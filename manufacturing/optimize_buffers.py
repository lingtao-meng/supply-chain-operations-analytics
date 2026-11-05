"""Enumerate Jackson station assignments and minimize buffer capacity under shift risk."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from math import ceil, floor
from pathlib import Path

import numpy as np

from line_balance import ROOT, Case, greedy, reference_plans, verify
from shift_capacity import (
    DESIGN_REPLICATIONS, DESIGN_SCENARIOS, REQUIRED_RELIABILITY, SHIFT_SECONDS, VALIDATION_REPLICATIONS,
    demand_completion_times, lognormal_multiplier, reliability_lower, simulate_shift,
)

DEMAND = 1780
STAFF = 4
PER_GAP_MAX = 2
DESIGN_SEED, VALIDATION_SEED = 20261107, 20261108
VALIDATION_SCENARIOS = (*DESIGN_SCENARIOS, (20, 5))
Plan = tuple[tuple[int, ...], ...]
Layout = tuple[int, ...]


def predecessor_ideals(case: Case) -> list[frozenset[int]]:
    ideals = [frozenset()]
    for task in case.order:
        ideals.extend([done | {task} for done in ideals if case.predecessors[task] <= done])
    return sorted(ideals, key=lambda done: (len(done), sorted(done)))


def enumerate_plans(case: Case, cycle: int, staff: int) -> list[Plan]:
    """Split the DAG into nonempty stations using nested predecessor-closed sets."""
    case.check_cycle(cycle)
    if not 1 <= staff <= len(case.task_times):
        raise ValueError("station count must be between one and the number of tasks")
    ideals = predecessor_ideals(case)
    all_tasks = frozenset(case.order)
    prefix_load = {done: sum(map(case.duration, done)) for done in ideals}
    plans = []

    def visit(done: frozenset[int], stations: Plan) -> None:
        remaining = staff - len(stations)
        if remaining == 1:
            if done < all_tasks and prefix_load[all_tasks] - prefix_load[done] <= cycle:
                plans.append((*stations, tuple(task for task in case.order if task not in done)))
            return
        for next_done in ideals:
            if (done < next_done < all_tasks and prefix_load[next_done] - prefix_load[done] <= cycle
                    and len(all_tasks - next_done) >= remaining - 1
                    and prefix_load[all_tasks] - prefix_load[next_done] <= cycle * (remaining - 1)):
                tasks = tuple(task for task in case.order if task in next_done - done)
                visit(next_done, (*stations, tasks))

    visit(frozenset(), ())
    return plans


def workloads_for(task_times: np.ndarray, plan: Plan) -> np.ndarray:
    return np.stack([task_times[:, np.asarray(tasks) - 1].sum(axis=1) for tasks in plan], axis=1)


def scenarios(case: Case, pairs: tuple, jobs: int, runs: int, seed: int):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((jobs, len(case.task_times), runs))
    shift_noise = rng.standard_normal((1, 1, runs))
    nominal = np.asarray(case.task_times)[None, :, None]
    for task_cv, shift_cv in pairs:
        yield (task_cv, shift_cv), (
            nominal * lognormal_multiplier(noise, task_cv) * lognormal_multiplier(shift_noise, shift_cv)
        )


def frozen_milp(case: Case, cycle: int) -> Plan:
    return tuple(tuple(tasks) for tasks in reference_plans(case, cycle)["MILP"])


def run_study() -> tuple[dict, list[dict]]:
    records = json.loads((ROOT / "data" / "benchmarks.json").read_text(encoding="utf-8"))
    case = Case(**next(record for record in records if record["name"] == "Jackson"))
    cycle = floor(SHIFT_SECONDS / DEMAND)
    plans = enumerate_plans(case, cycle, STAFF)
    loads = [verify(case, cycle, plan) for plan in plans]
    layouts = list(product(range(PER_GAP_MAX + 1), repeat=STAFF - 1))
    candidate_rows = []
    selected = None

    design_times = list(scenarios(case, DESIGN_SCENARIOS, DEMAND, DESIGN_REPLICATIONS, DESIGN_SEED))
    for budget in range(PER_GAP_MAX * (STAFF - 1) + 1):
        tier = [layout for layout in layouts if sum(layout) == budget]
        qualifying = []
        for index, plan in enumerate(plans):
            workloads = [workloads_for(times, plan) for _, times in design_times]
            for layout in tier:
                finish = [demand_completion_times(work, layout) for work in workloads]
                hits = [int((times <= SHIFT_SECONDS).sum()) for times in finish]
                lower = min(reliability_lower(count, DESIGN_REPLICATIONS) for count in hits)
                worst_p95 = max(float(np.quantile(times, 0.95)) for times in finish)
                qualifies = lower >= REQUIRED_RELIABILITY
                candidate_rows.append({
                    "plan_id": index + 1, "station_tasks": " | ".join(" ".join(map(str, s)) for s in plan),
                    "station_loads": " ".join(map(str, loads[index])),
                    "buffer_layout": " ".join(map(str, layout)), "buffer_slots": budget,
                    "hits_task20_shift0": hits[0], "hits_task20_shift3": hits[1],
                    "worst_hit_lower95": round(lower, 6), "worst_p95_finish_seconds": round(worst_p95, 3),
                    "qualifies": qualifies,
                })
                if qualifies:
                    qualifying.append((worst_p95, index, layout))
        if qualifying:
            _, index, layout = min(qualifying)
            selected = (plans[index], layout)
            break

    milp = frozen_milp(case, cycle)
    rpw = tuple(tuple(tasks) for tasks in greedy(case, cycle, "RPW"))
    alternatives = {
        "rpw_no_buffer": (rpw, (0,) * (STAFF - 1)),
        "milp_no_buffer": (milp, (0,) * (STAFF - 1)),
        "transfer_milp_uniform_1": (milp, (1,) * (STAFF - 1)),
    }
    if selected:
        alternatives["optimized"] = selected
    designs = {}
    for name, (plan, layout) in alternatives.items():
        designs[name] = {"station_tasks": plan, "station_loads": verify(case, cycle, plan),
                         "buffer_layout": layout, "buffer_slots": sum(layout), "validation": []}
    for (task_cv, shift_cv), times in scenarios(
        case, VALIDATION_SCENARIOS, 2 * ceil(SHIFT_SECONDS / (sum(case.task_times) / STAFF)),
        VALIDATION_REPLICATIONS, VALIDATION_SEED,
    ):
        for name, (plan, layout) in alternatives.items():
            output, state = simulate_shift(workloads_for(times, plan), layout, SHIFT_SECONDS)
            hits = int((output >= DEMAND).sum())
            designs[name]["validation"].append({
                "task_cv_pct": task_cv, "shift_cv_pct": shift_cv, "runs": VALIDATION_REPLICATIONS,
                "hits": hits, "hit_lower95": round(reliability_lower(hits, len(output)), 6),
                "mean_output": round(float(output.mean()), 2), "p05_output": round(float(np.quantile(output, 0.05)), 2),
                "mean_shortfall": round(float(np.maximum(DEMAND - output, 0).mean()), 2),
                "station_processing_blocked_idle_pct": (100 * state.mean(axis=1)).round(2).tolist(),
            })
    decision = {
        "case": case.name, "shift_seconds": SHIFT_SECONDS, "demand": DEMAND, "integer_design_cycle": cycle,
        "staff": STAFF, "per_gap_max": PER_GAP_MAX, "required_reliability": REQUIRED_RELIABILITY,
        "design_seed": DESIGN_SEED, "design_runs": DESIGN_REPLICATIONS, "design_scenarios": DESIGN_SCENARIOS,
        "validation_seed": VALIDATION_SEED, "validation_runs": VALIDATION_REPLICATIONS,
        "feasible_station_plans": len(plans), "possible_station_buffer_combinations": len(plans) * len(layouts),
        "screened_combinations": len(candidate_rows), "designs": designs,
        "passes_independent_screen": bool(selected) and all(
            reliability_lower(row["hits"], row["runs"]) >= REQUIRED_RELIABILITY
            for row in designs["optimized"]["validation"]
            if (row["task_cv_pct"], row["shift_cv_pct"]) in DESIGN_SCENARIOS
        ),
    }
    return decision, candidate_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    decision, candidates = run_study()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "jackson_buffer_decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "jackson_buffer_screen.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=candidates[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(candidates)
    for name, design in decision["designs"].items():
        print(f"{name}: {design['station_loads']} / {design['buffer_layout']}, "
              f"validation hits {[row['hits'] for row in design['validation']]} of {VALIDATION_REPLICATIONS}")
    print(f"Independent screen passed: {decision['passes_independent_screen']}")


if __name__ == "__main__":
    main()

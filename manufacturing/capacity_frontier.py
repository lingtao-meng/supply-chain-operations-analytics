"""Compare reliable shift capacity across buffer budgets and test rebalancing."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from functools import cache
from itertools import product
from math import ceil
from pathlib import Path
from time import perf_counter

import numpy as np

from line_balance import ROOT, Case, verify
from optimize_buffers import Layout, Plan, enumerate_plans, frozen_milp, predecessor_ideals, scenarios, workloads_for
from shift_capacity import DESIGN_SCENARIOS, REQUIRED_RELIABILITY, SHIFT_SECONDS, reliability_lower, shift_output_counts

DESIGN_RUNS, VALIDATION_RUNS = 120, 180
DESIGN_SEED, VALIDATION_SEED = 20261109, 20261110
CONFIRMATION_RUNS, CONFIRMATION_SEED = 300, 20261111
BEAM_WIDTH = 16
BATCH_SIZE = 8
SHOCK_TASK, SHOCK_FACTOR, SHOCK_TARGET = 4, 1.4, 1780


@dataclass(frozen=True)
class Candidate:
    plan: Plan
    layout: Layout
    capacity: int
    worst_mean: float


def load_score(loads: tuple[int, ...]) -> tuple[int, int]:
    return max(loads, default=0), sum(load * load for load in loads)


def beam_plans(case: Case, cycle: int, staff: int, width: int) -> list[Plan]:
    """Keep a fixed number of partial assignments at each station depth."""
    ideals = predecessor_ideals(case)
    all_tasks = frozenset(case.order)
    work = {done: sum(map(case.duration, done)) for done in ideals}

    @cache
    def transitions(done: frozenset[int], remaining: int) -> tuple[frozenset[int], ...]:
        if remaining == 0:
            return (all_tasks,) if done < all_tasks and work[all_tasks] - work[done] <= cycle else ()
        return tuple(end for end in ideals if done < end < all_tasks and work[end] - work[done] <= cycle
                     and len(all_tasks - end) >= remaining
                     and work[all_tasks] - work[end] <= remaining * cycle
                     and transitions(end, remaining - 1))

    states = [(frozenset(), (), ())]
    for station in range(staff):
        remaining = staff - station - 1
        expanded = []
        for done, plan, loads in states:
            for end in transitions(done, remaining):
                load = work[end] - work[done]
                group = tuple(task for task in case.order if task in end - done)
                expanded.append((end, (*plan, group), (*loads, load)))
        states = sorted(expanded, key=lambda state: (load_score(state[2]), state[1]))[:width]
    # Include the static benchmark even if the beam prunes its partial assignments.
    return list(dict.fromkeys([*(plan for _, plan, _ in states), frozen_milp(case, cycle)]))


def required_hits(runs: int) -> int:
    return next(hits for hits in range(runs + 1) if reliability_lower(hits, runs) >= REQUIRED_RELIABILITY)


def reliable_capacity(counts: np.ndarray, hits: int) -> np.ndarray:
    """Largest integer demand reached by at least the required number of replications."""
    position = counts.shape[-1] - hits
    return np.partition(counts, position, axis=-1)[..., position]


def scenario_arrays(case: Case, staff: int, runs: int, seed: int, shock: bool = False) -> list[np.ndarray]:
    jobs = ceil(1.45 * SHIFT_SECONDS * staff / sum(case.task_times))
    arrays = []
    for _, times in scenarios(case, DESIGN_SCENARIOS, jobs, runs, seed):
        if shock:
            times[:, SHOCK_TASK - 1, :] *= SHOCK_FACTOR
        arrays.append(times)
    return arrays


def evaluate_pool(
    plans: list[Plan], layouts: list[Layout], arrays: list[np.ndarray],
) -> list[Candidate]:
    runs = arrays[0].shape[-1]
    min_hits = required_hits(runs)
    candidates = []
    for begin in range(0, len(plans), BATCH_SIZE):
        batch = plans[begin:begin + BATCH_SIZE]
        # Pack candidate plans into the replication axis. Each sees the same task-time draws.
        workloads = []
        for times in arrays:
            grouped = np.stack([workloads_for(times, plan) for plan in batch], axis=2)
            workloads.append(grouped.reshape(grouped.shape[0], grouped.shape[1], -1))
        for layout in layouts:
            counts = [shift_output_counts(work, layout, SHIFT_SECONDS).reshape(len(batch), runs)
                      for work in workloads]
            capacities = np.min([reliable_capacity(count, min_hits) for count in counts], axis=0)
            means = np.min([count.mean(axis=-1) for count in counts], axis=0)
            candidates.extend(Candidate(plan, layout, int(capacity), float(mean))
                              for plan, capacity, mean in zip(batch, capacities, means))
    return candidates


def best_by_capacity(candidates: list[Candidate]) -> Candidate:
    return min(candidates, key=lambda item: (-item.capacity, sum(item.layout), -item.worst_mean, item.plan, item.layout))


def select_for_target(candidates: list[Candidate], target: int) -> Candidate:
    feasible = [item for item in candidates if item.capacity >= target]
    if feasible:
        minimum = min(sum(item.layout) for item in feasible)
        return best_by_capacity([item for item in feasible if sum(item.layout) == minimum])
    return best_by_capacity(candidates)


def validate(candidate: Candidate, arrays: list[np.ndarray], target: int) -> dict:
    counts = [shift_output_counts(workloads_for(times, candidate.plan), candidate.layout, SHIFT_SECONDS)
              for times in arrays]
    runs = arrays[0].shape[-1]
    hits = [int((count >= target).sum()) for count in counts]
    lower = min(reliability_lower(value, runs) for value in hits)
    return {
        "target": target, "validation_hits": hits, "worst_lower95": round(lower, 6),
        "passes": lower >= REQUIRED_RELIABILITY,
        "validation_capacity": min(int(reliable_capacity(count, required_hits(runs))) for count in counts),
        "validation_worst_mean": round(min(float(count.mean()) for count in counts), 2),
    }


def search_frontier(
    case: Case, staff: int, method: str, candidates: list[Candidate], validation: list[np.ndarray],
) -> list[dict]:
    maximum = max(sum(item.layout) for item in candidates)
    rows = []
    cached = {}
    for budget in range(maximum + 1):
        selected = best_by_capacity([item for item in candidates if sum(item.layout) <= budget])
        if selected not in cached:
            cached[selected] = validate(selected, validation, selected.capacity)
        verified = cached[selected]
        rows.append({
            "case": case.name, "staff": staff, "method": method, "slot_budget": budget,
            "used_slots": sum(selected.layout), "layout": " ".join(map(str, selected.layout)),
            "station_tasks": " | ".join(" ".join(map(str, group)) for group in selected.plan),
            "design_capacity": selected.capacity, "validation_capacity": verified["validation_capacity"],
            "hits_task20_shift0": verified["validation_hits"][0],
            "hits_task20_shift3": verified["validation_hits"][1],
            "worst_lower95": verified["worst_lower95"], "design_target_validated": verified["passes"],
        })
    return rows


def confirm_frontier(rows: list[dict], cases: dict[str, Case]) -> None:
    """Lock a conservative target after the pilot and test it with fresh replications."""
    for name, case in cases.items():
        subset = [row for row in rows if row["case"] == name]
        if not subset:
            continue
        arrays = scenario_arrays(case, subset[0]["staff"], CONFIRMATION_RUNS, CONFIRMATION_SEED)
        cached = {}
        for row in subset:
            plan = tuple(tuple(map(int, group.split())) for group in row["station_tasks"].split(" | "))
            layout = tuple(map(int, row["layout"].split()))
            # Hold back 1% of the lower capacity estimate, then round down to tens.
            capacity = min(int(row["design_capacity"]), int(row["validation_capacity"]))
            target = (99 * capacity // 1000) * 10
            key = (plan, layout, target)
            if key not in cached:
                cached[key] = validate(Candidate(plan, layout, capacity, 0), arrays, target)
            result = cached[key]
            row.update({
                "planning_target": target,
                "confirmation_hits_task20_shift0": result["validation_hits"][0],
                "confirmation_hits_task20_shift3": result["validation_hits"][1],
                "confirmation_worst_lower95": result["worst_lower95"],
                "planning_target_confirmed": result["passes"],
            })


def run_study() -> tuple[list[dict], dict]:
    records = json.loads((ROOT / "data" / "benchmarks.json").read_text(encoding="utf-8"))
    cases = {record["name"]: Case(**record) for record in records}
    configuration = {
        "shift_seconds": SHIFT_SECONDS, "required_reliability": REQUIRED_RELIABILITY,
        "design_scenarios_cv_pct": DESIGN_SCENARIOS, "design_runs": DESIGN_RUNS,
        "validation_runs": VALIDATION_RUNS, "design_seed": DESIGN_SEED, "validation_seed": VALIDATION_SEED,
        "confirmation_runs": CONFIRMATION_RUNS, "confirmation_seed": CONFIRMATION_SEED,
        "planning_target_rule": "round down to tens after subtracting 1% from min(design_capacity, validation_capacity)",
        "beam_width": BEAM_WIDTH,
    }
    rows, searches = [], []
    for name, cycle, staff, per_gap_max in (("Jackson", 14, 4, 2), ("Mitchell", 15, 8, 1)):
        case = cases[name]
        design = scenario_arrays(case, staff, DESIGN_RUNS, DESIGN_SEED)
        validation = scenario_arrays(case, staff, VALIDATION_RUNS, VALIDATION_SEED)
        layouts = list(product(range(per_gap_max + 1), repeat=staff - 1))
        methods = ["exact", "beam"] if name == "Jackson" else ["beam"]
        for method in methods:
            started = perf_counter()
            plans = (enumerate_plans(case, cycle, staff) if method == "exact"
                     else beam_plans(case, cycle, staff, BEAM_WIDTH))
            for plan in plans:
                verify(case, cycle, plan)
            candidates = evaluate_pool(plans, layouts, design)
            print(f"{name} {method}: {len(candidates)} candidates in {perf_counter() - started:.1f}s", flush=True)
            searches.append({"case": name, "method": method, "plans": len(plans), "layouts": len(layouts),
                             "candidates": len(candidates), "per_gap_max": per_gap_max, "nominal_cycle": cycle})
            rows.extend(search_frontier(case, staff, method, candidates, validation))

    confirm_frontier(rows, cases)
    case = cases["Jackson"]
    previous = json.loads((ROOT / "results" / "jackson_buffer_decision.json").read_text(encoding="utf-8"))
    locked = previous["designs"]["optimized"]
    incumbent = tuple(tuple(group) for group in locked["station_tasks"])
    incumbent_layout = tuple(locked["buffer_layout"])
    plans = enumerate_plans(case, 14, 4)
    layouts = list(product(range(3), repeat=3))
    shock_design = scenario_arrays(case, 4, DESIGN_RUNS, DESIGN_SEED, shock=True)
    shock_validation = scenario_arrays(case, 4, VALIDATION_RUNS, VALIDATION_SEED, shock=True)
    pool = evaluate_pool(plans, layouts, shock_design)
    policies = {
        "retain": [item for item in pool if item.plan == incumbent and item.layout == incumbent_layout],
        "buffers_only": [item for item in pool if item.plan == incumbent],
        "reassign_only": [item for item in pool if item.layout == incumbent_layout],
        "joint": pool,
    }
    shock_results = {}
    for policy, eligible in policies.items():
        selected = select_for_target(eligible, SHOCK_TARGET)
        shock_results[policy] = {
            "station_tasks": selected.plan, "layout": selected.layout, "slots": sum(selected.layout),
            "design_capacity": selected.capacity, "design_target_met": selected.capacity >= SHOCK_TARGET,
            **validate(selected, shock_validation, SHOCK_TARGET),
        }
    return rows, {"configuration": configuration, "searches": searches, "shock": {
        "case": "Jackson", "task": SHOCK_TASK, "duration_multiplier": SHOCK_FACTOR,
        "target": SHOCK_TARGET, "policies": shock_results,
    }}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    rows, report = run_study()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "capacity_frontier.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "frontier_experiments.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""Certify sampled throughput bounds for binary buffer allocations."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from functools import lru_cache
from heapq import heappop, heappush
from pathlib import Path
from time import perf_counter

import numpy as np
from numba import njit
from scipy.stats import beta

from capacity_frontier import (
    Candidate, CONFIRMATION_RUNS, DESIGN_RUNS, DESIGN_SEED, VALIDATION_RUNS, VALIDATION_SEED,
    reliable_capacity, required_hits, scenario_arrays, validate,
)
from line_balance import ROOT, Case, verify
from optimize_buffers import Layout, Plan, enumerate_plans, workloads_for
from shift_capacity import DESIGN_SCENARIOS, REQUIRED_RELIABILITY, SHIFT_SECONDS

ADDITIONAL_EVALUATION_LIMIT = 6000
KEY_BUDGETS = (0, 3, 7)
CONFIRMATION_SEED = 20261114


@njit
def binary_output_counts(service: np.ndarray, mask: int, duration: float) -> np.ndarray:
    """Same blocking-after-service recurrence, specialized to 0/1 slots per gap."""
    runs, jobs, stations = service.shape
    output = np.zeros(runs, dtype=np.int64)
    for run in range(runs):
        exits = np.zeros(stations)
        starts = np.zeros(stations)
        for job in range(jobs):
            upstream = 0.0
            for station in range(stations):
                start = max(upstream, exits[station])
                departure = start + service[run, job, station]
                if station < stations - 1:
                    # Downstream records still refer to the preceding product.
                    if mask & (1 << station):
                        departure = max(departure, starts[station + 1])
                    else:
                        departure = max(departure, exits[station + 1])
                starts[station] = start
                exits[station] = departure
                upstream = departure
            if upstream <= duration:
                output[run] += 1
            if upstream >= duration:
                break
        else:
            raise ValueError("the workload ends before the shift")
    return output


def layout_for(mask: int, gaps: int) -> Layout:
    return tuple((mask >> i) & 1 for i in range(gaps))


def mask_for(layout: Layout) -> int:
    if any(bit not in (0, 1) for bit in layout):
        raise ValueError("binary buffer layouts may only contain 0 or 1")
    return sum(bit << i for i, bit in enumerate(layout))


class Simulator:
    """Memoize sampled capacity per (plan, buffer mask) across budget searches."""

    def __init__(self, plans: list[Plan], arrays: list[np.ndarray], duration: float = SHIFT_SECONDS):
        self.plans = plans
        self.scores: dict[tuple[int, int], int] = {}
        self.minimum_hits = required_hits(arrays[0].shape[-1])
        self.duration = duration

        @lru_cache(maxsize=8)
        def workload(plan_index: int) -> tuple[np.ndarray, ...]:
            return tuple(np.ascontiguousarray(workloads_for(times, plans[plan_index]).transpose(2, 0, 1))
                         for times in arrays)

        self.workload = workload

    def score(self, plan: int, mask: int) -> int:
        key = (plan, mask)
        if key not in self.scores:
            self.scores[key] = min(int(reliable_capacity(
                binary_output_counts(work, mask, self.duration), self.minimum_hits,
            )) for work in self.workload(plan))
        return self.scores[key]


@dataclass
class Certificate:
    plan: int
    mask: int
    lower: int
    upper: int
    evaluations: int
    pruned: int = 0
    expanded: int = 0
    checkpoints: list[dict] = field(default_factory=list)


def children(mask: int, next_gap: int, gaps: int, budget: int) -> list[tuple[int, int]]:
    """Remove gaps in ascending order so each feasible leaf has one parent path."""
    remove = mask.bit_count() - budget
    if budget == 0:
        return [(0, gaps)]
    return [(mask ^ (1 << gap), gap + 1) for gap in range(next_gap, gaps - remove + 1)
            if mask & (1 << gap)]


def solve_budget(
    simulator: Simulator, budget: int, gaps: int, roots: list[int],
    seed: tuple[int, int], evaluation_limit: int = ADDITIONAL_EVALUATION_LIMIT,
) -> Certificate:
    """Best-bound-first search with an admissible relaxation at every node."""
    seed_plan, seed_mask = seed
    lower = simulator.score(seed_plan, seed_mask)
    before = len(simulator.scores)
    best = seed
    all_slots = (1 << gaps) - 1
    queue = []
    pruned = expanded = 0
    if budget == gaps:
        winner = max(range(len(roots)), key=roots.__getitem__)
        return Certificate(winner, all_slots, roots[winner], roots[winner], len(simulator.scores) - before)

    for plan, bound in enumerate(roots):
        if bound > lower:
            heappush(queue, (-bound, plan, all_slots, 0))
        else:
            pruned += 1

    def checkpoint() -> dict:
        return {"additional_evaluations": len(simulator.scores) - before, "lower": lower,
                "upper": max(lower, -queue[0][0]) if queue else lower}

    checkpoints = [checkpoint()]
    milestones = (100, 500, 1000, 2500, 5000)
    next_milestone = 0
    while queue and -queue[0][0] > lower:
        negative_bound, plan, mask, next_gap = heappop(queue)
        options = children(mask, next_gap, gaps, budget)
        new_scores = sum((plan, smaller) not in simulator.scores for smaller, _ in options)
        if len(simulator.scores) - before + new_scores > evaluation_limit:
            heappush(queue, (negative_bound, plan, mask, next_gap))
            break
        expanded += 1
        for smaller, following in options:
            bound = simulator.score(plan, smaller)
            if bound > -negative_bound:
                raise AssertionError("buffer monotonicity violated for the same plan and samples")
            if smaller.bit_count() == budget:
                if bound > lower:
                    lower, best = bound, (plan, smaller)
            elif bound > lower:
                heappush(queue, (-bound, plan, smaller, following))
            else:
                pruned += 1
        while next_milestone < len(milestones) and len(simulator.scores) - before >= milestones[next_milestone]:
            checkpoints.append(checkpoint())
            next_milestone += 1
    final = checkpoint()
    if final != checkpoints[-1]:
        checkpoints.append(final)
    return Certificate(*best, lower, final["upper"], len(simulator.scores) - before, pruned, expanded, checkpoints)


def canonical(plan: Plan, case: Case) -> Plan:
    return tuple(tuple(task for task in case.order if task in group) for group in plan)


def existing_references(case: Case, method: str, budgets: tuple[int, ...]) -> dict[int, tuple[Plan, Layout, int]]:
    with (ROOT / "results" / "capacity_frontier.csv").open(encoding="utf-8", newline="") as stream:
        return {
            int(row["slot_budget"]): (
                canonical(tuple(tuple(map(int, group.split())) for group in row["station_tasks"].split(" | ")), case),
                tuple(map(int, row["layout"].split())), int(row["design_capacity"]),
            )
            for row in csv.DictReader(stream) if row["case"] == case.name and row["method"] == method
            and int(row["slot_budget"]) in budgets
        }


def case_study(
    case: Case, cycle: int, staff: int, budgets: tuple[int, ...], method: str, evaluation_limit: int,
) -> dict:
    plans = enumerate_plans(case, cycle, staff)
    plan_indices = {plan: i for i, plan in enumerate(plans)}
    gaps = staff - 1
    design = scenario_arrays(case, staff, DESIGN_RUNS, DESIGN_SEED)
    validation = scenario_arrays(case, staff, VALIDATION_RUNS, VALIDATION_SEED)
    confirmation = scenario_arrays(case, staff, CONFIRMATION_RUNS, CONFIRMATION_SEED)
    simulator = Simulator(plans, design)
    full = (1 << gaps) - 1
    roots = [simulator.score(plan, full) for plan in range(len(plans))]
    references = existing_references(case, method, budgets)
    results = []
    print(f"{case.name}: {len(plans)} feasible station plans", flush=True)
    for budget in budgets:
        plan, layout, expected = references[budget]
        index = plan_indices[plan]
        mask = mask_for(layout)
        if len(layout) != gaps or sum(layout) > budget:
            raise ValueError("reference uses a layout outside the binary budget")
        if simulator.score(index, mask) != expected:
            raise AssertionError("accelerated simulation does not reproduce the saved reference")
        cert = solve_budget(simulator, budget, gaps, roots, (index, mask), evaluation_limit)
        winner = plans[cert.plan]
        slots = layout_for(cert.mask, gaps)
        verify(case, cycle, winner)
        selected = Candidate(winner, slots, cert.lower, 0)
        if validate(selected, design, cert.lower)["validation_capacity"] != cert.lower:
            raise AssertionError("the reference simulator disagrees with the certified design score")
        checked = validate(selected, validation, cert.lower)
        planning_target = (99 * min(cert.lower, checked["validation_capacity"]) // 1000) * 10
        final = validate(selected, confirmation, planning_target)
        results.append({
            "budget": budget, "reference_method": method, "reference_capacity": expected,
            "lower": cert.lower, "upper": cert.upper, "gap_units": cert.upper - cert.lower,
            "station_tasks": winner, "layout": slots, "additional_evaluations": cert.evaluations,
            "expanded_nodes": cert.expanded, "pruned_nodes": cert.pruned, "checkpoints": cert.checkpoints,
            "independent_validation": checked, "planning_confirmation": final,
        })
        print(f"  budget {budget}: [{cert.lower}, {cert.upper}], reference {expected}, "
              f"{cert.evaluations} additional simulations", flush=True)
    return {"case": case.name, "cycle": cycle, "staff": staff, "feasible_plans": len(plans),
            "binary_layouts_per_plan": full + 1, "root_evaluations": len(roots),
            "unique_simulated_combinations": len(simulator.scores), "results": results}


def run_study(evaluation_limit: int) -> dict:
    rows = json.loads((ROOT / "data" / "benchmarks.json").read_text(encoding="utf-8"))
    cases = {row["name"]: Case(**row) for row in rows}
    return {
        "configuration": {
            "shift_seconds": SHIFT_SECONDS, "scenarios_cv_pct": DESIGN_SCENARIOS,
            "required_reliability": REQUIRED_RELIABILITY, "evaluation_limit_per_budget": evaluation_limit,
            "design_seed": DESIGN_SEED, "design_runs": DESIGN_RUNS,
            "validation_seed": VALIDATION_SEED, "validation_runs": VALIDATION_RUNS,
            "confirmation_seed": CONFIRMATION_SEED, "confirmation_runs": CONFIRMATION_RUNS,
        },
        "cases": [
            case_study(cases["Jackson"], 14, 4, (0, 1, 3), "exact", evaluation_limit),
            case_study(cases["Mitchell"], 15, 8, KEY_BUDGETS, "beam", evaluation_limit),
        ],
    }


def add_confirmation_bounds(report: dict, alpha: float = 0.05) -> None:
    """Add simultaneous one-sided exact bounds for the locked confirmation targets."""
    rows = [row for case in report["cases"] for row in case["results"]]
    tests = sum(len(row["planning_confirmation"]["validation_hits"]) for row in rows)
    runs = report["configuration"]["confirmation_runs"]
    report["configuration"]["confirmation_family_alpha"] = alpha
    report["configuration"]["confirmation_family_size"] = tests
    for row in rows:
        final = row["planning_confirmation"]
        bounds = [float(beta.ppf(alpha / tests, hits, runs - hits + 1)) if hits else 0.0
                  for hits in final["validation_hits"]]
        final["simultaneous_lower95_by_scenario"] = [round(value, 6) for value in bounds]
        final["passes_simultaneous_confirmation"] = min(bounds) >= REQUIRED_RELIABILITY


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "search_certificate.json")
    parser.add_argument("--evaluation-limit", type=int, default=ADDITIONAL_EVALUATION_LIMIT)
    args = parser.parse_args()
    if args.evaluation_limit < 0:
        parser.error("the additional evaluation limit must be nonnegative")
    started = perf_counter()
    report = run_study(args.evaluation_limit)
    add_confirmation_bounds(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Elapsed: {perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()

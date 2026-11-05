"""Test whether Mitchell's buffer rule transfers to the published Buxey benchmark."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from math import ceil, comb
from pathlib import Path
from time import perf_counter

from capacity_frontier import Candidate, scenario_arrays, validate
from certified_search import Simulator, add_confirmation_bounds, layout_for, mask_for, solve_budget
from line_balance import ROOT, Case, verify
from optimize_buffers import enumerate_plans, scenarios
from shift_capacity import DESIGN_SCENARIOS, REQUIRED_RELIABILITY, SHIFT_SECONDS

STAFF, BUDGET = 8, 3
DESIGN_RUNS, VALIDATION_RUNS, CONFIRMATION_RUNS = 120, 180, 300
DESIGN_SEED, VALIDATION_SEED, CONFIRMATION_SEED, STRESS_SEED = 20261115, 20261116, 20261117, 20261118
EVALUATION_LIMIT = 6000


def run_study() -> dict:
    case = Case(**json.loads((ROOT / "data" / "buxey.json").read_text(encoding="utf-8")))
    cycle = case.reference_cycle
    plans = enumerate_plans(case, cycle, STAFF)
    loads = [verify(case, cycle, plan) for plan in plans]
    static = min(range(len(plans)), key=lambda i: (max(loads[i]), sum(load * load for load in loads[i]), plans[i]))

    previous = json.loads((ROOT / "results" / "search_certificate.json").read_text(encoding="utf-8"))
    mitchell = next(row for row in previous["cases"] if row["case"] == "Mitchell")
    transferred = tuple(next(row["layout"] for row in mitchell["results"] if row["budget"] == BUDGET))
    gaps = STAFF - 1
    masks = [sum(1 << gap for gap in occupied) for occupied in combinations(range(gaps), BUDGET)]
    transferred_mask = mask_for(transferred)
    if len(transferred) != gaps or transferred_mask not in masks:
        raise ValueError("the transferred layout does not match the trial budget")

    design = scenario_arrays(case, STAFF, DESIGN_RUNS, DESIGN_SEED)
    simulator = Simulator(plans, design)
    baseline_mask = max(masks, key=lambda mask: (simulator.score(static, mask), -mask))
    full = (1 << gaps) - 1
    print(f"{len(plans)} feasible station plans, {len(masks)} layouts with {BUDGET} slots", flush=True)
    roots = [simulator.score(index, full) for index in range(len(plans))]
    certificate = solve_budget(simulator, BUDGET, gaps, roots, (static, baseline_mask), EVALUATION_LIMIT)
    print(f"joint search [{certificate.lower}, {certificate.upper}], "
          f"{certificate.evaluations} additional evaluations", flush=True)

    options = {
        "transferred_positions": (static, transferred_mask),
        "optimize_buffers_only": (static, baseline_mask),
        "optimize_assignment_and_buffers": (certificate.plan, certificate.mask),
    }
    validation = scenario_arrays(case, STAFF, VALIDATION_RUNS, VALIDATION_SEED)
    confirmation = scenario_arrays(case, STAFF, CONFIRMATION_RUNS, CONFIRMATION_SEED)
    jobs = ceil(1.45 * SHIFT_SECONDS * STAFF / sum(case.task_times))
    stress = [times for _, times in scenarios(case, ((20, 5),), jobs, CONFIRMATION_RUNS, STRESS_SEED)]

    comparisons = []
    for name, (index, mask) in options.items():
        capacity = simulator.score(index, mask)
        layout = layout_for(mask, gaps)
        plan = plans[index]
        candidate = Candidate(plan, layout, capacity, 0)
        if validate(candidate, design, capacity)["validation_capacity"] != capacity:
            raise AssertionError("the reference and accelerated simulators disagree")
        checked = validate(candidate, validation, capacity)
        planning_target = (99 * min(capacity, checked["validation_capacity"]) // 1000) * 10
        comparisons.append({
            "policy": name, "station_tasks": plan, "station_loads": loads[index], "layout": layout,
            "design_capacity": capacity, "independent_validation": checked,
            "planning_confirmation": validate(candidate, confirmation, planning_target),
        })

    common_target = comparisons[-1]["planning_confirmation"]["target"]
    for row, (index, mask) in zip(comparisons, options.values()):
        candidate = Candidate(plans[index], layout_for(mask, gaps), row["design_capacity"], 0)
        row["confirmation_at_common_target"] = validate(candidate, confirmation, common_target)
        row["stress_at_common_target"] = validate(candidate, stress, common_target)

    report = {
        "configuration": {
            "shift_seconds": SHIFT_SECONDS, "required_reliability": REQUIRED_RELIABILITY,
            "scenario_cv_pct": DESIGN_SCENARIOS, "stress_cv_pct": (20, 5),
            "design_seed": DESIGN_SEED, "design_runs": DESIGN_RUNS,
            "validation_seed": VALIDATION_SEED, "validation_runs": VALIDATION_RUNS,
            "confirmation_seed": CONFIRMATION_SEED, "confirmation_runs": CONFIRMATION_RUNS,
            "stress_seed": STRESS_SEED, "evaluation_limit": EVALUATION_LIMIT,
        },
        "cases": [{
            "case": case.name, "staff": STAFF, "nominal_cycle": cycle, "buffer_budget": BUDGET,
            "feasible_plans": len(plans), "same_budget_combinations": len(plans) * comb(gaps, BUDGET),
            "unique_simulated_combinations": len(simulator.scores), "transferred_layout": transferred,
            "design_certificate": {"lower": certificate.lower, "upper": certificate.upper,
                                   "gap_units": certificate.upper - certificate.lower,
                                   "checkpoints": certificate.checkpoints},
            "common_comparison_target": common_target, "results": comparisons,
        }],
    }
    add_confirmation_bounds(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "transfer_validation.json")
    args = parser.parse_args()
    started = perf_counter()
    report = run_study()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Elapsed: {perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()

"""Find resource-efficient line changes when the slowdown estimate is uncertain."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path

from capacity_frontier import (
    Candidate, SHOCK_TARGET, SHOCK_TASK, best_by_capacity, evaluate_pool, scenario_arrays, validate,
)
from line_balance import ROOT, Case, verify
from optimize_buffers import PER_GAP_MAX, STAFF, Layout, Plan, enumerate_plans
from shift_capacity import DESIGN_SCENARIOS, SHIFT_SECONDS

CYCLE = 14
DESIGN_SLOWDOWN_PCT = 45
CHECK_SLOWDOWN_PCT = (0, 40, DESIGN_SLOWDOWN_PCT, 50, 60)
DESIGN_RUNS, VALIDATION_RUNS = 120, 300
DESIGN_SEED, VALIDATION_SEED = 20261112, 20261113


def moved_tasks(incumbent: Plan, candidate: Plan) -> tuple[int, ...]:
    """Count tasks assigned to a different station, irrespective of within-station order."""
    original = {task: station for station, group in enumerate(incumbent) for task in group}
    return tuple(sorted(task for station, group in enumerate(candidate) for task in group
                        if original[task] != station))


def slowdown_scenarios(baseline, percent: int):
    scenarios = [times.copy() for times in baseline]
    for times in scenarios:
        times[:, SHOCK_TASK - 1, :] *= 1 + percent / 100
    return scenarios


def resource_frontier(
    candidates: list[Candidate], incumbent: Plan,
) -> tuple[list[dict], list[tuple[tuple[int, int], Candidate]]]:
    groups: dict[tuple[int, int], list[Candidate]] = {}
    moves = {plan: moved_tasks(incumbent, plan) for plan in dict.fromkeys(item.plan for item in candidates)}
    for item in candidates:
        groups.setdefault((len(moves[item.plan]), sum(item.layout)), []).append(item)

    best = {key: best_by_capacity(group) for key, group in groups.items()}
    feasible = {key for key, item in best.items() if item.capacity >= SHOCK_TARGET}
    pareto = sorted(key for key in feasible if not any(
        other != key and other[0] <= key[0] and other[1] <= key[1] for other in feasible
    ))
    rows = [
        {
            "moved_tasks": key[0], "buffer_slots": key[1], "candidates": len(groups[key]),
            "best_design_capacity": item.capacity, "design_feasible": key in feasible,
            "pareto": key in pareto, "moved_task_ids": " ".join(map(str, moves[item.plan])),
            "station_tasks": " | ".join(" ".join(map(str, group)) for group in item.plan),
            "layout": " ".join(map(str, item.layout)),
        }
        for key, item in sorted(best.items())
    ]
    return rows, [(key, best[key]) for key in pareto]


def run_study() -> tuple[list[dict], dict]:
    records = json.loads((ROOT / "data" / "benchmarks.json").read_text(encoding="utf-8"))
    case = Case(**next(record for record in records if record["name"] == "Jackson"))
    prior = json.loads((ROOT / "results" / "jackson_buffer_decision.json").read_text(encoding="utf-8"))
    incumbent = tuple(tuple(group) for group in prior["designs"]["optimized"]["station_tasks"])
    initial_layout = tuple(prior["designs"]["optimized"]["buffer_layout"])
    earlier = json.loads((ROOT / "results" / "frontier_experiments.json").read_text(encoding="utf-8"))
    prior_repair = earlier["shock"]["policies"]["joint"]
    repair = tuple(tuple(group) for group in prior_repair["station_tasks"])
    repair_layout = tuple(prior_repair["layout"])

    plans = enumerate_plans(case, CYCLE, STAFF)
    layouts: list[Layout] = list(product(range(PER_GAP_MAX + 1), repeat=STAFF - 1))
    # All candidates face the upper end of the uncertainty range. The flow
    # recurrences are monotone in durations for a fixed line configuration.
    baseline = scenario_arrays(case, STAFF, DESIGN_RUNS, DESIGN_SEED)
    candidates = evaluate_pool(plans, layouts, slowdown_scenarios(baseline, DESIGN_SLOWDOWN_PCT))
    rows, frontier = resource_frontier(candidates, incumbent)
    indexed = {(item.plan, item.layout): item for item in candidates}
    options = {
        "incumbent": indexed[(incumbent, initial_layout)],
        "previous_point_estimate_repair": indexed[(repair, repair_layout)],
        **{f"pareto_{moved}_moves_{slots}_slots": item for (moved, slots), item in frontier},
    }

    results = {
        name: {
            "station_tasks": item.plan, "nominal_loads": verify(case, CYCLE, item.plan),
            "moved_task_ids": moved_tasks(incumbent, item.plan), "layout": item.layout,
            "slots": sum(item.layout), "design_capacity": item.capacity,
            "validation": [],
        }
        for name, item in options.items()
    }
    validation = scenario_arrays(case, STAFF, VALIDATION_RUNS, VALIDATION_SEED)
    for percent in CHECK_SLOWDOWN_PCT:
        arrays = slowdown_scenarios(validation, percent)
        for name, item in options.items():
            results[name]["validation"].append({"slowdown_pct": percent, **validate(item, arrays, SHOCK_TARGET)})
    for result in results.values():
        result["passes_uncertainty_limit"] = next(row["passes"] for row in result["validation"]
                                                 if row["slowdown_pct"] == DESIGN_SLOWDOWN_PCT)

    report = {
        "case": case.name, "staff": STAFF, "shift_seconds": SHIFT_SECONDS, "nominal_cycle": CYCLE,
        "per_gap_max": PER_GAP_MAX, "slow_task": SHOCK_TASK, "target": SHOCK_TARGET,
        "design_slowdown_pct": DESIGN_SLOWDOWN_PCT, "check_slowdown_pct": CHECK_SLOWDOWN_PCT,
        "scenario_cv_pct": DESIGN_SCENARIOS, "design_seed": DESIGN_SEED, "design_runs": DESIGN_RUNS,
        "validation_seed": VALIDATION_SEED, "validation_runs": VALIDATION_RUNS,
        "candidates": len(candidates), "design_feasible_candidates": sum(item.capacity >= SHOCK_TARGET for item in candidates),
        "design_pareto_resources": [key for key, _ in frontier], "options": results,
    }
    return rows, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    rows, report = run_study()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "reconfiguration_frontier.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "reconfiguration_decisions.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    print(f"{report['candidates']} candidates, Pareto resources: {report['design_pareto_resources']}")
    for name, option in report["options"].items():
        print(name, option["passes_uncertainty_limit"])


if __name__ == "__main__":
    main()

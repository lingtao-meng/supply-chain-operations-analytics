"""Set a release-wait limit after a mechanical stop and check it on held-out shifts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from functools import cache
from itertools import product
from pathlib import Path

import numpy as np
from scipy.stats import beta

from downtime_recovery import completion_matrix, minimum_hits, support_time
from line_balance import ROOT, Case, verify
from optimize_buffers import scenarios, workloads_for
from shift_capacity import REQUIRED_RELIABILITY

MECHANICAL_STOP_MINUTES = 25
RELEASE_WAITS = (0, 5, 10, 15)
DESIGN_STARTS, VALIDATION_STARTS = (120, 300), (90, 330)
DESIGN_SEED, VALIDATION_SEED = 2026112203, 2026112204
RUNS, ALPHA = 300, 0.05


def maximum_wait(required_time: Callable[[int], int], deadline: int, upper: int) -> int | None:
    """Find the last feasible whole minute when required completion time is nondecreasing."""
    feasible, infeasible = -1, upper + 1
    while feasible + 1 < infeasible:
        minute = (feasible + infeasible) // 2
        if required_time(minute) <= deadline:
            feasible = minute
        else:
            infeasible = minute
    return feasible if feasible >= 0 else None


def run_study() -> dict:
    previous = json.loads((ROOT / "results" / "downtime_decision.json").read_text(encoding="utf-8"))
    records = json.loads((ROOT / "data" / "benchmarks.json").read_text(encoding="utf-8"))
    case = Case(**next(row for row in records if row["name"] == previous["case"]))
    plan = tuple(tuple(tasks) for tasks in previous["station_tasks"])
    layouts = {name: tuple(previous["layouts"][name]) for name in ("incumbent", "selected")}
    loads = verify(case, case.reference_cycle, plan)
    shift_seconds, target = previous["shift_seconds"], previous["target_units"]
    variation = previous["task_cv_pct"], previous["shift_cv_pct"]

    def sample(seed: int) -> np.ndarray:
        _, times = next(scenarios(case, (variation,), target, RUNS, seed))
        return workloads_for(times, plan)

    design = sample(DESIGN_SEED)
    design_alpha = ALPHA / (len(plan) * len(DESIGN_STARTS))
    design_hits = minimum_hits(RUNS, design_alpha)

    @cache
    def design_required_time(wait: int) -> int:
        completion = completion_matrix(design, layouts["selected"], MECHANICAL_STOP_MINUTES + wait, DESIGN_STARTS)
        return support_time(completion, design_hits)

    max_search_minutes = shift_seconds // 60 - max(DESIGN_STARTS) - MECHANICAL_STOP_MINUTES
    locked_wait = maximum_wait(design_required_time, shift_seconds, max_search_minutes)
    design_checks = [0] if locked_wait is None else [locked_wait]
    if locked_wait is not None and locked_wait < max_search_minutes:
        design_checks.append(locked_wait + 1)

    waits = sorted({*RELEASE_WAITS, locked_wait}) if locked_wait is not None else list(RELEASE_WAITS)
    validation = sample(VALIDATION_SEED)
    conditions = list(product(range(1, len(plan) + 1), VALIDATION_STARTS))
    validation_alpha = ALPHA / (len(layouts) * len(waits) * len(conditions))
    validation_hits = minimum_hits(RUNS, validation_alpha)
    results = []
    for name, layout in layouts.items():
        for wait in waits:
            completion = completion_matrix(validation, layout, MECHANICAL_STOP_MINUTES + wait, VALIDATION_STARTS)
            counts = np.count_nonzero(completion <= shift_seconds, axis=1)
            lower = np.zeros(len(counts))
            positive = counts > 0
            lower[positive] = beta.ppf(validation_alpha, counts[positive], RUNS - counts[positive] + 1)
            deadline = support_time(completion, validation_hits)
            results.append({
                "policy": name,
                "release_wait_minutes": wait,
                "worst_hits": int(counts.min()),
                "worst_lower95_adjusted": round(float(lower.min()), 6),
                "pass": bool((counts >= validation_hits).all()),
                "required_make_up_seconds": max(0, deadline - shift_seconds),
                "conditions": [{"station": station, "stop_start_minute": start, "hits": int(hits)}
                               for (station, start), hits in zip(conditions, counts, strict=True)],
            })

    locked_result = next((row["pass"] for row in results if row["policy"] == "selected"
                          and row["release_wait_minutes"] == locked_wait), None)
    return {
        "case": case.name,
        "station_tasks": plan,
        "nominal_loads": loads,
        "layouts": layouts,
        "shift_seconds": shift_seconds,
        "target_units": target,
        "required_reliability": REQUIRED_RELIABILITY,
        "task_cv_pct": variation[0],
        "shift_cv_pct": variation[1],
        "mechanical_stop_minutes": MECHANICAL_STOP_MINUTES,
        "design": {"seed": DESIGN_SEED, "runs": RUNS, "starts": DESIGN_STARTS, "alpha_per_condition": design_alpha,
                   "required_hits": design_hits, "maximum_search_minutes": max_search_minutes,
                   "locked_wait_minutes": locked_wait,
                   "boundary_checks": [{"release_wait_minutes": wait, "required_seconds": design_required_time(wait)}
                                       for wait in design_checks]},
        "validation": {"seed": VALIDATION_SEED, "runs": RUNS, "starts": VALIDATION_STARTS,
                       "release_waits_minutes": waits, "alpha_per_condition": validation_alpha,
                       "required_hits": validation_hits, "locked_wait_pass": locked_result, "results": results},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "restart_release.json")
    args = parser.parse_args()
    report = run_study()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Design release limit: {report['design']['locked_wait_minutes']} min; "
          f"held-out pass: {report['validation']['locked_wait_pass']}")


if __name__ == "__main__":
    main()

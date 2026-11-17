"""Screen buffer layouts and confirm a shift target after a single station outage."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from math import ceil
from pathlib import Path

import numpy as np
from scipy.stats import beta

from line_balance import ROOT, Case, verify
from optimize_buffers import Layout, scenarios, workloads_for
from shift_capacity import REQUIRED_RELIABILITY, SHIFT_SECONDS, demand_completion_times

MAX_SLOTS, MAX_PER_GAP = 3, 2
TARGET_UNITS = 1780
DESIGN_RUNS, VALIDATION_RUNS = 120, 300
DESIGN_SEED, VALIDATION_SEED = 20261116, 20261117
TASK_CV, SHIFT_CV = 20, 3
OUTAGE_STARTS = (120, 300)
DESIGN_STOP = 30
STRESS_STOPS = (0, 20, DESIGN_STOP, 45, 60)


def minimum_hits(runs: int, alpha: float) -> int:
    """Return the first count whose exact one-sided lower bound meets the gate."""
    counts = np.arange(1, runs + 1)
    lower = beta.ppf(alpha, counts, runs - counts + 1)
    return int(next(count for count, bound in zip(counts, lower) if bound >= REQUIRED_RELIABILITY))


def completion_matrix(work: np.ndarray, layout: Layout, stop_minutes: int) -> np.ndarray:
    """Use the same task-time draws for all station/start combinations."""
    if stop_minutes == 0:
        return demand_completion_times(work, layout)[None, :]
    return np.stack([
        demand_completion_times(work, layout, outage=(station, start * 60, (start + stop_minutes) * 60))
        for station, start in product(range(work.shape[1]), OUTAGE_STARTS)
    ])


def support_time(completions: np.ndarray, hits: int) -> int:
    """Whole-second deadline needed to achieve the hit count in every scenario."""
    return ceil(float(np.partition(completions, hits - 1, axis=1)[:, hits - 1].max()))


def run_study() -> tuple[list[dict], dict]:
    records = json.loads((ROOT / "data" / "benchmarks.json").read_text(encoding="utf-8"))
    case = Case(**next(record for record in records if record["name"] == "Jackson"))
    baseline = json.loads((ROOT / "results" / "jackson_buffer_decision.json").read_text(encoding="utf-8"))
    reference = baseline["designs"]["optimized"]
    plan = tuple(tuple(group) for group in reference["station_tasks"])
    initial = tuple(reference["buffer_layout"])
    loads = verify(case, baseline["integer_design_cycle"], plan)
    conditions = list(product(range(1, len(plan) + 1), OUTAGE_STARTS))
    layouts = [layout for layout in product(range(MAX_PER_GAP + 1), repeat=len(plan) - 1)
               if sum(layout) <= MAX_SLOTS]
    alpha = 0.05 / len(conditions)

    def sample(runs: int, seed: int) -> np.ndarray:
        _, times = next(scenarios(case, ((TASK_CV, SHIFT_CV),), TARGET_UNITS, runs, seed))
        return workloads_for(times, plan)

    design = sample(DESIGN_RUNS, DESIGN_SEED)
    # Design uses a pointwise gate for screening. Confirmation uses an exact
    # Bonferroni bound across the predefined scenarios for the fixed selection.
    design_hits = minimum_hits(DESIGN_RUNS, 0.05)
    deadlines = {layout: support_time(completion_matrix(design, layout, DESIGN_STOP), design_hits)
                 for layout in layouts}
    eligible = [layout for layout in layouts if deadlines[layout] <= SHIFT_SECONDS]
    if eligible:
        chosen = min(eligible, key=lambda layout: (sum(layout), deadlines[layout], layout))
    else:
        chosen = min(layouts, key=lambda layout: (deadlines[layout], sum(layout), layout))

    validation = sample(VALIDATION_RUNS, VALIDATION_SEED)
    validation_hits = minimum_hits(VALIDATION_RUNS, alpha)
    rows, summary = [], []
    comparisons = {"incumbent": initial, "selected": chosen}
    for name, layout in comparisons.items():
        for stop in STRESS_STOPS:
            times = completion_matrix(validation, layout, stop)
            hits = np.count_nonzero(times <= SHIFT_SECONDS, axis=1)
            lower = [float(beta.ppf(alpha, count, VALIDATION_RUNS - count + 1)) if count else 0.0
                     for count in hits]
            deadline = support_time(times, validation_hits)
            for (station, start), count, bound, completion in zip(
                conditions if stop else [(0, 0)], hits, lower, times, strict=True,
            ):
                rows.append({
                    "plan": name, "stop_minutes": stop, "station": station, "start_minute": start,
                    "hits": int(count), "lower95_adjusted": round(bound, 6),
                    "median_completion_seconds": round(float(np.median(completion)), 2),
                    "required_deadline_seconds": support_time(completion[None, :], validation_hits),
                })
            summary.append({
                "plan": name, "stop_minutes": stop, "worst_hits": int(hits.min()),
                "worst_lower95_adjusted": round(min(lower), 6),
                "passes": min(lower) >= REQUIRED_RELIABILITY,
                "extra_seconds": max(0, deadline - SHIFT_SECONDS),
            })

    return rows, {
        "case": case.name, "station_tasks": plan, "nominal_loads": loads, "target_units": TARGET_UNITS,
        "shift_seconds": SHIFT_SECONDS, "stop_start_minutes": OUTAGE_STARTS, "design_stop_minutes": DESIGN_STOP,
        "task_cv_pct": TASK_CV, "shift_cv_pct": SHIFT_CV, "required_reliability": REQUIRED_RELIABILITY,
        "max_slots": MAX_SLOTS, "max_per_gap": MAX_PER_GAP,
        "design_seed": DESIGN_SEED, "design_runs": DESIGN_RUNS, "design_min_hits": design_hits,
        "design_candidates": [{"layout": layout, "slots": sum(layout), "deadline_seconds": deadlines[layout]}
                              for layout in layouts],
        "design_feasible_count": len(eligible), "layouts": comparisons,
        "validation_seed": VALIDATION_SEED, "validation_runs": VALIDATION_RUNS,
        "validation_scenario_alpha": alpha, "validation_min_hits": validation_hits, "validation": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    rows, report = run_study()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "downtime_scenarios.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "downtime_decision.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for row in report["validation"]:
        print(row)


if __name__ == "__main__":
    main()

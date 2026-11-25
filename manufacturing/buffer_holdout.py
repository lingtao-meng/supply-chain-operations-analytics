"""Compare buffer placement rules on Jackson and transfer the protocol to Sawyer."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from math import ceil
from pathlib import Path

import numpy as np
from scipy.stats import beta

from downtime_recovery import (
    DESIGN_RUNS, DESIGN_STOP, MAX_PER_GAP, MAX_SLOTS, OUTAGE_STARTS, SHIFT_CV,
    TARGET_UNITS, TASK_CV, VALIDATION_RUNS, completion_matrix, minimum_hits, support_time,
)
from line_balance import ROOT, Case, verify
from optimize_buffers import Layout, Plan, scenarios, workloads_for
from shift_capacity import (
    REQUIRED_RELIABILITY, SHIFT_SECONDS, demand_completion_times, recovery_shift, station_trace,
)

SLOTS = 2
HOLDOUT_STARTS, HOLDOUT_STOPS = (90, 330), (25, 40)
JACKSON_SEEDS, SAWYER_SEEDS = (20261116, 20261119), (20261118, 20261120)
SAWYER_PLAN = ((2, 11, 12, 13), (1, 3, 4, 5), (16, 17, 14, 20), (10, 6, 7, 8, 24),
               (18, 15, 9, 21), (22, 25, 26), (23, 27), (19, 28, 29, 30))
SAWYER_TARGET = 520
BOOTSTRAP_RUNS = 2000


def rule_layout(loads: list[int], rule: str) -> Layout:
    """Place two single slots evenly or in the two gaps closest to the first bottleneck."""
    gaps = len(loads) - 1
    if rule == "even":
        positions = [int((slot + 0.5) * gaps / SLOTS) for slot in range(SLOTS)]
    else:
        bottleneck = loads.index(max(loads))
        positions = sorted(range(gaps), key=lambda gap: (abs(gap + 0.5 - bottleneck), gap))[:SLOTS]
    return tuple(int(gap in positions) for gap in range(gaps))


def workloads(case: Case, plan: Plan, jobs: int, runs: int, seed: int) -> np.ndarray:
    _, times = next(scenarios(case, ((TASK_CV, SHIFT_CV),), jobs, runs, seed))
    return workloads_for(times, plan)


def screen(work: np.ndarray, max_slots: int) -> dict[Layout, int]:
    hits = minimum_hits(work.shape[-1], 0.05)
    gaps = work.shape[1] - 1
    return {
        layout: support_time(completion_matrix(work, layout, DESIGN_STOP), hits)
        for slots in range(max_slots + 1)
        for layout in product(range(min(MAX_PER_GAP, slots) + 1), repeat=gaps)
        if sum(layout) == slots
    }


def evaluate(
    case: Case, plan: Plan, target: int, seeds: tuple[int, int],
) -> tuple[list[dict], dict, np.ndarray]:
    loads = verify(case, case.reference_cycle, plan)
    design = workloads(case, plan, target, DESIGN_RUNS, seeds[0])
    deadlines = screen(design, SLOTS)
    selected = min((layout for layout in deadlines if sum(layout) == SLOTS),
                   key=lambda layout: (deadlines[layout], layout))
    layouts = {rule: rule_layout(loads, rule) for rule in ("even", "bottleneck")}
    layouts["optimized"] = selected
    jobs = 2 * ceil(SHIFT_SECONDS / max(loads))
    validation = workloads(case, plan, jobs, VALIDATION_RUNS, seeds[1])
    conditions = list(product(range(len(plan)), HOLDOUT_STARTS, HOLDOUT_STOPS))
    alpha = 0.05 / (len(conditions) * len(layouts))
    required_hits = minimum_hits(VALIDATION_RUNS, alpha)
    rows = []
    evaluated = {}
    shortfalls = {}
    for name, layout in layouts.items():
        for station, start, length in conditions:
            key = layout, station, start, length
            if key not in evaluated:
                outage = station, start * 60, (start + length) * 60
                finish = demand_completion_times(validation[:target], layout, outage=outage)
                counts, wip = recovery_shift(validation, layout, SHIFT_SECONDS, outage)
                hits = int(np.count_nonzero(counts >= target))
                if not np.array_equal(counts >= target, finish <= SHIFT_SECONDS):
                    raise AssertionError("target completion and shift counts disagree")
                shortfalls[key] = np.maximum(target - counts, 0)
                lower = float(beta.ppf(alpha, hits, VALIDATION_RUNS - hits + 1)) if hits else 0.0
                evaluated[key] = {
                    "hits": hits, "lower95_adjusted": round(lower, 6),
                    "passes": lower >= REQUIRED_RELIABILITY,
                    "mean_output": round(float(counts.mean()), 3),
                    "mean_shortfall": round(float(shortfalls[key].mean()), 3),
                    "extra_seconds": max(0, support_time(finish[None, :], required_hits) - SHIFT_SECONDS),
                    "mean_waiting_wip": round(float(wip.mean()), 4),
                }
            rows.append({"case": case.name, "policy": name, "station": station + 1,
                         "start_minute": start, "stop_minutes": length, **evaluated[key]})

    separate_durations = []
    for name in layouts:
        for length in HOLDOUT_STOPS:
            block = [row for row in rows if row["policy"] == name and row["stop_minutes"] == length]
            separate_durations.append({
                "policy": name, "stop_minutes": length,
                "worst_hits": min(row["hits"] for row in block),
                "worst_lower95_adjusted": min(row["lower95_adjusted"] for row in block),
                "worst_mean_shortfall": max(row["mean_shortfall"] for row in block),
                "max_extra_seconds": max(row["extra_seconds"] for row in block),
                "mean_waiting_wip_equal_scenarios": round(float(np.mean(
                    [row["mean_waiting_wip"] for row in block])), 4),
                "passes_every_condition": all(row["passes"] for row in block),
            })
    bootstrap_seed = seeds[1] + 10_000
    resamples = np.random.default_rng(bootstrap_seed).integers(
        VALIDATION_RUNS, size=(BOOTSTRAP_RUNS, VALIDATION_RUNS),
    )
    differences = []
    for name in ("even", "bottleneck"):
        for length in HOLDOUT_STOPS:
            by_shift = np.mean([
                shortfalls[layouts[name], station, start, length] - shortfalls[selected, station, start, length]
                for station, start, stop in conditions if stop == length
            ], axis=0)
            low, high = np.quantile(by_shift[resamples].mean(axis=1), (0.025, 0.975))
            differences.append({
                "baseline": name, "stop_minutes": length,
                "mean_shortfall_reduction": round(float(by_shift.mean()), 3),
                "paired_bootstrap95": [round(float(bound), 3) for bound in (low, high)],
            })
    return rows, {
        "case": case.name, "target": target, "nominal_loads": loads, "plan": plan,
        "design_seed": seeds[0], "holdout_seed": seeds[1], "design_candidates": len(deadlines),
        "holdout_conditions": len(conditions), "holdout_alpha": alpha,
        "holdout_required_hits": required_hits,
        "bootstrap_seed": bootstrap_seed, "bootstrap_runs": BOOTSTRAP_RUNS,
        "policies": {name: {"layout": layout, "design_deadline_seconds": deadlines[layout]}
                     for name, layout in layouts.items()},
        "holdout": separate_durations, "paired_shortfall_comparisons": differences,
    }, validation


def phase_seconds(trace: np.ndarray, station: int, outage, begin: float, end: float) -> tuple[float, ...]:
    """Time in processing, blocked, waiting for entry, and stopped over a display interval."""
    admission, finish, departure = trace[:, station].T
    job_entry_gap_starts = np.r_[0, departure[:-1]]

    def overlap(left, right, low, high) -> float:
        return float(np.maximum(0, np.minimum(right, high) - np.maximum(left, low)).sum())

    phases = [(admission, finish), (finish, departure), (job_entry_gap_starts, admission)]
    stop_station, stop_start, stop_end = outage
    stop_seconds = overlap(stop_start, stop_end, begin, end) if station == stop_station else 0.0
    seconds = []
    for left, right in phases:
        value = overlap(left, right, begin, end)
        if stop_seconds:
            value -= overlap(left, right, max(begin, stop_start), min(end, stop_end))
        seconds.append(max(0, value))
    return (*seconds, stop_seconds)


def write_timeline(path: Path, work: np.ndarray, report: dict) -> None:
    """Replay the first held-out Jackson shift for an announced station and start time."""
    station, start, length = 0, HOLDOUT_STARTS[-1], HOLDOUT_STOPS[-1]
    outage = station, start * 60, (start + length) * 60
    first, last = start - 10, start + length + 10
    width, x_start, bin_minutes = 780, 110, 5
    pixels_per_minute = (width - x_start - 34) / (last - first)
    bar = pixels_per_minute * bin_minutes
    colors = ("#527186", "#af7b52", "#dce0e1", "#37383c")
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="324" viewBox="0 0 {width} 324">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="Georgia,serif" font-size="13" fill="#263341">',
             f'<text x="20" y="25" font-size="18" font-weight="normal">Jackson | Station {station + 1} '
             f'outage, {start}-{start + length} min</text>']
    for column, (color, label) in enumerate(zip(colors, ("Processing", "Blocked", "Waiting", "Stopped"))):
        x = 110 + column * 135
        parts.append(f'<rect x="{x}" y="38" width="12" height="12" fill="{color}"/>'
                     f'<text x="{x + 18}" y="49">{label}</text>')
    for panel, (name, label) in enumerate((("bottleneck", "Bottleneck rule"), ("optimized", "Optimized placement"))):
        layout = tuple(report["policies"][name]["layout"])
        trace = station_trace(work[:, :, :1], layout, outage)
        top = 67 + panel * 115
        parts.append(f'<text x="20" y="{top}">{label} | {" / ".join(map(str, layout))}</text>')
        for index in range(work.shape[1]):
            y = top + 8 + index * 20
            parts.append(f'<text x="46" y="{y + 12}">S{index + 1}</text>')
            for minute in range(first, last, bin_minutes):
                x = x_start + (minute - first) * pixels_per_minute
                bin_seconds = bin_minutes * 60
                phases = phase_seconds(trace, index, outage, minute * 60, minute * 60 + bin_seconds)
                if not np.isclose(sum(phases), bin_seconds):
                    raise AssertionError("timeline phases do not cover the interval")
                for seconds, color in zip(phases, colors, strict=True):
                    size = seconds / bin_seconds * bar
                    if size >= 0.015:
                        parts.append(f'<rect x="{x:.3f}" y="{y}" width="{size:.3f}" height="14" fill="{color}"/>')
                    x += size
    for minute in range(first, last + 1, 10):
        x = x_start + (minute - first) * pixels_per_minute
        parts.append(f'<text x="{x:.1f}" y="287" text-anchor="middle">{minute}</text>')
    parts.append(f'<text x="430" y="310" text-anchor="middle">Minutes after shift start | '
                 f'{bin_minutes}-minute state shares</text></g></svg>')
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    base = json.loads((ROOT / "results" / "downtime_decision.json").read_text(encoding="utf-8"))
    cases = json.loads((ROOT / "data" / "benchmarks.json").read_text(encoding="utf-8"))
    jackson = Case(**next(case for case in cases if case["name"] == "Jackson"))
    sawyer = Case(**json.loads((ROOT / "data" / "sawyer.json").read_text(encoding="utf-8")))
    plan = tuple(tuple(group) for group in base["station_tasks"])
    jackson_rows, jackson_summary, trace_work = evaluate(jackson, plan, TARGET_UNITS, JACKSON_SEEDS)
    sawyer_rows, sawyer_summary, _ = evaluate(sawyer, SAWYER_PLAN, SAWYER_TARGET, SAWYER_SEEDS)
    if tuple(jackson_summary["policies"]["optimized"]["layout"]) != tuple(base["layouts"]["selected"]):
        raise AssertionError("the fixed two-slot Jackson comparison differs from the registered layout")

    sensitivity = []
    design = workloads(jackson, plan, TARGET_UNITS, DESIGN_RUNS, JACKSON_SEEDS[0])
    for target in (1720, 1750, TARGET_UNITS):
        deadlines = screen(design[:target], MAX_SLOTS)
        row = {"target": target, "budgets": []}
        for slots in range(MAX_SLOTS + 1):
            options = [layout for layout in deadlines if sum(layout) <= slots]
            best = min(options, key=lambda layout: (deadlines[layout], sum(layout), layout))
            row["budgets"].append({"max_slots": slots, "layout": best,
                                   "extra_seconds": max(0, deadlines[best] - SHIFT_SECONDS)})
        sensitivity.append(row)

    report = {
        "shift_seconds": SHIFT_SECONDS, "buffer_slots": SLOTS,
        "task_cv_pct": TASK_CV, "shift_cv_pct": SHIFT_CV, "required_reliability": REQUIRED_RELIABILITY,
        "design_runs": DESIGN_RUNS, "design_start_minutes": OUTAGE_STARTS, "design_stop_minutes": DESIGN_STOP,
        "holdout_runs": VALIDATION_RUNS, "holdout_start_minutes": HOLDOUT_STARTS, "holdout_stop_minutes": HOLDOUT_STOPS,
        "cases": [jackson_summary, sawyer_summary], "jackson_design_target_sensitivity": sensitivity,
    }
    rows = jackson_rows + sawyer_rows
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "buffer_holdout.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "buffer_holdout.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_timeline(args.output_dir / "buffer_holdout_timeline.svg", trace_work, jackson_summary)
    for case in report["cases"]:
        print(case["case"], case["policies"], case["holdout"])


if __name__ == "__main__":
    main()

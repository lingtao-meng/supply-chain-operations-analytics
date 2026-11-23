"""Choose a fixed or queue-triggered rework support window under equal staffing time."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from heapq import heapify, heappop, heapreplace
from pathlib import Path

import numpy as np
from scipy.stats import beta

from downtime_recovery import SHIFT_CV, TASK_CV, minimum_hits
from good_output import (
    DIRECT_SCRAP, FAST_FACTOR, FIRST_PASS, INSPECTION_SECONDS, REPAIR_SECONDS, REWORK_PASS,
    STRESS_DEFECT_PCT, assembly_quality_batches, fifo_finishes, metric, paired_gain, quality_routes,
)
from line_balance import ROOT, Case, reference_plans
from shift_capacity import DEMAND, REQUIRED_RELIABILITY, SHIFT_SECONDS

SUPPORT_MINUTES, CHECK_MINUTES, RESPONSE_MINUTES = 90, 5, 10
PULSE_MINUTES, FAILURE_SHARE_PCT = 120, 65
DESIGN_STARTS = {"early": 30, "middle": 130, "late": 240}
HOLDOUT_STARTS = {"uniform": None, "early": 55, "middle": 165, "late": 285}
DESIGN_RUNS, HOLDOUT_RUNS = 300, 600
DESIGN_SEEDS, HOLDOUT_SEEDS = (2026112213, 2026112214), (2026112215, 2026112216)
FIXED_MINUTES = tuple(range(0, SHIFT_SECONDS // 60 - SUPPORT_MINUTES + 1, 15))
THRESHOLDS = (4, 8, 12, 16, 20, 24)


@dataclass(frozen=True)
class Policy:
    name: str
    start_minute: int | None = None
    threshold: int | None = None
    support_minutes: int = SUPPORT_MINUTES
    support_people = 1


def concentrate_failures(
    routes: np.ndarray, inspection_finishes: np.ndarray, pulse_start: int,
    positions: np.ndarray, failures: np.ndarray,
) -> np.ndarray:
    """Move failures between inspection positions while retaining all disposition counts."""
    in_pulse = (inspection_finishes >= pulse_start * 60) & (
        inspection_finishes < (pulse_start + PULSE_MINUTES) * 60
    )
    inside = positions[in_pulse[positions]]
    outside = positions[~in_pulse[positions]]
    desired = (len(failures) * FAILURE_SHARE_PCT + 50) // 100
    inside_count = min(len(inside), len(failures), max(desired, len(failures) - len(outside)))
    shifted = np.full_like(routes, FIRST_PASS)
    shifted[inside[:inside_count]] = failures[:inside_count]
    shifted[outside[:len(failures) - inside_count]] = failures[inside_count:]
    return shifted


def observed_waiting(arrivals: np.ndarray, checks: np.ndarray) -> np.ndarray:
    """Count rework waiting immediately after core staff starts available jobs at each check."""
    core_starts = fifo_finishes(arrivals, 1, REPAIR_SECONDS) - REPAIR_SECONDS
    return np.searchsorted(arrivals, checks, side="right") - np.searchsorted(core_starts, checks, side="right")


def support_start(checks: np.ndarray, waiting: np.ndarray, threshold: int) -> tuple[int, bool]:
    crossings = np.flatnonzero(waiting >= threshold)
    if len(crossings):
        return int(checks[crossings[0]]) + 60 * RESPONSE_MINUTES, True
    return SHIFT_SECONDS - 60 * SUPPORT_MINUTES, False


def dispatch_rework_windows(
    arrivals: np.ndarray, windows: tuple[tuple[int, int], ...],
) -> tuple[np.ndarray, int]:
    """Assign FIFO jobs to the core server or a helper whose window contains the job."""
    helpers = [(float(start), start + duration - REPAIR_SECONDS) for start, duration in windows
               if duration >= REPAIR_SECONDS]
    if not helpers:
        return fifo_finishes(arrivals, 1, REPAIR_SECONDS), 0
    heapify(helpers)
    core_free, helper_jobs = 0.0, 0
    finishes = np.empty(len(arrivals))
    for job, arrival in enumerate(arrivals.tolist()):
        core_start = max(arrival, core_free)
        while helpers and max(arrival, helpers[0][0]) > helpers[0][1]:
            heappop(helpers)
        if helpers and max(arrival, helpers[0][0]) < core_start:
            available, last_start = helpers[0]
            finish = max(arrival, available) + REPAIR_SECONDS
            finishes[job] = finish
            heapreplace(helpers, (finish, last_start))
            helper_jobs += 1
        else:
            core_free = finishes[job] = core_start + REPAIR_SECONDS
    return finishes, helper_jobs


def dispatch_rework(arrivals: np.ndarray, start: int, duration: int, helpers: int = 1) -> tuple[np.ndarray, int]:
    return dispatch_rework_windows(arrivals, ((start, duration),) * helpers)


def assemble_inputs(runs: int, seeds: tuple[int, int], pulse_starts: dict[str, int | None]):
    records = json.loads((ROOT / "data" / "benchmarks.json").read_text(encoding="utf-8"))
    case = Case(**next(record for record in records if record["name"] == "Mitchell"))
    plan = reference_plans(case, 15)["MILP"]
    buffers = (1,) * (len(plan) - 1)
    shuffle_rng = np.random.default_rng(seeds[1])
    for arrivals_by_speed, draws in assembly_quality_batches(case, plan, buffers, {FAST_FACTOR}, runs, seeds[0]):
        for exits, rep_draws in zip(arrivals_by_speed[FAST_FACTOR].T, draws, strict=True):
            gross = int(np.searchsorted(exits, SHIFT_SECONDS, side="right"))
            passes = fifo_finishes(exits[:gross], 1, INSPECTION_SECONDS)
            inspected = int(np.searchsorted(passes, SHIFT_SECONDS, side="right"))
            finishes = passes[:inspected]
            routes = quality_routes(rep_draws[:inspected], STRESS_DEFECT_PCT)
            positions = shuffle_rng.permutation(inspected)
            failures = shuffle_rng.permutation(routes[routes != FIRST_PASS])
            yield gross, finishes, {
                name: (routes if start is None else concentrate_failures(routes, finishes, start, positions, failures))
                for name, start in pulse_starts.items()
            }


def evaluate(inputs, policies: tuple[Policy, ...]) -> dict[tuple[str, str], list[dict]]:
    latest_call = SHIFT_SECONDS - 60 * (SUPPORT_MINUTES + RESPONSE_MINUTES)
    checks = np.arange(0, latest_call + 1, CHECK_MINUTES * 60)
    cases = {}
    for gross, inspection_finishes, scenarios in inputs:
        for condition, routes in scenarios.items():
            repairable = routes >= REWORK_PASS
            arrivals = inspection_finishes[repairable]
            successes = routes[repairable] == REWORK_PASS
            waiting = observed_waiting(arrivals, checks)
            first_pass = int(np.count_nonzero(routes == FIRST_PASS))
            direct_scrap = int(np.count_nonzero(routes == DIRECT_SCRAP))
            potential_good = int(np.count_nonzero(successes))
            for policy in policies:
                if policy.threshold is None:
                    start, triggered = 60 * policy.start_minute, False
                else:
                    start, triggered = support_start(checks, waiting, policy.threshold)
                finishes, helper_jobs = dispatch_rework(
                    arrivals, start, 60 * policy.support_minutes, policy.support_people,
                )
                completed = finishes <= SHIFT_SECONDS
                repaired_good = int(np.count_nonzero(completed & successes))
                scrap = direct_scrap + int(np.count_nonzero(completed & ~successes))
                row = {
                    "gross": gross, "inspected": len(routes), "first_pass": first_pass,
                    "direct_scrap": direct_scrap, "rework_candidates": len(arrivals), "rework_released": repaired_good,
                    "potential_rework_good": potential_good,
                    "released": first_pass + repaired_good, "scrap": scrap,
                    "pending_inspection": gross - len(routes),
                    "pending_rework": len(arrivals) - int(np.count_nonzero(completed)),
                    "shift_mean_open_rework": round(float((np.minimum(finishes, SHIFT_SECONDS) - arrivals).sum()
                                                          / SHIFT_SECONDS), 6),
                    "reserved_minutes": policy.support_minutes * policy.support_people,
                    "helper_busy_minutes": helper_jobs * REPAIR_SECONDS / 60,
                    "support_start_minute": start / 60 if policy.support_minutes else None,
                    "queue_triggered": triggered,
                }
                cases.setdefault((policy.name, condition), []).append(row)
    return cases


def design_score(
    rows: dict[tuple[str, str], list[dict]], name: str, scenarios: tuple[str, ...] = tuple(DESIGN_STARTS),
) -> tuple[int, int, int]:
    released = [metric(rows[name, scenario], "released") for scenario in scenarios]
    minimum_hits = min(int(np.count_nonzero(output >= DEMAND)) for output in released)
    max_shortfall = max(int(np.maximum(DEMAND - output, 0).sum()) for output in released)
    return -minimum_hits, max_shortfall, -sum(int(output.sum()) for output in released)


def summarize_case(name: str, scenario: str, rows: list[dict], alpha: float) -> dict:
    output = metric(rows, "released")
    hits = int(np.count_nonzero(output >= DEMAND))
    lower = float(beta.ppf(alpha, hits, len(rows) - hits + 1)) if hits else 0.0
    start_times = [row["support_start_minute"] for row in rows if row["support_start_minute"] is not None]
    return {
        "policy": name, "condition": scenario, "hits": hits, "lower95_adjusted": round(lower, 6),
        "passes": lower >= REQUIRED_RELIABILITY, "mean_released": round(float(output.mean()), 3),
        "mean_shortfall": round(float(np.maximum(DEMAND - output, 0).mean()), 3),
        "mean_pending_rework": round(float(metric(rows, "pending_rework").mean()), 3),
        "mean_open_rework": round(float(metric(rows, "shift_mean_open_rework").mean()), 3),
        "mean_helper_busy_minutes": round(float(metric(rows, "helper_busy_minutes").mean()), 3),
        "median_support_start_minute": float(np.median(start_times)) if start_times else None,
        "queue_triggered_shifts": int(metric(rows, "queue_triggered").sum()),
    }


def run_study() -> tuple[dict, list[dict]]:
    fixed = tuple(Policy(f"fixed_{minute:03d}", start_minute=minute) for minute in FIXED_MINUTES)
    adaptive = tuple(Policy(f"trigger_{threshold:02d}", threshold=threshold) for threshold in THRESHOLDS)
    design = evaluate(assemble_inputs(DESIGN_RUNS, DESIGN_SEEDS, DESIGN_STARTS), (*fixed, *adaptive))
    chosen_fixed = min(fixed, key=lambda policy: design_score(design, policy.name))
    chosen_adaptive = min(adaptive, key=lambda policy: design_score(design, policy.name))
    policies = (
        Policy("no_support", start_minute=0, support_minutes=0),
        Policy("fixed", start_minute=chosen_fixed.start_minute),
        Policy("adaptive", threshold=chosen_adaptive.threshold),
        Policy("full_shift", start_minute=0, support_minutes=SHIFT_SECONDS // 60),
    )
    holdout = evaluate(assemble_inputs(HOLDOUT_RUNS, HOLDOUT_SEEDS, HOLDOUT_STARTS), policies)
    alpha = 0.05 / (len(policies) * len(HOLDOUT_STARTS))
    detail, raw = [], []
    for (name, scenario), rows in holdout.items():
        detail.append(summarize_case(name, scenario, rows, alpha))
        raw.extend({"scenario": scenario, "policy": name, "replication": i, **row}
                   for i, row in enumerate(rows, start=1))
    comparisons = {
        scenario: paired_gain(metric(holdout["adaptive", scenario], "released"),
                              metric(holdout["fixed", scenario], "released"))
        for scenario in HOLDOUT_STARTS
    }
    report = {
        "case": "Mitchell", "target_released": DEMAND, "shift_seconds": SHIFT_SECONDS,
        "task_cv_pct": TASK_CV, "shift_cv_pct": SHIFT_CV, "first_defect_pct": STRESS_DEFECT_PCT,
        "pulse_minutes": PULSE_MINUTES, "pulse_failure_share_pct": FAILURE_SHARE_PCT,
        "inspection_seconds": INSPECTION_SECONDS, "rework_seconds": REPAIR_SECONDS,
        "reserved_minutes": SUPPORT_MINUTES, "review_minutes": CHECK_MINUTES, "response_minutes": RESPONSE_MINUTES,
        "design": {"runs": DESIGN_RUNS, "seeds": DESIGN_SEEDS, "pulse_starts": DESIGN_STARTS,
                   "candidates": [{**asdict(policy), "score": design_score(design, policy.name)}
                                  for policy in (*fixed, *adaptive)]},
        "holdout": {"runs": HOLDOUT_RUNS, "seeds": HOLDOUT_SEEDS, "pulse_starts": HOLDOUT_STARTS,
                    "alpha_per_scenario_policy": alpha, "required_hits": minimum_hits(HOLDOUT_RUNS, alpha),
                    "policies": [asdict(policy) for policy in policies], "results": detail,
                    "paired_adaptive_gain_vs_fixed": comparisons},
    }
    return report, raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    report, rows = run_study()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "rework_support_holdout.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "rework_support.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

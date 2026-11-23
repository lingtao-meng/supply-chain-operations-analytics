"""Compare gross assembly completions with quality-released units by shift end."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from math import ceil, sqrt
from pathlib import Path

import numpy as np
from scipy.stats import beta, t

from downtime_recovery import SHIFT_CV, TASK_CV, minimum_hits
from line_balance import ROOT, Case, reference_plans, verify
from optimize_buffers import workloads_for
from shift_capacity import DEMAND, REQUIRED_RELIABILITY, SHIFT_SECONDS, lognormal_multiplier, shift_exit_times

INSPECTION_SECONDS, REPAIR_SECONDS = 12, 120
REPAIRABLE_PROBABILITY, REPAIR_SUCCESS_PROBABILITY = 0.8, 0.85
FAST_FACTOR = 0.93
BASE_DEFECT_PCT, STRESS_DEFECT_PCT = 4, 18
FIRST_PASS, DIRECT_SCRAP, REWORK_PASS, REWORK_SCRAP = range(4)
DESIGN_RATES = tuple(range(4, 25, 2))
DESIGN_RUNS, HOLDOUT_RUNS, BATCH_SIZE = 300, 600, 100
DESIGN_SEED, HOLDOUT_SEED = 2026112211, 2026112212
SHIFT_METRICS = (
    "gross", "first_inspected", "first_pass", "routed_to_rework", "rework_pass",
    "direct_scrap", "rework_scrap", "scrap", "pending_inspection", "pending_rework", "released",
)


@dataclass(frozen=True)
class Policy:
    name: str
    time_factor: float
    defect_pct: int
    inspectors: int = 1
    rework_servers: int = 1


STANDARD = Policy("standard", 1.0, BASE_DEFECT_PCT)
HOLDOUT_POLICIES = (
    STANDARD,
    Policy("fast_stable", FAST_FACTOR, BASE_DEFECT_PCT),
    Policy("fast_drift", FAST_FACTOR, STRESS_DEFECT_PCT),
    Policy("fast_extra_inspection", FAST_FACTOR, STRESS_DEFECT_PCT, inspectors=2),
    Policy("fast_extra_rework", FAST_FACTOR, STRESS_DEFECT_PCT, rework_servers=2),
)


def fifo_finishes(arrivals: np.ndarray, servers: int, service_seconds: int) -> np.ndarray:
    """Complete a FIFO queue with identical servers and constant service time."""
    finishes = np.empty(arrivals.shape, dtype=float)
    for offset in range(min(servers, len(arrivals))):
        stream = arrivals[offset::servers]
        position = np.arange(len(stream))
        # One stream per server unrolls finish[j] = max(arrival[j], finish[j-servers]) + service.
        finishes[offset::servers] = (
            np.maximum.accumulate(stream - position * service_seconds) + (position + 1) * service_seconds
        )
    return finishes


def quality_routes(draws: np.ndarray, defect_pct: int) -> np.ndarray:
    """Assign each inspected unit its first result and eventual repair outcome."""
    failed = draws[:, 0] < defect_pct / 100
    repairable = failed & (draws[:, 1] < REPAIRABLE_PROBABILITY)
    routes = np.full(len(draws), FIRST_PASS, dtype=np.int8)
    routes[failed] = DIRECT_SCRAP
    routes[repairable] = np.where(draws[repairable, 2] < REPAIR_SUCCESS_PROBABILITY, REWORK_PASS, REWORK_SCRAP)
    return routes


def quality_shift(
    arrivals: np.ndarray, draws: np.ndarray, policy: Policy, horizon: float,
) -> dict[str, int]:
    """Follow each unique assembly exit through first inspection and at most one repair."""
    gross = int(np.searchsorted(arrivals, horizon, side="right"))
    first_finishes = fifo_finishes(arrivals[:gross], policy.inspectors, INSPECTION_SECONDS)
    inspected = int(np.searchsorted(first_finishes, horizon, side="right"))
    routes = quality_routes(draws[:inspected], policy.defect_pct)
    can_rework = routes >= REWORK_PASS
    routed = int(np.count_nonzero(can_rework))
    first_pass = int(np.count_nonzero(routes == FIRST_PASS))
    direct_scrap = int(np.count_nonzero(routes == DIRECT_SCRAP))
    rework_finishes = fifo_finishes(
        first_finishes[:inspected][can_rework], policy.rework_servers, REPAIR_SECONDS,
    )
    repaired = int(np.searchsorted(rework_finishes, horizon, side="right"))
    rework_pass = int(np.count_nonzero(routes[can_rework][:repaired] == REWORK_PASS))
    rework_scrap = repaired - rework_pass
    return {
        "gross": gross, "first_inspected": inspected, "first_pass": first_pass,
        "routed_to_rework": routed, "rework_pass": rework_pass,
        "direct_scrap": direct_scrap, "rework_scrap": rework_scrap,
        "scrap": direct_scrap + rework_scrap, "pending_inspection": gross - inspected,
        "pending_rework": routed - repaired, "released": first_pass + rework_pass,
    }


def assembly_quality_batches(
    case: Case, plan: list[list[int]], buffers: tuple[int, ...],
    time_factors: set[float], runs: int, seed: int,
):
    """Generate paired assembly exits and per-unit quality draws without storing all shifts."""
    max_workload = max(verify(case, 15, plan))
    longest_horizon = SHIFT_SECONDS / min(time_factors)
    jobs = 2 * ceil(longest_horizon / max_workload)
    nominal = np.asarray(case.task_times)[None, :, None]
    time_seed, quality_seed = np.random.SeedSequence(seed).spawn(2)
    time_rng = np.random.default_rng(time_seed)
    quality_rng = np.random.default_rng(quality_seed)
    for offset in range(0, runs, BATCH_SIZE):
        size = min(BATCH_SIZE, runs - offset)
        times = nominal * lognormal_multiplier(time_rng.standard_normal((jobs, len(case.task_times), size)), TASK_CV)
        times *= lognormal_multiplier(time_rng.standard_normal((1, 1, size)), SHIFT_CV)
        assembly = shift_exit_times(workloads_for(times, plan), buffers, longest_horizon)
        quality_draws = quality_rng.random((size, jobs, 3))
        # A common time scale preserves the order and buffer constraints of the fixed line.
        yield {factor: assembly * factor for factor in time_factors}, quality_draws


def simulate_policies(
    case: Case, plan: list[list[int]], buffers: tuple[int, ...],
    policies: tuple[Policy, ...], runs: int, seed: int,
) -> dict[str, list[dict[str, int]]]:
    results = {policy.name: [] for policy in policies}
    time_factors = {policy.time_factor for policy in policies}
    for arrivals_by_speed, quality_draws in assembly_quality_batches(case, plan, buffers, time_factors, runs, seed):
        for policy in policies:
            arrivals = arrivals_by_speed[policy.time_factor]
            results[policy.name].extend(
                quality_shift(arrivals[:, rep], quality_draws[rep], policy, SHIFT_SECONDS)
                for rep in range(len(quality_draws))
            )
    return results


def metric(samples: list[dict[str, int]], name: str) -> np.ndarray:
    return np.asarray([shift[name] for shift in samples])


def paired_gain(values: np.ndarray, reference: np.ndarray) -> dict:
    diff = values - reference
    mean = diff.mean()
    radius = t.ppf(0.975, len(diff) - 1) * diff.std(ddof=1) / sqrt(len(diff))
    return {"mean": round(float(mean), 3), "ci95": [round(float(mean - radius), 3), round(float(mean + radius), 3)],
            "lower_above_zero": bool(mean - radius > 0)}


def shift_summary(samples: list[dict[str, int]], baseline: np.ndarray, alpha: float, staff: int) -> dict:
    gross = metric(samples, "gross")
    released = metric(samples, "released")
    hit = released >= DEMAND
    hits = int(hit.sum())
    lower = float(beta.ppf(alpha, hits, len(samples) - hits + 1)) if hits else 0.0
    inspected = metric(samples, "first_inspected").sum()
    first_pass_yield = float(metric(samples, "first_pass").sum() / inspected) if inspected else None
    mean_released = released.mean()
    staffed_hours = staff * SHIFT_SECONDS / 3600
    return {
        "means": {name: round(float(metric(samples, name).mean()), 3) for name in SHIFT_METRICS},
        "pooled_first_pass_yield": round(first_pass_yield, 6) if first_pass_yield is not None else None,
        "gross_hits": int((gross >= DEMAND).sum()), "released_hits": hits,
        "gross_pass_released_fail": int(((gross >= DEMAND) & ~hit).sum()),
        "released_adjusted_lower": round(lower, 6), "passes": lower >= REQUIRED_RELIABILITY,
        "mean_released_shortfall": round(float(np.maximum(DEMAND - released, 0).mean()), 3),
        "staff": staff, "staffed_hours_per_100_released": round(float(100 * staffed_hours / mean_released), 3)
        if mean_released else None,
        "paired_released_gain_vs_standard": paired_gain(released, baseline),
    }


def run_study() -> tuple[dict, list[dict]]:
    cases = json.loads((ROOT / "data" / "benchmarks.json").read_text(encoding="utf-8"))
    case = Case(**next(record for record in cases if record["name"] == "Mitchell"))
    plan = reference_plans(case, 15)["MILP"]
    buffers = (1,) * (len(plan) - 1)
    screen_policies = tuple(Policy(f"screen_{rate:02d}", FAST_FACTOR, rate) for rate in DESIGN_RATES)
    design = simulate_policies(case, plan, buffers, (STANDARD, *screen_policies), DESIGN_RUNS, DESIGN_SEED)
    design_baseline = metric(design[STANDARD.name], "released")
    design_gate = minimum_hits(DESIGN_RUNS, 0.05)
    screen = []
    for policy in screen_policies:
        released = metric(design[policy.name], "released")
        hits = int((released >= DEMAND).sum())
        gain = paired_gain(released, design_baseline)
        screen.append({"defect_pct": policy.defect_pct, "released_hits": hits,
                       "paired_gain_vs_standard": gain, "eligible": hits >= design_gate and gain["lower_above_zero"]})
    locked_rate = max((row["defect_pct"] for row in screen if row["eligible"]), default=None)
    policies = HOLDOUT_POLICIES
    if locked_rate is not None:
        policies = (*policies, Policy("fast_locked_limit", FAST_FACTOR, locked_rate))

    holdout = simulate_policies(case, plan, buffers, policies, HOLDOUT_RUNS, HOLDOUT_SEED)
    holdout_baseline = metric(holdout[STANDARD.name], "released")
    alpha = 0.05 / len(policies)
    report = {
        "case": case.name, "target_released_units": DEMAND, "shift_seconds": SHIFT_SECONDS,
        "assembly_plan": plan, "assembly_buffers": buffers,
        "task_cv_pct": TASK_CV, "shift_cv_pct": SHIFT_CV, "batch_size": BATCH_SIZE,
        "inspection_seconds": INSPECTION_SECONDS, "rework_and_recheck_seconds": REPAIR_SECONDS,
        "repairable_probability": REPAIRABLE_PROBABILITY, "rework_pass_probability": REPAIR_SUCCESS_PROBABILITY,
        "design": {"runs": DESIGN_RUNS, "seed": DESIGN_SEED, "required_hits": design_gate,
                   "screen": screen, "locked_defect_pct": locked_rate},
        "holdout": {"runs": HOLDOUT_RUNS, "seed": HOLDOUT_SEED, "required_reliability": REQUIRED_RELIABILITY,
                    "alpha_per_policy": alpha, "required_hits": minimum_hits(HOLDOUT_RUNS, alpha), "policies": []},
    }
    for policy in policies:
        staff = len(plan) + policy.inspectors + policy.rework_servers
        report["holdout"]["policies"].append({
            **asdict(policy), **shift_summary(holdout[policy.name], holdout_baseline, alpha, staff),
        })
    report["holdout"]["extra_rework_vs_extra_inspection"] = paired_gain(
        metric(holdout["fast_extra_rework"], "released"), metric(holdout["fast_extra_inspection"], "released"),
    )
    rows = [
        {"policy": name, "replication": rep, **shift}
        for name, samples in holdout.items() for rep, shift in enumerate(samples, start=1)
    ]
    return report, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    report, rows = run_study()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "good_output_holdout.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "good_output.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""Stress-test shift targets when consecutive units have correlated task times."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from math import ceil, sqrt
from pathlib import Path

import numpy as np
from scipy.signal import lfilter
from scipy.stats import beta, t

from downtime_recovery import SHIFT_CV, TASK_CV, minimum_hits
from line_balance import ROOT, Case, reference_plans, verify
from optimize_buffers import Layout, Plan, workloads_for
from shift_capacity import REQUIRED_RELIABILITY, SHIFT_SECONDS, buffer_layout, lognormal_multiplier, shift_output_counts

CORRELATIONS = (0.0, 0.5, 0.9)
DESIGN_RUNS, HOLDOUT_RUNS, BATCH_SIZE = 300, 600, 100
HEADROOM_PCT, TARGET_INCREMENT = 1, 10


@dataclass(frozen=True)
class Policy:
    name: str
    plan: Plan
    layout: Layout


def correlated_noise(innovations: np.ndarray, rho: float) -> np.ndarray:
    """Stationary Gaussian AR(1) along the product axis, independently per task and shift."""
    if not 0 <= rho < 1:
        raise ValueError("rho must be in [0, 1)")
    if rho == 0:
        return innovations
    scale = sqrt(1 - rho**2)
    initial = (1 - scale) * innovations[:1]
    noise, _ = lfilter([scale], [1, -rho], innovations, axis=0, zi=initial)
    return noise


def simulate_policies(
    case: Case, cycle: int, policies: list[Policy], runs: int, seed: int,
) -> dict[tuple[str, float], np.ndarray]:
    loads = [verify(case, cycle, policy.plan) for policy in policies]
    # The existing flow simulator checks that all replications cover the shift.
    jobs = 2 * ceil(SHIFT_SECONDS / min(map(max, loads)))
    nominal = np.asarray(case.task_times)[None, :, None]
    rng = np.random.default_rng(seed)
    batches = {(policy.name, rho): [] for policy in policies for rho in CORRELATIONS}
    for offset in range(0, runs, BATCH_SIZE):
        size = min(BATCH_SIZE, runs - offset)
        innovations = rng.standard_normal((jobs, len(case.task_times), size))
        shift_noise = rng.standard_normal((1, 1, size))
        shift_multiplier = lognormal_multiplier(shift_noise, SHIFT_CV)
        for rho in CORRELATIONS:
            task_times = nominal * lognormal_multiplier(correlated_noise(innovations, rho), TASK_CV)
            task_times *= shift_multiplier
            for policy in policies:
                loads_for_policy = workloads_for(task_times, policy.plan)
                output = shift_output_counts(loads_for_policy, policy.layout, SHIFT_SECONDS)
                batches[policy.name, rho].append(output)
    return {key: np.concatenate(values) for key, values in batches.items()}


def capacity_target(counts: np.ndarray, required_hits: int) -> int:
    """Largest whole-unit target met by at least required_hits simulations."""
    index = len(counts) - required_hits
    return int(np.partition(counts, index)[index])


def planned_target(boundary: int) -> int:
    return (boundary * (100 - HEADROOM_PCT) // (100 * TARGET_INCREMENT)) * TARGET_INCREMENT


def hit_result(counts: np.ndarray, target: int, alpha: float) -> dict:
    hits = int(np.count_nonzero(counts >= target))
    lower = float(beta.ppf(alpha, hits, len(counts) - hits + 1)) if hits else 0.0
    return {
        "hits": hits, "adjusted_lower": round(lower, 6), "passes": lower >= REQUIRED_RELIABILITY,
        "mean_shortfall": round(float(np.maximum(target - counts, 0).mean()), 3),
    }


def paired_output_loss(counts: np.ndarray, independent: np.ndarray) -> dict:
    loss = independent - counts
    mean = float(loss.mean())
    margin = float(t.ppf(0.975, len(loss) - 1) * loss.std(ddof=1) / sqrt(len(loss)))
    return {"mean": round(mean, 3), "ci95": [round(mean - margin, 3), round(mean + margin, 3)]}


def evaluate(
    case: Case, cycle: int, target: int, policies: list[Policy], seeds: tuple[int, int],
) -> tuple[dict, list[dict]]:
    alpha = 0.05 / (len(policies) * len(CORRELATIONS) * 2)
    design = simulate_policies(case, cycle, policies, DESIGN_RUNS, seeds[0])
    design_gate = minimum_hits(DESIGN_RUNS, alpha)
    boundaries = {key: capacity_target(counts, design_gate) for key, counts in design.items()}
    plans = {
        policy.name: planned_target(min(boundaries[policy.name, rho] for rho in CORRELATIONS))
        for policy in policies
    }
    holdout = simulate_policies(case, cycle, policies, HOLDOUT_RUNS, seeds[1])
    report = {
        "case": case.name, "cycle": cycle, "original_target": target, "design_seed": seeds[0],
        "holdout_seed": seeds[1], "alpha_per_endpoint": alpha, "design_required_hits": design_gate,
        "holdout_required_hits": minimum_hits(HOLDOUT_RUNS, alpha), "policies": {},
    }
    for policy in policies:
        scenarios = []
        for rho in CORRELATIONS:
            counts = holdout[policy.name, rho]
            scenarios.append({
                "rho": rho, "design_boundary_units": boundaries[policy.name, rho],
                "mean_output": round(float(counts.mean()), 3), "p05_output": round(float(np.quantile(counts, 0.05)), 3),
                "original_target": hit_result(counts, target, alpha),
                "planned_target": hit_result(counts, plans[policy.name], alpha),
                "paired_loss_vs_independent": paired_output_loss(counts, holdout[policy.name, 0.0]),
            })
        report["policies"][policy.name] = {
            "station_tasks": policy.plan, "station_loads": verify(case, cycle, policy.plan),
            "layout": policy.layout, "planned_target_units": plans[policy.name],
            "original_passes_all": all(row["original_target"]["passes"] for row in scenarios),
            "planned_passes_all": all(row["planned_target"]["passes"] for row in scenarios),
            "scenarios": scenarios,
        }
    output_rows = []
    for stage, data, runs in (("design", design, DESIGN_RUNS), ("holdout", holdout, HOLDOUT_RUNS)):
        for rep in range(runs):
            output_rows.append({
                "case": case.name, "stage": stage, "replication": rep + 1,
                **{f"{name}_rho_{rho}": int(counts[rep]) for (name, rho), counts in data.items()},
            })
    return report, output_rows


def run_study() -> tuple[dict, dict[str, list[dict]]]:
    cases = {record["name"]: Case(**record) for record in json.loads(
        (ROOT / "data" / "benchmarks.json").read_text(encoding="utf-8"))}
    mitchell = cases["Mitchell"]
    plans = {name: tuple(map(tuple, plan)) for name, plan in reference_plans(mitchell, 15).items()}
    concentrated = buffer_layout(verify(mitchell, 15, plans["MILP"]), "targeted_1")
    mitchell_policies = [
        Policy("rpw_7_uniform", plans["RPW"], (1,) * 7),
        Policy("milp_7_uniform", plans["MILP"], (1,) * 7),
        Policy("milp_7_concentrated", plans["MILP"], concentrated),
        Policy("milp_14_uniform", plans["MILP"], (2,) * 7),
    ]
    outage = json.loads((ROOT / "results" / "downtime_decision.json").read_text(encoding="utf-8"))
    jackson_plan = tuple(map(tuple, outage["station_tasks"]))
    jackson_policies = [
        Policy("existing_zero", jackson_plan, (0, 0, 0)),
        Policy("two_end_slots", jackson_plan, tuple(outage["layouts"]["selected"])),
        Policy("three_uniform", jackson_plan, (1, 1, 1)),
    ]
    report = {
        "shift_seconds": SHIFT_SECONDS, "task_cv_pct": TASK_CV, "shift_cv_pct": SHIFT_CV,
        "correlations": CORRELATIONS, "design_runs": DESIGN_RUNS, "holdout_runs": HOLDOUT_RUNS,
        "batch_size": BATCH_SIZE, "reliability": REQUIRED_RELIABILITY,
        "headroom_pct": HEADROOM_PCT, "target_increment": TARGET_INCREMENT, "cases": [],
    }
    replications = {}
    for case, cycle, target, policies, seeds in (
        (mitchell, 15, 1650, mitchell_policies, (2026112205, 2026112206)),
        (cases["Jackson"], 14, 1780, jackson_policies, (2026112207, 2026112208)),
    ):
        result, rows = evaluate(case, cycle, target, policies, seeds)
        report["cases"].append(result)
        replications[case.name.lower()] = rows
    return report, replications


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    report, replications = run_study()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for case, rows in replications.items():
        with (args.output_dir / f"persistent_workloads_{case}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    (args.output_dir / "persistent_workloads.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

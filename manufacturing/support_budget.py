"""Certify two consecutive shifts against a delivery promise and a labor budget."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

from good_output import FIRST_PASS, REPAIR_SECONDS, REWORK_PASS
from shift_dispatch import ADVERSE, DEMAND, INSPECT, ROOT, SHIFT_SECONDS, Case, reference_plans, shift_counts, trace_windows, trajectory
from staggered_support import simulate

MINUTES = (120, 125, 130, 135, 140, 150, 165, 180, 210)
POLICY = (6, 10, 0)
DESIGN = (480, 2026112261)
HOLDOUT = (1600, 2026112262)


def independent_repairs(work, routes, offers):
    """Infinite repair servers start each repair at the actual first-inspection finish."""
    clean = np.zeros_like(routes)
    trace = trace_windows(work, clean, offers, np.empty(0, dtype=np.int64), 0, 0, 12)
    ends = trace[:, INSPECT] + (routes >= REWORK_PASS) * REPAIR_SECONDS
    good = (routes == FIRST_PASS) | (routes == REWORK_PASS)
    return [int(np.sum(good & (ends <= (day + 1) * SHIFT_SECONDS) & (offers == day * SHIFT_SECONDS)))
            for day in ADVERSE]


def run_stage(case, plan, stage, minutes):
    rows = []
    for rep, seed in enumerate(np.random.SeedSequence(stage[1]).spawn(stage[0]), 1):
        work, routes, offers, _ = trajectory(case, plan, seed)
        upper = independent_repairs(work, routes, offers)
        for per_worker in minutes:
            duration = per_worker * 60
            fallback = np.repeat((np.array(ADVERSE) + 1) * SHIFT_SECONDS - duration, 2)
            initial = trace_windows(work, routes, offers, fallback, duration, 0, 12)
            trace, starts = simulate(work, routes, offers, POLICY, initial, duration)
            for i, day in enumerate(ADVERSE):
                metrics = shift_counts(trace, routes, offers, day)
                if metrics["on_time"] > upper[i] or metrics["helper_busy_min"] > 2 * per_worker:
                    raise AssertionError("line output or labor exceeds its physical bound")
                rows.append(dict(rep=rep, shift=day+1, minutes=per_worker,
                                 start_min=int((starts[2*i]-day*SHIFT_SECONDS)/60),
                                 perfect_repair=upper[i], **metrics))
    return rows


def levels(rows, minutes, field):
    selected = [r for r in rows if r["minutes"] == minutes]
    one, two = [[r for r in selected if r["shift"] == d+1] for d in ADVERSE]
    assert [r["rep"] for r in one] == [r["rep"] for r in two]
    return np.array([[r[field] for r in group] for group in (one, two)])


def choose_promise(joint):
    """Round down to tens using the 2% lower tail of the design histories."""
    value = np.sort(joint)[int(np.floor(.02 * len(joint)))]
    return int(min(DEMAND, value // 10 * 10))


def describe(output, target, alpha):
    passes = output >= target
    hits = int(np.all(passes, axis=0).sum())
    n = output.shape[1]
    lower = float(beta.ppf(alpha, hits, n-hits+1)) if hits else 0.
    return dict(target=target, each_shift=passes.sum(axis=1).tolist(), joint_hits=hits,
                lower=round(lower, 6), accepted=bool(lower >= .95))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"results")
    args = parser.parse_args()
    case = Case(**next(r for r in json.loads((ROOT/"data/benchmarks.json").read_text()) if r["name"] == "Mitchell"))
    plan = reference_plans(case, 15)["MILP"]
    design = run_stage(case, plan, DESIGN, MINUTES)
    promises = {m: choose_promise(levels(design, m, "on_time").min(axis=0)) for m in MINUTES}
    holdout = run_stage(case, plan, HOLDOUT, MINUTES)
    # Each budget has up to two predeclared claims. The last claim is the diagnostic ceiling.
    alpha = .05 / (2*len(MINUTES)+1)
    report = {}
    previous = levels(holdout, MINUTES[0], "on_time")
    ceiling = levels(holdout, MINUTES[0], "perfect_repair")
    for m in MINUTES:
        output = levels(holdout, m, "on_time")
        both = np.all(output >= DEMAND, axis=0)
        base = np.all(previous >= DEMAND, axis=0)
        report[m] = dict(full=describe(output, DEMAND, alpha),
                         design_promise=describe(output, promises[m], alpha),
                         paired_rescued=int(np.sum(both & ~base)), paired_lost=int(np.sum(~both & base)),
                         mean_busy=levels(holdout, m, "helper_busy_min").mean(axis=1).round(2).tolist(),
                         ceiling_also_misses=int(np.sum(~both & np.any(ceiling < DEMAND, axis=0))))
    summary = dict(design=DESIGN, holdout=HOLDOUT, policy=POLICY, confidence_family_alpha=.05,
                   alpha_per_claim=alpha, results=report, unlimited_repair=describe(ceiling, DEMAND, alpha))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/"support_budget.json").write_text(json.dumps(summary, indent=2)+"\n")
    with (args.output_dir/"support_budget.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=holdout[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(holdout)


if __name__ == "__main__":
    main()

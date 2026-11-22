"""Test whether additional marked waiting places can replace extra crew time."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from shift_dispatch import ADVERSE, DEMAND, DURATION, FALLBACK, HANDOFF, INSPECT, REPAIR, ROOT, SHIFT_SECONDS, Case, reference_plans, shift_counts, trace_windows, trajectory
from staggered_support import simulate
from support_budget import POLICY, describe, independent_repairs

SLOTS = (12, 16, 20, 24, 28, 32, 40, 48, 64, 96)
HOLDOUT = (2000, 2026112281)


def peak_waiting(trace, left, right):
    repairs = np.isfinite(trace[:, REPAIR])
    handoffs = trace[repairs, HANDOFF]
    starts = trace[repairs, REPAIR]
    # Equal-time service starts free a place before measuring occupancy.
    points = np.r_[left, handoffs[(handoffs > left) & (handoffs <= right)]]
    return int(max(np.searchsorted(handoffs, points, side="right") -
                   np.searchsorted(starts, points, side="right")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"results")
    parser.add_argument("--confirm", action="store_true", help="independently replay the locked 12 and 32 place comparison")
    args = parser.parse_args()
    options, stage = ((12, 32), (3000, 2026112282)) if args.confirm else (SLOTS, HOLDOUT)
    stem = "repair_space_confirmation" if args.confirm else "repair_space_budget"
    case = Case(**next(r for r in json.loads((ROOT/"data/benchmarks.json").read_text()) if r["name"] == "Mitchell"))
    plan = reference_plans(case, 15)["MILP"]
    rows = []
    for rep, seed in enumerate(np.random.SeedSequence(stage[1]).spawn(stage[0]), 1):
        work, routes, offers, _ = trajectory(case, plan, seed)
        upper = independent_repairs(work, routes, offers)
        fallback = np.repeat(np.array(ADVERSE)*SHIFT_SECONDS + FALLBACK, 2)
        for slots in options:
            initial = trace_windows(work, routes, offers, fallback, DURATION, 0, slots)
            trace, _ = simulate(work, routes, offers, POLICY, initial, repair_slots=slots)
            for i,day in enumerate(ADVERSE):
                left, right = day*SHIFT_SECONDS, (day+1)*SHIFT_SECONDS
                metrics = shift_counts(trace, routes, offers, day)
                peak = peak_waiting(trace, left, right)
                blocked = np.maximum(0., np.minimum(trace[:, HANDOFF], right) - np.maximum(trace[:, INSPECT], left))
                assert peak <= slots and metrics["helper_busy_min"] <= 240 and metrics["on_time"] <= upper[i]
                rows.append(dict(rep=rep, shift=day+1, slots=slots, peak_waiting=peak,
                                 blocked_min=round(float(blocked.sum()/60), 6), **metrics))
    summary = {}
    for slots in options:
        by_day = [[r for r in rows if r["slots"] == slots and r["shift"] == d+1] for d in ADVERSE]
        counts = np.array([[r["on_time"] for r in rs] for rs in by_day])
        both = np.all(counts >= DEMAND, axis=0)
        if slots == options[0]:
            base = both
        summary[slots] = {**describe(counts, DEMAND, .05/len(options)),
                          "rescued":int((both & ~base).sum()), "lost":int((~both & base).sum()),
                          "mean_blocked_min":[round(float(np.mean([r["blocked_min"] for r in rs])),3) for rs in by_day]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(holdout=stage, slots=options, crew="2x120", target=DEMAND, results=summary)
    (args.output_dir/f"{stem}.json").write_text(json.dumps(report,indent=2)+"\n")
    with (args.output_dir/f"{stem}.csv").open("w", newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0],lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

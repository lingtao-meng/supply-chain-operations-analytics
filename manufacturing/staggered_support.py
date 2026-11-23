"""Select and test equal-budget arrival times for two repair helpers."""

import argparse
import csv
import json
from itertools import product
from pathlib import Path

import numpy as np
from scipy.stats import beta, binomtest

from shift_dispatch import (ADVERSE, CHECK, DAYS, DEMAND, DURATION, FALLBACK, INSPECT, LOT, REPAIR,
                            RESPONSE, ROOT, SHIFT_SECONDS, Case, paired_interval, reference_plans,
                            shift_counts, trace_windows, trajectory)

DESIGN = (160, 2026112251)
HOLDOUT = (800, 2026112252)
BASELINE = (8, 0, 0)
HISTORY = 40


def trigger(waiting, seen, recent_repairs, threshold, minimum_repairs):
    return ((waiting >= threshold)
            & ((minimum_repairs == 0) | ((seen >= HISTORY) & (recent_repairs >= minimum_repairs))))


def crew_times(trace, offers, day, policy, duration=DURATION):
    threshold, minimum_repairs, lag_minutes = policy
    latest_start = SHIFT_SECONDS - duration
    day_start = day * SHIFT_SECONDS
    checks = day_start + np.arange(0, latest_start - RESPONSE + 1, CHECK)
    repairs = np.isfinite(trace[:, REPAIR])
    inspected = trace[repairs, INSPECT]
    waiting = np.searchsorted(inspected, checks, side="right") - np.searchsorted(trace[repairs, REPAIR], checks, side="right")
    counts = recent = 0
    if minimum_repairs:
        # Inspect only the newest batch; previous-shift carryover can create a false alarm.
        current = offers == day_start
        counts = np.searchsorted(trace[current, INSPECT], checks, side="right")
        cumulative = np.r_[0, np.cumsum(repairs[current])]
        recent = cumulative[counts] - cumulative[np.maximum(counts - HISTORY, 0)]
    eligible = trigger(waiting, counts, recent, threshold, minimum_repairs)
    crossings = np.flatnonzero(eligible)
    first = checks[crossings[0]] + RESPONSE if len(crossings) else day_start + latest_start
    return (first, min(first + lag_minutes * 60, day_start + latest_start))


def simulate(work, routes, offers, policy, fallback_trace, duration=DURATION, crew_size=2, repair_slots=12):
    if crew_size < 2 or (crew_size != 2 and policy[2] != 0):
        raise ValueError("larger crews require simultaneous starts")
    starts = np.repeat(np.asarray(ADVERSE) * SHIFT_SECONDS + SHIFT_SECONDS - duration, crew_size)
    trace = fallback_trace
    for i, day in enumerate(ADVERSE):
        pair = slice(crew_size*i, crew_size*(i+1))
        arrivals = crew_times(trace, offers, day, policy, duration)
        if crew_size != 2:
            arrivals = np.full(crew_size, arrivals[0])
        if np.any(starts[pair] != arrivals):
            starts[pair] = arrivals
            trace = trace_windows(work, routes, offers, starts, duration, 0, repair_slots)
    return trace, starts


def study(case, plan, stage, candidates, raw):
    scores = {p: [0, 0, 0] for p in candidates}
    rows = []
    for rep, seed in enumerate(np.random.SeedSequence(stage[1]).spawn(stage[0]), 1):
        work, routes, offers, _ = trajectory(case, plan, seed)
        fallback = np.repeat(np.asarray(ADVERSE)*SHIFT_SECONDS + FALLBACK, 2)
        fallback_trace = trace_windows(work, routes, offers, fallback, DURATION, 0, 12)
        for policy in candidates:
            trace, starts = simulate(work, routes, offers, policy, fallback_trace)
            for i, day in enumerate(ADVERSE):
                row = shift_counts(trace, routes, offers, day)
                scores[policy][i] += int(row["on_time"] >= DEMAND)
                scores[policy][2] -= max(0, DEMAND - row["on_time"])
                if raw:
                    first, second = (starts[2*i:2*i+2] - day*SHIFT_SECONDS)//60
                    rows.append(dict(rep=rep, shift=day+1, policy=label(policy), first_min=int(first),
                                     second_min=int(second), **row))
    return scores, rows


def label(policy):
    return "_".join(map(str, policy))


def summarize(baseline, proposed):
    assert [r["rep"] for r in baseline] == [r["rep"] for r in proposed]
    a, b = [np.array([r["on_time"] for r in rows]) for rows in (baseline, proposed)]
    first, second = a >= DEMAND, b >= DEMAND
    rescued, lost = int(sum(second & ~first)), int(sum(first & ~second))
    hits = int(sum(second))
    return dict(n=len(a), baseline_hits=int(sum(first)), hits=hits, rescued=rescued, lost=lost,
                paired_exact_p=round(float(binomtest(rescued, rescued+lost).pvalue), 8) if rescued+lost else 1.,
                lower_onesided_97_5=round(float(beta.ppf(.025, hits, len(a)-hits+1)), 5) if hits else 0.,
                paired_on_time=paired_interval(b-a),
                mean_shortfall=[round(float(np.maximum(DEMAND-v, 0).mean()), 3) for v in (a,b)],
                mean_backlog=[round(float(np.mean([r["backlog"] for r in g])),3) for g in (baseline, proposed)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"results")
    args = parser.parse_args()
    case = Case(**next(r for r in json.loads((ROOT/"data/benchmarks.json").read_text()) if r["name"] == "Mitchell"))
    plan = reference_plans(case, 15)["MILP"]
    candidates = list(product((2, 4, 6, 8, 10), (0, 6, 8, 10), (0, 10, 20, 30)))
    scores, _ = study(case, plan, DESIGN, candidates, False)
    # Protect the worse of the two consecutive adverse shifts first.
    def rank(p):
        h1, h2, shortage = scores[p]
        return min(h1,h2), h1+h2, shortage, -p[2]
    selected = max(candidates, key=rank)
    best_simultaneous = max((p for p in candidates if p[2]==0), key=rank)
    best_staggered = max((p for p in candidates if p[2]>0), key=rank)
    paired_start = (*selected[:2], 0)
    frozen = list(dict.fromkeys((BASELINE, selected, best_simultaneous, best_staggered, paired_start)))
    _, rows = study(case, plan, HOLDOUT, frozen, True)
    def records(p, day):
        return [r for r in rows if r["shift"]==day+1 and r["policy"]==label(p)]
    summary = {label(p): {day+1:summarize(records(BASELINE, day),records(p, day)) for day in ADVERSE}
               for p in frozen}
    report = dict(design=DESIGN, holdout=HOLDOUT, candidates={label(p):scores[p] for p in candidates},
                  selected=selected, best_simultaneous=best_simultaneous, best_staggered=best_staggered,
                  baseline=BASELINE, comparison=summary)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/"staggered_support.json").write_text(json.dumps(report,indent=2)+"\n")
    with (args.output_dir/"staggered_support.csv").open("w",newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

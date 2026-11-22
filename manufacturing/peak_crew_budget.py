"""Select an on-call crew on design histories, validate on fresh paired histories."""

import argparse
import csv
import json
from itertools import product
from pathlib import Path

import numpy as np

from shift_dispatch import ADVERSE, DEMAND, ROOT, SHIFT_SECONDS, Case, reference_plans, shift_counts, trace_windows, trajectory
from staggered_support import simulate
from support_budget import POLICY, describe, independent_repairs

BASE = (2, 120)
SAME_COST = (3, 80)
DESIGN = (320, 2026112271)
HOLDOUT = (1600, 2026112272)
OPTIONS = sorted({BASE, SAME_COST, (2, 210), *product((3, 4, 5, 6), (75, 90, 105, 120, 135, 150, 165, 180, 210))})


def run(case, plan, stage, options, details=False):
    counts = {option: [] for option in options}
    rows = []
    for rep, seed in enumerate(np.random.SeedSequence(stage[1]).spawn(stage[0]), 1):
        work, routes, offers, _ = trajectory(case, plan, seed)
        ceiling = independent_repairs(work, routes, offers)
        for people, minutes in options:
            duration = minutes * 60
            fallback = np.repeat((np.asarray(ADVERSE)+1)*SHIFT_SECONDS-duration, people)
            initial = trace_windows(work, routes, offers, fallback, duration, 0, 12)
            trace, starts = simulate(work, routes, offers, POLICY, initial, duration, people)
            output = []
            for i, day in enumerate(ADVERSE):
                data = shift_counts(trace, routes, offers, day)
                output.append(data["on_time"])
                assert data["on_time"] <= ceiling[i] and data["helper_busy_min"] <= people*minutes
                if details:
                    rows.append(dict(rep=rep, shift=day+1, people=people, minutes=minutes,
                                     start_min=int((starts[i*people]-day*SHIFT_SECONDS)/60),
                                     unlimited=ceiling[i], **data))
            counts[people, minutes].append(output)
    return {k: np.asarray(v).T for k,v in counts.items()}, rows


def name(option):
    return f"{option[0]}x{option[1]}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"results")
    args = parser.parse_args()
    case = Case(**next(r for r in json.loads((ROOT/"data/benchmarks.json").read_text()) if r["name"] == "Mitchell"))
    plan = reference_plans(case, 15)["MILP"]
    design, _ = run(case, plan, DESIGN, OPTIONS)
    hits = {k:int((v.min(axis=0) >= DEMAND).sum()) for k,v in design.items()}
    cheapest = lambda k: (k[0]*k[1], k[0], k[1])
    ordered = sorted(OPTIONS, key=cheapest)
    # Two prespecified design margins yield economical and conservative options.
    winners = [next((k for k in ordered if hits[k]/DESIGN[0] >= gate), max(ordered,key=hits.get))
               for gate in (.965, .98)]
    strongest = max(ordered, key=hits.get)
    frozen = list(dict.fromkeys([BASE, SAME_COST, (2, 210), *winners, strongest]))
    test, rows = run(case, plan, HOLDOUT, frozen, True)
    alpha = .05 / len(frozen)
    summary = {}
    base_hits = test[BASE].min(axis=0) >= DEMAND
    for k, data in test.items():
        passed = data.min(axis=0) >= DEMAND
        summary[name(k)] = {**describe(data, DEMAND, alpha),
                            "rescued":int((passed & ~base_hits).sum()), "lost":int((base_hits & ~passed).sum())}
    result = dict(design=DESIGN, holdout=HOLDOUT, design_hits={name(k):v for k,v in hits.items()},
                  nominees=[name(k) for k in winners], strongest=name(strongest), alpha=alpha, holdout_results=summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/"peak_crew_budget.json").write_text(json.dumps(result,indent=2)+"\n")
    with (args.output_dir/"peak_crew_budget.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0],lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__=="__main__":
    main()

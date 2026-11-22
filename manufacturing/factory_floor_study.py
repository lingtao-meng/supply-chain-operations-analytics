"""Compare layout, ergonomics, material routes and repaired-unit flow."""

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from factory_dispatch_study import PLANT, SHIPPING, evaluate, summarize
from factory_floor import ACTIONS, CLOSE_LAYOUT, Floor
from factory_inputs import ADVERSE, SHIFTS, SHIFT_SECONDS, load_case, sample
from factory_receiving import ReceivingPlan, receive_kits
from factory_simulation import END, ENTRY, simulate
from good_output import REWORK_PASS, REWORK_SCRAP
from line_balance import ROOT

DESIGN = (120, 2026112331)
HOLDOUT = (600, 2026112332)
POSTURE_LIMIT = 90
ACTIONS_LIMIT = 42


@dataclass(frozen=True)
class Policy:
    floor: Floor = Floor()
    second_repair: bool = True
    faults: bool = True


BASE = Policy()
CLOSE = replace(BASE, floor=replace(BASE.floor, layout=CLOSE_LAYOUT))
AID = replace(CLOSE, floor=replace(CLOSE.floor, posture_aid=True))
OPTIONS = {
    "baseline": BASE,
    "zero_buffer": replace(BASE, floor=replace(BASE.floor, buffers=(0,)*7)),
    "two_per_gap": replace(BASE, floor=replace(BASE.floor, buffers=(2,)*7)),
    "compact_layout": CLOSE,
    "symmetric_setup": replace(BASE, floor=replace(BASE.floor, directional_setup=False)),
    "no_fault": replace(BASE, faults=False),
    "no_second_repair": replace(BASE, second_repair=False),
    "slow_kanban": replace(BASE, floor=replace(BASE.floor, route_minutes=25)),
    "no_route_or_kanban_wait": replace(BASE, floor=replace(BASE.floor, route_minutes=0)),
    "compact_with_aid": AID,
    "compact_aid_faster_route": replace(AID, floor=replace(AID.floor, route_minutes=10)),
    "compact_aid_faster_single_card": replace(AID, floor=replace(AID.floor, route_minutes=10, cards=1)),
    "compact_aid_two_per_gap": replace(AID, floor=replace(AID.floor, buffers=(2,)*7)),
}
PRACTICAL = tuple(k for k in OPTIONS if k.startswith("compact") and k != "compact_layout")


def calendar(starts, lengths, faults):
    intervals = [(d*SHIFT_SECONDS + 6000, d*SHIFT_SECONDS + 7080) for d in range(SHIFTS)]
    if faults:
        for day, start, length in zip(ADVERSE, starts, lengths, strict=True):
            left = day*SHIFT_SECONDS + start*60
            intervals.append((left, left+length*60))
    # Planned point checks and faults can overlap.
    merged = []
    for left, right in sorted(intervals):
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(right, merged[-1][1]))
        else:
            merged.append((left, right))
    return np.asarray(merged)


def second_attempt(ready, routes, draws):
    """A physically separate bench serializes repair and recheck of eligible failures."""
    new_ready, new_routes = ready.copy(), routes.copy()
    candidates = np.flatnonzero((routes == REWORK_SCRAP) & (draws[:, 0] < .60))
    order = candidates[np.argsort(ready[candidates], kind="stable")]
    free = 0.
    for unit in order:
        free = max(free, ready[unit]) + 210.
        new_ready[unit] = free
        if draws[unit, 1] < .45:
            new_routes[unit] = REWORK_PASS
    assert np.all(new_ready[order] >= ready[order]+210)
    return new_ready, new_routes, order


def replay(scenario, gates, starts, lengths, retries, policy):
    planned = replace(scenario, kits=gates)
    windows = calendar(starts, lengths, policy.faults)
    model = simulate(planned, replace(PLANT, kit_advance=0), maintenance=windows, floor=policy.floor)
    assert np.all(model.trace[:, ENTRY] >= gates)
    check_start, check_end = model.maintenance_operations.T
    for a,b in windows:
        assert not np.any((check_start < b) & (check_end > a))
    attempted = np.empty(0, dtype=int)
    if policy.second_repair:
        ready, routes, attempted = second_attempt(model.trace[:, END], scenario.routes, retries)
        trace = model.trace.copy()
        trace[:, END] = ready
        planned = replace(planned, routes=routes)
        model = replace(model, trace=trace)
    rows = evaluate(planned, model, (), SHIPPING["faster_loading"])
    for day, row in enumerate(rows):
        data = model.floor_stats[day]
        repetitions = data[:, :2] * np.array(ACTIONS).T
        posture = data[4, 2] / 60
        peak_actions = repetitions.sum(axis=1).max() / 420
        end = model.trace[:, END]
        completed = (end > day*SHIFT_SECONDS) & (end <= (day+1)*SHIFT_SECONDS)
        row.update(
            peak_actions_per_min=float(peak_actions), station_five_posture_min=float(posture),
            ergonomics_pass=int(peak_actions <= ACTIONS_LIMIT and posture <= POSTURE_LIMIT),
            logistics_bins=int(model.delivery_stats[day,0]),
            wait_kit_min=float(model.delivery_stats[day,1]/60),
            transfer_m=float(data[:,3].sum()*1.25),
            second_attempts=int(completed[attempted].sum()),
            second_passes=int((completed[attempted] & (planned.routes[attempted] == REWORK_PASS)).sum()),
        )
    return rows


def collect(case, plan, stage, keys):
    result = {k: [] for k in keys}
    for seed in np.random.SeedSequence(stage[1]).spawn(stage[0]):
        regular, extra = seed.spawn(2)
        scenario = sample(case, plan, regular)
        rng = np.random.default_rng(extra)
        starts = rng.integers(145, 345, size=2)
        lengths = rng.integers(12, 31, size=2)
        retries = rng.random((len(scenario.kits), 2))
        gates, _ = receive_kits(scenario, ReceivingPlan(), advance_minutes=30)
        for key in keys:
            result[key].append(replay(scenario, gates, starts, lengths, retries, OPTIONS[key]))
    return result


def choose(histories):
    def score(k):
        shifts = [r for history in histories[k] for r in (history[d] for d in ADVERSE)]
        return (min(r["ergonomics_pass"] for r in shifts),
                sum(r["ergonomics_pass"] for r in shifts), -sum(r["shipped_shortfall"] for r in shifts))
    return max(PRACTICAL, key=score)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"results")
    args = parser.parse_args()
    case, plan = load_case()
    design = collect(case, plan, DESIGN, PRACTICAL)
    winner = choose(design)
    holdout = collect(case, plan, HOLDOUT, OPTIONS)
    alpha = .05 / len(holdout)
    result = dict(case=case.name, design=DESIGN, holdout=HOLDOUT, winner=winner,
                  limits=dict(actions_per_min=ACTIONS_LIMIT, posture_minutes=POSTURE_LIMIT),
                  configurations={k:asdict(p) for k,p in OPTIONS.items()},
                  results={k:summarize(h,holdout["baseline"],alpha) for k,h in holdout.items()})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/"factory_floor_study.json").write_text(json.dumps(result,indent=2)+"\n")
    with (args.output_dir/"factory_floor_study.csv").open("w",newline="") as f:
        fields = ["policy","trajectory",*holdout["baseline"][0][0]]
        writer = csv.DictWriter(f,fieldnames=fields,lineterminator="\n")
        writer.writeheader()
        for key, cohorts in holdout.items():
            for i, cohort in enumerate(cohorts,1):
                for row in cohort:
                    writer.writerow({"policy":key,"trajectory":i,**{
                        k:round(v,6) if isinstance(v,float) else v for k,v in row.items()
                    }})


if __name__ == "__main__":
    main()

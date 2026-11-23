"""Compare prebooked and queue-triggered repair crews on a continuous finite line."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from numba import njit
from scipy.stats import t

from downtime_recovery import SHIFT_CV, TASK_CV
from good_output import (BASE_DEFECT_PCT, FAST_FACTOR, FIRST_PASS, INSPECTION_SECONDS,
                         REPAIR_SECONDS, REWORK_PASS, STRESS_DEFECT_PCT, quality_routes)
from line_balance import ROOT, Case, reference_plans
from optimize_buffers import scenarios, workloads_for
from rework_support import concentrate_failures
from shift_capacity import DEMAND, SHIFT_SECONDS

LOT = 1850
DAYS = 4
ADVERSE = (1, 2)  # zero-based
DURATION = 120 * 60
FALLBACK = SHIFT_SECONDS - DURATION
CHECK = 5 * 60
RESPONSE = 10 * 60
DESIGN, HOLDOUT = (160, 2026112241), (400, 2026112243)
FIXED = tuple(range(0, 301, 5))
THRESHOLDS = (2, 4, 6, 8, 10, 12)
# One timestamp per offered unit and event. NaN means the route skips repair.
ENTRY, TRANSFER, INSPECT, HANDOFF, REPAIR, END, HELPER = range(7)


@njit(cache=True)
def trace_windows(work, routes, offers, starts, duration, inspection_slots, repair_slots):
    """One temporary worker per window. Distinct windows may overlap."""
    jobs, stations = work.shape
    trace = np.full((jobs, 7), np.nan)
    exits = np.zeros(stations + 1)
    ring = np.zeros((max(1, inspection_slots) + 1, stations + 1))
    repair_history = np.zeros(repair_slots + 1)
    repair_count = 0
    core_free = 0.
    helpers = starts.astype(np.float64)
    for job in range(jobs):
        arrival = offers[job]
        for station in range(stations):
            begin = max(arrival, exits[station])
            if station == 0:
                trace[job, ENTRY] = begin
            slots = inspection_slots if station == stations - 1 else 1
            leave = begin + work[job, station]
            if slots == 0:
                leave = max(leave, exits[station + 1])
            elif job >= slots:
                leave = max(leave, ring[(job - slots) % len(ring), station + 1])
            ring[job % len(ring), station] = begin
            exits[station] = arrival = leave
        trace[job, TRANSFER] = arrival
        inspect_begin = max(arrival, exits[-1])
        ring[job % len(ring), -1] = inspect_begin
        finish = inspect_begin + INSPECTION_SECONDS
        handoff = end = finish
        trace[job, INSPECT] = finish
        if routes[job] >= REWORK_PASS:
            core_start = max(finish, core_free)
            repair_start = core_start
            helper_used = False
            chosen = -1
            for worker in range(len(helpers)):
                available = max(finish, helpers[worker])
                if available < repair_start and available + REPAIR_SECONDS <= starts[worker] + duration:
                    repair_start = available
                    chosen = worker
            if chosen >= 0:
                helpers[chosen] = repair_start + REPAIR_SECONDS
                helper_used = True
            if not helper_used:
                core_free = core_start + REPAIR_SECONDS
            if repair_slots == 0:
                handoff = repair_start
            elif repair_count >= repair_slots:
                handoff = max(finish, repair_history[(repair_count - repair_slots) % len(repair_history)])
            repair_history[repair_count % len(repair_history)] = repair_start
            repair_count += 1
            end = repair_start + REPAIR_SECONDS
            trace[job, REPAIR] = repair_start
            trace[job, HELPER] = int(helper_used)
        exits[-1] = handoff
        trace[job, HANDOFF] = handoff
        trace[job, END] = end
    return trace


def trace_line(work, routes, offers, starts, duration, inspection_slots, repair_slots):
    """Original policy reserves two simultaneous workers per window."""
    return trace_windows(work, routes, offers, np.repeat(starts, 2), duration, inspection_slots, repair_slots)


def observed_start(trace, day, threshold):
    # Includes one inspected unit that may be blocked outside the waiting area.
    mask = np.isfinite(trace[:, REPAIR])
    arrivals, service = trace[mask, INSPECT], trace[mask, REPAIR]
    checks = day * SHIFT_SECONDS + np.arange(0, FALLBACK - RESPONSE + 1, CHECK)
    waiting = np.searchsorted(arrivals, checks, side="right") - np.searchsorted(service, checks, side="right")
    crossings = np.flatnonzero(waiting >= threshold)
    if len(crossings):
        return checks[crossings[0]] + RESPONSE, True
    return day * SHIFT_SECONDS + FALLBACK, False


def run_policy(work, routes, offers, kind, value):
    starts = np.array(ADVERSE) * SHIFT_SECONDS + (value * 60 if kind == "fixed" else FALLBACK)
    trace = trace_line(work, routes, offers, starts, DURATION, 0, 12)
    signals = [False] * len(ADVERSE)
    if kind == "queue":
        for i, day in enumerate(ADVERSE):
            start, signals[i] = observed_start(trace, day, value)
            if start != starts[i]:
                starts[i] = start
                trace = trace_line(work, routes, offers, starts, DURATION, 0, 12)
    return trace, starts, signals


def trajectory(case, plan, seed):
    rng = np.random.default_rng(seed)
    lots, dispositions, pulses = [], [], []
    for day in range(DAYS):
        _, task = next(scenarios(case, ((TASK_CV, SHIFT_CV),), LOT, 1, int(rng.integers(2**32))))
        lots.append(workloads_for(task, plan)[:, :, 0] * FAST_FACTOR)
        risk = day in ADVERSE
        routes = quality_routes(rng.random((LOT, 3)), STRESS_DEFECT_PCT if risk else BASE_DEFECT_PCT)
        onset = int(rng.integers(20, 301)) if risk else -1
        if risk:
            routes = concentrate_failures(routes, np.arange(LOT) * SHIFT_SECONDS / LOT, onset,
                                          rng.permutation(LOT), rng.permutation(routes[routes != FIRST_PASS]))
        pulses.append(onset)
        dispositions.append(routes)
    return np.vstack(lots), np.concatenate(dispositions), np.repeat(np.arange(DAYS) * SHIFT_SECONDS, LOT), pulses


def shift_counts(trace, routes, offers, day):
    left, right = day * SHIFT_SECONDS, (day + 1) * SHIFT_SECONDS
    good = (routes == FIRST_PASS) | (routes == REWORK_PASS)
    completed = trace[:, END] <= right
    offered = offers <= left
    # The next lot is offered at the boundary, after closing the period.
    entered = offered & (trace[:, ENTRY] < right)
    remains = offered & ~completed
    unstarted = remains & ~entered
    assembly = remains & entered & (trace[:, TRANSFER] > right)
    inspect = remains & (trace[:, TRANSFER] <= right) & (trace[:, INSPECT] > right)
    blocked = remains & (trace[:, INSPECT] <= right) & (trace[:, HANDOFF] > right)
    repair = remains & (trace[:, HANDOFF] <= right)
    backlog = int(remains.sum())
    assert backlog == int(unstarted.sum() + assembly.sum() + inspect.sum() + blocked.sum() + repair.sum())
    physically = completed & (trace[:, END] > left)
    paid = (trace[:, HELPER] == 1) & (trace[:, REPAIR] >= left) & (trace[:, END] <= right)
    current = offers == left
    return dict(on_time=int((current & completed & good).sum()), physical_good=int((physically & good).sum()),
                late_good=int((physically & ~current & good).sum()), scrap=int((physically & ~good).sum()),
                backlog=backlog, unstarted=int(unstarted.sum()), assembly=int(assembly.sum()),
                inspection=int(inspect.sum()), blocked=int(blocked.sum()), repair=int(repair.sum()),
                helper_busy_min=int(paid.sum() * REPAIR_SECONDS / 60))


def evaluate(case, plan, stage, policies, design_only=False):
    runs, seed = stage
    rows = []
    for rep, child in enumerate(np.random.SeedSequence(seed).spawn(runs), 1):
        work, routes, offers, pulses = trajectory(case, plan, child)
        for kind, value in policies:
            trace, starts, signals = run_policy(work, routes, offers, kind, value)
            for day in (ADVERSE if design_only else range(DAYS)):
                i = ADVERSE.index(day) if day in ADVERSE else None
                rows.append(dict(rep=rep, kind=kind, value=value, shift=day+1, onset=pulses[day],
                                 support_start_min=int((starts[i]-day*SHIFT_SECONDS)/60) if i is not None else -1,
                                 triggered=int(signals[i]) if i is not None else 0,
                                 reserved_min=240 if i is not None else 0, **shift_counts(trace, routes, offers, day)))
    return rows


def outcome(rows):
    return (sum(r["on_time"] >= DEMAND for r in rows),
            -sum(max(DEMAND-r["on_time"], 0) for r in rows), -sum(r["backlog"] for r in rows))


def paired_interval(differences):
    values = np.asarray(differences)
    mean = values.mean()
    radius = t.ppf(.975, len(values)-1) * values.std(ddof=1) / np.sqrt(len(values))
    return [round(float(x), 3) for x in (mean, mean-radius, mean+radius)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    records = json.loads((ROOT / "data/benchmarks.json").read_text())
    case = Case(**next(row for row in records if row["name"] == "Mitchell"))
    plan = reference_plans(case, 15)["MILP"]
    policies = [("fixed", v) for v in FIXED] + [("queue", v) for v in THRESHOLDS]
    design = evaluate(case, plan, DESIGN, policies, design_only=True)
    scores = {p: outcome([r for r in design if (r["kind"], r["value"]) == p and r["shift"]-1 in ADVERSE])
              for p in policies}
    locked = [max((p for p in policies if p[0] == kind), key=scores.get) for kind in ("fixed", "queue")]
    shortfall_control = max((p for p in policies if p[0] == "fixed"),
                            key=lambda p: (scores[p][1], scores[p][0], scores[p][2]))
    evaluated = list(dict.fromkeys([*locked, shortfall_control]))
    rows = evaluate(case, plan, HOLDOUT, evaluated)
    results = {}
    for day in range(DAYS):
        a, b, c = [[r for r in rows if (r["kind"], r["value"]) == policy and r["shift"] == day+1]
                   for policy in [*locked, shortfall_control]]
        assert [r["rep"] for r in a] == [r["rep"] for r in b]
        def describe(group):
            return dict(hits=outcome(group)[0], **{f"mean_{k}": round(float(np.mean([r[k] for r in group])), 3)
                        for k in ("on_time", "backlog", "physical_good", "late_good", "helper_busy_min")})
        results[day+1] = dict(fixed=describe(a), queue=describe(b), fixed_shortfall_control=describe(c),
                              paired_vs_shortfall_control=paired_interval([y["on_time"]-x["on_time"] for x,y in zip(c,b)]),
                              paired_on_time_delta=paired_interval([y["on_time"]-x["on_time"] for x,y in zip(a,b)]))
    report = dict(design=DESIGN, holdout=HOLDOUT, scores={f"{k}_{v}":s for (k,v),s in scores.items()},
                  selected=locked, fixed_shortfall_control=shortfall_control, target=DEMAND, lot=LOT, results=results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/"shift_dispatch.json").write_text(json.dumps(report, indent=2)+"\n")
    with (args.output_dir/"shift_dispatch.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

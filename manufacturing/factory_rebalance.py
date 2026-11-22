"""Compare adjacent task transfers, release priorities and a staffed duplicate bench on one floor."""

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from functools import partial
from itertools import product
from pathlib import Path

import numpy as np
from scipy.stats import beta, t as student_t

from factory_capacity_plan import Design, ORIGINAL_TARGET, design_promise, violations
from factory_dispatch_study import PLANT, SHIPPING
from factory_floor import ACTIONS
from factory_floor_study import ACTIONS_LIMIT, OPTIONS, POSTURE_LIMIT, calendar, second_attempt
from factory_inputs import ADVERSE, SHIFTS, SHIFT_SECONDS, load_case, sample
from factory_receiving import ReceivingPlan, receive_kits
from factory_shipping import dispatch
from factory_simulation import ASSEMBLED, END, ENTRY, simulate
from factory_study import SKU_TARGETS
from good_output import FIRST_PASS, REWORK_PASS
from line_balance import ROOT

DESIGN = (120, 2026112341)
PILOT_SEED = 2026112342
HOLDOUT = (600, 2026112343)
FLOOR = OPTIONS["compact_aid_two_per_gap"].floor
RULES = ("ready", "current_ready", "reserve_old")
BASE_DESIGN = Design()
PARALLEL_DESIGN = Design(location="added_bay")
TIME_NAMES = ("processing", "setup", "blocked", "transfer", "calendar_wait", "other_idle")


@dataclass(frozen=True)
class Choice:
    task: int = 0
    source: int = -1
    dest: int = -1
    rule: str = "ready"
    parallel: bool = False

    @property
    def name(self):
        return f"t{self.task}_{self.rule}_{'parallel' if self.parallel else 'serial'}"


def candidate_moves(case, plan):
    """One adjacent transfer; keep the dedicated fixture work and specialist zone intact."""
    yield Choice()
    for src, group in enumerate(plan):
        for dst in (src-1, src+1):
            if not 0 <= dst < len(plan) or 4 in (src, dst) or len(group) == 1:
                continue
            for task in group:
                trial = [list(tasks) for tasks in plan]
                trial[src].remove(task)
                trial[dst].append(task)
                station = {t: i for i, tasks in enumerate(trial) for t in tasks}
                if any(station[a] > station[b] for a, b in case.precedence):
                    continue
                if max(sum(case.task_times[t-1] for t in tasks) for tasks in trial) > 18:
                    continue
                if not violations(BASE_DESIGN, trial) and not violations(PARALLEL_DESIGN, trial):
                    yield Choice(task, src, dst)


def workload(scenario, tasks, choice):
    if not choice.task:
        return scenario.work
    work = scenario.work.copy()
    moved = tasks[:, choice.task-1]
    work[:, choice.source] -= moved
    work[:, choice.dest] += moved
    assert work.min() > 0 and np.allclose(work.sum(axis=1), scenario.work.sum(axis=1))
    return work


def action_rates(case, plan, choice):
    actions = np.array(ACTIONS, dtype=float)
    if choice.task:
        source_load = sum(case.task_times[t-1] for t in plan[choice.source])
        share = actions[:, choice.source] * case.task_times[choice.task-1] / source_load
        actions[:, choice.source] -= share
        actions[:, choice.dest] += share
    return actions


def cohort_rows(scenario, ready, good, shipped):
    rows = []
    for day in range(SHIFTS):
        left, right = day*SHIFT_SECONDS, (day+1)*SHIFT_SECONDS
        current = scenario.orders == left
        on_time = current & (shipped <= right)
        a, b = np.bincount(scenario.sku[on_time], minlength=2)
        # A scrapped piece leaves the outstanding physical-job queue upon disposition.
        pending = (scenario.orders <= left) & (shipped > right) & (good | (ready > right))
        due = scenario.orders + SHIFT_SECONDS
        departed = (shipped > left) & (shipped <= right)
        # Follow terminal backlog as right-censored and retain realized late deliveries.
        observed_delay = np.maximum(np.minimum(shipped, right) - due, 0)
        rows.append(dict(shift=day+1, a=int(a), b=int(b),
                         short_a=max(0, int(SKU_TARGETS[0]-a)), short_b=max(0, int(SKU_TARGETS[1]-b)),
                         qualified=int((current & good & (ready <= right)).sum()),
                         old_sent=int((departed & ~current).sum()),
                         old_open=int((pending & (due < right)).sum()), closing_open=int(pending.sum()),
                         scrap=int(((scenario.orders <= left) & ~good & (ready <= right)).sum()),
                         late_minutes=float(observed_delay[pending | departed].max(initial=0)/60)))
        assert (int((shipped <= right).sum()) + rows[-1]["closing_open"] + rows[-1]["scrap"]
                == int((scenario.orders <= left).sum()))
    return rows


def trial(case, plan, choices, detail, seed):
    base_seed, extra_seed, roster_seed = seed.spawn(3)
    blocks = []
    scenario = sample(case, plan, base_seed, task_blocks=blocks)
    tasks = np.concatenate(blocks)
    extra = np.random.default_rng(extra_seed)
    windows = calendar(extra.integers(145, 345, 2), extra.integers(12, 31, 2), True)
    retries = extra.random((len(tasks), 2))
    staffed = np.random.default_rng(roster_seed).random(SHIFTS) < .82
    gates, _ = receive_kits(scenario, ReceivingPlan(), advance_minutes=30)
    results = {}
    for choice in choices:
        planned = replace(scenario, work=workload(scenario, tasks, choice), kits=gates)
        model = simulate(planned, replace(PLANT, kit_dispatch=choice.rule, kit_advance=0),
                         maintenance=windows, parallel_available=staffed if choice.parallel else None,
                         floor=FLOOR, diagnose=detail)
        assert np.all(model.trace[:, ENTRY] >= gates)
        starts, ends = model.maintenance_operations.T
        for alternate in (False, True):
            mask = model.parallel_used == alternate
            order = np.argsort(starts[mask])
            assert np.all(starts[mask][order][1:] >= ends[mask][order][:-1])
            for left, right in windows + 2400*alternate:
                assert not np.any(mask & (starts < right) & (ends > left))
        alt_start = starts[model.parallel_used]
        alt_day = (alt_start // SHIFT_SECONDS).astype(int)
        assert np.all(staffed[alt_day]) and np.all(ends[model.parallel_used] <= (alt_day+1)*SHIFT_SECONDS)
        ready, routes, _ = second_attempt(model.trace[:, END], scenario.routes, retries)
        good = (routes == FIRST_PASS) | (routes == REWORK_PASS)
        shipped = dispatch(ready, scenario.orders, good, SHIPPING["faster_loading"])
        rows = cohort_rows(scenario, ready, good, shipped)
        actions = action_rates(case, plan, choice)
        for day, row in enumerate(rows):
            cutoff = (day+1)*SHIFT_SECONDS
            current = planned.orders == day*SHIFT_SECONDS
            in_process = current & (ready > cutoff)
            entry = model.trace[:, ENTRY]
            assembled = model.trace[:, ASSEMBLED]
            row["not_started"] = int((in_process & (entry > cutoff)).sum())
            row["on_line"] = int((in_process & (entry <= cutoff) & (assembled > cutoff)).sum())
            row["in_quality"] = int((in_process & (assembled <= cutoff)).sum())
            row["scrapped_current"] = int((current & ~good & (ready <= cutoff)).sum())
            row["awaiting_truck"] = row["qualified"]-row["a"]-row["b"]
            assert (row["qualified"]+row["not_started"]+row["on_line"]+row["in_quality"]+
                    row["scrapped_current"] == int(current.sum()))
            counts = model.floor_stats[day, :, :2].copy()
            mask = model.parallel_used & (starts >= day*SHIFT_SECONDS) & (starts < (day+1)*SHIFT_SECONDS)
            alt = np.bincount(scenario.sku[mask], minlength=2)
            counts[3] -= alt
            peak = max((counts * actions.T).sum(axis=1).max(), alt @ actions[:, 3])/420
            row["ergo_pass"] = int(peak <= ACTIONS_LIMIT and model.floor_stats[day, 4, 2]/60 <= POSTURE_LIMIT)
            row["parallel_jobs"] = int(alt.sum())
        minutes = None
        if detail:
            ledger = model.station_seconds[:, :8+(1 if choice.parallel else 0)] / 60
            idle = 420-ledger.sum(axis=2)
            assert idle.min() >= -1e-7
            minutes = np.dstack((ledger, idle))
        results[choice.name] = rows, minutes
    return results


def collect(case, plan, choices, stage, detail, workers):
    seeds = np.random.SeedSequence(stage[1]).spawn(stage[0])
    run = partial(trial, case, plan, choices, detail)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run, seeds, chunksize=4))


def totals(records, choice):
    rows = [rec[choice.name][0] for rec in records]
    adverse = np.array([[[r["a"], r["b"]] for r in (hist[d] for d in ADVERSE)] for hist in rows])
    mean = lambda values: round(float(np.mean(values)), 4)
    return adverse, dict(
        mean_shortfall=mean([sum(r["short_a"]+r["short_b"] for r in hist) for hist in rows]),
        max_delay_min_mean=mean([max(r["late_minutes"] for r in hist) for hist in rows]),
        final_open_mean=mean([hist[-1]["closing_open"] for hist in rows]),
        adverse_ergo_pass=mean([all(hist[d]["ergo_pass"] for d in ADVERSE) for hist in rows]),
        mean_shift_3={k:mean([h[2][k] for h in rows]) for k in rows[0][2] if k != "shift"},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"results")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    case, plan = load_case()
    options = [replace(move, rule=rule, parallel=parallel)
               for move, rule, parallel in product(candidate_moves(case, plan), RULES, (False, True))]
    assert len(set(c.name for c in options)) == len(options)
    design = collect(case, plan, options, DESIGN, False, args.workers)
    scores, promises = {}, {}
    for c in options:
        adverse, scores[c.name] = totals(design, c)
        promises[c.name] = design_promise(adverse)
    def rank(c):
        s = scores[c.name]
        return (-s["adverse_ergo_pass"], s["mean_shortfall"], s["max_delay_min_mean"], c.name)
    serial = min((c for c in options if not c.parallel), key=rank)
    parallel = min((c for c in options if c.parallel), key=rank)
    move_only = min((c for c in options if c.task and not c.parallel and c.rule=="ready"), key=rank)
    def service_rank(c):
        return (-scores[c.name]["adverse_ergo_pass"], -promises[c.name], *rank(c)[1:])
    reliable = [min((c for c in options if c.parallel == duplicate), key=service_rank)
                for duplicate in (False, True)]
    locked = list(dict.fromkeys([*(Choice(rule=r) for r in RULES), move_only, serial,
                                 replace(serial, parallel=True), parallel, *reliable]))
    print(f"Locked {len(locked)} of {len(options)} alternatives", flush=True)
    histories = collect(case, plan, locked, HOLDOUT, True, args.workers)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    alpha = .05/(2*len(locked))
    with (out/"factory_rebalance.csv").open("w",newline="") as f:
        writer = None
        for c in locked:
            adverse, result = totals(histories, c)
            result["station_minutes_by_shift"] = np.mean(
                [h[c.name][1] for h in histories], axis=0).round(4).tolist()
            # One paired difference per independent four-shift trajectory.
            def shortage(history):
                return sum(row["short_a"] + row["short_b"] for row in history)
            gains = np.array([shortage(h[Choice().name][0]) - shortage(h[c.name][0]) for h in histories])
            half_width = student_t.ppf(.975, len(gains)-1)*gains.std(ddof=1)/np.sqrt(len(gains))
            result["four_shift_shortfall_reduction_vs_base"] = [
                round(float(x), 4) for x in (gains.mean(), gains.mean()-half_width, gains.mean()+half_width)]
            result["reliability"] = {}
            for target in (promises[c.name], ORIGINAL_TARGET):
                hits = int(np.all((adverse[:,:,0] >= target*7//10)
                                  & (adverse[:,:,1] >= target*3//10), axis=1).sum())
                lower = float(beta.ppf(alpha, hits, len(histories)-hits+1)) if hits else 0.
                result["reliability"][target] = dict(hits=hits, lower=round(lower,6), passed=target>0 and lower>=.95)
            summary[c.name] = result
            for i,h in enumerate(histories,1):
                for row in h[c.name][0]:
                    if writer is None:
                        writer = csv.DictWriter(f,fieldnames=["policy","trajectory",*row],lineterminator="\n")
                        writer.writeheader()
                    writer.writerow(dict(policy=c.name,trajectory=i,**row))
    result = dict(design=DESIGN, exploratory_holdout_seed=PILOT_SEED, holdout=HOLDOUT, floor="compact_aid_two_per_gap", original_plan=plan,
                  candidates={c.name:dict(source=c.source+1,dest=c.dest+1,task=c.task,rule=c.rule,
                                          equipment_points=PARALLEL_DESIGN.cost if c.parallel else 0,
                                          promise=promises[c.name],design_score=scores[c.name]) for c in options},
                  selected_serial=serial.name, selected_parallel=parallel.name,
                  service_priority=[c.name for c in reliable], move_only=move_only.name,
                  station_minutes_columns=TIME_NAMES, simultaneous_alpha=alpha, results=summary)
    (out/"factory_rebalance.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__ == "__main__":
    main()

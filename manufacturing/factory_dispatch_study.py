"""Compare receiving, equipment availability and shipping limits on paired line replays."""

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path

import numpy as np
from scipy.stats import beta

from factory_inputs import ADVERSE, SHIFTS, SHIFT_SECONDS, load_case, sample
from factory_receiving import ReceivingPlan, receive_kits
from factory_shipping import Outbound, dispatch
from factory_simulation import END, ENTRY, Plant, simulate
from factory_study import RELIABILITY, SKU_TARGETS
from good_output import FIRST_PASS, REWORK_PASS, paired_gain
from line_balance import ROOT

DESIGN = (120, 2026112311)
HOLDOUT = (600, 2026112312)
CONFIRMATION = (700, 2026112313)
MAINTENANCE_START = 230
MAINTENANCE_DURATION = 18
START_MINUTES = (100, MAINTENANCE_START, 300)
BASE_TRUCKS = Outbound()
SAME_CAPACITY = replace(BASE_TRUCKS, capacities=(500, 600, 850))
SHIPPING = {
    "normal": BASE_TRUCKS,
    "equal_capacity": SAME_CAPACITY,
    "extra_capacity": replace(BASE_TRUCKS, capacities=(650, 650, 900)),
    "faster_loading": replace(BASE_TRUCKS, cutoff_minutes=5),
    "equal_capacity_faster_loading": replace(SAME_CAPACITY, cutoff_minutes=5),
}
PLANT = Plant(kit_dispatch="ready", cover=True, kit_advance=30)


@dataclass(frozen=True)
class Policy:
    lanes: int = 1
    maintenance_start: int | None = MAINTENANCE_START
    outbound: str | None = "normal"


def policy_label(policy):
    return f"lanes{policy.lanes}_minute{policy.maintenance_start}_{policy.outbound}"


def windows_at(minute):
    if minute is None:
        return None
    starts = np.arange(SHIFTS) * SHIFT_SECONDS + minute * 60
    return np.column_stack((starts, starts + MAINTENANCE_DURATION * 60))


def production(scenario, lanes, maintenance_start):
    inspection = ReceivingPlan(lanes=lanes) if lanes else None
    ready, lots = receive_kits(scenario, inspection, advance_minutes=PLANT.kit_advance)
    assert np.all(ready >= scenario.orders)
    planned = replace(scenario, kits=ready)
    windows = windows_at(maintenance_start)
    model = simulate(planned, replace(PLANT, kit_advance=0), maintenance=windows)
    assert np.all(model.trace[:, ENTRY] >= ready), "unit released before incoming inspection"
    if windows is not None:
        starts, ends = model.maintenance_operations.T
        for start, finish in windows:
            assert not np.any((starts < finish) & (ends > start)), "work during maintenance"
    return model, lots


def count_units(sku, mask):
    return np.bincount(sku[mask], minlength=len(SKU_TARGETS))


def evaluate(scenario, model, lots, outbound):
    good = (scenario.routes == FIRST_PASS) | (scenario.routes == REWORK_PASS)
    ready = model.trace[:, END]
    if outbound is None:
        shipped = np.where(good, ready, np.inf)
    else:
        shipped = dispatch(ready, scenario.orders, good, outbound, shifts=SHIFTS)
        finite = np.isfinite(shipped)
        assert not np.any(finite & (~good | (shipped < ready + outbound.cutoff_minutes * 60)))
        departures = np.arange(SHIFTS)[:, None] * SHIFT_SECONDS + np.asarray(outbound.departure_minutes)[None, :] * 60
        shipped_times, loads = np.unique(shipped[finite], return_counts=True)
        capacity = dict(zip(departures.ravel(), outbound.capacities * SHIFTS, strict=True))
        assert all(time in capacity and load <= capacity[time] for time, load in zip(shipped_times, loads))
    rows = []
    for day in range(SHIFTS):
        left, right = day * SHIFT_SECONDS, (day + 1) * SHIFT_SECONDS
        current = scenario.orders == left
        open_orders = scenario.orders <= left
        complete = ready <= right
        sent = np.isfinite(shipped) & (shipped <= right)
        released_awaiting_ship = good & complete & ~sent
        scrap = ~good & complete
        work_in_process = open_orders & ~complete
        assert not np.any((complete | sent) & ~open_orders)
        assert sum(np.count_nonzero(mask) for mask in
                   (sent, released_awaiting_ship, scrap, work_in_process)) == np.count_nonzero(open_orders)
        good_count = count_units(scenario.sku, current & good & complete)
        shipped_count = count_units(scenario.sku, current & sent)
        assert np.all(shipped_count <= good_count)
        shortage = np.maximum(SKU_TARGETS - shipped_count, 0)
        arrived = [lot for lot in lots if left <= lot.arrival < right]
        rows.append({
            "shift": day + 1,
            "qualified_a": int(good_count[0]), "qualified_b": int(good_count[1]),
            "qualified_on_time": int(good_count.sum()), "shipped_a": int(shipped_count[0]),
            "shipped_b": int(shipped_count[1]), "shipped_on_time": int(shipped_count.sum()),
            "shipped_short_a": int(shortage[0]), "shipped_short_b": int(shortage[1]),
            "shipped_shortfall": int(shortage.sum()), "hit": int(not shortage.any()),
            "qualified_not_shipped_on_time": int(good_count.sum() - shipped_count.sum()),
            "late_shipped": int(np.count_nonzero((shipped > left) & sent & ~current)),
            "dock_backlog": int(np.count_nonzero(released_awaiting_ship)),
            "unfinished_production": int(np.count_nonzero(work_in_process)),
            "receiving_wait_lot_min": sum(lot.start - lot.arrival for lot in arrived) / 60,
            "incoming_lots": len(arrived),
            "maintenance_block_min": float(model.maintenance_wait[day]) / 60,
        })
    return rows


def collect(case, plan, stage, policies):
    count, seed = stage
    histories = {name: [] for name in policies}
    for stream in np.random.SeedSequence(seed).spawn(count):
        scenario = sample(case, plan, stream)
        lines = {}
        for name, policy in policies.items():
            key = (policy.lanes, policy.maintenance_start)
            if key not in lines:
                lines[key] = production(scenario, *key)
            model, lots = lines[key]
            outbound = SHIPPING[policy.outbound] if policy.outbound is not None else None
            histories[name].append(evaluate(scenario, model, lots, outbound))
    return histories


def score(histories):
    worst_hits = min(sum(history[day]["hit"] for history in histories) for day in ADVERSE)
    joint_hits = sum(all(history[day]["hit"] for day in ADVERSE) for history in histories)
    shortage = sum(history[day]["shipped_shortfall"] for history in histories for day in ADVERSE)
    return joint_hits, worst_hits, -shortage


def summarize(histories, baseline, alpha):
    passed = np.asarray([[history[day]["hit"] for day in ADVERSE] for history in histories]).all(axis=1)
    reference = np.asarray([[history[day]["hit"] for day in ADVERSE] for history in baseline]).all(axis=1)
    repetitions, count = len(histories), int(passed.sum())
    lower = float(beta.ppf(alpha, count, repetitions - count + 1)) if count else 0.
    shifts = {}
    for day in range(SHIFTS):
        rows = [history[day] for history in histories]
        reference_rows = [history[day] for history in baseline]
        shifts[day + 1] = {
            "hits": sum(row["hit"] for row in rows),
            "means": {key: round(float(np.mean([row[key] for row in rows])), 3)
                      for key in rows[0] if key not in {"shift", "hit"}},
            "paired_delta_vs_baseline": {
                key: paired_gain(np.asarray([row[key] for row in rows]), np.asarray([row[key] for row in reference_rows]))
                for key in ("qualified_on_time", "shipped_on_time", "shipped_shortfall", "dock_backlog")
            },
        }
    return {
        "both_hit": count, "both_rate": round(count / repetitions, 6), "both_adjusted_lower": round(lower, 6),
        "passes_reliability_gate": lower >= RELIABILITY, "rescued_vs_baseline": int(np.sum(passed & ~reference)),
        "lost_vs_baseline": int(np.sum(~passed & reference)), "shifts": shifts,
    }


def study():
    case, plan = load_case()
    candidates = {policy_label(policy): policy for policy in
                  (Policy(*params) for params in product((1, 2), START_MINUTES, SHIPPING))}
    scores = {label: score(histories) for label, histories in collect(case, plan, DESIGN, candidates).items()}
    winner = max(scores, key=scores.get)
    same_budget = max((label for label in scores if candidates[label].outbound != "extra_capacity"), key=scores.get)
    base = Policy()
    fixed = {
        "baseline": base,
        "relax_receiving": replace(base, lanes=0),
        "relax_maintenance": replace(base, maintenance_start=None),
        "relax_shipping": replace(base, outbound=None),
        "add_receiving_lane": replace(base, lanes=2),
        "reschedule_maintenance": max((p for p in candidates.values() if p.lanes == 1 and p.outbound == "normal"
                                       and p.maintenance_start != MAINTENANCE_START),
                                      key=lambda p: scores[policy_label(p)]),
        "rebalance_same_truck_capacity": replace(base, outbound="equal_capacity"),
        "add_final_truck_capacity": replace(base, outbound="extra_capacity"),
        "faster_manifest": replace(base, outbound="faster_loading"),
        "design_best_same_capacity": candidates[same_budget],
        "design_best": candidates[winner],
    }
    holdout = collect(case, plan, HOLDOUT, fixed)
    alpha = .05 / len(fixed)
    report = {
        "case": case.name, "layout": plan, "plant": asdict(PLANT),
        "metadata": {
            "shift_seconds": SHIFT_SECONDS, "adverse_shifts": [day + 1 for day in ADVERSE],
            "joint_target_by_sku": SKU_TARGETS.tolist(), "maintenance_station": 4,
            "maintenance_duration_minutes": MAINTENANCE_DURATION,
            "receiving": asdict(ReceivingPlan()), "outbound": {name: asdict(config) for name, config in SHIPPING.items()},
        },
        "design": {"n": DESIGN[0], "seed": DESIGN[1], "candidates": scores,
                   "selected": winner, "selected_same_truck_capacity": same_budget},
        "evaluation": {"n": HOLDOUT[0], "seed": HOLDOUT[1], "alpha_per_policy": alpha},
        "evaluated": {name: asdict(policy) for name, policy in fixed.items()},
        "results": {name: summarize(histories, holdout["baseline"], alpha) for name, histories in holdout.items()},
    }
    return report, result_rows(holdout)


def result_rows(histories):
    return [{"policy": name, "replication": rep, **row}
            for name, trials in histories.items() for rep, history in enumerate(trials, 1) for row in history]


def confirm_policy():
    case, plan = load_case()
    one_lane = Policy(maintenance_start=100, outbound="faster_loading")
    fixed = {"baseline": Policy(), "one_lane": one_lane, "two_lanes": replace(one_lane, lanes=2)}
    histories = collect(case, plan, CONFIRMATION, fixed)
    alpha = .05 / len(fixed)
    report = {
        "case": case.name, "joint_target_by_sku": SKU_TARGETS.tolist(),
        "adverse_shifts": [day + 1 for day in ADVERSE],
        "evaluation": {"n": CONFIRMATION[0], "seed": CONFIRMATION[1], "alpha_per_policy": alpha},
        "evaluated": {name: asdict(policy) for name, policy in fixed.items()},
        "results": {name: summarize(trials, histories["baseline"], alpha) for name, trials in histories.items()},
        "paired_two_lanes_vs_one_lane": {
            day + 1: {
                key: paired_gain(
                    np.asarray([history[day][key] for history in histories["two_lanes"]]),
                    np.asarray([history[day][key] for history in histories["one_lane"]]),
                ) for key in ("qualified_on_time", "shipped_on_time", "shipped_shortfall", "dock_backlog")
            } for day in ADVERSE
        },
    }
    return report, result_rows(histories)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--confirm", action="store_true", help="run the independent one- versus two-lane check")
    args = parser.parse_args()
    report, rows = confirm_policy() if args.confirm else study()
    stem = "factory_dispatch_confirmation" if args.confirm else "factory_dispatch_study"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{stem}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    with (args.output_dir / f"{stem}.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: round(value, 6) if isinstance(value, float) else value
                             for key, value in row.items()})


if __name__ == "__main__":
    main()

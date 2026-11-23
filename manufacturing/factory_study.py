"""Compare dispatch and resource choices on paired four-shift factory histories."""

import argparse
import csv
import json
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path

import numpy as np
from scipy.stats import beta

from factory_inputs import ADVERSE, LOT, SHIFTS, SHIFT_SECONDS, load_case, sample
from factory_simulation import (
    ASSEMBLED, DURATION, END, ENTRY, HANDOFF, HELPER, INSPECTED, REPAIR_END, REPAIR_START,
    TEST_START, Plant, simulate,
)
from good_output import FIRST_PASS, REWORK_PASS, paired_gain
from line_balance import ROOT

DESIGN = (120, 2026112291)
HOLDOUT = (600, 2026112292)
SKU_TARGETS = np.array([1155, 495])
RELIABILITY = .95
BASE = Plant()
ABLATIONS = {
    "relax_space": replace(BASE, parking_bays=8, carts=8, parking_length_cm=960),
    "relax_staffing": replace(BASE, staffing=False),
    "relax_fixture": replace(BASE, testers=4),
    "relax_changeover": replace(BASE, setup=False),
    "relax_kits": replace(BASE, kits=False),
}
PRACTICAL = {}
for (policy, run_length), coverage, fixture, advance in product(
    (("fifo", 80), ("ready", 80), ("campaign", 40), ("campaign", 80), ("campaign", 120)),
    (False, True), (1, 2), (0, 30),
):
    label = policy if policy != "campaign" else f"campaign_{run_length}"
    name = f"{label}_cover{int(coverage)}_test{fixture}_early{advance}"
    PRACTICAL[name] = replace(
        BASE, kit_dispatch=policy, max_run=run_length, cover=coverage, testers=fixture, kit_advance=advance,
    )


def interval_seconds(start, end, left, right):
    """Person- or unit-seconds within a particular shift, excluding nonexistent events."""
    valid = np.isfinite(start) & np.isfinite(end)
    return float(np.maximum(np.minimum(end[valid], right) - np.maximum(start[valid], left), 0).sum())


def peak_overlap(start, end, left, right):
    """Reachable simultaneous occupancy at a start event under half-open leases."""
    valid = np.isfinite(start) & np.isfinite(end) & (start < right) & (end > left)
    starts = np.sort(start[valid])
    finishes = np.sort(end[valid])
    if not len(starts):
        return 0
    checks = np.maximum(starts, left)
    occupancy = np.searchsorted(starts, checks, side="right") - np.searchsorted(finishes, checks, side="right")
    return int(occupancy.max())


def measure(scenario, model, plant):
    trace = model.trace
    end = trace[:, END]
    good = (scenario.routes == FIRST_PASS) | (scenario.routes == REWORK_PASS)
    supported = trace[:, HELPER] >= 0
    rework = np.isfinite(trace[:, REPAIR_START])
    leases = trace[rework, HANDOFF]
    lease_returns = trace[rework, REPAIR_START] + plant.return_bin
    eligible = np.maximum(scenario.orders, scenario.kits - 60 * plant.kit_advance)
    if plant.kits and np.any(trace[:, ENTRY] < eligible):
        raise AssertionError("a unit started without a complete kit")
    for index, (start, finish) in enumerate(model.staffing_windows):
        assigned = trace[:, HELPER] == index
        if np.any(trace[assigned, REPAIR_START] < start) or np.any(trace[assigned, REPAIR_END] > finish):
            raise AssertionError("repair extends outside qualified staffing window")
    rows = []
    old_backlog = 0
    for shift in range(SHIFTS):
        left = shift * SHIFT_SECONDS
        right = left + SHIFT_SECONDS
        finished = (end > left) & (end <= right)
        new_orders = scenario.orders == left
        open_orders = scenario.orders <= left
        pending = open_orders & ((end > right) | ~np.isfinite(end))
        started = np.isfinite(trace[:, ENTRY]) & (trace[:, ENTRY] < right)
        on_time = new_orders & finished & good
        released = int((finished & good).sum())
        scrap = int((finished & ~good).sum())
        backlog = int(pending.sum())
        if old_backlog + int(new_orders.sum()) != released + scrap + backlog:
            raise AssertionError(f"unit conservation failed in shift {shift+1}")
        if np.any(finished & ~open_orders):
            raise AssertionError("closed a unit before releasing its order")
        old_backlog = backlog
        amount = np.bincount(scenario.sku[on_time], minlength=len(SKU_TARGETS))
        missing = np.maximum(SKU_TARGETS - amount, 0)
        busy = interval_seconds(trace[supported, REPAIR_START], trace[supported, REPAIR_END], left, right)
        booked = float(model.booked[shift])
        used_windows = model.staffing_windows
        overlapping = (used_windows[:, 0] < right) & (used_windows[:, 1] > left)
        peak_carts = peak_overlap(leases, lease_returns, left, right)
        peak_testers = peak_overlap(trace[:, TEST_START], end, left, right)
        if busy > booked + 1e-5 or booked > 2 * DURATION + 1e-5 or overlapping.sum() > 2:
            raise AssertionError("used more repair support than the roster allowed")
        if peak_carts > plant.waiting_places or peak_testers > plant.testers:
            raise AssertionError("cart or fixture capacity exceeded")
        assembled = np.isfinite(trace[:, ASSEMBLED]) & (trace[:, ASSEMBLED] <= right)
        inspected = np.isfinite(trace[:, INSPECTED]) & (trace[:, INSPECTED] <= right)
        handed = np.isfinite(trace[:, HANDOFF]) & (trace[:, HANDOFF] <= right)
        repairing = np.isfinite(trace[:, REPAIR_START]) & (trace[:, REPAIR_START] <= right)
        test_queue = np.isfinite(trace[:, REPAIR_END]) & (trace[:, REPAIR_END] <= right)
        unstarted = int((pending & ~started).sum())
        assembly = int((pending & started & ~assembled).sum())
        waiting_inspection = int((pending & assembled & ~inspected).sum())
        held_by_inspection = int((pending & inspected & ~handed).sum())
        waiting_repair = int((pending & handed & ~repairing).sum())
        in_repair = int((pending & repairing & ~test_queue).sum())
        retest = int((pending & test_queue).sum())
        if unstarted + assembly + waiting_inspection + held_by_inspection + waiting_repair + in_repair + retest != backlog:
            raise AssertionError("unfinished units do not reconcile by operation")
        rows.append({
            "shift": shift + 1, "offered": int(new_orders.sum()), "on_time_a": int(amount[0]),
            "on_time_b": int(amount[1]), "on_time_good": int(amount.sum()),
            "short_a": int(missing[0]), "short_b": int(missing[1]), "shortfall": int(missing.sum()),
            "hit": int(not missing.any()),
            "physical_good": released, "late_good": released - int(amount.sum()), "scrap": scrap,
            "backlog": backlog, "unstarted": unstarted, "assembly": assembly,
            "waiting_inspection": waiting_inspection, "held_by_inspection": held_by_inspection,
            "waiting_repair": waiting_repair, "in_repair": in_repair, "waiting_or_retesting": retest,
            "inspection_blocked_min": interval_seconds(trace[:, INSPECTED], trace[:, HANDOFF], left, right) / 60,
            "retest_wait_unit_min": interval_seconds(trace[:, REPAIR_END], trace[:, TEST_START], left, right) / 60,
            "kit_wait_min": float(model.kit_wait[shift]) / 60,
            "setup_station_min": float(model.setups[shift].sum()) / 60,
            "changeovers": int(model.changeovers[shift].sum()),
            "support_people": int(overlapping.sum()),
            "support_booked_person_min": booked / 60, "support_busy_person_min": busy / 60,
            "peak_cart_places": peak_carts, "peak_testers": peak_testers,
        })
    return rows


def collect(case, plan, stage, policies, export=False):
    count, seed = stage
    by_policy = {name: [] for name in policies}
    for rep, stream in enumerate(np.random.SeedSequence(seed).spawn(count), 1):
        scenario = sample(case, plan, stream)
        if len(scenario.orders) != LOT * SHIFTS:
            raise AssertionError("unexpected order volume")
        for name, plant in policies.items():
            rows = measure(scenario, simulate(scenario, plant), plant)
            by_policy[name].append(rows if export else [rows[shift] for shift in ADVERSE])
    return by_policy


def design_scores(by_policy):
    scores = {}
    for name, histories in by_policy.items():
        worst_shift_hits = min(sum(history[i]["hit"] for history in histories) for i in range(len(ADVERSE)))
        both = sum(all(row["hit"] for row in history) for history in histories)
        shortage = sum(row["shortfall"] for history in histories for row in history)
        scores[name] = (both, worst_shift_hits, -shortage)
    return scores


def summarize(histories, baseline, alpha):
    measured = (
        "on_time_a", "on_time_b", "on_time_good", "short_a", "short_b", "shortfall", "physical_good",
        "late_good", "scrap", "backlog", "unstarted", "assembly", "waiting_inspection",
        "held_by_inspection", "waiting_repair", "in_repair", "waiting_or_retesting", "inspection_blocked_min",
        "retest_wait_unit_min", "kit_wait_min", "setup_station_min", "changeovers",
        "support_people", "support_booked_person_min", "support_busy_person_min", "peak_cart_places", "peak_testers",
    )
    passes = np.array([[h[day]["hit"] for day in ADVERSE] for h in histories], dtype=bool).all(axis=1)
    reference = np.array([[h[day]["hit"] for day in ADVERSE] for h in baseline], dtype=bool).all(axis=1)
    n, hits = len(histories), int(passes.sum())
    lower = float(beta.ppf(alpha, hits, n - hits + 1)) if hits else 0
    shifts = {}
    for day in range(SHIFTS):
        data = [h[day] for h in histories]
        base = [h[day] for h in baseline]
        shifts[day + 1] = {
            "hits": sum(r["hit"] for r in data),
            "means": {metric: round(float(np.mean([r[metric] for r in data])), 3) for metric in measured},
            "paired_delta_vs_baseline": {
                metric: paired_gain(np.array([r[metric] for r in data]), np.array([r[metric] for r in base]))
                for metric in ("on_time_good", "shortfall", "backlog", "inspection_blocked_min")
            },
        }
    return {
        "replications": n, "both_hit": hits, "both_rate": round(hits / n, 6),
        "both_adjusted_lower": round(lower, 6), "passes_reliability_gate": bool(lower >= RELIABILITY),
        "rescued_vs_baseline": int((passes & ~reference).sum()),
        "lost_vs_baseline": int((~passes & reference).sum()), "shifts": shifts,
    }


def study():
    case, plan = load_case()
    # The unmodified FIFO entry is part of the search family and shares the baseline configuration.
    design = design_scores(collect(case, plan, DESIGN, PRACTICAL))
    winner = max(design, key=design.get)
    dispatch_only = max((name for name in design if name.endswith("_cover0_test1_early0")), key=design.get)
    controls = ("fifo_cover1_test1_early0", "fifo_cover0_test2_early0", "fifo_cover0_test1_early30")
    one_action = max(controls, key=design.get)
    policies = {"baseline": BASE, **ABLATIONS}
    # Preserve known practical controls even when not chosen by the screening.
    for name in dict.fromkeys((*controls, dispatch_only, one_action, winner)):
        if PRACTICAL[name] != BASE:
            policies[name] = PRACTICAL[name]
    raw = collect(case, plan, HOLDOUT, policies, export=True)
    alpha = .05 / len(policies)
    results = {name: summarize(h, raw["baseline"], alpha) for name, h in raw.items()}
    report = {
        "case": case.name, "assembly_plan": plan, "joint_target_by_sku": SKU_TARGETS.tolist(),
        "adverse_shifts": [day + 1 for day in ADVERSE], "design": {"n": DESIGN[0], "seed": DESIGN[1]},
        "evaluation": {"n": HOLDOUT[0], "seed": HOLDOUT[1], "alpha_per_policy": alpha},
        "design_candidates": {
            name: {"plant": asdict(PRACTICAL[name]), "scores": score} for name, score in design.items()
        },
        "selected": {"dispatch_only": dispatch_only, "single_resource_action": one_action, "combined": winner},
        "ablation_names": list(ABLATIONS),
        "evaluated": {name: asdict(plant) for name, plant in policies.items()},
        "results": results,
    }
    rows = [
        {"policy": name, "replication": rep, **row}
        for name, histories in raw.items()
        for rep, history in enumerate(histories, 1)
        for row in history
    ]
    return report, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    report, rows = study()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "factory_study.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    with (args.output_dir / "factory_study.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: round(v, 6) if isinstance(v, float) else v for k, v in row.items()})


if __name__ == "__main__":
    main()

"""Track inspection and rework carryover through normal, adverse, and recovery shifts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from good_output import (
    BASE_DEFECT_PCT, FAST_FACTOR, FIRST_PASS, INSPECTION_SECONDS, REPAIR_SECONDS, REWORK_PASS,
    STRESS_DEFECT_PCT, assembly_quality_batches, fifo_finishes, paired_gain, quality_routes,
)
from line_balance import ROOT, Case, reference_plans
from rework_support import concentrate_failures, dispatch_rework_windows
from shift_capacity import DEMAND, SHIFT_SECONDS

PHASES = ("normal",) * 2 + ("adverse",) * 4 + ("recovery",) * 3
PULSE_START_MINUTE = 285
HISTORIES, SEED = 400, 2026112225
POLICIES = {
    "no_support": (0, 0),
    "one_90": (1, 90),
    "one_240": (1, 240),
    "two_120": (2, 120),
}
Lot = tuple[np.ndarray, np.ndarray]


def support_windows(phases: tuple[str, ...], people: int, minutes: int) -> tuple[tuple[int, int], ...]:
    """Place each helper's window at the end of every predeclared adverse shift."""
    duration = 60 * minutes
    return tuple(
        ((day + 1) * SHIFT_SECONDS - duration, duration)
        for day, phase in enumerate(phases) if phase == "adverse" for _ in range(people)
    )


def sample_lots(case: Case, plan: list[list[int]], phase: str, count: int, seed: np.random.SeedSequence) -> list[Lot]:
    times, placement = seed.spawn(2)
    shuffle = np.random.default_rng(placement)
    result = []
    buffers = (1,) * (len(plan) - 1)
    adverse = phase == "adverse"
    assembly_seed = int(times.generate_state(1, dtype=np.uint64)[0])
    for exits_by_speed, draws in assembly_quality_batches(case, plan, buffers, {FAST_FACTOR}, count, assembly_seed):
        for exits, shift_draws in zip(exits_by_speed[FAST_FACTOR].T, draws, strict=True):
            gross = int(np.searchsorted(exits, SHIFT_SECONDS, side="right"))
            actual_exits = exits[:gross].copy()
            routes = quality_routes(shift_draws[:gross], STRESS_DEFECT_PCT if adverse else BASE_DEFECT_PCT)
            if adverse:
                projected_inspection = fifo_finishes(actual_exits, 1, INSPECTION_SECONDS)
                routes = concentrate_failures(
                    routes, projected_inspection, PULSE_START_MINUTE,
                    shuffle.permutation(gross), shuffle.permutation(routes[routes != FIRST_PASS]),
                )
            result.append((actual_exits, routes))
    return result


def period_counts(events: np.ndarray, cutoffs: np.ndarray) -> np.ndarray:
    """Assign each event to the first shift cutoff at or after its timestamp."""
    return np.diff(np.searchsorted(np.sort(events), cutoffs, side="right"), prepend=0)


def inspect_history(lots: tuple[Lot, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    origin = np.repeat(np.arange(len(lots)), [len(exits) for exits, _ in lots])
    exits = np.concatenate([exits + day * SHIFT_SECONDS for day, (exits, _) in enumerate(lots)])
    routes = np.concatenate([routes for _, routes in lots])
    return origin, fifo_finishes(exits, 1, INSPECTION_SECONDS), routes


def measure_history(
    origin: np.ndarray, inspection: np.ndarray, routes: np.ndarray, phases: tuple[str, ...],
    people: int, minutes: int,
) -> tuple[list[dict], int]:
    """Credit conforming units to their assembly cohort and preserve unfinished work at every cutoff."""
    days = len(phases)
    cutoffs = SHIFT_SECONDS * np.arange(1, days + 1)
    recovery_start = phases.index("recovery")
    rework_mask = routes >= REWORK_PASS
    good_mask = (routes == FIRST_PASS) | (routes == REWORK_PASS)
    rework_entries = inspection[rework_mask]
    rework_finishes, helper_jobs = dispatch_rework_windows(
        rework_entries, support_windows(phases, people, minutes),
    )
    dispositions = inspection.copy()
    dispositions[rework_mask] = rework_finishes
    on_time = good_mask & (dispositions <= cutoffs[origin])
    due_released = np.bincount(origin[on_time], minlength=days)
    physically_released = period_counts(dispositions[good_mask], cutoffs)
    disposed_scrap = period_counts(dispositions[~good_mask], cutoffs)
    gross = np.bincount(origin, minlength=days)
    inspected = period_counts(inspection, cutoffs)
    rework_arrivals = period_counts(rework_entries, cutoffs)
    rework_completed = period_counts(rework_finishes, cutoffs)
    pending_inspection = np.cumsum(gross - inspected)
    pending_rework = np.cumsum(rework_arrivals - rework_completed)
    if not np.array_equal(np.cumsum(gross - physically_released - disposed_scrap), pending_inspection + pending_rework):
        raise AssertionError("shift cutoff does not conserve the assembled cohort")
    rows = []
    for day, cutoff in enumerate(cutoffs):
        old = (origin < recovery_start) & (origin <= day)
        rows.append({
            "shift": day + 1, "phase": phases[day], "gross": int(gross[day]),
            "physical_released": int(physically_released[day]), "origin_released_on_time": int(due_released[day]),
            "late_released": int(physically_released[day] - due_released[day]),
            "scrap": int(disposed_scrap[day]), "pending_inspection": int(pending_inspection[day]),
            "rework_arrivals": int(rework_arrivals[day]), "rework_completed": int(rework_completed[day]),
            "pending_rework": int(pending_rework[day]),
            "pre_recovery_origin_pending": int(np.count_nonzero(old & (dispositions > cutoff))),
            "reserved_person_minutes": people * minutes if phases[day] == "adverse" else 0,
        })
    return rows, helper_jobs


def recovery_shifts(rows: list[dict]) -> int | None:
    """Count recovery shifts until all pre-recovery units have been released or scrapped."""
    first_recovery = next(i for i, row in enumerate(rows) if row["phase"] == "recovery")
    return next(
        (i - first_recovery + 1 for i in range(first_recovery - 1, len(rows))
         if rows[i]["pre_recovery_origin_pending"] == 0), None,
    )


def run_study() -> tuple[dict, list[dict]]:
    records = json.loads((ROOT / "data" / "benchmarks.json").read_text(encoding="utf-8"))
    case = Case(**next(record for record in records if record["name"] == "Mitchell"))
    plan = reference_plans(case, 15)["MILP"]
    origins = [sample_lots(case, plan, phase, HISTORIES, seed)
               for phase, seed in zip(PHASES, np.random.SeedSequence(SEED).spawn(len(PHASES)), strict=True)]
    by_policy = {name: [] for name in POLICIES}
    helper_busy = {name: [] for name in POLICIES}
    raw = []
    for history, lots in enumerate(zip(*origins, strict=True), start=1):
        origin, inspection, routes = inspect_history(lots)
        for name, (people, minutes) in POLICIES.items():
            rows, helper_jobs = measure_history(origin, inspection, routes, PHASES, people, minutes)
            busy_minutes = helper_jobs * REPAIR_SECONDS / 60
            if busy_minutes > sum(row["reserved_person_minutes"] for row in rows):
                raise AssertionError("helpers exceeded their reserved labor budget")
            by_policy[name].append(rows)
            helper_busy[name].append(busy_minutes)
            raw.extend({"history": history, "policy": name, **row} for row in rows)

    first_adverse, last_adverse = PHASES.index("adverse"), len(PHASES) - 1 - PHASES[::-1].index("adverse")
    summary = {}
    for name, histories in by_policy.items():
        by_shift = []
        for day, cohort in enumerate(zip(*histories, strict=True), start=1):
            by_shift.append({
                "shift": day, "phase": PHASES[day - 1], "histories_at_due_target": sum(
                    row["origin_released_on_time"] >= DEMAND for row in cohort),
                "histories_at_physical_target": sum(row["physical_released"] >= DEMAND for row in cohort),
                "histories_only_at_physical_target": sum(
                    row["physical_released"] >= DEMAND > row["origin_released_on_time"] for row in cohort),
                **{f"mean_{key}": round(float(np.mean([row[key] for row in cohort])), 3)
                   for key in ("gross", "origin_released_on_time", "late_released", "rework_arrivals", "rework_completed",
                               "pending_inspection", "pending_rework", "pre_recovery_origin_pending")},
            })
        recoveries = [recovery_shifts(history) for history in histories]
        summary[name] = {
            "by_shift": by_shift,
            "origins_recovered_by_recovery_shift": [sum(value is not None and value <= day for value in recoveries)
                                                   for day in range(PHASES.count("recovery") + 1)],
            "mean_stress_helper_busy_person_minutes": round(float(np.mean(helper_busy[name])), 3),
            "mean_adverse_backlog_growth_per_shift": round(float(np.mean([
                (history[last_adverse]["pending_rework"] - history[first_adverse - 1]["pending_rework"])
                / PHASES.count("adverse") for history in histories])), 3),
            "mean_backlog_growth_after_first_adverse_shift": round(float(np.mean([
                (history[last_adverse]["pending_rework"] - history[first_adverse]["pending_rework"])
                / (last_adverse - first_adverse) for history in histories])), 3),
        }

    backlog = {name: np.array([history[last_adverse]["pending_rework"] for history in histories])
               for name, histories in by_policy.items()}
    return {
        "case": case.name, "histories": HISTORIES, "seed": SEED, "phases": PHASES,
        "shift_seconds": SHIFT_SECONDS, "due_target": DEMAND, "adverse_pulse_start_minute": PULSE_START_MINUTE,
        "normal_first_defect_pct": BASE_DEFECT_PCT, "adverse_first_defect_pct": STRESS_DEFECT_PCT,
        "policies": {name: {"people": people, "window_minutes": minutes,
                            "person_minutes_per_adverse_shift": people * minutes}
                     for name, (people, minutes) in POLICIES.items()},
        "summary": summary,
        "last_adverse_pending_difference_two_120_minus_one_240": paired_gain(backlog["two_120"], backlog["one_240"]),
    }, raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    report, rows = run_study()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "rework_carryover.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "rework_carryover_histories.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

"""Factorial experiment on material release, assembly capacity and repair in a mixed-model line."""

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from functools import partial
from itertools import combinations, product
from pathlib import Path

import numpy as np
from scipy.stats import beta

from decision_economics import paired
from factory_capacity_plan import ORIGINAL_TARGET, design_promise
from factory_dispatch_study import PLANT, SHIPPING
from factory_floor import ACTIONS
from factory_floor_study import ACTIONS_LIMIT, POSTURE_LIMIT, calendar, second_attempt
from factory_inputs import ADVERSE, SHIFTS, SHIFT_SECONDS, load_case, sample
from factory_rebalance import FLOOR, cohort_rows, update_cohort_stages
from factory_receiving import ReceivingPlan, receive_kits
from factory_shipping import dispatch
from factory_simulation import END, ENTRY, TEST_START, simulate
from good_output import FIRST_PASS, REWORK_PASS
from line_balance import ROOT

DESIGN = (160, 2026112531)
HOLDOUT = (800, 2026112532)
COMMON_TARGETS = (330, 400, 500, ORIGINAL_TARGET)
CAPITAL_LIMITS = (0, 20000, 50000, 85000, 90000, 110000)
FACTORS = ("supply", "assembly", "rework")


@dataclass(frozen=True)
class Policy:
    supply: bool
    assembly: bool
    rework: bool

    @property
    def name(self):
        return "+".join(f for f in FACTORS if getattr(self, f)) or "base"

    def capital(self, rates):
        return (self.supply * rates["supply_change_investment"]
                + self.assembly * rates["parallel_station_investment"]
                + self.rework * rates["rework_fixture_investment"])


POLICIES = tuple(Policy(*flags) for flags in product((False, True), repeat=len(FACTORS)))


def cost_components(policy, rates, row):
    """Recurring and annualized costs for the four-shift operating regime."""
    hours = SHIFT_SECONDS / 3600
    annual_shifts = rates["shifts_per_year"]
    operating = (
        hours * (8 * rates["assembly_hourly_rate"] + policy.assembly * rates["parallel_operator_hourly_rate"])
        + sum(FLOOR.buffers) * rates["buffer_space_per_slot_year"] / annual_shifts
        + row["primary_stop_minutes"] * rates["primary_station_downtime_per_minute"]
        + row["first_attempts"] * (rates["first_repair_per_completed_attempt"]
                                   + rates["first_recheck_per_completed_attempt"])
        + row["second_attempts"] * rates["second_repair_and_recheck_per_completed_attempt"]
        + row["scrap_this_shift"] * rates["scrap_per_disposed_unit"]
        + policy.supply * rates["supply_change_extra_cost_per_shift"]
        + policy.assembly * rates["parallel_maintenance_extra_cost_per_shift"]
        + policy.rework * rates["rework_fixture_extra_cost_per_shift"]
    )
    return operating, policy.capital(rates) / (annual_shifts * rates["investment_life_years"])


def shift_measures(scenario, model, gates, windows, retries, staffed):
    ready, routes, attempted = second_attempt(model.trace[:, END], scenario.routes, retries)
    good = (routes == FIRST_PASS) | (routes == REWORK_PASS)
    shipped = dispatch(ready, scenario.orders, good, SHIPPING["faster_loading"])
    finite = np.isfinite(shipped)
    assert np.all(model.trace[:, ENTRY] >= gates)
    cutoff_seconds = SHIPPING["faster_loading"].cutoff_minutes * 60
    assert np.all(good[finite]) and np.all(shipped[finite] >= ready[finite] + cutoff_seconds)
    starts, ends = model.maintenance_operations.T
    for alternate in (False, True):
        mask = model.parallel_used == alternate
        order = np.argsort(starts[mask])
        assert np.all(starts[mask][order][1:] >= ends[mask][order][:-1])
        for start, finish in windows + 2400 * alternate:
            assert not np.any(mask & (starts < finish) & (ends > start))
    alt_day = (starts[model.parallel_used] // SHIFT_SECONDS).astype(int)
    assert np.all(staffed[alt_day])
    assert np.all(ends[model.parallel_used] <= (alt_day+1) * SHIFT_SECONDS)

    ledger = model.station_seconds
    assert np.all(ledger.sum(axis=2) <= SHIFT_SECONDS + 1e-6)
    rows = cohort_rows(scenario, ready, good, shipped)
    first_end = model.trace[:, END]
    for day, row in enumerate(rows):
        left, right = day*SHIFT_SECONDS, (day+1)*SHIFT_SECONDS
        update_cohort_stages(row, day, scenario.orders, model.trace, ready, good)
        first_done = np.isfinite(model.trace[:, TEST_START]) & (first_end > left) & (first_end <= right)
        finished = (ready > left) & (ready <= right)
        scheduled = model.parallel_used & (starts >= left) & (starts < right)
        alt = np.bincount(scenario.sku[scheduled], minlength=2)
        counts = model.floor_stats[day, :, :2].copy()
        counts[3] -= alt
        peak = max((counts * np.asarray(ACTIONS).T).sum(axis=1).max(), alt @ np.asarray(ACTIONS)[:, 3])/420
        posture = model.floor_stats[day, 4, 2] / 60
        stopped = np.maximum(0, np.minimum(windows[:, 1], right)-np.maximum(windows[:, 0], left)).sum()/60
        row.update(
            first_attempts=int(first_done.sum()),
            second_attempts=int(finished[attempted].sum()),
            scrap_this_shift=int((finished & ~good).sum()),
            primary_stop_minutes=float(stopped),
            peak_actions_per_min=float(peak),
            station_five_posture_min=float(posture),
            ergo_pass=int(peak <= ACTIONS_LIMIT and posture <= POSTURE_LIMIT),
            parallel_jobs=int(alt.sum()),
            kit_wait_min=float(model.kit_wait[day]/60),
            station3_block_min=float(ledger[day, 2, 2]/60),
            station4_calendar_wait_min=float(ledger[day, 3, 4]/60),
            station6_processing_min=float(ledger[day, 5, 0]/60),
        )
    return rows


@dataclass(frozen=True)
class Pressure:
    late_receipts_minutes: int = 0
    adverse_qualified_availability: float = .82
    improved_repair_seconds: int = 60


def trial(case, plan, seed, *, pressure=Pressure(), policies=POLICIES, release_rule="ready", batch_size=80,
          measure_release=False):
    base_seed, extra_seed, roster_seed = seed.spawn(3)
    scenario = sample(case, plan, base_seed)
    extra = np.random.default_rng(extra_seed)
    windows = calendar(extra.integers(145, 345, 2), extra.integers(12, 31, 2), True)
    retries = extra.random((len(scenario.kits), 2))
    availability = np.full(SHIFTS, .82)
    availability[list(ADVERSE)] = pressure.adverse_qualified_availability
    staffed = np.random.default_rng(roster_seed).random(SHIFTS) < availability
    if pressure.late_receipts_minutes:
        affected = np.isin(scenario.orders, np.array(ADVERSE)*SHIFT_SECONDS)
        affected &= scenario.kits > scenario.orders
        scenario = replace(scenario, kits=scenario.kits + affected*60*pressure.late_receipts_minutes)
    gates = {supply: receive_kits(scenario, ReceivingPlan(), advance_minutes=(55 if supply else 30))[0]
             for supply in (False, True)}
    scenarios = {supply: replace(scenario, kits=gates[supply]) for supply in gates}
    result = {}
    for policy in policies:
        plant = replace(PLANT, kit_advance=0, kit_dispatch=release_rule, max_run=batch_size,
                        repair_seconds=pressure.improved_repair_seconds if policy.rework else 90)
        floor = replace(FLOOR, route_minutes=10) if policy.supply else FLOOR
        model = simulate(scenarios[policy.supply], plant, maintenance=windows, floor=floor,
                         parallel_available=staffed if policy.assembly else None, diagnose=True)
        rows = shift_measures(scenario, model, gates[policy.supply], windows, retries, staffed)
        if measure_release:
            for day, row in enumerate(rows):
                row["entry_changeovers"] = int(model.changeovers[day, 0])
        result[policy.name] = rows
    return result


def collect(case, plan, stage, workers):
    seeds = np.random.SeedSequence(stage[1]).spawn(stage[0])
    run = partial(trial, case, plan)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run, seeds, chunksize=4))


def adverse_units(history, key):
    return np.array([[[r["a"], r["b"]] for r in (h[key][i] for i in ADVERSE)] for h in history])


def verify_target(adverse, target, alpha):
    quotas = np.array([target*7//10, target*3//10])
    hits = int(np.all(adverse >= quotas, axis=(1, 2)).sum())
    lower = beta.ppf(alpha, hits, len(adverse)-hits+1) if hits else 0.
    return dict(target=target, quotas=quotas.tolist(), hits=hits, lower=round(float(lower), 6),
                passed=bool(target > 0 and lower >= .95))


def contrasts(shortfalls):
    """Compare factorial averages and every one-step change on matched trajectories."""
    factorial = {}
    for n in (1, 2, 3):
        for factors in combinations(FACTORS, n):
            weights = [np.prod([1 if getattr(policy, factor) else -1 for factor in factors])
                       for policy in POLICIES]
            # Standard two-level factorial effect (one half of high-minus-low difference for interactions).
            effect = -(np.asarray(weights) @ shortfalls) / (len(POLICIES)/2)
            factorial["+".join(factors)] = paired(effect, 7)
    incremental = {}
    for policy in POLICIES:
        low = POLICIES.index(policy)
        for factor in FACTORS:
            if not getattr(policy, factor):
                high = POLICIES.index(replace(policy, **{factor: True}))
                incremental[f"{policy.name} -> {POLICIES[high].name}"] = paired(
                    shortfalls[low]-shortfalls[high], 12)
    return factorial, incremental


def report(holdout, promises, rates, out):
    alpha = .05/(len(POLICIES)*(len(COMMON_TARGETS)+1))
    costs, outcomes, shortfalls = {}, {}, []
    out.mkdir(parents=True, exist_ok=True)
    evidence_dir = out/"mixed_line"
    evidence_dir.mkdir(exist_ok=True)
    for policy in POLICIES:
        key = policy.name
        rows = [h[key] for h in holdout]
        adverse = adverse_units(holdout, key)
        shortfalls.append([sum(row["short_a"]+row["short_b"] for row in cohort) for cohort in rows])
        cost = np.array([[cost_components(policy, rates, row) for row in cohort] for cohort in rows])
        costs[key] = cost.sum(axis=2).mean(axis=1)
        total_shipped = sum(row["a"]+row["b"]+row["old_sent"] for cohort in rows for row in cohort)
        stats = {field:round(float(np.mean([h[2][field] for h in rows])), 5)
                 for field in rows[0][2] if field != "shift"}
        ergo_rate = np.mean([all(h[i]["ergo_pass"] for i in ADVERSE) for h in rows])
        targets = sorted(set((*COMMON_TARGETS, promises[key])))
        outcomes[key] = dict(
            capital=policy.capital(rates), annual_operating_cost=round(float(cost[:,:,0].mean()*rates["shifts_per_year"]), 4),
            mean_total_cost_per_shift=round(float(costs[key].mean()), 4),
            cost_per_physically_shipped=round(float(cost.sum()/total_shipped), 5) if total_shipped else None,
            mean_four_shift_shortfall=round(float(np.mean(shortfalls[-1])), 5),
            mean_shift_3=stats, joint_ergonomics_pass_rate=round(float(ergo_rate), 6),
            design_promise=promises[key],
            reliability={target:verify_target(adverse, target, alpha) for target in targets},
        )
        with (evidence_dir/f"{key}.csv").open("w", newline="") as file:
            writer = None
            for i, cohort in enumerate(rows, 1):
                for row, (operating_cost, capital_cost) in zip(cohort, cost[i-1], strict=True):
                    evidence = dict(policy=key, trajectory=i, **row, operating_cost=float(operating_cost),
                                    allocated_investment=float(capital_cost))
                    if writer is None:
                        writer = csv.DictWriter(file, fieldnames=evidence, lineterminator="\n")
                        writer.writeheader()
                    writer.writerow({k:round(v, 6) if isinstance(v, float) else v for k, v in evidence.items()})

    paired_shortfalls = np.asarray(shortfalls)
    factorial, incremental = contrasts(paired_shortfalls)
    for policy, shortage in zip(POLICIES, paired_shortfalls, strict=True):
        key = policy.name
        if key != "base":
            outcomes[key]["paired_four_shift_shortfall_reduction"] = paired(paired_shortfalls[0]-shortage, 7)
            outcomes[key]["paired_shift_cost_change"] = paired(costs[key]-costs["base"], 7)

    def qualified(key, target):
        outcome = outcomes[key]
        return (outcome["joint_ergonomics_pass_rate"] == 1.
                and outcome["reliability"].get(target, {}).get("passed", False))

    def cheapest(keys):
        return min(keys, key=lambda k: (outcomes[k]["mean_total_cost_per_shift"], k), default=None)

    same_target = {target:cheapest(k for k in outcomes if qualified(k, target)) for target in COMMON_TARGETS}
    budgets = {}
    for cap in CAPITAL_LIMITS:
        feasible = [(target, key) for key, outcome in outcomes.items() if outcome["capital"] <= cap
                    for target in outcome["reliability"] if qualified(key, target)]
        best_target = max((target for target, _ in feasible), default=None)
        budgets[cap] = dict(target=best_target, policy=cheapest(k for target,k in feasible if target==best_target))
    result = dict(design=DESIGN, holdout=HOLDOUT, assumptions=rates,
                  settings=dict(dispatch="ready", floor="compact_aid_two_per_gap", target_a_b=[1155,495],
                                alternate_operator_availability=.82, supply_advance=[30,55], route_minutes=[15,10],
                                first_repair_seconds=[90,60], unchanged_second_repair_seconds=210),
                  confidence_alpha_per_goal=alpha, outcomes=outcomes, factorial_shortfall_effects=factorial,
                  incremental_shortfall_reduction=incremental,
                  cheapest_at_common_target=same_target, largest_tested_target_by_capital=budgets)
    (out/"mixed_line_decisions.json").write_text(json.dumps(result, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"results")
    parser.add_argument("--rates", type=Path, default=ROOT/"data"/"mixed_line_costs.json")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    rates = json.loads(args.rates.read_text())
    if min(rates["shifts_per_year"], rates["investment_life_years"]) <= 0 or any(
            not np.isfinite(v) or v < 0 for k,v in rates.items() if k != "currency"):
        raise ValueError("cost rates must be finite and nonnegative, with positive life and shifts")
    case, plan = load_case()
    design = collect(case, plan, DESIGN, args.workers)
    promises = {p.name:design_promise(adverse_units(design, p.name)) for p in POLICIES}
    print(f"Design quantities locked: {promises}", flush=True)
    holdout = collect(case, plan, HOLDOUT, args.workers)
    report(holdout, promises, rates, args.output_dir)


if __name__ == "__main__":
    main()

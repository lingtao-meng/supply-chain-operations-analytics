"""Verify prelocked mixed-line delivery promises under three stresses and their combination."""

import argparse
import csv
import gzip
import hashlib
import io
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np

from decision_economics import paired
from factory_capacity_plan import design_promise
from factory_inputs import ADVERSE, load_case
from line_balance import ROOT
from mixed_line_decisions import POLICIES, Pressure, cost_components, trial, verify_target

DESIGN = (160, 2026112541)
HOLDOUT = (800, 2026112542)
COMMON_TARGETS = (330, 500, 640, 650)
CAPITAL_LIMITS = (0, 25000, 50000, 85000, 90000, 110000)
PRESSURES = {
    "reference": Pressure(),
    "late_receipts": Pressure(late_receipts_minutes=20),
    "low_staffing": Pressure(adverse_qualified_availability=.65),
    "slower_repair": Pressure(improved_repair_seconds=75),
    "combined": Pressure(20, .65, 75),
}
COLUMNS = ("a_first", "b_first", "a_second", "b_second", "four_shift_shortfall",
           "mean_shift_cost", "ergonomic_screen")


def replay_panel(case, plan, rates, seed):
    scenarios = []
    for pressure in PRESSURES.values():
        # Spawn the same task, receipt, repair and roster draws in every pressure scenario.
        identical_seed = np.random.SeedSequence(seed.entropy, spawn_key=seed.spawn_key)
        histories = trial(case, plan, identical_seed, pressure=pressure)
        values = []
        for policy in POLICIES:
            rows = histories[policy.name]
            by_sku = [rows[day][sku] for day in ADVERSE for sku in ("a", "b")]
            shortfall = sum(row["short_a"] + row["short_b"] for row in rows)
            cost = sum(sum(cost_components(policy, rates, row)) for row in rows)/len(rows)
            passed = all(rows[day]["ergo_pass"] for day in ADVERSE)
            values.append((*by_sku, shortfall, cost, int(passed)))
        scenarios.append(values)
    return scenarios


def collect_panel(case, plan, rates, stage, workers):
    seeds = np.random.SeedSequence(stage[1]).spawn(stage[0])
    replay = partial(replay_panel, case, plan, rates)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return np.asarray(list(pool.map(replay, seeds, chunksize=4)))


def summarize(panel, promises, rates):
    alpha = .05/(len(PRESSURES)*len(POLICIES)*(len(COMMON_TARGETS)+1))
    effect_family = (len(PRESSURES)-1)*len(POLICIES)
    outcomes = {}
    for index, policy in enumerate(POLICIES):
        robust = int(promises[index])
        verified_targets = sorted(set((*COMMON_TARGETS, robust)))
        scenarios = {}
        for pos, name in enumerate(PRESSURES):
            series = panel[:, pos, index]
            adverse = series[:, :4].reshape(-1, 2, 2)
            effect = series[:, 4] - panel[:, 0, index, 4]
            scenarios[name] = dict(
                mean_four_shift_shortfall=round(float(series[:, 4].mean()), 5),
                mean_abnormal_shift_2=[round(float(x), 5) for x in series[:, 2:4].mean(axis=0)],
                mean_shift_cost=round(float(series[:, 5].mean()), 4),
                ergonomic_passes=int(series[:, 6].sum()),
                paired_added_shortfall_vs_reference=None if not pos else paired(effect, effect_family),
                reliability={target:verify_target(adverse, target, alpha) for target in verified_targets},
            )
        passed_all = [
            target for target in verified_targets
            if all(s["reliability"][target]["passed"] and s["ergonomic_passes"] == len(panel)
                   for s in scenarios.values())
        ]
        outcomes[policy.name] = dict(
            capital=policy.capital(rates),
            locked_pressure_promise=robust,
            passed_all_pressure_targets=passed_all,
            scenarios=scenarios,
        )

    def worst_cost(key):
        return max(row["mean_shift_cost"] for row in outcomes[key]["scenarios"].values())

    same_service = {}
    for target in COMMON_TARGETS:
        candidates = [key for key, outcome in outcomes.items() if target in outcome["passed_all_pressure_targets"]]
        winner = min(candidates, key=lambda key:(worst_cost(key), key), default=None)
        same_service[target] = winner

    by_budget = {}
    for cap in CAPITAL_LIMITS:
        candidates = [(target, key) for key, outcome in outcomes.items() if outcome["capital"] <= cap
                      for target in outcome["passed_all_pressure_targets"]]
        selected = min(candidates, key=lambda item:(-item[0], worst_cost(item[1]), item[1]), default=None)
        by_budget[cap] = dict(target=selected[0], policy=selected[1]) if selected else None

    return dict(
        design=DESIGN, holdout=HOLDOUT, confidence_alpha_per_cell=alpha,
        assumptions=rates, pressures={name:asdict(pressure) for name,pressure in PRESSURES.items()},
        predeclared_common_targets=COMMON_TARGETS, outcomes=outcomes,
        lowest_worst_case_cost_at_common_target=same_service, highest_verified_by_capital=by_budget,
    )


def write_evidence(panel, out):
    data = io.StringIO(newline="")
    writer = csv.writer(data, lineterminator="\n")
    writer.writerow(("trajectory","pressure","policy",*COLUMNS))
    for trajectory_id, trajectory in enumerate(panel, 1):
        for name, cases in zip(PRESSURES, trajectory, strict=True):
            for policy, row in zip(POLICIES, cases, strict=True):
                writer.writerow((trajectory_id, name, policy.name, *map(int, row[:5]), round(float(row[5]), 6), int(row[6])))
    raw = data.getvalue().encode()
    with (out/"mixed_line_stress.csv.gz").open("wb") as file:
        with gzip.GzipFile(filename="", mode="wb", fileobj=file, mtime=0) as compressed:
            compressed.write(raw)
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"results")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    case, plan = load_case()
    rates = json.loads((ROOT/"data"/"mixed_line_costs.json").read_text())
    design = collect_panel(case, plan, rates, DESIGN, args.workers)
    promises = [min(design_promise(design[:, scenario, i, :4].reshape(-1, 2, 2))
                    for scenario in range(len(PRESSURES))) for i in range(len(POLICIES))]
    print("Locked pressure promises:", dict(zip((p.name for p in POLICIES), promises, strict=True)), flush=True)
    panel = collect_panel(case, plan, rates, HOLDOUT, args.workers)
    report = summarize(panel, promises, rates)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report["uncompressed_evidence_sha256"] = write_evidence(panel, args.output_dir)
    (args.output_dir/"mixed_line_stress.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()

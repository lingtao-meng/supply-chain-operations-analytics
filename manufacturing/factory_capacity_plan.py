"""Screen constrained factory designs and verify a joint two-shift delivery promise."""

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path

import numpy as np
from scipy.stats import beta

from factory_dispatch_study import PLANT, SHIPPING, evaluate
from factory_inputs import ADVERSE, SHIFTS, SHIFT_SECONDS, load_case, sample
from factory_receiving import ReceivingPlan, receive_kits
from factory_simulation import MAINTENANCE_STATION, simulate
from line_balance import ROOT

DESIGN = (175, 2026112321)
HOLDOUT = (800, 2026112322)
SENSITIVITY = (600, 2026112323)
BUDGETS = (0, 35, 60, 90, 120)
ORIGINAL_TARGET = 1650
# Invented space and qualifications used only with this benchmark.
LOCATIONS = {
    "none": (None, (), 4, 0),
    "converted_bay": ("assembly", (), 2, 35),
    "added_bay": ("assembly", (), 4, 55),
    "general_bay": ("general", (), 4, 20),
    "shared_inspection_bay": ("assembly", ("incoming_inspection",), 3, 25),
}
RESTRICTED_TASKS = {2, 6, 8}
INCOMPATIBLE_TASKS = ((8, 17), (2, 21))


@dataclass(frozen=True)
class Design:
    location: str = "none"
    carts: int = 3
    backup: bool = False
    earlier_kits: bool = False

    @property
    def cost(self):
        return LOCATIONS[self.location][3] + 12 * max(0, self.carts - 3) + 18 * self.backup + 25 * self.earlier_kits


@dataclass(frozen=True)
class Stress:
    qualified_availability: float = .82
    kit_delay: int = 0
    maintenance_minutes: int = 18


BASE = Design()
STRESSES = {
    "reference": Stress(),
    "operator_70pct": Stress(qualified_availability=.70),
    "operator_95pct": Stress(qualified_availability=.95),
    "kits_15min_late": Stress(kit_delay=15),
    "kits_30min_late": Stress(kit_delay=30),
    "maintenance_30min": Stress(maintenance_minutes=30),
    "combined": Stress(qualified_availability=.70, kit_delay=30, maintenance_minutes=30),
}


def violations(design, stations):
    zone, uses, cart_limit, _ = LOCATIONS[design.location]
    reasons = []
    station_four = set(stations[MAINTENANCE_STATION])
    if any(a in tasks and b in tasks for tasks in stations for a, b in INCOMPATIBLE_TASKS):
        reasons.append("incompatible_tasks_on_one_station")
    if not RESTRICTED_TASKS <= station_four:
        reasons.append("restricted_tasks_outside_station_four")
    if zone is not None and zone != "assembly" and RESTRICTED_TASKS & station_four:
        reasons.append("duplicate_outside_assembly_zone")
    if "incoming_inspection" in uses:
        reasons.append("assembly_and_incoming_inspection_share_bay")
    if design.carts > cart_limit:
        reasons.append("cart_bays_exceeded")
    if design.backup and zone is None:
        reasons.append("no_parallel_station_to_staff")
    return reasons


def run_one(scenario, draws, design, stress):
    quality_open = None
    if design.location != "none":
        quality_open = (draws[:, 0] < stress.qualified_availability)
        if design.backup:
            quality_open |= draws[:, 1] < .70
    gates, lots = receive_kits(scenario, ReceivingPlan(),
                              advance_minutes=(55 if design.earlier_kits else 30)-stress.kit_delay)
    scheduled = replace(scenario, kits=gates)
    plant = replace(PLANT, carts=design.carts, parking_bays=design.carts,
                    parking_length_cm=120*design.carts, kit_advance=0)
    starts = np.arange(SHIFTS)*SHIFT_SECONDS + 6000
    windows = np.column_stack((starts, starts+60*stress.maintenance_minutes))
    model = simulate(scheduled, plant, maintenance=windows, parallel_available=quality_open)
    assert np.all(model.trace[:, 0] >= gates)
    begin, end = model.maintenance_operations.T
    for offset, mask in ((0, ~model.parallel_used), (2400, model.parallel_used)):
        for left, right in windows + offset:
            assert not np.any(mask & (begin < right) & (end > left)), "bench worked during its own maintenance"
    if quality_open is not None:
        chosen = model.parallel_used
        days = (begin[chosen] // SHIFT_SECONDS).astype(int)
        assert quality_open[days].all() and np.all(end[chosen] <= (days+1)*SHIFT_SECONDS)
        for mask in (chosen, ~chosen):
            start = begin[mask]
            finish = end[mask]
            order = np.argsort(start)
            assert np.all(start[order][1:] >= finish[order][:-1])
    rows = evaluate(scheduled, model, lots, SHIPPING["faster_loading"])
    return [(r["shipped_a"], r["shipped_b"], r["shipped_on_time"],
             int(quality_open[d]) if quality_open is not None else 0,
             int(np.sum(model.parallel_used & (begin >= d*SHIFT_SECONDS) & (begin < (d+1)*SHIFT_SECONDS))))
            for d, r in enumerate(rows) if d in ADVERSE]


def collect(case, stations, stage, checks):
    history = {key: [] for key in checks}
    for seed in np.random.SeedSequence(stage[1]).spawn(stage[0]):
        work_seed, roster_seed = seed.spawn(2)
        scenario = sample(case, stations, work_seed)
        draws = np.random.default_rng(roster_seed).random((SHIFTS, 2))
        for key, (design, stress) in checks.items():
            history[key].append(run_one(scenario, draws, design, stress))
    return {key: np.asarray(rows) for key, rows in history.items()}


def deliverable(data):
    return np.minimum(data[:, :, 0]*10//7, data[:, :, 1]*10//3).min(axis=1)


def design_promise(data):
    # Reserve a further twenty units below the design lower-tail order statistic.
    ordered = np.sort(deliverable(data))
    return int(max(0, min(ORIGINAL_TARGET, (ordered[int(.015*len(ordered))]-20)//10*10)))


def certify(data, target, alpha):
    joint = deliverable(data)
    hits = int((joint >= target).sum())
    lower = float(beta.ppf(alpha, hits, len(joint)-hits+1)) if hits else 0.
    return {"target": target, "a": target*7//10, "b": target*3//10, "hits": hits,
            "n": len(joint), "lower": round(lower, 6), "accepted": lower >= .95,
            "sample_rate": round(hits/len(joint), 6),
            "mean_shipped": data[:, :, 2].mean(axis=0).round(3).tolist(),
            "availability": data[:, :, 3].mean(axis=0).round(4).tolist(),
            "mean_parallel_operations": data[:, :, 4].mean(axis=0).round(3).tolist()}


def policy_name(d):
    return f"{d.location}_c{d.carts}_backup{int(d.backup)}_earlier{int(d.earlier_kits)}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"results")
    args = parser.parse_args()
    case, stations = load_case()
    configs = [Design(*values) for values in product(LOCATIONS, (2, 3, 4), (False, True), (False, True))]
    rejected = {policy_name(d): violations(d, stations) for d in configs if violations(d, stations)}
    excluded = {policy_name(d): "budget" if d.cost > max(BUDGETS) else "unnecessary_cart_removal"
                for d in configs if policy_name(d) not in rejected
                and (d.cost > max(BUDGETS) or (d.carts < 3 and d.location != "converted_bay"))}
    eligible = [d for d in configs if policy_name(d) not in rejected and policy_name(d) not in excluded]
    options = {policy_name(d): d for d in eligible}
    design = collect(case, stations, DESIGN, {k: (d, Stress()) for k, d in options.items()})
    promises = {k: design_promise(data) for k, data in design.items()}

    def score(k):
        return promises[k], -options[k].cost, float(design[k][:, :, 2].mean())

    winners = {b: max((k for k,d in options.items() if d.cost <= b), key=score) for b in BUDGETS}
    parallel = max((k for k,d in options.items() if d.location != "none"), key=score)
    anchors = list(dict.fromkeys((winners[max(BUDGETS)], winners[60], parallel)))
    frozen = list(dict.fromkeys((policy_name(BASE), *winners.values(), *anchors)))
    confirm = collect(case, stations, HOLDOUT, {k:(options[k], Stress()) for k in frozen})
    alpha = .05/(2*len(frozen))
    validation = {k: {"promise": certify(v, promises[k], alpha), "original": certify(v, ORIGINAL_TARGET, alpha)}
                  for k,v in confirm.items()}
    stress_checks = {(k, name):(options[k], spec) for k in anchors for name,spec in STRESSES.items()}
    sensitive = collect(case, stations, SENSITIVITY, stress_checks)
    stress_alpha = .05/len(sensitive)
    report = {
        "benchmark": case.name, "stations": stations, "budgets": BUDGETS,
        "design": DESIGN, "holdout": HOLDOUT, "sensitivity": SENSITIVITY,
        "screened_count": len(configs), "infeasible_layout": rejected, "excluded_before_design": excluded,
        "feasible": {k: dict(config=asdict(d), cost=d.cost, frozen_promise=promises[k]) for k,d in options.items()},
        "winners": winners, "best_parallel": parallel, "validation": validation, "alpha": alpha,
        "stress_alpha": stress_alpha, "stress_settings": {k:asdict(v) for k,v in STRESSES.items()},
        "stress_results": {k:{name:certify(sensitive[k,name], promises[k], stress_alpha) for name in STRESSES}
                           for k in anchors},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/"factory_capacity_plan.json").write_text(json.dumps(report,indent=2)+"\n")
    with (args.output_dir/"factory_capacity_plan.csv").open("w",newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(("stage","policy","scenario","trajectory","shift","a","b","shipped","parallel_staffed","parallel_jobs"))
        for stage,data in (("holdout",{(k,"reference"):v for k,v in confirm.items()}),("sensitivity",sensitive)):
            for (key, scenario), values in data.items():
                for i, history in enumerate(values,1):
                    for day, row in zip(ADVERSE, history, strict=True):
                        writer.writerow((stage,key,scenario,i,day+1,*row))


if __name__ == "__main__":
    main()

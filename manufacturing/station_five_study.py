"""Independent 50:50 ready-policy replay for a proposed station-five motion reduction."""

import csv
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
from scipy.stats import beta

from factory_floor_study import ACTIONS_LIMIT, POSTURE_LIMIT
from factory_inputs import ADVERSE, load_case
from mixed_line_decisions import trial
from mixed_line_stress import PRESSURES
from order_mix_study import CONDITIONS, MIXES, RESOURCE, quotas

SEED = 2026112801
RUNS = 400
TARGET = 525


def measure(case, plan, seed):
    rows = []
    for condition in CONDITIONS:
        matched = np.random.SeedSequence(seed.entropy, spawn_key=seed.spawn_key)
        shifts = trial(case, plan, matched, pressure=PRESSURES[condition], policies=(RESOURCE,),
                       release_rule="ready", sku_counts=MIXES["50_50"], load_detail=True)[RESOURCE.name]
        a, b = quotas(TARGET, MIXES["50_50"])
        joint_delivery = all(shifts[i]["a"] >= a and shifts[i]["b"] >= b for i in ADVERSE)
        old, new, max_posture = [], [], 0.
        for i in ADVERSE:
            shift = shifts[i]
            movements = np.asarray(shift["action_totals"]).copy()
            old.append(shift["peak_actions_per_min"])
            movements[4] -= shift["station_five_b_units"]
            new.append(float(movements.max()/420))
            max_posture = max(max_posture, shift["station_five_posture_min"])
        rows.append(dict(trajectory=seed.spawn_key[-1]+1, pressure=condition, delivered=int(joint_delivery),
                         old_peak=max(old), new_peak=max(new), max_posture=max_posture))
    return rows


def summarize(rows):
    result = {}
    for condition in CONDITIONS:
        subset = [r for r in rows if r["pressure"] == condition]
        delivered = sum(r["delivered"] for r in subset)
        # One-sided familywise 95% lower bounds for the two delivery checks.
        lower = beta.ppf(.05/len(CONDITIONS), delivered, RUNS-delivered+1) if delivered else 0
        result[condition] = dict(delivered=delivered, delivery_lower=float(lower))
        for field in ("old_peak", "new_peak"):
            result[condition][field] = dict(
                breaches=sum(r[field] > ACTIONS_LIMIT or r["max_posture"] > POSTURE_LIMIT for r in subset),
                maximum=max(r[field] for r in subset))
    return result


if __name__ == "__main__":
    results = Path(__file__).resolve().parent / "results"
    case, plan = load_case()
    seeds = np.random.SeedSequence(SEED).spawn(RUNS)
    with ProcessPoolExecutor(max_workers=3) as pool:
        rows = [r for pair in pool.map(partial(measure, case, plan), seeds, chunksize=5) for r in pair]
    with (results / "station_five_study.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(seed=SEED, trials_per_condition=RUNS, target=TARGET, results=summarize(rows))
    (results / "station_five_study.json").write_text(json.dumps(summary, indent=2) + "\n")

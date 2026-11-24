"""Replay failed ergonomic checks and matched controls from the frozen release holdout."""

import csv
import gzip
import json
from pathlib import Path

import numpy as np

from factory_floor_study import ACTIONS_LIMIT, POSTURE_LIMIT
from factory_inputs import ADVERSE, load_case
from mixed_line_decisions import trial
from mixed_line_stress import PRESSURES
from order_mix_study import RESOURCE


def diagnose(directory):
    meta = json.loads((directory / "constrained_release.json").read_text())
    policies = {item["name"]: item for item in meta["candidates"]}
    count, root_seed = meta["holdout"]
    seeds = np.random.SeedSequence(root_seed).spawn(count)
    case, plan = load_case()
    controls = set()
    records = []
    with gzip.open(directory / "constrained_release.csv.gz", "rt") as stream:
        for row in csv.DictReader(stream):
            if row["stage"] != "holdout" or row["mix"] != "50_50":
                continue
            condition, name = row["pressure"], row["policy"]
            control = row["ergo_pass"] == "1"
            if control and (condition, name) in controls:
                continue
            if control:
                controls.add((condition, name))
            index = int(row["trajectory"])
            seed = seeds[index - 1]
            matched = np.random.SeedSequence(seed.entropy, spawn_key=seed.spawn_key)
            policy = policies[name]
            shifts = trial(case, plan, matched, pressure=PRESSURES[condition], policies=(RESOURCE,),
                           release_rule=policy["rule"], batch_size=policy["batch"], measure_release=True,
                           sku_counts=tuple(meta["mixes"][row["mix"]]), order_mix_band=policy["band"],
                           load_detail=True)[RESOURCE.name]
            passed = True
            for day, suffix in zip(ADVERSE, ("first", "second"), strict=True):
                shift = shifts[day]
                if any(shift[sku] != int(row[f"{sku}_{suffix}"]) for sku in ("a", "b")):
                    raise ValueError(f"Shipment replay mismatch: {row}")
                actions, posture = shift["peak_actions_per_min"], shift["station_five_posture_min"]
                passed &= actions <= ACTIONS_LIMIT and posture <= POSTURE_LIMIT
                records.append(dict(trajectory=index, pressure=condition, policy=name, shift=day+1,
                                    sample="control" if control else "flagged", actions_per_min=actions,
                                    peak_station=int(np.argmax(shift["action_totals"]))+1,
                                    posture_minutes=posture, actions_over=max(0, actions-ACTIONS_LIMIT),
                                    posture_over=max(0, posture-POSTURE_LIMIT)))
            if passed != control:
                raise ValueError(f"Load replay mismatch: {row}")
    return records


if __name__ == "__main__":
    results = Path(__file__).resolve().parent / "results"
    rows = diagnose(results)
    with (results / "release_load_diagnostics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

"""Check how a fixed release ratio transfers to different order mixes."""

import argparse
import csv
import gzip
import hashlib
import io
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from decision_economics import paired
from factory_capacity_plan import ORIGINAL_TARGET
from factory_inputs import ADVERSE, SKU_COUNTS, load_case
from line_balance import ROOT
from mixed_line_decisions import POLICIES, trial, verify_target
from mixed_line_stress import PRESSURES

HOLDOUT = (800, 2026112601)
MIXES = {"70_30": SKU_COUNTS, "50_50": (925, 925), "80_20": (1480, 370)}
CONDITIONS = ("reference", "combined")
RULES = ("ready", "balanced_batch", "balanced_order_mix")
RESOURCE = next(policy for policy in POLICIES if policy.name == "assembly+rework")
BATCH = 120
TARGETS = (330, 420, 500, 600, 650)
FIELDS = ("a_first", "b_first", "a_second", "b_second",
          "short_a", "short_b", "longest_delay_min", "changeovers", "ergo_pass")
EFFECTS = ("short_a", "short_b", "longest_delay_min", "changeovers")


def quotas(target, counts):
    a = target * counts[0] // sum(counts)
    return a, target - a


def trajectory_metrics(case, plan, seed, counts, condition, rule, batch, band=0):
    nominal_a, nominal_b = quotas(ORIGINAL_TARGET, counts)
    matched = np.random.SeedSequence(seed.entropy, spawn_key=seed.spawn_key)
    rows = trial(
        case, plan, matched, pressure=PRESSURES[condition], policies=(RESOURCE,),
        release_rule=rule, batch_size=batch, measure_release=True, sku_counts=counts, order_mix_band=band,
    )[RESOURCE.name]
    return (
        *(rows[i][sku] for i in ADVERSE for sku in ("a", "b")),
        sum(max(0, nominal_a - row["a"]) for row in rows),
        sum(max(0, nominal_b - row["b"]) for row in rows),
        max(row["late_minutes"] for row in rows),
        sum(row["entry_changeovers"] for row in rows),
        int(all(rows[i]["ergo_pass"] for i in ADVERSE)),
    )


def evaluate(case, plan, seed):
    return [
        trajectory_metrics(case, plan, seed, counts, condition, rule, BATCH)
        for counts in MIXES.values() for condition in CONDITIONS for rule in RULES
    ]


def report(panel, out):
    if not np.array_equal(panel[:, 0, :, 1], panel[:, 0, :, 2]):
        raise ValueError("fixed and order-matched rules must agree when the mix is 70:30")

    target_family = len(MIXES) * len(CONDITIONS) * len(RULES) * len(TARGETS)
    effect_family = len(MIXES) * len(CONDITIONS) * 2 * len(EFFECTS)
    evidence = io.StringIO(newline="")
    writer = csv.writer(evidence, lineterminator="\n")
    writer.writerow(("trajectory", "mix", "pressure", "rule", *FIELDS))
    outcomes = {}
    for i, (mix, counts) in enumerate(MIXES.items()):
        outcomes[mix] = {}
        for j, condition in enumerate(CONDITIONS):
            series = panel[:, i, j]
            outcomes[mix][condition] = scenario_report(series, counts, target_family, effect_family)
            for k, rule in enumerate(RULES):
                for n, row in enumerate(series[:, k], 1):
                    features = (*map(int, row[:6]), round(float(row[6]), 6), *map(int, row[7:]))
                    writer.writerow((n, mix, condition, rule, *features))

    raw = evidence.getvalue().encode()
    out.mkdir(parents=True, exist_ok=True)
    with (out / "order_mix.csv.gz").open("wb") as target:
        with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0) as archive:
            archive.write(raw)
    result = dict(
        holdout=HOLDOUT, order_counts=MIXES, resources=RESOURCE.name, minimum_batch=BATCH,
        shortage_reference_total=ORIGINAL_TARGET, common_commitment_targets=TARGETS,
        reliability_family=target_family, paired_effect_family=effect_family,
        uncompressed_evidence_sha256=hashlib.sha256(raw).hexdigest(), outcomes=outcomes,
    )
    (out / "order_mix.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def scenario_report(series, counts, target_family, effect_family):
    outcome = {}
    for k, rule in enumerate(RULES):
        rows = series[:, k]
        counts_in_adverse = rows[:, :4].reshape(-1, len(ADVERSE), 2)
        ergo_passed = bool(rows[:, -1].all())
        reliability = {
            target: verify_target(counts_in_adverse, target, .05 / target_family, quotas=quotas(target, counts))
            for target in TARGETS
        }
        outcome[rule] = dict(
            means={field: round(float(rows[:, col].mean()), 5) for col, field in enumerate(FIELDS)},
            reliability=reliability,
            largest_passing_target=max((t for t, r in reliability.items()
                                        if ergo_passed and r["passed"]), default=None),
        )
    matched = series[:, RULES.index("balanced_order_mix")]
    outcome["paired_order_mix_vs"] = {
        other: {
            field: paired(matched[:, col] - series[:, k, col], effect_family)
            for col, field in enumerate(FIELDS) if field in EFFECTS
        }
        for k, other in enumerate(RULES[:-1])
    }
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    case, plan = load_case()
    seeds = np.random.SeedSequence(HOLDOUT[1]).spawn(HOLDOUT[0])
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(partial(evaluate, case, plan), seeds, chunksize=4))
    panel = np.asarray(rows).reshape(HOLDOUT[0], len(MIXES), len(CONDITIONS), len(RULES), len(FIELDS))
    report(panel, args.output_dir)


if __name__ == "__main__":
    main()

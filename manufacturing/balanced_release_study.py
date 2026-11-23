"""Tune SKU-aware batches, then validate service and old-order delay with resources fixed."""

import argparse
import csv
import gzip
import hashlib
import io
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from itertools import product
from pathlib import Path

import numpy as np

from decision_economics import paired
from factory_capacity_plan import design_promise
from factory_inputs import ADVERSE, load_case
from line_balance import ROOT
from mixed_line_decisions import POLICIES, trial, verify_target
from mixed_line_stress import PRESSURES

DESIGN = (144, 2026112551)
HOLDOUT = (800, 2026112552)
BATCHES = (30, 60, 90, 120, 180)
TARGETS = (330, 400, 500, 640, 650)
TUNING_PRESSURES = ("reference", "combined")
RESOURCES = ("base", "assembly+rework")
DELAY_TOLERANCE_MIN = 10.0
FIELDS = ("a_first", "b_first", "a_second", "b_second",
          "short_a", "short_b", "longest_delay_min", "final_open", "entry_changeovers", "ergo_pass")


@dataclass(frozen=True)
class Rule:
    name: str
    batch: int = 80

    @property
    def label(self):
        return self.name if self.name == "ready" else f"{self.name}_{self.batch}"


BASE = Rule("ready")


def evaluate(case, plan, rules, resources, conditions, seed):
    configs = tuple(p for p in POLICIES if p.name in resources)
    panel = []
    for condition in conditions:
        variants = []
        for rule in rules:
            identical = np.random.SeedSequence(seed.entropy, spawn_key=seed.spawn_key)
            results = trial(case, plan, identical, pressure=PRESSURES[condition], policies=configs,
                            release_rule=rule.name, batch_size=rule.batch, measure_release=True)
            row = []
            for resource in resources:
                cohort = results[resource]
                counts = [cohort[day][sku] for day in ADVERSE for sku in ("a", "b")]
                row.append((*counts,
                            sum(s["short_a"] for s in cohort), sum(s["short_b"] for s in cohort),
                            max(s["late_minutes"] for s in cohort), cohort[-1]["closing_open"],
                            sum(s["entry_changeovers"] for s in cohort),
                            int(all(cohort[day]["ergo_pass"] for day in ADVERSE))))
            variants.append(row)
        panel.append(variants)
    return panel


def collect(case, plan, rules, stage, conditions, workers, resources=RESOURCES):
    replay = partial(evaluate, case, plan, rules, resources, conditions)
    seeds = np.random.SeedSequence(stage[1]).spawn(stage[0])
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return np.asarray(list(pool.map(replay, seeds, chunksize=4)))


def locked_target(values):
    return min(design_promise(series[:, :4].reshape(-1, 2, 2))
               for series in values.transpose(1, 0, 2))


def choose(panel, rules):
    """Choose from the design cohort, giving the service guards priority."""
    base = panel[:, :, 0]
    selected = {}
    for resource_idx, resource in enumerate(RESOURCES):
        reference = base[:, :, resource_idx]
        for name in ("campaign", "balanced_batch"):
            scores = []
            for policy_idx, rule in enumerate(rules):
                if rule.name != name:
                    continue
                sample = panel[:, :, policy_idx, resource_idx]
                deltas = (sample-reference).mean(axis=0)
                guards = (np.all(deltas[:, 4:6] <= 0)
                          and np.all(deltas[:, 6] <= DELAY_TOLERANCE_MIN)
                          and np.all(sample[:, :, 9]))
                average_shortage = sample[:, :, 4:6].sum(axis=2).mean()
                scores.append(((int(guards), locked_target(sample), -average_shortage, -rule.batch), rule))
            selected[resource, name] = max(scores, key=lambda item:item[0])[1]
    return selected


def report(out, panel, rules, promises):
    outcome = {}
    conditions = tuple(PRESSURES)
    target_alpha = .05/(len(conditions)*len(promises)*(len(TARGETS)+1))
    # A/B deficits, oldest observed delay, terminal stock, changes and total deficit.
    comparison_family = len(conditions)*(len(promises)-len(RESOURCES))*6
    evidence = io.StringIO(newline="")
    writer = csv.writer(evidence, lineterminator="\n")
    writer.writerow(("trajectory", "pressure", "resources", "rule", *FIELDS))
    for i, resource in enumerate(RESOURCES):
        outcome[resource] = {}
        for j, rule in enumerate(rules):
            if (resource, rule.label) not in promises:
                continue
            target = promises[resource, rule.label]
            scenarios = {}
            for k, condition in enumerate(conditions):
                series = panel[:, k, j, i]
                delta = series-panel[:, k, 0, i]
                targets = sorted(set((*TARGETS, target)))
                effects = {key:paired(delta[:, col], comparison_family) for col,key in
                           ((4,"short_a"), (5,"short_b"), (6,"longest_delay_min"),
                            (7,"final_open"), (8,"entry_changeovers"))} if j else None
                if effects is not None:
                    effects["total_shortfall"] = paired(delta[:, 4]+delta[:, 5], comparison_family)
                guard = (effects is None or (effects["short_a"]["simultaneous95"][1] <= 0
                         and effects["short_b"]["simultaneous95"][1] <= 0
                         and effects["longest_delay_min"]["simultaneous95"][1] <= DELAY_TOLERANCE_MIN))
                scenarios[condition] = dict(
                    means={f:round(float(series[:,col].mean()),5) for col,f in enumerate(FIELDS)},
                    effects_vs_ready=effects, protects_old_orders_and_both_skus=guard,
                    reliability={q:verify_target(series[:, :4].reshape(-1, 2, 2), q, target_alpha) for q in targets},
                )
                for n, values in enumerate(series, 1):
                    writer.writerow((n, condition, resource, rule.label, *map(int, values[:6]),
                                     round(float(values[6]),6), *map(int, values[7:])))
            outcome[resource][rule.label] = dict(locked_target=target, scenarios=scenarios,
                 guardrails_passed=all(s["protects_old_orders_and_both_skus"] and s["means"]["ergo_pass"]==1.
                                       for s in scenarios.values()),
                 verified_all_pressures=[q for q in targets
                                         if all(s["reliability"][q]["passed"] for s in scenarios.values())])
    out.mkdir(parents=True, exist_ok=True)
    raw = evidence.getvalue().encode()
    with (out/"balanced_release.csv.gz").open("wb") as blob:
        with gzip.GzipFile(filename="", mode="wb", fileobj=blob, mtime=0) as archive:
            archive.write(raw)
    result = dict(design=DESIGN, holdout=HOLDOUT, tuning_pressures=TUNING_PRESSURES,
                  tested_batch_sizes=BATCHES, delay_tolerance_minutes=DELAY_TOLERANCE_MIN,
                  reliability_alpha=target_alpha, pairwise_effect_family=comparison_family,
                  uncompressed_evidence_sha256=hashlib.sha256(raw).hexdigest(), outcomes=outcome)
    (out/"balanced_release.json").write_text(json.dumps(result, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"results")
    args = parser.parse_args()
    case, plan = load_case()
    rules = [BASE, *(Rule(method, batch) for method,batch in product(("campaign","balanced_batch"), BATCHES))]
    design = collect(case, plan, rules, DESIGN, TUNING_PRESSURES, args.workers)
    selected = choose(design, rules)
    promises = {}
    for i, resource in enumerate(RESOURCES):
        finalists = [BASE, selected[resource, "campaign"], selected[resource, "balanced_batch"]]
        for rule in finalists:
            promises[resource, rule.label] = locked_target(design[:, :, rules.index(rule), i])
    print("Locked before holdout:", promises, flush=True)
    # Simulate only each resource's locked designs; share the seed list across both resources.
    final_rules = [r for r in rules if any((resource, r.label) in promises for resource in RESOURCES)]
    # Exclude rules in the other resource group before simulating, then align the results.
    panel = np.zeros((HOLDOUT[0], len(PRESSURES), len(final_rules), len(RESOURCES), len(FIELDS)))
    for i, resource in enumerate(RESOURCES):
        finalists = [r for r in final_rules if (resource,r.label) in promises]
        samples = collect(case, plan, finalists, HOLDOUT, tuple(PRESSURES), args.workers, resources=(resource,))
        for j, rule in enumerate(finalists):
            panel[:, :, final_rules.index(rule), i] = samples[:, :, j, 0]
    report(args.output_dir, panel, final_rules, promises)


if __name__ == "__main__":
    main()

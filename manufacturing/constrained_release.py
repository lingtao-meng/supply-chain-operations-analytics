"""Select a release policy on design paths and verify its service and operating limits."""

import argparse
import csv
import gzip
import hashlib
import io
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np

from decision_economics import paired
from factory_capacity_plan import ORIGINAL_TARGET
from factory_inputs import ADVERSE, SHIFTS, load_case
from line_balance import ROOT
from mixed_line_decisions import verify_target
from order_mix_study import CONDITIONS, FIELDS, MIXES, RESOURCE, quotas, trajectory_metrics

DESIGN = (200, 2026112701)
HOLDOUT = (800, 2026112702)
TARGETS = (330, 420, *range(450, 701, 25))
DESIGN_PASS_RATE = .975
MAX_SHORTAGE_INCREASE = .01
MAX_DELAY_INCREASE = 10
MAX_CHANGEOVER_INCREASE = 2
EFFECT_FIELDS = ("short_a", "short_b", "longest_delay_min", "changeovers")


@dataclass(frozen=True)
class Policy:
    name: str
    rule: str
    batch: int = 120
    band: int = 0


READY = Policy("ready", "ready")
CANDIDATES = (
    READY,
    *(Policy(f"campaign_{n}", "campaign", n) for n in (60, 120)),
    *(Policy(f"fixed_{n}", "balanced_batch", n) for n in (60, 120)),
    *(Policy(f"order_{n}", "balanced_order_mix", n) for n in (60, 120)),
    *(Policy(f"band_{n}_{b}", "balanced_order_mix", n, b) for n, b in ((60, 20), (120, 20), (120, 50))),
)


def evaluate(case, plan, tasks, seed):
    return [
        trajectory_metrics(case, plan, seed, counts, condition, policy.rule, policy.batch, policy.band)
        for _, counts, condition, policy in tasks
    ]


def collect(case, plan, schedule, stage, workers):
    tasks = [
        (mix, MIXES[mix], condition, rule)
        for mix, rules in schedule.items() for condition in CONDITIONS for rule in rules
    ]
    seeds = np.random.SeedSequence(stage[1]).spawn(stage[0])
    replay = partial(evaluate, case, plan, tasks)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        draws = np.asarray(list(pool.map(replay, seeds, chunksize=4)))
    results = {mix: {condition: {} for condition in CONDITIONS} for mix in schedule}
    for col, (mix, _, condition, rule) in enumerate(tasks):
        results[mix][condition][rule.name] = draws[:, col]
    return results


def tolerances(counts):
    a, b = quotas(ORIGINAL_TARGET, counts)
    return dict(zip(EFFECT_FIELDS, (a*SHIFTS*MAX_SHORTAGE_INCREASE, b*SHIFTS*MAX_SHORTAGE_INCREASE,
                                     MAX_DELAY_INCREASE, MAX_CHANGEOVER_INCREASE), strict=True))


def assessment(panel, schedule, *, effect_family=None, target_family=None):
    outcome = {}
    for mix, counts in MIXES.items():
        limits = tolerances(counts)
        outcome[mix] = {}
        for policy in schedule[mix]:
            scenarios = {}
            for condition in CONDITIONS:
                series = panel[mix][condition][policy.name]
                delta = series[:, 4:8] - panel[mix][condition][READY.name][:, 4:8]
                effects, upper = effect_limits(delta, effect_family)
                violations = [key for key in EFFECT_FIELDS if upper[key] > limits[key]]
                ergo_breaches = int((series[:, -1] == 0).sum())
                if ergo_breaches:
                    violations.append("ergonomics")
                reliability = delivery_targets(series[:, :4].reshape(-1, len(ADVERSE), 2), counts, target_family)
                scenarios[condition] = dict(
                    effect_vs_ready=effects, breached_limits=violations,
                    ergo_trajectories_over_limit=ergo_breaches, reliability=reliability,
                    mean_combined_shortage=round(float(series[:, 4:6].sum(axis=1).mean()), 5),
                )
            best_service = max((target for target in TARGETS if all(
                result["reliability"][target]["passed"] for result in scenarios.values()
            )), default=None)
            acceptable = all(not result["breached_limits"] for result in scenarios.values())
            outcome[mix][policy.name] = dict(
                best_service=best_service, safeguards_pass=acceptable,
                best_acceptable_target=best_service if acceptable else None, scenarios=scenarios,
            )
    return outcome


def delivery_targets(adverse, counts, target_family):
    result = {}
    for target in TARGETS:
        sku_quotas = quotas(target, counts)
        if target_family is None:
            hits = int(np.all(adverse >= sku_quotas, axis=(1, 2)).sum())
            result[target] = dict(target=target, quotas=sku_quotas, hits=hits, passed=hits/len(adverse) >= DESIGN_PASS_RATE)
        else:
            result[target] = verify_target(adverse, target, .05/target_family, quotas=sku_quotas)
    return result


def effect_limits(delta, family):
    effects, upper = {}, {}
    for field, observations in zip(EFFECT_FIELDS, delta.T, strict=True):
        stats = (dict(mean=round(float(observations.mean()), 5)) if family is None
                 else paired(observations, family))
        effects[field] = stats
        upper[field] = stats["mean"] if family is None else stats["simultaneous95"][1]
    return effects, upper


def service_rank(result):
    return (result["best_service"] or 0, -sum(s["mean_combined_shortage"] for s in result["scenarios"].values()))


def choose(design):
    schedule, nominees = {}, {}
    for mix, outcomes in design.items():
        service = max(CANDIDATES, key=lambda policy: service_rank(outcomes[policy.name]))
        eligible = [policy for policy in CANDIDATES[1:] if outcomes[policy.name]["best_acceptable_target"] is not None]
        if eligible:
            protected = max(eligible, key=lambda policy: service_rank(outcomes[policy.name]))
        else:
            # Keep the closest challenger to show the constraint that blocks it on independent paths.
            def score(policy):
                row = outcomes[policy.name]
                failures = sum(len(s["breached_limits"]) for s in row["scenarios"].values())
                return (-failures, *service_rank(row))
            protected = max(CANDIDATES[1:], key=score)
        nominees[mix] = dict(best_design_service=service.name, design_challenger=protected.name,
                             challenger_passed_design=bool(eligible))
        schedule[mix] = tuple(dict.fromkeys((READY, service, protected)))
    return schedule, nominees


def write(out, design_panel, holdout_panel, nominees, design, holdout, families):
    evidence = io.StringIO(newline="")
    writer = csv.writer(evidence, lineterminator="\n")
    writer.writerow(("stage", "trajectory", "mix", "pressure", "policy", *FIELDS))
    for stage, panel in (("design", design_panel), ("holdout", holdout_panel)):
        for mix, conditions in panel.items():
            for condition, policies in conditions.items():
                for name, series in policies.items():
                    for n, row in enumerate(series, 1):
                        writer.writerow((stage, n, mix, condition, name, *map(int, row[:6]),
                                         round(float(row[6]), 6), *map(int, row[7:])))
    raw = evidence.getvalue().encode()
    out.mkdir(parents=True, exist_ok=True)
    with (out/"constrained_release.csv.gz").open("wb") as target:
        with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0) as archive:
            archive.write(raw)
    summary = dict(
        design=DESIGN, holdout=HOLDOUT, mixes=MIXES, resource=RESOURCE.name, targets=TARGETS,
        design_joint_pass_rate=DESIGN_PASS_RATE, tolerances={mix:tolerances(counts) for mix, counts in MIXES.items()},
        candidates=[vars(policy) for policy in CANDIDATES], nominees=nominees, holdout_families=families,
        design_assessment=design, holdout_assessment=holdout,
        uncompressed_evidence_sha256=hashlib.sha256(raw).hexdigest(),
    )
    (out/"constrained_release.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"results")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    case, plan = load_case()
    design_setup = dict.fromkeys(MIXES, CANDIDATES)
    design_panel = collect(case, plan, design_setup, DESIGN, args.workers)
    design = assessment(design_panel, design_setup)
    holdout_setup, nominees = choose(design)
    print(f"Locked before holdout: {nominees}", flush=True)
    tests = sum(map(len, holdout_setup.values()))*len(CONDITIONS)
    comparisons = tests - len(CONDITIONS)*len(holdout_setup)
    families = dict(effect_family=comparisons*len(EFFECT_FIELDS), target_family=tests*len(TARGETS))
    holdout_panel = collect(case, plan, holdout_setup, HOLDOUT, args.workers)
    holdout = assessment(holdout_panel, holdout_setup, **families)
    write(args.output_dir, design_panel, holdout_panel, nominees, design, holdout, families)


if __name__ == "__main__":
    main()

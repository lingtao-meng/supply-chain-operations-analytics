"""Choose inspection and rework waiting space when congestion can block assembly."""

from __future__ import annotations

import argparse
import csv
import json
from math import ceil
from pathlib import Path

import numpy as np
from scipy.stats import beta

from downtime_recovery import SHIFT_CV, TASK_CV, minimum_hits
from finite_staging_kernel import SHIFT_FIELDS, simulate_finite_shift
from good_output import BATCH_SIZE, FAST_FACTOR, FIRST_PASS, STRESS_DEFECT_PCT, paired_gain, quality_routes
from line_balance import ROOT, Case, reference_plans, verify
from optimize_buffers import scenarios, workloads_for
from rework_support import FAILURE_SHARE_PCT, PULSE_MINUTES, concentrate_failures
from shift_capacity import DEMAND, REQUIRED_RELIABILITY, SHIFT_SECONDS

AREA_BUDGETS = (0, 4, 8, 12, 24, 48)
FOCUS_BUDGET = 12
CREWS = {"long": (1, 180, 240), "peak": (2, 300, 120)}
DESIGN_PULSES = {"early": 35, "middle": 145, "late": 260, "latest": 290}
HOLDOUT_PULSES = {"uniform": None, "early": 50, "middle": 165, "late": 275, "latest": 300}
DESIGN_RUNS, HOLDOUT_RUNS = 300, 600
DESIGN_SEEDS, HOLDOUT_SEEDS = (2026112226, 2026112227), (2026112228, 2026112229)
EARLY_DESIGN_PULSES = {"d035": 35, "d065": 65, "d095": 95}
EARLY_HOLDOUT_PULSES = {"e020": 20, "e050": 50, "e080": 80}
EARLY_DESIGN_SEEDS, EARLY_HOLDOUT_SEEDS = (2026112230, 2026112231), (2026112232, 2026112233)
# The timing of defective lots is assigned by unit sequence, before congestion.
REFERENCE_UNITS = 1900
REFERENCE_TIMES = np.arange(REFERENCE_UNITS) * SHIFT_SECONDS / REFERENCE_UNITS
OUTCOME = SHIFT_FIELDS.index("released")

Layout = tuple[int, int]
Setting = tuple[str, Layout | None]


def layouts_for_budget(budget: int) -> tuple[Layout, ...]:
    choices = range(budget + 1) if budget == FOCUS_BUDGET else sorted({
        0, budget // 4, budget // 2, 3 * budget // 4, budget,
    })
    return tuple((inspection, budget - inspection) for inspection in choices)


def samples(case: Case, plan: list[list[int]], runs: int, seeds: tuple[int, int], pulses):
    worst_nominal_load = max(verify(case, 15, plan)) * FAST_FACTOR
    jobs = 2 * ceil(SHIFT_SECONDS / worst_nominal_load)
    time_batches = np.random.SeedSequence(seeds[0]).spawn(ceil(runs / BATCH_SIZE))
    quality = np.random.default_rng(seeds[1])
    for batch_index, offset in enumerate(range(0, runs, BATCH_SIZE)):
        size = min(BATCH_SIZE, runs - offset)
        time_seed = int(time_batches[batch_index].generate_state(1, dtype=np.uint64)[0])
        _, time = next(scenarios(case, ((TASK_CV, SHIFT_CV),), jobs, size, time_seed))
        station_times = np.ascontiguousarray(workloads_for(time, plan).transpose(2, 0, 1) * FAST_FACTOR)
        draws = quality.random((size, jobs, 3))
        for work, row in zip(station_times, draws, strict=True):
            uniform = quality_routes(row, STRESS_DEFECT_PCT)
            cohort = uniform[:REFERENCE_UNITS]
            order = quality.permutation(REFERENCE_UNITS)
            failures = quality.permutation(cohort[cohort != FIRST_PASS])
            routes = {}
            for label, onset in pulses.items():
                if onset is None:
                    routes[label] = uniform
                    continue
                assigned = uniform.copy()
                assigned[:REFERENCE_UNITS] = concentrate_failures(cohort, REFERENCE_TIMES, onset, order, failures)
                routes[label] = assigned
            yield work, routes


def run_stage(case: Case, plan: list[list[int]], runs: int, seeds: tuple[int, int], pulses, settings: list[Setting],
              crews=CREWS):
    assembly_buffers = np.ones(len(plan) - 1, dtype=np.int64)
    results = {(setting, label): np.empty((runs, len(SHIFT_FIELDS))) for setting in settings for label in pulses}
    for rep, (work, route_by_scenario) in enumerate(samples(case, plan, runs, seeds, pulses)):
        for label, routes in route_by_scenario.items():
            for setting in settings:
                crew, layout = setting
                inspection_slots, rework_slots = layout or (len(work), len(work))
                people, start, duration = crews[crew]
                results[setting, label][rep] = simulate_finite_shift(
                    work, routes, assembly_buffers, inspection_slots, rework_slots, SHIFT_SECONDS,
                    people, start * 60, duration * 60,
                )
    return results


def design_score(design, setting: Setting, pulses=DESIGN_PULSES) -> tuple[int, int, int]:
    output = [design[setting, label][:, OUTCOME] for label in pulses]
    return (-min(int((v >= DEMAND).sum()) for v in output),
            max(int(np.maximum(DEMAND - v, 0).sum()) for v in output),
            -sum(int(v.sum()) for v in output))


def policy_name(setting: Setting) -> str:
    crew, layout = setting
    return f"{crew}_unlimited" if layout is None else f"{crew}_i{layout[0]}_r{layout[1]}"


def describe(setting: Setting) -> dict:
    crew, layout = setting
    return {"name": policy_name(setting), "crew": crew, "area_slots": sum(layout) if layout is not None else None,
            "inspection_slots": layout[0] if layout is not None else None,
            "rework_slots": layout[1] if layout is not None else None}


def describe_holdout(holdout, alpha: float) -> tuple[list[dict], list[dict]]:
    details, raw = [], []
    for (setting, label), values in holdout.items():
        policy = describe(setting)
        output = values[:, OUTCOME]
        hits = int((output >= DEMAND).sum())
        lower = float(beta.ppf(alpha, hits, HOLDOUT_RUNS - hits + 1)) if hits else 0.0
        details.append({
            **policy, "scenario": label, "hits": hits, "adjusted_lower": round(lower, 6),
            "passes": lower >= REQUIRED_RELIABILITY,
            "means": {name: round(float(values[:, pos].mean()), 3) for pos, name in enumerate(SHIFT_FIELDS)},
            "mean_shortfall": round(float(np.maximum(DEMAND - output, 0).mean()), 3),
        })
        raw.extend({"policy": policy["name"], "scenario": label, "replication": rep, **{
            name: round(float(row[pos]), 6) if name.endswith("_seconds") else int(row[pos])
            for pos, name in enumerate(SHIFT_FIELDS)
        }} for rep, row in enumerate(values, start=1))
    return details, raw


def test_preplanned_early_support(case: Case, plan: list[list[int]], layout: Layout) -> tuple[dict, list[dict]]:
    """If a risky lot is known to run early, choose its support window on design shifts."""
    crews = {f"early_{start:03d}": (2, start, 120) for start in range(0, 106, 15)}
    crews["peak"] = CREWS["peak"]
    options = [(crew, layout) for crew in crews]
    design = run_stage(case, plan, DESIGN_RUNS, EARLY_DESIGN_SEEDS, EARLY_DESIGN_PULSES, options, crews)
    scores = {option: design_score(design, option, EARLY_DESIGN_PULSES) for option in options}
    baseline = ("peak", layout)
    chosen = min(options, key=scores.__getitem__)
    selected = list(dict.fromkeys((baseline, chosen)))
    actual = run_stage(case, plan, HOLDOUT_RUNS, EARLY_HOLDOUT_SEEDS, EARLY_HOLDOUT_PULSES, selected, crews)
    alpha = 0.05 / len(actual)
    detail, rows = describe_holdout(actual, alpha)
    return {
        "known_condition": "The higher-risk lot is scheduled in the early part of the shift before dispatch.",
        "layout": layout, "helpers": 2, "minutes_per_helper": 120,
        "reference_start_minute": CREWS["peak"][1], "selected_start_minute": crews[chosen[0]][1],
        "reference": policy_name(baseline), "selected": policy_name(chosen),
        "design": {"runs": DESIGN_RUNS, "seeds": EARLY_DESIGN_SEEDS,
                   "pulses_on_reference_clock": EARLY_DESIGN_PULSES,
                   "options": [{"start_minute": crews[option[0]][1], "worst_hits": -scores[option][0],
                                "worst_shortfall_total": scores[option][1],
                                "released_total": -scores[option][2]} for option in options]},
        "holdout": {"runs": HOLDOUT_RUNS, "seeds": EARLY_HOLDOUT_SEEDS,
                    "pulses_on_reference_clock": EARLY_HOLDOUT_PULSES,
                    "alpha": alpha, "required_hits": minimum_hits(HOLDOUT_RUNS, alpha), "results": detail,
                    "paired_release_gain": {
                        label: paired_gain(actual[chosen, label][:, OUTCOME], actual[baseline, label][:, OUTCOME])
                        for label in EARLY_HOLDOUT_PULSES
                    }},
    }, rows


def run_study() -> tuple[dict, list[dict]]:
    records = json.loads((ROOT / "data" / "benchmarks.json").read_text(encoding="utf-8"))
    case = Case(**next(record for record in records if record["name"] == "Mitchell"))
    plan = reference_plans(case, 15)["MILP"]
    candidates = [(crew, layout) for crew in CREWS for budget in AREA_BUDGETS
                  for layout in layouts_for_budget(budget)]
    design = run_stage(case, plan, DESIGN_RUNS, DESIGN_SEEDS, DESIGN_PULSES, candidates)
    scores = {setting: design_score(design, setting) for setting in candidates}
    winners = {(crew, budget): min(((crew, lay) for lay in layouts_for_budget(budget)), key=scores.__getitem__)
               for crew in CREWS for budget in AREA_BUDGETS}
    controls = [(crew, lay) for crew in CREWS for lay in ((0, 12), (6, 6), (12, 0), None)]
    settings = list(dict.fromkeys([*winners.values(), *controls]))
    holdout = run_stage(case, plan, HOLDOUT_RUNS, HOLDOUT_SEEDS, HOLDOUT_PULSES, settings)
    alpha = 0.05 / (len(settings) * len(HOLDOUT_PULSES))
    details, raw = describe_holdout(holdout, alpha)
    comparisons = []
    for (setting, label), values in holdout.items():
        if setting[1] is not None:
            reference = holdout[(setting[0], None), label][:, OUTCOME]
            comparisons.append({"policy": policy_name(setting), "scenario": label,
                                "loss_vs_unlimited_staging": paired_gain(reference, values[:, OUTCOME])})
    layout_comparisons = []
    for crew in CREWS:
        picked = winners[crew, FOCUS_BUDGET]
        for label in HOLDOUT_PULSES:
            output = holdout[picked, label][:, OUTCOME]
            for layout in ((0, 12), (6, 6), (12, 0)):
                other = (crew, layout)
                reference = holdout[other, label][:, OUTCOME]
                layout_comparisons.append({"selected": policy_name(picked), "baseline": policy_name(other),
                                           "scenario": label, "paired_release_gain": paired_gain(output, reference)})
    early, timing_rows = test_preplanned_early_support(case, plan, winners["peak", FOCUS_BUDGET][1])
    return {
        "case": case.name, "assembly_tasks": plan, "assembly_buffer_slots": 7, "shift_seconds": SHIFT_SECONDS,
        "target_good_units": DEMAND, "defect_pct": STRESS_DEFECT_PCT, "reference_units": REFERENCE_UNITS,
        "pulse_minutes_on_reference_clock": PULSE_MINUTES, "pulse_failure_share_pct": FAILURE_SHARE_PCT,
        "task_cv_pct": TASK_CV, "shift_cv_pct": SHIFT_CV, "fast_time_factor": FAST_FACTOR,
        "crews": {crew: {"helpers": people, "start_minute": start, "minutes_per_helper": duration}
                  for crew, (people, start, duration) in CREWS.items()},
        "design": {"runs": DESIGN_RUNS, "seeds": DESIGN_SEEDS, "pulses_on_reference_clock": DESIGN_PULSES,
                   "candidates": [{**describe(setting), "worst_hits": -scores[setting][0],
                                   "worst_shortfall_total": scores[setting][1], "released_total": -scores[setting][2]}
                                  for setting in candidates],
                   "selected": [describe(setting) for setting in winners.values()]},
        "holdout": {"runs": HOLDOUT_RUNS, "seeds": HOLDOUT_SEEDS, "pulses_on_reference_clock": HOLDOUT_PULSES,
                    "alpha": alpha, "required_hits": minimum_hits(HOLDOUT_RUNS, alpha),
                    "results": details, "vs_unlimited": comparisons, "same_budget_layout": layout_comparisons},
        "early_risk_support": early,
    }, [*raw, *timing_rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    report, rows = run_study()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "finite_staging_holdout.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "finite_staging.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

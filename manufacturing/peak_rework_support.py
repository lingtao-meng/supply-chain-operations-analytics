"""Compare long and concurrent peak rework support with equal person-minutes."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from downtime_recovery import minimum_hits
from good_output import metric, paired_gain
from line_balance import ROOT
from rework_support import (
    DESIGN_RUNS, FAILURE_SHARE_PCT, HOLDOUT_RUNS, PULSE_MINUTES,
    Policy, assemble_inputs, design_score, evaluate, summarize_case,
)
from shift_capacity import DEMAND, REQUIRED_RELIABILITY, SHIFT_SECONDS

PERSON_MINUTES, START_INCREMENT = 240, 15
DESIGN_STARTS = {"early": 35, "middle": 145, "late": 260, "latest": 290}
HOLDOUT_STARTS = {"uniform": None, "early": 50, "middle": 165, "late": 275, "latest": 300}
DESIGN_SEEDS, HOLDOUT_SEEDS = (2026112221, 2026112222), (2026112223, 2026112224)


@dataclass(frozen=True)
class PeakPolicy(Policy):
    support_people: int = 2


def candidate_windows(people: int) -> tuple[PeakPolicy, ...]:
    minutes = PERSON_MINUTES // people
    return tuple(PeakPolicy(
        f"{people}_helper_{start:03d}", start_minute=start, support_minutes=minutes, support_people=people,
    ) for start in range(0, SHIFT_SECONDS // 60 - minutes + 1, START_INCREMENT))


def run_study() -> tuple[dict, list[dict]]:
    one = candidate_windows(1)
    two = candidate_windows(2)
    design = evaluate(assemble_inputs(DESIGN_RUNS, DESIGN_SEEDS, DESIGN_STARTS), (*one, *two))
    scores = {p.name: design_score(design, p.name, tuple(DESIGN_STARTS)) for p in (*one, *two)}
    chosen_one = replace(min(one, key=lambda p: scores[p.name]), name="one_long")
    chosen_two = replace(min(two, key=lambda p: scores[p.name]), name="two_peak")
    policies = (PeakPolicy("no_support", start_minute=0, support_minutes=0, support_people=0), chosen_one, chosen_two)
    holdout = evaluate(assemble_inputs(HOLDOUT_RUNS, HOLDOUT_SEEDS, HOLDOUT_STARTS), policies)
    alpha = 0.05 / (len(policies) * len(HOLDOUT_STARTS))
    detail, rows = [], []
    for (name, scenario), samples in holdout.items():
        detail.append(summarize_case(name, scenario, samples, alpha))
        rows.extend({"scenario": scenario, "policy": name, "replication": i, **sample}
                    for i, sample in enumerate(samples, start=1))
    pairs = {
        scenario: paired_gain(metric(holdout["two_peak", scenario], "released"),
                              metric(holdout["one_long", scenario], "released"))
        for scenario in HOLDOUT_STARTS
    }
    return {
        "case": "Mitchell", "target_released": DEMAND, "shift_seconds": SHIFT_SECONDS,
        "helper_person_minutes": PERSON_MINUTES, "design_start_increment": START_INCREMENT,
        "pulse_minutes": PULSE_MINUTES, "pulse_failure_share_pct": FAILURE_SHARE_PCT,
        "design": {"runs": DESIGN_RUNS, "seeds": DESIGN_SEEDS, "pulse_starts": DESIGN_STARTS,
                   "candidates": [{**asdict(p), "worst_hits": -scores[p.name][0],
                                   "worst_shortfall_total": scores[p.name][1],
                                   "released_total": -scores[p.name][2]} for p in (*one, *two)]},
        "holdout": {"runs": HOLDOUT_RUNS, "seeds": HOLDOUT_SEEDS, "pulse_starts": HOLDOUT_STARTS,
                    "alpha_per_scenario_policy": alpha, "required_reliability": REQUIRED_RELIABILITY,
                    "required_hits": minimum_hits(HOLDOUT_RUNS, alpha), "policies": list(map(asdict, policies)),
                    "results": detail, "paired_two_peak_gain_vs_one_long": pairs},
    }, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    report, rows = run_study()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "peak_rework_holdout.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "peak_rework_support.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

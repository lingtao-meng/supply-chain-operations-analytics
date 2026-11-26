"""Cross-check the station benchmark using fixed-station integer feasibility models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from assignment_benchmark import FACTORS, Benchmark, read_benchmark, timing_candidates
from line_balance import ROOT
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


def feasible(benchmark: Benchmark, station_count: int, cycle: int, restrictions: tuple[str, ...]) -> bool:
    if station_count < 1:
        return False

    case = benchmark.case
    variable_count = len(case.task_times) * station_count
    constraints = []
    coefficients = []
    columns = []
    lower, upper = [], []

    def position(task: int, station: int) -> int:
        return (task - 1) * station_count + station

    def add_constraint(terms, lo: float, hi: float) -> None:
        for index, coefficient in terms:
            constraints.append(len(lower))
            coefficients.append(coefficient)
            columns.append(index)
        lower.append(lo)
        upper.append(hi)

    for task in case.order:
        add_constraint(((position(task, station), 1) for station in range(station_count)), 1, 1)
    for first, second in case.precedence:
        terms = [(position(first, station), station) for station in range(station_count)]
        terms += [(position(second, station), -station) for station in range(station_count)]
        add_constraint(terms, -np.inf, 0)

    for station in range(station_count):
        add_constraint(((position(task, station), case.duration(task)) for task in case.order), 0, cycle)
        if "resource_caps" in restrictions:
            for resource, cap in enumerate(benchmark.caps):
                terms = ((position(task, station), benchmark.resources[task - 1][resource])
                         for task in case.order)
                add_constraint(terms, 0, cap)
        if "linked_tasks" in restrictions:
            for first, second in benchmark.linked:
                add_constraint(((position(first, station), 1), (position(second, station), -1)), 0, 0)
        if "incompatible_tasks" in restrictions:
            for first, second in benchmark.incompatible:
                add_constraint(((position(first, station), 1), (position(second, station), 1)), 0, 1)

    # Empty stations are allowed here. Removing one preserves the order and all restrictions.
    matrix = coo_matrix((coefficients, (constraints, columns)), shape=(len(lower), variable_count)).tocsc()
    solution = milp(np.zeros(variable_count), integrality=np.ones(variable_count), bounds=Bounds(0, 1),
                    constraints=LinearConstraint(matrix, lower, upper), options={"time_limit": 30.0})
    if solution.status not in (0, 2):
        raise RuntimeError(f"{case.name}: feasibility check inconclusive: {solution.message}")
    return solution.status == 0


def verify_station_result(benchmark: Benchmark, result: dict, cycle: int, restrictions: tuple[str, ...]) -> int:
    if result["status"] == "infeasible":
        if feasible(benchmark, len(benchmark.case.task_times), cycle, restrictions):
            raise AssertionError(f"{benchmark.case.name}: reported infeasible but a plan exists")
        return 1
    stations = result["stations"]
    peak = result["peak_load"]
    checks = [
        feasible(benchmark, stations, peak, restrictions),
        not feasible(benchmark, stations - 1, cycle, restrictions),
        not feasible(benchmark, stations, peak - 1, restrictions),
    ]
    if not all(checks):
        raise AssertionError(f"{benchmark.case.name}: {cycle=}, {restrictions=}, {checks=}")
    return len(checks)


def verify_timing_candidates(benchmark: Benchmark, trial: dict) -> int:
    checks = 0
    for row in trial["budgets"]:
        expected = {tuple(sorted((int(task), amount) for task, amount in plan["reductions"].items()))
                    for plan in row["passing"]}
        actual = set()
        count = 0
        for reductions, candidate in timing_candidates(benchmark, row["total_reduction"]):
            count += 1
            if feasible(candidate, trial["station_limit"], trial["target_cycle"], FACTORS):
                actual.add(tuple(sorted(reductions.items())))
        if actual != expected or count != row["candidates"] or len(actual) != row["passing_count"]:
            raise AssertionError(f"Time-reduction results differ at budget {row['total_reduction']}")
        checks += count
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--report", type=Path, default=ROOT / "results" / "assignment_benchmark.json")
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    base_checks = timing_checks = 0
    with ZipFile(args.archive) as source:
        for row in report["cases"]:
            benchmark = read_benchmark(row["name"], source.read(row["source_entry"]))
            for profile in row["profiles"]:
                base_checks += verify_station_result(benchmark, profile, row["cycle"], tuple(profile["restrictions"]))
            stress = row["restricted_at_unrestricted_peak"]
            base_checks += verify_station_result(benchmark, stress, stress["cycle"], FACTORS)
            if row["name"] == "Mitchell":
                timing_checks = verify_timing_candidates(benchmark, report["mitchell_time_reductions"])
    print(f"Verified {base_checks} station and peak-load claims and {timing_checks} time-reduction candidates.")


if __name__ == "__main__":
    main()

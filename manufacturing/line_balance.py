"""Simple assembly line balancing with fixed cycle times (SALBP-1)."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from graphlib import TopologicalSorter
from math import ceil
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

ROOT = Path(__file__).resolve().parent
METHODS = ("LCR", "RPW", "MILP")


@dataclass(frozen=True)
class Case:
    name: str
    task_times: tuple[int, ...]
    precedence: tuple[tuple[int, int], ...]
    reference_cycle: int
    cycle_times: tuple[int, ...]
    predecessors: dict[int, set[int]] = field(init=False)
    order: tuple[int, ...] = field(init=False)

    def __post_init__(self):
        if not self.task_times or any(type(t) is not int or t <= 0 for t in self.task_times):
            raise ValueError("task times must be positive integers")
        predecessors = {task: set() for task in range(1, len(self.task_times) + 1)}
        for before, after in self.precedence:
            if before not in predecessors or after not in predecessors:
                raise ValueError("precedence references an unknown task")
            predecessors[after].add(before)
        object.__setattr__(self, "predecessors", predecessors)
        object.__setattr__(self, "order", tuple(TopologicalSorter(predecessors).static_order()))
        if self.reference_cycle not in self.cycle_times or len(set(self.cycle_times)) != len(self.cycle_times):
            raise ValueError("cycle times must be unique and include the reference cycle")
        for cycle in self.cycle_times:
            self.check_cycle(cycle)

    def duration(self, task: int) -> int:
        return self.task_times[task - 1]

    def check_cycle(self, cycle: int) -> None:
        if type(cycle) is not int or cycle < max(self.task_times):
            raise ValueError("cycle time must be an integer at least as long as the longest task")


def greedy(case: Case, cycle: int, method: str) -> list[list[int]]:
    case.check_cycle(cycle)
    if method not in ("LCR", "RPW"):
        raise ValueError(f"unknown greedy method: {method}")
    priority = {task: case.duration(task) for task in case.order}
    if method == "RPW":
        descendants: dict[int, set[int]] = {}
        for task in reversed(case.order):
            descendants[task] = set()
            for successor, predecessors in case.predecessors.items():
                if task in predecessors:
                    descendants[task].add(successor)
                    descendants[task].update(descendants[successor])
            priority[task] += sum(case.duration(child) for child in descendants[task])

    ranked = sorted(case.order, key=lambda task: (-priority[task], task))
    completed: set[int] = set()
    stations: list[list[int]] = []
    while len(completed) < len(ranked):
        station = []
        remaining = cycle
        while (task := next(
            (task for task in ranked if task not in completed
             and case.predecessors[task] <= completed and case.duration(task) <= remaining),
            None,
        )) is not None:
            station.append(task)
            completed.add(task)
            remaining -= case.duration(task)
        stations.append(station)
    return stations


def verify(case: Case, cycle: int, stations: list[list[int]]) -> list[int]:
    """Check exact coverage, work content and sequence; return station loads."""
    sequence = [task for station in stations for task in station]
    if not all(stations) or sorted(sequence) != sorted(case.order):
        raise ValueError("each task must be assigned exactly once to a nonempty station")
    position = {task: index for index, task in enumerate(sequence)}
    if any(position[before] >= position[after] for before, after in case.precedence):
        raise ValueError("assignment violates task precedence")
    loads = [sum(case.duration(task) for task in station) for station in stations]
    if max(loads) > cycle:
        raise ValueError("station load exceeds cycle time")
    return loads


def reference_plans(case: Case, cycle: int) -> dict[str, list[list[int]]]:
    """Read the fixed station assignments used to compare simulation policies."""
    groups: dict[str, list[tuple[int, list[int]]]] = {}
    with (ROOT / "data" / "reference_station_plans.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["case"] == case.name and int(row["cycle"]) == cycle:
                groups.setdefault(row["method"], []).append(
                    (int(row["station"]), list(map(int, row["tasks_in_order"].split())))
                )
    if not groups:
        raise ValueError(f"no reference station plans for {case.name} at cycle {cycle}")
    plans = {}
    for method, indexed in groups.items():
        numbers, stations = zip(*sorted(indexed, key=lambda row: row[0]))
        if list(numbers) != list(range(1, len(stations) + 1)):
            raise ValueError(f"invalid reference station numbering for {case.name} {method}")
        plans[method] = list(stations)
        verify(case, cycle, plans[method])
    return plans


def optimize(case: Case, cycle: int, max_stations: int) -> list[list[int]]:
    """Minimize active stations, then minimize the heaviest station's workload."""
    case.check_cycle(cycle)
    times = np.asarray(case.task_times)
    n = len(times)
    m = max_stations
    if not ceil(sum(case.task_times) / cycle) <= m <= n:
        raise ValueError("station bound must be between the workload lower bound and the task count")
    # Variables: x[i,s] = task assignment, y[s] = station open, z = peak load.
    x = np.arange(n * m).reshape(n, m)
    y = np.arange(n * m, n * m + m)
    z = n * m + m
    size = z + 1
    matrix = lil_matrix((n + 3 * m + len(case.precedence), size))
    lower = np.full(matrix.shape[0], -np.inf)
    upper = np.zeros(matrix.shape[0])
    row = 0
    for task in range(n):
        matrix[row, x[task]] = 1
        lower[row] = upper[row] = 1
        row += 1
    for station in range(m):
        matrix[row, x[:, station]] = times
        matrix[row, y[station]] = -cycle
        row += 1
        matrix[row, x[:, station]] = times
        matrix[row, z] = -1
        row += 1
    for before, after in case.precedence:
        matrix[row, x[before - 1]] = np.arange(m)
        matrix[row, x[after - 1]] = -np.arange(m)
        row += 1
    for left, right in zip(y[:-1], y[1:]):
        matrix[row, [left, right]] = [-1, 1]
        row += 1
    matrix[row, y] = 1
    lower[row] = ceil(sum(case.task_times) / cycle)
    upper[row] = m

    objective = np.zeros(size)
    objective[y] = cycle + 1  # One station outweighs any reduction in peak load.
    objective[z] = 1
    bounds = Bounds(np.zeros(size), np.r_[np.ones(z), cycle])
    integrality = np.r_[np.ones(z), 0]
    solved = milp(
        objective, integrality=integrality, bounds=bounds,
        constraints=LinearConstraint(matrix.tocsc(), lower, upper),
        options={"mip_rel_gap": 0.0, "time_limit": 60.0},
    )
    if solved.status != 0:
        raise RuntimeError(f"{case.name}, cycle={cycle}: no proven optimum ({solved.message})")
    choices = solved.x[x]
    stations = [
        [task for task in case.order if choices[task - 1, station] > 0.5]
        for station in range(m)
    ]
    stations = [station for station in stations if station]
    loads = verify(case, cycle, stations)
    objective_check = (cycle + 1) * len(stations) + max(loads)
    if not np.isclose(solved.fun, objective_check, rtol=0, atol=1e-5):
        raise RuntimeError("decoded assignment does not match the optimal objective")
    return stations


def analyze(case: Case) -> tuple[list[dict], list[dict]]:
    summary = []
    assignments = []
    work = sum(case.task_times)
    for cycle in sorted(case.cycle_times):
        plans = {method: greedy(case, cycle, method) for method in METHODS[:-1]}
        plans["MILP"] = optimize(case, cycle, min(map(len, plans.values())))
        for method, stations in plans.items():
            loads = verify(case, cycle, stations)
            total_time = len(loads) * cycle
            peak_load = max(loads)
            summary.append({
                "case": case.name, "cycle": cycle, "method": method,
                "tasks": len(case.task_times), "work_content": work,
                "workload_lower_bound": ceil(work / cycle), "stations": len(loads),
                "peak_load": peak_load, "total_idle": total_time - work,
                "balance_rate_pct": round(100 * work / total_time, 4),
            })
            for number, (station, load) in enumerate(zip(stations, loads), start=1):
                assignments.append({
                    "case": case.name, "cycle": cycle, "method": method,
                    "station": number, "tasks_in_order": " ".join(map(str, station)),
                    "load": load, "idle": cycle - load, "bottleneck": int(load == peak_load),
                })
    return summary, assignments


def plot(cases: list[Case], summary: list[dict], assignments: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"LCR": "#9c6334", "RPW": "#5b6e8e", "MILP": "#146a63"}
    with plt.rc_context({"svg.fonttype": "none", "svg.hashsalt": "salbp-workload"}):
        fig, axes = plt.subplots(len(cases), 2, squeeze=False, figsize=(11.2, 3.4 * len(cases)))
        for case, (loads_ax, cycle_ax) in zip(cases, axes):
            for offset, method in enumerate(METHODS):
                rows = [r for r in assignments if r["case"] == case.name
                        and r["cycle"] == case.reference_cycle and r["method"] == method]
                loads_ax.bar([r["station"] + (offset - 1) * 0.25 for r in rows],
                             [r["load"] for r in rows], 0.23, label=method, color=colors[method])
                points = [r for r in summary if r["case"] == case.name and r["method"] == method]
                cycle_ax.plot([r["cycle"] for r in points], [r["stations"] for r in points],
                              marker=("o", "s", "D")[offset], markersize=(9, 6, 4)[offset],
                              markerfacecolor="none", linestyle=("-", "--", ":")[offset],
                              linewidth=1.5, color=colors[method], label=method)
            loads_ax.axhline(case.reference_cycle, ls="--", color="#303842", lw=1)
            loads_ax.set(title=f"{case.name}  |  Workloads at cycle {case.reference_cycle}",
                         xlabel="Station", ylabel="Workload (time units)")
            loads_ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            loads_ax.set_ylim(0, case.reference_cycle * 1.14)
            cycle_ax.set(title=f"{case.name}  |  Stations required by cycle target",
                         xlabel="Cycle limit (time units)", ylabel="Stations", xticks=sorted(case.cycle_times))
            cycle_ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            for ax in (loads_ax, cycle_ax):
                ax.set_axisbelow(True)
                ax.grid(axis="y", alpha=0.22)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, ncols=3, loc="lower center")
        fig.tight_layout(pad=2, rect=(0, 0.05, 1, 1))
        fig.savefig(path, metadata={"Date": None})
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "benchmarks.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    cases = [Case(**record) for record in json.loads(args.data.read_text(encoding="utf-8"))]
    if not cases or len({case.name for case in cases}) != len(cases):
        parser.error("provide one or more cases with unique names")
    summary, assignments = [], []
    for case in cases:
        rows, stations = analyze(case)
        summary.extend(rows)
        assignments.extend(stations)
        for row in rows:
            if row["cycle"] == case.reference_cycle:
                print(f"{case.name:8}  cycle {row['cycle']:2}  {row['method']:4}  "
                      f"{row['stations']} stations  peak {row['peak_load']:2}  "
                      f"balance {row['balance_rate_pct']:.2f}%")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("method_comparison", summary), ("station_assignments", assignments)):
        with (args.output_dir / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    plot(cases, summary, assignments, args.output_dir / "line_balance.svg")


if __name__ == "__main__":
    main()

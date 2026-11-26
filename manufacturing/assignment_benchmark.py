"""Exact station balancing on four published instances with assignment restrictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, replace
from itertools import combinations_with_replacement, product
from pathlib import Path
from zipfile import ZipFile

from line_balance import Case, verify
from optimize_buffers import predecessor_ideals

INSTANCES = (("Jackson", 14), ("Mitchell", 15), ("Buxey", 41), ("Sawyer30", 41))
SOURCE = "https://assembly-line-balancing.de/aralbp/"
FACTORS = ("resource_caps", "linked_tasks", "incompatible_tasks")
Prefix = tuple[int, int, tuple[int, ...]]
PROFILES = [
    tuple(factor for factor, on in zip(FACTORS, switches) if on)
    for switches in product((False, True), repeat=len(FACTORS))
]


@dataclass(frozen=True)
class Benchmark:
    case: Case
    resources: tuple[tuple[int, ...], ...]
    caps: tuple[int, ...]
    linked: tuple[tuple[int, int], ...]
    incompatible: tuple[tuple[int, int], ...]
    plain_optimum: int


def read_benchmark(name: str, content: bytes) -> Benchmark:
    """Read the blocks used by the ARALBP-1 benchmark archive."""
    blocks = {}
    tags = re.compile(r"^<([^>]+)>[ \t]*(?:\n|$)", re.MULTILINE)
    text = content.decode("ascii").replace("\r\n", "\n")
    headers = list(tags.finditer(text))
    for index, tag in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        key = tag[1].strip().lower()
        if key in blocks:
            raise ValueError(f"Repeated block: {key}")
        blocks[key] = text[tag.end():end].strip().splitlines()
    supported = {
        "number of tasks", "cycle time", "task times", "precedence relations",
        "number of task attributes", "task attribute values", "attribute bounds per station",
        "linked tasks", "incompatible tasks", "end", "optimal salbp-1 value", "rand seed",
    }
    if blocks.keys() - supported:
        raise ValueError(f"Unknown benchmark blocks: {blocks.keys() - supported}")

    task_count = int(blocks["number of tasks"][0])
    cycle = int(blocks["cycle time"][0])
    times = dict(tuple(map(int, re.split(r"[\s:]+", line))) for line in blocks["task times"])
    if sorted(times) != list(range(1, task_count + 1)) or len(blocks["task times"]) != task_count:
        raise ValueError("Expected exactly one time for each task")

    def pairs(block: str) -> tuple[tuple[int, int], ...]:
        values = tuple(tuple(map(int, row.split(","))) for row in blocks.get(block, []))
        if any(len(pair) != 2 or pair[0] == pair[1] or not set(pair) <= times.keys() for pair in values):
            raise ValueError(f"Invalid task pair in {block}")
        return values

    dimensions = int(blocks["number of task attributes"][0])
    caps = dict(tuple(map(int, row.split(":"))) for row in blocks["attribute bounds per station"])
    if sorted(caps) != list(range(1, dimensions + 1)) or min(caps.values()) <= 0:
        raise ValueError("Expected one positive upper limit per resource")
    resources = [[0] * dimensions for _ in range(task_count)]
    seen = set()
    for row in blocks["task attribute values"]:
        index, value = row.split(":")
        task, resource = map(int, index.split(","))
        quantity = int(value)
        if task not in times or resource not in caps or quantity < 0 or (task, resource) in seen:
            raise ValueError(f"Invalid resource entry: {row}")
        resources[task - 1][resource - 1] = quantity
        seen.add((task, resource))
    case = Case(name, tuple(times[task] for task in range(1, task_count + 1)),
                pairs("precedence relations"), cycle, (cycle,))
    return Benchmark(case, tuple(map(tuple, resources)), tuple(caps[r] for r in sorted(caps)),
                     pairs("linked tasks"), pairs("incompatible tasks"),
                     int(blocks["optimal salbp-1 value"][0]))


def prefix_data(benchmark: Benchmark) -> list[Prefix]:
    """For each precedence-closed prefix, store its membership and cumulative loads."""
    case = benchmark.case
    return [
        (sum(1 << (task - 1) for task in prefix),
         sum(map(case.duration, prefix)),
         tuple(sum(benchmark.resources[task - 1][r] for task in prefix) for r in range(len(benchmark.caps))))
        for prefix in predecessor_ideals(case)
    ]


def exact_plan(
    benchmark: Benchmark, prefixes: list[Prefix], restrictions: tuple[str, ...], cycle: int,
) -> dict:
    """Find a shortest path between feasible station boundaries, then minimize peak workload."""
    case = benchmark.case
    case.check_cycle(cycle)
    linked = benchmark.linked if "linked_tasks" in restrictions else ()
    incompatible = benchmark.incompatible if "incompatible_tasks" in restrictions else ()
    pair_masks = [(1 << (first - 1)) | (1 << (second - 1)) for first, second in incompatible]
    caps = benchmark.caps if "resource_caps" in restrictions else ()
    scores = {0: (0, 0)}
    previous = {}
    reachable = [(0, 0, (0,) * len(benchmark.caps))]
    for current, total, resources in prefixes[1:]:
        if any(bool(current & (1 << (a - 1))) != bool(current & (1 << (b - 1))) for a, b in linked):
            continue  # Linked tasks must both be complete or both pending at each boundary.
        best = None
        for before, prior_total, prior_resources in reachable:
            if before & current != before:
                continue
            station_time = total - prior_total
            if station_time > cycle:
                continue
            station = current ^ before
            if any(station & pair == pair for pair in pair_masks):
                continue
            if any(current_load - prior_load > cap
                   for current_load, prior_load, cap in zip(resources, prior_resources, caps)):
                continue
            count, peak = scores[before]
            candidate = (count + 1, max(peak, station_time), before)
            if best is None or candidate < best:
                best = candidate
        if best is not None:
            scores[current] = best[:2]
            previous[current] = best[2]
            reachable.append((current, total, resources))

    current = prefixes[-1][0]
    if current not in scores:
        return {"status": "infeasible", "stations": None}
    optimum = scores[current]
    plan = []
    while current:
        before = previous[current]
        station = current ^ before
        plan.append([task for task in case.order if station & (1 << (task - 1))])
        current = before
    plan.reverse()

    loads = verify(case, cycle, plan)
    at = {task: station for station, tasks in enumerate(plan) for task in tasks}
    resource_loads = [
        [sum(benchmark.resources[task - 1][r] for task in tasks) for r in range(len(benchmark.caps))]
        for tasks in plan
    ]
    if ((len(plan), max(loads)) != optimum or any(at[a] != at[b] for a, b in linked)
            or any(at[a] == at[b] for a, b in incompatible)
            or any(value > cap for row in resource_loads for value, cap in zip(row, caps))):
        raise AssertionError("The decoded plan violates the benchmark or the optimum")
    return {"status": "optimal", "stations": len(plan), "peak_load": max(loads),
            "station_tasks": plan, "station_loads": loads, "station_resources": resource_loads}


def audit_plan(benchmark: Benchmark, plan: list[list[int]]) -> dict:
    """Check the unrestricted design against the three published restrictions."""
    station_of = {task: index for index, tasks in enumerate(plan, start=1) for task in tasks}
    over_caps = []
    for station, tasks in enumerate(plan, start=1):
        for resource, cap in enumerate(benchmark.caps):
            load = sum(benchmark.resources[task - 1][resource] for task in tasks)
            if load > cap:
                over_caps.append({"station": station, "resource": resource + 1, "load": load, "cap": cap})
    return {
        "resource_caps": over_caps,
        "linked_tasks": [(a, b) for a, b in benchmark.linked if station_of[a] != station_of[b]],
        "incompatible_tasks": [(a, b) for a, b in benchmark.incompatible if station_of[a] == station_of[b]],
    }


def timing_candidates(benchmark: Benchmark, budget: int):
    """Enumerate integer time reductions that keep every task duration positive."""
    for tasks in combinations_with_replacement(range(1, len(benchmark.case.task_times) + 1), budget):
        reductions = Counter(tasks)
        durations = list(benchmark.case.task_times)
        for task, reduction in reductions.items():
            durations[task - 1] -= reduction
        if min(durations) >= 1:
            case = replace(benchmark.case, task_times=tuple(durations))
            yield reductions, replace(benchmark, case=case)


def compare_time_reductions(benchmark: Benchmark, cycle: int, station_limit: int, max_budget: int) -> dict:
    """Evaluate each total reduction up to the stated budget under unchanged assignment rules."""
    rows = []
    for budget in range(max_budget + 1):
        candidates = 0
        best_station_count = None
        passing = []
        for reductions, candidate in timing_candidates(benchmark, budget):
            candidates += 1
            plan = exact_plan(candidate, prefix_data(candidate), FACTORS, cycle)
            station_count = plan["stations"]
            if station_count is None:
                continue
            if best_station_count is None or station_count < best_station_count:
                best_station_count = station_count
            if station_count <= station_limit:
                passing.append({"reductions": dict(reductions), **plan})
        rows.append({
            "total_reduction": budget, "candidates": candidates,
            "fewest_stations": best_station_count,
            "passing_count": len(passing), "passing": passing,
        })
    first_success = next((row["total_reduction"] for row in rows if row["passing"]), None)
    return {"target_cycle": cycle, "station_limit": station_limit, "minimum_reduction": first_success,
            "budgets": rows}


def run(archive: Path) -> dict:
    results = []
    process_trial = None
    with ZipFile(archive) as source:
        for name, cycle in INSTANCES:
            entry = f"3_10_10/{name.lower()}_c={cycle}.alb"
            content = source.read(entry)
            benchmark = read_benchmark(name, content)
            if benchmark.case.reference_cycle != cycle:
                raise ValueError(f"Cycle time does not match filename: {entry}")
            prefixes = prefix_data(benchmark)
            profiles = [{"restrictions": profile, **exact_plan(benchmark, prefixes, profile, cycle)}
                        for profile in PROFILES]
            if profiles[0]["stations"] != benchmark.plain_optimum:
                raise AssertionError(f"Baseline does not match the published SALBP-1 optimum: {entry}")
            baseline = profiles[0]
            tighter_cycle = baseline["peak_load"]
            if name == "Mitchell":
                process_trial = compare_time_reductions(benchmark, tighter_cycle, baseline["stations"], max_budget=2)
            results.append({
                "name": name, "cycle": cycle, "source_entry": entry,
                "sha256": hashlib.sha256(content).hexdigest(), "tasks": len(benchmark.case.task_times),
                "prefixes": len(prefixes), "published_plain_optimum": benchmark.plain_optimum,
                "resource_caps": benchmark.caps, "linked_pairs": benchmark.linked,
                "incompatible_pairs": benchmark.incompatible, "profiles": profiles,
                "unrestricted_plan_violations": audit_plan(benchmark, baseline["station_tasks"]),
                "restricted_at_unrestricted_peak": {
                    "cycle": tighter_cycle, **exact_plan(benchmark, prefixes, FACTORS, tighter_cycle),
                },
            })
    with archive.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"source": SOURCE, "archive_sha256": digest.hexdigest(), "cases": results, "mitchell_time_reductions": process_trial}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Original ARALBP-1-data-sets.zip")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(run(args.archive), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()

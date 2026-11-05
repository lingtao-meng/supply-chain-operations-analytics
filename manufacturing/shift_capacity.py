"""Compare finite-shift throughput under task-time variability and finite buffers."""

from __future__ import annotations

import argparse
import csv
import json
from math import ceil, floor
from pathlib import Path

import numpy as np
from scipy.stats import norm, t

from line_balance import ROOT, Case, reference_plans, verify

SHIFT_SECONDS = 7 * 60 * 60
DEMAND = 1650
DESIGN_SEED, VALIDATION_SEED = 20261105, 20261106
DESIGN_REPLICATIONS, VALIDATION_REPLICATIONS = 120, 180
DESIGN_SCENARIOS = ((20, 0), (20, 3))
VALIDATION_SCENARIOS = ((0, 0), (10, 0), *DESIGN_SCENARIOS, (20, 5))
MAX_STAFF = 8
MAX_BUFFER = 14
REQUIRED_RELIABILITY = 0.95
BUFFER_POLICIES = ("none", "uniform_1", "targeted_1", "uniform_2", "uniform_4")


def simulate_shift(service: np.ndarray, buffers: tuple[int, ...], duration: float) -> tuple[np.ndarray, np.ndarray]:
    """Return completions and per-station processing, blocking and idle fractions.

    service axes are product, station, and replication. Jobs share the station order.
    Buffers hold waiting jobs only. A blocked job stays on its upstream station.
    """
    return _flow_shop(service, buffers, duration)


def demand_completion_times(service: np.ndarray, buffers: tuple[int, ...]) -> np.ndarray:
    """Time at which the last input product clears the line in each replication."""
    return _flow_shop(service, buffers, None)


def shift_output_counts(service: np.ndarray, buffers: tuple[int, ...], duration: float) -> np.ndarray:
    """Count completed products without accumulating per-station time metrics."""
    return _flow_shop(service, buffers, duration, measure_stations=False)


def _flow_shop(
    service: np.ndarray, buffers: tuple[int, ...], duration: float | None, *, measure_stations: bool = True,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    if service.ndim != 3 or not all(service.shape) or not np.isfinite(service).all() or (service <= 0).any():
        raise ValueError("service must contain finite positive times for products, stations and replications")
    _, station_count, replications = service.shape
    if len(buffers) != station_count - 1 or any(type(b) is not int or b < 0 for b in buffers):
        raise ValueError("provide one nonnegative integer buffer per station gap")
    track_shift = duration is not None
    track_stations = track_shift and measure_stations
    if track_shift and (not np.isfinite(duration) or duration <= 0):
        raise ValueError("shift duration must be finite and positive")
    exits = np.zeros((station_count, replications))
    history_length = 1 + max(buffers, default=0)
    prior_starts = np.zeros((history_length, station_count, replications))
    output = np.zeros(replications, dtype=int)
    if track_stations:
        busy = np.zeros_like(exits)
        blocked = np.zeros_like(exits)

    for product, workloads in enumerate(service):
        if track_shift and (exits[0] >= duration).all():
            break
        upstream = 0.0
        start_slot = prior_starts[product % history_length]
        for station, load in enumerate(workloads):
            start = np.maximum(upstream, exits[station])
            finish = start + load
            departure = finish
            if station + 1 < station_count:
                buffer = buffers[station]
                if buffer == 0:
                    # The previous job must leave the downstream station before handoff.
                    departure = np.maximum(finish, exits[station + 1])
                elif product >= buffer:
                    # A waiting slot frees when the downstream station starts that job.
                    departure = np.maximum(finish, prior_starts[(product - buffer) % history_length, station + 1])
            start_slot[station] = start
            exits[station] = upstream = departure
            if track_stations:
                busy[station] += np.minimum(finish, duration) - np.minimum(start, duration)
                blocked[station] += np.minimum(departure, duration) - np.minimum(finish, duration)
        if track_shift:
            output += exits[-1] <= duration

    if not track_shift:
        return exits[-1]
    if (exits[0] < duration).any():
        raise ValueError("the input workload ends before the shift; supply more products")
    if not track_stations:
        return output
    idle = np.maximum(0, duration - busy - blocked)
    return output, np.stack((busy, blocked, idle), axis=-1) / duration


def deterministic_output(loads: list[int], duration: int) -> int:
    """An empty, deterministic line finishes its first product after the full work content."""
    return max(0, 1 + floor((duration - sum(loads)) / max(loads)))


def buffer_layout(loads: list[int], policy: str) -> tuple[int, ...]:
    """Concentrate one-slot-per-gap budget around the heaviest neighboring loads."""
    gaps = len(loads) - 1
    if policy == "targeted_1":
        # Take the total budget of uniform_1, with at most two slots per gap.
        ranked = sorted(range(gaps), key=lambda i: (-min(loads[i:i + 2]), -sum(loads[i:i + 2]), i))
        slots = [0] * gaps
        for gap in ranked[:ceil(gaps / 2)]:
            slots[gap] = min(2, gaps - sum(slots))
        return tuple(slots)
    return ({"none": 0, "uniform_1": 1, "uniform_2": 2, "uniform_4": 4}[policy],) * gaps


def reliability_lower(hits: int, runs: int) -> float:
    """One-sided 95% Wilson lower bound on the simulated hit probability."""
    observed = hits / runs
    z = norm.ppf(0.95)
    center = observed + z * z / (2 * runs)
    radius = z * np.sqrt(observed * (1 - observed) / runs + z * z / (4 * runs * runs))
    return float(max(0, (center - radius) / (1 + z * z / runs)))


def choose_design(design: list[dict]) -> tuple[str, str] | None:
    candidates = {}
    for row in design:
        candidates.setdefault((row["method"], row["policy"]), []).append(row)
    eligible = []
    for (method, policy), rows in candidates.items():
        first = rows[0]
        if (first["staffed_stations"] <= MAX_STAFF and first["total_buffer"] <= MAX_BUFFER
                and all(row["hit_lower95"] >= REQUIRED_RELIABILITY for row in rows)):
            eligible.append((first["total_buffer"], first["staffed_stations"],
                             -min(row["p05_output_units"] for row in rows), method, policy))
    if not eligible:
        return None
    return min(eligible)[-2:]


def lognormal_multiplier(noise: np.ndarray, cv_pct: int) -> np.ndarray:
    variance = np.log1p((cv_pct / 100) ** 2)
    return np.exp(np.sqrt(variance) * noise - variance / 2)


def run_study() -> tuple[list[dict], list[dict], dict]:
    records = json.loads((ROOT / "data" / "benchmarks.json").read_text(encoding="utf-8"))
    case = Case(**next(record for record in records if record["name"] == "Mitchell"))
    cycle = floor(SHIFT_SECONDS / DEMAND)
    plans = reference_plans(case, cycle)
    loads = {name: verify(case, cycle, stations) for name, stations in plans.items()}

    jobs = 2 * ceil(SHIFT_SECONDS / min(map(max, loads.values())))
    nominal = np.asarray(case.task_times)[None, :, None]
    summary, station_rows = [], []
    selected = None
    for stage, seed, replications, scenarios in (
        ("design", DESIGN_SEED, DESIGN_REPLICATIONS, DESIGN_SCENARIOS),
        ("validation", VALIDATION_SEED, VALIDATION_REPLICATIONS, VALIDATION_SCENARIOS),
    ):
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal((jobs, len(case.task_times), replications))
        shift_noise = rng.standard_normal((1, 1, replications))
        for task_cv, shift_cv in scenarios:
            if task_cv or shift_cv:
                task_times = nominal * lognormal_multiplier(noise, task_cv)
                task_times *= lognormal_multiplier(shift_noise, shift_cv)
            else:
                task_times = np.broadcast_to(nominal, (jobs, len(case.task_times), 1))
            workloads = {
                method: np.stack([task_times[:, np.asarray(station) - 1].sum(axis=1)
                                  for station in stations], axis=1)
                for method, stations in plans.items()
            }
            for policy in BUFFER_POLICIES:
                layouts = {method: buffer_layout(loads[method], policy) for method in plans}
                results = {method: simulate_shift(service, layouts[method], SHIFT_SECONDS)
                           for method, service in workloads.items()}
                reference = results["RPW"][0]
                for method, (output, state) in results.items():
                    workers = len(plans[method])
                    mean = float(output.mean())
                    hits = int((output >= DEMAND).sum())
                    gap = output - reference
                    paired_delta = float(gap.mean())
                    radius = (float(t.ppf(0.975, len(gap) - 1) * gap.std(ddof=1) / np.sqrt(len(gap)))
                              if len(gap) > 1 else 0.0)
                    summary.append({
                        "stage": stage, "seed": seed, "task_cv_pct": task_cv, "shift_cv_pct": shift_cv,
                        "method": method, "policy": policy, "staffed_stations": workers,
                        "buffer_layout": " ".join(map(str, layouts[method])), "total_buffer": sum(layouts[method]),
                        "demand_units": DEMAND, "runs": len(output), "hits": hits,
                        "nominal_peak_seconds": max(loads[method]),
                        "deterministic_empty_shift_units": deterministic_output(loads[method], SHIFT_SECONDS),
                        "mean_output_units": round(mean, 2),
                        "p05_output_units": round(float(np.quantile(output, 0.05)), 2),
                        "demand_hit_pct": round(100 * hits / len(output), 2),
                        "hit_lower95": reliability_lower(hits, len(output)) if len(output) > 1 else None,
                        "mean_shortfall_units": round(float(np.maximum(DEMAND - output, 0).mean()), 2),
                        "staffed_hours_per_100_units": round(100 * workers * SHIFT_SECONDS / 3600 / mean, 3),
                        "paired_delta_vs_rpw": round(paired_delta, 2),
                        "paired_delta_ci95_low": round(paired_delta - radius, 2),
                        "paired_delta_ci95_high": round(paired_delta + radius, 2),
                    })
                    if stage == "validation":
                        for index, station in enumerate(plans[method]):
                            processing, blocking, idle = 100 * state[index].mean(axis=0)
                            station_rows.append({
                                "task_cv_pct": task_cv, "shift_cv_pct": shift_cv, "policy": policy, "method": method,
                                "station": index + 1, "tasks_in_order": " ".join(map(str, station)),
                                "nominal_load_seconds": loads[method][index],
                                "processing_pct": round(float(processing), 2),
                                "blocked_pct": round(float(blocking), 2),
                                "idle_pct": round(float(idle), 2),
                            })
        if stage == "design":
            selected = choose_design(summary)

    validation = [
        {**row, "used_for_design_screen": (row["task_cv_pct"], row["shift_cv_pct"]) in DESIGN_SCENARIOS,
         "meets_reliability": row["hit_lower95"] >= REQUIRED_RELIABILITY}
        for row in summary if row["stage"] == "validation"
        and (row["method"], row["policy"]) == selected and row["hit_lower95"] is not None
    ]
    decision = {
        "net_shift_seconds": SHIFT_SECONDS, "demand_units": DEMAND, "demand_takt_seconds": SHIFT_SECONDS / DEMAND,
        "integer_design_cycle": cycle, "max_staff": MAX_STAFF, "max_buffer": MAX_BUFFER,
        "required_reliability": REQUIRED_RELIABILITY,
        "design_scenarios_cv_pct": DESIGN_SCENARIOS,
        "selected_method": selected[0] if selected else None,
        "selected_policy": selected[1] if selected else None,
        "passes_independent_screen": bool(validation) and all(
            row["meets_reliability"] for row in validation if row["used_for_design_screen"]
        ),
        "validation": validation,
    }
    return summary, station_rows, decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    summary, station_rows, decision = run_study()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("shift_capacity", summary), ("station_time_losses", station_rows)):
        with (args.output_dir / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    (args.output_dir / "shift_decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(f"Selected {decision['selected_method']} / {decision['selected_policy']} on design scenarios")
    for row in decision["validation"]:
        print(f"  task CV={row['task_cv_pct']}%, shift CV={row['shift_cv_pct']}%: "
              f"{row['mean_output_units']:.2f} units, hits {row['hits']}/{row['runs']}, "
              f"lower bound {row['hit_lower95']:.1%}")

if __name__ == "__main__":
    main()

"""Sorting-line IE model: line balance, tandem queue simulation and cost."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Station:
    name: str
    work_content: float  # seconds of work per parcel
    cv: float
    operators: int
    availability: float = 0.92

    @property
    def effective_cycle(self) -> float:
        return self.work_content / self.operators

    @property
    def capacity_per_hour(self) -> float:
        return 3600 * self.availability / self.effective_cycle


@dataclass
class LineMetrics:
    bottleneck: str
    bottleneck_cycle: float
    capacity_per_hour: float
    balance_rate: float
    effective_cycles: list[float]


@dataclass
class SimResult:
    parcels: int
    throughput_per_hour: float
    avg_flow_min: float
    p95_flow_min: float
    overtime_min: float
    on_time_rate: float
    cost_per_parcel: float
    labor_cost: float
    overtime_cost: float
    equipment_cost: float
    total_operators: int
    backlog: np.ndarray


def load_stations(path: Path) -> list[Station]:
    frame = pd.read_csv(path)
    return [
        Station(
            name=str(row.station),
            work_content=float(row.work_content_sec),
            cv=float(row.cv),
            operators=int(row.operators),
            availability=float(row.availability),
        )
        for row in frame.itertuples()
    ]


def line_metrics(stations: list[Station]) -> LineMetrics:
    cycles = [station.effective_cycle for station in stations]
    index = int(np.argmax(cycles))
    bottleneck = stations[index]
    capacities = [station.capacity_per_hour for station in stations]
    return LineMetrics(
        bottleneck=bottleneck.name,
        bottleneck_cycle=float(cycles[index]),
        capacity_per_hour=float(min(capacities)),
        balance_rate=float(sum(cycles) / (len(cycles) * max(cycles))),
        effective_cycles=cycles,
    )


def with_operator(stations: list[Station], station_name: str, operators: int) -> list[Station]:
    return [replace(station, operators=operators) if station.name == station_name else station for station in stations]


def _sample_service(rng: np.random.Generator, station: Station) -> float:
    mean = station.work_content / station.availability
    sigma2 = np.log(1 + station.cv**2)
    return float(rng.lognormal(np.log(mean) - sigma2 / 2, np.sqrt(sigma2)))


def simulate(
    stations: list[Station],
    rate_per_hour: float,
    hours: int = 8,
    profile: list[float] | None = None,
    seed: int = 42,
    labor_rate: float = 25.0,
    equipment_per_parcel: float = 0.10,
    service_target_min: float = 5.0,
) -> SimResult:
    """Tandem queue with infinite buffers, lognormal service times and overtime."""
    rng = np.random.default_rng(seed)
    if profile is None:
        profile = [1.0, 1.05, 1.15, 1.15, 1.05, 1.0, 0.95, 0.95]
    profile = np.asarray(profile, dtype=float)
    profile = profile / profile.mean()
    if len(profile) != hours:
        raise ValueError("profile length must equal hours")

    arrivals = []
    seq = 0
    for minute in range(hours * 60):
        lam = rate_per_hour / 60 * profile[minute // 60]
        for _ in range(rng.poisson(lam)):
            arrivals.append((minute * 60 + rng.uniform(0, 60), seq))
            seq += 1
    arrivals.sort()
    parcels = len(arrivals)
    arrival_times = [0.0] * parcels

    queues = [[] for _ in stations]
    busy = [0] * len(stations)
    completion_times = []
    flow_times = []
    events: list[tuple[float, int, str, object, int]] = []
    counter = 0
    for arrival_time, parcel_id in arrivals:
        arrival_times[parcel_id] = arrival_time
        heapq.heappush(events, (arrival_time, counter, "arrive", parcel_id, 0))
        counter += 1

    def start(station_index: int, now: float) -> None:
        nonlocal counter
        station = stations[station_index]
        while queues[station_index] and busy[station_index] < station.operators:
            parcel_id = queues[station_index].pop(0)
            busy[station_index] += 1
            service = _sample_service(rng, station)
            heapq.heappush(events, (now + service, counter, "done", parcel_id, station_index))
            counter += 1

    shift_end = hours * 3600
    while events:
        now, _, kind, parcel, station_index = heapq.heappop(events)
        if kind == "arrive":
            queues[station_index].append(parcel)
            start(station_index, now)
        else:
            busy[station_index] -= 1
            if station_index == len(stations) - 1:
                completion_times.append(now)
                flow_times.append(now - arrival_times[parcel])
            else:
                queues[station_index + 1].append(parcel)
                start(station_index + 1, now)
            start(station_index, now)

    completion = np.array(completion_times)
    flow = np.array(flow_times)
    arrival_array = np.array(arrival_times)
    last = float(completion.max()) if len(completion) else 0.0
    end_min = int(np.ceil(last / 60)) + 1
    minutes = np.arange(end_min + 1)
    arrivals_cum = np.searchsorted(np.sort(arrival_array), minutes * 60, side="right")
    completions_cum = np.searchsorted(np.sort(completion), minutes * 60, side="right")
    backlog = (arrivals_cum - completions_cum).astype(float)
    overtime_min = max(0.0, (last - shift_end) / 60)
    flow_min = flow / 60
    on_time = float(np.mean(flow_min <= service_target_min))
    operators = sum(station.operators for station in stations)
    regular = operators * labor_rate * hours
    overtime = operators * labor_rate * 1.5 * (overtime_min / 60)
    equipment = equipment_per_parcel * parcels
    total = regular + overtime + equipment
    return SimResult(
        parcels=parcels,
        throughput_per_hour=parcels / (hours + overtime_min / 60),
        avg_flow_min=float(flow_min.mean()) if len(flow) else 0.0,
        p95_flow_min=float(np.percentile(flow_min, 95)) if len(flow) else 0.0,
        overtime_min=overtime_min,
        on_time_rate=on_time,
        cost_per_parcel=total / parcels if parcels else 0.0,
        labor_cost=regular,
        overtime_cost=overtime,
        equipment_cost=equipment,
        total_operators=operators,
        backlog=backlog,
    )

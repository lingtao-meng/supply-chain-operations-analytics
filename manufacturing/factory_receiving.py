"""Release incoming material lots after a shared inspection queue."""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReceivingPlan:
    """Synthetic inspection duration by SKU and number of parallel benches."""

    lanes: int = 1
    minutes_a: int = 22
    minutes_b: int = 26

    def __post_init__(self) -> None:
        if min(self.lanes, self.minutes_a, self.minutes_b) < 1:
            raise ValueError("inspection lanes and durations must be positive")


@dataclass(frozen=True)
class ReceivingLot:
    """One lot from an order cohort. Cohort and event times are absolute seconds."""

    cohort: int
    sku: int
    units: int
    arrival: int
    start: int
    finished: int
    released: int
    lane: int


def receive_kits(
    scenario, plan: ReceivingPlan | None = ReceivingPlan(), advance_minutes: int = 0,
) -> tuple[np.ndarray, tuple[ReceivingLot, ...]]:
    """Release existing stock at order opening and inbound lots after inspection.

    A receipt is identified by its intended order cohort, SKU and original arrival
    time. Advancing it moves the receipt event, even if it moves all the way to
    opening; the lot still needs inspection. One bench can inspect one lot at a
    time. Lots queue by arrival, then older cohort, then SKU, then original time.
    Inspection can carry over a shift boundary. Released quantities stay with
    their designated SKU and lot; assembly applies its own FIFO rules.
    """
    if advance_minutes < 0:
        raise ValueError("receipt advance cannot be negative")
    orders = np.asarray(scenario.orders, dtype=np.int64)
    kits = np.asarray(scenario.kits, dtype=np.int64)
    sku = np.asarray(scenario.sku, dtype=np.int64)
    if orders.ndim != 1 or orders.shape != kits.shape or kits.shape != sku.shape:
        raise ValueError("orders, kits and SKUs must be matching vectors")
    if np.any(kits < orders) or np.any((sku != 0) & (sku != 1)):
        raise ValueError("receipts require a valid SKU and an open order cohort")

    arrivals = np.maximum(orders, kits - advance_minutes * 60)
    inbound = np.flatnonzero(kits > orders)
    if plan is None or not len(inbound):
        return arrivals, ()

    # Keep separate receipt lots even if early scheduling moves them to the same timestamp.
    order = np.lexsort((kits[inbound], sku[inbound], orders[inbound], arrivals[inbound]))
    jobs = inbound[order]
    new_lot = np.diff(orders[jobs]) != 0
    new_lot |= np.diff(sku[jobs]) != 0
    new_lot |= np.diff(kits[jobs]) != 0
    lots = np.split(jobs, np.flatnonzero(new_lot) + 1)
    free = [(0, lane) for lane in range(plan.lanes)]
    durations = (plan.minutes_a * 60, plan.minutes_b * 60)
    eligible = arrivals.copy()
    records = []
    for lot in lots:
        first = lot[0]
        model = int(sku[first])
        arrival = int(arrivals[first])
        available, lane = heapq.heappop(free)
        start = max(arrival, available)
        finished = start + durations[model]
        heapq.heappush(free, (finished, lane))
        eligible[lot] = finished
        records.append(ReceivingLot(int(orders[first]), model, len(lot), arrival, start, finished, finished, lane))
    return eligible, tuple(records)

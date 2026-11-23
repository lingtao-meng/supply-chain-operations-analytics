"""Load fixed-capacity trucks from the released finished-goods queue."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from shift_capacity import SHIFT_SECONDS


@dataclass(frozen=True)
class Outbound:
    departure_minutes: tuple[int, ...] = (245, 365, 420)
    capacities: tuple[int, ...] = (650, 650, 650)
    cutoff_minutes: int = 15

    def __post_init__(self):
        if len(self.departure_minutes) != len(self.capacities) or not self.departure_minutes:
            raise ValueError("each truck needs a departure time and capacity")
        if tuple(sorted(set(self.departure_minutes))) != self.departure_minutes:
            raise ValueError("departure times must be strictly increasing")
        if any(not isinstance(time, Integral) or not 0 < time * 60 <= SHIFT_SECONDS
               for time in self.departure_minutes):
            raise ValueError("truck departures must fall within the shift")
        if any(not isinstance(capacity, Integral) or capacity <= 0 for capacity in self.capacities):
            raise ValueError("truck capacities must be positive whole units")
        if not isinstance(self.cutoff_minutes, Integral) or not 0 <= self.cutoff_minutes < self.departure_minutes[0]:
            raise ValueError("loading cutoff must be after shift open and no later than departure")


def dispatch(good_ready, orders, good_mask, options: Outbound = Outbound(), shifts: int = 4) -> np.ndarray:
    """Return each unit's departure time, or infinity for scrap and unshipped units.

    Inputs are aligned by unit. Quality-release and order timestamps are absolute
    net seconds from the first shift's opening. A released good unit is eligible
    at or before the manifest cutoff. Among eligible units, load the oldest
    origin cohort first, then by quality-release time and original array order.
    """
    ready = np.asarray(good_ready)
    orders = np.asarray(orders)
    good = np.asarray(good_mask)
    if ready.ndim != 1 or ready.shape != orders.shape or ready.shape != good.shape or good.dtype.kind != "b":
        raise ValueError("quality release, order time and boolean good-unit arrays must align")
    if not isinstance(shifts, Integral) or shifts < 1:
        raise ValueError("shift count must be a positive integer")
    if not np.isfinite(orders).all() or np.any(orders < 0):
        raise ValueError("order timestamps must be finite and nonnegative")
    if np.isnan(ready[good]).any() or np.any(ready[good] < orders[good]):
        raise ValueError("good units cannot be released before the order first becomes available")

    shipped = np.full(len(ready), np.inf)
    queue = np.flatnonzero(good & np.isfinite(ready))
    queue = queue[np.lexsort((queue, ready[queue], orders[queue]))]
    for day in range(shifts):
        for minute, capacity in zip(options.departure_minutes, options.capacities):
            departure = day * SHIFT_SECONDS + minute * 60
            load_by = departure - options.cutoff_minutes * 60
            eligible_positions = np.flatnonzero(ready[queue] <= load_by)[:capacity]
            shipped[queue[eligible_positions]] = departure
            queue = np.delete(queue, eligible_positions)
    return shipped

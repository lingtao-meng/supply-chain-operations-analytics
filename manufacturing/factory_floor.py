"""Parameters and event helpers for the serial-floor experiment."""

from dataclasses import dataclass

import numpy as np
from numba import njit

BASE_LAYOUT = ((0, 0), (4, 0), (8, 0), (8, 3), (5, 3), (1, 3), (1, 8), (5, 8))
CLOSE_LAYOUT = ((0, 0), (2, 0), (4, 0), (4, 2), (2, 2), (0, 2), (0, 4), (2, 4))
# Assumed representative technical actions per unit, by model and physical station.
ACTIONS = ((6, 7, 5, 6, 9, 7, 4, 5), (7, 8, 5, 8, 11, 8, 4, 7))
# Seconds in a high-load posture per unit, at station 5, with/without a fixture aid.
POSTURE = ((5, 7), (2, 3))


@dataclass(frozen=True)
class Floor:
    buffers: tuple = (1, 1, 1, 1, 1, 1, 1)
    layout: tuple = BASE_LAYOUT
    walk: bool = True
    directional_setup: bool = True
    route_minutes: int = 15
    bin_units: int = 90
    cards: int = 2
    posture_aid: bool = False

    def distances(self):
        return np.abs(np.diff(np.asarray(self.layout), axis=0)).sum(axis=1)

    def arrays(self, setup, stations):
        if (len(self.buffers) != stations-1 or len(self.layout) != stations
                or min(self.buffers, default=0) < 0 or self.route_minutes < 0
                or min(self.bin_units, self.cards) < 1):
            raise ValueError("invalid buffers, layout or material route configuration")
        travel = self.distances() / 1.25 if self.walk else np.zeros(stations-1)
        changes = np.stack((setup, setup)).astype(float)
        if self.directional_setup:
            changes[0] *= 1.25  # A to B
            changes[1] *= .75   # B to A
        return (np.asarray(self.buffers, dtype=np.int64), travel, changes,
                self.route_minutes*60, self.bin_units, self.cards, np.asarray(POSTURE[self.posture_aid]))


@njit(cache=True)
def next_bin(available, card_released, previous, interval):
    """Next fixed milk run after an empty card returns; one bin per SKU per run."""
    ready_tick = np.ceil(available / interval)
    card_tick = np.floor(card_released / interval) + 1
    return max(ready_tick, card_tick, previous / interval + 1) * interval

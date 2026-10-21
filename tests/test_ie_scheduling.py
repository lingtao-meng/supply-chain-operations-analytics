"""IE scheduling simulation: dispatching rules, capacity and staffing."""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

spec = importlib.util.spec_from_file_location("ie_scheduling", ROOT / "python" / "04_ie_scheduling.py")
ie_scheduling = importlib.util.module_from_spec(spec)
sys.modules["ie_scheduling"] = ie_scheduling
spec.loader.exec_module(ie_scheduling)

Order = ie_scheduling.Order


class TestDispatching(unittest.TestCase):
    def test_edd_beats_fcfs_on_due_dates(self):
        # A is first in release order but has a later due date than B.
        orders = [
            Order(idx=0, release=0, due=1, units=1, mode="Standard Class"),
            Order(idx=1, release=0, due=0, units=1, mode="Standard Class"),
        ]
        fcfs = ie_scheduling.simulate(orders, capacity=1, rule="FCFS")
        edd = ie_scheduling.simulate(orders, capacity=1, rule="EDD")
        self.assertAlmostEqual(fcfs["on_time_rate"], 0.5)
        self.assertAlmostEqual(edd["on_time_rate"], 1.0)

    def test_capacity_is_respected_every_day(self):
        orders = [Order(idx=i, release=0, due=5, units=2, mode="Standard Class") for i in range(5)]
        result = ie_scheduling.simulate(orders, capacity=3, rule="FCFS")
        self.assertTrue((result["daily"]["processed_units"] <= 3).all())

    def test_tardiness_is_completion_minus_due(self):
        orders = [
            Order(idx=0, release=0, due=0, units=1, mode="Standard Class"),
            Order(idx=1, release=0, due=0, units=1, mode="Standard Class"),
        ]
        result = ie_scheduling.simulate(orders, capacity=1, rule="FCFS")
        # one order is served a day late, the other is on time
        self.assertEqual(result["max_tardiness"], 1)
        self.assertAlmostEqual(result["avg_tardiness"], 0.5)


class TestStaffing(unittest.TestCase):
    def test_workers_round_up_with_utilization(self):
        frame = pd.DataFrame(
            {
                ie_scheduling.DATE_COL: pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                "Order Item Quantity": [50, 50, 50],
            }
        )
        curve = ie_scheduling.staffing_curve(frame, units_per_worker=40, target_utilization=0.85)
        expected = math.ceil(50 / (40 * 0.85))
        self.assertEqual(int(curve["workers_needed"].iloc[-1]), expected)

    def test_rolling_window_smooths_volume(self):
        frame = pd.DataFrame(
            {
                ie_scheduling.DATE_COL: pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "Order Item Quantity": [0, 100],
            }
        )
        curve = ie_scheduling.staffing_curve(frame, units_per_worker=40, target_utilization=0.85)
        self.assertAlmostEqual(curve["rolling_units"].iloc[0], 0.0)
        self.assertAlmostEqual(curve["rolling_units"].iloc[1], 50.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

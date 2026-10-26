"""Line balance and sorting-line simulation tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.hub import line_metrics, load_stations, simulate, with_operator  # noqa: E402


class TestLineBalance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = load_stations(ROOT / "data" / "time_study.csv")

    def test_base_bottleneck_and_balance(self):
        metrics = line_metrics(self.base)
        self.assertEqual(metrics.bottleneck, "induct")
        self.assertAlmostEqual(metrics.bottleneck_cycle, 12.0)
        self.assertAlmostEqual(metrics.capacity_per_hour, 276.0, places=1)
        self.assertAlmostEqual(metrics.balance_rate, 0.75)

    def test_adding_induct_operator_moves_bottleneck(self):
        improved = with_operator(self.base, "induct", 2)
        metrics = line_metrics(improved)
        self.assertEqual(metrics.bottleneck, "load")
        self.assertAlmostEqual(metrics.bottleneck_cycle, 10.0)
        self.assertAlmostEqual(metrics.balance_rate, 0.78)
        self.assertGreater(metrics.capacity_per_hour, line_metrics(self.base).capacity_per_hour)


class TestSimulation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = load_stations(ROOT / "data" / "time_study.csv")
        cls.base_result = simulate(cls.base, 220.0, seed=42)
        cls.peak_result = simulate(cls.base, 286.0, seed=42)
        cls.improved_result = simulate(with_operator(cls.base, "induct", 2), 286.0, seed=42)

    def test_base_is_stable(self):
        self.assertGreater(self.base_result.parcels, 0)
        self.assertLess(self.base_result.avg_flow_min, 5.0)
        self.assertGreater(self.base_result.on_time_rate, 0.95)
        self.assertLess(self.base_result.overtime_min, 5.0)

    def test_peak_breaks_the_line(self):
        self.assertGreater(self.peak_result.avg_flow_min, self.base_result.avg_flow_min)
        self.assertLess(self.peak_result.on_time_rate, self.base_result.on_time_rate)
        self.assertGreater(self.peak_result.backlog.max(), self.base_result.backlog.max())

    def test_one_operator_at_bottleneck_fixes_peak(self):
        self.assertLess(self.improved_result.avg_flow_min, self.peak_result.avg_flow_min)
        self.assertGreater(self.improved_result.on_time_rate, self.peak_result.on_time_rate)
        self.assertLess(self.improved_result.overtime_min, self.peak_result.overtime_min)
        self.assertGreater(self.improved_result.cost_per_parcel, self.peak_result.cost_per_parcel)


if __name__ == "__main__":
    unittest.main(verbosity=2)

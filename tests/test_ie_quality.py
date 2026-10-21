"""IE quality tools: Laney p′ chart, capability, Pareto."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

spec = importlib.util.spec_from_file_location("ie_quality", ROOT / "python" / "03_ie_quality.py")
ie_quality = importlib.util.module_from_spec(spec)
sys.modules["ie_quality"] = ie_quality
spec.loader.exec_module(ie_quality)


class TestWesternElectric(unittest.TestCase):
    def test_one_point_beyond_three_sigma(self):
        z = np.zeros(12)
        z[-1] = 4.0
        flags = ie_quality.western_electric(z)
        self.assertTrue(flags["rule1"][-1])

    def test_eight_consecutive_on_one_side(self):
        z = np.array([0.5] * 9)
        flags = ie_quality.western_electric(z)
        self.assertTrue(flags["rule3"][7])

    def test_two_of_three_beyond_two_sigma(self):
        z = np.array([0.0, 2.5, 2.4, 0.0])
        flags = ie_quality.western_electric(z)
        self.assertTrue(flags["rule2"][2])


class TestLaneyChart(unittest.TestCase):
    def test_limits_and_summary(self):
        rows = []
        for month, late, total in [("2024-01", 50, 100), ("2024-02", 55, 100), ("2024-03", 60, 100)]:
            rows.append(
                pd.DataFrame(
                    {
                        ie_quality.DATE_COL: pd.to_datetime([f"{month}-01"] * total),
                        "delay_days": [1] * late + [0] * (total - late),
                    }
                )
            )
        df = pd.concat(rows, ignore_index=True)
        chart, summary = ie_quality.laney_p_chart(df, freq="M")
        self.assertEqual(len(chart), 3)
        self.assertAlmostEqual(summary["pbar"], 0.55)
        self.assertGreaterEqual(summary["laney_factor"], 1.0)
        self.assertTrue((chart["ucl"] > summary["pbar"]).all())
        self.assertTrue((chart["lcl"] < summary["pbar"]).all())


class TestCapability(unittest.TestCase):
    def test_standard_class_capability(self):
        df = pd.DataFrame(
            {
                "Shipping Mode": ["Standard Class"] * 5,
                ie_quality.SCHED_COL: [4] * 5,
                "transit_days": [2, 3, 4, 5, 6],
            }
        )
        cap = ie_quality.capability(df)
        row = cap.iloc[0]
        self.assertAlmostEqual(row["yield"], 0.6)
        self.assertAlmostEqual(row["ppk"], 0.0, places=6)
        self.assertAlmostEqual(row["dpmo"], 400_000, delta=1)

    def test_zero_variance_mode_is_flagged(self):
        df = pd.DataFrame(
            {
                "Shipping Mode": ["First Class"] * 3,
                ie_quality.SCHED_COL: [1] * 3,
                "transit_days": [2, 2, 2],
            }
        )
        cap = ie_quality.capability(df)
        self.assertEqual(cap.iloc[0]["yield"], 0.0)
        self.assertTrue(np.isnan(cap.iloc[0]["ppk"]))
        self.assertEqual(cap.iloc[0]["note"], "zero variance")


class TestPareto(unittest.TestCase):
    def test_cumulative_share_reaches_one(self):
        df = pd.DataFrame(
            {
                "Shipping Mode": ["A", "A", "B", "C"],
                "delay_days": [1, 2, 3, -1],
            }
        )
        par = ie_quality.pareto(df, "Shipping Mode")
        self.assertAlmostEqual(par["cumulative_share"].iloc[-1], 1.0)
        self.assertEqual(par["late_orders"].tolist(), [2, 1])


if __name__ == "__main__":
    unittest.main(verbosity=2)

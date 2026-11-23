import unittest

import numpy as np

from good_output import FIRST_PASS, REWORK_PASS, REWORK_SCRAP
from rework_carryover import inspect_history, measure_history, recovery_shifts, support_windows
from shift_capacity import SHIFT_SECONDS


class ReworkCarryoverTest(unittest.TestCase):
    def test_exact_cutoff_and_late_units_stay_with_their_origins(self):
        lots = (
            (np.array([SHIFT_SECONDS - 132, SHIFT_SECONDS - 1.]), np.array([REWORK_PASS, FIRST_PASS])),
            (np.array([0., SHIFT_SECONDS - 32.]), np.array([FIRST_PASS, REWORK_PASS])),
            (np.array([]), np.array([], dtype=int)),
        )
        origin, inspection, routes = inspect_history(lots)
        rows, helpers = measure_history(origin, inspection, routes, ("normal", "adverse", "recovery"), 0, 0)
        self.assertEqual(helpers, 0)
        self.assertEqual([row["origin_released_on_time"] for row in rows], [1, 1, 0])
        self.assertEqual([row["physical_released"] for row in rows], [1, 2, 1])
        self.assertEqual([row["late_released"] for row in rows], [0, 1, 1])
        self.assertEqual([row["pending_inspection"] for row in rows], [1, 0, 0])
        self.assertEqual([row["pending_rework"] for row in rows], [0, 1, 0])
        self.assertEqual(recovery_shifts(rows), 1)

    def test_rework_spans_net_cutoff_and_failed_unit_is_final_disposition(self):
        origin, inspection, routes = inspect_history((
            (np.array([SHIFT_SECONDS - 72.]), np.array([REWORK_SCRAP])),
            (np.array([]), np.array([], dtype=int)),
        ))
        rows, _ = measure_history(origin, inspection, routes, ("adverse", "recovery"), 0, 0)
        self.assertEqual([row["pending_rework"] for row in rows], [1, 0])
        self.assertEqual([row["scrap"] for row in rows], [0, 1])
        self.assertEqual([row["origin_released_on_time"] for row in rows], [0, 0])
        self.assertEqual(recovery_shifts(rows), 1)

    def test_two_helper_shapes_use_the_same_minutes_only_in_adverse_shifts(self):
        phases = ("normal", "adverse", "adverse", "recovery")
        one = support_windows(phases, 1, 240)
        two = support_windows(phases, 2, 120)
        self.assertEqual(len(one), 2)
        self.assertEqual(len(two), 4)
        self.assertEqual(sum(duration for _, duration in one), sum(duration for _, duration in two))
        self.assertEqual({start + duration for start, duration in two}, {2 * SHIFT_SECONDS, 3 * SHIFT_SECONDS})

    def test_saturated_core_and_short_help_leave_predictable_daily_growth(self):
        origin = np.repeat(np.arange(3), 273)
        inspection = 1. + origin * SHIFT_SECONDS
        routes = np.full(len(origin), REWORK_PASS)
        rows, helper_jobs = measure_history(origin, inspection, routes, ("adverse",) * 3 + ("recovery",), 1, 90)
        self.assertEqual(helper_jobs, 3 * 45)
        self.assertEqual([row["pending_rework"] for row in rows], [19, 37, 55, 0])
        self.assertEqual(recovery_shifts(rows), 1)

    def test_recovery_reports_zero_and_censoring(self):
        rows = [{"phase": "adverse", "pre_recovery_origin_pending": 0},
                {"phase": "recovery", "pre_recovery_origin_pending": 0}]
        self.assertEqual(recovery_shifts(rows), 0)
        for row in rows:
            row["pre_recovery_origin_pending"] = 1
        self.assertIsNone(recovery_shifts(rows))


if __name__ == "__main__":
    unittest.main()

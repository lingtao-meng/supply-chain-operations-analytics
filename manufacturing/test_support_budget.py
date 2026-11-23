import unittest

import numpy as np

from crew_call import decision
from repair_space_budget import peak_waiting
from shift_dispatch import SHIFT_SECONDS, ADVERSE, trace_windows, HANDOFF, REPAIR
from staggered_support import crew_times, simulate
from support_budget import choose_promise, describe, independent_repairs


class SupportBudgetTests(unittest.TestCase):
    def test_joint_gate_counts_pairs(self):
        result = describe(np.array([[1700, 1690, 1500], [1400, 1700, 1710]]), 1650, .05)
        self.assertEqual(result["each_shift"], [2, 2])
        self.assertEqual(result["joint_hits"], 1)
        self.assertFalse(result["accepted"])

    def test_extra_worker_can_start_third_repair_at_the_same_time(self):
        work = np.ones((120, 1))
        offers = np.full(120, SHIFT_SECONDS)
        routes = np.repeat(2, 120)
        duration = 80*60
        fallback = np.repeat((np.array(ADVERSE)+1)*SHIFT_SECONDS-duration, 3)
        initial = trace_windows(work, routes, offers, fallback, duration, 0, 12)
        trace, starts = simulate(work, routes, offers, (6, 10, 0), initial, duration, 3)
        self.assertEqual(len(starts), 6)
        self.assertEqual(sum(trace[:, REPAIR] == starts[0]), 3)
        self.assertIn("通知3 人", decision(125, 6, 100, 10, (6,10,0),80,3))

    def test_occupied_places_exclude_workers_and_blocked_inspector(self):
        trace = np.full((4, 7), np.nan)
        trace[:, HANDOFF] = [2, 5, 10, 11]
        trace[:, REPAIR] = [5, 10, 10, 12]
        self.assertEqual(peak_waiting(trace, 0, 10), 1)
        self.assertEqual(peak_waiting(trace, 10, 10), 0)
        self.assertIn("通知两人", decision(120, 15, 100, 10, (6,10,0), repair_slots=24))

    def test_design_quantile_and_cap(self):
        self.assertEqual(choose_promise(np.r_[100, 200, np.repeat(1578, 98)]), 1570)
        self.assertEqual(choose_promise(np.repeat(1720, 100)), 1650)

    def test_perfect_repairs_still_require_first_inspection_and_repair_time(self):
        offers = np.array([SHIFT_SECONDS, 2*SHIFT_SECONDS])
        work = np.array([[SHIFT_SECONDS-132.], [SHIFT_SECONDS-131.]])
        self.assertEqual(independent_repairs(work, np.array([2, 2]), offers), [1, 0])
        self.assertEqual(independent_repairs(work, np.array([1, 0]), offers), [0, 1])

    def test_longer_reservation_fits_inside_shift_even_without_signal(self):
        trace = np.empty((0, 7))
        starts = crew_times(trace, np.array([]), 1, (6, 10, 0), 150*60)
        self.assertEqual(starts, (SHIFT_SECONDS + 270*60,)*2)
        self.assertIn("第 270 分钟同时到岗，第 420 分钟结束", decision(260,0,0,0,(6,10,0),150))
        with self.assertRaises(ValueError):
            decision(265,0,0,0,(6,10,0),150)


if __name__ == "__main__":
    unittest.main()

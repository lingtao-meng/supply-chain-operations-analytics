import unittest

import numpy as np

from crew_call import decision
from shift_dispatch import (ADVERSE, DAYS, DURATION, END, FALLBACK, INSPECT, LOT, REPAIR, SHIFT_SECONDS,
                            run_policy, trace_windows)
from staggered_support import BASELINE, crew_times, simulate


class SplitCrewTests(unittest.TestCase):
    def test_action_matches_selected_rule_and_fallback(self):
        policy = (6, 10, 0)
        self.assertIn("第 135 分钟同时到岗", decision(125, 7, 100, 11, policy))
        self.assertIn("复查", decision(125, 7, 39, 11, policy))
        self.assertIn("第 300 分钟同时到岗", decision(290, 0, 0, 0, policy))
        with self.assertRaises(ValueError):
            decision(120, 3, 7, 8, policy)

    def test_individual_windows_accept_only_tasks_that_fit(self):
        work = np.ones((6, 1))
        routes = np.full(6, 2)
        trace = trace_windows(work, routes, np.zeros(6), np.array([25, 55]), 120, 0, 12)
        np.testing.assert_equal(trace[:4, REPAIR], [13, 25, 55, 133])
        np.testing.assert_equal(trace[:4, END], [133, 145, 175, 253])

    def test_quality_signal_uses_only_finished_current_batch_inspections(self):
        origin = SHIFT_SECONDS
        offers = np.r_[np.zeros(40), np.full(70, origin)]
        trace = np.full((110, 7), np.nan)
        trace[:, INSPECT] = origin - 500 + np.arange(110)*12
        trace[:40, REPAIR] = origin + 10000 + np.arange(40)*120
        self.assertEqual(crew_times(trace, offers, 1, (2, 6, 20)), (origin+FALLBACK, origin+FALLBACK))
        trace[40:, REPAIR] = origin + 20000 + np.arange(70)*120
        first, second = crew_times(trace, offers, 1, (2, 6, 20))
        self.assertEqual(first-origin, 1200)  # first poll with 40 current inspections is at minute 10
        self.assertEqual(second-first, 1200)
        trace[trace[:, INSPECT] > first-600, REPAIR] = np.nan
        self.assertEqual(crew_times(trace, offers, 1, (2, 6, 20)), (first,second))

    def test_baseline_replay_and_budget_at_fallback(self):
        rng = np.random.default_rng(503)
        offers = np.repeat(np.arange(DAYS)*SHIFT_SECONDS, LOT)
        work = rng.uniform(6, 14, (len(offers), 3))
        routes = rng.choice([0,1,2,3], len(offers), p=[.84,.03,.11,.02])
        fallback = np.repeat(np.array(ADVERSE)*SHIFT_SECONDS + FALLBACK, 2)
        base = trace_windows(work, routes, offers, fallback, DURATION, 0, 12)
        trace, starts = simulate(work, routes, offers, BASELINE, base)
        old, original_starts, _ = run_policy(work, routes, offers, "queue", 8)
        np.testing.assert_equal(trace, old)
        np.testing.assert_equal(starts, np.repeat(original_starts,2))
        never = crew_times(base, offers, 1, (40, 40, 30))
        self.assertEqual(never,(SHIFT_SECONDS+FALLBACK,)*2)


if __name__ == "__main__":
    unittest.main()

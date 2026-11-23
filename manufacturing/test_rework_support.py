from unittest import TestCase, main

import numpy as np

from good_output import DIRECT_SCRAP, FIRST_PASS, REWORK_PASS, REWORK_SCRAP
from rework_support import concentrate_failures, dispatch_rework, observed_waiting, support_start


class ReworkSupportTests(TestCase):
    def test_pulse_preserves_counts_and_moves_only_failure_timing(self):
        routes = np.array([FIRST_PASS, REWORK_PASS, FIRST_PASS, DIRECT_SCRAP, FIRST_PASS,
                           REWORK_SCRAP, REWORK_PASS, DIRECT_SCRAP, FIRST_PASS, FIRST_PASS])
        times = np.arange(len(routes)) * 30 * 60
        shifted = concentrate_failures(routes, times, 60, np.arange(len(routes)), routes[routes != FIRST_PASS])
        np.testing.assert_array_equal(np.bincount(shifted), np.bincount(routes))
        self.assertEqual(int(np.count_nonzero(shifted[2:6] != FIRST_PASS)), 3)

    def test_helper_finishes_at_cutoff_but_never_overtimes(self):
        arrivals = np.zeros(4)
        finishes, helper_jobs = dispatch_rework(arrivals, start=80, duration=120)
        np.testing.assert_array_equal(finishes, [120, 200, 240, 360])
        self.assertEqual(helper_jobs, 1)
        finishes, helper_jobs = dispatch_rework(arrivals, start=80, duration=119)
        np.testing.assert_array_equal(finishes, [120, 240, 360, 480])
        self.assertEqual(helper_jobs, 0)

    def test_waiting_excludes_active_job_and_trigger_uses_only_observed_prefix(self):
        checks = np.array([0, 300, 600, 900])
        first = np.array([0, 1, 2, 3, 1000])
        second = np.array([0, 1, 2, 3, 301, 302, 303, 304])
        waiting = observed_waiting(first, checks)
        np.testing.assert_array_equal(waiting[:2], [0, 1])
        self.assertEqual(support_start(checks, waiting, threshold=1), (900, True))
        self.assertEqual(support_start(checks, observed_waiting(second, checks), threshold=1), (900, True))
        self.assertEqual(support_start(checks, waiting, threshold=5), (330 * 60, False))


if __name__ == "__main__":
    main()

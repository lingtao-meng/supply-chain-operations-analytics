from unittest import TestCase, main

import numpy as np

from peak_rework_support import PERSON_MINUTES, candidate_windows
from rework_support import dispatch_rework_windows
from shift_capacity import SHIFT_SECONDS


class PeakReworkTests(TestCase):
    def test_candidate_windows_use_equal_person_minutes(self):
        for people in (1, 2):
            for policy in candidate_windows(people):
                self.assertEqual(policy.support_people * policy.support_minutes, PERSON_MINUTES)
                self.assertLessEqual(60 * (policy.start_minute + policy.support_minutes), SHIFT_SECONDS)

    def test_concurrent_slots_address_a_late_burst(self):
        arrivals = np.full(150, 300 * 60)
        long, _ = dispatch_rework_windows(arrivals, ((180 * 60, 240 * 60),))
        peak, _ = dispatch_rework_windows(arrivals, ((300 * 60, 120 * 60),) * 2)
        self.assertEqual(np.count_nonzero(long <= SHIFT_SECONDS), 120)
        self.assertEqual(np.count_nonzero(peak <= SHIFT_SECONDS), 150)
        self.assertEqual(peak[-1], 400 * 60)

    def test_both_allocations_provide_equal_helper_capacity_with_backlog(self):
        arrivals = np.zeros(1000)
        for windows in (((180 * 60, 240 * 60),), ((300 * 60, 120 * 60),) * 2):
            _, helper_jobs = dispatch_rework_windows(arrivals, windows)
            self.assertEqual(helper_jobs, 120)

    def test_future_block_is_used_only_after_the_earlier_block_expires(self):
        arrivals = np.zeros(5)
        finishes, helper_jobs = dispatch_rework_windows(arrivals, ((0, 120), (180, 120)))
        np.testing.assert_array_equal(finishes, [120, 120, 240, 300, 360])
        self.assertEqual(helper_jobs, 2)


if __name__ == "__main__":
    main()

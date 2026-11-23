import unittest

import numpy as np

from finite_staging_kernel import SHIFT_FIELDS, simulate_finite_shift
from shift_dispatch import (ADVERSE, DAYS, DEMAND, DURATION, END, ENTRY, FALLBACK, FIRST_PASS, HANDOFF, HELPER,
                            INSPECT, LOT, REPAIR, REWORK_PASS, SHIFT_SECONDS, TRANSFER, observed_start,
                            run_policy, shift_counts, trace_line)


class ShiftDispatchTests(unittest.TestCase):
    def test_block_and_unfinished_service_across_boundary(self):
        # These three arrive 30 seconds before handover. First repair has 107 s left.
        origin = SHIFT_SECONDS - 30
        trace = trace_line(np.full((3, 1), 5.), np.array([2, 2, 0]), np.full(3, origin),
                           np.array([], dtype=int), 0, 0, 0)
        np.testing.assert_equal(trace[:, ENTRY] - origin, [0, 5, 17])
        np.testing.assert_equal(trace[:, TRANSFER] - origin, [5, 17, 137])
        np.testing.assert_equal(trace[:, INSPECT] - origin, [17, 29, 149])
        np.testing.assert_equal(trace[:, HANDOFF] - origin, [17, 137, 149])
        np.testing.assert_equal(trace[:, END] - origin, [137, 257, 149])

    def test_fixed_calendar_matches_existing_single_shift_oracle(self):
        rng = np.random.default_rng(264)
        for slots in (0, 3, 12):
            for _ in range(15):
                work = rng.integers(5, 12, (180, 3)).astype(float)
                routes = rng.integers(0, 4, 180)
                expected = dict(zip(SHIFT_FIELDS, simulate_finite_shift(work, routes, np.ones(2, dtype=int),
                                                                        0, slots, 750, 2, 210, 360)))
                trace = trace_line(work, routes, np.zeros(180), np.array([210]), 360, 0, slots)
                self.assertEqual((trace[:, TRANSFER] <= 750).sum(), expected["assembly_transferred"])
                self.assertEqual(((trace[:, END] <= 750) & np.isin(routes, [0, 2])).sum(), expected["released"])
                self.assertEqual(((trace[:, HANDOFF] <= 750) & (trace[:, END] > 750)).sum(),
                                 expected["pending_rework"])

    def test_queue_dispatch_uses_only_observations_before_call(self):
        jobs = LOT * DAYS
        offers = np.repeat(np.arange(DAYS)*SHIFT_SECONDS, LOT)
        work = np.full((jobs, 2), 10.)
        routes = np.full(jobs, REWORK_PASS)
        trace, starts, signals = run_policy(work, routes, offers, "queue", 6)
        self.assertTrue(all(signals))
        for day, start in zip(ADVERSE, starts):
            self.assertEqual(observed_start(trace, day, 6), (start, True))
        address = starts[0] - 600
        changed = routes.copy()
        changed[trace[:, INSPECT] > address] = FIRST_PASS
        _, modified, _ = run_policy(work, changed, offers, "queue", 6)
        self.assertEqual(modified[0], starts[0])
        pending = 0
        for day in range(DAYS):
            row = shift_counts(trace, routes, offers, day)
            self.assertEqual(pending + LOT, row["backlog"] + row["physical_good"] + row["scrap"])
            self.assertLessEqual(row["helper_busy_min"], 240 if day in ADVERSE else 0)
            pending = row["backlog"]

    def test_fallback_and_exact_deadline(self):
        work = np.ones((4, 1))
        offers = np.zeros(4)
        trace = trace_line(work, np.full(4, REWORK_PASS), offers, np.array([25]), 120, 0, 12)
        np.testing.assert_equal(trace[:, HELPER], [0, 1, 0, 0])
        self.assertEqual(trace[1, END], 145)
        self.assertEqual(observed_start(trace, 0, 12), (FALLBACK, False))


if __name__ == "__main__":
    unittest.main()

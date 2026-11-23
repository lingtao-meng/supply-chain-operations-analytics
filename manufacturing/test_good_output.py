from unittest import TestCase, main

import numpy as np

from good_output import Policy, fifo_finishes, quality_shift
from shift_capacity import shift_exit_times, shift_output_counts


class GoodOutputTests(TestCase):
    def test_line_exits_match_counts_including_exact_cutoff(self):
        work = np.full((7, 2, 2), 1.0)
        work[:, :, 1] = 1.5
        exits = shift_exit_times(work, (1,), 4)
        np.testing.assert_array_equal(exits[:4, 0], [2, 3, 4, np.inf])
        np.testing.assert_array_equal(exits[:3, 1], [3, np.inf, np.inf])
        np.testing.assert_array_equal(np.isfinite(exits).sum(axis=0), shift_output_counts(work, (1,), 4))

    def test_parallel_fifo_handles_simultaneous_arrivals(self):
        arrivals = np.array([0, 0, 1, 1, 6])
        np.testing.assert_array_equal(fifo_finishes(arrivals, 2, 4), [4, 4, 8, 8, 12])

    def test_quality_conserves_units_at_cutoff(self):
        arrivals = np.array([0, 0, 0, 1, 144, 145])
        draws = np.array([
            [.9, .1, .1], [.1, .1, .1], [.1, .9, .1],
            [.1, .1, .9], [.9, .1, .1], [.9, .1, .1],
        ])
        self.assertEqual(quality_shift(arrivals, draws, Policy("mixed", 1, 50), 144), {
            "gross": 5, "first_inspected": 4, "first_pass": 1, "routed_to_rework": 2,
            "rework_pass": 1, "direct_scrap": 1, "rework_scrap": 0, "scrap": 1,
            "pending_inspection": 1, "pending_rework": 1, "released": 2,
        })

    def test_failed_rework_is_scrap_only_after_processing(self):
        arrivals = np.array([0, 0])
        draws = np.array([[.1, .1, .99], [.1, .1, .1]])
        result = quality_shift(arrivals, draws, Policy("base", 1, 50), 144)
        self.assertEqual((result["scrap"], result["pending_rework"], result["rework_pass"]), (1, 1, 0))
        result = quality_shift(arrivals, draws, Policy("extra", 1, 50, rework_servers=2), 144)
        self.assertEqual((result["scrap"], result["pending_rework"], result["rework_pass"]), (1, 0, 1))


if __name__ == "__main__":
    main()

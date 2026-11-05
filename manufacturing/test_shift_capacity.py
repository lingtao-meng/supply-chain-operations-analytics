import unittest

import numpy as np

from shift_capacity import buffer_layout, choose_design, deterministic_output, reliability_lower, simulate_shift


class TestShiftCapacity(unittest.TestCase):
    def test_deterministic_startup_and_exact_shift_cutoff(self):
        service = np.broadcast_to(np.array([3, 5, 2])[None, :, None], (20, 3, 2))
        for layout in ((0, 0), (0, 2), (2, 0), (1, 2)):
            counts, state = simulate_shift(service, layout, 25)
            np.testing.assert_array_equal(counts, [deterministic_output([3, 5, 2], 25)] * 2)
            np.testing.assert_array_equal(counts, [4, 4])
            np.testing.assert_allclose(state.sum(axis=2), 1)
        np.testing.assert_array_equal(simulate_shift(service, (0, 0), 24.99)[0], [3, 3])

    def test_finite_buffer_releases_upstream_station(self):
        # The first job occupies station 2 until time 6. Job 2 can wait in a buffer
        # at time 2; that lets station 1 finish the long third job at time 7 vs 11.
        service = np.asarray([[1, 5], [1, 1], [5, 1], [10, 10]])[:, :, None]
        no_buffer, blocked = simulate_shift(service, (0,), 8)
        one_slot, buffered = simulate_shift(service, (1,), 8)
        np.testing.assert_array_equal(no_buffer, [2])
        np.testing.assert_array_equal(one_slot, [3])
        self.assertAlmostEqual(blocked[0, 0, 1], 4 / 8)
        self.assertEqual(buffered[0, 0, 1], 0)

    def test_short_workload_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "supply more"):
            simulate_shift(np.ones((3, 2, 1)), (0,), 20)

    def test_buffer_budget_and_reliability_screen(self):
        self.assertEqual(sum(buffer_layout([13, 14, 11, 14, 14, 14, 13, 12], "targeted_1")), 7)
        self.assertGreater(reliability_lower(178, 180), 0.95)
        self.assertLess(reliability_lower(115, 120), 0.95)
        first = {"method": "MILP", "staffed_stations": 8, "p05_output_units": 1660}
        self.assertEqual(choose_design([
            {**first, "policy": "none", "total_buffer": 0, "hit_lower95": 0.98},
            {**first, "policy": "none", "total_buffer": 0, "hit_lower95": 0.94},
            {**first, "policy": "uniform_1", "total_buffer": 7, "hit_lower95": 0.96},
            {**first, "policy": "uniform_2", "total_buffer": 14, "hit_lower95": 0.98},
        ]), ("MILP", "uniform_1"))


if __name__ == "__main__":
    unittest.main(verbosity=2)

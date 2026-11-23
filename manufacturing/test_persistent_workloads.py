from math import sqrt
from unittest import TestCase, main

import numpy as np

from persistent_workloads import capacity_target, correlated_noise, hit_result, planned_target


class PersistentWorkloadTests(TestCase):
    def test_independence_is_the_existing_unit_noise(self):
        innovations = np.arange(12.0).reshape(3, 2, 2)
        np.testing.assert_array_equal(correlated_noise(innovations, 0.0), innovations)

    def test_stationary_start_and_product_axis_recurrence(self):
        innovations = np.arange(24.0).reshape(4, 2, 3) / 5
        rho = 0.7
        correlated = correlated_noise(innovations, rho)
        np.testing.assert_allclose(correlated[0], innovations[0])
        np.testing.assert_allclose(
            correlated[1:], rho * correlated[:-1] + sqrt(1 - rho**2) * innovations[1:],
        )

    def test_capacity_boundary_keeps_ties_and_uses_integer_units(self):
        counts = np.array([101, 100, 100, 104, 99])
        self.assertEqual(capacity_target(counts, 4), 100)
        self.assertEqual(capacity_target(counts, 5), 99)
        self.assertEqual(planned_target(1700), 1680)

    def test_empirical_rate_is_not_the_adjusted_acceptance_gate(self):
        counts = np.repeat([1000, 900], [570, 30])
        result = hit_result(counts, 1000, 0.05 / 24)
        self.assertEqual(result["hits"], 570)
        self.assertEqual(result["mean_shortfall"], 5.0)
        self.assertFalse(result["passes"])


if __name__ == "__main__":
    main()

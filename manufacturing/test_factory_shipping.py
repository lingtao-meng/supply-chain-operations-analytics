import unittest
from dataclasses import replace

import numpy as np

from factory_shipping import Outbound, dispatch
from shift_capacity import SHIFT_SECONDS


class FactoryShippingTests(unittest.TestCase):
    def test_cutoff_is_inclusive_and_scrap_cannot_ship(self):
        options = Outbound((20, 40), (4, 4), cutoff_minutes=10)
        shipped = dispatch([600, 601, 300, 1801], [0] * 4, [True, True, False, True], options, shifts=1)
        np.testing.assert_array_equal(shipped, [1200, 2400, np.inf, np.inf])

    def test_late_cohort_keeps_its_priority_in_next_shift(self):
        options = Outbound((10,), (1,), cutoff_minutes=1)
        released = np.array([SHIFT_SECONDS + 100, SHIFT_SECONDS + 1, SHIFT_SECONDS + 2])
        orders = np.array([0, SHIFT_SECONDS, SHIFT_SECONDS])
        shipped = dispatch(released, orders, [True] * 3, options, shifts=3)
        np.testing.assert_array_equal(shipped, [SHIFT_SECONDS + 600, 2 * SHIFT_SECONDS + 600, np.inf])
        self.assertGreater(shipped[0], orders[0] + SHIFT_SECONDS)

    def test_capacity_and_stable_priority_at_the_last_truck(self):
        options = Outbound((420,), (2,), cutoff_minutes=10)
        ready = np.array([20000, 10000, 10000, 24900, np.inf, 17000])
        good = np.array([True, True, True, True, True, False])
        shipped = dispatch(ready, np.zeros(6), good, options, shifts=1)
        np.testing.assert_array_equal(shipped, [np.inf, SHIFT_SECONDS, SHIFT_SECONDS, np.inf, np.inf, np.inf])
        expanded = dispatch(ready, np.zeros(6), good, replace(options, capacities=(3,)), shifts=1)
        np.testing.assert_array_equal(expanded, [SHIFT_SECONDS, SHIFT_SECONDS, SHIFT_SECONDS, np.inf, np.inf, np.inf])

    def test_more_space_does_not_admit_a_unit_after_the_manifest_cutoff(self):
        options = Outbound((420,), (3,), cutoff_minutes=15)
        ready = [SHIFT_SECONDS - 800]
        self.assertTrue(np.isinf(dispatch(ready, [0], [True], options, shifts=1)[0]))
        cutover = replace(options, cutoff_minutes=10)
        self.assertEqual(dispatch(ready, [0], [True], cutover, shifts=1)[0], SHIFT_SECONDS)

    def test_invalid_calendar_and_trace(self):
        with self.assertRaises(ValueError):
            Outbound((245, 420), (650,))
        with self.assertRaises(ValueError):
            Outbound((420, 365), (650, 650))
        with self.assertRaises(ValueError):
            dispatch([40], [60], [True], shifts=1)
        with self.assertRaises(ValueError):
            dispatch([40], [0], [1], shifts=1)


if __name__ == "__main__":
    unittest.main()

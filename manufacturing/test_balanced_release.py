import unittest

import numpy as np

from factory_simulation import SHIFT_SECONDS, _select_balanced


class BalancedReleaseTests(unittest.TestCase):
    def test_batch_is_respected_then_the_lagging_sku_is_selected(self):
        heads = np.array([0, 1])
        orders = np.zeros(2, dtype=int)
        ready = np.zeros(2, dtype=float)
        cohort = np.zeros(2, dtype=int)
        released = np.array([[80, 12]])
        self.assertEqual(_select_balanced(heads, 10, orders, ready, 0, 19, 20, cohort, released, 0)[0], 0)
        self.assertEqual(_select_balanced(heads, 10, orders, ready, 0, 20, 20, cohort, released, 0)[0], 1)

    def test_ready_older_order_and_material_gate_take_priority(self):
        orders = np.array([0, SHIFT_SECONDS])
        released = np.zeros((2, 2), dtype=int)
        heads = np.array([0, 1])
        free = SHIFT_SECONDS+8
        self.assertEqual(_select_balanced(heads, free, orders, orders.astype(float), 1, 1, 20,
                                          np.array([0, 1]), released, 1)[0], 0)
        self.assertEqual(_select_balanced(heads, free, orders, np.array([free+1, SHIFT_SECONDS], dtype=float),
                                          0, 1, 20, np.array([0, 1]), released, 0)[0], 1)

    def test_previous_batches_do_not_carry_over_across_order_cohorts(self):
        orders = np.full(2, SHIFT_SECONDS)
        released = np.array([[500, 1], [5, 18]])
        heads = np.array([0, 1])
        self.assertEqual(_select_balanced(heads, SHIFT_SECONDS+8, orders, orders.astype(float),
                                          1, 3, 20, np.array([1, 1]), released, 0)[0], 0)

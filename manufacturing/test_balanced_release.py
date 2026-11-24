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

    def test_changed_order_mix_can_reverse_the_fixed_ratio_choice(self):
        heads = np.array([0, 1])
        orders = np.zeros(2, dtype=int)
        ready = orders.astype(float)
        released = np.array([[80, 50]])
        cohort = np.zeros(2, dtype=int)
        args = (heads, 10, orders, ready, 0, 20, 20, cohort, released, 0)
        self.assertEqual(_select_balanced(*args)[0], 0)
        self.assertEqual(_select_balanced(*args, np.array([[925, 925]]))[0], 1)
        self.assertEqual(_select_balanced(*args, np.array([[1295, 555]]))[0], 0)

    def test_changed_mix_cannot_release_unready_model(self):
        heads = np.array([0, 1])
        orders = np.zeros(2, dtype=int)
        ready = np.array([0., 50.])
        released = np.array([[120, 0]])
        cohort = np.zeros(2, dtype=int)
        demand = np.array([[925, 925]])
        self.assertEqual(_select_balanced(heads, 10, orders, ready, 0, 120, 120,
                                          cohort, released, 0, demand)[0], 0)

    def test_order_mix_band_defers_small_changes_and_corrects_larger_ones(self):
        heads = np.array([0, 1])
        orders = np.zeros(2, dtype=int)
        cohort = np.zeros(2, dtype=int)
        demand = np.array([[925, 925]])
        released = np.array([[160, 130]])
        args = (heads, 10, orders, orders.astype(float), 0, 120, 120, cohort, released, 0, demand)
        self.assertEqual(_select_balanced(*args, 0)[0], 1)
        self.assertEqual(_select_balanced(*args, 15)[0], 0)
        self.assertEqual(_select_balanced(*args, 14)[0], 1)

    def test_order_mix_band_does_not_bypass_a_ready_older_cohort(self):
        heads = np.array([0, 1])
        orders = np.array([0, SHIFT_SECONDS])
        cohort = np.array([0, 1])
        demand = np.array([[925, 925], [1480, 370]])
        released = np.array([[160, 140], [160, 130]])
        self.assertEqual(_select_balanced(heads, SHIFT_SECONDS + 10, orders, orders.astype(float),
                                          1, 120, 120, cohort, released, 1, demand, 50)[0], 0)

import unittest
from types import SimpleNamespace

import numpy as np

from factory_inputs import SHIFTS, SHIFT_SECONDS
from factory_simulation import ASSEMBLED, END, ENTRY, HANDOFF, HELPER, INSPECTED, Plant
from factory_study import measure, peak_overlap
from good_output import DIRECT_SCRAP, FIRST_PASS


class FactoryStudyTests(unittest.TestCase):
    def test_cutoff_uses_original_batch_and_reconciles_late_product(self):
        orders = np.repeat(np.arange(SHIFTS) * SHIFT_SECONDS, 2)
        trace = np.full((len(orders), 9), np.nan)
        trace[:, HELPER] = -1
        trace[:, ENTRY] = orders + 1
        trace[:, ASSEMBLED] = orders + 2
        fixed = orders + 14
        fixed[0] = SHIFT_SECONDS
        fixed[1] = SHIFT_SECONDS + 1
        trace[:, INSPECTED] = trace[:, HANDOFF] = trace[:, END] = fixed
        routes = np.full(len(orders), FIRST_PASS)
        routes[2] = DIRECT_SCRAP
        scenario = SimpleNamespace(sku=np.tile([0, 1], SHIFTS), orders=orders, kits=orders, routes=routes)
        model = SimpleNamespace(
            trace=trace, booked=np.zeros(SHIFTS), kit_wait=np.zeros(SHIFTS),
            staffing_windows=np.empty((0, 2)), setups=np.zeros((SHIFTS, 1)),
            changeovers=np.zeros((SHIFTS, 1)),
        )
        first, second, *_ = measure(scenario, model, Plant())
        self.assertEqual((first["on_time_a"], first["on_time_b"], first["backlog"]), (1, 0, 1))
        self.assertEqual((second["on_time_b"], second["late_good"], second["scrap"], second["backlog"]), (1, 1, 1, 0))

    def test_cart_and_fixture_intervals_use_half_open_events(self):
        begin, end = np.array([1, 2, 4]), np.array([3, 4, 5])
        self.assertEqual(peak_overlap(begin, end, 0, 6), 2)
        self.assertEqual(peak_overlap(begin, end, 3, 4), 1)
        self.assertEqual(peak_overlap(begin, end, 4, 5), 1)


if __name__ == "__main__":
    unittest.main()

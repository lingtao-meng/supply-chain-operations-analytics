import unittest
from types import SimpleNamespace

import numpy as np

from factory_receiving import ReceivingPlan, receive_kits
from shift_capacity import SHIFT_SECONDS


def scenario(sku, kits, orders=None):
    return SimpleNamespace(
        sku=np.array(sku),
        kits=np.array(kits),
        orders=np.zeros(len(kits), dtype=int) if orders is None else np.array(orders),
    )


class FactoryReceivingTests(unittest.TestCase):
    def test_stock_bypasses_inspection_but_receipts_queue_by_bench(self):
        source = scenario([0, 0, 1, 1, 0], [0, 60, 60, 60, 120])
        one, lots = receive_kits(source, ReceivingPlan(lanes=1, minutes_a=2, minutes_b=3))
        two, parallel = receive_kits(source, ReceivingPlan(lanes=2, minutes_a=2, minutes_b=3))
        np.testing.assert_array_equal(one, [0, 180, 360, 360, 480])
        np.testing.assert_array_equal(two, [0, 180, 240, 240, 300])
        self.assertEqual([(lot.sku, lot.units, lot.start) for lot in lots], [(0, 1, 60), (1, 2, 180), (0, 1, 360)])
        self.assertEqual([lot.lane for lot in parallel], [0, 1, 0])

    def test_advance_reduces_arrival_and_preserves_distinct_inbound_lots(self):
        source = scenario([0, 0, 0], [0, 60, 120])
        eligible, lots = receive_kits(source, ReceivingPlan(minutes_a=2), advance_minutes=5)
        np.testing.assert_array_equal(eligible, [0, 120, 240])
        self.assertEqual([lot.arrival for lot in lots], [0, 0])
        self.assertEqual([lot.released for lot in lots], [120, 240])

    def test_pending_inspection_keeps_its_lot_and_carries_across_shift(self):
        source = scenario(
            [0, 1, 1],
            [SHIFT_SECONDS-60, SHIFT_SECONDS, SHIFT_SECONDS+60],
            [0, SHIFT_SECONDS, SHIFT_SECONDS],
        )
        eligible, lots = receive_kits(source, ReceivingPlan(minutes_a=2, minutes_b=3))
        np.testing.assert_array_equal(eligible, [SHIFT_SECONDS+60, SHIFT_SECONDS, SHIFT_SECONDS+240])
        self.assertEqual([lot.cohort for lot in lots], [0, SHIFT_SECONDS])
        self.assertEqual(lots[1].start, lots[0].finished)
        self.assertEqual(lots[1].released, eligible[2])

    def test_disabled_inspection_matches_advanced_delivery_time(self):
        source = scenario([0, 1], [0, 300])
        eligible, lots = receive_kits(source, None, advance_minutes=2)
        np.testing.assert_array_equal(eligible, [0, 180])
        self.assertEqual(lots, ())


if __name__ == "__main__":
    unittest.main()

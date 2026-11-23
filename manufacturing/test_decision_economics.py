import unittest

import numpy as np

from decision_economics import paired_hits, shift_cost
from shift_capacity import _flow_shop

class DecisionEconomicsTests(unittest.TestCase):
    def test_joint_recording_keeps_exit_times_and_buffer_occupancy(self):
        # Work reaches a slower station; the single waiting slot is used.
        service = np.tile(np.array([3., 5.])[None,:,None], (10,1,1))
        outage = (1, 7., 10.)
        times = _flow_shop(service, (1,), 25., outage=outage, record_exits=True)
        units, occupied = _flow_shop(service, (1,), 25., outage=outage, measure_buffers=True)
        combined_times, combined_occupied = _flow_shop(
            service, (1,), 25., outage=outage, record_exits=True, measure_buffers=True)
        np.testing.assert_array_equal(combined_times, times)
        np.testing.assert_array_equal(combined_occupied, occupied)
        self.assertEqual(np.isfinite(times).sum(), units[0])
        self.assertGreater(occupied[0], 0)

    def test_paired_rate_has_uncertainty_even_with_no_observed_changes(self):
        hits = np.ones(30, dtype=bool)
        effect = paired_hits(hits, hits, 14)
        self.assertEqual(effect["mean"], 0)
        self.assertIsNone(effect["dz"])
        self.assertLess(effect["simultaneous95"][0], effect["ci95"][0])
        self.assertLess(effect["ci95"][0], 0)
        self.assertGreater(effect["ci95"][1], 0)

    def test_cost_uses_fixed_space_and_time_average_occupancy_separately(self):
        rates = dict(shifts_per_year=250, assembly_hourly_rate=38, buffer_space_per_slot_year=900,
                     wip_value_per_unit=260, annual_holding_rate=.2, direct_downtime_cost_per_minute=8,
                     repair_per_completed_attempt=6, recheck_per_completed_attempt=2)
        base = shift_cost(rates,8,7,2,15,10)
        self.assertAlmostEqual(base, 8*7*38 + 7*900/250 + 2*260*.2/250 + 15*8 + 10*8)
        self.assertAlmostEqual(shift_cost(rates,9,8,2,15,10)-base, 7*38+900/250)

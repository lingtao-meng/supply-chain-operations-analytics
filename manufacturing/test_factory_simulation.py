import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from factory_receiving import ReceivingPlan, receive_kits
from factory_shipping import Outbound, dispatch
from factory_simulation import (ENTRY, END, HANDOFF, HELPER, INSPECTED, REPAIR_END, REPAIR_START,
                                SHIFT_SECONDS, TEST_START, Plant, _calendar, _line, simulate)
from good_output import FIRST_PASS, REWORK_PASS, REWORK_SCRAP


def scenario(sku, kits, routes, start=0):
    jobs = len(sku)
    return SimpleNamespace(work=np.full((jobs, 1), 5.), sku=np.array(sku), orders=np.full(jobs, start),
                           kits=np.array(kits)+start, routes=np.array(routes),
                           staffing=np.zeros((2, 3, 2), dtype=int), staffing_floater=np.zeros((2, 3, 2), dtype=int))


class FactorySimulationTests(unittest.TestCase):
    def test_ready_dispatch_starts_material_complete_order(self):
        case = scenario([0, 1, 0], [100, 0, 100], [FIRST_PASS]*3)
        locked = Plant(setup=False)
        fifo = simulate(case, locked)
        ready = simulate(case, replace(locked, kit_dispatch="ready"))
        self.assertEqual(fifo.trace[1, ENTRY], 105)
        self.assertEqual(ready.trace[1, ENTRY], 0)
        self.assertEqual(fifo.kit_wait[0], 100)
        self.assertEqual(ready.kit_wait[0], 95)

    def test_setup_is_charged_at_each_transition(self):
        case = scenario([0, 1, 0], [0]*3, [FIRST_PASS]*3)
        result = simulate(case, Plant())
        self.assertEqual(result.setups[0, 0], 48)
        self.assertEqual(result.changeovers[0, 0], 2)
        self.assertEqual(list(result.trace[:, INSPECTED]), [17, 46, 75])

    def test_cart_capacity_return_time_and_transfer(self):
        case = scenario([0, 0, 0], [0]*3, [REWORK_PASS]*3)
        constrained = simulate(case, Plant(parking_bays=1, carts=2, slots_per_cart=1,
                                           transfer=15, return_bin=20))
        np.testing.assert_equal(constrained.trace[:, INSPECTED], [17, 29, 64])
        np.testing.assert_equal(constrained.trace[:, HANDOFF], [17, 52, 142])
        np.testing.assert_equal(constrained.trace[:, REPAIR_START], [32, 122, 212])

    def test_maintenance_defers_entire_operation_to_window_end(self):
        case = scenario([0, 0], [0, 0], [FIRST_PASS, FIRST_PASS])
        case.work = np.full((2, 4), 5.)
        plant = Plant(setup=False)
        open_schedule = simulate(case, plant)
        exact_cutoff = simulate(case, plant, [(20, 50)])
        starts_during_shutdown = simulate(case, plant, [(18, 50)])
        np.testing.assert_equal(open_schedule.trace[:, END], [32, 44])
        np.testing.assert_equal(exact_cutoff.maintenance_operations, [[15, 20], [50, 55]])
        np.testing.assert_equal(starts_during_shutdown.maintenance_operations, [[50, 55], [55, 60]])
        self.assertEqual(starts_during_shutdown.maintenance_wait[0], 35.)
        np.testing.assert_equal(starts_during_shutdown.trace[:, END], [67, 79])

    def test_maintenance_rejects_reversed_and_overlapping_windows(self):
        case = scenario([0], [0], [FIRST_PASS])
        case.work = np.full((1, 4), 5.)
        for windows in ([(9, 5)], [(1, 6), (5, 11)]):
            with self.assertRaises(ValueError):
                simulate(case, Plant(), windows)

    def test_receiving_maintenance_and_dispatch_share_the_same_clock(self):
        case = scenario([0, 1], [0, 60], [FIRST_PASS, FIRST_PASS])
        case.work = np.full((2, 4), 5.)
        case.kits, lots = receive_kits(case, ReceivingPlan(minutes_a=1, minutes_b=1))
        self.assertEqual(lots[0].released, 120)
        result = simulate(case, Plant(setup=False), [(130, 200)])
        np.testing.assert_equal(result.trace[:, ENTRY], [0, 120])
        np.testing.assert_equal(result.trace[:, END], [32, 217])
        shipped = dispatch(result.trace[:, END], case.orders, np.array([True, True]),
                           Outbound((2, 4, 5), (1, 1, 1), 1))
        np.testing.assert_equal(shipped, [120, 300])

    def test_parallel_bench_needs_qualified_staff_and_merges_in_order(self):
        case = scenario([0, 0], [0, 0], [FIRST_PASS, FIRST_PASS])
        case.work = np.array([[5., 5., 5., 30.], [5., 5., 5., 6.]])
        plant = Plant(setup=False)
        single = simulate(case, plant)
        unavailable = simulate(case, plant, parallel_available=[False]*4)
        np.testing.assert_equal(single.trace, unavailable.trace)
        staffed = simulate(case, plant, parallel_available=[True, False, False, False])
        np.testing.assert_equal(staffed.parallel_used, [False, True])
        np.testing.assert_equal(staffed.maintenance_operations, [[15, 45], [20, 26]])
        self.assertLess(staffed.trace[0, END], staffed.trace[1, END])

    def test_cart_footprint_does_not_use_reserved_aisle(self):
        plant = Plant(parking_bays=4, carts=4)
        self.assertEqual(plant.waiting_places, 24)
        self.assertEqual(replace(plant, parking_length_cm=480).waiting_places, 32)
        self.assertEqual(replace(plant, parking_width_cm=219).waiting_places, 0)
        case = scenario([0], [0], [FIRST_PASS])
        with self.assertRaises(ValueError):
            simulate(case, replace(plant, parking_width_cm=219))

    def test_planned_earlier_receipt_never_precedes_order_release(self):
        case = scenario([0, 1], [45*60, 50*60], [FIRST_PASS]*2, start=SHIFT_SECONDS)
        result = simulate(case, Plant(kit_advance=50))
        self.assertEqual(result.trace[0, ENTRY], SHIFT_SECONDS)
        self.assertGreaterEqual(result.trace[1, ENTRY], SHIFT_SECONDS)

    def test_shared_tester_carries_queue_after_repair(self):
        windows = np.array([[12, 300], [12, 300]])
        case = scenario([0]*3, [0]*3, [REWORK_PASS, REWORK_SCRAP, REWORK_PASS])
        args = (case.work, case.sku, case.orders, case.kits, case.routes, windows, np.zeros(1, dtype=int),
                24, 0, 0)
        one, *_ = _line(*args, 1, 90, 30, 0, 80)
        two, *_ = _line(*args, 2, 90, 30, 0, 80)
        np.testing.assert_equal(one[:, REPAIR_END], [107, 119, 131])
        np.testing.assert_equal(one[:, TEST_START], [107, 137, 167])
        np.testing.assert_equal(two[:, END], [137, 149, 167])

    def test_donor_coverage_and_training_limit_helper_windows(self):
        case = scenario([0], [0], [FIRST_PASS], start=SHIFT_SECONDS)
        case.staffing[0] = [[SHIFT_SECONDS+200, SHIFT_SECONDS+290], [SHIFT_SECONDS+300, SHIFT_SECONDS+350], [0, 0]]
        case.staffing_floater[:] = case.staffing
        case.staffing_floater[0, 2] = [SHIFT_SECONDS+100, SHIFT_SECONDS+500]
        baseline = _calendar(case, Plant(), 0, SHIFT_SECONDS+10)
        covered = _calendar(case, Plant(cover=True), 0, SHIFT_SECONDS+10)
        self.assertEqual(len(baseline), 1)
        self.assertEqual(len(covered), 2)
        self.assertEqual(covered[0, 0], SHIFT_SECONDS+100)
        self.assertTrue(np.all(simulate(case, Plant()).trace[:, HELPER] == -1))


if __name__ == "__main__":
    unittest.main()

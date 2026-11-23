import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from factory_floor import CLOSE_LAYOUT, Floor, next_bin
from factory_floor_study import second_attempt
from factory_simulation import END, ENTRY, Plant, simulate
from good_output import FIRST_PASS, REWORK_PASS, REWORK_SCRAP


def units(count):
    return SimpleNamespace(work=np.full((count,8), 5.), sku=np.zeros(count,dtype=int),
                           orders=np.zeros(count,dtype=int), kits=np.zeros(count,dtype=int),
                           routes=np.full(count,FIRST_PASS), staffing=np.zeros((2,3,2),dtype=int))


class FloorTests(unittest.TestCase):
    def test_one_slot_with_no_additional_rules_matches_existing_flow(self):
        case = units(40)
        case.work[::2,3] = 30
        policy = Floor(walk=False, directional_setup=False, route_minutes=0)
        standard = simulate(case, Plant())
        upgraded = simulate(case, Plant(), floor=policy)
        np.testing.assert_equal(standard.trace, upgraded.trace)
        direct = simulate(case, Plant(), floor=replace(policy,buffers=(0,)*7))
        self.assertTrue(np.all(direct.trace[:,END] >= upgraded.trace[:,END]))

    def test_cards_and_routes_release_only_cleared_bins(self):
        self.assertEqual(next_bin(65, 60, 0, 60), 120)
        self.assertEqual(next_bin(0, 60, 0, 60), 120)
        self.assertEqual(next_bin(0, 63, 120, 60), 180)
        case = units(5)
        case.kits[2:]=61
        policy = Floor(walk=False, route_minutes=1, bin_units=2, cards=1)
        trace = simulate(case, Plant(), floor=policy).trace
        np.testing.assert_equal(trace[:,ENTRY], [0, 5, 120, 125, 180])

    def test_direction_of_change_and_layout_distance(self):
        original = Floor()
        compact = replace(original,layout=CLOSE_LAYOUT)
        self.assertLess(compact.distances().sum(),original.distances().sum())
        settings = original.arrays(np.full(8,20),8)
        np.testing.assert_equal(settings[2][:,0], [25,15])

    def test_second_repair_respects_a_single_shared_bench(self):
        routes = np.array([REWORK_SCRAP, FIRST_PASS, REWORK_SCRAP])
        end, outcomes, used = second_attempt(np.array([200., 500.,100.]), routes,
                                            np.array([[0.1,.6],[.1,.1],[.1,.1]]))
        np.testing.assert_equal(used,[2,0])
        np.testing.assert_equal(end,[520,500,310])
        np.testing.assert_equal(outcomes,[REWORK_SCRAP,FIRST_PASS,REWORK_PASS])


if __name__ == "__main__":
    unittest.main()

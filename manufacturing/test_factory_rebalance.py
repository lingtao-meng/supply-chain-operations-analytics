import unittest
from dataclasses import replace

import numpy as np

from factory_capacity_plan import violations
from factory_floor import Floor
from factory_inputs import SHIFT_SECONDS, load_case
from factory_rebalance import (BASE_DESIGN, PARALLEL_DESIGN, Choice, candidate_moves,
                               cohort_rows, workload)
from factory_simulation import END, Plant, _select, simulate
from test_factory_floor import units


class RebalanceTests(unittest.TestCase):
    def test_every_transfer_preserves_coverage_precedence_and_zones(self):
        case, plan = load_case()
        moves = list(candidate_moves(case, plan))
        self.assertGreater(len(moves), 1)
        for c in moves[1:]:
            trial = [list(s) for s in plan]
            trial[c.source].remove(c.task)
            trial[c.dest].append(c.task)
            pos = {t:i for i, group in enumerate(trial) for t in group}
            self.assertEqual(sorted(pos), list(range(1,1+len(case.task_times))))
            self.assertTrue(all(pos[a] <= pos[b] for a,b in case.precedence))
            self.assertEqual(plan[4], trial[4])
            self.assertFalse(violations(BASE_DESIGN, trial) or violations(PARALLEL_DESIGN, trial))
        s = units(3)
        result = workload(s, np.full((3,1), 2.), Choice(1, 0, 1))
        np.testing.assert_equal(result[:,0], 3)
        np.testing.assert_equal(result[:,1], 7)
        np.testing.assert_equal(result.sum(axis=1), s.work.sum(axis=1))

    def test_choice_only_at_a_released_competing_head(self):
        orders = np.array([0, SHIFT_SECONDS])
        def pick(rule, count, kits):
            return _select(np.array([0,1]), SHIFT_SECONDS+3, orders, np.array(kits),
                           0, 1, 80, rule, count)[0]
        self.assertEqual([pick(4,i,orders) for i in range(6)], [0,1,1,0,1,1])
        self.assertEqual(pick(3,0,orders), 1)
        self.assertEqual(pick(1,0,orders), 0)
        self.assertEqual(pick(3,1,[0,SHIFT_SECONDS+4]), 0)

    def test_secondary_bench_keeps_own_changeover_state_and_fifo_exit(self):
        case = units(8)
        case.sku = np.array([0,1]*4)
        case.work[:,3] = 60.
        floor = Floor(route_minutes=0, buffers=(2,)*7)
        staffed = simulate(case, Plant(), parallel_available=[True]*4, floor=floor, diagnose=True)
        closed = simulate(case, Plant(), parallel_available=[False]*4, floor=floor)
        serial = simulate(case, Plant(), floor=floor)
        np.testing.assert_allclose(closed.trace, serial.trace)
        self.assertTrue(staffed.parallel_used.any())
        self.assertTrue(np.all(np.diff(staffed.trace[:, END])>=0))
        expected = 0.
        for use in (False,True):
            sku = case.sku[staffed.parallel_used==use]
            expected += sum(55*(1.25 if a==0 else .75) for a,b in zip(sku,sku[1:]) if a != b)
        self.assertAlmostEqual(staffed.setups[:,3].sum(),expected)
        np.testing.assert_allclose(staffed.station_seconds[:,:,0].sum(), case.work.sum())
        self.assertTrue(np.all(staffed.station_seconds.sum(axis=2) <= SHIFT_SECONDS+1e-8))
        with self.assertRaises(ValueError):
            simulate(case, Plant(), parallel_available=[True]*4,
                     floor=replace(floor,buffers=(0,)*7))

    def test_overdue_clock_retains_terminal_backlog_and_excludes_disposed_scrap(self):
        case = units(3)
        ready = np.array([10,20,30])
        good = np.array([True,True,False])
        shipped = np.array([SHIFT_SECONDS+60., np.inf, np.inf])
        rows = cohort_rows(case, ready, good, shipped)
        self.assertEqual(rows[0]["closing_open"],2)
        self.assertEqual(rows[1]["old_sent"],1)
        self.assertEqual(rows[1]["old_open"],1)
        self.assertEqual(rows[-1]["late_minutes"],3*SHIFT_SECONDS/60)
        self.assertEqual(rows[-1]["scrap"],1)


if __name__ == "__main__":
    unittest.main()

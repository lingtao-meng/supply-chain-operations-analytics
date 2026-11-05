import json
import unittest

import numpy as np

from line_balance import ROOT, Case
from optimize_buffers import enumerate_plans
from shift_capacity import demand_completion_times, simulate_shift


class TestJointOptimization(unittest.TestCase):
    def test_jackson_candidates_match_direct_assignment(self):
        records = json.loads((ROOT / "data" / "benchmarks.json").read_text())
        case = Case(**next(row for row in records if row["name"] == "Jackson"))
        direct = set()
        placement = {}
        loads = [0] * 4

        def assign(index):
            if index == len(case.order):
                if all(loads):
                    direct.add(tuple(placement[task] for task in case.order))
                return
            task = case.order[index]
            first = max((placement[p] for p in case.predecessors[task]), default=0)
            for station in range(first, 4):
                duration = case.duration(task)
                if loads[station] + duration <= 14:
                    placement[task] = station
                    loads[station] += duration
                    assign(index + 1)
                    loads[station] -= duration

        assign(0)
        plans = enumerate_plans(case, 14, 4)
        encoded = [{task: station for station, group in enumerate(plan) for task in group}
                   for plan in plans]
        self.assertEqual(len(plans), 209)
        self.assertEqual({tuple(plan[task] for task in case.order) for plan in encoded}, direct)

    def test_demand_completion_matches_throughput_at_shift_cutoff(self):
        rng = np.random.default_rng(17)
        work = rng.uniform(1, 4, (120, 4, 30))
        demand, duration = 30, 100
        for layout in ((0, 0, 0), (0, 1, 2), (2, 1, 0)):
            finish = demand_completion_times(work[:demand], layout)
            output, _ = simulate_shift(work, layout, duration)
            self.assertTrue((finish <= duration).any())
            self.assertTrue((finish > duration).any())
            np.testing.assert_array_equal(finish <= duration, output >= demand)


if __name__ == "__main__":
    unittest.main(verbosity=2)

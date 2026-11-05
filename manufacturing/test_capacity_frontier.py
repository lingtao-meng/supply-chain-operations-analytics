import json
import unittest

import numpy as np

from capacity_frontier import beam_plans, evaluate_pool, reliable_capacity, required_hits
from line_balance import ROOT, Case, verify
from optimize_buffers import enumerate_plans, workloads_for
from shift_capacity import SHIFT_SECONDS, reliability_lower, shift_output_counts


class TestCapacityFrontier(unittest.TestCase):
    def test_capacity_is_largest_passing_integer_target(self):
        counts = np.arange(1500, 1680)
        cutoff = int(reliable_capacity(counts, required_hits(len(counts))))
        self.assertGreaterEqual(reliability_lower(int((counts >= cutoff).sum()), len(counts)), 0.95)
        self.assertLess(reliability_lower(int((counts >= cutoff + 1).sum()), len(counts)), 0.95)

    def test_packed_candidate_results_match_individual_simulations(self):
        rng = np.random.default_rng(102)
        times = rng.uniform(160, 300, (180, 3, 60))
        plans = [((1, 2), (3,)), ((1,), (2, 3))]
        layouts = [(0,), (1,)]
        actual = evaluate_pool(plans, layouts, [times])
        for candidate in actual:
            work = workloads_for(times, candidate.plan)
            output = shift_output_counts(work, candidate.layout, SHIFT_SECONDS)
            self.assertEqual(candidate.capacity, int(reliable_capacity(output, required_hits(len(output)))))
            self.assertEqual(candidate.worst_mean, output.mean())

    def test_beam_preserves_constraints_on_both_benchmarks(self):
        cases = json.loads((ROOT / "data" / "benchmarks.json").read_text())
        for record, cycle, staff in zip(cases, (14, 15), (4, 8)):
            case = Case(**record)
            plans = beam_plans(case, cycle, staff, 16)
            self.assertLessEqual(len(plans), 17)
            self.assertGreater(len(plans), 1)
            for plan in plans:
                self.assertEqual(len(plan), staff)
                verify(case, cycle, plan)
            if case.name == "Jackson":
                exact = {tuple(frozenset(group) for group in plan)
                         for plan in enumerate_plans(case, cycle, staff)}
                self.assertTrue(all(tuple(frozenset(group) for group in plan) in exact for plan in plans))


if __name__ == "__main__":
    unittest.main(verbosity=2)

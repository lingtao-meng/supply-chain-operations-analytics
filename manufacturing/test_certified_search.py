import unittest

import numpy as np

from certified_search import add_confirmation_bounds, binary_output_counts, layout_for, mask_for, solve_budget
from shift_capacity import shift_output_counts


class ToyScores:
    def __init__(self):
        self.scores = {}
        self.data = {(plan, mask): 20 + plan + sum((gap + 2) * (plan + 1)
                     for gap in range(4) if mask & (1 << gap))
                     for plan in range(3) for mask in range(16)}

    def score(self, plan, mask):
        key = (plan, mask)
        self.scores[key] = self.data[key]
        return self.scores[key]


class TestCertifiedSearch(unittest.TestCase):
    def test_binary_layout_rejects_values_that_alias_another_gap(self):
        self.assertEqual(mask_for((0, 1, 0)), 2)
        with self.assertRaises(ValueError):
            mask_for((2, 0, 0))

    def test_accelerated_recurrence_matches_reference_for_every_layout(self):
        rng = np.random.default_rng(802)
        service = rng.uniform(0.3, 3, (90, 5, 24))
        packed = np.ascontiguousarray(service.transpose(2, 0, 1))
        for mask in range(16):
            layout = layout_for(mask, 4)
            reference = shift_output_counts(service, layout, 35)
            accelerated = binary_output_counts(packed, mask, 35)
            np.testing.assert_array_equal(accelerated, reference)
            for gap in range(4):
                enlarged = binary_output_counts(packed, mask | (1 << gap), 35)
                self.assertTrue(np.all(enlarged >= accelerated))

    def test_complete_search_matches_independent_enumeration(self):
        for budget in range(5):
            scores = ToyScores()
            roots = [scores.score(plan, 15) for plan in range(3)]
            expected = max(value for (_, mask), value in scores.data.items() if mask.bit_count() <= budget)
            result = solve_budget(scores, budget, 4, roots, (0, 0))
            self.assertEqual(result.lower, expected)
            self.assertEqual(result.upper, expected)
            self.assertLessEqual(result.mask.bit_count(), budget)

    def test_interrupted_search_retains_a_valid_upper_bound(self):
        scores = ToyScores()
        roots = [scores.score(plan, 15) for plan in range(3)]
        expected = max(value for (_, mask), value in scores.data.items() if mask.bit_count() <= 2)
        result = solve_budget(scores, 2, 4, roots, (0, 0), evaluation_limit=3)
        self.assertLessEqual(result.lower, expected)
        self.assertGreaterEqual(result.upper, expected)
        self.assertLessEqual(result.evaluations, 3)
        self.assertGreater(result.upper, result.lower)

    def test_joint_confirmation_accounts_for_all_scenario_claims(self):
        targets = [{"planning_confirmation": {"validation_hits": [300, 299]}} for _ in range(6)]
        report = {"configuration": {"confirmation_runs": 300}, "cases": [{"results": targets}]}
        add_confirmation_bounds(report)
        self.assertEqual(report["configuration"]["confirmation_family_size"], 12)
        result = targets[0]["planning_confirmation"]
        self.assertAlmostEqual(result["simultaneous_lower95_by_scenario"][0], (0.05 / 12) ** (1 / 300), places=6)
        self.assertGreaterEqual(result["simultaneous_lower95_by_scenario"][1], 0.95)
        self.assertTrue(result["passes_simultaneous_confirmation"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

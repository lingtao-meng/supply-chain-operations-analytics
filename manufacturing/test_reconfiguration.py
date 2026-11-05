import unittest

import numpy as np

from capacity_frontier import Candidate
from reconfiguration import SHOCK_TARGET, moved_tasks, resource_frontier, slowdown_scenarios
from shift_capacity import shift_output_counts


class TestReconfiguration(unittest.TestCase):
    def test_pareto_filter_requires_both_feasibility_and_no_resource_dominance(self):
        incumbent = ((1, 2), (3,), (4,))
        one_move = ((1,), (2, 3), (4,))
        two_moves = ((1,), (2,), (3, 4))
        candidates = [
            Candidate(incumbent, (0, 0), SHOCK_TARGET - 1, SHOCK_TARGET + 100),
            Candidate(one_move, (1, 1), SHOCK_TARGET, SHOCK_TARGET + 10),
            Candidate(two_moves, (0, 1), SHOCK_TARGET, SHOCK_TARGET + 20),
            Candidate(two_moves, (1, 1), SHOCK_TARGET + 10, SHOCK_TARGET + 30),
        ]
        _, efficient = resource_frontier(candidates, incumbent)
        self.assertEqual([key for key, _ in efficient], [(1, 2), (2, 1)])
        self.assertEqual(moved_tasks(incumbent, ((2, 1), (3,), (4,))), ())
        self.assertEqual(moved_tasks(incumbent, two_moves), (2, 3))

    def test_larger_slowdown_cannot_increase_output_for_a_fixed_line(self):
        rng = np.random.default_rng(913)
        base = rng.uniform(0.8, 2.5, (100, 4, 20))
        stressed = slowdown_scenarios([base], 45)[0]
        np.testing.assert_array_equal(stressed[:, :3], base[:, :3])
        np.testing.assert_allclose(stressed[:, 3], base[:, 3] * 1.45)
        for layout in ((0, 0, 0), (0, 1, 2), (2, 0, 1)):
            normal = shift_output_counts(base, layout, 45)
            slower = shift_output_counts(stressed, layout, 45)
            self.assertTrue(np.all(slower <= normal))
            self.assertTrue(np.any(slower < normal))


if __name__ == "__main__":
    unittest.main(verbosity=2)

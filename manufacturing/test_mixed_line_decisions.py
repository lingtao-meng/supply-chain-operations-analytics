import unittest

import numpy as np

from factory_inputs import load_case
from factory_rebalance import Choice, trial as rebalance_trial
from mixed_line_decisions import POLICIES, contrasts, trial, verify_target


class MixedLineDecisionTests(unittest.TestCase):
    def test_base_and_parallel_match_previous_simulation_on_same_seed(self):
        case, plan = load_case()
        seed = 2026113201
        actual = trial(case, plan, np.random.SeedSequence(seed))
        choices = (Choice(), Choice(parallel=True))
        old = rebalance_trial(case, plan, choices, False, np.random.SeedSequence(seed))
        for key, choice in zip(("base", "assembly"), choices, strict=True):
            for a, b in zip(actual[key], old[choice.name][0], strict=True):
                self.assertEqual({k: a[k] for k in b}, b)

    def test_delivery_requires_both_models_in_both_shifts(self):
        units = np.array([[[231, 99], [232, 99]], [[300, 90], [231, 100]], [[240, 99], [229, 101]]])
        target = verify_target(units, 330, .05)
        self.assertEqual(target["hits"], 1)
        self.assertEqual(target["quotas"], [231, 99])

    def test_factorial_contrasts_distinguish_base_and_conditional_gains(self):
        shortage = np.array([[100-4*p.supply-5*p.assembly - 9*(p.supply and p.assembly)] * 4
                             for p in POLICIES])
        result, incremental = contrasts(shortage)
        self.assertEqual(result["supply+assembly"]["mean"], 4.5)
        self.assertEqual(result["rework"]["mean"], 0)
        self.assertEqual(incremental["base -> supply"]["mean"], 4)
        self.assertEqual(incremental["assembly -> supply+assembly"]["mean"], 13)

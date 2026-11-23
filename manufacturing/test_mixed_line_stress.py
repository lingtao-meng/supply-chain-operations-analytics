import json
import unittest

import numpy as np

from factory_inputs import load_case
from line_balance import ROOT
from mixed_line_decisions import POLICIES, Pressure, trial
from mixed_line_stress import PRESSURES, summarize


class MixedLineStressTests(unittest.TestCase):
    def test_pressure_targets_only_intended_resources(self):
        case, plan = load_case()
        seed = 2026112590
        nominal = trial(case, plan, np.random.SeedSequence(seed))
        absent = trial(case, plan, np.random.SeedSequence(seed),
                       pressure=Pressure(adverse_qualified_availability=0))
        for policy in POLICIES:
            if not policy.assembly:
                self.assertEqual(absent[policy.name], nominal[policy.name])
            else:
                self.assertEqual([absent[policy.name][i]["parallel_jobs"] for i in (1, 2)], [0, 0])
        slower = trial(case, plan, np.random.SeedSequence(seed), pressure=Pressure(improved_repair_seconds=90))
        for name, unchanged in zip(("rework","assembly+rework","supply+rework","supply+assembly+rework"),
                                   ("base","assembly","supply","supply+assembly"), strict=True):
            self.assertEqual(slower[name], nominal[unchanged])

    def test_cost_ranking_excludes_a_configuration_that_fails_combined_stress(self):
        rates = json.loads((ROOT/"data"/"mixed_line_costs.json").read_text())
        panel = np.zeros((800, len(PRESSURES), len(POLICIES), 7))
        panel[..., :4] = [350, 150, 350, 150]
        panel[..., 5] = 5200
        panel[..., 6] = 1
        panel[:, :, 0, 5] = 4700
        panel[:, -1, 0, :4] = [300, 145, 300, 145]
        expected = summarize(panel, [500]*len(POLICIES), rates)
        self.assertTrue(expected["outcomes"]["base"]["scenarios"]["reference"]["reliability"][500]["passed"])
        self.assertFalse(expected["outcomes"]["base"]["scenarios"]["combined"]["reliability"][500]["passed"])
        self.assertNotEqual(expected["lowest_worst_case_cost_at_common_target"][500], "base")
        self.assertEqual(expected["highest_verified_by_capital"][0], {"target":330,"policy":"base"})

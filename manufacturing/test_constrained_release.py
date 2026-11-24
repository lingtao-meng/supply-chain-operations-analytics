import unittest

import numpy as np

from constrained_release import CANDIDATES, READY, TARGETS, assessment, choose
from order_mix_study import CONDITIONS, MIXES


class ConstrainedReleaseTests(unittest.TestCase):
    def test_highest_service_and_protected_nominees_are_locked_separately(self):
        # Two candidates can ship the same mix, but one exceeds the SKU shortage guard.
        strong = CANDIDATES[1]
        unprotected = CANDIDATES[2]
        setup = dict.fromkeys(MIXES, CANDIDATES)
        panel = {}
        for mix in MIXES:
            baseline = np.tile((300, 170, 300, 170, 500, 400, 210, 35, 1), (200, 1))
            protected = baseline.copy()
            protected[:, :4] = (420, 300, 420, 300)
            preferred_for_service = protected.copy()
            preferred_for_service[:, 4:6] = (600, 200)
            variants = dict.fromkeys((policy.name for policy in CANDIDATES), baseline)
            variants.update({strong.name: protected, unprotected.name: preferred_for_service})
            panel[mix] = dict.fromkeys(CONDITIONS, variants)
        design = assessment(panel, setup)
        _, nominees = choose(design)
        self.assertEqual(nominees["50_50"]["best_design_service"], unprotected.name)
        self.assertEqual(nominees["50_50"]["design_challenger"], strong.name)

    def test_proxy_breach_rejects_all_targets_even_at_high_service(self):
        schedule = dict.fromkeys(MIXES, (READY,))
        rows = np.tile((600, 400, 600, 400, 0, 0, 0, 0, 1), (800, 1))
        rows[-1, -1] = 0
        panel = dict.fromkeys(MIXES, dict.fromkeys(CONDITIONS, {READY.name: rows}))
        holdout = assessment(panel, schedule, effect_family=24, target_family=78)
        row = holdout["70_30"][READY.name]
        self.assertEqual(row["best_service"], TARGETS[-1])
        self.assertIsNone(row["best_acceptable_target"])
        self.assertEqual(row["scenarios"]["combined"]["breached_limits"], ["ergonomics"])

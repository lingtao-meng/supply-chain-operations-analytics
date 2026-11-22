import unittest

import numpy as np

from factory_capacity_plan import Design, certify, violations
from factory_inputs import load_case


class CapacityPlanTests(unittest.TestCase):
    def test_cheaper_bays_fail_zoning_or_separation(self):
        _, stations = load_case()
        self.assertIn("duplicate_outside_assembly_zone", violations(Design("general_bay"), stations))
        self.assertIn("assembly_and_incoming_inspection_share_bay",
                      violations(Design("shared_inspection_bay"), stations))
        self.assertIn("cart_bays_exceeded", violations(Design("converted_bay", carts=3), stations))
        self.assertEqual(violations(Design("added_bay", carts=3), stations), [])

    def test_promise_requires_both_models_in_each_shift(self):
        data = np.array([[[800, 200], [700, 300]], [[700, 300], [700, 300]]])
        data = np.pad(data, ((0,0), (0,0), (0,3)))
        result = certify(data, 1000, .05)
        self.assertEqual(result["hits"], 1)
        self.assertFalse(result["accepted"])


if __name__ == "__main__":
    unittest.main()

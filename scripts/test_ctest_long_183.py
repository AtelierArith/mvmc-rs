#!/usr/bin/env python3
"""Infrastructure tests for the long ctest driver (no model execution)."""
import unittest

import run_ctest_long_183 as driver


class LongCtestDriver(unittest.TestCase):
    def test_pass_requires_every_gate_exit_zero(self):
        self.assertEqual(driver.classify(["ok"], [0, 0]), "Pass")
        self.assertNotEqual(driver.classify(["ok"], [0, 101]), "Pass")

    def test_failure_classification(self):
        self.assertEqual(driver.classify(["x Unsupported: y"], [101, 0]), "Unsupported")
        self.assertEqual(driver.classify(["MissingFixture: y"], [0, 101]), "MissingFixture")
        self.assertEqual(driver.classify(["UNVERIFIED models"], [101, 101]), "Unverified")
        self.assertEqual(driver.classify(["boom"], [101, 101]), "Failure")

    def test_selection_and_unknown_model_provenance(self):
        self.assertIn("heisenberg_chain_real", driver.requested_models("all"))
        entry = driver.model_provenance("../etc")
        self.assertFalse(entry["reference_present"])
        self.assertFalse(driver.model_provenance("no_such_model")["reference_present"])


if __name__ == "__main__":
    unittest.main()

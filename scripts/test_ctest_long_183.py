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

    def test_verdict_gate_is_upstream_rule_and_supplementary_never_decides(self):
        self.assertEqual(driver.VERDICT_GATE[1], "rust_ctest_upstream_rule_selected_models")
        self.assertEqual(driver.VERDICT_GATE[2], "MVMC_RS_CTEST_UPSTREAM_MODELS")
        self.assertNotIn(driver.VERDICT_GATE, driver.SUPPLEMENTARY_GATES)
        self.assertEqual(driver.classify(["MissingFixture"], [101]), "MissingFixture")

    def test_every_known_model_has_upstream_reference(self):
        models = driver.known_models()
        self.assertEqual(len(models), 13)
        for model in models:
            self.assertTrue(driver.upstream_provenance(model)["upstream_reference_present"], model)
        self.assertFalse(driver.upstream_provenance("../etc")["upstream_reference_present"])


if __name__ == "__main__":
    unittest.main()

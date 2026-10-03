"""Adversarial tests for optional reference comparisons; no oracle invocation."""
import math
import unittest
from numerical_comparison import GREEN, compare_text, within
from check_interall_real_c_parity import bits


class ComparisonTests(unittest.TestCase):
    def test_computation_is_semantic_but_input_and_structure_are_exact(self):
        reference = "# source sha256=abc\n7 " + bits(0.5) + " " + bits(1.0) + "\n"
        selector = lambda r, c, fields: GREEN if c == 2 else None
        changed = reference.replace(bits(1.0), bits(math.nextafter(1.0, 2.0)))
        compare_text(changed, reference, selector)
        for bad in (reference.replace("sha256=abc", "sha256=abd"),
                    reference.replace("7 ", "8 "),
                    reference.replace(bits(0.5), bits(math.nextafter(0.5, 1.0))),
                    reference.replace(bits(1.0), bits(1.001)),
                    reference + "0\n"):
            with self.assertRaises(AssertionError):
                compare_text(bad, reference, selector)

    def test_nonfinite_and_zero_semantics(self):
        self.assertTrue(within(0.0, -0.0, *GREEN))
        self.assertTrue(within(math.nan, math.nan, *GREEN))
        self.assertFalse(within(math.nan, 0.0, *GREEN))
        self.assertFalse(within(math.inf, -math.inf, *GREEN))
        self.assertFalse(within(math.inf, 1e308, *GREEN))

    def test_invalid_bounds_fail(self):
        for bounds in ((-1, 0), (0, math.inf), (math.nan, 0)):
            with self.assertRaises(AssertionError):
                within(1.0, 1.0, *bounds)

    def test_finite_subtraction_overflow_keeps_significant_difference(self):
        maximum = float.fromhex("0x1.fffffffffffffp+1023")
        self.assertFalse(within(maximum, -maximum, maximum, 1e-10))
        self.assertTrue(within(maximum, maximum, 0, 0))


if __name__ == "__main__":
    unittest.main()

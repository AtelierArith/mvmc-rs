"""Adversarial protocol tests, not numerical expected-value generation."""
from pathlib import Path
import tempfile
import unittest

from mpi_issue179_cg_diagnostics import diagnose, read


CAPTURE = """d:events 3
d:event-000000-kind 0
d:event-000000-mapping 1
n:event-000000-mean 1
n:event-000000-diagonal 1
n:event-000000-gradient 1
n:event-000000-real-samples 1
d:event-000001-kind 1
d:event-000001-phase 2
n:event-000001-search 1
n:event-000001-product 1
d:event-000002-kind 3
d:event-000002-iterations 1
n:event-000002-solution 1
n:event-000002-residual 1
n:event-000002-direction 1
"""


class CgDiagnosticsTests(unittest.TestCase):
    def parse(self, text):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "capture.txt"
            file.write_text(text)
            return read(file)

    def test_difference_is_diagnostic_not_acceptance(self):
        a = self.parse(CAPTURE)
        b = self.parse(CAPTURE.replace("product 1", "product 2"))
        result = diagnose(a, b)
        self.assertEqual(result["first_numeric_difference"]["absolute_difference"], 1)
        self.assertEqual(result["scope"], "DIAGNOSTIC_ONLY_NO_NUMERICAL_ACCEPTANCE")

    def test_malformed_observations_fail(self):
        variants = ("", CAPTURE.replace("d:events 3", "d:events 4"),
                    CAPTURE.replace("n:event-000001-product 1\n", ""),
                    CAPTURE.replace("product 1", "product nan"),
                    CAPTURE.replace("phase 2", "phase 9"),
                    CAPTURE.replace("mapping 1", "mapping -1"),
                    CAPTURE.replace("solution 1", "solution 1 2"),
                    CAPTURE + "n:event-000003-product 1\n",
                    CAPTURE + "n:event-000001-unexpected 1\n",
                    CAPTURE + "d:events 3\n")
        for text in variants:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.parse(text)

    def test_iteration_difference_is_not_silently_accepted(self):
        with self.assertRaises(ValueError):
            diagnose(self.parse(CAPTURE), self.parse(CAPTURE.replace("iterations 1", "iterations 2")))


if __name__ == "__main__":
    unittest.main()

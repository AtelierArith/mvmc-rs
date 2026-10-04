"""Adversarial checks for the optional MPI evidence gate (run through uv)."""
import tempfile
import unittest
import re
import subprocess
from pathlib import Path
from compare_mpi_issue179 import compare, compare_root, read, read_c_flags, validate_trace


class EvidenceChecks(unittest.TestCase):
    def test_c_mask_cannot_disable_all_written_fields(self):
        valid = "flags 1 0\nwritten 1 0\nC_source_sha256 6c53cb832f93d6cbfd7cea955fbb693738af5536b913d36af32b98eed38c32d9\nenumerator_sha256 " + "a" * 64 + "\ncontract_sha256 " + "b" * 64 + "\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flags.txt"
            path.write_text(valid)
            self.assertEqual(read_c_flags(path), ([1, 0], [1, 0]))
            for text in (valid.replace("written 1 0", "written 0 0"),
                         valid.replace("written 1 0", "written 1 2"),
                         valid.replace("written 1 0", "written 1"),
                         valid.replace("enumerator_sha256 " + "a" * 64, "enumerator_sha256 unknown")):
                with self.subTest(text=text), self.assertRaises(ValueError):
                    path.write_text(text)
                    read_c_flags(path)

    def test_actual_trace_is_required_and_shaped(self):
        valid = {"d:trace-events": ["2"], "d:trace-000000": ["9", "4294967295"],
                 "d:trace-000001": ["7", "1", "4294967295"]}
        validate_trace(valid, 0)
        for invalid in ({}, {"d:trace-events": ["0"]},
                        {**valid, "d:trace-events": ["3"]},
                        {**valid, "d:trace-000000": ["9", "1"]},
                        {**valid, "d:trace-000000": ["0", "1"]},
                        {**valid, "d:trace-000001": ["7", "2", "0"]},
                        {**valid, "d:trace-000001": ["7", "1", "4294967296"]}):
            with self.assertRaises(ValueError):
                validate_trace(invalid, 0)

    def test_discrete_drift_is_never_tolerated(self):
        with self.assertRaisesRegex(ValueError, "discrete mismatch"):
            compare({"d:rng": ["1"]}, {"d:rng": ["2"]}, 1e9, 1e9)

    def test_c_written_flags_preserve_unwritten_raw_difference(self):
        key = "d:sr-system-000000-flags"
        a = {key: ["1", "0"], "d:sr-system-000000-active": ["0"]}
        b = {**a, key: ["1", "1"]}
        compare(a, b, 0, 0, c_flags=([1, 0], [1, 0]))
        self.assertEqual(b[key], ["1", "1"], "reference raw metadata must remain unchanged")
        with self.assertRaises(ValueError):
            compare(a, b, 0, 0)
        for invalid in ({**b, key: ["0", "1"]},
                        {**b, "d:sr-system-000000-active": ["1"]},
                        {key: ["1", "1"]}):
            with self.assertRaises(ValueError):
                compare(a, invalid, 0, 0, c_flags=([1, 0], [1, 0]))

    def test_written_imaginary_family_exception_is_exact(self):
        key = "d:sr-system-000000-flags"
        a = {key: ["1", "0"], "d:sr-system-000000-active": ["0"]}
        with self.assertRaises(ValueError):
            compare(a, {**a, key: ["1", "1"]}, 1e9, 1e9, c_flags=([1, 0], [1, 1]))

    def test_acceptance_metadata_must_match_actual_decisions(self):
        rows = {"d:trace-events": ["2"], "d:trace-000000": ["9", "42"],
                "d:trace-000001": ["7", "0", "42"],
                "d:sampling-draw-count": ["1"], "d:acceptance-events": ["0", "1"]}
        validate_trace(rows, 0)
        for counts in (["1", "0"], [], ["0", "1", "0"]):
            with self.assertRaises(ValueError):
                validate_trace({**rows, "d:acceptance-events": counts}, 0)

    def test_numeric_mismatch_and_nonfinite_fail(self):
        for expected in ("2", "nan", "inf", "garbled"):
            with self.assertRaises(ValueError):
                compare({"n:energy": ["1"]}, {"n:energy": [expected]}, 1e-12, 1e-12)

    def test_shapes_and_missing_records_fail(self):
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            compare({"d:counter": ["1"]}, {"d:counter": ["1", "2"]}, 0, 0)
        with self.assertRaisesRegex(ValueError, "record mismatch"):
            compare({"d:seed": ["1"]}, {}, 0, 0)

    def test_empty_duplicate_and_malformed_evidence_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.txt"
            for content in ("", "d:rng\n", "d:seed 1\nd:seed 2\n"):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    read(path)

    def test_discrete_diagnostic_does_not_claim_numeric_parity(self):
        compare({"d:seed": ["1"], "n:energy": ["1"]}, {"d:seed": ["1"]}, 0, 0, discrete_only=True)

    def test_finite_componentwise_budget(self):
        compare({"n:energy": ["0", "1"]}, {"n:energy": ["1e-14", "1.00000000000001"]}, 1e-12, 1e-12)

    def test_root_output_is_finite_shaped_and_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "rust", Path(directory) / "reference"
            a.write_text("1 2\n")
            for content in ("", "1\n", "1 nan\n", "1 1e999\n", "1 3\n", "1 2\n1 2\n"):
                b.write_text(content)
                with self.assertRaises(ValueError):
                    compare_root(a, b, 1e-12, 1e-12)
            b.write_text("1 2\n")
            compare_root(a, b, 1e-12, 1e-12)

    def test_actual_cli_root_gate_propagates_mismatch_and_malformed_data(self):
        source = Path(__file__).with_name("verify_mpi_issue179.sh").read_text()
        match = re.search(r'awk -v atol=.*?\'(.*?)\' "\$dir/output/zvo_out.dat"', source, re.S)
        self.assertIsNotNone(match, "root gate changed; update this observation boundary")
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "rust", Path(directory) / "julia"
            a.write_text("1 2\n")
            for content, succeeds in (("1 2\n", True), ("1 3\n", False), ("1 nan\n", False), ("1 1e999\n", False), ("1\n", False), ("1 2\n1 2\n", False), ("", False)):
                with self.subTest(content=content):
                    b.write_text(content)
                    result = subprocess.run(["awk", "-v", "atol=1e-12", "-v", "rtol=1e-12", match[1], str(a), str(b)], capture_output=True, text=True)
                    self.assertEqual(result.returncode == 0, succeeds, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()

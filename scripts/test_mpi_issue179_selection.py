"""Independent negative tests for fail-closed shell gate preflight."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SelectionTests(unittest.TestCase):
    def select(self, text, scope="states", **settings):
        with tempfile.TemporaryDirectory() as directory:
            matrix = Path(directory) / "matrix.tsv"
            matrix.write_text(text)
            env = {k: v for k, v in os.environ.items() if not k.startswith("MPI179_")}
            env.update(settings)
            return subprocess.run(
                ["bash", "-c", 'source scripts/mpi_issue179_selection.sh; mpi179_validate_selection "$1" "$2"', "test", str(matrix), scope],
                cwd=ROOT, env=env, capture_output=True, text=True,
            ).returncode

    @staticmethod
    def rows():
        return "".join(f"r{n}-real-s1-cg0-store1\t{n}\treal\t1\t0\t1\tidentity\tsuccess\tEXECUTION_ONLY\t0\n" for n in (2, 4))

    def test_valid_axes(self):
        for scope in ("states", "workers", "rust"):
            self.assertEqual(self.select(self.rows(), scope), 0)

    def test_long20_is_explicit_and_coverage_counts_remain_exact(self):
        self.assertEqual(self.select(self.rows(), MPI179_STEPS_LIST="20",
                                     MPI179_EXPECTED_CELLS="2", MPI179_EXPECTED_TOTAL="6"), 0)
        self.assertEqual(self.select(self.rows(), MPI179_STEPS_LIST="1 2 3 20",
                                     MPI179_EXPECTED_CELLS="2", MPI179_EXPECTED_TOTAL="24"), 0)
        for steps in ("20 20", "50", "4", "120", "20 x", "020", ""):
            self.assertNotEqual(self.select(self.rows(), MPI179_STEPS_LIST=steps), 0, steps)

    def test_empty_and_ineligible_matrix_fail(self):
        for scope in ("states", "workers", "rust"):
            self.assertNotEqual(self.select("", scope), 0)
            self.assertNotEqual(self.select(self.rows().replace("success", "reject"), scope), 0)

    def test_invalid_filters_and_steps_fail(self):
        for settings in ({"MPI179_CELL_REGEX": "no-match"}, {"MPI179_CELL_REGEX": "["}, {"MPI179_CELL_REGEX": ""}, {"MPI179_STEPS_LIST": ""}, {"MPI179_STEPS_LIST": "0"}, {"MPI179_STEPS_LIST": "1 1"}, {"MPI179_STEPS_LIST": "1 x"}):
            self.assertNotEqual(self.select(self.rows(), **settings), 0, settings)

    def test_missing_rank_axis_fails_unfiltered_gate(self):
        for scope in ("states", "workers", "rust"):
            self.assertNotEqual(self.select(self.rows().splitlines()[0] + "\n", scope), 0)

    def test_duplicate_and_malformed_matrix_rows_fail(self):
        first = self.rows().splitlines()[0] + "\n"
        for text in (self.rows() + first, self.rows().replace("\treal\t", "\tunknown\t"),
                     self.rows().replace("\t1\t0\t1\t", "\t0\t0\t1\t"),
                     self.rows().replace("\tidentity\t", "\tunknown\t"),
                     self.rows().replace("\tEXECUTION_ONLY\t0", "\tEXECUTION_ONLY\t0\textra"),
                     self.rows().replace("r2-real-s1-cg0-store1", "../outside")):
            with self.subTest(text=text):
                self.assertNotEqual(self.select(text), 0)

    def test_declared_negative_cells_are_not_success_axes(self):
        text = self.rows()
        for expected in ("rejection", "missing", "output-failure"):
            row = self.rows().splitlines()[0].replace("r2-real-s1-cg0-store1", "r2-invalid")
            row = row.replace("\t0\t1\tidentity\tsuccess", f"\t2\t1\tidentity\t{expected}")
            self.assertEqual(self.select(text + row + "\n"), 0)

    def test_declared_coverage_counts_fail_closed(self):
        self.assertEqual(self.select(self.rows(), MPI179_EXPECTED_CELLS="2", MPI179_EXPECTED_TOTAL="18"), 0)
        for settings in ({"MPI179_EXPECTED_CELLS": "3"}, {"MPI179_EXPECTED_CELLS": ""},
                         {"MPI179_EXPECTED_TOTAL": "0"}, {"MPI179_EXPECTED_TOTAL": "17"},
                         {"MPI179_EXPECTED_TOTAL": "18", "MPI179_STEPS_LIST": "1 2"}):
            with self.subTest(settings=settings):
                self.assertNotEqual(self.select(self.rows(), **settings), 0)

    def test_disabled_feature_binary_fails(self):
        result = subprocess.run(["bash", "-c", "source scripts/mpi_issue179_selection.sh; mpi179_require_test /bin/true issue179_state"], cwd=ROOT)
        self.assertNotEqual(result.returncode, 0)

    def test_actual_scripts_reject_header_only_before_builds(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "matrix.tsv").write_text("cell\tranks\tmode\tsplit\tcg\tstore\tprojection\texpected\tstatus\texit\n")
            for script in ("verify_mpi_issue179_states.sh", "verify_mpi_issue179_workers.sh", "verify_mpi_issue179_rust_states.sh"):
                result = subprocess.run(["bash", str(ROOT / "scripts" / script), directory], cwd=ROOT, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0, script)
                self.assertIn("zero eligible verification cells", result.stderr, script)


if __name__ == "__main__":
    unittest.main()

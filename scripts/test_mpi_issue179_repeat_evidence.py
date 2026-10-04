import tempfile
import unittest
from pathlib import Path

from mpi_issue179_repeat_evidence import validate


class RepeatEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.roots = [Path(self.temp.name) / name for name in ("first", "second")]
        for root in self.roots:
            root.mkdir()
            for name in ("rank-0.txt", "rank-1.txt", "zvo_out.dat", "zvo_var.dat", "zqp_opt.dat"):
                (root / name).write_text("protocol-test-only\n")

    def check(self, cg=0, exits=(0, 0)):
        return validate(*self.roots, 2, cg, *exits)

    def test_direct_without_srinfo_passes(self):
        self.assertIn("files=5", self.check())

    def test_cg_without_srinfo_fails(self):
        for root in self.roots:
            for rank in (0, 1):
                (root / f"cg-rank-{rank}.txt").write_text("protocol-test-only\n")
        with self.assertRaisesRegex(ValueError, "zvo_SRinfo"):
            self.check(cg=1)

    def test_changed_rank_fails(self):
        (self.roots[1] / "rank-1.txt").write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.check()

    def test_changed_output_fails(self):
        (self.roots[1] / "zvo_var.dat").write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            self.check()

    def test_nonzero_launches_fail_even_with_identical_files(self):
        for exits in ((1, 0), (0, 1), (124, 0), (0, 137)):
            with self.subTest(exits=exits), self.assertRaisesRegex(ValueError, "launcher"):
                self.check(exits=exits)

    def test_missing_rank_fails(self):
        (self.roots[1] / "rank-0.txt").unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            self.check()

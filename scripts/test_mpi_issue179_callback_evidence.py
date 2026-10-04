import tempfile
import unittest
from pathlib import Path

from mpi_issue179_callback_evidence import validate


class CallbackEvidenceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        checkpoint = [8] + [0] * 5 + [10] + [0] * 10 + [624] + [0] * 624
        self.rows = {"d:status": [1], "d:steps": [3], "n:sr-step-0": [0],
                     "d:trace-events": [1], "d:trace-000000": checkpoint,
                     "d:initial-rng": [0] * 624, "d:rng": [0] * 624}
        self.write(self.rows)
        for filename in ("zvo_out.dat", "zvo_var.dat"):
            (self.root / filename).write_text("0\n")

    def write(self, rows):
        for rank in range(2):
            rows = {**rows, "d:callback-injected-rank": [0], "d:callback-calls": [1],
                    "d:callback-local-errors": [int(rank == 0)]}
            (self.root / f"rank-{rank}.txt").write_text("\n".join(
                key + " " + " ".join(map(str, value)) for key, value in rows.items()) + "\n")
            (self.root / f"callback-result-rank-{rank}.txt").write_text(
                "issue179 injected rank 0 callback failure" if rank == 0 else "callback failed on another rank")

    def test_actual_first_step_failure(self):
        self.assertIn("completed_steps=1", validate(self.root, 2, 0))

    def test_success_or_later_callback_or_missing_rank_fails(self):
        for rows in ({**self.rows, "d:status": [0]},
                     {**self.rows, "n:sr-step-1": [0]},
                     {**self.rows, "d:trace-000000": [8]}):
            with self.subTest(rows=rows):
                self.write(rows)
                with self.assertRaises(ValueError):
                    validate(self.root, 2, 0)
        self.write(self.rows)
        (self.root / "rank-1.txt").unlink()
        with self.assertRaises(FileNotFoundError):
            validate(self.root, 2, 0)

    def test_identity_reason_counts_and_output_boundary_fail_closed(self):
        with self.assertRaises(ValueError):
            validate(self.root, 2, 1)
        path = self.root / "rank-0.txt"
        original = path.read_text()
        for needle, replacement in (("d:callback-calls 1", "d:callback-calls 2"),
                                    ("d:callback-local-errors 1", "d:callback-local-errors 0")):
            path.write_text(original.replace(needle, replacement))
            with self.assertRaises(ValueError):
                validate(self.root, 2, 0)
        path.write_text(original)
        reason = self.root / "callback-result-rank-0.txt"
        reason.write_text("unrelated sampling failure")
        with self.assertRaises(ValueError):
            validate(self.root, 2, 0)
        self.write(self.rows)
        (self.root / "zvo_out.dat").write_text("0\n0\n")
        with self.assertRaises(ValueError):
            validate(self.root, 2, 0)


if __name__ == "__main__":
    unittest.main()

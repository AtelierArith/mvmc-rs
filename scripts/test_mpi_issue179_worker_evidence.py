import tempfile
import unittest
from pathlib import Path

from mpi_issue179_worker_evidence import validate


class WorkerEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.header = "requested=2 configured=2 threshold=1 observed_pool=2 qp_work=3\n"
        self.snapshot = (
            "kernel_parallel_calls=1 kernel_serial_calls=0 kernel_qp_items=3 "
            "kernel_parallel_qp_items=3 kernel_serial_qp_items=0 kernel_term_items=0 "
            "kernel_parallel_term_items=0 kernel_serial_term_items=0 "
            "kernel_worker_entries=3 kernel_workers_seen=1 kernel_worker_ids=[0]\n"
        )
        self.write(self.header + self.snapshot)

    def write(self, text):
        for rank in range(2):
            (self.root / f"workers-rank-{rank}.txt").write_text(text)

    def test_actual_entries_do_not_require_all_workers_scheduled(self):
        self.assertTrue(all("ACTUAL_PARALLEL_QP" in label for label in validate(self.root, 2, 2)))

    def test_capacity_only_fails(self):
        self.write(self.header)
        with self.assertRaises(ValueError):
            validate(self.root, 2, 2)

    def test_empty_rank_file_fails(self):
        (self.root / "workers-rank-1.txt").unlink()
        with self.assertRaises(FileNotFoundError):
            validate(self.root, 2, 2)

    def test_malformed_item_totals_fail(self):
        self.write(self.header + self.snapshot.replace("kernel_qp_items=3", "kernel_qp_items=4"))
        with self.assertRaises(ValueError):
            validate(self.root, 2, 2)

    def test_eligible_serial_execution_fails(self):
        serial = self.snapshot.replace("kernel_parallel_qp_items=3", "kernel_parallel_qp_items=0").replace(
            "kernel_serial_qp_items=0", "kernel_serial_qp_items=3")
        self.write(self.header + serial)
        with self.assertRaises(ValueError):
            validate(self.root, 2, 2)

    def test_invalid_actual_worker_id_fails(self):
        self.write(self.header + self.snapshot.replace("kernel_worker_ids=[0]", "kernel_worker_ids=[2]"))
        with self.assertRaises(ValueError):
            validate(self.root, 2, 2)

    def test_malformed_worker_id_delimiters_fail(self):
        for ids in ("[0,]", "[,0]", "[0,,]", "[0 1]"):
            with self.subTest(ids=ids):
                self.write(self.header + self.snapshot.replace("[0]", ids))
                with self.assertRaises(ValueError):
                    validate(self.root, 2, 2)

    def test_term_only_parallel_without_entries_fails(self):
        text = self.snapshot.replace("kernel_qp_items=3", "kernel_qp_items=0").replace(
            "kernel_parallel_qp_items=3", "kernel_parallel_qp_items=0").replace(
            "kernel_term_items=0", "kernel_term_items=1").replace(
            "kernel_parallel_term_items=0", "kernel_parallel_term_items=1").replace(
            "kernel_worker_entries=3", "kernel_worker_entries=0").replace(
            "kernel_workers_seen=1", "kernel_workers_seen=0").replace(
            "kernel_worker_ids=[0]", "kernel_worker_ids=[]")
        self.write(self.header + text)
        with self.assertRaises(ValueError):
            validate(self.root, 2, 2)

    def test_parallel_entries_lower_bound_fails(self):
        self.write(self.header + self.snapshot.replace("kernel_worker_entries=3", "kernel_worker_entries=2"))
        with self.assertRaises(ValueError):
            validate(self.root, 2, 2)

    def test_worker1_term_only_parallel_fails(self):
        header = self.header.replace("requested=2", "requested=1").replace(
            "configured=2", "configured=1").replace("observed_pool=2", "observed_pool=1")
        text = self.snapshot.replace("kernel_qp_items=3", "kernel_qp_items=0").replace(
            "kernel_parallel_qp_items=3", "kernel_parallel_qp_items=0").replace(
            "kernel_term_items=0", "kernel_term_items=1").replace(
            "kernel_parallel_term_items=0", "kernel_parallel_term_items=1").replace(
            "kernel_parallel_calls=1", "kernel_parallel_calls=0").replace(
            "kernel_worker_entries=3", "kernel_worker_entries=0").replace(
            "kernel_workers_seen=1", "kernel_workers_seen=0").replace(
            "kernel_worker_ids=[0]", "kernel_worker_ids=[]")
        self.write(header + text)
        with self.assertRaises(ValueError):
            validate(self.root, 2, 1)

    def test_serial_items_without_calls_fail(self):
        text = self.snapshot.replace("kernel_term_items=0", "kernel_term_items=1").replace(
            "kernel_serial_term_items=0", "kernel_serial_term_items=1")
        self.write(self.header + text)
        with self.assertRaises(ValueError):
            validate(self.root, 2, 2)

    def test_negative_malformed_and_extra_fields_fail(self):
        for replacement in ("kernel_worker_entries=-1", "kernel_worker_entries=x",
                            "kernel_worker_entries=3 unexpected=1"):
            with self.subTest(replacement=replacement):
                self.write(self.header + self.snapshot.replace("kernel_worker_entries=3", replacement))
                with self.assertRaises(ValueError):
                    validate(self.root, 2, 2)


if __name__ == "__main__":
    unittest.main()

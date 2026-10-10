"""Runner validation without compiling/running a reference implementation."""
import tempfile
import unittest
from pathlib import Path

from bench_mpi import parse_measurements, write_report


class EvidenceTests(unittest.TestCase):
    def test_requires_each_actual_rank_exactly_once(self):
        valid = "WORLD 1 2\nWORLD 0 2\nBENCH 1 2.5 -0.3\n"
        self.assertEqual(parse_measurements(valid, 2, 1), [(1, 2.5, -0.3)])
        for bad in (valid.replace("WORLD 1 2", "WORLD 1 1"),
                    valid.replace("WORLD 1 2", "WORLD 0 2"),
                    valid.replace("WORLD 1 2\n", ""),
                    valid + "WORLD 1 2\n"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_measurements(bad, 2, 1)

    def test_rejects_missing_duplicate_or_nonfinite_measurements(self):
        for bad in ("", "BENCH 2 1 -0.3\n", "BENCH 1 nan -0.3\n",
                    "BENCH 1 1 inf\n", "BENCH 1 0 -0.3\n",
                    "BENCH 1 1 -0.3\nBENCH 1 1 -0.3\n"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_measurements("WORLD 0 1\n" + bad, 1, 1)

    def test_requires_actual_threads_blas_and_kernel_workers(self):
        valid = "WORLD 0 1\nTHREADS 0 4\nBLAS_THREADS 0 1\nEXECUTION 0 20 4\nBENCH 1 2.5 -0.3\n"
        self.assertEqual(parse_measurements(valid, 1, 1, 4), [(1, 2.5, -0.3)])
        for bad in (valid.replace("THREADS 0 4", "THREADS 0 1"),
                    valid.replace("BLAS_THREADS 0 1", "BLAS_THREADS 0 4"),
                    valid.replace("EXECUTION 0 20 4", "EXECUTION 0 20 1"),
                    valid.replace("EXECUTION 0 20 4", "EXECUTION 0 0 4")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_measurements(bad, 1, 1, 4)
        serial = valid.replace("EXECUTION 0 20 4", "EXECUTION 0 0 0")
        self.assertEqual(parse_measurements(serial, 1, 1, 4, require_parallel=False),
                         [(1, 2.5, -0.3)])
        with self.assertRaises(ValueError):
            parse_measurements(valid, 1, 1, 4, require_native_blas=True)
        self.assertEqual(parse_measurements(valid + "NATIVE_BLAS_THREADS 0 1\n", 1, 1, 4,
                                           require_native_blas=True), [(1, 2.5, -0.3)])

    def test_report_keeps_thread_layouts_separate(self):
        rows = [dict(model="L16", implementation=impl, ranks=1, threads=threads,
                     seconds=seconds, samples_per_rank=300, steps=20)
                for impl in ("rust", "julia") for threads, seconds in ((1, 8), (4, 4))]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_report(output, rows, "total", 1)
            report = (output / "report.md").read_text()
            self.assertIn("| L16 | 1 | 1 | 300 | 8.000000 |", report)
            self.assertIn("| L16 | 1 | 4 | 300 | 4.000000 |", report)

    def test_throughput_distinguishes_total_and_per_rank_work(self):
        rows = [dict(model="L16", implementation=impl, ranks=ranks, seconds=seconds,
                     samples_per_rank=100, steps=20)
                for impl in ("rust", "julia") for ranks, seconds in ((1, 8), (4, 4))]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_report(output, rows, "total", 1)
            self.assertIn("| L16 | rust | 4x1 | 2.000 | 2.000 |", (output / "report.md").read_text())
            write_report(output, rows, "per-rank", 1)
            self.assertIn("| L16 | rust | 4x1 | 2.000 | 8.000 |", (output / "report.md").read_text())


if __name__ == "__main__":
    unittest.main()

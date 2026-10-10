"""Validate benchmark evidence and complete Markdown cells without runtimes."""
import argparse
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("bench_runner", Path(__file__).with_name("run.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ReportTests(unittest.TestCase):
    def test_rejects_missing_rank_and_incomplete_group(self):
        text = "WORLD 0 1\nTHREADS 0 4\nBLAS_THREADS 0 1\nBENCH 1 2.5 3\n"
        self.assertEqual(runner.measurements(text, 1, 4, 1, groups=3), [(1, 2.5, 3.0)])
        with self.assertRaises(ValueError):
            runner.measurements(text, 2, 4, 1)
        with self.assertRaises(ValueError):
            runner.measurements(text, 1, 4, 1, groups=4)

    def test_rejects_invalid_timing_and_repetition(self):
        metadata = "WORLD 0 1\nTHREADS 0 1\nBLAS_THREADS 0 1\n"
        for records in ("BENCH 1 nan 1\n", "BENCH 1 -1 1\n",
                        "BENCH 1 1 1\nBENCH 1 2 1\n", "BENCH 2 1 1\n"):
            with self.assertRaises(ValueError):
                runner.measurements(metadata + records, 1, 1, 1)

    def test_median_ratios_and_missing_cells(self):
        args = argparse.Namespace(ranks=4, threads=4, samples=300, steps=300,
                                  groups=100, warmups=1, reps=3, sites=[32, 64])
        rows = [dict(workload=mode, sites=size, implementation=impl, seconds=t * scale)
                for mode in ("Opt", "PhysCal") for size in args.sites
                for impl, scale in (("C", 1), ("Julia", 2), ("Rust", 0.5))
                for t in (1, 9, 3)]
        table = runner.report(rows, args)
        self.assertIn("| Opt | 32 | 3.000000 | 6.000000 | 1.500000 | 2.000 | 0.500 |", table)
        self.assertIn("| PhysCal | 64 |", table)
        with self.assertRaises(ValueError):
            runner.report(rows[:-1], args)


if __name__ == "__main__":
    unittest.main()

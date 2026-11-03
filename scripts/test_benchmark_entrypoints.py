"""Bounded orchestration regressions; no Rust/C/Julia/Docker/GPU execution."""
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('benchmark_entrypoints', ROOT / 'scripts/benchmark/run.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class BenchmarkEntrypoints(unittest.TestCase):
    def test_dry_run_from_unrelated_directory_has_no_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            for mode in ('cpu', 'gpu'):
                out = Path(tmp) / mode
                result = subprocess.run(['bash', str(ROOT / f'scripts/benchmark_{mode}.sh'),
                                         '--dry-run', '--output', str(out)],
                                        cwd=tmp, capture_output=True, text=True, check=True)
                self.assertIn('uv run --no-project python', result.stdout)
                self.assertIn('steps=20; samples=320', result.stdout)
                self.assertFalse(out.exists())

    def test_gpu_selection_preserves_visible_set_and_remaps_by_uuid(self):
        query = '0, GPU-aaa, RTX 3060, 580, 15 MiB\n1, GPU-bbb, RTX 3060, 580, 160 MiB\n'
        self.assertEqual(bench.select_gpu(query, None, '0,1'), 'GPU-aaa')
        self.assertEqual(bench.select_gpu(query, None, '1'), 'GPU-bbb')
        self.assertEqual(bench.select_gpu(query, '1', '0'), 'GPU-bbb')
        self.assertEqual(bench.select_gpu(query, None, 'GPU-bbb'), 'GPU-bbb')
        for visible in ('', '-1', '2', 'GPU-unknown'):
            with self.assertRaises(ValueError):
                bench.select_gpu(query, None, visible)

    def test_inherited_selectors_cannot_change_benchmark_work(self):
        with patch.dict(os.environ, {'MVMC_RS_MEASURE_BACKEND': 'cuda',
                                    'MVMC_RS_INNER_THRESHOLD': '999999',
                                    'MVMC_RS_INNER_PROFILE': '1',
                                    'JULIA_MVMC_INNER_THREADS': '16'}):
            env = bench.clean_environment()
        self.assertNotIn('MVMC_RS_MEASURE_BACKEND', env)
        self.assertNotIn('MVMC_RS_INNER_THRESHOLD', env)
        self.assertNotIn('MVMC_RS_INNER_PROFILE', env)
        self.assertEqual(env['JULIA_MVMC_INNER_THREADS'], '1')
        self.assertEqual(env['MVMC_RS_MEASURE_PF_BACKEND'], 'c-order')

    def test_incomplete_duplicate_and_nonfinite_repetitions_are_rejected(self):
        self.assertEqual(len(bench.parse_times('BENCH rep=0 seconds=2\nBENCH rep=1 seconds=1\n', 1)), 2)
        for text in ['BENCH rep=0 seconds=1', 'rep=0 seconds=1\nrep=0 seconds=1',
                     'rep=0 seconds=1\nrep=1 seconds=1e999', 'rep=0 seconds=1\nrep=1 seconds=-1']:
            with self.assertRaises(AssertionError):
                bench.parse_times(text, 1)

    def test_smoke_inputs_preserve_fixed_parameters_and_reference_sources(self):
        original = (bench.INPUTS / 'opt/modpara.def').read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench.prepare_inputs(root, 2, 16)
            for mode in ('opt', 'phys'):
                text = (root / f'inputs/{mode}/modpara.def').read_text()
                self.assertIn('NVMCSample 16', text)
                self.assertIn('NSROptItrStep 2', text)
            self.assertEqual(bench.sha(root / 'inputs/phys/zqp_opt.dat'), bench.PARAM_SHA)
        self.assertEqual((bench.INPUTS / 'opt/modpara.def').read_bytes(), original)

    def test_summary_never_publishes_missing_implementation(self):
        import argparse
        args = argparse.Namespace(mode='cpu', workloads=['opt'], threads=[1],
                                  steps=20, samples=320, reps=1, smoke=False)
        rows = [dict(workload='opt', threads=1, kind='Rust', rep=r, seconds=1.) for r in (0, 1)]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(AssertionError):
                bench.write_report(args, Path(tmp), rows)
            self.assertFalse((Path(tmp) / 'report.md').exists())


if __name__ == '__main__':
    unittest.main()

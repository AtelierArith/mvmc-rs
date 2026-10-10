import tempfile
import unittest
from pathlib import Path
from bench_native_c import compare_file


class NumericalComparison(unittest.TestCase):
    def compare(self, expected, actual):
        with tempfile.TemporaryDirectory() as scratch:
            a, b = Path(scratch) / "c", Path(scratch) / "rust"
            a.write_text(expected)
            b.write_text(actual)
            return compare_file(a, b, 1e-11, 1e-11)

    def test_rejects_missing_rows_and_columns(self):
        self.assertEqual(
            self.compare("1 2.0\n2 3.0\n", "1 2.0\n")["first"]["reason"], "shape"
        )
        self.assertEqual(self.compare("1 2.0\n", "1\n")["first"]["reason"], "shape")

    def test_discrete_indices_remain_exact(self):
        self.assertEqual(
            self.compare("1 2.0\n", "2 2.0\n")["first"]["reason"], "integer"
        )

    def test_roundoff_and_near_zero_use_explicit_bounds(self):
        self.assertEqual(
            self.compare("1 2.0 0.0\n", "1 2.000000000001 1e-12\n")["failures"], 0
        )
        self.assertEqual(self.compare("1 2.0\n", "1 2.01\n")["failures"], 1)

    def test_nonfinite_does_not_pass(self):
        self.assertEqual(self.compare("1 nan\n", "1 nan\n")["failures"], 1)
        self.assertEqual(self.compare("1 2.0\n", "1 inf\n")["failures"], 1)


class OutputSetComparison(unittest.TestCase):
    def test_optimization_filename_mapping_preserves_data_index(self):
        from bench_native_c import compare_outputs

        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            for name, filename in [("c", "zvo_out_007.dat"), ("rust", "zvo_out.dat")]:
                d = root / name
                (d / "output").mkdir(parents=True)
                (d / "modpara.def").write_text("NVMCCalMode 0\nNDataIdxStart 7\n")
                (d / "output" / filename).write_text("1.0 0.0\n")
            result = compare_outputs(root / "c", root / "rust", None, None)
            self.assertFalse(result["missing"])
            self.assertFalse(result["extra"])
            self.assertIn("zvo_out_007.dat", result["files"])

    def test_missing_and_extra_files_are_reported(self):
        from bench_native_c import compare_outputs

        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            for name in ["c", "rust"]:
                d = root / name
                (d / "output").mkdir(parents=True)
                (d / "modpara.def").write_text("NVMCCalMode 1\n")
            (root / "c/output/zvo_cisajs_001.dat").write_text("0 0 1.0\n")
            (root / "rust/output/zvo_out_001.dat").write_text("1.0\n")
            result = compare_outputs(root / "c", root / "rust", None, None)
            self.assertEqual(result["missing"], ["zvo_cisajs_001.dat"])
            self.assertEqual(result["extra"], ["zvo_out_001.dat"])


if __name__ == "__main__":
    unittest.main()

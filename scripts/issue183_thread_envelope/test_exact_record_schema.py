"""Synthetic-only source proposal controls; NOT RUN."""
from copy import deepcopy
import unittest
from exact_record_schema import validate


class Controls(unittest.TestCase):
    def setUp(self):
        self.settings = dict(npara=2, nelec=1, nqp=1, samples=2, onebody=1,
                             twobody=0, store=0, cg=0, input_seed=1, steps=20, window=20,
                             groups=1, ranks=1, all_complex=True, physical=True, launch_seed=1)
        self.source = {"emitter.rs": "a" * 64}
        self.inputs = {"orbital.def": "b" * 64}
        self.case = "case/exact"
        self.reviewed = {"schema": 1, "cases": [{"id": self.case,
            "settings": deepcopy(self.settings), "source": self.source.copy(),
            "inputs": self.inputs.copy(), "records": [
                {"label": "exact-pf-real", "kind": "pf-real", "dimension": 0,
                 "zero_reason": "inactive-real-buffer"},
                {"label": "exact-pf", "kind": "pf", "dimension": 2, "zero_reason": None},
                {"label": "exact-twobody", "kind": "twobody", "dimension": 0,
                 "zero_reason": "zero-green-definition-count"},
            ]}]}
        self.actual = {self.case: deepcopy(self.settings)}
        self.records = {"exact-pf-real": "", "exact-pf": "1.0 2.0", "exact-twobody": ""}

    def check(self):
        return validate(self.reviewed, self.actual, self.records, self.source, self.inputs, {self.case})

    def test_exact_structural_empty_not_numeric_comparison(self):
        self.assertEqual(self.check()["numeric_comparisons"], 0)

    def test_active_empty(self):
        self.records["exact-pf"] = ""
        with self.assertRaises(ValueError): self.check()

    def test_whitespace_not_zero_encoding(self):
        for payload in (" ", "\n", "0", "0.0"):
            self.records["exact-pf-real"] = payload
            with self.subTest(payload=payload), self.assertRaises(ValueError): self.check()

    def test_wrong_label_or_extra_record(self):
        self.records["other-pf-real"] = self.records.pop("exact-pf-real")
        with self.assertRaises(ValueError): self.check()

    def test_wrong_case(self):
        self.reviewed["cases"][0]["id"] = "case/other"
        with self.assertRaises(ValueError): self.check()

    def test_wrong_dimension_and_bool(self):
        record = self.reviewed["cases"][0]["records"][1]
        for value in (0, False, 2.0, 3):
            record["dimension"] = value
            with self.subTest(value=value), self.assertRaises(ValueError): self.check()

    def test_changed_runtime_settings(self):
        for key, value in (("all_complex", False), ("samples", 3), ("npara", 3),
                           ("twobody", 1), ("input_seed", 2), ("launch_seed", 2), ("steps", 3), ("cg", 1)):
            actual = deepcopy(self.settings)
            actual[key] = value
            self.actual[self.case] = actual
            with self.subTest(key=key), self.assertRaises(ValueError): self.check()

    def test_settings_bool_float_rejected(self):
        for key, value in (("samples", True), ("npara", 2.0), ("all_complex", 1)):
            self.actual[self.case] = dict(self.settings, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError): self.check()

    def test_missing_actual_layout(self):
        self.actual.clear()
        with self.assertRaises(ValueError): self.check()

    def test_changed_source_or_input(self):
        for closure in (self.source, self.inputs):
            key = next(iter(closure))
            original = closure[key]
            closure[key] = "changed"
            with self.assertRaises(ValueError): self.check()
            closure[key] = original

    def test_wildcard_and_duplicate_labels(self):
        record = self.reviewed["cases"][0]["records"][0]
        record["label"] = "*-pf-real"
        with self.assertRaises(ValueError): self.check()
        record["label"] = "exact-pf"
        with self.assertRaises(ValueError): self.check()

    def test_missing_structural_authority(self):
        self.reviewed["cases"][0]["records"][0]["zero_reason"] = None
        with self.assertRaises(ValueError): self.check()

    def test_nonfinite_and_wrong_active_length(self):
        for payload in ("NaN 1", "inf 1", "1", "1 2 3"):
            self.records["exact-pf"] = payload
            with self.subTest(payload=payload), self.assertRaises(ValueError): self.check()


if __name__ == "__main__":
    unittest.main()

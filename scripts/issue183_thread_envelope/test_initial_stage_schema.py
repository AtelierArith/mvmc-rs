"""Synthetic controls only; no RNG golden or model execution."""
import copy
import unittest

from initial_stage_schema import validate


class InitialStageTests(unittest.TestCase):
    def setUp(self):
        self.review = {"seed": dict(stage="seeded", launch_seed=1, npara=0, nqp=0),
                       "init": dict(stage="initialized", launch_seed=1, npara=2, nqp=1)}
        self.values = [dict(run_uuid="synthetic", invocation=0, label=label,
                            raw624=[0] * 624, next624=[1] * 624, index=624,
                            draw_count=0, **rule) for label, rule in self.review.items()]
        self.discrete = {}
        for item in self.values:
            label = item["label"]
            self.discrete.update({label + "-raw624": item["raw624"],
                                  label + "-rng": item["next624"],
                                  label + "-rng-index": item["index"],
                                  label + "-draw-count": item["draw_count"]})
        self.numeric = {"init-parameters": [0.] * 4, "init-qpweights": [1., 0.]}

    def check(self):
        return validate(self.values, self.review, "synthetic", 0, self.discrete, self.numeric)

    def test_shape_binding_is_not_numeric_verification(self):
        self.assertEqual(self.check()["independent_comparisons"], 0)

    def test_missing(self):
        self.values.pop()
        with self.assertRaises(ValueError): self.check()

    def test_duplicate(self):
        self.values.append(copy.deepcopy(self.values[0]))
        with self.assertRaises(ValueError): self.check()

    def test_wrong_stage(self):
        self.values[0]["stage"] = "initialized"
        with self.assertRaises(ValueError): self.check()

    def test_zero_active_even_when_self_reported(self):
        self.review["init"]["npara"] = self.values[1]["npara"] = 0
        with self.assertRaises(ValueError): self.check()

    def test_fabricated_state(self):
        self.numeric["seed-oo"] = []
        with self.assertRaises(ValueError): self.check()

    def test_bool_count(self):
        self.values[0]["draw_count"] = False
        with self.assertRaises(ValueError): self.check()

    def test_truncated_raw(self):
        self.values[0]["raw624"] = [0]
        with self.assertRaises(ValueError): self.check()

    def test_wrong_stdout(self):
        self.discrete["seed-rng-index"] = 1
        with self.assertRaises(ValueError): self.check()

    def test_no_shared_alias(self):
        self.values[1]["label"] = "seed"
        with self.assertRaises(ValueError): self.check()


if __name__ == "__main__":
    unittest.main()

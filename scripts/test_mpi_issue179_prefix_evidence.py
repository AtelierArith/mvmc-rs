"""Synthetic checker-protocol regressions, not MPI/model evidence."""
import tempfile
import unittest
import hashlib
import json
from pathlib import Path

from mpi_issue179_prefix_evidence import validate, validate_build


class PrefixEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.populate()

    def populate(self, ranks=2, width=2):
        for steps in (1, 2, 3):
            root = self.root / f"prefix{steps}" / "w1"
            root.mkdir(parents=True, exist_ok=True)
            for repeat in (1, 2):
                (root / f"launch{repeat}.exit").write_text("0\n")
                directory = root / f"repeat{repeat}"
                directory.mkdir(exist_ok=True)
                for name in ("zvo_out.dat", "zvo_var.dat", "zqp_opt.dat"):
                    (directory / name).write_text("checker-protocol-only\n")
                for rank in range(ranks):
                    rows = {"steps": [steps], "status": [0], "requested-width": [width],
                            "group": [rank // width, rank % width, width],
                            "seed": [7 + rank // width], "configured-samples": [1],
                            "chain-samples": [1], "initial-draw-count": [0],
                            "total-draw-count": [steps], "sampling-draw-count": [steps],
                            "before-init-rng-cursor": [624], "initial-rng-cursor": [624],
                            "rng-cursor": [steps], "ele_idx": [0], "ele_cfg": [0],
                            "ele_num": [1], "counter": [0] * 10,
                            "acceptance-events": [steps, 0]}
                    for key in ("before-init-raw-rng", "initial-raw-rng", "raw-rng", "initial-rng", "rng"):
                        rows[key] = [0] * 624
                    checkpoint = [8]
                    for values in ([0], [0], [1], [0], [0], [0] * 10, [0] * 624):
                        checkpoint += [len(values), *values]
                    events = []
                    for _ in range(steps):
                        events += [[1, 0, 0, 1, 0, 0], [9, 123], [7, 1, 123], checkpoint]
                    rows["trace-events"] = [len(events)]
                    rows.update({f"trace-{i:06}": event for i, event in enumerate(events)})
                    (directory / f"rank-{rank}.txt").write_text("".join(
                        f"d:{key} {' '.join(map(str, values))}\n" for key, values in rows.items()))

    def mutate(self, key, values=None, rank=0, steps=3, repeats=(1, 2), kind="d"):
        for repeat in repeats:
            file = self.root / f"prefix{steps}/w1/repeat{repeat}/rank-{rank}.txt"
            lines = [line for line in file.read_text().splitlines() if not line.startswith(f"{kind}:{key} ")]
            if values is not None:
                lines.append(f"{kind}:{key} " + " ".join(map(str, values)))
            file.write_text("\n".join(lines) + "\n")

    def populate_store1(self):
        self.populate()
        for steps in (1, 2, 3):
            for rank in (0, 1):
                self.mutate("sr-kind", [0], rank=rank, steps=steps)
                self.mutate("sr-systems", [steps], rank=rank, steps=steps)
                for key in ("initial-parameters", "parameters"):
                    self.mutate(key, [1.0, 0.0], rank=rank, steps=steps, kind="n")
                for step in range(steps):
                    stem = f"sr-system-{step:06}"
                    for key, values in {"settings": [7, steps, steps, 0, 1],
                                        "triangle": [85], "nrhs": [1], "status": [0],
                                        "dimension": [1], "not-solved": [0],
                                        "active": [0], "flags": [1, 0],
                                        "factor-info": [0], "solve-info": [0]}.items():
                        self.mutate(stem + "-" + key, values, rank=rank, steps=steps)
                    self.mutate(stem + "-regularization", [1e-5, 1e-10, 0.01], rank=rank, steps=steps, kind="n")
                    for name in ("matrix", "rhs", "increment"):
                        self.mutate(stem + "-" + name, [1.0], rank=rank, steps=steps, kind="n")
                    self.mutate(f"sr-step-{step}", [1.0, 0.0], rank=rank, steps=steps, kind="n")
                for repeat in (1, 2):
                    (self.root / f"prefix{steps}/w1/repeat{repeat}/cg-rank-{rank}.txt").write_text("d:events 0\n")

    def test_store1_actual_metadata_passes(self):
        self.populate_store1()
        self.assertIn("actual_sr_settings=verified", validate(self.root, 2, 2, "direct-store1"))

    def test_store1_missing_actual_metadata_fails(self):
        for key in ("sr-kind", "sr-systems", "sr-system-000000-settings",
                    "sr-system-000000-factor-info", "sr-system-000000-solve-info"):
            with self.subTest(key=key):
                self.populate_store1()
                self.mutate(key)
                with self.assertRaises(ValueError):
                    validate(self.root, 2, 2, "direct-store1")

    def test_store1_wrong_actual_settings_fails(self):
        for values in ([7, 3, 3, 0, 0], [7, 3, 3, 1, 1], [8, 3, 3, 0, 1],
                       [7, 2, 3, 0, 1], [7, 3, 2, 0, 1], [7, 3, 3, 0]):
            with self.subTest(values=values):
                self.populate_store1()
                self.mutate("sr-system-000000-settings", values)
                with self.assertRaisesRegex(ValueError, "settings"):
                    validate(self.root, 2, 2, "direct-store1")

    def test_store1_wrong_system_count_fails(self):
        self.populate_store1()
        self.mutate("sr-systems", [2])
        with self.assertRaisesRegex(ValueError, "observation"):
            validate(self.root, 2, 2, "direct-store1")

    def test_store1_missing_sidecar_fails(self):
        self.populate_store1()
        (self.root / "prefix1/w1/repeat1/cg-rank-0.txt").unlink()
        with self.assertRaises(OSError):
            validate(self.root, 2, 2, "direct-store1")

    def test_store1_cg_events_and_srinfo_fail(self):
        self.populate_store1()
        file = self.root / "prefix1/w1/repeat1/cg-rank-0.txt"
        for text in ("d:events 1\n", "d:events 0\nd:extra 1\n"):
            file.write_text(text)
            with self.assertRaisesRegex(ValueError, "sidecar"):
                validate(self.root, 2, 2, "direct-store1")
        file.write_text("d:events 0\n")
        (file.parent / "zvo_SRinfo.dat").write_text("unexpected CG output\n")
        with self.assertRaisesRegex(ValueError, "SRinfo"):
            validate(self.root, 2, 2, "direct-store1")

    def test_store1_native_failure_is_not_expected_success(self):
        self.populate_store1()
        (self.root / "prefix1/w1/launch1.exit").write_text("101\n")
        with self.assertRaisesRegex(ValueError, "launcher"):
            validate(self.root, 2, 2, "direct-store1")

    def test_store1_failed_or_missing_operands_fail(self):
        self.populate_store1()
        self.mutate("sr-system-000000-factor-info", [1])
        with self.assertRaisesRegex(ValueError, "INFO"):
            validate(self.root, 2, 2, "direct-store1")
        for key, values in (("matrix", None), ("rhs", [float("nan")]), ("increment", [1, 2])):
            with self.subTest(key=key):
                self.populate_store1()
                self.mutate("sr-system-000000-" + key, values, kind="n")
                with self.assertRaisesRegex(ValueError, "operands"):
                    validate(self.root, 2, 2, "direct-store1")

    def test_store1_observed_not_solved_is_explicit(self):
        self.populate_store1()
        stem = "sr-system-000000"
        for key, values in (("not-solved", [2]), ("dimension", [0]),
                            ("active", None), ("factor-info", None), ("solve-info", None)):
            self.mutate(stem + "-" + key, values, steps=1)
        for key in ("matrix", "rhs", "increment"):
            self.mutate(stem + "-" + key, steps=1, kind="n")
        self.assertIn("verified", validate(self.root, 2, 2, "direct-store1"))
        self.mutate(stem + "-solve-info", [0], steps=1)
        with self.assertRaisesRegex(ValueError, "not-solved"):
            validate(self.root, 2, 2, "direct-store1")

    def test_cg_width3_stage_not_enabled(self):
        for stage in ("cg-store1", "direct-store2", ""):
            with self.assertRaisesRegex(ValueError, "stage"):
                validate(self.root, 2, 2, stage)
        with self.assertRaisesRegex(ValueError, "axes"):
            validate(self.root, 4, 3, "direct-store1")

    def test_complete_protocol(self):
        self.assertEqual("PREFIX_GROUP_EVIDENCE ranks=2 width=2 prefixes=1,2,3 repeats=2 workers=1",
                         validate(self.root, 2, 2))

    def test_store1_orphan_system_metadata_fails(self):
        self.populate_store1()
        self.mutate("sr-system-000003-dimension", [1])
        with self.assertRaisesRegex(ValueError, "extra actual SR"):
            validate(self.root, 2, 2, "direct-store1")

    def test_store1_missing_nonfinite_update_fails(self):
        for values in (None, [1], [1, 0, 2, 0], [1, float("inf")]):
            with self.subTest(values=values):
                self.populate_store1()
                self.mutate("sr-step-0", values, kind="n")
                with self.assertRaisesRegex(ValueError, "update"):
                    validate(self.root, 2, 2, "direct-store1")

    def test_store1_active_missing_extra_out_of_range_fails(self):
        for values in (None, [0, 0], [0, 1], [-1], [2], [1]):
            with self.subTest(values=values):
                self.populate_store1()
                self.mutate("sr-system-000000-active", values)
                with self.assertRaisesRegex(ValueError, "active"):
                    validate(self.root, 2, 2, "direct-store1")

    def test_store1_flags_missing_wrong_shape_or_integer_fails(self):
        for values in (None, [1], [1, 0, 0], ["NaN", 0], ["1.5", 0]):
            with self.subTest(values=values):
                self.populate_store1()
                self.mutate("sr-system-000000-flags", values)
                with self.assertRaises(ValueError):
                    validate(self.root, 2, 2, "direct-store1")

    def test_store1_regularization_missing_nonfinite_wrong_shape_fails(self):
        for values in (None, [1e-5, 1e-10], [1e-5, 1e-10, 0.01, 1],
                       [float("nan"), 1e-10, 0.01], [1e-5, float("inf"), 0.01]):
            with self.subTest(values=values):
                self.populate_store1()
                self.mutate("sr-system-000000-regularization", values, kind="n")
                with self.assertRaisesRegex(ValueError, "regularization"):
                    validate(self.root, 2, 2, "direct-store1")

    def test_store1_parameter_count_records_missing_or_inconsistent_fail(self):
        for key, values in (("initial-parameters", None), ("initial-parameters", [1]),
                            ("parameters", [1, 0, 2, 0]), ("parameters", [float("nan"), 0])):
            with self.subTest(key=key, values=values):
                self.populate_store1()
                self.mutate(key, values, kind="n")
                with self.assertRaisesRegex(ValueError, "parameter count"):
                    validate(self.root, 2, 2, "direct-store1")

    def test_store1_no_parameters_contradicts_actual_parameter_records(self):
        self.populate_store1()
        self.mutate("sr-system-000000-not-solved", [1])
        with self.assertRaisesRegex(ValueError, "NoParameters"):
            validate(self.root, 2, 2, "direct-store1")

    def test_width1_independent_chains(self):
        self.populate(width=1)
        self.assertIn("width=1", validate(self.root, 2, 1))

    def test_missing_raw_and_count(self):
        for key in ("raw-rng", "rng-cursor", "initial-raw-rng", "total-draw-count", "acceptance-events"):
            with self.subTest(key=key):
                self.populate()
                self.mutate(key)
                with self.assertRaises(ValueError):
                    validate(self.root, 2, 2)

    def test_malformed_words(self):
        for values in ([0] * 623, [2**32] * 624, [-1] * 624):
            self.mutate("raw-rng", values)
            with self.assertRaises(ValueError):
                validate(self.root, 2, 2)

    def test_cursor_count_mismatch(self):
        self.mutate("rng-cursor", [4])
        with self.assertRaisesRegex(ValueError, "cursor"):
            validate(self.root, 2, 2)

    def test_group_sampling_difference(self):
        self.mutate("raw-rng", [1] * 624, rank=1)
        with self.assertRaisesRegex(ValueError, "group sampling"):
            validate(self.root, 2, 2)

    def test_common_prefix_difference(self):
        for rank in (0, 1):
            self.mutate("trace-000000", [1, 0, 0, 2, 0, 0], rank=rank)
        with self.assertRaisesRegex(ValueError, "common prefix"):
            validate(self.root, 2, 2)

    def test_missing_prefix(self):
        (self.root / "prefix2/w1/launch1.exit").unlink()
        with self.assertRaises(OSError):
            validate(self.root, 2, 2)

    def test_initial_prefix_state_difference(self):
        for rank in (0, 1):
            self.mutate("initial-raw-rng", [1] * 624, rank=rank)
        with self.assertRaisesRegex(ValueError, "initial prefix"):
            validate(self.root, 2, 2)

    def test_nonzero_launch(self):
        (self.root / "prefix3/w1/launch1.exit").write_text("101\n")
        with self.assertRaisesRegex(ValueError, "launcher"):
            validate(self.root, 2, 2)

    def test_reject_flag_domain(self):
        self.mutate("trace-000000", [1, 0, 0, 1, 0, 2])
        with self.assertRaisesRegex(ValueError, "reject"):
            validate(self.root, 2, 2)

    def test_decision_word_mismatch(self):
        self.mutate("trace-000002", [7, 1, 124])
        with self.assertRaisesRegex(ValueError, "consumed word"):
            validate(self.root, 2, 2)

    def test_bad_checkpoint(self):
        self.mutate("trace-000003", [8, 1000, 0])
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            validate(self.root, 2, 2)

    def test_base_offset_mismatch(self):
        self.populate(width=1)
        self.mutate("seed", [99], rank=1, steps=1)
        with self.assertRaisesRegex(ValueError, "base seed"):
            validate(self.root, 2, 1)

    def test_repeat_difference(self):
        self.mutate("raw-rng", [1] * 624, repeats=(2,))
        with self.assertRaisesRegex(ValueError, "repeat mismatch"):
            validate(self.root, 2, 2)

    def test_build_association_rejects_drift(self):
        snapshot = self.root / "snapshot"
        (snapshot / "scripts").mkdir(parents=True)
        names = ("mpi_issue179_prefix_evidence.py", "test_mpi_issue179_prefix_evidence.py",
                 "verify_mpi_issue179_prefixes.sh")
        for name in names:
            (snapshot / "scripts" / name).write_text("synthetic provenance test only\n")
        commit = "a" * 40
        for name in ("build-command.txt", "build.log", "build.json", "compiler.txt", "baseline.sha256"):
            (snapshot / name).write_text("synthetic provenance test only\n")
        (snapshot / "git-rev-parse.txt").write_text(commit + "\n")
        (snapshot / "build.exit").write_text("0\n")
        binary, manifest, file = (snapshot / name for name in ("binary", "manifest", "proof.json"))
        binary.write_text("synthetic binary\n")
        manifest.write_text("synthetic manifest\n")
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        proof = {"source_commit": commit, "build_exit": 0,
                 "binary_sha256": digest(binary), "source_manifest_sha256": digest(manifest),
                 "owner_overlay_sha256": {"scripts/" + name: digest(snapshot / "scripts" / name) for name in names},
                 "build_records_sha256": {name: digest(snapshot / name) for name in (
                     "build-command.txt", "build.log", "build.exit", "build.json", "compiler.txt", "git-rev-parse.txt", "baseline.sha256")}}
        file.write_text(json.dumps(proof))
        validate_build(file, binary, manifest, commit, snapshot)
        for key, value in (("build_exit", 101), ("source_commit", "b" * 40),
                           ("binary_sha256", "0" * 64), ("source_manifest_sha256", "0" * 64),
                           ("owner_overlay_sha256", {}), ("build_records_sha256", {})):
            with self.subTest(key=key):
                file.write_text(json.dumps(dict(proof, **{key: value})))
                with self.assertRaises(ValueError):
                    validate_build(file, binary, manifest, commit, snapshot)


if __name__ == "__main__":
    unittest.main()

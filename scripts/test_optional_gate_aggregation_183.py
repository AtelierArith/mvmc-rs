"""Synthetic infrastructure evidence only; never runs numerical gates/oracles."""
import copy
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

import aggregate_optional_gates_183 as audit
from run_optional_gates_183 import family_ledger

BINDING = audit.identity("a" * 40, "123", "2", "owner/repo/.github/workflows/optional-gates.yml@refs/heads/main")


def put(path, value):
    path.write_text(json.dumps(value) + "\n")


def fixture(root, packages, plan, family):
    evidence = root / f"source-{family}"
    evidence.mkdir()
    rows = family_ledger(family)
    identities = {"general": list(audit.GENERAL), "lanczos": [audit.LANCZOS],
                  "mpi": [audit.MPI], "thread": [audit.THREAD]}[family]
    completed = {"general": list(audit.GENERAL), "lanczos": [f"{m}-{c}" for m in audit.MODELS for c in ("real", "cmp")],
                 "mpi": ["world2", "world4"], "thread": [audit.THREAD]}[family]
    rows[family].update(status="Pass", started_driver_invocations={"general": 1, "lanczos": 6, "mpi": 2, "thread": 1}[family],
                        selected_test_identities=identities, completed_selection_identities=completed)
    if family == "lanczos":
        rows[family].update(numeric_reference_comparisons=8, empty_contracts=4, comparison_evidence="Verified")
        records = [{"model": m, "mode": c, "file": f, "status": s}
                   for m in audit.MODELS[1:] for c in ("real", "cmp")
                   for f, s in (("zvo_ls_cisajs_001.dat", "REFERENCE_COMPARED"),
                                ("zvo_ls_cisajscktalt_001.dat", "REFERENCE_COMPARED"),
                                ("zvo_ls_cisajscktaltex_001.dat", "EMPTY_CONTRACT"))]
        put(evidence / "dc-comparisons.json", {"records": records})
    put(evidence / "terminal.json", {"status": "Pass", "exit_status": 0, "completed": completed})
    put(evidence / "family-ledger.json", {"families": rows})
    settings = {
        "general": {"model": "general_rbm_cmp", "seed": 12395, "prefixes_and_windows": [1, 2, 3, 20], "NSRCG": 0, "NStore": 1, "ranks": 1, "workers": 1, "threshold": 32},
        "lanczos": {"models": list(audit.MODELS), "modes": ["real", "cmp"], "seed_override": 1, "ranks": 1, "groups": 1, "sample_counts": [100, 1000, 5000], "workers": 1},
        "mpi": {"model": "heisenberg_chain_real", "ranks": [2, 4], "group_widths": [1, 2], "repeats": 2, "seed": 1, "steps": 1, "samples": 3, "warmup": 1, "workers": 1},
        "thread": {"workers": [1, 2, 4], "threshold": 32, "sizes": [31, 32, 33], "steps": 2, "samples": 200, "ranks": 1, "seed": 1},
    }[family]
    put(evidence / "metadata.json", {"head": plan["head"], "family": family, "profile": "test-fast",
        "features": "mpi" if family == "mpi" else "default", "configuration": settings,
        "compiler_environment": {"CARGO_TARGET_DIR": "/synthetic/target"}})
    for stem in ("source", "fixtures", "binary"):
        put(evidence / f"{stem}.before.json", {"synthetic-path": "b" * 64})
        put(evidence / ("binary.json" if stem == "binary" else f"{stem}.after.json"), {"synthetic-path": "b" * 64})
    put(evidence / "backend.json", {"actual_threads": 1, "actual_config": "OpenBLAS 0.3.26 synthetic",
                                   "actual_core": "Haswell", "library_sha256": "c" * 64})
    for name in ("commands.json", "selection.json"):
        put(evidence / name, {"synthetic": True})
    for name in ("linkage.txt", "rust-version.stdout", "nextest-version.stdout", "reference-revisions.stdout"):
        (evidence / name).write_text("synthetic infrastructure evidence only\n")
    put(evidence / "artifacts.json", {str(path): audit.digest(path) for path in evidence.iterdir()})
    package = packages / audit.artifact_name(family, plan)
    with patch.object(audit, "checkout_head", return_value=plan["head"]):
        audit.seal(plan, family, evidence, package, "success", plan["head"], "never")
    return package


def refresh(package):
    envelope = audit.read_json(package / "envelope.json")
    evidence = package / "evidence"
    put(evidence / "artifacts.json", {
        str(Path(envelope["evidence_root"]) / path.relative_to(evidence)): audit.digest(path)
        for path in evidence.rglob("*") if path.is_file() and path.name != "artifacts.json"})
    envelope["driver_hashes"] = {name: audit.digest(evidence / name)
                                for name in ("terminal.json", "family-ledger.json", "artifacts.json")}
    put(package / "envelope.json", envelope)


class AggregationContract(unittest.TestCase):
    def test_actual_checkout_head_must_match_expected_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = audit.make_plan("general", BINDING)
            output = root / "package"
            with patch.object(audit.subprocess, "check_output", return_value="b" * 40 + "\n") as git:
                with self.assertRaises(ValueError):
                    audit.seal(plan, "general", root / "absent", output, "failure", BINDING["head"], "never")
                git.assert_called_once()
                self.assertEqual(git.call_args.args[0], ["git", "rev-parse", "HEAD"])
                self.assertFalse(output.exists())
            with patch.object(audit.subprocess, "check_output", return_value=BINDING["head"] + "\n"):
                audit.seal(plan, "general", root / "absent", output, "failure", BINDING["head"], "never")
                self.assertEqual(audit.read_json(output / "envelope.json")["checkout_head"], BINDING["head"])

    def test_cli_plan_envelope_and_no_evidence_failure_without_gate_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = ["--head", BINDING["head"], "--run-id", BINDING["run_id"],
                       "--attempt", BINDING["attempt"], "--workflow", BINDING["workflow"]]
            plan_path = root / "plan.json"
            with patch.dict(audit.os.environ, {"GITHUB_OUTPUT": str(root / "outputs")}), \
                    patch.object(audit.sys, "argv", ["audit", "plan", *context, "--scope", "general", "--output", str(plan_path)]):
                self.assertEqual(audit.main(), 0)
            self.assertIn('"general"', (root / "outputs").read_text())
            packages = root / "packages"
            packages.mkdir()
            package = packages / audit.artifact_name("general", audit.read_json(plan_path))
            with patch.dict(audit.os.environ, {"CARGO_TERM_COLOR": "never"}), \
                    patch.object(audit, "checkout_head", return_value=BINDING["head"]), \
                    patch.object(audit.sys, "argv", ["audit", "seal", *context, "--plan", str(plan_path),
                        "--family", "general", "--evidence", str(root / "absent"), "--job-status", "failure", "--output", str(package)]):
                self.assertEqual(audit.main(), 0)
            result = root / "result.json"
            with patch.object(audit.sys, "argv", ["audit", "aggregate", *context, "--plan", str(plan_path),
                        "--packages", str(packages), "--output", str(result)]):
                self.assertEqual(audit.main(), 1)
            self.assertEqual(audit.read_json(result)["families"]["general"]["status"], "Failure")
            missing_plan_result = root / "missing-plan-result.json"
            with patch.object(audit.sys, "argv", ["audit", "aggregate", *context, "--plan", str(root / "missing-plan"),
                        "--packages", str(packages), "--output", str(missing_plan_result)]):
                self.assertEqual(audit.main(), 1)
            self.assertEqual(audit.read_json(missing_plan_result)["status"], "Failure")

    def test_plan_full_family_selection_and_binding_fail_closed(self):
        for scope in (*audit.FAMILIES, "all"):
            plan = audit.make_plan(scope, BINDING)
            audit.validate_plan(plan, BINDING)
            self.assertEqual(set(plan["families"]), set(audit.FAMILIES))
            self.assertTrue(all(row["status"] == "NotRun" for row in plan["families"].values()))
        for scope in ("", "unknown", "general;echo injected"):
            with self.assertRaises(ValueError):
                audit.make_plan(scope, BINDING)
        for key in audit.BINDING:
            changed = dict(BINDING, **{key: "wrong"})
            with self.assertRaises(ValueError):
                audit.validate_plan(plan, changed)
        for selected in ([], ["general", "general"], ["unknown"]):
            mutant = copy.deepcopy(plan)
            mutant["selected"] = selected
            with self.assertRaises(ValueError):
                audit.validate_plan(mutant, BINDING)

    def test_all_four_synthetic_packages_pass_without_inventing_numeric_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packages = root / "packages"
            packages.mkdir()
            plan = audit.make_plan("all", BINDING)
            for family in audit.FAMILIES:
                fixture(root, packages, plan, family)
            result = audit.aggregate(plan, packages, BINDING)
            self.assertEqual(result["exit_status"], 0)
            self.assertTrue(all(row["status"] == "Pass" for row in result["families"].values()))
            self.assertEqual(result["families"]["lanczos"]["numeric_reference_comparisons"], 8)
            self.assertEqual(result["families"]["lanczos"]["empty_contracts"], 4)
            self.assertIsNone(result["families"]["general"]["numeric_reference_comparisons"])

    def test_unselected_notrun_missing_duplicate_wrong_attempt_packages_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packages = root / "packages"
            packages.mkdir()
            plan = audit.make_plan("general", BINDING)
            missing = audit.aggregate(plan, packages, BINDING)
            self.assertEqual(missing["families"]["general"]["status"], "Failure")
            self.assertEqual(missing["families"]["mpi"]["status"], "NotRun")
            fixture(root, packages, plan, "general")
            extra = packages / "optional-family-general-123-1"
            extra.mkdir()
            self.assertEqual(audit.aggregate(plan, packages, BINDING)["exit_status"], 1)

    def test_envelope_head_run_attempt_workflow_color_and_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packages = root / "packages"
            packages.mkdir()
            plan = audit.make_plan("general", BINDING)
            package = fixture(root, packages, plan, "general")
            original = audit.read_json(package / "envelope.json")
            for key in (*audit.BINDING, "family", "checkout_head"):
                mutant = dict(original, **{key: "wrong"})
                put(package / "envelope.json", mutant)
                self.assertEqual(audit.aggregate(plan, packages, BINDING)["exit_status"], 1)
            mutant = dict(original, gate_environment={"CARGO_TERM_COLOR": "always"})
            put(package / "envelope.json", mutant)
            self.assertEqual(audit.aggregate(plan, packages, BINDING)["exit_status"], 1)
            put(package / "envelope.json", original)
            (package / "evidence/terminal.json").write_text("tampered")
            self.assertEqual(audit.aggregate(plan, packages, BINDING)["exit_status"], 1)

    def test_rehashed_semantic_status_identity_closure_backend_mutations_fail(self):
        mutations = [
            ("metadata.json", lambda data: data.update(head="d" * 40)),
            ("metadata.json", lambda data: data["configuration"].update(seed=99)),
            ("terminal.json", lambda data: data.update(exit_status=1)),
            ("terminal.json", lambda data: data.update(completed=[])),
            ("family-ledger.json", lambda data: data["families"]["general"].update(selected_test_identities=["helper"])),
            ("family-ledger.json", lambda data: data["families"]["general"].update(started_driver_invocations=2)),
            ("family-ledger.json", lambda data: data["families"]["general"].update(numeric_reference_comparisons=8)),
            ("source.after.json", lambda data: data.update(extra="e" * 64)),
            ("backend.json", lambda data: data.update(actual_threads=2)),
            ("backend.json", lambda data: data.update(actual_config="OpenBLAS unknown")),
        ]
        for name, mutation in mutations:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                packages = root / "packages"
                packages.mkdir()
                plan = audit.make_plan("general", BINDING)
                package = fixture(root, packages, plan, "general")
                path = package / "evidence" / name
                data = audit.read_json(path)
                mutation(data)
                put(path, data)
                refresh(package)
                self.assertEqual(audit.aggregate(plan, packages, BINDING)["exit_status"], 1, name)

    def test_missing_nonempty_dc_reference_not_replaced_by_empty_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packages = root / "packages"
            packages.mkdir()
            plan = audit.make_plan("lanczos", BINDING)
            package = fixture(root, packages, plan, "lanczos")
            path = package / "evidence/dc-comparisons.json"
            data = audit.read_json(path)
            data["records"][0]["status"] = "EMPTY_CONTRACT"
            put(path, data)
            refresh(package)
            self.assertEqual(audit.aggregate(plan, packages, BINDING)["exit_status"], 1)

    def test_prebuild_no_evidence_cancel_and_failed_driver_never_pass(self):
        for status in ("failure", "cancelled", "skipped"):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                packages = root / "packages"
                packages.mkdir()
                plan = audit.make_plan("general", BINDING)
                package = packages / audit.artifact_name("general", plan)
                with patch.object(audit, "checkout_head", return_value=plan["head"]):
                    audit.seal(plan, "general", root / "absent-evidence", package, status, plan["head"], "never")
                self.assertEqual(audit.aggregate(plan, packages, BINDING)["exit_status"], 1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packages = root / "packages"
            packages.mkdir()
            plan = audit.make_plan("general", BINDING)
            package = fixture(root, packages, plan, "general")
            terminal = audit.read_json(package / "evidence/terminal.json")
            terminal.update(exit_status=1, status="MissingFixture")
            put(package / "evidence/terminal.json", terminal)
            ledger = audit.read_json(package / "evidence/family-ledger.json")
            ledger["families"]["general"]["status"] = "MissingFixture"
            put(package / "evidence/family-ledger.json", ledger)
            refresh(package)
            envelope = audit.read_json(package / "envelope.json")
            envelope["job_status"] = "failure"
            put(package / "envelope.json", envelope)
            result = audit.aggregate(plan, packages, BINDING)
            self.assertEqual(result["exit_status"], 1)
            self.assertEqual(result["families"]["general"]["status"], "MissingFixture")

    def test_unsafe_paths_symlinks_json_duplicates_and_reuse_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "duplicate.json"
            path.write_text('{"status":"Pass","status":"Failure"}')
            with self.assertRaises(ValueError):
                audit.read_json(path)
            evidence = root / "evidence"
            evidence.mkdir()
            (evidence / "actual").write_text("actual")
            for name in ("/outside/actual", "/original/../actual", "relative"):
                with self.assertRaises(ValueError):
                    audit.validate_manifest(evidence, "/original", {name: "b" * 64})
            (evidence / "link").symlink_to(evidence / "actual")
            with self.assertRaises(ValueError):
                audit.safe_tree(evidence)
            with self.assertRaises(FileExistsError):
                audit.write_new(path, {})


def historical_evidence_regression(evidence):
    """Explicit readonly real schema audit, NOT a current dispatch/gate execution."""
    audit.validate_manifest(evidence, str(evidence.resolve()), audit.read_json(evidence / "artifacts.json"))
    assert audit.digest(evidence / "terminal.json") == "29a66818dcdc58907e95109dced643ff89a9b53b801126e4c5f24a72c0d37214"
    assert audit.digest(evidence / "source.before.json") == "1b62cafd5335bb35649e1cf4a0cb535e83df46127c02d0e09cbbc889f7821f23"
    metadata = audit.read_json(evidence / "metadata.json")
    assert metadata["head"] == "c8db43bf461fe6fb426b5fee117319849c1b9c21"
    assert metadata["family"] == "lanczos" and metadata["profile"] == "test-fast" and metadata["features"] == "default"
    assert metadata["configuration"]["seed_override"] == 1
    assert metadata["configuration"]["modes"] == ["real", "cmp"]
    terminal = audit.read_json(evidence / "terminal.json")
    assert terminal["exit_status"] == 0
    assert terminal["completed"] == [f"{model}-{mode}" for model in audit.MODELS for mode in ("real", "cmp")]
    for stem in ("source", "fixtures", "binary"):
        audit.stable_closure(evidence, stem)
    backend = audit.read_json(evidence / "backend.json")
    assert backend["actual_threads"] == 1 and backend["actual_config"].startswith("OpenBLAS 0.3.26 ")
    dc = audit.read_json(evidence / "dc-comparisons.json")
    assert len(dc["records"]) == 12 and dc["reference_comparisons"] == 8 and dc["empty_contracts"] == 4
    assert dc["empty_contracts_are_numeric_comparisons"] is False
    for name in ("commands.json", "selection.json", "linkage.txt", "rust-version.stdout", "nextest-version.stdout", "reference-revisions.stdout"):
        assert audit.regular(evidence / name).stat().st_size > 0
    # This historical producer predates the new ledger; do not manufacture one.
    assert not (evidence / "family-ledger.json").exists()
    print("historical real driver schema/hash regression PASS: original c8db43bf/handle22750 only; NO current run/head/attempt binding or numerical rerun")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--historical-evidence":
        historical_evidence_regression(Path(sys.argv[2]))
    else:
        unittest.main()

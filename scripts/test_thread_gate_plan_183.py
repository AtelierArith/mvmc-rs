#!/usr/bin/env python3
"""Synthetic infrastructure controls only; never execute numerical gates."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import thread_gate_plan_183 as gate


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="thread-plan-183-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = {gate.TEST_SOURCE: gate.AUDITED_TEST_SHA}
        self.fixtures = {
            root + "/fixture.txt": "a" * 64
            for case in gate.cases() for root in case["reference_roots"]
        }
        self.fixtures.update({path: "a" * 64 for path in gate.required_fixture_files(list(gate.IDENTITIES))})

    def plan(self, selected=None):
        selected = selected or ["transfer"]
        closure = {path: {path: self.fixtures[path]} for path in gate.required_fixture_files(selected)
                   if path.endswith("/namelist.def")}
        return gate.build_plan(selected, "1" * 40, self.source, self.fixtures, closure)

    def evidence(self, plan, status="Pass", invocations=1):
        evidence = {"schema": 1, **{key: copy.deepcopy(plan[key]) for key in ("head", "source", "fixtures")},
                "binary": {"executed-test": "b" * 64},
                "runtime": {"profile": "test-fast", "features": "default", "ranks": 1,
                            "rust_version": "SYNTHETIC rust test metadata, not execution evidence",
                            "blas": {"version": "SYNTHETIC", "threads": 1,
                                     "observation": "runtime-query", "libraries": {"libblas": "c" * 64}}},
                "selection": [gate.IDENTITIES[name] for name in plan["selected"]],
                "gates": [{"identity": gate.IDENTITIES[name], "status": status,
                           "invocations": invocations,
                           "exit_status": (0 if status == "Pass" else 1) if invocations else None}
                          for name in plan["selected"]], "cases": []}
        for key in ("source", "fixtures", "binary"):
            evidence[key + "_after"] = copy.deepcopy(evidence[key])
        return evidence

    def add_records(self, plan, evidence):
        for index, case in enumerate(plan["cases"]):
            if case["gate"] not in plan["selected"]:
                continue
            body = {"case": case["id"], "identity": gate.IDENTITIES[case["gate"]],
                    **{key: evidence[key] for key in ("head", "source", "fixtures", "binary")},
                    "workers": case["workers"], "repeats": case["repeats"],
                    "checks": {check: True for check in case["checks"]},
                    "rejection": case["rejection"]}
            body["metadata"] = {
                "model": "SYNTHETIC " + case["id"], "seed": 1,
                "steps": (2 if case["id"].endswith("/physcal") else 20)
                         if case["gate"] in ("long20", "cg20") and "/prefix/" not in case["id"] else 1,
                "ranks": 1, "groups": 1, "threads": [1, 2, 4], "oracle_execution": "none",
                "reference": ({"label": "historical-fixture", "julia_version": "1.11-historical-synthetic",
                               "manifest_status": "Unavailable", "manifest_sha256": None}
                              if case["planned_kind"] == "independent_comparison"
                              else {"status": "NotApplicable"})}
            body["reference_workspace"] = plan["reference_workspace"]
            path = self.root / f"case-{index}.json"
            path.write_text(json.dumps(body))
            evidence["cases"].append({"id": case["id"],
                                      "artifact": {"path": path.name, "sha256": gate.sha(path)}})

    def test_explicit_empty_unknown_duplicate_selection_rejected(self):
        for selected in ([], ["unsupported"], ["prefix", "prefix"]):
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                gate.build_plan(selected, "1" * 40, self.source, self.fixtures)

    def test_exact_seven_identity_inventory_and_long20_counts(self):
        plan = self.plan(list(gate.IDENTITIES))
        self.assertEqual(len(plan["identities"]), 7)
        long = [case for case in plan["cases"] if case["gate"] == "long20"]
        self.assertEqual(len(long), 65)
        self.assertEqual(sum(case["planned_kind"] == "expected_rejection" for case in long), 3)
        self.assertEqual(sum(case["planned_kind"] == "invariance" for case in long), 57)
        self.assertEqual(sum(case["planned_kind"] == "independent_comparison" for case in long), 5)
        self.assertEqual(plan["settings"]["long_steps"], 20)
        self.assertEqual(plan["settings"]["long_window"], 20)
        self.assertEqual(len(plan["selection_commands"]), 7)
        self.assertTrue(all("--run-ignored" in command for command in plan["selection_commands"]))

    def test_all_selected_synthetic_records_have_separate_kinds(self):
        plan = self.plan(list(gate.IDENTITIES))
        evidence = self.evidence(plan)
        self.add_records(plan, evidence)
        result = gate.report(plan, evidence, self.root)
        self.assertEqual(result["status"], "Incomplete")
        self.assertEqual(result["outer_driver_invocations"], 7)
        self.assertEqual(result["reported_attested_cases"], 83)
        self.assertEqual(result["verified_independent_comparisons"], 0)
        self.assertTrue(all(row["verification"] == "NotVerified" for row in result["cases"]))
        self.assertEqual(result["reported_counts_by_kind"], {
            "independent_comparison": 5 + 5 + 4 + 1 + 4, "invariance": 57,
            "expected_rejection": 4, "activation": 3})

    def test_generic_gate_pass_is_incomplete_not_independent(self):
        plan = self.plan(["prefix"])
        result = gate.report(plan, self.evidence(plan), self.root)
        self.assertEqual(result["status"], "Incomplete")
        self.assertEqual(result["reported_counts_by_kind"]["independent_comparison"], 0)
        self.assertEqual(result["outer_driver_invocations"], 1)
        self.assertEqual(result["selected_test_identities"], 1)
        self.assertTrue(all(row["evidence_kind"] == "none" for row in result["cases"]))

    def test_unselected_cases_notrun_and_selected_explicit_skip(self):
        plan = self.plan()
        result = gate.report(plan, self.evidence(plan, "ExplicitSkip", 0), self.root)
        for row in result["cases"]:
            self.assertEqual(row["status"], "ExplicitSkip" if row["id"].startswith("transfer/") else "NotRun")
        self.assertEqual(result["outer_driver_invocations"], 0)
        self.assertEqual(result["status"], "Incomplete")

    def test_missing_gate_is_notrun_not_pass(self):
        plan = self.plan()
        evidence = self.evidence(plan)
        evidence["gates"] = []
        result = gate.report(plan, evidence, self.root)
        self.assertEqual(result["status"], "Incomplete")
        self.assertEqual(result["observed_gate_identities"], 0)
        self.assertTrue(all(row["status"] == "NotRun" for row in result["cases"]))

    def test_selected_failure_missing_unsupported_never_pass(self):
        for status in ("Failure", "MissingFixture", "Unsupported"):
            plan = self.plan()
            result = gate.report(plan, self.evidence(plan, status), self.root)
            self.assertEqual(result["status"], status)
            self.assertEqual(result["reported_attested_cases"], 0)
            self.assertEqual(result["outer_driver_invocations"], 1)

    def test_preflight_failure_counts_zero_executed_invocations(self):
        plan = self.plan()
        for status in ("Failure", "MissingFixture", "Unsupported"):
            result = gate.report(plan, self.evidence(plan, status, 0), self.root)
            self.assertEqual(result["status"], status)
            self.assertEqual(result["outer_driver_invocations"], 0)

    def test_after_binding_selection_and_terminal_controls(self):
        plan = self.plan()
        for key in ("source_after", "fixtures_after", "binary_after", "selection", "terminal"):
            evidence = self.evidence(plan)
            if key == "terminal":
                evidence["gates"][0]["exit_status"] = 1
            else:
                evidence[key] = [] if key == "selection" else {}
            with self.subTest(key=key), self.assertRaises(ValueError):
                gate.report(plan, evidence, self.root)

    def test_runtime_backend_linkage_only_and_wrong_thread_metadata_rejected(self):
        plan = self.plan()
        for field, value in (("profile", "release"), ("features", "mpi"), ("ranks", 2),
                             ("rust_version", ""), ("blas", {})):
            evidence = self.evidence(plan)
            evidence["runtime"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                gate.report(plan, evidence, self.root)
        for field, value in (("threads", 2), ("threads", True), ("version", ""),
                             ("observation", "linkage-only"), ("libraries", {})):
            evidence = self.evidence(plan)
            evidence["runtime"]["blas"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                gate.report(plan, evidence, self.root)

    def test_duplicate_wrong_or_unselected_test_and_count_rejected(self):
        plan = self.plan()
        for mutation in ("duplicate", "unselected", "zero", "two", "bool"):
            evidence = self.evidence(plan)
            if mutation == "duplicate":
                evidence["gates"] *= 2
            elif mutation == "unselected":
                evidence["gates"][0]["identity"] = gate.IDENTITIES["prefix"]
            else:
                evidence["gates"][0]["invocations"] = {"zero": 0, "two": 2, "bool": True}[mutation]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                gate.report(plan, evidence, self.root)

    def test_changed_source_head_fixture_and_binary_binding_rejected(self):
        plan = self.plan()
        for key in ("head", "source", "fixtures", "binary"):
            evidence = self.evidence(plan)
            if key == "head":
                evidence[key] = "2" * 40
            else:
                evidence[key] = {}
            with self.subTest(key=key), self.assertRaises(ValueError):
                gate.report(plan, evidence, self.root)

    def test_case_and_settings_tampered_plan_rejected(self):
        for key in ("cases", "settings", "identities"):
            plan = self.plan()
            plan[key] = [] if key == "cases" else {}
            with self.subTest(key=key), self.assertRaises(ValueError):
                gate.report(plan, self.evidence(plan), self.root)

    def test_changed_audited_test_source_requires_new_mapping(self):
        with self.assertRaises(ValueError):
            gate.build_plan(["transfer"], "1" * 40, {gate.TEST_SOURCE: "f" * 64}, self.fixtures)

    def test_workspace_version_none_empty_or_nonversion_rejected_not_historical(self):
        plan = self.plan(["prefix"])
        for version in (None, "", " ", 1131, True, "unknown", "1.13"):
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, "workspace Julia version"):
                gate.build_plan(plan["selected"], plan["head"], plan["source"], plan["fixtures"],
                                plan["input_closure"], version)

    def test_complete_synthetic_transfer_is_activation_not_independent(self):
        plan = self.plan()
        evidence = self.evidence(plan)
        self.add_records(plan, evidence)
        result = gate.report(plan, evidence, self.root)
        self.assertEqual(result["status"], "Incomplete")
        self.assertEqual(result["reported_attested_cases"], 3)
        self.assertEqual(result["reported_counts_by_kind"]["activation"], 3)
        self.assertEqual(result["verified_independent_comparisons"], 0)
        self.assertTrue(all(row["evidence_kind"] == "reported_assertion"
                            for row in result["cases"] if row["id"].startswith("transfer/")))

    def test_missing_case_incomplete_and_duplicate_unexpected_cases_rejected(self):
        plan = self.plan()
        evidence = self.evidence(plan)
        self.add_records(plan, evidence)
        removed = evidence["cases"].pop()
        self.assertEqual(gate.report(plan, evidence, self.root)["status"], "Incomplete")
        evidence["cases"].append(removed)
        evidence["cases"].append(removed)
        with self.assertRaises(ValueError):
            gate.report(plan, evidence, self.root)
        evidence["cases"][-1] = {**removed, "id": "unsupported/case"}
        with self.assertRaises(ValueError):
            gate.report(plan, evidence, self.root)

    def test_missing_changed_symlink_and_escape_artifacts_rejected(self):
        plan = self.plan()
        evidence = self.evidence(plan)
        self.add_records(plan, evidence)
        row = evidence["cases"][0]
        original = copy.deepcopy(row)
        for path in ("missing.json", "../escape.json", "/tmp/escape.json"):
            row["artifact"]["path"] = path
            with self.subTest(path=path), self.assertRaises(ValueError):
                gate.report(plan, evidence, self.root)
        row.update(original)
        file = self.root / row["artifact"]["path"]
        data = file.read_bytes()
        file.write_bytes(data + b" ")
        with self.assertRaises(ValueError):
            gate.report(plan, evidence, self.root)
        file.unlink()
        file.symlink_to(self.root / "case-79.json")
        with self.assertRaises(ValueError):
            gate.report(plan, evidence, self.root)

    def test_case_missing_check_worker_repeat_and_identity_rejected(self):
        plan = self.plan()
        for key in ("checks", "workers", "repeats", "identity", "source", "binary"):
            evidence = self.evidence(plan)
            self.add_records(plan, evidence)
            row = evidence["cases"][0]
            path = self.root / row["artifact"]["path"]
            body = json.loads(path.read_text())
            body[key] = None
            path.write_text(json.dumps(body))
            row["artifact"]["sha256"] = gate.sha(path)
            with self.subTest(key=key), self.assertRaises(ValueError):
                gate.report(plan, evidence, self.root)

    def test_independent_references_required_not_label_only(self):
        plan = self.plan(["prefix"])
        evidence = self.evidence(plan)
        self.add_records(plan, evidence)
        result = gate.report(plan, evidence, self.root)
        self.assertEqual(result["reported_counts_by_kind"]["independent_comparison"], 5)
        self.assertEqual(result["verified_independent_comparisons"], 0)
        plan["fixtures"] = {"unrelated/reference.txt": "a" * 64}
        evidence = self.evidence(plan)
        self.add_records(plan, evidence)
        with self.assertRaises(ValueError):
            gate.report(plan, evidence, self.root)

    def test_expected_rejection_not_success_or_numeric_comparison(self):
        plan = self.plan(["failure"])
        evidence = self.evidence(plan)
        self.add_records(plan, evidence)
        result = gate.report(plan, evidence, self.root)
        self.assertEqual(result["reported_counts_by_kind"]["expected_rejection"], 1)
        self.assertEqual(result["verified_independent_comparisons"], 0)
        row = evidence["cases"][0]
        path = self.root / row["artifact"]["path"]
        body = json.loads(path.read_text())
        body["rejection"] = [10, 0]
        path.write_text(json.dumps(body))
        row["artifact"]["sha256"] = gate.sha(path)
        with self.assertRaises(ValueError):
            gate.report(plan, evidence, self.root)

    def test_selected_missing_fixture_preflight_fails_before_plan(self):
        with patch.object(gate.subprocess, "check_output", return_value=""):
            with self.assertRaisesRegex(ValueError, "MissingFixture.*threaded_182_real_fsz"):
                gate.capture(self.root, ["real-fsz"])

    def test_any_file_under_root_cannot_replace_required_fixture_inventory(self):
        for selected in (["prefix"], ["physcal"], ["cg20"], ["real-fsz"]):
            incomplete = copy.deepcopy(self.fixtures)
            required = gate.required_fixture_files(selected)
            missing = next(path for path in required if not path.endswith("Manifest-v1.13.toml"))
            del incomplete[missing]
            # All other files/root markers remain; one required file still matters.
            with self.subTest(selected=selected), self.assertRaisesRegex(ValueError, "MissingFixture"):
                gate.build_plan(selected, "1" * 40, self.source, incomplete)

    def test_forged_true_checks_and_verified_claims_never_promote_execution(self):
        plan = self.plan(["prefix"])
        evidence = self.evidence(plan)
        self.add_records(plan, evidence)
        evidence["claimed_verified"] = True
        evidence["approved_emitter"] = "FORGED"
        evidence["run_uuid"] = "FORGED"
        for row in evidence["cases"]:
            path = self.root / row["artifact"]["path"]
            body = json.loads(path.read_text())
            body["claimed_verified"] = True
            path.write_text(json.dumps(body))
            row["artifact"]["sha256"] = gate.sha(path)
        result = gate.report(plan, evidence, self.root)
        self.assertEqual(result["status"], "Incomplete")
        self.assertEqual(result["execution_verification"], "NotVerified")
        self.assertIsNone(result["approved_emitter"])
        self.assertEqual(result["verified_independent_comparisons"], 0)
        self.assertTrue(all(row["status"] == "AttestedOnly" for row in result["cases"]
                            if row["id"].startswith("prefix/")))

    def test_case_metadata_and_reference_manifest_controls(self):
        plan = self.plan()
        for mutation in ("missing", "seed", "steps", "model", "ranks", "groups", "threads", "oracle", "manifest"):
            evidence = self.evidence(plan)
            self.add_records(plan, evidence)
            row = evidence["cases"][0]
            path = self.root / row["artifact"]["path"]
            body = json.loads(path.read_text())
            metadata = body["metadata"]
            if mutation == "missing":
                del body["metadata"]
            elif mutation in ("seed", "steps", "model"):
                metadata[mutation] = None
            elif mutation in ("ranks", "groups"):
                metadata[mutation] = 2
            elif mutation == "threads":
                metadata[mutation] = [1]
            elif mutation == "oracle":
                metadata["oracle_execution"] = "Julia"
            else:
                metadata["reference"] = {"label": "historical-fixture", "manifest_sha256": "f" * 64}
            path.write_text(json.dumps(body))
            row["artifact"]["sha256"] = gate.sha(path)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                gate.report(plan, evidence, self.root)

    def test_historical_111_current_manifest_substitution_rejected(self):
        plan = self.plan(["prefix"])
        evidence = self.evidence(plan)
        self.add_records(plan, evidence)
        row = evidence["cases"][0]
        path = self.root / row["artifact"]["path"]
        body = json.loads(path.read_text())
        body["metadata"]["reference"] = {
            "label": "historical-fixture", "julia_version": "1.11.0",
            "manifest_status": "Available", "manifest_role": "historical-generation",
            "manifest_sha256": plan["reference_workspace"]["manifest_sha256"]}
        path.write_text(json.dumps(body))
        row["artifact"]["sha256"] = gate.sha(path)
        with self.assertRaisesRegex(ValueError, "different historical Julia"):
            gate.report(plan, evidence, self.root)

    def test_unknown_historical_manifest_and_version_remain_missing_evidence(self):
        plan = self.plan(["prefix"])
        evidence = self.evidence(plan)
        self.add_records(plan, evidence)
        for row in evidence["cases"]:
            path = self.root / row["artifact"]["path"]
            body = json.loads(path.read_text())
            body["metadata"]["reference"] = {"label": "historical-fixture", "julia_version": None,
                                             "manifest_status": "MissingEvidence", "manifest_sha256": None}
            path.write_text(json.dumps(body))
            row["artifact"]["sha256"] = gate.sha(path)
        result = gate.report(plan, evidence, self.root)
        self.assertEqual(result["status"], "Incomplete")
        self.assertEqual(result["verified_independent_comparisons"], 0)
        for row in result["cases"]:
            if row["id"].startswith("prefix/"):
                self.assertEqual(row["reported_metadata"]["reference"]["manifest_status"], "MissingEvidence")
                self.assertIsNone(row["reported_metadata"]["reference"]["manifest_sha256"])
                self.assertEqual(row["settings_verification"], "NotVerified")

    def test_namelist_closure_all_definitions_and_implicit_initial_bound(self):
        input_dir = self.root / "inputs"
        input_dir.mkdir()
        for name, text in (("namelist.def", "ModPara modpara.def\nInGutzwiller overlay.def\n"),
                           ("modpara.def", "settings"), ("overlay.def", "overlay"),
                           ("initial.def", "implicit")):
            (input_dir / name).write_text(text)
        fixtures = {"inputs/" + path.name: gate.sha(path) for path in input_dir.iterdir()}
        closure = gate.definition_closure(self.root, "inputs/namelist.def", fixtures)
        self.assertEqual(closure, fixtures)
        for name in ("inputs/modpara.def", "inputs/overlay.def", "inputs/initial.def"):
            incomplete = {key: value for key, value in fixtures.items() if key != name}
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "MissingFixture"):
                gate.definition_closure(self.root, "inputs/namelist.def", incomplete)
        (input_dir / "overlay.def").write_text("changed")
        with self.assertRaisesRegex(ValueError, "MissingFixture"):
            gate.definition_closure(self.root, "inputs/namelist.def", fixtures)

    def test_escaping_namelist_definition_rejected(self):
        path = self.root / "namelist.def"
        for text in ("ModPara ../outside.def\n", "ModPara /tmp/outside.def\n"):
            path.write_text(text)
            with self.subTest(text=text), self.assertRaises(ValueError):
                gate.definition_closure(self.root, "namelist.def", {"namelist.def": gate.sha(path)})

    def test_oracle_input_sha_must_cover_loaded_definition_closure(self):
        inputs = self.root / "inputs"
        inputs.mkdir()
        (inputs / "namelist.def").write_text("ModPara modpara.def\n")
        (inputs / "modpara.def").write_text("settings")
        (inputs / "initial.def").write_text("implicit overlay")
        closure = {"inputs/" + path.name: gate.sha(path) for path in inputs.iterdir()}
        hash_manifest = self.root / "inputs.sha256"
        text = "".join(f"{digest} {Path(name).name}\n" for name, digest in closure.items())
        hash_manifest.write_text(text)
        fixtures = {**closure, "inputs.sha256": gate.sha(hash_manifest)}
        gate.bind_declared_input_hashes(self.root, "inputs.sha256", "inputs/namelist.def", fixtures, closure)
        for content in ("\n".join(text.splitlines()[:-1]) + "\n", text + text.splitlines()[0] + "\n",
                        text.replace(closure["inputs/modpara.def"], "f" * 64)):
            hash_manifest.write_text(content)
            fixtures["inputs.sha256"] = gate.sha(hash_manifest)
            with self.subTest(content=content), self.assertRaises(ValueError):
                gate.bind_declared_input_hashes(self.root, "inputs.sha256", "inputs/namelist.def", fixtures, closure)


if __name__ == "__main__":
    unittest.main()

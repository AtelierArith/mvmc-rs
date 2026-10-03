"""Synthetic actual-format package controls; NOT executed parity records."""
import json
import os
from pathlib import Path
import tempfile
import subprocess
import signal
import unittest
from unittest.mock import patch

from thread_envelope import BASE, GATES, case_ids, digest, metadata, validate
from run_thread_envelope import run
import thread_envelope
import run_thread_envelope
import exact_record_schema


class Controls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace = self.root / "work"
        self.package = self.root / "package"
        self.workspace.mkdir()
        self.package.mkdir()
        (self.package / "records").mkdir()
        self.uuid = "00000000-0000-4000-8000-000000000001"
        self.identity, self.job, _, self.pairs = GATES["failure"]
        self.case = "failure/hubbard_chain_real/qp32/store0/cg0"
        for name in ("emitter.rs", "fixture.dat", "binary"):
            (self.workspace / name).write_bytes(b"synthetic:" + name.encode())
        self.reviewed = {
            "schema": 1, "head": BASE, "run_uuid": self.uuid, "gate": "failure",
            "suite": "mvmc-core::threaded_issue182", "settings": {"threshold": 32, "workers": [1, 2, 4]},
            "source": {"emitter.rs": digest((self.workspace / "emitter.rs").read_bytes())},
            "fixtures": {"fixture.dat": digest((self.workspace / "fixture.dat").read_bytes())},
            "binary": {"path": str(self.workspace / "binary"), "sha256": digest((self.workspace / "binary").read_bytes())},
            "reference_metadata": {"role": "historical", "manifest": None},
            "backend": {"role": "synthetic-only"}, "toolchain": {"role": "synthetic-only"},
            "case_map": [{"id": self.case, "gate": "failure", "boundary": "runtime-outcome", "required_record_labels": ["case-outcome"],
                          "required_discrete_values": {"case-outcome": "verified-non-spd"}}],
            "timeout_seconds": 10, "launch_environment": {"OPENBLAS_NUM_THREADS": "1"},
        }
        self.reviewed["emitter_sources"] = dict(self.reviewed["source"])
        self.layout_settings = dict(npara=2, nelec=1, nqp=1, samples=2,
            onebody=0, twobody=0, store=0, cg=0, input_seed=1, launch_seed=1,
            steps=20, window=20, groups=1, ranks=1, all_complex=True, physical=False)
        self.reviewed["numeric_schema"] = {"schema": 1, "cases": [{
            "id": self.case, "settings": self.layout_settings.copy(),
            "source": dict(self.reviewed["source"]), "inputs": dict(self.reviewed["fixtures"]),
            "records": [{"label": "numeric", "kind": "pf", "dimension": 2, "zero_reason": None}],
        }]}
        self.reviewed["layout_bindings"] = {self.case: "synthetic-exact"}
        self.reviewed["adapter_sha256"] = digest(Path(thread_envelope.__file__).read_bytes())
        self.reviewed["schema_adapter_sha256"] = digest(Path(exact_record_schema.__file__).read_bytes())
        self.reviewed["wrapper_sha256"] = digest(Path(run_thread_envelope.__file__).read_bytes())
        self.reviewed["binary"]["source_manifest_sha256"] = digest(json.dumps(self.reviewed["source"], sort_keys=True).encode())
        selection = {"rust-suites": {self.reviewed["suite"]: {
            "binary-path": self.reviewed["binary"]["path"], "testcases": {
                self.identity: {"ignored": True, "filter-match": {"status": "matches"}}}}}}
        self.put("selection.json", json.dumps(selection).encode())
        self.reviewed["selection_artifact"] = digest((self.package / "selection.json").read_bytes())
        argv = [self.reviewed["binary"]["path"], "--ignored", "--exact", self.identity, "--nocapture", "--test-threads=1"]
        self.put("parent.json", json.dumps({"argv": argv, "run_uuid": self.uuid, "head": BASE,
                  "binary_sha256": self.reviewed["binary"]["sha256"], "exit_code": 0,
                  "cleanup": {"timed_out": False, "status": "NotRequired", "errors": [], "races": []},
                  "post_inputs": True}).encode())
        self.put("parent.stdout", (f"running 1 test\ntest {self.identity} ... ok\n"
                  "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 36 filtered out; finished in 0.01s\n").encode())
        self.put("parent.stderr", b"synthetic captured parent stderr\n")
        for index in range(6):
            worker, repeat = [1, 2, 4][index // 2], index % 2 + 1
            prefix = f"records/child-{index}"
            request = metadata(self.uuid, index, self.job, worker, 32, repeat)
            for suffix in ("requested", "actual-settings"):
                self.put(prefix + "." + suffix, request)
            self.put(prefix + ".terminal", request + b"status=Some(0)\n")
            self.put(prefix + ".events", (f"0|{self.uuid}|START|{self.case}|runtime-outcome|{worker}|{repeat}\n"
                      f"1|{self.uuid}|COMPLETE|{self.case}|runtime-outcome|{worker}|{repeat}\n").encode())
            self.put(prefix + ".stdout", b"running 1 test\nD|case-outcome|verified-non-spd\nN|numeric|1.0 2.0\n")
            self.put(prefix + ".stderr", b"synthetic observation stderr\n")
            self.put(prefix + ".layout-synthetic-exact.json", json.dumps({
                "run_uuid": self.uuid, "invocation": index, "label": "synthetic-exact",
                "settings": self.layout_settings.copy(),
            }).encode())
        operand = b'{"case-outcome": (\'D\', "verified-non-spd"), "numeric": (\'N\', "1.0 2.0")}\n'
        for index in range(len(self.pairs)):
            prefix = f"records/comparison-{index}"
            self.put(prefix + ".expected", operand)
            self.put(prefix + ".actual", operand)
            self.put(prefix + ".terminal", f"run={self.uuid}\ncomparison={index}\nboundary=original-worker-compare-returned\n".encode())
        self.refresh()

    def put(self, name, data):
        (self.package / name).write_bytes(data)

    def refresh(self):
        self.envelope = {"reviewed_sha256": digest(json.dumps(self.reviewed, sort_keys=True).encode()),
                         "run_uuid": self.uuid, "head": BASE, "parent_exit_code": 0,
                         "artifacts": {str(p.relative_to(self.package)): digest(p.read_bytes())
                            for p in self.package.rglob("*") if p.is_file() and p.name != "envelope.json"}}
        self.put("envelope.json", json.dumps(self.envelope).encode())

    def reject(self, mutation, rehash=True):
        mutation()
        if rehash:
            self.refresh()
        else:
            self.put("envelope.json", json.dumps(self.envelope).encode())
        with self.assertRaises((ValueError, KeyError, SyntaxError)):
            validate(self.package, self.workspace, self.envelope, self.reviewed)

    def change(self, name, before, after):
        path = self.package / name
        path.write_bytes(path.read_bytes().replace(before, after))

    def test_valid_execution_bound_not_numeric(self):
        result = validate(self.package, self.workspace, self.envelope, self.reviewed)
        self.assertEqual(result["numerical_verification"], "NotVerified")
        self.assertEqual(result["verified_independent_comparisons"], 0)

    def test_full_83_inventory_not_execution(self):
        self.assertEqual(sum(len(case_ids(g)) for g in GATES), 83)

    def test_missing_child(self):
        self.reject(lambda: (self.package / "records/child-5.events").unlink())

    def test_duplicate_child(self):
        self.reject(lambda: self.put("records/child-6.requested", b"extra\n"))

    def test_tamper_without_rehash(self):
        self.reject(lambda: self.put("records/child-0.stderr", b"tampered"), False)

    def test_wrong_actual_repeat(self):
        self.reject(lambda: self.change("records/child-0.actual-settings", b"repeat=1", b"repeat=2"))

    def test_wrong_threshold(self):
        self.reject(lambda: self.change("records/child-0.actual-settings", b"threshold=32", b"threshold=99"))

    def test_wrong_run(self):
        self.reject(lambda: self.change("records/child-0.events", self.uuid.encode(), b"00000000-0000-4000-8000-000000000002"))

    def test_wrong_child_outcome(self):
        self.reject(lambda: self.change("records/child-0.terminal", b"Some(0)", b"Some(1)"))

    def test_wrong_case_outcome_even_matching_parent_operands(self):
        def mutation():
            for path in self.package.rglob("*"):
                if path.is_file() and path.name != "envelope.json":
                    path.write_bytes(path.read_bytes().replace(b"verified-non-spd", b"unexpected-success"))
        self.reject(mutation)

    def test_wrong_parent_exit(self):
        self.envelope["parent_exit_code"] = 1
        self.reject(lambda: None, False)

    def test_missing_parent_normal_summary(self):
        self.reject(lambda: self.put("parent.stdout", b"generic PASS\n"))

    def test_wrong_exact_identity(self):
        self.reject(lambda: self.change("parent.stdout", self.identity.encode(), b"other_identity"))

    def test_duplicate_event(self):
        data = (self.package / "records/child-0.events").read_bytes()
        self.reject(lambda: self.put("records/child-0.events", data + data))

    def test_truncated_event(self):
        data = (self.package / "records/child-0.events").read_bytes()
        self.reject(lambda: self.put("records/child-0.events", data[:-1]))

    def test_wrong_comparison_operands(self):
        self.reject(lambda: self.change("records/comparison-0.actual", b"1.0 2.0", b"1.0 3.0"))

    def test_duplicate_comparison_key(self):
        self.reject(lambda: self.put("records/comparison-0.actual", b'{"numeric": (\'N\', "1"), "numeric": (\'N\', "2")}\n'))

    def test_missing_fixture(self):
        self.reject(lambda: (self.workspace / "fixture.dat").unlink())

    def test_missing_emitter_binding(self):
        self.reject(lambda: self.reviewed["source"].clear())

    def test_unexpected_source_file(self):
        self.reject(lambda: (self.workspace / "unreviewed.rs").write_bytes(b"extra source"))

    def test_wrong_binary_source_association(self):
        self.reject(lambda: self.reviewed["binary"].update(source_manifest_sha256="f" * 64))

    def test_wrong_adapter_source(self):
        self.reject(lambda: self.reviewed.update(adapter_sha256="f" * 64))

    def test_wrong_binary(self):
        self.reject(lambda: (self.workspace / "binary").write_bytes(b"different"))

    def test_forged_self_checks(self):
        self.envelope["checks"] = {"all_passed": True}
        self.reject(lambda: None, False)

    def test_wrapper_no_authorization_never_launches(self):
        with patch("run_thread_envelope.subprocess.Popen") as launch:
            with self.assertRaises(ValueError):
                run(self.workspace, self.root / "not-created", self.reviewed,
                    self.package / "selection.json", execute=False)
            launch.assert_not_called()
        self.assertFalse((self.root / "not-created").exists())

    def test_wrapper_wrong_token_never_launches(self):
        with patch.dict(os.environ, {"ISSUE182_APPROVED_EXECUTION_TOKEN": "wrong"}), patch("run_thread_envelope.subprocess.Popen") as launch:
            with self.assertRaises(ValueError):
                run(self.workspace, self.root / "not-created", self.reviewed,
                    self.package / "selection.json", execute=True)
            launch.assert_not_called()

    def test_wrapper_parent_failure_retains_artifacts(self):
        output = self.root / "failed-package"
        with patch.dict(os.environ, {"ISSUE182_APPROVED_EXECUTION_TOKEN": self.uuid}), patch("run_thread_envelope.subprocess.Popen") as launch:
            launch.return_value.wait.return_value = 2
            with self.assertRaises(ValueError):
                run(self.workspace, output, self.reviewed, self.package / "selection.json", execute=True)
            launch.assert_called_once()
        self.assertEqual(json.loads((output / "parent.json").read_text())["exit_code"], 2)
        self.assertTrue((output / "envelope.json").is_file())

    def test_wrapper_wrong_selection_never_launches(self):
        data = (self.package / "selection.json").read_bytes().replace(self.identity.encode(), b"wrong_identity")
        self.put("selection.json", data)
        self.reviewed["selection_artifact"] = digest(data)
        with patch.dict(os.environ, {"ISSUE182_APPROVED_EXECUTION_TOKEN": self.uuid}), patch("run_thread_envelope.subprocess.Popen") as launch:
            with self.assertRaises(ValueError):
                run(self.workspace, self.root / "not-created", self.reviewed, self.package / "selection.json", execute=True)
            launch.assert_not_called()

    def test_wrapper_timeout_is_not_normal_parent_return(self):
        output = self.root / "timeout-package"
        with patch.dict(os.environ, {"ISSUE182_APPROVED_EXECUTION_TOKEN": self.uuid}), patch("run_thread_envelope.subprocess.Popen") as launch, patch("run_thread_envelope.os.killpg") as terminate, patch("run_thread_envelope.group_members", return_value=[]):
            launch.return_value.pid = 123456  # synthetic PID; killpg is mocked
            launch.return_value.wait.side_effect = [subprocess.TimeoutExpired("synthetic", 10), 0]
            with self.assertRaises(ValueError):
                run(self.workspace, output, self.reviewed, self.package / "selection.json", execute=True)
            self.assertEqual([call.args for call in terminate.call_args_list],
                             [(123456, signal.SIGTERM), (123456, signal.SIGKILL)])
        self.assertIsNone(json.loads((output / "parent.json").read_text())["exit_code"])

    def test_package_symlink_rejected(self):
        path = self.package / "parent.stderr"
        path.unlink()
        path.symlink_to(self.package / "parent.stdout")
        self.reject(lambda: None)

    def test_reviewed_scope_cannot_drop_group(self):
        self.reject(lambda: self.reviewed.update(case_map=[]))

    def test_wrong_selection_status_even_rehashed(self):
        data = (self.package / "selection.json").read_bytes().replace(b'"ignored": true', b'"ignored": false')
        self.put("selection.json", data)
        self.reviewed["selection_artifact"] = digest(data)
        self.reject(lambda: None)

    def test_strict_reviewed_integer_fields(self):
        for target, field, value in [
            (self.reviewed, "schema", True), (self.reviewed, "schema", 1.0),
            (self.reviewed, "timeout_seconds", True), (self.reviewed, "timeout_seconds", 10.0),
            (self.reviewed["settings"], "threshold", 32.0),
            (self.reviewed["settings"], "threshold", True),
            (self.reviewed["settings"], "workers", [1.0, 2, 4]),
            (self.reviewed["settings"], "workers", [True, 2, 4]),
        ]:
            original = target[field]
            with self.subTest(field=field, value=value):
                self.reject(lambda: target.update({field: value}))
            target[field] = original
        for field in ("seed", "steps", "window", "ranks", "groups", "store", "cg", "samples", "warmup"):
            for value in (True, 1.0):
                with self.subTest(field=field, value=value):
                    self.reject(lambda: self.reviewed["settings"].update({field: value}))
                del self.reviewed["settings"][field]

    def test_strict_parent_status_and_flags(self):
        original = json.loads((self.package / "parent.json").read_text())
        for field, value in (("exit_code", False), ("exit_code", 0.0), ("post_inputs", 1)):
            parent = dict(original, **{field: value})
            with self.subTest(field=field, value=value):
                self.reject(lambda: self.put("parent.json", json.dumps(parent).encode()))
        parent = dict(original, cleanup={"timed_out": 0, "status": "NotRequired", "errors": [], "races": []})
        self.reject(lambda: self.put("parent.json", json.dumps(parent).encode()))

    def test_strict_envelope_exit_status(self):
        for value in (False, 0.0):
            with self.subTest(value=value):
                self.reject(lambda: self.envelope.update(parent_exit_code=value), rehash=False)

    def test_empty_numeric_operands_need_explicit_schema(self):
        self.reject(lambda: self.change("records/child-0.stdout", b"N|numeric|1.0 2.0", b"N|numeric|"))

    def test_early_parent_exit_term_ignoring_child_still_killed(self):
        # Mock parent already returns zero; membership shows a surviving child.
        # wait must occur AFTER final KILL, not suppress it on early parent exit.
        from unittest.mock import Mock
        process = Mock(pid=123456)
        order = []
        process.wait.side_effect = lambda **kwargs: order.append("wait") or 0
        with patch("run_thread_envelope.os.killpg", side_effect=lambda pid, sig: order.append(sig)), \
             patch("run_thread_envelope.group_members", side_effect=[[777], []]), \
             patch("run_thread_envelope.time.sleep"):
            result = run_thread_envelope.terminate_owned_group(process, grace=0, deadline=1)
        self.assertEqual(order, [signal.SIGTERM, signal.SIGKILL, "wait"])
        self.assertEqual(result["status"], "Terminated")

    def test_process_lookup_race_requires_empty_membership(self):
        from unittest.mock import Mock
        process = Mock(pid=123456)
        with patch("run_thread_envelope.os.killpg", side_effect=[ProcessLookupError(), None]) as kill, \
             patch("run_thread_envelope.group_members", return_value=[]):
            result = run_thread_envelope.terminate_owned_group(process, grace=0, deadline=0)
        self.assertEqual(kill.call_count, 2)
        self.assertEqual(result["races"], ["SIGTERM"])
        self.assertEqual(result["status"], "Terminated")

    def test_second_wait_timeout_retains_partial_evidence(self):
        output = self.root / "second-timeout-package"
        with patch.dict(os.environ, {"ISSUE182_APPROVED_EXECUTION_TOKEN": self.uuid}), \
             patch("run_thread_envelope.subprocess.Popen") as launch, \
             patch("run_thread_envelope.os.killpg") as kill, \
             patch("run_thread_envelope.group_members", return_value=[]):
            launch.return_value.pid = 123456
            launch.return_value.wait.side_effect = subprocess.TimeoutExpired("synthetic", 10)
            with self.assertRaises(ValueError):
                run(self.workspace, output, self.reviewed, self.package / "selection.json", execute=True)
        parent = json.loads((output / "parent.json").read_text())
        self.assertEqual(kill.call_count, 2)
        self.assertIsNone(parent["exit_code"])
        self.assertEqual(parent["cleanup"]["status"], "CleanupIncomplete")
        self.assertIn("reap:TimeoutExpired", parent["cleanup"]["errors"])
        self.assertTrue((output / "envelope.json").is_file())

    def test_unknown_membership_and_signal_error_fail_closed(self):
        from unittest.mock import Mock
        with patch("run_thread_envelope.os.killpg", side_effect=[PermissionError(), None]) as kill, \
             patch("run_thread_envelope.group_members", side_effect=PermissionError()):
            result = run_thread_envelope.terminate_owned_group(Mock(pid=123456), grace=0, deadline=0)
        self.assertEqual(kill.call_count, 2)
        self.assertEqual(result["status"], "CleanupIncomplete")
        self.assertTrue(result["errors"])

    def test_post_input_failure_still_preserves_package(self):
        output = self.root / "post-input-failure"
        with patch.dict(os.environ, {"ISSUE182_APPROVED_EXECUTION_TOKEN": self.uuid}), \
             patch("run_thread_envelope.subprocess.Popen") as launch, \
             patch("run_thread_envelope.verify_inputs", side_effect=[None, ValueError("changed")]):
            launch.return_value.wait.return_value = 0
            with self.assertRaises(ValueError):
                run(self.workspace, output, self.reviewed, self.package / "selection.json", execute=True)
        parent = json.loads((output / "parent.json").read_text())
        self.assertIs(parent["post_inputs"], False)
        envelope = json.loads((output / "envelope.json").read_text())
        self.assertEqual(envelope["preservation_failures"], ["post_inputs:ValueError"])
        self.assertTrue((output / "parent.stdout").is_file())

    def test_missing_actual_layout_artifact(self):
        self.reject(lambda: (self.package / "records/child-0.layout-synthetic-exact.json").unlink())

    def test_layout_integer_invocation_bool_rejected(self):
        name = "records/child-0.layout-synthetic-exact.json"
        value = json.loads((self.package / name).read_text())
        value["invocation"] = False
        self.reject(lambda: self.put(name, json.dumps(value).encode()))

    def test_actual_launch_seed_changes_fail_closed(self):
        name = "records/child-0.layout-synthetic-exact.json"
        value = json.loads((self.package / name).read_text())
        value["settings"]["launch_seed"] = 2
        self.reject(lambda: self.put(name, json.dumps(value).encode()))

    def test_unreviewed_numeric_schema_missing(self):
        self.reject(lambda: self.reviewed.pop("numeric_schema"))


if __name__ == "__main__":
    unittest.main()

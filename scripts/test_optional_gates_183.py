"""Infrastructure tests, NOT executed numerical gate coverage."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch, Mock

import run_optional_gates_183 as gate


def listing(names=gate.GENERAL):
    return {"rust-suites": {"one": {"binary-path": "/tmp/not-executed",
            "testcases": {name: {"ignored": True, "filter-match": {"status": "matches"}}
                          for name in names}}}}


class OptionalGateContract(unittest.TestCase):
    def test_mpi_provider_capture_precedes_cargo_and_failure_is_not_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "fixed-source"
            source.write_bytes(b"synthetic nonexecutable input")
            with patch.object(gate.platform, "system", return_value="Linux"), \
                    patch.dict(gate.os.environ, {"CARGO_TARGET_DIR": str(root / "target")}), \
                    patch.object(gate.subprocess, "check_output", return_value="synthetic"), \
                    patch.object(gate.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run, \
                    patch.object(gate, "source_files", return_value=[source]), \
                    patch.object(gate, "fixture_files", return_value=[source]), \
                    patch.object(gate.mpi_provider, "capture", side_effect=ValueError("startup failure")) as capture:
                output = root / "evidence"
                with self.assertRaisesRegex(ValueError, "startup failure"):
                    gate.run("mpi", output)
                capture.assert_called_once()
                cargo_commands = [call.args[0] for call in run.call_args_list
                                  if call.args[0][0] == "cargo"]
                self.assertEqual(cargo_commands, [["cargo", "nextest", "--version"]])
                terminal = json.loads((output / "terminal.json").read_text())
                self.assertEqual((terminal["status"], terminal["exit_status"]), ("Failure", 1))
                self.assertEqual(terminal["completed"], [])

    def test_family_ledger_unselected_explicit_skip_and_no_fake_counts(self):
        ledger = gate.family_ledger("lanczos", ["mpi"])
        self.assertEqual(set(ledger), set(gate.FAMILIES))
        self.assertEqual(ledger["mpi"]["status"], "ExplicitSkip")
        self.assertTrue(ledger["lanczos"]["selected"])
        self.assertEqual(ledger["general"]["status"], "NotRun")
        for row in ledger.values():
            self.assertEqual(row["started_driver_invocations"], 0)
            self.assertEqual(row["completed_selection_identities"], [])
            self.assertEqual(row["selected_test_identities"], [])
            self.assertEqual(row["helper_tests"], 0)
            self.assertEqual(row["empty_contracts"], 0)
            self.assertIsNone(row["numeric_reference_comparisons"])
        for selected, excluded in (("unknown", []), ("mpi", ["mpi"]),
                                   ("mpi", ["thread", "thread"]), ("mpi", ["unknown"])):
            with self.assertRaises(gate.UnsupportedError):
                gate.family_ledger(selected, excluded)

    def test_failure_classification_and_actual_nonrunning_terminal(self):
        cases = [(gate.MissingFixtureError("fixture"), "MissingFixture"),
                 (gate.UnsupportedError("selection"), "Unsupported"),
                 (RuntimeError("execution"), "Failure"), (ValueError("hash"), "Failure")]
        for error, status in cases:
            self.assertEqual(gate.failure_status(error), status)
            with tempfile.TemporaryDirectory() as tmp, \
                    patch.object(gate.platform, "system", return_value="Linux"), \
                    patch.dict(gate.os.environ, {"CARGO_TARGET_DIR": str(Path(tmp) / "target")}), \
                    patch.object(subprocess, "check_output", side_effect=error), \
                    patch.object(subprocess, "run") as run:
                root = Path(tmp) / "evidence"
                with self.assertRaises(type(error)):
                    gate.run("general", root, ["mpi"])
                run.assert_not_called()
                terminal = json.loads((root / "terminal.json").read_text())
                ledger = json.loads((root / "family-ledger.json").read_text())["families"]
                self.assertEqual(terminal["exit_status"], 1)
                self.assertEqual(terminal["status"], status)
                self.assertEqual(ledger["general"]["status"], status)
                self.assertEqual(ledger["general"]["started_driver_invocations"], 0)
                self.assertEqual(ledger["thread"]["status"], "NotRun")
                self.assertEqual(ledger["mpi"]["status"], "ExplicitSkip")

    def test_unfinished_identity_and_hash_closure_cannot_pass(self):
        before = {"source": "pinned"}
        gate.validate_completion("general", list(gate.GENERAL), before, before)
        for completed, after in (([], before), (list(gate.GENERAL[:1]), before),
                                 ([gate.GENERAL[0]] * 2, before),
                                 (list(gate.GENERAL), {"source": "changed"}),
                                 (list(gate.GENERAL), {})):
            with self.assertRaises(ValueError):
                gate.validate_completion("general", completed, before, after)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "evidence.txt"
            path.write_text("actual")
            hashes = gate.digest_files([path])
            gate.verify_artifact_hashes(root, hashes)
            path.write_text("tampered")
            with self.assertRaises(ValueError):
                gate.verify_artifact_hashes(root, hashes)
            path.unlink()
            for manifest in (hashes, {}, {"/outside/evidence": "bad"}):
                with self.assertRaises(ValueError):
                    gate.verify_artifact_hashes(root, manifest)

    def test_mocked_completion_pass_is_downgraded_on_artifact_failure(self):
        # Simulated subprocesses only: this does not execute numerical gates.
        for artifact_failure in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                binary = root / "mock-binary"
                binary.write_bytes(b"not executable; simulated infrastructure")
                data = listing()
                data["rust-suites"]["one"]["binary-path"] = str(binary)
                def fake_run(args, **kwargs):
                    if args[:3] == ["cargo", "nextest", "list"]:
                        kwargs["stdout"].write(json.dumps(data))
                    return SimpleNamespace(returncode=0)
                def fake_backend(_, linkage):
                    linkage.write_text("mock linkage, not native backend evidence\n")
                    return {"mock": True}
                with patch.object(gate.platform, "system", return_value="Linux"), \
                        patch.dict(gate.os.environ, {"CARGO_TARGET_DIR": str(root / "target")}), \
                        patch.object(subprocess, "check_output", return_value="mock revision"), \
                        patch.object(subprocess, "run", side_effect=fake_run), \
                        patch.object(gate, "source_files", return_value=[binary]), \
                        patch.object(gate, "fixture_files", return_value=[binary]), \
                        patch.object(gate, "backend", side_effect=fake_backend):
                    output = root / "evidence"
                    if artifact_failure:
                        with patch.object(gate, "validate_artifacts", side_effect=ValueError("missing artifact")):
                            with self.assertRaises(ValueError):
                                gate.run("general", output)
                    else:
                        gate.run("general", output)
                    terminal = json.loads((output / "terminal.json").read_text())
                    ledger = json.loads((output / "family-ledger.json").read_text())["families"]
                    self.assertEqual(terminal["exit_status"], int(artifact_failure))
                    self.assertEqual(ledger["general"]["status"], "Failure" if artifact_failure else "Pass")
                    self.assertEqual(ledger["general"]["started_driver_invocations"], 1)
                    self.assertEqual(ledger["general"]["completed_selection_identities"], list(gate.GENERAL))
                    self.assertEqual(ledger["general"]["selected_test_identities"], list(gate.GENERAL))
                    self.assertIsNone(ledger["general"]["numeric_reference_comparisons"])
                    self.assertEqual(ledger["general"]["helper_tests"], 0)

    def test_backend_runtime_api_library_config_and_threads_fail_closed(self):
        def library(config=b"OpenBLAS 0.3.26 test", core=b"Haswell", threads=1):
            result = Mock(spec=[])
            result.openblas_get_config = Mock(return_value=config)
            result.openblas_get_corename = Mock(return_value=core)
            result.openblas_get_num_threads = Mock(return_value=threads)
            return result
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            libpath = root / "libopenblas.so"
            libpath.write_bytes(b"mock library; no native code executed")
            linkage = f"libopenblas.so => {libpath}\n"
            with patch.object(subprocess, "check_output", return_value=linkage), \
                    patch.object(gate.ctypes, "CDLL", return_value=library()):
                self.assertEqual(gate.backend(root / "binary", root / "ldd")['actual_threads'], 1)
            for output in ("", "libopenblas.so => not found\n",
                           linkage + f"libopenblas_other.so => {root / 'other.so'}\n"):
                with patch.object(subprocess, "check_output", return_value=output), \
                        patch.object(gate.ctypes, "CDLL") as load:
                    with self.assertRaises(ValueError):
                        gate.backend(root / "binary", root / "ldd")
                    load.assert_not_called()
            missing_apis = []
            for api in ("openblas_get_config", "openblas_get_corename", "openblas_get_num_threads"):
                missing = library()
                delattr(missing, api)
                missing_apis.append(missing)
            for bad in (*missing_apis, object(), library(config=None), library(core=None),
                        library(config=b"OpenBLAS unknown"), library(config=b""),
                        library(core=b""), library(threads=0), library(threads=2)):
                with patch.object(subprocess, "check_output", return_value=linkage), \
                        patch.object(gate.ctypes, "CDLL", return_value=bad):
                    with self.assertRaises(ValueError):
                        gate.backend(root / "binary", root / "ldd")
            with patch.object(subprocess, "check_output", return_value=linkage), \
                    patch.object(gate.ctypes, "CDLL", side_effect=OSError("load failed")):
                with self.assertRaises(OSError):
                    gate.backend(root / "binary", root / "ldd")

    def test_failed_selected_rust_markers_not_unrelated_compile_words(self):
        identity = gate.GENERAL[0]
        failure_line = f"FAIL [   0.017s] (1/2) mvmc-core::ctest_general_reference {identity}\n"
        cases = [
            (True, failure_line + "parity gate ctest-general: MissingFixture: absent\n", "MissingFixture"),
            (True, failure_line + "parity gate ctest-general: Unsupported: input\n", "Unsupported"),
            (True, "compiler error mentions MissingFixture and Unsupported\n", "Failure"),
            (True, "test helper ... FAILED\nparity gate ctest-general: MissingFixture: absent\n", "Failure"),
            (True, f"test {identity} ... FAILED\nparity gate unrelated: MissingFixture: absent\n", "Failure"),
            (False, f"test {identity} ... FAILED\nparity gate ctest-general: MissingFixture: compile text\n", "Failure"),
        ]
        for fail_gate, diagnostic, expected in cases:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                binary = root / "mock-binary"
                binary.write_bytes(b"simulated; not executable")
                data = listing()
                data["rust-suites"]["one"]["binary-path"] = str(binary)
                def fake_run(args, **kwargs):
                    is_gate = args[:3] == ["cargo", "nextest", "run"]
                    if is_gate == fail_gate:
                        kwargs["stderr"].write(diagnostic)
                        return SimpleNamespace(returncode=100)
                    if args[:3] == ["cargo", "nextest", "list"]:
                        kwargs["stdout"].write(json.dumps(data))
                    return SimpleNamespace(returncode=0)
                def fake_backend(_, path):
                    path.write_text("mock linkage\n")
                    return {"mock": True}
                with patch.object(gate.platform, "system", return_value="Linux"), \
                        patch.dict(gate.os.environ, {"CARGO_TARGET_DIR": str(root / "target")}), \
                        patch.object(subprocess, "check_output", return_value="mock revision"), \
                        patch.object(subprocess, "run", side_effect=fake_run), \
                        patch.object(gate, "source_files", return_value=[binary]), \
                        patch.object(gate, "fixture_files", return_value=[binary]), \
                        patch.object(gate, "backend", side_effect=fake_backend):
                    output = root / "evidence"
                    with self.assertRaises((ValueError, RuntimeError)):
                        gate.run("general", output)
                    terminal = json.loads((output / "terminal.json").read_text())
                    row = json.loads((output / "family-ledger.json").read_text())["families"]["general"]
                    self.assertEqual(terminal["status"], expected)
                    self.assertEqual(row["status"], expected)
                    self.assertEqual(row["started_driver_invocations"], int(fail_gate))
                    self.assertEqual(row["completed_selection_identities"], [])
                    self.assertEqual(row["selected_test_identities"], list(gate.GENERAL) if fail_gate else [])

    def test_representative_retained_nextest_layout_strict_suite_identity(self):
        # Layout transcribed from actual failed run2d16f9fd; no runtime artifact dependency.
        diagnostic = (
            "    Starting 1 test across 1 binary (8 tests skipped)\n"
            f"        FAIL [   3.949s] (1/1) mvmc-core::lanczos_transfer_physcal {gate.LANCZOS}\n"
            "  stdout ───\n\n    running 1 test\n"
            "  stderr ───\n"
            "    parity gate lanczos-physcal: MissingFixture: missing independent reference\n"
        )
        self.assertEqual(gate.selected_failure_status("lanczos", [gate.LANCZOS], diagnostic), "MissingFixture")
        self.assertEqual(gate.selected_failure_status("lanczos", [gate.LANCZOS], diagnostic.replace("MissingFixture:", "Unsupported:")), "Unsupported")
        for bad in (diagnostic.replace("lanczos_transfer_physcal", "unrelated_suite"),
                    diagnostic.replace(gate.LANCZOS, "support::helper"),
                    diagnostic.replace("        FAIL", "compiler quotes: FAIL"),
                    diagnostic.replace("parity gate lanczos-physcal:", "parity gate unrelated:"),
                    "parity gate lanczos-physcal: MissingFixture: absent\n",
                    f"test {gate.LANCZOS} ... FAILED\nparity gate lanczos-physcal: MissingFixture: absent\n"):
            self.assertEqual(gate.selected_failure_status("lanczos", [gate.LANCZOS], bad), "Failure")
        mpi = f"test {gate.MPI} ... FAILED\nparity gate mpi-physcal: Unsupported: input\n"
        self.assertEqual(gate.selected_failure_status("mpi", [gate.MPI], mpi), "Unsupported")

    def test_artifact_failure_invalidates_numeric_totals_not_execution_facts(self):
        row = gate.family_ledger("lanczos")["lanczos"]
        row.update(started_driver_invocations=6, completed_selection_identities=["all six verified selections"],
                   numeric_reference_comparisons=8, empty_contracts=4, comparison_evidence="Verified")
        gate.invalidate_comparison_counts(row)
        self.assertIsNone(row["numeric_reference_comparisons"])
        self.assertEqual(row["empty_contracts"], 0)
        self.assertEqual(row["comparison_evidence"], "Unverified")
        self.assertEqual(row["started_driver_invocations"], 6)
        self.assertEqual(row["completed_selection_identities"], ["all six verified selections"])

    def test_dc_exact_selected_identity_and_empty_not_numeric(self):
        for model in gate.MODELS[1:]:
            for mode in ("real", "cmp"):
                lines = [f"OPTIONAL183_DC model={model} mode={mode} file={name} status={status}"
                         for name, status in (
                             ("zvo_ls_cisajs_001.dat", "REFERENCE_COMPARED"),
                             ("zvo_ls_cisajscktalt_001.dat", "REFERENCE_COMPARED"),
                             ("zvo_ls_cisajscktaltex_001.dat", "EMPTY_CONTRACT"))]
                text = "\n".join(lines)
                records = gate.validate_dc_run(text, model, mode)
                # Actual nextest layout: empty stdout, markers on stderr.
                self.assertEqual(gate.validate_dc_run("" + "\n" + text, model, mode), records)
                # Split streams must retain exact identity/count checks.
                self.assertEqual(gate.validate_dc_run(lines[0] + "\n" + "\n".join(lines[1:]), model, mode), records)
                for stdout, stderr in ((text, lines[0]), ("", "\n".join(lines[:2])),
                                       ("", text.replace(f"model={model}", "model=helper"))):
                    with self.assertRaises(ValueError):
                        gate.validate_dc_run(stdout + "\n" + stderr, model, mode)
                self.assertEqual(sum(r["status"] == "REFERENCE_COMPARED" for r in records), 2)
                self.assertEqual(sum(r["status"] == "EMPTY_CONTRACT" for r in records), 1)
                for bad in ("", "\n".join(lines[:2]), text + "\n" + lines[0],
                            text.replace("EMPTY_CONTRACT", "PASS"),
                            text.replace(f"model={model}", "model=helper"),
                            text.replace(f"mode={mode}", "mode=unknown")):
                    with self.assertRaises(ValueError):
                        gate.validate_dc_run(bad, model, mode)
                with self.assertRaises(ValueError):
                    gate.validate_dc_run(text, "hubbard_chain_real", mode)
        self.assertEqual(gate.validate_dc_run("", "hubbard_chain_real", "real"), [])
        for model, mode in (("interall", "real"), ("spin_chain_lanczos", "unknown")):
            with self.assertRaises(ValueError):
                gate.validate_dc_run("", model, mode)

    def test_exact_count_and_identities_not_helpers(self):
        self.assertEqual(gate.selected_binary(listing(), gate.GENERAL), Path("/tmp/not-executed"))
        for names in ((), gate.GENERAL[:1], (*gate.GENERAL, "support::helper")):
            with self.assertRaises(ValueError):
                gate.selected_binary(listing(names), gate.GENERAL)
        data = listing()
        data["rust-suites"]["one"]["testcases"][gate.GENERAL[0]]["ignored"] = False
        with self.assertRaises(ValueError):
            gate.selected_binary(data, gate.GENERAL)
        data = listing()
        data["rust-suites"]["duplicate"] = copy.deepcopy(data["rust-suites"]["one"])
        with self.assertRaises(ValueError):
            gate.selected_binary(data, gate.GENERAL)

    def test_cargo_exact_one_executable(self):
        record = {"reason": "compiler-artifact", "target": {"name": "mpi_physcal"},
                  "executable": "/tmp/not-executed"}
        self.assertEqual(gate.cargo_binary([record], "mpi_physcal"), Path("/tmp/not-executed"))
        for records in ([], [record, record]):
            with self.assertRaises(ValueError):
                gate.cargo_binary(records, "mpi_physcal")

    def test_unknown_empty_scope_no_commands(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(subprocess, "run") as run:
            for family in ("", "all", "general;echo injected", "interall"):
                with self.assertRaises(ValueError):
                    gate.run(family, Path(tmp) / "unused")
            run.assert_not_called()
            self.assertFalse((Path(tmp) / "unused").exists())

    def test_missing_changed_empty_artifacts_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                gate.validate_artifacts(root, ["terminal.json"])
            (root / "terminal.json").touch()
            with self.assertRaises(ValueError):
                gate.validate_artifacts(root, ["terminal.json"])
            (root / "terminal.json").write_text('{"exit_status":0}\n')
            gate.validate_artifacts(root, ["terminal.json"])
            before = gate.digest_files([root / "terminal.json"])
            (root / "terminal.json").write_text('{"exit_status":1}\n')
            self.assertNotEqual(before, gate.digest_files([root / "terminal.json"]))
            with self.assertRaises(ValueError):
                gate.digest_files([])

    def test_interall_and_missing_closure_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "namelist.def"
            (root / "modpara.def").write_text("Nsite 2\n")
            path.write_text("ModPara modpara.def\n")
            gate.validate_namelist(path)
            path.write_text("# InterAll absent.def\nModPara modpara.def # valid inline comment\n")
            gate.validate_namelist(path)
            path.write_text("ModPara absent.def\n")
            with self.assertRaises(gate.MissingFixtureError):
                gate.validate_namelist(path)
            path.write_text("InterAll modpara.def\n")
            with self.assertRaises(gate.UnsupportedError):
                gate.validate_namelist(path)
        # Every permitted original model is checked, not just a name whitelist.
        gate.fixture_files("lanczos")

    def test_backend_linkage_without_runtime_identity_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(subprocess, "check_output", return_value="libblas.so => /mock/libblas.so\n"):
                with self.assertRaises(ValueError):
                    gate.backend(Path("/not/executed"), Path(tmp) / "ldd.txt")

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(gate.platform, "system", return_value="Linux"):
            with self.assertRaises(FileExistsError):
                gate.run("general", Path(tmp))

    def test_mpi_rank_pass_and_actual_world_group_counts(self):
        for ranks in (2, 4):
            summaries = "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured;\n" * ranks
            markers = "".join(f"OPTIONAL183_MPI rank={rank} world={ranks} width={width} group_size={ranks if width == 1 else 2}\n"
                              for rank in range(ranks) for width in (1, 2))
            gate.validate_mpi_run(summaries + markers, ranks)
            for bad in ("", summaries, markers, summaries + markers + markers,
                        summaries + markers.replace("rank=0", "rank=99"),
                        summaries + markers.replace(f"world={ranks}", "world=1"),
                        summaries + markers.replace("group_size=2", "group_size=1"),
                        summaries.replace("0 failed", "1 failed") + markers):
                with self.assertRaises(ValueError):
                    gate.validate_mpi_run(bad, ranks)

    def test_compile_closure_contains_numerical_support(self):
        paths = gate.source_files()
        self.assertIn(gate.ROOT / "tests/support/numerical_comparison.rs", paths)
        self.assertIn(gate.ROOT / "rustfmt.toml", paths)
        self.assertTrue(any("xtask" in p.parts for p in paths))


if __name__ == "__main__":
    unittest.main()

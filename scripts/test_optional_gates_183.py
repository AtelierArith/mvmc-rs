"""Infrastructure tests, NOT executed numerical gate coverage."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import run_optional_gates_183 as gate


def listing(names=gate.GENERAL):
    return {"rust-suites": {"one": {"binary-path": "/tmp/not-executed",
            "testcases": {name: {"ignored": True, "filter-match": {"status": "matches"}}
                          for name in names}}}}


class OptionalGateContract(unittest.TestCase):
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
            with self.assertRaises(ValueError):
                gate.validate_namelist(path)
            path.write_text("InterAll modpara.def\n")
            with self.assertRaises(ValueError):
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

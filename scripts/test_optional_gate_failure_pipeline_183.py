"""Real driver/seal/aggregate subprocess errors; no numerical gate execution."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import run_optional_gates_183 as driver_source


SCRIPTS = Path(__file__).resolve().parent
MODULES = ("run_optional_gates_183.py", "aggregate_optional_gates_183.py",
           "optional_mpi_provider_183.py", "optional_fixture_metadata_183.py",
           "optional_reference_provenance_183.py")
MODELS = ("hubbard_chain_real", "hubbard_chain_lanczos", "spin_chain_lanczos")


class RealFailurePipeline(unittest.TestCase):
    def test_actual_command_lifecycle_discovery_failure_timeout_and_incomplete(self):
        # Real tiny Python children exercise the production command boundary;
        # none is a selected numerical gate or a mocked driver success.
        cases = (("success", [sys.executable, "-B", "-c", "print('discovery')"], None, 0),
                 ("nonzero", [sys.executable, "-B", "-c", "raise SystemExit(7)"], None, 7),
                 ("timeout", [sys.executable, "-B", "-c", "import time; time.sleep(1)"], subprocess.TimeoutExpired, None),
                 ("missing", ["/nonexistent-issue183-command"], FileNotFoundError, None))
        for name, argv, error, code in cases:
            with self.subTest(case=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                commands = []
                if error is None:
                    result = driver_source.execute_command(argv, name, root, commands,
                                                           os.environ.copy(), root, timeout=1)
                    self.assertEqual(result.returncode, code)
                else:
                    with self.assertRaises(error):
                        driver_source.execute_command(argv, name, root, commands,
                                                      os.environ.copy(), root, timeout=0.05)
                receipt = json.loads((root / "commands.json").read_text())
                self.assertEqual(receipt, commands)
                self.assertEqual(len(receipt), 1)
                record = receipt[0]
                self.assertIsNone(record["selected"])
                self.assertEqual(record["argv"], argv)
                self.assertEqual(record["returncode"], code)
                self.assertEqual(record["state"], "Timeout" if name == "timeout" else
                                 "Incomplete" if name == "missing" else "Completed")
                self.assertEqual(record["exception"], error.__name__ if error else None)
                self.assertGreaterEqual(record["ended_monotonic_ns"], record["started_monotonic_ns"])
                self.assertTrue((root / record["stdout"]).is_file())
                self.assertTrue((root / record["stderr"]).is_file())
                if name == "success":
                    self.assertEqual((root / record["stdout"]).read_text(), "discovery\n")

    def command(self, argv, root, env, expected=0):
        result = subprocess.run(argv, cwd=root, env=env, capture_output=True,
                                text=True, timeout=30)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def test_real_classified_failures_survive_seal_and_aggregate(self):
        for case, status in (("manifest", "MissingFixture"),
                             ("referenced_input", "MissingFixture"),
                             ("interall", "Unsupported"), ("invalid_utf8", "Failure"),
                             ("metadata_mismatch", "Failure"),
                             ("metadata_missing", "Failure"),
                             ("metadata_duplicate", "Failure")):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                scripts = root / "scripts"
                scripts.mkdir()
                for name in MODULES:
                    shutil.copyfile(SCRIPTS / name, scripts / name)
                env = os.environ.copy()
                env.update(PYTHONDONTWRITEBYTECODE="1", CARGO_TERM_COLOR="never",
                           CARGO_TARGET_DIR=str(root / "unused-target"))
                for key in ("GITHUB_OUTPUT", "MVMC_ISSUE234_MPI_PREFIX",
                            "MVMC_ISSUE234_MPI_RECEIPT"):
                    env.pop(key, None)
                self.command(["git", "init", "--quiet"], root, env)
                self.command(["git", "add", "scripts"], root, env)
                self.command(["git", "-c", "user.name=infra-test", "-c",
                              "user.email=infra-test@example.invalid", "commit", "--quiet",
                              "-m", "isolated exact driver sources"], root, env)
                head = self.command(["git", "rev-parse", "HEAD"], root, env).stdout.strip()

                reference = root / "extern/Julia-mVMC"
                reference.mkdir(parents=True)
                manifest = reference / "Manifest-v1.13.toml"
                manifest.write_text('julia_version = "1.13.1"\n# synthetic; no Julia runtime\n')
                for model, samples in zip(MODELS, (100, 1000, 5000)):
                    inputs = reference / "test/integration/reference" / model / "physcal_ref/inputs"
                    inputs.mkdir(parents=True)
                    (inputs / "modpara.def").write_text(
                        f"Nsite 2\nNVMCSample {samples}\nNDataQtySmp 1\n"
                        "NVMCWarmUp 10\nNVMCInterval 1\nRndSeed 1\n"
                        "NVMCCalMode 1\nNLanczosMode 2\n")
                    (inputs / "namelist.def").write_text("ModPara modpara.def\n")
                # Real fixture selection succeeds before intentional mutation; this
                # is only a filesystem closure control, never a simulated model Pass.
                self.command([sys.executable, "-B", "-c",
                              "import sys;sys.path.insert(0,'scripts');"
                              "import run_optional_gates_183 as g;"
                              "assert len(set(g.fixture_files('lanczos')))==7;"
                              "r=g.fixture_metadata.capture(g.ROOT);"
                              "assert [v['declared']['NVMCSample'] for v in r['fixtures']]==[100,1000,5000];"
                              "assert r['explicit_gate_overrides']=={'seed':1,'modes':['real','cmp']}"], root, env)
                namelist = reference / "test/integration/reference" / MODELS[0] / "physcal_ref/inputs/namelist.def"
                if case == "manifest":
                    manifest.unlink()
                elif case == "referenced_input":
                    namelist.write_text("ModPara absent.def\n")
                elif case == "interall":
                    namelist.write_text("InterAll modpara.def\n")
                elif case == "invalid_utf8":
                    namelist.write_bytes(b"ModPara \xff\n")
                else:
                    modpara = namelist.parent / "modpara.def"
                    original = modpara.read_text()
                    if case == "metadata_mismatch":
                        modpara.write_text(original.replace("NVMCSample 100\n", "NVMCSample 101\n"))
                        expected_metadata_error = "fixture declarations mismatch reviewed gate report"
                    elif case == "metadata_missing":
                        modpara.write_text(original.replace("NVMCSample 100\n", ""))
                        expected_metadata_error = "missing required fixture report fields"
                    else:
                        modpara.write_text(original + "NVMCSample 100\n")
                        expected_metadata_error = "ambiguous fixture report field: NVMCSample"
                    self.assertNotEqual(modpara.read_text(), original)

                # Guard executable boundary, not the driver: only existing Cargo
                # version discovery may reach the real Cargo binary. No build/test.
                real_cargo = shutil.which("cargo")
                self.assertIsNotNone(real_cargo)
                guard = root / "guard"
                guard.mkdir()
                calls = root / "cargo-calls.jsonl"
                shim = guard / "cargo"
                shim.write_text("#!" + sys.executable + "\nimport os,sys,json\n"
                                "with open(os.environ['CARGO_CALL_RECEIPT'],'a') as f:"
                                "f.write(json.dumps(sys.argv[1:])+'\\n')\n"
                                "if sys.argv[1:] != ['nextest','--version']: sys.exit(97)\n"
                                "os.execv(os.environ['REAL_CARGO'],"
                                "[os.environ['REAL_CARGO']]+sys.argv[1:])\n")
                shim.chmod(0o700)
                env.update(PATH=str(guard) + os.pathsep + env["PATH"],
                           REAL_CARGO=real_cargo, CARGO_CALL_RECEIPT=str(calls))
                originals = {name: hashlib.sha256((scripts / name).read_bytes()).hexdigest()
                             for name in MODULES}
                evidence = root / "evidence"
                driver = self.command([sys.executable, "-B", str(scripts / MODULES[0]),
                                      "lanczos", str(evidence)], root, env, expected=1)
                if case.startswith("metadata_"):
                    self.assertIn("optional gate failed: " + expected_metadata_error,
                                  driver.stderr)
                self.assertEqual([json.loads(line) for line in calls.read_text().splitlines()],
                                 [["nextest", "--version"]])
                command_receipts = json.loads((evidence / "commands.json").read_text())
                self.assertEqual([record["name"] for record in command_receipts],
                                 ["rust-version", "nextest-version", "reference-revisions", "working-tree-status"])
                for record in command_receipts:
                    self.assertIsNone(record["selected"], "discovery cannot become selected completion")
                    self.assertEqual((record["state"], record["returncode"], record["exception"]),
                                     ("Completed", 0, None))
                    self.assertGreaterEqual(record["ended_monotonic_ns"], record["started_monotonic_ns"])
                terminal = json.loads((evidence / "terminal.json").read_text())
                row = json.loads((evidence / "family-ledger.json").read_text())["families"]["lanczos"]
                self.assertEqual((terminal["exit_status"], terminal["status"], terminal["completed"]),
                                 (1, status, []))
                self.assertEqual((row["status"], row["started_driver_invocations"]), (status, 0))
                self.assertFalse((root / "unused-target").exists())

                aggregate = str(scripts / MODULES[1])
                binding = ["--head", head, "--run-id", "183", "--attempt", "1",
                           "--workflow", "local-real-failure-infrastructure"]
                plan = root / "plan.json"
                self.command([sys.executable, "-B", aggregate, "plan", *binding,
                              "--scope", "lanczos", "--output", str(plan)], root, env)
                packages = root / "packages"
                packages.mkdir()
                package = packages / "optional-family-lanczos-183-1"
                self.command([sys.executable, "-B", aggregate, "seal", *binding,
                              "--plan", str(plan), "--family", "lanczos", "--evidence",
                              str(evidence), "--job-status", "failure", "--output",
                              str(package)], root, env)
                output = root / "aggregate.json"
                self.command([sys.executable, "-B", aggregate, "aggregate", *binding,
                              "--plan", str(plan), "--packages", str(packages),
                              "--output", str(output)], root, env, expected=1)
                result = json.loads(output.read_text())
                self.assertEqual(result["exit_status"], 1)
                self.assertEqual(result["families"]["lanczos"]["status"], status)
                self.assertTrue(result["families"]["lanczos"]["selected"])
                self.assertEqual(result["families"]["lanczos"]["comparison_evidence"],
                                 "Unverified")
                self.assertIsNone(result["families"]["lanczos"]["numeric_reference_comparisons"])
                self.assertEqual(result["families"]["lanczos"]["empty_contracts"], 0)
                for family in ("general", "mpi", "thread"):
                    self.assertEqual(result["families"][family],
                                     {"selected": False, "status": "NotRun"})
                self.assertEqual(originals, {name: hashlib.sha256((scripts / name).read_bytes()).hexdigest()
                                             for name in MODULES})


if __name__ == "__main__":
    unittest.main()

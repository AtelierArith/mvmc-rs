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
    # Literal independent fresh-B closure declarations, not production-generated
    # POSTs or numerical execution. Legacy 305 evidence is not rewritten.
    posts = {"source": {"status": "UNCHANGED", "error": None},
             "fixtures": {"status": "UNCHANGED", "error": None},
             "binary": {"status": "UNCHANGED", "error": None}}
    put(evidence / "closure-posts.json", posts)
    put(evidence / "terminal.json", {"status": "Pass", "exit_status": 0,
                                      "completed": completed, "closure_posts": posts})
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
        if stem == "binary":
            put(evidence / "binary.after.json", {"synthetic-path": "b" * 64})
    if family == "lanczos":
        report = {"schema": 1, "authority": "static fixture declaration; NOT runtime observation",
                  "explicit_gate_overrides": {"seed": 1, "modes": ["real", "cmp"]},
                  "fixtures": [{"model": model,
                                "path": f"/synthetic/{model}/physcal_ref/inputs/modpara.def",
                                "sha256": "b" * 64,
                                "declared": audit.fixture_metadata.expected(model)}
                               for model in audit.MODELS]}
        metadata = audit.read_json(evidence / "metadata.json")
        metadata["fixture_modpara"] = report
        put(evidence / "metadata.json", metadata)
        for suffix in ("before", "after"):
            put(evidence / f"fixtures.{suffix}.json",
                {row["path"]: row["sha256"] for row in report["fixtures"]})
    # Independent literal declarations, not a simulated Julia/model execution.
    manifest_path = "/synthetic/extern/Julia-mVMC/Manifest-v1.13.toml"
    declaration_path = "/synthetic/reference/provenance.txt"
    manifest_text = 'julia_version = "1.13.1"\n'
    declaration_text = "C historical acquisition; Julia version not recorded\n"
    import hashlib
    declarations = [
        {"path": manifest_path, "sha256": hashlib.sha256(manifest_text.encode()).hexdigest(),
         "text": manifest_text, "kind": "checkout_manifest", "julia_version": "1.13.1"},
        {"path": declaration_path, "sha256": hashlib.sha256(declaration_text.encode()).hexdigest(),
         "text": declaration_text, "kind": "historical_acquisition_declaration", "julia_versions": "NotRecorded"},
    ]
    for suffix in ("before", "after"):
        path = evidence / f"fixtures.{suffix}.json"
        state = audit.read_json(path)
        state.update({item["path"]: item["sha256"] for item in declarations})
        put(path, state)
    metadata = audit.read_json(evidence / "metadata.json")
    metadata.update(oracle_execution="none; checked-in fixtures only",
                    reference_version="per-fixture provenance, NOT current Julia runtime verification",
                    not_claimed="full13 matrix, fullJulia features, fullC sampler, InterAll",
                    offline_reference={"schema": 1,
                        "authority": "hash-bound offline declarations; NOT current runtime verification",
                        "julia_runtime": {"status": "NotRun", "version": None, "blas": None},
                        "declarations": declarations})
    put(evidence / "metadata.json", metadata)
    put(evidence / "backend.json", {"actual_threads": 1, "actual_config": "OpenBLAS 0.3.26 synthetic",
                                   "actual_core": "Haswell", "library_sha256": "c" * 64})
    put(evidence / "selection.json", {"synthetic": True})
    # Independent literal command receipts, not subprocess/model execution.
    calls = {"general": ["general"], "lanczos": [f"{m}-{c}" for m in audit.MODELS for c in ("real", "cmp")],
             "mpi": ["mpi-2", "mpi-4"], "thread": ["thread"]}[family]
    commands = []
    for name in calls:
        env = {key: "1" for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS")}
        env["CARGO_TARGET_DIR"] = "/synthetic/target"
        if family in ("general", "lanczos"):
            target = "ctest_general_reference" if family == "general" else "lanczos_transfer_physcal"
            expr = " | ".join(f"test(={identity})" for identity in identities)
            argv = ["cargo", "nextest", "run", "--locked", "-p", "mvmc-core", "--cargo-profile", "test-fast",
                    "--test", target, "--run-ignored", "only", "-E", expr, "--no-fail-fast", "--retries", "0",
                    "--success-output", "immediate", "--failure-output", "immediate"]
            if family == "general":
                env["MVMC_RS_CTEST_GENERAL"] = "1"
            else:
                model, mode = name.rsplit("-", 1)
                env.update(MVMC_RS_LANCZOS_PHYSICAL="1", MVMC_RS_LANCZOS_MODEL=model, MVMC_RS_LANCZOS_MODE=mode)
        elif family == "mpi":
            world = name.split("-")[1]
            argv = ["mpiexec", "-n", world, "synthetic-path", "--ignored", "--exact", audit.MPI, "--nocapture"]
            env.update(MVMC_RS_MPI_PHYSICAL="1", MPI179_PHYSCAL_OUTPUT=str(evidence / f"world-{world}"))
        else:
            argv = ["bash", "scripts/verify_threaded_issue182.sh"]
            env.update(MVMC_RS_THREADED_TARGET_DIR="/synthetic/target", TMPDIR=str(evidence / "thread-artifacts"))
        commands.append({"schema": 1, "name": name, "argv": argv, "cwd": "/synthetic/checkout", "env": env,
                         "selected": {"family": family, "profile": "test-fast",
                                      "features": "mpi" if family == "mpi" else "default", "identities": identities},
                         "timeout_seconds": 2400, "state": "Completed", "returncode": 0, "exception": None,
                         "started_unix_ns": 100, "ended_unix_ns": 200, "started_monotonic_ns": 10,
                         "ended_monotonic_ns": 20, "stdout": f"{name}.stdout", "stderr": f"{name}.stderr"})
        for stream in ("stdout", "stderr"):
            (evidence / f"{name}.{stream}").write_text("")
    if family == "mpi":
        acquisition_argv = [
            ("build", ["cargo", "test", "--locked", "-p", "mvmc-core", "--profile", "test-fast",
                       "--features", "mpi", "--test", "mpi_physcal", "--no-run", "--message-format=json"]),
            ("selection", ["synthetic-path", "--list"])]
    else:
        target = {"general": "ctest_general_reference", "lanczos": "lanczos_transfer_physcal",
                  "thread": "threaded_issue182"}[family]
        expr = " | ".join(f"test(={identity})" for identity in identities)
        acquisition_argv = [("selection", ["cargo", "nextest", "list", "--locked", "-p", "mvmc-core",
                            "--cargo-profile", "test-fast", "--test", target, "--run-ignored", "only", "-E", expr,
                            "--message-format", "json"])]
    acquisitions = []
    for name, argv in acquisition_argv:
        acquisitions.append({"schema": 1, "name": name, "argv": argv, "cwd": "/synthetic/checkout",
                             "env": {"CARGO_TARGET_DIR": "/synthetic/target"}, "selected": None,
                             "timeout_seconds": 2400, "state": "Completed", "returncode": 0, "exception": None,
                             "started_unix_ns": 1, "ended_unix_ns": 2, "started_monotonic_ns": 1,
                             "ended_monotonic_ns": 2, "stdout": f"{name}.stdout", "stderr": f"{name}.stderr"})
        for stream in ("stdout", "stderr"):
            (evidence / f"{name}.{stream}").write_text("")
    put(evidence / "commands.json", acquisitions + commands)
    metadata = audit.read_json(evidence / "metadata.json")
    metadata["command_receipt_schema"] = 1
    metadata["closure_post_schema"] = 1
    metadata["checkout_root"] = "/synthetic/checkout"
    put(evidence / "metadata.json", metadata)
    for name in ("linkage.txt", "rust-version.stdout", "nextest-version.stdout", "reference-revisions.stdout"):
        (evidence / name).write_text("synthetic infrastructure evidence only\n")
    if family == "mpi":
        mpi_fixture(evidence)
    put(evidence / "artifacts.json", {str(path): audit.digest(path)
        for path in evidence.rglob("*") if path.is_file()})
    package = packages / audit.artifact_name(family, plan)
    with patch.object(audit, "checkout_head", return_value=plan["head"]):
        audit.seal(plan, family, evidence, package, "success", plan["head"], "never")
    return package


def mpi_fixture(evidence):
    """Synthetic dependency receipts, not an installed MPI or native run."""
    prefix = Path("/synthetic/mpich")
    receipt = evidence / "mpi-provider-receipt"
    receipt.mkdir()
    (receipt / "terminal.txt").write_text("prior=0 post=0\n")
    (receipt / "ready.txt").write_text(audit.mpi_provider.READY)
    (receipt / "mpichversion.txt").write_text(
        f"MPICH Version: 4.2.0\nMPICH configure: --prefix={prefix} --with-pm=hydra --with-pmi=pmi1 --without-pmix\n")
    for name in audit.mpi_provider.MANIFESTS:
        (receipt / f"{name}.sha256").write_text("b" * 64 + "  /synthetic/source\n")
        (receipt / f"{name}.post.log").write_text("")
    for world in (2, 4):
        (receipt / f"world-{world}.native.status").write_text("0\n")
        (receipt / f"world-{world}.stdout").write_text("".join(
            f"MPI_STARTUP rank={rank} size={world} required=1 provided=1 rank_sum={world*(world-1)//2} ok=1\n"
            for rank in range(world)))
    state = {str(prefix / path): "b" * 64 for path in
             ("bin/mpicc", "bin/mpiexec.hydra", "include/mpi.h", "lib/libmpi.so.12.4.0")}
    for suffix in ("before", "after"):
        put(evidence / f"mpi-provider.{suffix}.json", state)
    library = {"path": str(prefix / "lib/libmpi.so.12"),
               "resolved": str(prefix / "lib/libmpi.so.12.4.0"), "sha256": "b" * 64}
    (evidence / "mpi-linkage.txt").write_text(f"libmpi.so.12 => {library['path']} (0x123)\n")
    put(evidence / "mpi-provider.json", {"schema": 1, "prefix": str(prefix),
        "receipt_root": "/synthetic/receipt", "receipt_hashes": audit.mpi_provider.receipt_state(receipt),
        "binary_sha256": "b" * 64, "library": library,
        "tools": {"mpicc": {"path": str(prefix / "bin/mpicc"), "sha256": "b" * 64},
                  "mpiexec": {"path": str(prefix / "bin/mpiexec.hydra"), "sha256": "b" * 64}}})


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
    def test_rehashed_closure_posts_join_terminal_and_fresh_schema(self):
        variants = ("schema_missing", "schema_bool", "schema_float", "schema_old",
                    "file_missing", "terminal_missing", "terminal_list", "file_list",
                    "missing_closure", "extra_closure", "terminal_contradiction",
                    "record_list", "record_extra", "error", "status_bool",
                    "CHANGED", "NOT_STARTED", "UNAVAILABLE", "binary_hash_contradiction")
        for family in audit.FAMILIES:
            for variant in variants:
                with self.subTest(family=family, variant=variant), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    packages = root / "packages"
                    packages.mkdir()
                    plan = audit.make_plan(family, BINDING)
                    package = fixture(root, packages, plan, family)
                    self.assertEqual(audit.aggregate(plan, packages, BINDING)["exit_status"], 0)
                    evidence = package / "evidence"
                    metadata = audit.read_json(evidence / "metadata.json")
                    terminal = audit.read_json(evidence / "terminal.json")
                    posts = audit.read_json(evidence / "closure-posts.json")
                    expected = "terminal/file closure POST mismatch"
                    if variant.startswith("schema_"):
                        if variant == "schema_missing":
                            metadata.pop("closure_post_schema")
                        else:
                            metadata["closure_post_schema"] = {"schema_bool": True, "schema_float": 1.0,
                                                               "schema_old": 0}[variant]
                        expected = "historical closure POST receipts lack current schema; not current-eligible"
                    elif variant == "file_missing":
                        (evidence / "closure-posts.json").unlink()
                        expected = "missing fresh closure POST file"
                    elif variant == "terminal_missing":
                        terminal.pop("closure_posts")
                    elif variant == "terminal_list":
                        terminal["closure_posts"] = []
                    elif variant == "file_list":
                        posts = []
                        expected = "expected JSON object"
                    elif variant == "missing_closure":
                        posts.pop("binary")
                        terminal["closure_posts"] = copy.deepcopy(posts)
                    elif variant == "extra_closure":
                        posts["unknown"] = {"status": "UNCHANGED", "error": None}
                        terminal["closure_posts"] = copy.deepcopy(posts)
                    elif variant == "terminal_contradiction":
                        terminal["closure_posts"]["source"]["status"] = "CHANGED"
                    elif variant == "binary_hash_contradiction":
                        put(evidence / "binary.after.json", {"synthetic-path": "c" * 64})
                        expected = "closure POST claim contradicts captured before/after hashes"
                    else:
                        expected = "successful driver requires three unchanged closure POSTs"
                        if variant == "record_list":
                            posts["source"] = []
                        elif variant == "record_extra":
                            posts["source"]["extra"] = None
                        elif variant == "error":
                            posts["source"]["error"] = "preserved POST error"
                        elif variant == "status_bool":
                            posts["source"]["status"] = True
                        else:
                            posts["source"]["status"] = variant
                        terminal["closure_posts"] = copy.deepcopy(posts)
                    put(evidence / "metadata.json", metadata)
                    put(evidence / "terminal.json", terminal)
                    if variant != "file_missing":
                        put(evidence / "closure-posts.json", posts)
                    self.assert_rehashed_type_failure(package, plan, packages, family, expected)

    def test_rehashed_command_receipts_bind_actual_calls_not_discovery(self):
        variants = ("absent_schema", "schema_bool", "legacy", "discovery_only", "duplicate", "missing_call",
                    "profile", "features", "identity", "argv", "environment", "target", "thread_setting",
                    "started", "timeout", "incomplete", "nonzero", "bool_exit", "backwards", "stream",
                    "checkout", "acquisition", "acquisition_missing", "discovery_selected")
        for family in audit.FAMILIES:
            for variant in variants:
                with self.subTest(family=family, variant=variant), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    packages = root / "packages"
                    packages.mkdir()
                    plan = audit.make_plan(family, BINDING)
                    package = fixture(root, packages, plan, family)
                    self.assertEqual(audit.aggregate(plan, packages, BINDING)["exit_status"], 0)
                    evidence = package / "evidence"
                    commands = json.loads((evidence / "commands.json").read_text())
                    record = next(command for command in commands if command["selected"] is not None)
                    expected = "unfinished/failed/invalid command lifecycle"
                    if variant in ("absent_schema", "schema_bool"):
                        metadata = audit.read_json(evidence / "metadata.json")
                        if variant == "absent_schema":
                            metadata.pop("command_receipt_schema")
                        else:
                            metadata["command_receipt_schema"] = True
                        put(evidence / "metadata.json", metadata)
                        expected = "historical command receipts lack execution schema; not current-eligible"
                    elif variant == "legacy":
                        commands = [{"argv": record["argv"], "env": record["env"]}]
                        expected = "invalid/duplicate command receipt identity"
                    elif variant == "discovery_only":
                        for command in commands:
                            command["selected"] = None
                        expected = "selected command coverage/order mismatch; discovery is not completion"
                    elif variant == "duplicate":
                        commands.append(copy.deepcopy(record))
                        expected = "invalid/duplicate command receipt identity"
                    elif variant == "missing_call":
                        commands.pop()
                        expected = ("actual command receipt list missing" if not commands else
                                    "selected command coverage/order mismatch; discovery is not completion")
                    elif variant in ("profile", "features", "identity"):
                        if variant == "identity":
                            record["selected"]["identities"] = ["helper"]
                        else:
                            record["selected"][variant] = "wrong"
                        expected = "command selection/profile/features mismatch"
                    elif variant == "argv":
                        record["argv"] = ["cargo", "nextest", "--version"]
                        expected = "selected command argv/gate environment mismatch"
                    elif variant == "environment":
                        record["env"] = {key: value for key, value in record["env"].items()
                                         if not key.startswith(("MVMC_", "MPI179_", "TMPDIR"))}
                        expected = "selected command argv/gate environment mismatch"
                    elif variant == "target":
                        record["env"]["CARGO_TARGET_DIR"] = "/other-target"
                        expected = "selected command target binding mismatch"
                    elif variant == "thread_setting":
                        record["env"]["OPENBLAS_NUM_THREADS"] = "2"
                        expected = "selected command backend thread setting mismatch"
                    elif variant == "checkout":
                        record["cwd"] = "/different-checkout"
                        expected = "invalid command argv/cwd/environment"
                    elif variant == "acquisition":
                        commands[0]["argv"] = ["cargo", "nextest", "--version"]
                        expected = "build/selection command profile/features/order mismatch"
                    elif variant == "acquisition_missing":
                        commands.pop(0)
                        expected = "build/selection command profile/features/order mismatch"
                    elif variant == "discovery_selected":
                        commands[0]["selected"] = copy.deepcopy(record["selected"])
                        expected = "selected command coverage/order mismatch; discovery is not completion"
                    elif variant in ("started", "timeout", "incomplete"):
                        record["state"] = {"started": "Started", "timeout": "Timeout", "incomplete": "Incomplete"}[variant]
                        record["returncode"] = None
                    elif variant == "nonzero":
                        record["returncode"] = 7
                    elif variant == "bool_exit":
                        record["returncode"] = False
                    elif variant == "backwards":
                        record["ended_monotonic_ns"] = 9
                    else:
                        record["stdout"] = "rust-version.stdout"
                        expected = "command raw stream identity mismatch"
                    put(evidence / "commands.json", commands)
                    self.assert_rehashed_type_failure(package, plan, packages, family, expected)

    def assert_rehashed_type_failure(self, package, plan, packages, family, diagnostic):
        refresh(package)
        envelope = audit.read_json(package / "envelope.json")
        evidence = package / "evidence"
        audit.validate_manifest(evidence, envelope["evidence_root"],
                                audit.read_json(evidence / "artifacts.json"))
        for name, sha in envelope["driver_hashes"].items():
            self.assertEqual(audit.digest(evidence / name), sha)
        result = audit.aggregate(plan, packages, BINDING)
        row = result["families"][family]
        self.assertEqual(result["exit_status"], 1)
        self.assertEqual(row["status"], "Failure")
        self.assertEqual(row["detail"], diagnostic)
        self.assertEqual(row["comparison_evidence"], "Unverified")
        self.assertIsNone(row["numeric_reference_comparisons"])
        self.assertEqual(row["empty_contracts"], 0)

    def test_rehashed_success_exit_numeric_types_fail_semantically(self):
        for family in audit.FAMILIES:
            for replacement in (False, 0.0):
                with self.subTest(family=family, type=type(replacement).__name__), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    packages = root / "packages"
                    packages.mkdir()
                    plan = audit.make_plan(family, BINDING)
                    package = fixture(root, packages, plan, family)
                    self.assertEqual(audit.aggregate(plan, packages, BINDING)["exit_status"], 0)
                    path = package / "evidence/terminal.json"
                    terminal = audit.read_json(path)
                    self.assertEqual(terminal["exit_status"], replacement)
                    terminal["exit_status"] = replacement
                    put(path, terminal)
                    self.assert_rehashed_type_failure(package, plan, packages, family,
                                                      "unfinished/inconsistent selected driver terminal")

    def test_rehashed_failed_exit_numeric_types_preserve_failure(self):
        for classification in ("MissingFixture", "Unsupported", "Failure"):
            for replacement in (True, 1.0):
                with self.subTest(classification=classification, type=type(replacement).__name__), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    packages = root / "packages"
                    packages.mkdir()
                    plan = audit.make_plan("general", BINDING)
                    package = fixture(root, packages, plan, "general")
                    terminal_path = package / "evidence/terminal.json"
                    terminal = audit.read_json(terminal_path)
                    terminal.update(exit_status=1, status=classification)
                    put(terminal_path, terminal)
                    ledger_path = package / "evidence/family-ledger.json"
                    ledger = audit.read_json(ledger_path)
                    ledger["families"]["general"]["status"] = classification
                    put(ledger_path, ledger)
                    refresh(package)
                    envelope_path = package / "envelope.json"
                    envelope = audit.read_json(envelope_path)
                    envelope["job_status"] = "failure"
                    put(envelope_path, envelope)
                    positive = audit.aggregate(plan, packages, BINDING)
                    self.assertEqual(positive["exit_status"], 1)
                    self.assertEqual(positive["families"]["general"]["status"], classification)
                    self.assertEqual(terminal["exit_status"], replacement)
                    terminal["exit_status"] = replacement
                    put(terminal_path, terminal)
                    self.assert_rehashed_type_failure(package, plan, packages, "general",
                                                      "failed driver exit status must be an integer")

    def test_rehashed_comparison_count_numeric_types_fail_semantically(self):
        cases = [("lanczos", "numeric_reference_comparisons", 8.0),
                 ("lanczos", "empty_contracts", 4.0)]
        cases += [(family, "empty_contracts", value)
                  for family in ("general", "mpi", "thread") for value in (False, 0.0)]
        for family, field, replacement in cases:
            with self.subTest(family=family, field=field, type=type(replacement).__name__), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                packages = root / "packages"
                packages.mkdir()
                plan = audit.make_plan(family, BINDING)
                package = fixture(root, packages, plan, family)
                self.assertEqual(audit.aggregate(plan, packages, BINDING)["exit_status"], 0)
                path = package / "evidence/family-ledger.json"
                ledger = audit.read_json(path)
                self.assertEqual(ledger["families"][family][field], replacement)
                ledger["families"][family][field] = replacement
                put(path, ledger)
                diagnostic = "DC numeric/empty evidence mismatch" if family == "lanczos" else "invented uninstrumented comparison totals"
                self.assert_rehashed_type_failure(package, plan, packages, family, diagnostic)

    def test_rehashed_configuration_numeric_types_fail_semantically(self):
        for family in audit.FAMILIES:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                packages = root / "packages"
                packages.mkdir()
                plan = audit.make_plan(family, BINDING)
                package = fixture(root, packages, plan, family)
                positive = audit.aggregate(plan, packages, BINDING)
                self.assertEqual(positive["exit_status"], 0)
                self.assertEqual(positive["families"][family]["status"], "Pass")
                path = package / "evidence/metadata.json"
                original = audit.read_json(path)
                settings = original["configuration"]
                mutations = []
                for key, value in settings.items():
                    if type(value) is int:
                        mutations.append((key, None, float(value)))
                        if value in (0, 1):
                            mutations.append((key, None, bool(value)))
                    elif type(value) is list:
                        for index, item in enumerate(value):
                            if type(item) is int:
                                mutations.append((key, index, float(item)))
                                if item in (0, 1):
                                    mutations.append((key, index, bool(item)))
                self.assertTrue(mutations)
                for key, index, replacement in mutations:
                    with self.subTest(family=family, field=key, index=index,
                                      replacement_type=type(replacement).__name__):
                        metadata = copy.deepcopy(original)
                        if index is None:
                            metadata["configuration"][key] = replacement
                        else:
                            metadata["configuration"][key][index] = replacement
                        # Equal numeric values are deliberate: only type is wrong.
                        self.assertEqual(metadata["configuration"], settings)
                        put(path, metadata)
                        refresh(package)
                        result = audit.aggregate(plan, packages, BINDING)
                        row = result["families"][family]
                        self.assertEqual(result["exit_status"], 1)
                        self.assertEqual(row["status"], "Failure")
                        self.assertEqual(row["detail"],
                                         "bounded model/seed/steps/ranks/groups/workers settings mismatch")
                        self.assertEqual(row["comparison_evidence"], "Unverified")
                        self.assertIsNone(row["numeric_reference_comparisons"])
                        put(path, original)
                        refresh(package)

    def test_rehashed_offline_reference_claims_fail_semantically(self):
        cases = ("missing", "runtime", "manifest_version", "acquisition_version",
                 "duplicate", "hash", "authority", "oracle_execution",
                 "reference_version", "not_claimed")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                packages = root / "packages"
                packages.mkdir()
                plan = audit.make_plan("general", BINDING)
                package = fixture(root, packages, plan, "general")
                self.assertEqual(audit.aggregate(plan, packages, BINDING)["exit_status"], 0)
                path = package / "evidence/metadata.json"
                metadata = audit.read_json(path)
                report = metadata["offline_reference"]
                error = "reference declaration semantics/hash mismatch"
                if case == "missing":
                    del metadata["offline_reference"]
                    error = "invalid offline reference report schema"
                elif case == "runtime":
                    report["julia_runtime"]["status"] = "Pass"
                    error = "offline gate cannot claim Julia runtime execution"
                elif case == "manifest_version":
                    report["declarations"][0]["julia_version"] = "1.11.0"
                elif case == "acquisition_version":
                    report["declarations"][1]["julia_versions"] = ["1.13.1"]
                elif case == "duplicate":
                    report["declarations"].append(copy.deepcopy(report["declarations"][0]))
                    error = "missing/duplicate/off-closure reference declarations"
                elif case == "hash":
                    report["declarations"][0]["sha256"] = "d" * 64
                elif case == "authority":
                    report["authority"] = "current Julia verified"
                    error = "invalid offline reference authority"
                else:
                    metadata[case] = "current Julia verified"
                    error = "offline reference scope/claim mismatch"
                put(path, metadata)
                refresh(package)
                envelope = audit.read_json(package / "envelope.json")
                audit.validate_manifest(package / "evidence", envelope["evidence_root"],
                                        audit.read_json(package / "evidence/artifacts.json"))
                for name, sha in envelope["driver_hashes"].items():
                    self.assertEqual(audit.digest(package / "evidence" / name), sha)
                result = audit.aggregate(plan, packages, BINDING)
                row = result["families"]["general"]
                self.assertEqual(result["exit_status"], 1)
                self.assertEqual(row["detail"], error)
                self.assertEqual(row["status"], "Failure")
                self.assertEqual(row["comparison_evidence"], "Unverified")
                self.assertIsNone(row["numeric_reference_comparisons"])
                self.assertEqual(row["empty_contracts"], 0)

    def test_failed_installer_sidecar_is_retained_not_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt = root / "install-failure"
            receipt.mkdir()
            (receipt / "terminal.txt").write_text("prior=1 post=0\n")
            plan = audit.make_plan("mpi", BINDING)
            packages = root / "packages"
            packages.mkdir()
            package = packages / audit.artifact_name("mpi", plan)
            with patch.object(audit, "checkout_head", return_value=plan["head"]):
                audit.seal(plan, "mpi", root / "driver-not-run", package,
                           "failure", plan["head"], "never", receipt)
            self.assertEqual((package / "mpi-provider-install/terminal.txt").read_text(),
                             "prior=1 post=0\n")
            self.assertEqual(audit.aggregate(plan, packages, BINDING)["exit_status"], 1)

    def test_mpi_provider_rehashed_semantic_failures_cannot_pass(self):
        # Refresh artifact/envelope hashes after each deliberate mutation: the
        # actual aggregate must reject semantic defects, not merely stale SHA.
        cases = ("missing", "terminal", "startup-duplicate", "startup-world", "startup-thread",
                 "startup-native", "receipt-tamper", "runtime", "library",
                 "linkage", "binary", "tool", "header")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                packages = root / "packages"
                packages.mkdir()
                plan = audit.make_plan("mpi", BINDING)
                package = fixture(root, packages, plan, "mpi")
                evidence = package / "evidence"
                binding = audit.read_json(evidence / "mpi-provider.json")
                receipt = evidence / "mpi-provider-receipt"
                if case == "missing":
                    (evidence / "mpi-provider.json").unlink()
                elif case == "terminal":
                    (receipt / "terminal.txt").write_text("prior=1 post=0\n")
                elif case in ("startup-duplicate", "startup-world"):
                    path = receipt / "world-2.stdout"
                    path.write_text(path.read_text().replace(
                        "rank=1" if case == "startup-duplicate" else "size=2",
                        "rank=0" if case == "startup-duplicate" else "size=4"))
                elif case == "startup-native":
                    (receipt / "world-4.native.status").write_text("1\n")
                elif case == "startup-thread":
                    path = receipt / "world-2.stdout"
                    path.write_text(path.read_text().replace("required=1 provided=1", "required=0 provided=0"))
                elif case == "receipt-tamper":
                    (receipt / "ready.txt").write_text("changed\n")
                elif case == "runtime":
                    put(evidence / "mpi-provider.after.json", {"/changed": "b" * 64})
                elif case == "library":
                    binding["library"]["resolved"] = "/system/libmpi.so"
                elif case == "linkage":
                    (evidence / "mpi-linkage.txt").write_text("libmpi.so.12 => /system/libmpi.so\n")
                elif case == "binary":
                    binding["binary_sha256"] = "c" * 64
                elif case == "tool":
                    binding["tools"]["mpiexec"]["path"] = "/system/mpiexec"
                elif case == "header":
                    for suffix in ("before", "after"):
                        path = evidence / f"mpi-provider.{suffix}.json"
                        state = audit.read_json(path)
                        del state["/synthetic/mpich/include/mpi.h"]
                        put(path, state)
                if case not in ("missing", "receipt-tamper"):
                    binding["receipt_hashes"] = audit.mpi_provider.receipt_state(receipt)
                    put(evidence / "mpi-provider.json", binding)
                refresh(package)
                result = audit.aggregate(plan, packages, BINDING)
                self.assertEqual(result["exit_status"], 1)
                self.assertEqual(result["families"]["mpi"]["status"], "Failure")

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

    def test_rehashed_lanczos_metadata_semantic_rejection_not_hash_failure(self):
        variants = [("missing_report", None), ("override_bool", None),
                    ("unbound_hash", None)]
        variants += [(kind, field) for field in audit.fixture_metadata.FIELDS
                     for kind in ("missing_field", "mismatch")]
        for kind, field in variants:
            with self.subTest(kind=kind, field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                packages = root / "packages"
                packages.mkdir()
                plan = audit.make_plan("lanczos", BINDING)
                package = fixture(root, packages, plan, "lanczos")
                positive = audit.aggregate(plan, packages, BINDING)
                self.assertEqual(positive["exit_status"], 0)
                self.assertEqual(positive["families"]["lanczos"]["status"], "Pass")
                evidence = package / "evidence"
                path = evidence / "metadata.json"
                metadata = audit.read_json(path)
                report = metadata["fixture_modpara"]
                expected_error = "fixture declarations mismatch reviewed gate report"
                if kind == "missing_report":
                    metadata.pop("fixture_modpara")
                    expected_error = "invalid fixture report schema"
                elif kind == "override_bool":
                    report["explicit_gate_overrides"]["seed"] = True
                    expected_error = "wrong explicit gate overrides"
                elif kind == "unbound_hash":
                    report["fixtures"][0]["sha256"] = "d" * 64
                    expected_error = "fixture report hash not bound"
                elif kind == "missing_field":
                    report["fixtures"][0]["declared"].pop(field)
                else:
                    report["fixtures"][0]["declared"][field] += 1
                put(path, metadata)
                refresh(package)
                # Prove legitimate artifact/envelope rehashing succeeded before
                # asking the public aggregate to reject metadata semantics.
                envelope = audit.read_json(package / "envelope.json")
                for name, sha in envelope["driver_hashes"].items():
                    self.assertEqual(audit.digest(evidence / name), sha)
                audit.validate_manifest(evidence, envelope["evidence_root"],
                                        audit.read_json(evidence / "artifacts.json"))
                result = audit.aggregate(plan, packages, BINDING)
                self.assertEqual(result["exit_status"], 1)
                row = result["families"]["lanczos"]
                self.assertTrue(row["selected"])
                self.assertEqual(row["status"], "Failure")
                self.assertEqual(row["comparison_evidence"], "Unverified")
                self.assertIsNone(row["numeric_reference_comparisons"])
                self.assertEqual(row["empty_contracts"], 0)
                self.assertEqual(row["detail"], expected_error)
                for family in ("general", "mpi", "thread"):
                    self.assertEqual(result["families"][family],
                                     {"selected": False, "status": "NotRun"})

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

#!/usr/bin/env python3
"""Explicit Linux-only offline-reference gates; never called by Cargo/normal CI."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import optional_mpi_provider_183 as mpi_provider
import optional_fixture_metadata_183 as fixture_metadata
import optional_reference_provenance_183 as reference_provenance

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("general", "lanczos", "mpi", "thread")
GENERAL = (
    "corrected_general_all_four_prefixes_match_independent_reference",
    "corrected_general_twenty_step_public_runner_is_repeatable",
)
LANCZOS = "serial_lanczos_matches_hubbard_and_exchange_references"
MODELS = ("hubbard_chain_real", "hubbard_chain_lanczos", "spin_chain_lanczos")
THREAD = "runner_workers_preserve_rng_configurations_direct_store_cg_and_physcal"
MPI = "mpi_physcal_reduces_fixed_parameter_samples"


class MissingFixtureError(ValueError):
    """A required independent input/reference is absent."""


class UnsupportedError(ValueError):
    """The explicit requested input or platform is unsupported."""


def failure_status(error):
    if isinstance(error, MissingFixtureError):
        return "MissingFixture"
    if isinstance(error, UnsupportedError):
        return "Unsupported"
    return "Failure"


def family_ledger(selected, excluded=()):
    if selected not in FAMILIES or any(name not in FAMILIES for name in excluded):
        raise UnsupportedError("unknown or empty family selection/exclusion")
    if len(set(excluded)) != len(excluded) or selected in excluded:
        raise UnsupportedError("duplicate exclusion or selected family excluded")
    return {name: {"selected": name == selected,
                   "status": "ExplicitSkip" if name in excluded else "NotRun",
                   "started_driver_invocations": 0, "completed_selection_identities": [],
                   "selected_test_identities": [], "numeric_reference_comparisons": None,
                   "empty_contracts": 0, "comparison_evidence": "Unverified", "helper_tests": 0}
            for name in FAMILIES}


def validate_completion(family, completed, before, after):
    expected = {"general": list(GENERAL),
                "lanczos": [f"{model}-{mode}" for model in MODELS for mode in ("real", "cmp")],
                "mpi": ["world2", "world4"], "thread": [THREAD]}[family]
    if sorted(completed) != sorted(expected):
        raise ValueError("incomplete/duplicate/wrong bounded execution identities")
    if not before or before != after:
        raise ValueError("empty or changed source/fixture/binary closure")


def selected_failure_status(family, identities, diagnostic):
    """Require actual selected-suite failure layout AND its gate status marker."""
    target = {"general": "ctest_general_reference", "lanczos": "lanczos_transfer_physcal",
              "thread": "threaded_issue182"}.get(family)
    for identity in identities:
        if family == "mpi":
            pattern = rf"^\s*test {re.escape(identity)} \.\.\. FAILED\s*$"
        else:
            pattern = (rf"^\s*FAIL \[\s*\d+(?:\.\d+)?s\] \(\d+/\d+\) "
                       rf"mvmc-core::{target} {re.escape(identity)}\s*$")
        if re.search(pattern, diagnostic, re.MULTILINE):
            gate_name = {"general": "ctest-general", "lanczos": "lanczos-physcal",
                         "mpi": "mpi-physcal", "thread": "threaded-issue182"}[family]
            for category in ("MissingFixture", "Unsupported"):
                if re.search(rf"^\s*parity gate {gate_name}: {category}:", diagnostic, re.MULTILINE):
                    return category
    return "Failure"


def invalidate_comparison_counts(row):
    row.update(numeric_reference_comparisons=None, empty_contracts=0,
               comparison_evidence="Unverified")


def selected_binary(listing, expected):
    """Reject helpers, omitted names, duplicate suites, nonignored and zero selection."""
    matches = []
    binaries = set()
    for suite in listing["rust-suites"].values():
        for name, case in suite["testcases"].items():
            if case["filter-match"]["status"] == "matches":
                if not case["ignored"]:
                    raise ValueError("selected nonignored/helper test")
                matches.append(name)
                binaries.add(suite["binary-path"])
    if sorted(matches) != sorted(expected) or len(binaries) != 1:
        raise ValueError(f"selection differs from exact declared tests: {matches}")
    return Path(binaries.pop())


def cargo_binary(records, target):
    matches = [r["executable"] for r in records if r.get("reason") == "compiler-artifact"
               and r["target"]["name"] == target and r.get("executable")]
    if len(matches) != 1:
        raise ValueError("Cargo must emit exactly one target executable")
    return Path(matches[0])


def validate_namelist(path):
    text = path.read_text()
    references = []
    for line in text.splitlines():
        fields = line.split("#", 1)[0].split()
        if not fields:
            continue
        if len(fields) != 2 or fields[0].lower() == "interall":
            raise UnsupportedError(f"unsupported/malformed/InterAll namelist: {path}")
        if not (path.parent / fields[1]).is_file():
            raise MissingFixtureError(f"missing referenced input: {path.parent / fields[1]}")
        references.append(path.parent / fields[1])
    return references


def digest_files(paths):
    result = {}
    for path in sorted(set(paths)):
        if not path.is_file():
            raise ValueError(f"missing artifact/input: {path}")
        result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not result:
        raise ValueError("empty file closure")
    return result


def source_files():
    raw = subprocess.check_output(
        ["git", "ls-files", "-co", "--exclude-standard", "-z", "--", "Cargo.toml",
         "Cargo.lock", "rust-toolchain.toml", "rustfmt.toml", ".cargo", "tests/support",
         "crates", "xtask", "scripts", ".github", "third_party"],
        cwd=ROOT,
    )
    return [ROOT / os.fsdecode(x) for x in raw.split(b"\0")
            if x and "__pycache__" not in Path(os.fsdecode(x)).parts]


def fixture_files(family):
    if family == "general":
        directories = [ROOT / "tests/fixtures/ctest_general_pr54_3d0fd263"]
    elif family == "lanczos":
        directories = [ROOT / f"extern/Julia-mVMC/test/integration/reference/{m}/physcal_ref"
                       for m in MODELS]
    elif family == "mpi":
        directories = [ROOT / "tests/fixtures/physcal_181/heisenberg_chain_real"]
    else:
        directories = [ROOT / "tests/fixtures/ctest_model_prefixes",
                       ROOT / "tests/fixtures/reviewed_cg_62b/canonical_general_rbm"]
        for model in (*MODELS[:1], "heisenberg_chain_real", "heisenberg_chain_cmp",
                      "heisenberg_chain_fsz", "general_rbm_cmp"):
            directories.append(ROOT / f"extern/Julia-mVMC/test/integration/reference/{model}")
    files = []
    manifest = ROOT / "extern/Julia-mVMC/Manifest-v1.13.toml"
    if not manifest.is_file():
        raise MissingFixtureError("missing pinned reference Manifest-v1.13.toml")
    files.append(manifest)
    for directory in directories:
        if not directory.is_dir():
            raise MissingFixtureError(f"missing fixture directory: {directory}")
        contents = list(directory.rglob("*"))
        files.extend(p for p in contents if p.is_file())
        for path in contents:
            if path.name == "namelist.def" and path.is_file():
                files.extend(validate_namelist(path))
    return files


def backend(binary, linkage):
    """Query the actually linked Linux OpenBLAS library, not env/pkg-config alone."""
    output = subprocess.check_output(["ldd", str(binary)], text=True)
    linkage.write_text(output)
    paths = {line.split("=>", 1)[1].split()[0] for line in output.splitlines()
             if "openblas" in line.lower() and "=>" in line and "not found" not in line}
    if len(paths) != 1:
        raise ValueError("cannot identify exactly one linked OpenBLAS backend")
    path = paths.pop()
    library = ctypes.CDLL(path)
    def call(names, restype):
        for name in names:
            function = getattr(library, name, None)
            if function is not None:
                function.argtypes = []
                function.restype = restype
                return function()
        raise ValueError("linked backend lacks runtime version/configuration API")
    raw_config = call(("openblas_get_config", "openblas_get_config64_"), ctypes.c_char_p)
    raw_core = call(("openblas_get_corename", "openblas_get_corename64_"), ctypes.c_char_p)
    if not isinstance(raw_config, bytes) or not isinstance(raw_core, bytes):
        raise ValueError("backend configuration/core API returned no string")
    config, core = raw_config.decode(), raw_core.decode()
    threads = call(("openblas_get_num_threads", "openblas_get_num_threads64_"), ctypes.c_int)
    if not re.match(r"^OpenBLAS \d+\.\d+\.\d+(?:\s|$)", config) or not core.strip() or threads != 1:
        raise ValueError("actual backend version or single-thread configuration unverified")
    return {"library": path, "library_sha256": digest_files([Path(path)])[path],
            "actual_config": config, "actual_core": core, "actual_threads": threads}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def execute_command(args, name, output, commands, env, cwd, selected=None, timeout=2400):
    """Record actual process lifecycle; selected context is a request, not a model verdict."""
    record = {"schema": 1, "name": name, "argv": list(map(str, args)), "cwd": str(cwd),
              "env": {k: v for k, v in env.items() if k.startswith(("MVMC_", "MPI179_"))
                      or k in ("CARGO_TARGET_DIR", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                               "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "TMPDIR")},
              "selected": selected, "timeout_seconds": timeout, "state": "Started",
              "started_unix_ns": time.time_ns(), "started_monotonic_ns": time.monotonic_ns(),
              "ended_unix_ns": None, "ended_monotonic_ns": None, "returncode": None,
              "exception": None, "stdout": f"{name}.stdout", "stderr": f"{name}.stderr"}
    commands.append(record)
    write_json(output / "commands.json", commands)
    try:
        with (output / record["stdout"]).open("w") as stdout, (output / record["stderr"]).open("w") as stderr:
            result = subprocess.run(record["argv"], cwd=cwd, env=env, stdout=stdout,
                                    stderr=stderr, timeout=timeout)
        record.update(state="Completed", returncode=result.returncode)
        return result
    except subprocess.TimeoutExpired as error:
        record.update(state="Timeout", exception=type(error).__name__)
        raise
    except BaseException as error:
        record.update(state="Incomplete", exception=type(error).__name__)
        raise
    finally:
        record.update(ended_unix_ns=time.time_ns(), ended_monotonic_ns=time.monotonic_ns())
        write_json(output / "commands.json", commands)


def validate_artifacts(root, required):
    for name in required:
        path = root / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing/empty required artifact: {name}")


def verify_artifact_hashes(root, manifest):
    if not manifest:
        raise ValueError("empty artifact hash manifest")
    for name, expected in manifest.items():
        path = Path(name)
        if not path.is_absolute() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("artifact outside exclusive evidence root")
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("missing or changed hashed artifact")


def validate_mpi_run(text, ranks):
    summaries = re.findall(r"test result: ok\. 1 passed; 0 failed; 0 ignored;", text)
    markers = re.findall(r"OPTIONAL183_MPI rank=(\d+) world=(\d+) width=(\d+) group_size=(\d+)", text)
    expected = {(rank, ranks, width, ranks if width == 1 else 2)
                for rank in range(ranks) for width in (1, 2)}
    actual = [tuple(map(int, marker)) for marker in markers]
    if len(summaries) != ranks or len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("missing/duplicate/wrong actual rank/world/group or per-rank PASS summaries")


def validate_dc_run(text, model, mode):
    """Only the exact selected gate emits these records; helpers are not selected."""
    if model not in MODELS or mode not in ("real", "cmp"):
        raise ValueError("unsupported Lanczos DC case")
    lines = [line for line in text.splitlines() if "OPTIONAL183_DC" in line]
    records = []
    for line in lines:
        match = re.fullmatch(r"OPTIONAL183_DC model=(\S+) mode=(\S+) file=(\S+) status=(\S+)", line.strip())
        if match is None:
            raise ValueError("malformed DC marker")
        records.append(match.groups())
    expected = [] if model == "hubbard_chain_real" else [
        (model, mode, "zvo_ls_cisajs_001.dat", "REFERENCE_COMPARED"),
        (model, mode, "zvo_ls_cisajscktalt_001.dat", "REFERENCE_COMPARED"),
        (model, mode, "zvo_ls_cisajscktaltex_001.dat", "EMPTY_CONTRACT"),
    ]
    if sorted(records) != sorted(expected):
        raise ValueError("missing/duplicate/wrong DC case identity or comparison status")
    return [{"model": m, "mode": c, "file": f, "status": s} for m, c, f, s in records]


def run(family, output, excluded=()):
    ledger = family_ledger(family, excluded)
    output.mkdir(parents=False, exist_ok=False)
    write_json(output / "family-ledger.json", {"scope": "driver invocation only; NOT workflow aggregate", "families": ledger})
    env = os.environ.copy()
    env.update({key: "1" for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                                    "MKL_NUM_THREADS", "BLIS_NUM_THREADS")})
    env.update({"JULIA_MVMC_ROOT": str(ROOT / "extern/Julia-mVMC"),
                "MVMC_RS_THREADED_FIXTURE_ROOT": str(ROOT),
                "MVMC_RS_INNER_THREADS": "1", "MVMC_RS_INNER_THRESHOLD": "32"})
    for key in ("MVMC_RS_THREADED_ARCHIVE", "MVMC_RS_THREADED_FILTER"):
        env.pop(key, None)
    # The read-only ctypes backend observer runs in this driver, not a child;
    # initialize its library with the same requested thread settings as gates.
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[key] = env[key]
    # A dedicated CI target is mandatory; never silently reuse another checkout.
    commands = []
    completed = []
    required = ["metadata.json", "commands.json", "source.before.json", "source.after.json",
                "fixtures.before.json", "fixtures.after.json", "backend.json", "terminal.json", "family-ledger.json"]
    status = 1
    outcome = "Failure"
    def execute(args, name, extra=None, selected_gate=False):
        call_env = env | (extra or {})
        selected = None
        if selected_gate:
            selected = {"family": family, "profile": "test-fast",
                        "features": "mpi" if family == "mpi" else "default",
                        "identities": list(ledger[family]["selected_test_identities"])}
            ledger[family]["started_driver_invocations"] += 1
        result = execute_command(args, name, output, commands, call_env, ROOT, selected)
        if result.returncode:
            diagnostic = (output / f"{name}.stderr").read_text() + (output / f"{name}.stdout").read_text()
            if selected_gate:
                category = selected_failure_status(family, ledger[family]["selected_test_identities"], diagnostic)
                if category == "MissingFixture":
                    raise MissingFixtureError(f"{name} selected gate reported {category}")
                if category == "Unsupported":
                    raise UnsupportedError(f"{name} selected gate reported {category}")
            raise RuntimeError(f"{name} failed with exit {result.returncode}")
        return (output / f"{name}.stdout").read_text()
    try:
        if platform.system() != "Linux":
            raise UnsupportedError("bounded dispatch supports Linux only")
        if not env.get("CARGO_TARGET_DIR"):
            raise ValueError("explicit checkout-specific CARGO_TARGET_DIR required")
        metadata = {
            "family": family, "scope": {
                "general": "one General model, independent prefixes1/2/3/20 plus public20 repeat",
                "lanczos": "three historical models x real/cmp; eight DC references and four empty GEx contracts (not numeric comparisons)",
                "mpi": "one fixed Heisenberg real PhysCal, actual worlds2/4, groups1/2 and Lanczos rejection",
                "thread": "ONE primary runner test; workers1/2/4, steps2/samples200; NOT long45",
            }[family], "oracle_execution": "none; checked-in fixtures only",
            "profile": "test-fast", "features": "mpi" if family == "mpi" else "default",
            "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "command_receipt_schema": 1,
            "checkout_root": str(ROOT),
            "platform": platform.platform(), "started_unix": time.time(),
            "reference_version": "per-fixture provenance, NOT current Julia runtime verification",
            "not_claimed": "full13 matrix, fullJulia features, fullC sampler, InterAll",
            "compiler_environment": {k: v for k, v in env.items() if k in (
                "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER",
                "RUSTC", "CARGO_BUILD_TARGET", "CARGO_TARGET_DIR", "KACHE_SERVER",
                "CC", "CFLAGS", "FC", "FFLAGS", "MPICC", "LIBCLANG_PATH")
                or k.startswith("CARGO_PROFILE_")},
            "configuration": {
                "general": {"model": "general_rbm_cmp", "seed": 12395,
                            "prefixes_and_windows": [1, 2, 3, 20], "NSRCG": 0, "NStore": 1,
                            "ranks": 1, "workers": 1, "threshold": 32},
                "lanczos": {"models": MODELS, "modes": ["real", "cmp"], "seed_override": 1,
                            "ranks": 1, "groups": 1, "steps": "PhysCal fixture quantity count",
                            "sample_counts": [100, 1000, 5000], "workers": 1},
                "mpi": {"model": "heisenberg_chain_real", "ranks": [2, 4],
                        "group_widths": [1, 2], "repeats": 2,
                        "seed": 1, "steps": 1, "samples": 3, "warmup": 1, "workers": 1},
                "thread": {"workers": [1, 2, 4], "threshold": 32, "sizes": [31, 32, 33],
                           "steps": 2, "samples": 200, "ranks": 1,
                           "seed": 1},
            }[family],
        }
        write_json(output / "metadata.json", metadata)
        execute(["rustc", "-Vv"], "rust-version")
        execute(["cargo", "nextest", "--version"], "nextest-version")
        execute(["git", "submodule", "status", "--recursive"], "reference-revisions")
        execute(["git", "status", "--short"], "working-tree-status")
        source = digest_files(source_files())
        fixtures = digest_files(fixture_files(family))
        metadata["offline_reference"] = reference_provenance.capture(
            [Path(path) for path in fixtures])
        reference_provenance.validate(metadata["offline_reference"], fixtures)
        write_json(output / "metadata.json", metadata)
        if family == "lanczos":
            metadata["fixture_modpara"] = fixture_metadata.capture(ROOT)
            fixture_metadata.validate(metadata["fixture_modpara"], fixtures)
            write_json(output / "metadata.json", metadata)
        write_json(output / "source.before.json", source)
        write_json(output / "fixtures.before.json", fixtures)
        if family == "mpi":
            # Installer/startup receipts and live prefix must be valid before Cargo.
            mpi_binding = mpi_provider.capture(output, env)
            text = execute(["cargo", "test", "--locked", "-p", "mvmc-core", "--profile",
                            "test-fast", "--features", "mpi", "--test", "mpi_physcal",
                            "--no-run", "--message-format=json"], "build")
            binary = cargo_binary([json.loads(line) for line in text.splitlines()], "mpi_physcal")
            listing = execute([binary, "--list"], "selection")
            if listing.splitlines().count(f"{MPI}: test") != 1:
                raise ValueError("exact MPI gate absent/duplicated")
            write_json(output / "selection.json", {"selected": [MPI], "count": 1})
            ledger[family]["selected_test_identities"] = [MPI]
            write_json(output / "backend.json", backend(binary, output / "linkage.txt"))
            write_json(output / "binary.before.json", digest_files([binary]))
            execute(["mpiexec", "--version"], "mpi-version")
            for ranks in (2, 4):
                text = execute(["mpiexec", "-n", ranks, binary, "--ignored", "--exact", MPI,
                         "--nocapture"], f"mpi-{ranks}", {
                             "MVMC_RS_MPI_PHYSICAL": "1",
                             "MPI179_PHYSCAL_OUTPUT": str(output / f"world-{ranks}")}, selected_gate=True)
                text += (output / f"mpi-{ranks}.stderr").read_text()
                validate_mpi_run(text, ranks)
                completed.append(f"world{ranks}")
            mpi_provider.finish(output, mpi_binding, binary)
            required.extend(["mpi-provider.json", "mpi-provider.before.json",
                             "mpi-provider.after.json", "mpi-linkage.txt"])
        else:
            target, names = {"general": ("ctest_general_reference", GENERAL),
                             "lanczos": ("lanczos_transfer_physcal", (LANCZOS,)),
                             "thread": ("threaded_issue182", (THREAD,))}[family]
            filter_expr = " | ".join(f"test(={name})" for name in names)
            common = ["--locked", "-p", "mvmc-core", "--cargo-profile", "test-fast",
                      "--test", target, "--run-ignored", "only", "-E", filter_expr]
            text = execute(["cargo", "nextest", "list", *common, "--message-format", "json"], "selection")
            binary = selected_binary(json.loads(text), names)
            ledger[family]["selected_test_identities"] = list(names)
            write_json(output / "selection.json", json.loads(text))
            write_json(output / "backend.json", backend(binary, output / "linkage.txt"))
            write_json(output / "binary.before.json", digest_files([binary]))
            gate = ["cargo", "nextest", "run", *common, "--no-fail-fast", "--retries", "0",
                    "--success-output", "immediate", "--failure-output", "immediate"]
            if family == "general":
                execute(gate, "general", {"MVMC_RS_CTEST_GENERAL": "1"}, selected_gate=True)
                completed.extend(names)
            elif family == "lanczos":
                dc_records = []
                for model in MODELS:
                    for mode in ("real", "cmp"):
                        text = execute(gate, f"{model}-{mode}", {"MVMC_RS_LANCZOS_PHYSICAL": "1",
                                "MVMC_RS_LANCZOS_MODEL": model, "MVMC_RS_LANCZOS_MODE": mode}, selected_gate=True)
                        # Nextest's successful test output is recorded on stderr.
                        # Read only the matching selected call; do not scan helper logs.
                        text += "\n" + (output / f"{model}-{mode}.stderr").read_text()
                        dc_records.extend(validate_dc_run(text, model, mode))
                        completed.append(f"{model}-{mode}")
                if len(dc_records) != 12:
                    raise ValueError("incomplete DC accounting")
                write_json(output / "dc-comparisons.json", {
                    "records": dc_records, "reference_comparisons": 8,
                    "empty_contracts": 4, "empty_contracts_are_numeric_comparisons": False,
                })
                required.append("dc-comparisons.json")
            else:
                nested = output / "thread-artifacts"
                nested.mkdir()
                text = execute(["bash", "scripts/verify_threaded_issue182.sh"], "thread", {
                    "TMPDIR": str(nested), "MVMC_RS_THREADED_TARGET_DIR": env["CARGO_TARGET_DIR"]}, selected_gate=True)
                if "Passed: complete Rust-only matrix and actual worker observations" not in text:
                    raise ValueError("thread wrapper completion evidence absent")
                wrappers = [p for p in nested.iterdir() if p.is_dir() and (p / "tests.tar.zst").is_file()]
                if len(wrappers) != 1:
                    raise ValueError("exactly one executed thread archive artifact required")
                wrapper = wrappers[0]
                terminal = (wrapper / "terminal.txt").read_text().splitlines()
                if terminal.count("exit_status=0") != 1:
                    raise ValueError("nested thread wrapper terminal is not successful")
                # Query the exact immutable archive used by the wrapper, not a
                # separately compiled initial binary. Extraction never runs tests.
                extracted = output / "executed-thread-archive"
                extracted.mkdir()
                archive_listing = execute(["cargo", "nextest", "list", "--archive-file",
                    wrapper / "tests.tar.zst", "--extract-to", extracted,
                    "--run-ignored", "only", "-E", filter_expr, "--message-format", "json"],
                    "executed-thread-selection")
                binary = selected_binary(json.loads(archive_listing), names)
                actual_hash = next(iter(digest_files([binary]).values()))
                initial_hash = next(iter(json.loads((output / "binary.before.json").read_text()).values()))
                if actual_hash != initial_hash:
                    raise ValueError("wrapper executed archive differs from source-associated initial build")
                write_json(output / "thread-executed-archive.json", {
                    "archive": digest_files([wrapper / "tests.tar.zst"]),
                    "binary": digest_files([binary]), "wrapper": str(wrapper),
                    "selected": names, "source_association": "source stable; archive binary hash equals initial build"})
                write_json(output / "backend.json", backend(binary, output / "executed-thread-linkage.txt"))
                write_json(output / "binary.before.json", digest_files([binary]))
                required.extend(["thread-executed-archive.json", "executed-thread-linkage.txt"])
                completed.append(THREAD)
        write_json(output / "binary.json", digest_files([binary]))
        if json.loads((output / "binary.before.json").read_text()) != digest_files([binary]):
            raise ValueError("selected executable changed during validation")
        required.extend(["selection.json", "binary.json", "linkage.txt"])
        source_after = digest_files(source_files())
        fixture_after = digest_files(fixture_files(family))
        write_json(output / "source.after.json", source_after)
        write_json(output / "fixtures.after.json", fixture_after)
        if source != source_after or fixtures != fixture_after:
            raise ValueError("source or fixture closure changed during validation")
        validate_completion(family, completed, source, source_after)
        validate_completion(family, completed, fixtures, fixture_after)
        status = 0
    except Exception as error:
        outcome = failure_status(error)
        (output / "failure.txt").write_text(f"{type(error).__name__}: {error}\n")
        raise
    finally:
        write_json(output / "terminal.json", {"exit_status": status, "completed": completed,
                                               "status": "Pass" if status == 0 else outcome,
                                               "finished_unix": time.time()})
        ledger[family]["completed_selection_identities"] = list(completed)
        ledger[family]["status"] = outcome
        if status == 0:
            try:
                validate_artifacts(output, required)
                ledger[family]["status"] = "Pass"
                if family == "lanczos":
                    ledger[family]["numeric_reference_comparisons"] = 8
                    ledger[family]["empty_contracts"] = 4
                    ledger[family]["comparison_evidence"] = "Verified"
                write_json(output / "family-ledger.json", {"scope": "driver invocation only; NOT workflow aggregate", "families": ledger})
                hashes = digest_files(p for p in output.rglob("*") if p.is_file())
                verify_artifact_hashes(output, hashes)
                write_json(output / "artifacts.json", hashes)
            except Exception:
                ledger[family]["status"] = "Failure"
                invalidate_comparison_counts(ledger[family])
                write_json(output / "terminal.json", {"exit_status": 1, "completed": completed,
                                                       "status": "Failure",
                                                       "artifact_validation": "failed"})
                write_json(output / "family-ledger.json", {"scope": "driver invocation only; NOT workflow aggregate", "families": ledger})
                raise
        else:
            write_json(output / "family-ledger.json", {"scope": "driver invocation only; NOT workflow aggregate", "families": ledger})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=FAMILIES)
    parser.add_argument("output", type=Path, help="exclusive NEW artifact directory")
    parser.add_argument("--exclude", action="append", default=[], choices=FAMILIES,
                        help="record another family as ExplicitSkip; does not run or change workflow jobs")
    args = parser.parse_args()
    try:
        run(args.family, args.output.resolve(), args.exclude)
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f"optional gate failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

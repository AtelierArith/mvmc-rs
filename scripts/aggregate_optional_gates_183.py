#!/usr/bin/env python3
"""Dispatch-only evidence packaging/aggregation; no numerical or oracle execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import optional_mpi_provider_183 as mpi_provider
import optional_fixture_metadata_183 as fixture_metadata
import optional_reference_provenance_183 as reference_provenance

from run_optional_gates_183 import FAMILIES, GENERAL, LANCZOS, MODELS, MPI, THREAD, validate_general_settings

SCHEMA = 1
BINDING = ("head", "run_id", "attempt", "workflow")
STATUSES = ("Pass", "MissingFixture", "Unsupported", "Failure")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def regular(path):
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing/nonregular evidence: {path}")
    return path


def read_json(path):
    value = json.loads(regular(path).read_text(), object_pairs_hook=unique_object)
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def digest(path):
    return hashlib.sha256(regular(path).read_bytes()).hexdigest()


def write_new(path, value):
    with path.open("x") as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write("\n")


def identity(head, run_id, attempt, workflow):
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise ValueError("invalid full head SHA")
    if not re.fullmatch(r"[1-9][0-9]*", run_id) or not re.fullmatch(r"[1-9][0-9]*", attempt):
        raise ValueError("invalid run ID/attempt")
    if not workflow or any(char in workflow for char in "\r\n\0"):
        raise ValueError("missing/invalid workflow identity")
    return dict(head=head, run_id=run_id, attempt=attempt, workflow=workflow)


def make_plan(scope, binding):
    if scope not in (*FAMILIES, "all"):
        raise ValueError("unknown/empty dispatch selection")
    selected = list(FAMILIES) if scope == "all" else [scope]
    return {"schema": SCHEMA, **binding, "selected": selected,
            "families": {name: {"selected": name in selected, "status": "NotRun"}
                         for name in FAMILIES},
            "scope": "four bounded families only; NOT full issue183 coverage"}


def validate_plan(plan, expected):
    if type(plan.get("schema")) is not int or plan["schema"] != SCHEMA or any(plan.get(key) != expected[key] for key in BINDING):
        raise ValueError("plan schema/head/run/attempt/workflow mismatch")
    selected = plan.get("selected")
    if not isinstance(selected, list) or not selected or len(set(selected)) != len(selected):
        raise ValueError("empty/duplicate selected families")
    if any(name not in FAMILIES for name in selected) or set(plan.get("families", {})) != set(FAMILIES):
        raise ValueError("unknown/missing family")
    for name in FAMILIES:
        if type(plan["families"][name].get("selected")) is not bool or plan["families"][name] != {"selected": name in selected, "status": "NotRun"}:
            raise ValueError("plan falsely reports execution/skip")


def artifact_name(family, plan):
    return f"optional-family-{family}-{plan['run_id']}-{plan['attempt']}"


def safe_tree(root):
    if root.is_symlink() or not root.is_dir():
        raise ValueError("missing/unsafe evidence directory")
    for path in root.rglob("*"):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError("symlink/nonregular evidence tree")


def checkout_head():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip()


def seal(plan, family, evidence, output, job_status, head, color, mpi_receipt=None):
    actual_head = checkout_head()
    if family not in plan["selected"] or head != plan["head"] or actual_head != head:
        raise ValueError("job not selected or checkout head mismatch")
    if job_status not in ("success", "failure", "cancelled", "skipped") or color != "never":
        raise ValueError("invalid job status or gate color setting")
    output.mkdir(parents=False, exist_ok=False)
    if family == "mpi" and mpi_receipt is not None and mpi_receipt.exists():
        # Preserve failed startup/install receipts even if the driver never ran.
        # This sidecar is not successful execution evidence.
        safe_tree(mpi_receipt)
        shutil.copytree(mpi_receipt, output / "mpi-provider-install")
    hashes = {}
    if evidence.exists():
        safe_tree(evidence)
        shutil.copytree(evidence, output / "evidence")
        for name in ("terminal.json", "family-ledger.json", "artifacts.json"):
            path = output / "evidence" / name
            if path.exists():
                hashes[name] = digest(path)
    envelope = {"schema": SCHEMA, **{key: plan[key] for key in BINDING},
                "family": family, "job_status": job_status, "checkout_head": actual_head,
                "evidence_root": str(evidence.resolve()), "driver_hashes": hashes,
                "gate_environment": {"CARGO_TERM_COLOR": color},
                "scope": "job context envelope; not numerical coverage"}
    write_new(output / "envelope.json", envelope)


def validate_manifest(evidence, original_root, manifest):
    safe_tree(evidence)
    origin = PurePosixPath(original_root)
    if not origin.is_absolute() or ".." in origin.parts or not manifest:
        raise ValueError("invalid original evidence root/hash manifest")
    actual = {path.relative_to(evidence).as_posix() for path in evidence.rglob("*")
              if path.is_file() and path != evidence / "artifacts.json"}
    listed = set()
    for name, expected in manifest.items():
        path = PurePosixPath(name)
        if not path.is_absolute() or ".." in path.parts or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("unsafe manifest path/hash")
        relative = path.relative_to(origin).as_posix()
        if relative in listed or relative == "artifacts.json":
            raise ValueError("duplicate/self-referential artifact")
        listed.add(relative)
        if digest(evidence / relative) != expected:
            raise ValueError("changed artifact hash")
    if actual != listed:
        raise ValueError("missing/unlisted artifact closure")


def stable_closure(evidence, stem):
    before = read_json(evidence / f"{stem}.before.json")
    after_name = "binary.json" if stem == "binary" else f"{stem}.after.json"
    if not before or before != read_json(evidence / after_name):
        raise ValueError(f"empty/changed {stem} closure")
    if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
           for value in before.values()):
        raise ValueError("invalid source/fixture/binary digest")


def exact_setting(actual, expected):
    """JSON report settings must preserve declared types, not numeric coercions."""
    if type(actual) is not type(expected):
        return False
    if type(expected) is list:
        return len(actual) == len(expected) and all(
            exact_setting(item, value) for item, value in zip(actual, expected))
    return actual == expected


def validate_command_receipts(evidence, family, metadata, identities):
    if type(metadata.get("command_receipt_schema")) is not int or metadata["command_receipt_schema"] != 1:
        raise ValueError("historical command receipts lack execution schema; not current-eligible")
    commands = json.loads(regular(evidence / "commands.json").read_text(), object_pairs_hook=unique_object)
    if not isinstance(commands, list) or not commands:
        raise ValueError("actual command receipt list missing")
    checkout = metadata.get("checkout_root")
    if not isinstance(checkout, str) or not Path(checkout).is_absolute():
        raise ValueError("actual command checkout root missing")
    names = set()
    selected = []
    fields = {"schema", "name", "argv", "cwd", "env", "selected", "timeout_seconds", "state",
              "started_unix_ns", "started_monotonic_ns", "ended_unix_ns", "ended_monotonic_ns",
              "returncode", "exception", "stdout", "stderr"}
    for record in commands:
        if (not isinstance(record, dict) or set(record) != fields or
                type(record["schema"]) is not int or record["schema"] != 1 or
                not isinstance(record["name"], str) or
                not re.fullmatch(r"[A-Za-z0-9_-]+", record["name"]) or record["name"] in names):
            raise ValueError("invalid/duplicate command receipt identity")
        names.add(record["name"])
        if (not isinstance(record["argv"], list) or not record["argv"] or
                any(not isinstance(arg, str) or not arg for arg in record["argv"]) or
                record["cwd"] != checkout or
                not isinstance(record["env"], dict) or
                any(not isinstance(k, str) or not isinstance(v, str) for k, v in record["env"].items())):
            raise ValueError("invalid command argv/cwd/environment")
        if (record["state"] != "Completed" or type(record["returncode"]) is not int or
                record["returncode"] != 0 or record["exception"] is not None or
                type(record["timeout_seconds"]) is not int or record["timeout_seconds"] != 2400 or
                any(type(record[key]) is not int or record[key] < 0 for key in
                    ("started_unix_ns", "ended_unix_ns", "started_monotonic_ns", "ended_monotonic_ns")) or
                record["ended_monotonic_ns"] < record["started_monotonic_ns"]):
            raise ValueError("unfinished/failed/invalid command lifecycle")
        for stream in ("stdout", "stderr"):
            if record[stream] != f"{record['name']}.{stream}":
                raise ValueError("command raw stream identity mismatch")
            regular(evidence / record[stream])  # Empty output is legitimate; do not invent it.
        if record["selected"] is not None:
            context = {"family": family, "profile": metadata["profile"],
                       "features": metadata["features"], "identities": identities}
            if record["selected"] != context:
                raise ValueError("command selection/profile/features mismatch")
            selected.append(record)
    call_names = {"general": ["general"], "lanczos": [f"{m}-{c}" for m in MODELS for c in ("real", "cmp")],
                  "mpi": ["mpi-2", "mpi-4"], "thread": ["thread"]}[family]
    if [record["name"] for record in selected] != call_names:
        raise ValueError("selected command coverage/order mismatch; discovery is not completion")
    binary = read_json(evidence / "binary.before.json")
    if len(binary) != 1:
        raise ValueError("selected executable binding is not unique")
    target = {"general": "ctest_general_reference", "lanczos": "lanczos_transfer_physcal"}.get(family)
    for record in selected:
        env = record["env"]
        if family in ("general", "lanczos"):
            expr = " | ".join(f"test(={name})" for name in identities)
            argv = ["cargo", "nextest", "run", "--locked", "-p", "mvmc-core", "--cargo-profile", "test-fast",
                    "--test", target, "--run-ignored", "only", "-E", expr, "--no-fail-fast", "--retries", "0",
                    "--success-output", "immediate", "--failure-output", "immediate"]
            if family == "general":
                required_env = {"MVMC_RS_CTEST_GENERAL": "1"}
            else:
                model, mode = record["name"].rsplit("-", 1)
                required_env = {"MVMC_RS_LANCZOS_PHYSICAL": "1", "MVMC_RS_LANCZOS_MODEL": model,
                                "MVMC_RS_LANCZOS_MODE": mode}
        elif family == "mpi":
            world = record["name"].split("-", 1)[1]
            argv = ["mpiexec", "-n", world, next(iter(binary)), "--ignored", "--exact", MPI, "--nocapture"]
            required_env = {"MVMC_RS_MPI_PHYSICAL": "1", "MPI179_PHYSCAL_OUTPUT":
                            str(PurePosixPath(read_json(evidence.parent / "envelope.json")["evidence_root"]) / f"world-{world}")}
        else:
            argv = ["bash", "scripts/verify_threaded_issue182.sh"]
            required_env = {"MVMC_RS_THREADED_TARGET_DIR": metadata["compiler_environment"].get("CARGO_TARGET_DIR"),
                            "TMPDIR": str(PurePosixPath(read_json(evidence.parent / "envelope.json")["evidence_root"]) / "thread-artifacts")}
        if record["argv"] != argv or any(env.get(key) != value for key, value in required_env.items()):
            raise ValueError("selected command argv/gate environment mismatch")
        target_dir = metadata["compiler_environment"].get("CARGO_TARGET_DIR")
        if not isinstance(target_dir, str) or not target_dir or env.get("CARGO_TARGET_DIR") != target_dir:
            raise ValueError("selected command target binding mismatch")
        if any(env.get(key) != "1" for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS")):
            raise ValueError("selected command backend thread setting mismatch")
    # Discovery/listing is not completion, but actual build profile/features must
    # still agree with the selected command, particularly for mpiexec's ELF.
    if family == "mpi":
        acquisitions = {
            "build": ["cargo", "test", "--locked", "-p", "mvmc-core", "--profile", "test-fast",
                      "--features", "mpi", "--test", "mpi_physcal", "--no-run", "--message-format=json"],
            "selection": [next(iter(binary)), "--list"],
        }
    else:
        target = {"general": "ctest_general_reference", "lanczos": "lanczos_transfer_physcal",
                  "thread": "threaded_issue182"}[family]
        expr = " | ".join(f"test(={name})" for name in identities)
        acquisitions = {"selection": ["cargo", "nextest", "list", "--locked", "-p", "mvmc-core",
                        "--cargo-profile", "test-fast", "--test", target, "--run-ignored", "only",
                        "-E", expr, "--message-format", "json"]}
    for name, argv in acquisitions.items():
        matching = [record for record in commands if record["name"] == name]
        if (len(matching) != 1 or matching[0]["selected"] is not None or matching[0]["argv"] != argv or
                matching[0]["env"].get("CARGO_TARGET_DIR") != metadata["compiler_environment"].get("CARGO_TARGET_DIR") or
                commands.index(matching[0]) >= commands.index(selected[0])):
            raise ValueError("build/selection command profile/features/order mismatch")


def validate_closure_posts(evidence, metadata, terminal):
    if type(metadata.get("closure_post_schema")) is not int or metadata["closure_post_schema"] != 1:
        raise ValueError("historical closure POST receipts lack current schema; not current-eligible")
    if not (evidence / "closure-posts.json").is_file():
        raise ValueError("missing fresh closure POST file")
    posts = read_json(evidence / "closure-posts.json")
    if (type(posts) is not dict or set(posts) != {"source", "fixtures", "binary"} or
            type(terminal.get("closure_posts")) is not dict or terminal["closure_posts"] != posts):
        raise ValueError("terminal/file closure POST mismatch")
    for record in posts.values():
        if (type(record) is not dict or set(record) != {"status", "error"} or
                record["status"] != "UNCHANGED" or record["error"] is not None):
            raise ValueError("successful driver requires three unchanged closure POSTs")
    for name in ("source", "fixtures", "binary"):
        if read_json(evidence / f"{name}.before.json") != read_json(evidence / f"{name}.after.json"):
            raise ValueError("closure POST claim contradicts captured before/after hashes")


def validate_package(package, family, plan):
    safe_tree(package)
    envelope = read_json(package / "envelope.json")
    if type(envelope.get("schema")) is not int or envelope["schema"] != SCHEMA or envelope.get("family") != family:
        raise ValueError("envelope schema/family mismatch")
    if any(envelope.get(key) != plan[key] for key in BINDING):
        raise ValueError("envelope head/run/attempt/workflow mismatch")
    if envelope.get("checkout_head") != plan["head"]:
        raise ValueError("actual checkout head not verified")
    if envelope.get("gate_environment") != {"CARGO_TERM_COLOR": "never"}:
        raise ValueError("selected gate color/context not recorded")
    evidence = package / "evidence"
    if envelope.get("job_status") not in ("success", "failure", "cancelled", "skipped"):
        raise ValueError("invalid job status")
    if envelope["job_status"] in ("cancelled", "skipped"):
        raise ValueError("cancelled/skipped selected job; not completed")
    if envelope.get("job_status") != "success":
        # Preserve a classified failed driver only with a verifiable envelope hash.
        if evidence.is_dir() and set(envelope.get("driver_hashes", {})) >= {"terminal.json", "family-ledger.json"}:
            for name in ("terminal.json", "family-ledger.json"):
                if digest(evidence / name) != envelope["driver_hashes"][name]:
                    raise ValueError("failed driver envelope hash mismatch")
            terminal = read_json(evidence / "terminal.json")
            row = read_json(evidence / "family-ledger.json")["families"][family]
            if (evidence / "metadata.json").exists():
                metadata = read_json(evidence / "metadata.json")
                if metadata.get("head") != plan["head"] or metadata.get("family") != family:
                    raise ValueError("failed driver belongs to another head/family")
            if type(terminal.get("exit_status")) is not int:
                raise ValueError("failed driver exit status must be an integer")
            if terminal.get("exit_status") == 1 and row.get("selected") is True and terminal.get("status") == row.get("status") in STATUSES[1:]:
                return {"selected": True, "status": row["status"], "comparison_evidence": "Unverified",
                        "numeric_reference_comparisons": None, "empty_contracts": 0,
                        "detail": "failed job/driver; no successful evidence closure claimed"}
        raise ValueError("failed/cancelled/skipped job or absent driver evidence")
    if set(envelope.get("driver_hashes", {})) != {"terminal.json", "family-ledger.json", "artifacts.json"}:
        raise ValueError("missing driver completion hashes")
    for name, expected in envelope["driver_hashes"].items():
        if digest(evidence / name) != expected:
            raise ValueError("driver envelope hash mismatch")
    validate_manifest(evidence, envelope["evidence_root"], read_json(evidence / "artifacts.json"))
    for name in ("commands.json", "selection.json", "linkage.txt", "rust-version.stdout",
                 "nextest-version.stdout", "reference-revisions.stdout"):
        if regular(evidence / name).stat().st_size == 0:
            raise ValueError("missing/empty command/selection/version evidence")
    terminal = read_json(evidence / "terminal.json")
    metadata = read_json(evidence / "metadata.json")
    rows = read_json(evidence / "family-ledger.json")["families"]
    row = rows[family]
    if set(rows) != set(FAMILIES) or metadata.get("head") != plan["head"] or metadata.get("family") != family:
        raise ValueError("driver head/family/ledger identity mismatch")
    if metadata.get("profile") != "test-fast" or metadata.get("features") != ("mpi" if family == "mpi" else "default") or not metadata.get("configuration") or not metadata.get("compiler_environment"):
        raise ValueError("profile/features/configuration/compiler environment absent")
    configuration = metadata["configuration"]
    expected_settings = {
        "general": {"model": "general_rbm_cmp", "seed": 12395, "prefixes_and_windows": [1, 2, 3, 20], "NSRCG": 0, "NStore": 1, "ranks": 1, "workers": 1, "threshold": 32},
        "lanczos": {"models": list(MODELS), "modes": ["real", "cmp"], "seed_override": 1, "ranks": 1, "groups": 1, "sample_counts": [100, 1000, 5000], "workers": 1},
        "mpi": {"model": "heisenberg_chain_real", "ranks": [2, 4], "group_widths": [1, 2], "repeats": 2, "seed": 1, "steps": 1, "samples": 3, "warmup": 1, "workers": 1},
        "thread": {"workers": [1, 2, 4], "threshold": 32, "sizes": [31, 32, 33], "steps": 2, "samples": 200, "ranks": 1, "seed": 1},
    }[family]
    if type(configuration) is not dict or any(
            not exact_setting(configuration.get(key), value)
            for key, value in expected_settings.items()):
        raise ValueError("bounded model/seed/steps/ranks/groups/workers settings mismatch")
    if type(terminal.get("exit_status")) is not int or terminal.get("exit_status") != 0 or terminal.get("status") != "Pass" or row.get("status") != "Pass" or row.get("selected") is not True:
        raise ValueError("unfinished/inconsistent selected driver terminal")
    expected_identities = {"general": list(GENERAL), "lanczos": [LANCZOS], "mpi": [MPI], "thread": [THREAD]}[family]
    expected_completed = {"general": list(GENERAL), "lanczos": [f"{model}-{mode}" for model in MODELS for mode in ("real", "cmp")],
                          "mpi": ["world2", "world4"], "thread": [THREAD]}[family]
    if sorted(row.get("selected_test_identities", [])) != sorted(expected_identities):
        raise ValueError("selected test identity mismatch")
    if sorted(row.get("completed_selection_identities", [])) != sorted(expected_completed) or terminal.get("completed") != row["completed_selection_identities"]:
        raise ValueError("incomplete/duplicate completed selections")
    if type(row.get("started_driver_invocations")) is not int or row["started_driver_invocations"] != {"general": 1, "lanczos": 6, "mpi": 2, "thread": 1}[family] or type(row.get("helper_tests")) is not int or row["helper_tests"] != 0:
        raise ValueError("invocation/helper count mismatch")
    for other in FAMILIES:
        if other != family and (rows[other].get("selected") is not False or rows[other].get("status") not in ("NotRun", "ExplicitSkip")):
            raise ValueError("unselected driver family masquerades as executed")
    for stem in ("source", "fixtures", "binary"):
        stable_closure(evidence, stem)
    validate_closure_posts(evidence, metadata, terminal)
    validate_command_receipts(evidence, family, metadata, expected_identities)
    if family == "general":
        text = regular(evidence / "general.stdout").read_text() + "\n" + regular(evidence / "general.stderr").read_text()
        observed = validate_general_settings(text)
        report = read_json(evidence / "general-settings.json")
        if (type(report.get("schema")) is not int or report != observed or
                any(type(row[key]) is not type(value)
                    for row, expected in zip(report["records"], observed["records"])
                    for key, value in expected.items())):
            raise ValueError("General settings report/raw selected call mismatch")
    if (metadata.get("oracle_execution") != "none; checked-in fixtures only" or
            metadata.get("reference_version") != "per-fixture provenance, NOT current Julia runtime verification" or
            metadata.get("not_claimed") != "full13 matrix, fullJulia features, fullC sampler, InterAll"):
        raise ValueError("offline reference scope/claim mismatch")
    reference_provenance.validate(metadata.get("offline_reference"),
                                  read_json(evidence / "fixtures.before.json"))
    if family == "mpi":
        mpi_provider.validate_package(evidence, read_json)
    backend = read_json(evidence / "backend.json")
    if type(backend.get("actual_threads")) is not int or backend["actual_threads"] != 1 or not re.match(r"^OpenBLAS \d+\.\d+\.\d+(?:\s|$)", backend.get("actual_config", "")) or not backend.get("actual_core"):
        raise ValueError("actual single-thread backend identity absent")
    if not re.fullmatch(r"[0-9a-f]{64}", backend.get("library_sha256", "")):
        raise ValueError("backend library digest absent")
    if family == "lanczos":
        fixture_metadata.validate(read_json(evidence / "metadata.json").get("fixture_modpara"),
                                  read_json(evidence / "fixtures.before.json"))
        records = read_json(evidence / "dc-comparisons.json")["records"]
        expected = {(model, mode, file, status) for model in MODELS[1:] for mode in ("real", "cmp")
                    for file, status in (("zvo_ls_cisajs_001.dat", "REFERENCE_COMPARED"),
                                         ("zvo_ls_cisajscktalt_001.dat", "REFERENCE_COMPARED"),
                                         ("zvo_ls_cisajscktaltex_001.dat", "EMPTY_CONTRACT"))}
        actual = [(r["model"], r["mode"], r["file"], r["status"]) for r in records]
        if len(actual) != 12 or set(actual) != expected or type(row.get("numeric_reference_comparisons")) is not int or row.get("numeric_reference_comparisons") != 8 or type(row.get("empty_contracts")) is not int or row.get("empty_contracts") != 4 or row.get("comparison_evidence") != "Verified":
            raise ValueError("DC numeric/empty evidence mismatch")
    elif row.get("numeric_reference_comparisons") is not None or type(row.get("empty_contracts")) is not int or row.get("empty_contracts") != 0:
        raise ValueError("invented uninstrumented comparison totals")
    return {"selected": True, **row, "detail": "validated package; bounded scope only"}


def aggregate(plan, packages, expected):
    validate_plan(plan, expected)
    selected = plan["selected"]
    expected_names = {artifact_name(name, plan) for name in selected}
    found = {path.name for path in packages.iterdir()} if packages.is_dir() else set()
    result = {"schema": SCHEMA, **expected, "families": {}, "scope": plan["scope"], "exit_status": 0}
    extras = found - expected_names
    for family in FAMILIES:
        if family not in selected:
            result["families"][family] = {"selected": False, "status": "NotRun"}
            continue
        try:
            if packages.is_symlink():
                raise ValueError("unsafe package download root")
            if extras:
                raise ValueError("unexpected/duplicate/wrong-attempt family package")
            row = validate_package(packages / artifact_name(family, plan), family, plan)
        except (ValueError, OSError, KeyError, TypeError) as error:
            row = {"selected": True, "status": "Failure", "detail": str(error),
                   "comparison_evidence": "Unverified", "numeric_reference_comparisons": None, "empty_contracts": 0}
        result["families"][family] = row
        if row["status"] != "Pass":
            result["exit_status"] = 1
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("plan", "seal", "aggregate"))
    parser.add_argument("--head", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--packages", type=Path)
    parser.add_argument("--job-status")
    parser.add_argument("--mpi-provider-receipt", type=Path)
    args = parser.parse_args()
    try:
        binding = identity(args.head, args.run_id, args.attempt, args.workflow)
        if args.operation == "plan":
            plan = make_plan(args.scope, binding)
            write_new(args.output, plan)
            if os.environ.get("GITHUB_OUTPUT"):
                with Path(os.environ["GITHUB_OUTPUT"]).open("a") as file:
                    file.write("matrix=" + json.dumps({"family": plan["selected"]}) + "\n")
            return 0
        if args.plan is None:
            raise ValueError("plan artifact is required")
        plan = read_json(args.plan)
        validate_plan(plan, binding)
        if args.operation == "seal":
            if args.family is None or args.evidence is None or args.job_status is None:
                raise ValueError("seal requires selected family/evidence/job status")
            seal(plan, args.family, args.evidence, args.output, args.job_status,
                 args.head, os.environ.get("CARGO_TERM_COLOR"), args.mpi_provider_receipt)
            return 0
        if args.packages is None:
            raise ValueError("aggregate requires package download root")
        result = aggregate(plan, args.packages, binding)
        write_new(args.output, result)
        return result["exit_status"]
    except (ValueError, OSError, KeyError, TypeError) as error:
        if args.operation == "aggregate" and not args.output.exists():
            write_new(args.output, {"exit_status": 1, "status": "Failure", "detail": str(error)})
        print(f"optional aggregation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

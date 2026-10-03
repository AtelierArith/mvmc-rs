"""Bounded proposal wrapper. No capture without explicit reviewed approval token.

Runs one precompiled exact libtest identity. nextest LIST provenance is distinct
from parent libtest execution. Never claims a nextest execution it did not run.
"""
import argparse
import json
import os
import signal
from pathlib import Path
import subprocess
import time

from thread_envelope import digest, load_json, require, review_contract, validate, verify_inputs, verify_selection


def group_members(pgid):
    """Linux observation: non-zombie members, independent of parent wait status."""
    require(type(pgid) is int and pgid > 0, "owned positive process group required")
    require(Path("/proc/self/stat").is_file(), "Linux /proc required for group observation")
    alive = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdecimal():
            continue
        try:
            text = (directory / "stat").read_text()
        except (FileNotFoundError, ProcessLookupError):
            continue  # process disappeared during enumeration
        fields = text[text.rfind(")") + 2:].split()
        require(len(fields) >= 3, "unreadable process-group state")
        if int(fields[2]) == pgid and fields[0] not in ("Z", "X"):
            alive.append(int(directory.name))
    return alive


def terminate_owned_group(process, grace=1.0, deadline=2.0):
    """Never reap the leader during TERM grace; final KILL precedes parent wait.

    The unreaped leader holds its PID/PGID against reuse while signalling. No
    further signals are sent after reaping. Unknown membership fails closed.
    """
    result = {"timed_out": True, "status": "CleanupIncomplete", "errors": [], "races": []}
    pgid = process.pid
    require(type(pgid) is int and pgid > 0, "owned positive process group required")
    def signal_group(sig):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            result["races"].append(sig.name)  # observation below must confirm no live members
        except OSError as error:
            result["errors"].append(type(error).__name__)
    def observe():
        try:
            return group_members(pgid)
        except (OSError, ValueError) as error:
            result["errors"].append("observation:" + type(error).__name__)
            return None
    try:
        signal_group(signal.SIGTERM)
        end = time.monotonic() + grace
        while time.monotonic() < end:
            members = observe()
            if members is None or not members:
                break
            time.sleep(0.02)
    finally:
        # Mandatory even if parent exited first or TERM lookup raced. Do NOT
        # call wait()/poll() before this step; they could reap the leader.
        signal_group(signal.SIGKILL)
        try:
            process.wait(timeout=deadline)
        except (subprocess.TimeoutExpired, OSError) as error:
            result["errors"].append("reap:" + type(error).__name__)
        end = time.monotonic() + deadline
        remaining = observe()
        while remaining and time.monotonic() < end:
            time.sleep(0.02)
            remaining = observe()
        if remaining == [] and not result["errors"]:
            result["status"] = "Terminated"
    return result


def run(workspace, output, reviewed, selection_path, execute=False):
    require(execute and os.environ.get("ISSUE182_APPROVED_EXECUTION_TOKEN") == reviewed["run_uuid"],
            "explicit reviewer execution authorization required")
    identity, _, _, _ = review_contract(reviewed)
    workspace = Path(workspace).resolve(strict=True)
    require(reviewed.get("wrapper_sha256") == digest(Path(__file__).read_bytes()), "unreviewed wrapper source")
    verify_inputs(workspace, reviewed)
    selection_data = Path(selection_path).read_bytes()
    require(digest(selection_data) == reviewed["selection_artifact"], "wrong precompiled selection binding")
    verify_selection(selection_data, reviewed)
    output = Path(output)
    require(not output.resolve().is_relative_to(Path(workspace).resolve()), "output must be outside frozen workspace")
    output.mkdir(parents=False, exist_ok=False)
    records_dir = output / "records"
    records_dir.mkdir()
    (output / "selection.json").write_bytes(selection_data)
    env = os.environ.copy()
    for name in list(env):
        if name.startswith(("ISSUE182_", "MVMC_RS_")) or name == "JULIA_MVMC_ROOT":
            env.pop(name)
    for name in ("LD_PRELOAD", "DYLD_INSERT_LIBRARIES"):
        require(not env.get(name), "unreviewed runtime interposition")
    env.update(reviewed["launch_environment"])
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
        env[name] = "1"
    for name, target in (("MVMC_RS_THREADED_FIXTURE_ROOT", workspace),
                         ("JULIA_MVMC_ROOT", workspace / "extern/Julia-mVMC")):
        require(name not in reviewed["launch_environment"] or
                Path(reviewed["launch_environment"][name]).resolve() == target,
                "fixture root must remain within reviewed workspace closure")
        env[name] = str(target)
    if "MVMC_RS_THREADED_REAL_FSZ_ROOT" in reviewed["launch_environment"]:
        require(Path(reviewed["launch_environment"]["MVMC_RS_THREADED_REAL_FSZ_ROOT"]).resolve().is_relative_to(workspace),
                "external real-FSZ root requires separate binding adapter")
    env.update(ISSUE182_CASE_RECORD_DIR=str(records_dir.resolve()),
               ISSUE182_CASE_RUN_UUID=reviewed["run_uuid"], MVMC_RS_THREADED_182="1")
    argv = [reviewed["binary"]["path"], "--ignored", "--exact", identity, "--nocapture", "--test-threads=1"]
    code = None
    cleanup = {"timed_out": False, "status": "NotRequired", "errors": [], "races": []}
    failures = []
    post_inputs = False
    try:
        with (output / "parent.stdout").open("wb") as stdout, (output / "parent.stderr").open("wb") as stderr:
            # Exclusive process group; timeout must not leave nested children.
            process = subprocess.Popen(argv, cwd=workspace, env=env, stdout=stdout,
                                       stderr=stderr, start_new_session=True)
            try:
                code = process.wait(timeout=reviewed["timeout_seconds"])
            except subprocess.TimeoutExpired:
                cleanup = terminate_owned_group(process)
                code = None  # timeout is never a normal successful return
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        failures.append("execution:" + type(error).__name__)
        code = None
    finally:
        # Post-check failure must not bypass preservation of partial evidence.
        try:
            verify_inputs(workspace, reviewed)
            post_inputs = True
        except (OSError, ValueError) as error:
            failures.append("post_inputs:" + type(error).__name__)
        parent = {"argv": argv, "run_uuid": reviewed["run_uuid"], "head": reviewed["head"],
                  "binary_sha256": reviewed["binary"]["sha256"], "exit_code": code,
                  "cleanup": cleanup, "post_inputs": post_inputs}
        (output / "parent.json").write_text(json.dumps(parent, sort_keys=True) + "\n")
        hashes = {str(p.relative_to(output)): digest(p.read_bytes()) for p in output.rglob("*") if p.is_file()}
        envelope = {"reviewed_sha256": digest(json.dumps(reviewed, sort_keys=True).encode()),
                    "run_uuid": reviewed["run_uuid"], "head": reviewed["head"],
                    "parent_exit_code": code, "artifacts": hashes}
        if failures:
            envelope["preservation_failures"] = failures
        (output / "envelope.json").write_text(json.dumps(envelope, sort_keys=True) + "\n")
    require(not failures, "execution/post-input failure; partial artifacts retained")
    return validate(output, workspace, envelope, reviewed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewed", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(Path(args.workspace), Path(args.output),
                         load_json(Path(args.reviewed).read_bytes()), args.selection, args.execute)))

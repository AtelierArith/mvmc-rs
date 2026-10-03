#!/usr/bin/env -S uv run --no-project python
"""Staging/provenance support for generate-numerical-references.sh; no Rust oracle."""
import hashlib
import ctypes
import ctypes.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

COPIED = ("scripts", "c_toolbox", "tests/fixtures", "extern/mVMC-1.3.0", "extern/Julia-mVMC")
OUTPUTS = ("tests/fixtures/", "c_toolbox/")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes(tree):
    return {path.relative_to(tree).as_posix(): sha(path)
            for folder in COPIED if (tree / folder).is_dir()
            for path in sorted((tree / folder).rglob("*"))
            if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts}


def command(argv, required=False):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, check=required)
        return {"argv": argv, "exit_code": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.CalledProcessError) as error:
        if required:
            raise RuntimeError(f"Required provenance command failed: {argv}: {error}") from error
        return {"argv": argv, "unavailable": str(error)}


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def openblas_metadata():
    name = ctypes.util.find_library("openblas")
    if not name:
        return {"unavailable": "System OpenBLAS library was not found"}
    try:
        library = ctypes.CDLL(name)
        library.openblas_get_config.restype = ctypes.c_char_p
        library.openblas_get_corename.restype = ctypes.c_char_p
        return {"library": name, "configuration": library.openblas_get_config().decode(),
                "cpu_backend": library.openblas_get_corename().decode()}
    except (OSError, AttributeError) as error:
        return {"library": name, "unavailable": str(error)}


def git_metadata(root):
    prefix = ["git", "-c", f"safe.directory={root}", "-C", str(root)]
    result = {"checkout": command(prefix + ["rev-parse", "HEAD"], required=True),
              "worktree": command(prefix + ["status", "--short"], required=True),
              "gitlinks": command(prefix + ["ls-tree", "-r", "HEAD", "extern/"], required=True)}
    result["reference_heads"] = {}
    for name in ("extern/mVMC-1.3.0", "extern/Julia-mVMC"):
        source = root / name
        if (source / ".git").exists():
            result["reference_heads"][name] = command(
                ["git", "-c", f"safe.directory={source}", "-C", str(source), "rev-parse", "HEAD"],
                required=True)
        else:
            result["reference_heads"][name] = {
                "git_checkout": False, "source_present": source.is_dir()}
    return result


def prepare(root, output):
    if any(output.iterdir()):
        raise RuntimeError(f"Staging directory must be empty: {output}")
    for directory in COPIED:
        source = root / directory
        if source == output or source in output.parents:
            raise RuntimeError("Output must be outside copied source/reference directories")
    if output == root:
        raise RuntimeError("Output cannot be the source checkout")
    tree = output / "workspace"
    tree.mkdir()
    for directory in COPIED:
        source = root / directory
        if source.is_dir():
            # Copy data, never writable symlinks to the source checkout. Omit Git
            # pointer files: references retain provenance, not invalid copied .git.
            shutil.copytree(source, tree / directory, symlinks=False,
                            ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"))
    for directory in ("tests/fixtures", "c_toolbox", "scripts"):
        if not (tree / directory).is_dir():
            raise RuntimeError(f"Missing required input directory: {directory}")
    baseline = hashes(tree)
    save(output / "baseline.json", baseline)
    save(output / "manifest.json", {
        "schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkout": str(root), "staging_workspace": str(tree),
        "status": "prepared", "source_git": git_metadata(root),
        "input_sha256": baseline,
        "note": "Independent C/Julia oracles only; generation never reads Rust results."})


def environment(root, output):
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["environment"] = {
        "os": platform.system(), "architecture": platform.machine(),
        "platform": platform.platform(), "libc": platform.libc_ver(),
        "system_openblas": openblas_metadata(),
        "thread_variables": {key: os.environ.get(key, "unset") for key in (
            "OPENBLAS_NUM_THREADS", "OPENBLAS_CORETYPE", "OMP_NUM_THREADS",
            "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "JULIA_NUM_THREADS")},
        "oracle_variables": {key: os.environ.get(key, "unset") for key in (
            "REFERENCE_CC", "CC", "CXX", "FC", "JULIA_DEPOT_PATH", "UV_CACHE_DIR")},
        "commands": [command(argv) for argv in (
            [os.environ.get("REFERENCE_CC", "cc"), "--version"],
            ["g++", "--version"], ["gfortran", "--version"],
            ["ldd", "--version"], ["pkg-config", "--modversion", "openblas"],
            ["pkg-config", "--cflags", "--libs", "openblas"],
            ["mpichversion"], ["uv", "--version"], ["uname", "-a"])],
        "rust_blas_abi": "LP64 (workspace blas/lapack bindings); Julia backend recorded separately"}
    save(output / "manifest.json", manifest)


def finish(root, output, status):
    manifest = json.loads((output / "manifest.json").read_text())
    before = json.loads((output / "baseline.json").read_text())
    after = hashes(output / "workspace")
    changes = []
    for path in sorted(set(before) | set(after)):
        if before.get(path) != after.get(path):
            changes.append({"path": path, "before_sha256": before.get(path),
                            "after_sha256": after.get(path)})
    supporting = [change for change in changes if not change["path"].startswith(OUTPUTS)]
    changes = [change for change in changes if change["path"].startswith(OUTPUTS)]
    save(output / "changes.json", changes)
    manifest["status"] = "success" if status == 0 else "failed"
    manifest["exit_code"] = status
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["changes"] = changes
    manifest["supporting_changes"] = supporting
    commands = output / "commands.txt"
    manifest["exact_commands"] = commands.read_text() if commands.exists() else ""
    compiler_log = output / "compiler-commands.log"
    manifest["exact_compiler_commands"] = compiler_log.read_text() if compiler_log.exists() else ""
    julia_log = output / "julia-environment.log"
    if julia_log.exists():
        manifest["julia_environment"] = julia_log.read_text()
    manifest["logs"] = {path.name: sha(path) for path in output.glob("*.log")}
    save(output / "manifest.json", manifest)
    print(f"{manifest['status']}: {len(changes)} changed files; review changes.json and manifest.json")
    print(f"Compare: diff -ru {root / 'tests/fixtures'} {output / 'workspace/tests/fixtures'}")


def julia_overlay(root, output, check):
    """Retain independent Linux outputs while restoring the archived inputs."""
    tree = output / "workspace"
    fixture_root = tree / "tests/fixtures"
    overlay = fixture_root / "linux_gnu_julia"
    baseline = json.loads((output / "baseline.json").read_text())
    digest_path = overlay / "SHA256.json"
    digests = json.loads(digest_path.read_text()) if digest_path.exists() else {}
    generated = {}
    mismatches = []
    for relative in sorted(set((output / "julia-generated-paths.txt").read_text().splitlines())):
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or relative.startswith("linux_gnu_julia/"):
            raise RuntimeError(f"Invalid generated fixture path: {relative}")
        source = fixture_root / path
        if not source.is_file():
            raise RuntimeError(f"Missing independently generated output: {relative}")
        original = root / "tests/fixtures" / path
        digest = sha(source)
        generated[relative] = digest
        # Solver diagnostics and descriptive headers are recorded, but are not
        # read as runner expectations. Loaded normal-mode files generated by
        # history jobs are likewise retained in the external staging manifest.
        omitted = path.name in ("fixed-input.txt", "gram.txt", "reference.txt") or path.name.startswith("loaded-") or (path.name.startswith("history-") and "fsz" not in path.name)
        if not omitted:
            destination = overlay / path
            differs = not original.is_file() or original.read_bytes() != source.read_bytes()
            expected = destination if destination.is_file() else original
            if check and (not expected.is_file() or expected.read_bytes() != source.read_bytes()):
                mismatches.append(relative)
            if not check:
                if differs:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                elif destination.exists():
                    destination.unlink()
                digests[relative] = digest
        # Always restore the historical source fixtures; review/apply can only
        # touch the independent Linux overlay, never these macOS archives.
        if original.is_file():
            if sha(original) != baseline.get("tests/fixtures/" + relative):
                raise RuntimeError(f"Source changed during generation: {relative}")
            shutil.copy2(original, source)
        else:
            source.unlink()
    save(output / "julia-generated-sha256.json", generated)
    if not check:
        save(digest_path, digests)
        save(overlay / "regeneration.json", {
            "schema": 1, "generated_utc": datetime.now(timezone.utc).isoformat(),
            "generated_sha256": generated,
            "environment": json.loads((output / "manifest.json").read_text()).get("environment"),
            "julia_environment": (output / "julia-environment.log").read_text(),
            "exact_commands": (output / "commands.txt").read_text(),
            "source_git": json.loads((output / "manifest.json").read_text())["source_git"],
            "observer_instrumentation": "Isolated modules; Main.capture_ target qualification; verify path logging only; no numerical kernels or RNG changes."})
    print(f"Independent Linux Julia outputs: {len(generated)}; archive restored; overlay mismatches: {len(mismatches)}")
    if mismatches:
        raise RuntimeError("Linux Julia overlay check failed: " + ", ".join(mismatches[:10]))


def apply(root, output):
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest.get("schema") != 1 or manifest.get("status") != "success":
        raise RuntimeError("Only successful staging runs can be applied")
    changes = manifest["changes"]
    # Reject stale expectations whenever their full input snapshot changed.
    # Unrelated Rust implementation edits live outside these copied directories.
    before = manifest["input_sha256"]
    current = hashes(root)
    if current != before:
        changed_inputs = sorted(path for path in set(current) | set(before)
                                if current.get(path) != before.get(path))
        raise RuntimeError("Source inputs changed since staging; regenerate before applying: "
                           + ", ".join(changed_inputs[:10]))
    for change in changes:
        relative = change["path"]
        if not relative.startswith(OUTPUTS) or ".." in Path(relative).parts:
            raise RuntimeError(f"Refusing to apply outside fixtures/toolbox: {relative}")
        destination = root / relative
        source = output / "workspace" / relative
        if destination.is_symlink() or any(parent.is_symlink() for parent in destination.parents):
            raise RuntimeError(f"Refusing a symlink destination: {destination}")
        actual = sha(destination) if destination.is_file() else None
        if actual != change["before_sha256"]:
            raise RuntimeError(f"Source changed since staging; regenerate before applying: {relative}")
        if not source.is_file() or sha(source) != change["after_sha256"]:
            raise RuntimeError(f"Staged output changed or was deleted: {relative}")
    # Validate every file first, then atomically replace individual outputs.
    for change in changes:
        destination = root / change["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
            temporary.write((output / "workspace" / change["path"]).read_bytes())
            temporary_name = temporary.name
        os.chmod(temporary_name, destination.stat().st_mode & 0o777 if destination.exists() else 0o644)
        os.replace(temporary_name, destination)
    print(f"Applied {len(changes)} reviewed fixture/toolbox files; inspect git diff and run Rust tests.")


def main():
    action, root_arg, output_arg, *extra = sys.argv[1:]
    root, output = Path(root_arg).resolve(), Path(output_arg).resolve()
    if action == "prepare": prepare(root, output)
    elif action == "environment": environment(root, output)
    elif action == "finish": finish(root, output, int(extra[0]))
    elif action == "julia-overlay": julia_overlay(root, output, extra[0] == "true")
    elif action == "apply": apply(root, output)
    else: raise RuntimeError(f"Unknown action: {action}")


if __name__ == "__main__":
    main()

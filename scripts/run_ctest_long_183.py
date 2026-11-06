#!/usr/bin/env python3
"""Explicit long ctest gate driver for the #183 optional-gates workflow.

The verdict gate is the upstream C/Julia ctest rule at the upstream run length
(`rust_ctest_upstream_rule_selected_models`: fail only when difference >= 3*ref_std and
>= 1e-8 against the C-shipped ref_mean/ref_std). It is the only long ctest gate: the
20-step prefix gates (`MVMC_RS_CTEST_MODELS`, `MVMC_RS_CTEST_PREFIX_MODELS`) and their
Julia/1.3.0 prefix fixtures were removed in #702 (#695, #689).

Never called by Cargo or normal CI. Runs the ignored ctest gates for the requested
models, records the same provenance classes as the bounded families (exact head,
profile/features, actual linked BLAS getter, threads, seeds, steps, fixture hashes)
and classifies the terminal status. Only `Pass` exits 0; Unsupported, MissingFixture,
Unverified and Failure all exit non-zero. Fixtures are never generated here.
"""
import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import run_optional_gates_183 as gates

ROOT = gates.ROOT
UPSTREAM = ROOT / "tests/fixtures/ctest_upstream_reference"
JULIA = ROOT / "extern/Julia-mVMC"
VERDICT_GATE = ("ctest_equivalent", "rust_ctest_upstream_rule_selected_models",
                "MVMC_RS_CTEST_UPSTREAM_MODELS")
NAME = re.compile(r"^[a-z0-9_]+$")


def known_models():
    text = (ROOT / "crates/mvmc-core/tests/ctest_equivalent.rs").read_text()
    return re.findall(r'fixture: "([a-z0-9_]+)"', text)


def requested_models(selection):
    if selection.strip() == "all":
        return known_models()
    return [name.strip() for name in selection.split(",")]


def classify(texts, returncodes):
    """Terminal status from selected-gate diagnostics; Pass requires every exit 0."""
    if all(code == 0 for code in returncodes):
        return "Pass"
    joined = "\n".join(texts)
    for marker, status in (("Unsupported", "Unsupported"), ("MissingFixture", "MissingFixture"),
                           ("UNVERIFIED", "Unverified")):
        if marker in joined:
            return status
    return "Failure"


def upstream_provenance(model):
    """Presence and hashes of the C-shipped ref_mean/ref_std used by the verdict gate."""
    directory = UPSTREAM / model
    if not NAME.match(model) or not directory.is_dir():
        return {"model": model, "upstream_reference_present": False}
    files = [directory / n for n in ("ref_mean.dat", "ref_std.dat", "inputs.sha256")]
    present = all(f.is_file() for f in files)
    entry = {"model": model, "upstream_reference_present": present}
    if present:
        inputs = JULIA / "test/integration/reference" / model / "inputs"
        if inputs.is_dir():
            files.extend(p for p in sorted(inputs.rglob("*")) if p.is_file())
        entry["upstream_reference_sha256"] = gates.digest_files(files)
    return entry


def run(selection, output):
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    if not env.get("CARGO_TARGET_DIR"):
        raise ValueError("explicit checkout-specific CARGO_TARGET_DIR required")
    if platform.system() != "Linux":
        raise ValueError("long ctest dispatch supports Linux only")
    models = requested_models(selection)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    expected = env.get("EXPECTED_HEAD")
    if expected and expected != head:
        raise ValueError(f"checkout head {head} differs from dispatched {expected}")
    metadata = {
        "gate": "long ctest (upstream rule at upstream run length)", "selection": selection, "models": models,
        "head": head, "run_id": env.get("RUN_ID"), "attempt": env.get("ATTEMPT"),
        "workflow": env.get("WORKFLOW_ID"), "profile": "test-fast", "features": "default",
        "platform": platform.platform(), "started_unix": time.time(),
        "oracle_execution": "none; checked-in fixtures only",
        "reference_version": "C-shipped ctest references (mVMC 1.4.0 e83aa6cb, "
                             "byte-identical to 1.3.0); inputs from the Julia reference checkout",
        "not_claimed": "model coverage beyond the selection; full13 matrix is #180",
    }
    (output / "rustc.txt").write_text(subprocess.check_output(["rustc", "-Vv"], text=True))
    (output / "reference-revisions.txt").write_text(subprocess.check_output(
        ["git", "submodule", "status", "--recursive"], cwd=ROOT, text=True))
    metadata["reference_manifest_sha256"] = gates.digest_files(
        [JULIA / "Manifest-v1.13.toml"])
    metadata["per_model"] = [upstream_provenance(m) for m in models]
    gates.write_json(output / "metadata.json", metadata)
    build = subprocess.run(
        ["cargo", "test", "-p", "mvmc-core", "--profile", "test-fast",
         "--test", VERDICT_GATE[0],
         "--no-run", "--message-format=json"],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, text=True)
    if build.returncode:
        raise RuntimeError("build failed")
    records = [json.loads(line) for line in build.stdout.splitlines() if line.startswith("{")]
    target, test, selector = VERDICT_GATE
    binary = gates.cargo_binary(records, target)
    gates.write_json(output / "binaries.json", gates.digest_files([binary]))
    backend = gates.backend(binary, output / "linkage.txt")
    gates.write_json(output / "backend.json", backend)
    run_env = dict(env, **{selector: selection})
    result = subprocess.run([binary, "--ignored", "--exact", test, "--nocapture"],
                            cwd=ROOT, env=run_env, capture_output=True, text=True)
    (output / f"{test}.stdout").write_text(result.stdout)
    (output / f"{test}.stderr").write_text(result.stderr)
    texts = [result.stdout, result.stderr]
    verdict_code = result.returncode
    status = classify(texts, [verdict_code])
    gate_lines = [line for t in texts for line in t.splitlines()
                  if "parity gate" in line or "UNVERIFIED" in line
                  or line.startswith("ctest-upstream result")]
    result = {"status": status, "verdict_gate": VERDICT_GATE[1], "verdict_exit_code": verdict_code,
              "head": head, "selection": selection, "blas": backend,
              "threads": backend["actual_threads"], "profile": "test-fast",
              "features": "default", "gate_lines": gate_lines}
    gates.write_json(output / "status.json", result)
    summary = [f"### Long ctest gate: **{status}**", f"- selection: `{selection}`",
               f"- head: `{head}`", "- profile/features: test-fast / default",
               f"- BLAS: {backend['actual_config']} core={backend['actual_core']} "
               f"threads={backend['actual_threads']}"]
    for entry in metadata["per_model"]:
        summary.append(f"- {entry['model']}: upstream_reference="
                       f"{entry['upstream_reference_present']}")
    summary += ["```", *(gate_lines[-120:] or ["no gate status lines"]), "```"]
    (output / "summary.md").write_text("\n".join(summary) + "\n")
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        status = run(args.selection, args.output.resolve())
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f"long ctest driver failed: {error}", file=sys.stderr)
        return 1
    print(f"long ctest status: {status}")
    return 0 if status == "Pass" else 1


if __name__ == "__main__":
    sys.exit(main())

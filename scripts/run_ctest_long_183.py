#!/usr/bin/env python3
"""Explicit long ctest gate driver for the #183 optional-gates workflow.

The verdict gate is the upstream C/Julia ctest rule at the upstream run length
(`rust_ctest_upstream_rule_selected_models`: fail only when difference >= 3*ref_std and
>= 1e-8 against the C-shipped ref_mean/ref_std). The 20-step independent-reference gates
are still executed and reported, but are supplementary: they stay MissingFixture/Unverified
until independent step-20 references exist and never contribute a Pass.

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
FIXTURES = ROOT / "tests/fixtures/ctest_model_prefixes"
UPSTREAM = ROOT / "tests/fixtures/ctest_upstream_reference"
JULIA = ROOT / "extern/Julia-mVMC"
STEPS = [1, 2, 3, 20]
VERDICT_GATE = ("ctest_equivalent", "rust_ctest_upstream_rule_selected_models",
                "MVMC_RS_CTEST_UPSTREAM_MODELS")
SUPPLEMENTARY_GATES = (
    ("ctest_equivalent", "rust_ctest_equivalent_selected_models",
     "MVMC_RS_CTEST_MODELS"),
    ("ctest_model_prefixes", "canonical_models_match_independent_prefix_oracles",
     "MVMC_RS_CTEST_PREFIX_MODELS"),
)
GATES = (VERDICT_GATE, *SUPPLEMENTARY_GATES)
TARGETS = tuple(dict.fromkeys(target for target, _, _ in GATES))
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
        return {"upstream_reference_present": False}
    files = [directory / n for n in ("ref_mean.dat", "ref_std.dat", "inputs.sha256")]
    present = all(f.is_file() for f in files)
    entry = {"upstream_reference_present": present}
    if present:
        entry["upstream_reference_sha256"] = gates.digest_files(files)
    return entry


def model_provenance(model):
    entry = upstream_provenance(model)
    entry.update(_step_provenance(model))
    return entry


def _step_provenance(model):
    if not NAME.match(model) or not (FIXTURES / model).is_dir():
        return {"model": model, "reference_present": False, "steps_requested": STEPS}
    steps_present = sorted(int(p.name[5:]) for p in (FIXTURES / model).glob("step-*")
                           if p.name[5:].isdigit())
    seeds = set()
    for step in steps_present:
        settings = FIXTURES / model / f"step-{step}" / "model-settings.txt"
        if settings.is_file():
            seeds.update(m.group(1) for m in
                         re.finditer(r"(?:seed|RndSeed)=(\d+)", settings.read_text()))
    files = [p for p in (FIXTURES / model).rglob("*") if p.is_file()]
    files.append(FIXTURES / "provenance.txt")
    inputs = JULIA / "test/integration/reference" / model / "inputs"
    if inputs.is_dir():
        files.extend(p for p in inputs.rglob("*") if p.is_file())
    return {"model": model, "reference_present": True, "steps_requested": STEPS,
            "steps_present": steps_present,
            "twenty_step_reference_present": 20 in steps_present,
            "seeds": sorted(seeds), "fixture_sha256": gates.digest_files(files)}


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
        "reference_version": "per-fixture provenance (Julia 1.13.1 / C contract), "
                             "NOT current Julia runtime verification",
        "not_claimed": "model coverage beyond the selection; full13 matrix is #180",
    }
    (output / "rustc.txt").write_text(subprocess.check_output(["rustc", "-Vv"], text=True))
    (output / "reference-revisions.txt").write_text(subprocess.check_output(
        ["git", "submodule", "status", "--recursive"], cwd=ROOT, text=True))
    metadata["reference_manifest_sha256"] = gates.digest_files(
        [JULIA / "Manifest-v1.13.toml"])
    metadata["per_model"] = [model_provenance(m) for m in models]
    gates.write_json(output / "metadata.json", metadata)
    build = subprocess.run(
        ["cargo", "test", "-p", "mvmc-core", "--profile", "test-fast",
         *(arg for target in TARGETS for arg in ("--test", target)),
         "--no-run", "--message-format=json"],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, text=True)
    if build.returncode:
        raise RuntimeError("build failed")
    records = [json.loads(line) for line in build.stdout.splitlines() if line.startswith("{")]
    binaries = {target: gates.cargo_binary(records, target) for target, _, _ in GATES}
    gates.write_json(output / "binaries.json", gates.digest_files(binaries.values()))
    backend = gates.backend(binaries[GATES[0][0]], output / "linkage.txt")
    gates.write_json(output / "backend.json", backend)
    texts, codes, supplementary = [], [], {}
    for target, test, selector in GATES:
        run_env = dict(env, **{selector: selection})
        result = subprocess.run([binaries[target], "--ignored", "--exact", test, "--nocapture"],
                                cwd=ROOT, env=run_env, capture_output=True, text=True)
        (output / f"{test}.stdout").write_text(result.stdout)
        (output / f"{test}.stderr").write_text(result.stderr)
        pair = [result.stdout, result.stderr]
        if (target, test, selector) == VERDICT_GATE:
            texts.extend(pair)
            codes.append(result.returncode)
            verdict_code = result.returncode
        else:
            supplementary[test] = {"exit_code": result.returncode,
                                   "classification": classify(pair, [result.returncode]),
                                   "contributes_to_verdict": False}
    status = classify(texts, codes)
    gate_lines = [line for t in texts for line in t.splitlines()
                  if "parity gate" in line or "UNVERIFIED" in line
                  or line.startswith("ctest-upstream result")]
    result = {"status": status, "verdict_gate": VERDICT_GATE[1], "verdict_exit_code": verdict_code,
              "supplementary_20step_gates": supplementary,
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
                       f"{entry['upstream_reference_present']} reference_present={entry['reference_present']} "
                       f"steps_present={entry.get('steps_present')} seeds={entry.get('seeds')} "
                       f"20-step reference={entry.get('twenty_step_reference_present', False)}")
    for test, entry in supplementary.items():
        summary.append(f"- supplementary (not part of verdict) {test}: "
                       f"{entry['classification']}")
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

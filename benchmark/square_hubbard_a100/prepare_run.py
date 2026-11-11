#!/usr/bin/env python3
"""Assemble a run directory for one square-lattice Hubbard benchmark case.

Copies ``inputs/<CASE>/``, generates ``greenone.def``/``greentwo.def`` with
``generate_green_defs.py`` and checks every file against ``cases.json``
(SHA-256 of the inputs the C-mVMC reference run executed). Then, optionally,
replaces sampling keys in ``modpara.def`` and writes ``run_manifest.json``
with the original and final values and hashes.

Usage:
    uv run --no-project python prepare_run.py --list
    uv run --no-project python prepare_run.py CASE OUTDIR \
        [--walkers W] [--windows K] [--warmup N] [--interval N] [--seed S]

``--walkers`` sets ``NVMCSample`` and ``--windows`` sets ``NDataQtySmp``.
The resident GPU PhysCal uses these as its walker and window counts.
``--warmup``/``--interval`` set ``NVMCWarmUp``/``NVMCInterval`` (in sweeps
of Nsite proposals, as in C). ``--seed`` sets ``RndSeed``. OUTDIR must not
exist. Physical parameters are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from generate_green_defs import build  # noqa: E402

SAMPLING_KEYS = {
    "walkers": "NVMCSample",
    "windows": "NDataQtySmp",
    "warmup": "NVMCWarmUp",
    "interval": "NVMCInterval",
    "seed": "RndSeed",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cases() -> dict[str, dict]:
    doc = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))
    return {case["name"]: case for case in doc["cases"]}


def patch_modpara(path: Path, replacements: dict[str, int]) -> dict[str, list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    changed: dict[str, list[str]] = {}
    out = []
    for line in lines:
        fields = line.split()
        if len(fields) == 2 and fields[0] in replacements:
            key = fields[0]
            changed[key] = [fields[1], str(replacements[key])]
            out.append(f"{key:<15} {replacements[key]}")
        else:
            out.append(line)
    missing = sorted(set(replacements) - set(changed))
    if missing:
        raise SystemExit(f"error: modpara.def has no line for {missing}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("case", nargs="?")
    parser.add_argument("outdir", nargs="?", type=Path)
    parser.add_argument("--list", action="store_true", help="list the case names and exit")
    for option in SAMPLING_KEYS:
        parser.add_argument(f"--{option}", type=int)
    args = parser.parse_args()

    cases = load_cases()
    if args.list:
        for name, case in cases.items():
            ref = case["reference"]
            print(
                f"{name:16s} {case['boundary']:36s} NSPGaussLeg={case['NSPGaussLeg']} "
                f"NMPTrans={case['NMPTrans']:>2} origins={case['measurement_origins']:10s} "
                f"E/N={ref['E_over_N']:.6f}({ref['E_over_N_se']:.1e})"
            )
        return 0
    if args.case is None or args.outdir is None:
        parser.error("CASE and OUTDIR are required (or use --list)")
    if args.case not in cases:
        parser.error(f"unknown case {args.case!r}; see --list")
    if args.outdir.exists():
        parser.error(f"{args.outdir} already exists")
    case = cases[args.case]

    args.outdir.mkdir(parents=True)
    for src in sorted((HERE / "inputs" / args.case).iterdir()):
        shutil.copyfile(src, args.outdir / src.name)
    one, two = build(case["L"], case["measurement_origins"])
    for name, text in (("greenone.def", one), ("greentwo.def", two)):
        (args.outdir / name).write_text(text, encoding="utf-8", newline="\n")

    expected = {**case["files"], **case["generated_files"]}
    present = {p.name for p in args.outdir.iterdir()}
    if present != set(expected):
        raise SystemExit(f"error: file set differs: {sorted(present ^ set(expected))}")
    for name, digest in sorted(expected.items()):
        if sha256(args.outdir / name) != digest:
            raise SystemExit(f"error: {name} does not match cases.json")

    replacements = {
        SAMPLING_KEYS[k]: v for k, v in vars(args).items() if k in SAMPLING_KEYS and v is not None
    }
    changed = patch_modpara(args.outdir / "modpara.def", replacements) if replacements else {}
    manifest = {
        "case": args.case,
        "verified_against_reference_inputs": True,
        "modpara_changes": changed,
        "files_sha256": {p.name: sha256(p) for p in sorted(args.outdir.iterdir())},
    }
    (args.outdir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8"
    )
    print(f"prepared {args.outdir} ({args.case}); modpara changes: {changed or 'none'}")
    print("GPU: mvmc-cuda resident-physcal namelist.def zqp_opt.dat <new-output-dir>")
    print("CPU: mvmc namelist.def zqp_opt.dat")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

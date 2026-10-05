#!/usr/bin/env python3
"""Capture step-1 sampled SR operands from the instrumented C vmc.out (issue #358).

Run inside the Linux x86_64 Dev Container from the repository root:

    uv run --no-project python c_toolbox/sr_operand_dump/capture.py \
        --vmc /tmp/mvmc-sr-dump/build/src/mVMC/vmc.out --out tests/fixtures/c_order_sr_operands

Each case runs one optimization step of the real C executable with the
Rust test's seed (1), NSRCG/NStore as listed, one MPI rank and one thread.
The operands are the post-WeightAverage SROptOO/SROptHO/SROptO (and
SROptO_Store) arrays that C prints before it solves the SR equation, in %.17e.
The step-1 energy is read from zvo_out_001.dat. Cargo never runs this script.
"""
import argparse, os, re, shutil, struct, subprocess, sys, tempfile

CASES = [
    # name, namelist (relative to repo root), cg, store, extra vmc.out args
    ("real", "extern/Julia-mVMC/examples/inputs/heisenberg_chain_real/namelist.def", 1, 0, []),
    ("real", "extern/Julia-mVMC/examples/inputs/heisenberg_chain_real/namelist.def", 0, 0, []),
    ("real", "extern/Julia-mVMC/examples/inputs/heisenberg_chain_real/namelist.def", 0, 1, []),
    ("hubbard", "extern/Julia-mVMC/examples/inputs/hubbard_chain_real/namelist.def", 1, 0, []),
    ("dh2_real", "tests/fixtures/dh2/production_real/namelist.def", 1, 0, []),
    ("dh4_real", "tests/fixtures/dh4/production_dh4_real/namelist.def", 1, 0, []),
    ("dh24_real", "tests/fixtures/dh4/production_dh24_real/namelist.def", 1, 0, []),
    ("pairhop_real",
     "extern/Julia-mVMC/test/integration/reference/hubbard_chain_pairhop_real/inputs/namelist.def", 1, 0, []),
    ("opt_real", "tests/fixtures/opttrans/run_opt_real/namelist.def", 1, 0, ["-o"]),
]


def hexs(values):
    return " ".join(struct.pack(">d", v).hex() for v in values)


def prepare(rundir, namelist, cg, store):
    src = os.path.dirname(namelist)
    lines = []
    for line in open(namelist):
        tokens = line.split()
        if len(tokens) >= 2 and tokens[0] == "ModPara":
            text = open(os.path.join(src, tokens[1])).read()
            for key, value in (("NSROptItrStep", 1), ("NSROptItrSmp", 1), ("NSRCG", cg),
                               ("NStore", store), ("RndSeed", 1)):
                if re.search(rf"^\s*{key}\b", text, re.M):
                    text = re.sub(rf"^\s*{key}\b.*$", f"{key} {value}", text, flags=re.M)
                else:
                    text += f"{key} {value}\n"
            open(os.path.join(rundir, "modpara.def"), "w").write(text)
            lines.append("ModPara modpara.def\n")
        elif len(tokens) >= 2:
            lines.append(f"{tokens[0]} {os.path.normpath(os.path.join(src, tokens[1]))}\n")
        else:
            lines.append(line)
    open(os.path.join(rundir, "namelist.def"), "w").write("".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vmc", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = os.getcwd()
    os.makedirs(args.out, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    for name, namelist, cg, store, extra in CASES:
        mode = "cg" if cg else "direct"
        with tempfile.TemporaryDirectory() as rundir:
            prepare(rundir, os.path.join(root, namelist), cg, store)
            r = subprocess.run(["mpiexec", "-n", "1", os.path.abspath(args.vmc), *extra, "namelist.def"],
                               cwd=rundir, env=env, capture_output=True, text=True)
            if r.returncode != 0:
                sys.exit(f"{name} {mode}: vmc.out failed\n{r.stderr[-500:]}")
            arrays = {}
            for line in r.stderr.splitlines():
                m = re.match(r"DEBUG: (SROpt\w+)\[(\d+)\]=(\S+) \+I\*(\S+)", line)
                if m:
                    arrays.setdefault(m.group(1), []).append(float(m.group(3)))
            energy = float(open(os.path.join(rundir, "output/zvo_out_001.dat")).read().split()[0])
        records = [("energy", [energy])]
        if "SROptOO_real" in arrays and store == 0:
            records.append(("oo", arrays["SROptOO_real"]))
        if store == 1:
            records.append(("o_store", arrays["SROptO_Store_real"]))
        records.append(("ho", arrays["SROptHO_real"]))
        records.append(("o", arrays["SROptO_real"]))
        path = os.path.join(args.out, f"{name}-{mode}-store{store}.txt")
        with open(path, "w") as f:
            f.write(f"# native C vmc.out step-1 operands, case={name} mode={mode} NStore={store} seed=1\n")
            f.write("# name hex(IEEE-754 binary64, big-endian) ...; see PROVENANCE.md\n")
            for key, values in records:
                f.write(f"{key} {hexs(values)}\n")
        print(path, {k: len(v) for k, v in records})


if __name__ == "__main__":
    main()

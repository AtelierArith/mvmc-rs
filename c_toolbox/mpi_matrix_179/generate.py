#!/usr/bin/env python3
"""Generate the native-C rank-wise MPI matrix fixtures of issue #179.

Explicit developer command; never invoked by Cargo or Rust tests. Run inside the
Linux x86_64 Dev Container from the repository root:

    c_toolbox/mpi_matrix_179/build.sh /tmp/mvmc-179-c
    uv run --no-project python c_toolbox/mpi_matrix_179/generate.py \
        --vmc /tmp/mvmc-179-c/build/src/mVMC/vmc.out --work /tmp/mvmc-179-runs \
        --out tests/fixtures/mpi_matrix_179

Every cell runs ONE optimization step (NSROptItrStep = NSROptItrSmp = 1, seed from
the input, no initial-parameter file) of the instrumented C executable at the given
world size / NSplitSize and stores, per rank, the sampler state after the step
(Counter[], every saved EleIdx, the full SFMT state) and, from rank 0, the reduced
SR operands (SROptOO/HO/O in %.17e) and the step energy. See README.md there.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

INPUTS = "tests/fixtures/physcal_181"

# model id: (physcal_181 model dir, extra modpara overrides, -o flag)
MODELS = {
    "real": ("heisenberg_chain_real", {}, False),
    "cmp": ("heisenberg_chain_cmp", {}, False),
    "fsz1": ("heisenberg_chain_fsz", {}, False),
    "fsz2": ("heisenberg_chain_fsz", {"NMPTrans": 2}, False),
    "ot": ("hubbard_chain_dh_opttrans", {}, True),
}
SOLVERS = {"d0": {"NStore": 0, "NSRCG": 0}, "d1": {"NStore": 1, "NSRCG": 0},
           "cg": {"NStore": 0, "NSRCG": 1}}


def cells():
    out = []
    for model in MODELS:
        for solver in SOLVERS:
            for ranks, widths in ((2, (1, 2)), (4, (1, 2, 4))):
                for width in widths:
                    out.append((model, solver, ranks, width, 7))
    # Uneven / empty work: fewer saved samples than ranks (4 ranks, 3 samples), and
    # NQPFull = 1 with a width-4 group (three ranks own an empty QP range).
    for model in ("real", "fsz1"):
        for ranks, width in ((2, 2), (4, 1), (4, 2), (4, 4)):
            out.append((model, "d0", ranks, width, 3))
    return out


def cell_id(model, solver, ranks, width, samples):
    return f"{model}-{solver}-r{ranks}w{width}-s{samples}"


def write_modpara(path, overrides):
    lines = open(path).read().splitlines()
    for key, value in overrides.items():
        for i, line in enumerate(lines):
            if line.split()[:1] == [key]:
                lines[i] = f"{key} {value}"
                break
        else:
            raise SystemExit(f"{key} missing from {path}")
    open(path, "w").write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vmc", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mpiexec", default="/opt/mpich/bin/mpiexec")
    args = ap.parse_args()
    root = os.getcwd()
    shutil.rmtree(args.work, ignore_errors=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    index = []
    for model, solver, ranks, width, samples in cells():
        cid = cell_id(model, solver, ranks, width, samples)
        directory, extra, opttrans = MODELS[model]
        run = os.path.join(args.work, cid)
        os.makedirs(run)
        for name in os.listdir(f"{root}/{INPUTS}/{directory}/inputs"):
            shutil.copy(f"{root}/{INPUTS}/{directory}/inputs/{name}", run)
        overrides = {"NVMCCalMode": 0, "NSROptItrStep": 1, "NSROptItrSmp": 1,
                     "NVMCSample": samples, "NSplitSize": width, **SOLVERS[solver], **extra}
        write_modpara(f"{run}/modpara.def", overrides)
        cmd = [args.mpiexec, "-n", str(ranks), os.path.abspath(args.vmc)]
        cmd += (["-o"] if opttrans else []) + ["-e", "namelist.def"]
        r = subprocess.run(cmd, cwd=run, env=env, capture_output=True, text=True,
                           stdin=subprocess.DEVNULL, timeout=300)
        open(f"{run}/stdout.log", "w").write(r.stdout)
        open(f"{run}/stderr.log", "w").write(r.stderr)
        if r.returncode != 0:
            sys.exit(f"{cid}: vmc.out failed\n{r.stderr[-800:]}")
        dest = f"{args.out}/{cid}"
        shutil.rmtree(dest, ignore_errors=True)
        os.makedirs(dest)
        for rank in range(ranks):
            shutil.copy(f"{run}/mvmc179_rank{rank}.txt", f"{dest}/rank{rank}.txt")
        arrays = {}
        for line in r.stderr.splitlines():
            m = re.match(r"DEBUG: SROpt(OO|HO|O)(?:_real)?\[(\d+)\]=(\S+) \+I\*(\S+)", line)
            if m:
                arrays.setdefault(m.group(1).lower(), []).append((m.group(3), m.group(4)))
        energy = open(f"{run}/output/zvo_out_001.dat").read().split()[0]
        with open(f"{dest}/operands.txt", "w") as f:
            f.write(f"energy {energy}\n")
            # Grouped SR-CG reads unwritten malloc memory for <OO> (vmccal.c:314-318):
            # the values differ between identical runs (denormal garbage), so they
            # are not stored.
            undefined_oo = solver == "cg" and width > 1
            for name in ("ho", "o") if undefined_oo else ("ho", "o", "oo"):
                f.write(name + " " + " ".join(f"{a} {b}" for a, b in arrays[name]) + "\n")
        sr_error = int("Error: StcOpt" in r.stderr)
        ov = ",".join(f"{k}={v}" for k, v in overrides.items())
        index.append(f"{cid} {directory} {int(opttrans)} {model} {solver} {ranks} {width} {samples} {sr_error} {ov}")
    with open(f"{args.out}/cells.txt", "w") as f:
        f.write("# id physcal_181-model opttrans model solver ranks width samples c_sr_error overrides(k=v,...)\n")
        f.write("\n".join(index) + "\n")
    import hashlib
    import platform

    def sha(path):
        return hashlib.sha256(open(path, "rb").read()).hexdigest()

    def out_of(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True).stdout.splitlines()[0]
        except Exception:
            return "unknown"

    with open(f"{args.out}/PROVENANCE.txt", "w") as f:
        f.write("generator=c_toolbox/mpi_matrix_179/generate.py (build.sh builds the executable)\n")
        f.write(f"source=extern/mVMC-1.3.0 submodule commit {os.environ.get('SOURCE_COMMIT', 'unknown')}\n")
        f.write(f"vmc_out_sha256={sha(args.vmc)}\n")
        for patch in ("sr_operand_dump/dump_sr_operands.patch", "mpi_matrix_179/rank_state.patch"):
            f.write(f"patch_sha256 {patch} {sha(os.path.join(root, 'c_toolbox', patch))}\n")
        f.write("cmake_c_flags=-D_DEBUG_DUMP_SROPTOO -D_MVMC179_RANKSTATE (Release)\n")
        f.write(f"arch={platform.machine()} kernel={platform.platform()}\n")
        f.write(f"gcc={out_of(['gcc', '--version'])}\n")
        f.write(f"mpich={out_of([os.path.join(os.path.dirname(args.mpiexec), 'mpichversion')])}\n")
        f.write("blas=system libopenblas.so.0 (Ubuntu 24.04 openblas 0.3.26)\n")
        f.write("threads=OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\n")
        f.write(f"cells={len(index)}; per cell: one SR step, seed from the input, no initial-parameter file\n")
    print(f"{len(index)} cells")


if __name__ == "__main__":
    main()

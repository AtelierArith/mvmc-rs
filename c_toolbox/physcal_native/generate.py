#!/usr/bin/env python3
"""Generate native C vmc.out PhysCal/Lanczos references (issue #181).

Run inside the Linux x86_64 Dev Container from the repository root, after
`c_toolbox/physcal_native/build.sh /tmp/mvmc-physcal-native`:

    python3 c_toolbox/physcal_native/generate.py \
        --vmc /tmp/mvmc-physcal-native/build/src/mVMC/vmc.out \
        --out tests/fixtures/native_c_physcal_181 [--only NAME ...]

`tests/fixtures/native_c_physcal_181/scenarios.tsv` lists every scenario. For
each one the inputs and fixed `zqp_opt.dat` are copied from the listed source
directory, modpara overrides are applied, and the UNMODIFIED C executable is run
one MPI rank / one thread with `namelist.def zqp_opt.dat`. Stages (separated by
`;`) run in the same working directory, so the final `output/` directory shows
C's truncate/append behaviour across reruns. Cargo never runs this script;
the checked-in `expected/` files are the only inputs of the Rust tests.
`zvo_CalcTimer.dat` and `zvo_time_*.dat` carry wall-clock data and are kept only
as a name inventory (`time-files.txt`), their bodies are not stored.
"""
import argparse, hashlib, math, os, platform, re, shutil, subprocess, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def apply_overrides(text, overrides):
    for key, value in overrides:
        if re.search(rf"^\s*{key}\b", text, re.M):
            text = re.sub(rf"^\s*{key}\b.*$", f"{key} {value}", text, flags=re.M)
        else:
            text = text.rstrip("\n") + f"\n{key} {value}\n"
    return text


def perturb_zqp(path, amplitude, complex_mode):
    """Deterministically perturb the variational triples (re, im, err) of a fixed
    parameter file; the 6-number header is kept. Used where the original fixed
    state is an exact eigenstate (zero energy variance), for which the Lanczos
    alpha is singular."""
    tokens = open(path).read().split()
    head, body = tokens[:6], tokens[6:]
    assert len(body) % 3 == 0
    out = list(head)
    for k in range(len(body) // 3):
        re_, im_, err = body[3 * k : 3 * k + 3]
        re_ = float(re_) + amplitude * math.sin(1.3 * k + 0.4)
        im_ = float(im_) + (amplitude * 0.5 * math.sin(0.9 * k + 0.1) if complex_mode else 0.0)
        out += [f"{re_: .18e}", f"{im_: .18e}", err]
    open(path, "w").write("  ".join(out) + "\n")


def parse_stage(stage):
    return [tuple(item.split("=", 1)) for item in stage.split(",") if item]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vmc", required=True)
    ap.add_argument("--vmc-dump", help="probe build (build.sh ... dump); adds native-state/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mvmc-commit", default="unknown", help="extern/mVMC-1.3.0 submodule commit (from the host)")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    out_root = os.path.abspath(args.out)
    rows = [
        line.rstrip("\n").split("\t")
        for line in open(os.path.join(out_root, "scenarios.tsv"))
        if line.strip() and not line.startswith("#")
    ]
    vmc = os.path.abspath(args.vmc)
    vmc_dump = os.path.abspath(args.vmc_dump) if args.vmc_dump else None
    for name, source, mode, opttrans, stages, perturb, klass, *rest in rows:
        variant = rest[0] if rest else ""
        if args.only and name not in args.only:
            continue
        src = os.path.join(ROOT, source)
        dest = os.path.join(out_root, name)
        shutil.rmtree(dest, ignore_errors=True)
        os.makedirs(os.path.join(dest, "inputs"))
        for entry in sorted(os.listdir(os.path.join(src, "inputs"))):
            p = os.path.join(src, "inputs", entry)
            if os.path.isfile(p):
                shutil.copy(p, os.path.join(dest, "inputs", entry))
        shutil.copy(os.path.join(src, "zqp_opt.dat"), os.path.join(dest, "zqp_opt.dat"))
        ranks = 1
        for item in filter(None, variant.split(";")):
            key, _, value = item.partition("=")
            if key == "ranks":  # MPI ranks of the measured run (zqp=c_opt always uses one)
                ranks = int(value)
                continue
            if key == "drop":  # remove namelist keywords (e.g. DH4,InDH4)
                path = os.path.join(dest, "inputs", "namelist.def")
                kept = [l for l in open(path) if (l.split() or [""])[0] not in value.split(",")]
                open(path, "w").write("".join(kept))
            elif key == "zero_in":  # zero the value columns of overlay files (e.g. inrbm.def)
                for fname in value.split(","):
                    path = os.path.join(dest, "inputs", fname)
                    lines = open(path).read().split("\n")
                    out_lines = lines[:5] + [
                        (f"{l.split()[0]} 0.0 0.0" if l.split() else l) for l in lines[5:]]
                    open(path, "w").write("\n".join(out_lines))
            elif key == "zqp" and value == "c_opt":
                # fixed parameters produced by one native-C optimization step (seed 1)
                tmp = os.path.join("/tmp", f"physcal-native-opt-{name}")
                shutil.rmtree(tmp, ignore_errors=True)
                os.makedirs(tmp)
                for entry in os.listdir(os.path.join(dest, "inputs")):
                    shutil.copy(os.path.join(dest, "inputs", entry), os.path.join(tmp, entry))
                mp = os.path.join(tmp, "modpara.def")
                text = apply_overrides(open(mp).read(), [
                    ("NVMCCalMode", "0"), ("NLanczosMode", "0"), ("NSROptItrStep", "1"), ("NSROptItrSmp", "1")])
                open(mp, "w").write(text)
                cmd = ["/opt/mpich/bin/mpiexec", "-n", "1", vmc]
                if opttrans == "1":
                    cmd.append("-o")
                cmd += ["namelist.def"]
                res = subprocess.run(cmd, cwd=tmp, env=dict(os.environ, OMP_NUM_THREADS="1"), capture_output=True, text=True)
                assert res.returncode == 0, res.stderr
                # the *_var_* line has the (re, im, err) triple layout that ReadInitParameter reads
                var = sorted(f for f in os.listdir(os.path.join(tmp, "output")) if f.startswith("zvo_var_"))[0]
                shutil.copy(os.path.join(tmp, "output", var), os.path.join(dest, "zqp_opt.dat"))
        if float(perturb) != 0.0:
            perturb_zqp(os.path.join(dest, "zqp_opt.dat"), float(perturb), mode == "cmp")
        log = []
        works = {}
        for tag, binary in (("plain", vmc), ("dump", vmc_dump)):
            if binary is None:
                continue
            work = os.path.join("/tmp", f"physcal-native-{tag}-{name}")
            shutil.rmtree(work, ignore_errors=True)
            os.makedirs(work)
            works[tag] = work
            for index, stage in enumerate(stages.split(";")):
                for entry in os.listdir(os.path.join(dest, "inputs")):
                    shutil.copy(os.path.join(dest, "inputs", entry), os.path.join(work, entry))
                shutil.copy(os.path.join(dest, "zqp_opt.dat"), os.path.join(work, "zqp_opt.dat"))
                for entry in os.listdir(work):
                    if entry.startswith("state_dump_"):  # keep only the final stage's frames
                        os.remove(os.path.join(work, entry))
                mp = os.path.join(work, "modpara.def")
                if index > 0:
                    open(mp, "w").write(open(os.path.join(dest, "inputs", "modpara.def")).read())
                text = apply_overrides(open(mp).read(), parse_stage(stage))
                open(mp, "w").write(text)
                if tag == "plain":
                    if index == 0:
                        open(os.path.join(dest, "inputs", "modpara.def"), "w").write(text)
                    else:
                        open(os.path.join(dest, f"modpara-stage{index}.def"), "w").write(text)
                cmd = ["/opt/mpich/bin/mpiexec", "-n", str(ranks), binary]
                if opttrans == "1":
                    cmd.append("-o")
                cmd += ["namelist.def", "zqp_opt.dat"]
                env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
                res = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True)
                log.append(f"{tag} stage {index} overrides [{stage}] exit {res.returncode}")
                if res.returncode != 0:
                    log.append(res.stdout[-2000:] + res.stderr[-2000:])
        work = works["plain"]
        exp = os.path.join(dest, "expected")
        os.makedirs(exp)
        timefiles = []
        for entry in sorted(os.listdir(os.path.join(work, "output"))):
            p = os.path.join(work, "output", entry)
            if entry == "zvo_CalcTimer.dat" or entry.startswith("zvo_time_"):
                timefiles.append(entry)
                if entry.startswith("zvo_time_"):
                    # keep the deterministic columns; the trailing ctime() stamp is wall-clock
                    os.makedirs(os.path.join(dest, "time-rows"), exist_ok=True)
                    rows_ = [l.rsplit(": ", 1)[0] for l in open(p).read().splitlines()]
                    open(os.path.join(dest, "time-rows", entry), "w").write("\n".join(rows_) + "\n")
            else:
                shutil.copy(p, os.path.join(exp, entry))
        open(os.path.join(dest, "time-files.txt"), "w").write("\n".join(timefiles) + "\n")
        if "dump" in works:
            # the probe build must reproduce the unmodified build's outputs exactly
            for entry in sorted(os.listdir(os.path.join(work, "output"))):
                if entry == "zvo_CalcTimer.dat" or entry.startswith("zvo_time_"):
                    continue
                same = open(os.path.join(work, "output", entry), "rb").read() == open(
                    os.path.join(works["dump"], "output", entry), "rb"
                ).read()
                log.append(f"probe output identical {entry} {same}")
                assert same, f"{name}/{entry}: probe build differs from unmodified build"
            nat = os.path.join(dest, "native-state")
            os.makedirs(nat)
            for entry in sorted(os.listdir(works["dump"])):
                if entry.startswith("state_dump_"):
                    shutil.copy(os.path.join(works["dump"], entry), os.path.join(nat, entry))
        prov = [
            f"scenario {name}",
            f"source {source}",
            f"rust_mode {mode} opttrans_flag {opttrans}",
            f"stages {stages}",
            f"zqp_perturb {perturb} class {klass} variant [{variant}]",
            f"vmc.out sha256 {sha(vmc)}",
            f"probe vmc.out sha256 {sha(vmc_dump) if vmc_dump else 'none'}",
            f"extern/mVMC-1.3.0 commit {args.mvmc_commit}",
            *[f"source sha256 {rel} {sha(os.path.join(ROOT, 'extern/mVMC-1.3.0/src', rel))}"
              for rel in ("mVMC/vmcmain.c", "mVMC/physcal_lanczos.c", "mVMC/vmcmake.c", "mVMC/vmcmake_fsz.c", "sfmt/SFMT.c")],
            f"host {platform.machine()} {platform.platform()}",
            "gcc " + subprocess.run(["gcc", "--version"], capture_output=True, text=True).stdout.splitlines()[0],
            "mpi one rank, OMP_NUM_THREADS=1, build Release (CMake default flags), unmodified source",
        ] + log
        open(os.path.join(dest, "provenance.txt"), "w").write("\n".join(prov) + "\n")
        print(name, "ok" if all("exit 0" in l or "stage" not in l.split(" overrides")[0] for l in log) else "FAILED")


if __name__ == "__main__":
    main()

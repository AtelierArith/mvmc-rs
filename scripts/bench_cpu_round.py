#!/usr/bin/env python3
"""CPU performance round (issue #448): Rust vs native C on representative inputs.

Runs each workload with interleaved repetitions (Rust, C, Rust, C, ...), one thread on both
sides unless `--inner N`, and records wall time, the C-compatible section timers
(`zvo_CalcTimer.dat`: [3] VMCMakeSample, [4] VMCMainCal, [5] ... [8] SR, ...), the host load
average before and after every run and a metadata block.

Usage (from the repository root):
    cargo build --release -p mvmc-cli
    c_toolbox/perf_448/build_c.sh /tmp/c448          # unpatched native C vmc.out (Dev Container)
    uv run --no-project python scripts/bench_cpu_round.py --c-dir /tmp/c448 --reps 3 \
        --out benchmark/cpu_round/results/cpu_round.csv

Native C runs inside the Dev Container image (mpich, `mpirun -np 1`, `OMP_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`); the Rust binary runs on the host with `OPENBLAS_NUM_THREADS=1`.
"""
import argparse
import csv
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
P181 = ROOT / "tests/fixtures/physcal_181"
HUB = ROOT / "benchmark/hubbard_chain/inputs"
PHY = ROOT / "benchmark/physcal/inputs"

# name: (input dir, overrides, mode flags, optpara file or None)
OPT = {"NVMCCalMode": "0", "NLanczosMode": "0"}


def phys(extra=None):
    d = {"NVMCCalMode": "1"}
    d.update(extra or {})
    return d


def opt(steps, samples, extra=None):
    d = dict(OPT, NSROptItrStep=str(steps), NSROptItrSmp=str(min(steps, 10)),
             NVMCSample=str(samples), NDataQtySmp="1")
    d.update(extra or {})
    return d


WORKLOADS = {
    # optimization (NVMCCalMode=0)
    "opt_hubbard_L16": (HUB / "hubbard_chain_L16", opt(40, 300), [], None),
    "opt_hubbard_L32": (HUB / "hubbard_chain_L32", opt(30, 300), [], None),
    "opt_hubbard_L64": (HUB / "hubbard_chain_L64", opt(10, 300), [], None),
    "opt_heisenberg_real": (P181 / "heisenberg_chain_real/inputs", opt(100, 2000), [], None),
    "opt_heisenberg_cmp": (P181 / "heisenberg_chain_cmp/inputs", opt(100, 2000), [], None),
    "opt_heisenberg_fsz": (P181 / "heisenberg_chain_fsz/inputs", opt(100, 2000), [], None),
    "opt_kondo_real": (P181 / "kondo_chain_real/inputs", opt(100, 2000), [], None),
    "opt_hubbard_dh": (P181 / "hubbard_chain_dh_real/inputs", opt(100, 2000), [], None),
    "opt_hubbard_rbm_opttrans": (P181 / "hubbard_chain_dh_rbm_opttrans/inputs",
                                 opt(60, 1000), ["-o"], None),
    # PhysCal (NVMCCalMode=1, fixed parameters)
    "phys_hubbard_L16": (PHY / "hubbard_chain_L16", phys(), [], "zqp_opt.dat"),
    "phys_hubbard_L32": (PHY / "hubbard_chain_L32", phys(), [], "zqp_opt.dat"),
    "phys_heisenberg_real": (P181 / "heisenberg_chain_real/inputs",
                             phys({"NVMCSample": "20000", "NDataQtySmp": "20"}), [],
                             "../zqp_opt.dat"),
    "phys_lanczos": (P181 / "hubbard_all_terms_lanczos1/inputs",
                     phys({"NVMCSample": "20000", "NDataQtySmp": "20"}), [],
                     "../zqp_opt.dat"),
}


def patch_modpara(path: Path, overrides: dict):
    lines = path.read_text().splitlines()
    out = []
    seen = set()
    for line in lines:
        key = line.split()[0] if line.split() else ""
        if key in overrides:
            out.append(f"{key} {overrides[key]}")
            seen.add(key)
        else:
            out.append(line)
    for key, value in overrides.items():
        if key not in seen:
            out.append(f"{key} {value}")
    path.write_text("\n".join(out) + "\n")


def prepare(name, workdir: Path):
    src, overrides, flags, optpara = WORKLOADS[name]
    dest = workdir / name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    patch_modpara(dest / "modpara.def", overrides)
    para = None
    if optpara:
        p = (src / optpara).resolve()
        if not p.exists():
            sys.exit(f"missing parameter file {p}")
        shutil.copy(p, dest / "zqp_opt.dat")
        para = "zqp_opt.dat"
    return dest, flags, para


def parse_timer(path: Path):
    """Return {section id: seconds} from zvo_CalcTimer.dat (lines `[id] name : seconds`)."""
    secs = {}
    path = path.parent / "output" / path.name
    if not path.exists():
        return secs
    for line in path.read_text().splitlines():
        m = re.search(r"\[(\d+)\]\s+([0-9.eE+-]+)\s*$", line)
        if m:
            secs[int(m.group(1))] = float(m.group(2))
    return secs


def load():
    return os.getloadavg()[0]


def run_rust(binary, dest: Path, flags, para, inner, env_extra):
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MVMC_C_TIMER="1")
    env.update(env_extra)
    if inner > 1:
        env["MVMC_RS_INNER_THREADS"] = str(inner)
    else:
        env.pop("MVMC_RS_INNER_THREADS", None)
    shutil.rmtree(dest / "output", ignore_errors=True)
    for f in dest.glob("zvo_*"):
        f.unlink()
    cmd = [str(binary), *flags, "namelist.def"] + ([para] if para else [])
    t0 = time.perf_counter()
    subprocess.run(cmd, cwd=dest, env=env, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    return time.perf_counter() - t0, parse_timer(dest / "zvo_CalcTimer.dat")


def run_c(c_dir: Path, image, dest: Path, flags, para, threads):
    shutil.rmtree(dest / "output", ignore_errors=True)
    for f in dest.glob("zvo_*"):
        f.unlink()
    inner = (f"cd /work && s=$(date +%s.%N) && OMP_NUM_THREADS={threads} OPENBLAS_NUM_THREADS=1 "
             f"/opt/mpich/bin/mpirun -np 1 /c/build/src/mVMC/vmc.out {' '.join(flags)} namelist.def "
             f"{para or ''} > /dev/null 2>&1; e=$(date +%s.%N); python3 -c \"print('CWALL', $e - $s)\"")
    out = subprocess.run(
        ["docker", "run", "--rm", "-v", f"{dest}:/work", "-v", f"{c_dir}:/c:ro", "-u",
         f"{os.getuid()}:{os.getgid()}", image, "bash", "-lc", inner],
        check=True, capture_output=True, text=True).stdout
    m = re.search(r"CWALL\s+([0-9.]+)", out)
    return float(m.group(1)), parse_timer(dest / "zvo_CalcTimer.dat")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rust", default=str(ROOT / "target/release/mvmc"))
    ap.add_argument("--c-dir", default="/tmp/claude-1000/c448")
    ap.add_argument("--image", default="vsc-mvmc-rs-741feed754ed4827a429135e5fcc0e7623f1f847d228698b066da094a8819e32-uid:latest")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--inner", type=int, default=1, help="Rust inner threads / C OMP threads")
    ap.add_argument("--workloads", default="all")
    ap.add_argument("--no-c", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    ap.add_argument("--rust-env", action="append", default=[], help="KEY=VALUE for the Rust run")
    args = ap.parse_args()
    names = list(WORKLOADS) if args.workloads == "all" else args.workloads.split(",")
    env_extra = dict(kv.split("=", 1) for kv in args.rust_env)
    work = Path(tempfile.mkdtemp(prefix="bench448-", dir="/tmp/claude-1000"))
    meta = {
        "cpu": next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo")
                     if l.startswith("model name")), "unknown"),
        "threads_available": os.cpu_count(),
        "os": platform.platform(),
        "inner": args.inner,
        "load_start": f"{load():.2f}",
    }
    rows = []
    for name in names:
        dest, flags, para = prepare(name, work)
        for rep in range(args.reps + 1):  # first repetition is a warm-up
            for impl in ([] if False else ["rust"] + ([] if args.no_c else ["c"])):
                l0 = load()
                if impl == "rust":
                    wall, secs = run_rust(args.rust, dest, flags, para, args.inner, env_extra)
                else:
                    wall, secs = run_c(Path(args.c_dir), args.image, dest, flags, para,
                                       args.inner)
                l1 = load()
                if rep > 0:
                    rows.append({"workload": name, "impl": impl, "rep": rep, "wall_s": wall,
                                 "sec3_makesample": secs.get(3), "sec4_maincal": secs.get(4),
                                 "sec2_total": secs.get(0), "load_before": l0, "load_after": l1,
                                 "tag": args.tag})
                    print(f"{name:28s} {impl:5s} rep{rep} {wall:8.3f}s  load {l0:.1f}->{l1:.1f}",
                          flush=True)
    meta["load_end"] = f"{load():.2f}"
    # summary
    print("\n=== medians ===")
    summary = []
    for name in names:
        row = {"workload": name}
        for impl in ("rust", "c"):
            walls = [r["wall_s"] for r in rows if r["workload"] == name and r["impl"] == impl]
            if walls:
                row[impl] = statistics.median(walls)
        if "rust" in row and "c" in row:
            row["c_over_rust"] = row["c"] / row["rust"]
        summary.append(row)
        print(row)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        (out.with_suffix(".meta")).write_text(
            "\n".join(f"{k}={v}" for k, v in meta.items()) + "\n")
    shutil.rmtree(work, ignore_errors=True)
    print(meta)


if __name__ == "__main__":
    main()

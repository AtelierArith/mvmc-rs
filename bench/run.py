"""One-command native C / Julia / Rust Opt and PhysCal MPI comparison."""
import argparse
import csv
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from bench_mpi import prepare_project
from bench_cpu_round import patch_modpara, parse_timer


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(command, log, env, timeout, cwd=ROOT):
    print(f"Running: {' '.join(map(str, command))}", file=sys.stderr, flush=True)
    with log.open("w") as stream:
        process = subprocess.Popen(list(map(str, command)), cwd=cwd, env=env,
                                   stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    if code:
        raise RuntimeError(f"command failed ({code}); see {log}")
    return log.read_text()


def validate_metadata(text, ranks, threads, native=False):
    for label, value in (("WORLD", ranks), ("THREADS", threads), ("BLAS_THREADS", 1)):
        actual = re.findall(rf"^{label} (\d+) (\d+)$", text, re.M)
        if sorted((int(a), int(b)) for a, b in actual) != [(r, value) for r in range(ranks)]:
            raise ValueError(f"invalid {label} evidence")
    if native:
        actual = re.findall(r"^NATIVE_BLAS_THREADS (\d+) (\d+)$", text, re.M)
        if sorted((int(a), int(b)) for a, b in actual) != [(r, 1) for r in range(ranks)]:
            raise ValueError("native BLAS must use one thread per rank")


def measurements(text, ranks, threads, reps, groups=None, native=False):
    validate_metadata(text, ranks, threads, native)
    values = [(int(rep), float(seconds), float(result)) for rep, seconds, result
              in re.findall(r"^BENCH (\d+) (\S+) (\S+)$", text, re.M)]
    if [rep for rep, _, _ in values] != list(range(1, reps + 1)):
        raise ValueError("missing/duplicate benchmark repetitions")
    if any(t <= 0 or not math.isfinite(t) or not math.isfinite(result)
           or (groups is not None and result != groups) for _, t, result in values):
        raise ValueError("invalid timing or incomplete PhysCal groups")
    return values


def report(rows, args):
    lines = ["# C / Julia / Rust MPI benchmark", "",
             "Periodic half-filled Hubbard chain: t=1, U=4, Lsub=4, NSPGaussLeg=8.",
             f"MPI ranks: {args.ranks}; compute threads/rank: {args.threads}; BLAS threads: 1.",
             f"Saved configurations: {args.samples} total per step/group ({args.samples // args.ranks}/rank).",
             f"Opt: {args.steps} SR steps; PhysCal: {args.groups} measurement groups.",
             f"Warmups: {args.warmups}; measured repetitions: {args.reps}; times are medians in seconds.",
             "PhysCal uses the same C-generated parameter file for all implementations.", "",
             "| Workload | Sites | C (s) | Julia (s) | Rust (s) | Julia/C | Rust/C |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for mode in ("Opt", "PhysCal"):
        for size in args.sites:
            times = []
            for impl in ("C", "Julia", "Rust"):
                values = [row["seconds"] for row in rows
                          if (row["workload"], row["sites"], row["implementation"]) == (mode, size, impl)]
                if len(values) != args.reps:
                    raise ValueError(f"incomplete report cell: {mode} L{size} {impl}")
                times.append(statistics.median(values))
            c, j, r = times
            lines.append(f"| {mode} | {size} | {c:.6f} | {j:.6f} | {r:.6f} | {j/c:.3f} | {r/c:.3f} |")
    lines += ["", "C: fresh-process rank-zero internal All timer, after MPI_Init.",
              "Julia/Rust: warmed production API maximum over ranks, including parsing,",
              "initialization/parameter loading, sampling, optimization or observables, and output.",
              "Process startup, JIT, build/setup and warmups are excluded. Timing boundaries differ.",
              "Raw measurements, outputs, input hashes, source hashes, versions and linked libraries",
              "are retained beside this report. Timing runs are sequential. Energies are not parity proof."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", type=int, nargs="+", choices=(16, 24, 32, 64), default=[32, 64])
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--groups", type=int, default=100)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--output", type=Path, default=ROOT / "bench-out" / time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--mpi-prefix", type=Path, default=Path("/opt/mpich"))
    parser.add_argument("--julia-bin", type=Path, default=Path.home() / ".cache/mvmc/tools/julia-1.13.1/bin/julia")
    parser.add_argument("--julia-source", type=Path, default=ROOT / "extern/Julia-mVMC")
    args = parser.parse_args()
    if min(args.ranks, args.threads, args.steps, args.groups, args.samples, args.reps,
           args.warmups, args.timeout) < 1 or args.samples % args.ranks:
        parser.error("positive counts required; total samples must be divisible by ranks")
    if len(set(args.sites)) != len(args.sites):
        parser.error("site counts must be unique")
    out = args.output.resolve()
    if out.exists():
        parser.error("output directory must be new")
    prefix = args.mpi_prefix.resolve()
    for tool in (prefix / "bin/mpiexec", prefix / "bin/mpicc", args.julia_bin):
        if not tool.is_file():
            parser.error(f"missing {tool}; use the Linux Dev Container and .devcontainer/install-julia.sh")
    out.mkdir(parents=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", BLIS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OMP_NUM_THREADS="1", JULIA_NUM_THREADS=f"{args.threads},0", JULIA_NUM_GC_THREADS="1",
               MVMC_RS_INNER_THREADS=str(args.threads), JULIA_MVMC_MPI="1",
               JULIA_MVMC_INNER_THREADS="1", JULIA_MVMC_PFAPACK_THREADS="0",
               MPICC=str(prefix / "bin/mpicc"), MVMC_BLAS_PROVIDER="openblas",
               MVMC_RS_SR_BACKEND="c-order", MVMC_RS_MEASURE_PF_BACKEND="c-order",
               UCX_ERROR_SIGNALS="SIGILL,SIGBUS,SIGFPE", UCX_MEMTYPE_CACHE="no")
    env["LD_LIBRARY_PATH"] = str(prefix / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    for key in ("LD_PRELOAD", "MVMC_C_TIMER", "MVMC_CALHAM_DIAGNOSTICS", "MVMC_RS_INNER_PROFILE",
                "MVMC_RS_INNER_THRESHOLD", "MVMC_RS_INNER_MIN_WORK_NS", "MVMC_RS_INNER_MIN_SIZE"):
        env.pop(key, None)
    metadata = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    environment={k: v for k, v in env.items() if k.endswith("_THREADS") or
                                 k.startswith(("MVMC_", "JULIA_", "UCX_"))})
    (out / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for name, command in {"host": ["uname", "-a"], "cpu": ["lscpu"], "rust": ["rustc", "-Vv"],
                          "source": ["git", "rev-parse", "HEAD"], "dirty": ["git", "diff", "HEAD"],
                          "submodules": ["git", "submodule", "status", "--recursive"],
                          "compiler": [prefix / "bin/mpicc", "-show"],
                          "gcc": ["gcc", "--version"], "fortran": ["gfortran", "--version"],
                          "cmake": ["cmake", "--version"],
                          "mpi": [prefix / "bin/mpichversion"], "blas": ["pkg-config", "--modversion", "openblas"]}.items():
        execute(command, out / f"{name}.log", env, args.timeout)
    build = execute(["cargo", "build", "--release", "--locked", "-p", "mvmc-core", "--features", "mpi",
                     "--example", "mpi_benchmark", "--example", "mpi_physcal_benchmark", "--message-format=json"],
                    out / "rust-build.log", env, args.timeout)
    binaries = {item["target"]["name"]: Path(item["executable"])
                for line in build.splitlines() if line.startswith("{")
                for item in [json.loads(line)] if item.get("reason") == "compiler-artifact" and item.get("executable")}
    csource = ROOT / "extern/mVMC-1.3.0"
    copied = out / "c-source"
    shutil.copytree(csource, copied, ignore=shutil.ignore_patterns(".git"))
    cbuild = out / "c-build"
    execute(["cmake", "-S", copied, "-B", cbuild, "-DCMAKE_BUILD_TYPE=Release", "-DTesting=OFF",
             "-DCMAKE_C_COMPILER=gcc", "-DCMAKE_CXX_COMPILER=g++", "-DCMAKE_Fortran_COMPILER=gfortran",
             "-DGIT_SUBMODULE_UPDATE=OFF", "-DPFAFFIAN_BLOCKED=OFF", "-DUSE_GEMMT=ON",
             "-DBLA_VENDOR=OpenBLAS", f"-DMPI_C_COMPILER={prefix / 'bin/mpicc'}",
             "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"], out / "c-configure.log", env, args.timeout)
    execute(["cmake", "--build", cbuild, "--target", "vmc.out", "--parallel", "4"], out / "c-build.log", env, args.timeout)
    cbin = cbuild / "src/mVMC/vmc.out"
    observer = out / "world-observer.so"
    execute([prefix / "bin/mpicc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
             "-fopenmp", ROOT / "c_toolbox/mpi_comparison_495/world_observer.c", "-lopenblas", "-o", observer],
            out / "observer-build.log", env, args.timeout)
    with contextlib.redirect_stdout(sys.stderr):
        project = prepare_project(out, args.julia_bin, env, prefix, args.julia_source.resolve())
    hashes = {}
    for name, binary in {**binaries, "C": cbin}.items():
        linkage = execute(["ldd", binary], out / f"{name}-libraries.log", env, args.timeout)
        mpi = re.findall(r"libmpi\.so\S* => (\S+)", linkage)
        if len(mpi) != 1 or not Path(mpi[0]).resolve().is_relative_to(prefix):
            raise RuntimeError(f"{name} uses a different MPI provider")
        hashes[str(binary)] = digest(binary)
    for source in (copied, args.julia_source.resolve()):
        for path in sorted(source.rglob("*")):
            if path.is_file() and ".git" not in path.parts and path.suffix in (".c", ".h", ".jl", ".f", ".f90", ".tcc", ".toml"):
                hashes[str(path)] = digest(path)
    for path in (Path(__file__), ROOT / "bench/run.sh", observer, project / "Manifest-v1.13.toml",
                 project / "LocalPreferences.toml", ROOT / "bench/physcal_worker.jl",
                 ROOT / "benchmark/mpi_comparison/worker.jl",
                 ROOT / "crates/mvmc-core/examples/mpi_benchmark.rs",
                 ROOT / "crates/mvmc-core/examples/mpi_physcal_benchmark.rs",
                 ROOT / "scripts/bench_mpi.py", ROOT / "scripts/bench_cpu_round.py",
                 ROOT / "c_toolbox/mpi_comparison_495/world_observer.c",
                 project / "PfaPack.jl/deps/libltl2inv.so"):
        hashes[str(path)] = digest(path)
    (out / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    rows = []
    launcher = [prefix / "bin/mpiexec", "-n", str(args.ranks)]
    for size in args.sites:
        cell = out / f"L{size}"
        cell.mkdir()
        inputs = cell / "opt-inputs"
        shutil.copytree(ROOT / f"benchmark/hubbard_chain/inputs/hubbard_chain_L{size}", inputs)
        patch_modpara(inputs / "modpara.def", {"NVMCSample": str(args.samples // args.ranks),
                      "NSROptItrStep": str(args.steps), "NSROptItrSmp": str(args.steps), "NVMCCalMode": "0"})
        fixed_params = None
        for mode in ("Opt", "PhysCal"):
            if mode == "PhysCal":
                phys = cell / "physcal-inputs"
                shutil.copytree(inputs, phys)
                patch_modpara(phys / "modpara.def", {"NVMCCalMode": "1", "NDataQtySmp": str(args.groups)})
                shutil.copy2(fixed_params, phys / "zqp_opt.dat")
                inputs = phys
            (cell / f"{mode}-input-sha256.json").write_text(json.dumps({p.name: digest(p) for p in inputs.iterdir() if p.is_file()}, indent=2) + "\n")
            for impl in ("C", "Julia", "Rust"):
                dest = cell / f"{mode}-{impl}"
                dest.mkdir()
                if impl == "C":
                    cenv = dict(env, OMP_NUM_THREADS=str(args.threads), LD_PRELOAD=str(observer))
                    for iteration in range(args.warmups + args.reps):
                        work = dest / f"run-{iteration}"
                        shutil.copytree(inputs, work)
                        (work / "output").mkdir(exist_ok=True)
                        command = launcher + [cbin, "namelist.def"]
                        if mode == "PhysCal":
                            command.append("zqp_opt.dat")
                        text = execute(command, work / "run.log", cenv, args.timeout, cwd=work)
                        # The observer emits the same metadata as the warmed workers.
                        validate_metadata(text, args.ranks, args.threads)
                        seconds = parse_timer(work / "zvo_CalcTimer.dat")[0]
                        if not math.isfinite(seconds) or seconds <= 0:
                            raise ValueError("invalid C All timing")
                        outputs = sorted((work / "output").glob("zvo_out_*.dat"))
                        if len(outputs) != (1 if mode == "Opt" else args.groups):
                            raise ValueError("incomplete C output groups")
                        for path in outputs:
                            records = [[float(v) for v in line.split()] for line in path.read_text().splitlines()]
                            if len(records) != (args.steps if mode == "Opt" else 1) or any(
                                len(record) != 6 or not all(math.isfinite(v) for v in record) for record in records
                            ):
                                raise ValueError(f"incomplete or nonfinite C output: {path}")
                        if iteration >= args.warmups:
                            rows.append(dict(workload=mode, sites=size, implementation=impl,
                                             rep=iteration-args.warmups+1, seconds=seconds))
                        if mode == "Opt" and iteration == args.warmups:
                            fixed_params = work / "output/zqp_opt.dat"
                            if not fixed_params.is_file():
                                raise RuntimeError("C did not generate PhysCal parameters")
                else:
                    count = args.steps if mode == "Opt" else args.groups
                    command = launcher + ([binaries["mpi_benchmark" if mode == "Opt" else "mpi_physcal_benchmark"]]
                        if impl == "Rust" else [args.julia_bin, f"--project={project}", "--startup-file=no",
                            ROOT / ("benchmark/mpi_comparison/worker.jl" if mode == "Opt" else "bench/physcal_worker.jl")])
                    command += [inputs / "namelist.def"]
                    if mode == "PhysCal":
                        command.append(inputs / "zqp_opt.dat")
                    command += [str(count), str(args.warmups), str(args.reps), dest, str(args.ranks)]
                    text = execute(command, dest / "run.log", env, args.timeout)
                    values = measurements(text, args.ranks, args.threads, args.reps,
                                          groups=args.groups if mode == "PhysCal" else None, native=impl == "Julia")
                    if mode == "Opt":
                        for rep, _, _ in values:
                            work = dest / f"run-{args.warmups + rep - 1}"
                            if len((work / "zvo_out.dat").read_text().splitlines()) != args.steps or not (work / "zqp_opt.dat").is_file():
                                raise ValueError(f"incomplete Opt output: {work}")
                    rows += [dict(workload=mode, sites=size, implementation=impl, rep=rep, seconds=seconds)
                             for rep, seconds, _ in values]
                (out / "measurements.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (out / "measurements.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    markdown = report(rows, args)
    (out / "report.md").write_text(markdown)
    print(markdown)
    print(f"Report: {out / 'report.md'}", file=sys.stderr)


if __name__ == "__main__":
    main()

"""Optional developer benchmark, run through uv inside the Dev Container."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def run(command, log, env, timeout=1800):
    print(shlex.join(map(str, command)), flush=True)
    with log.open("w") as stream:
        stream.write(shlex.join(map(str, command)) + "\n")
        stream.flush()
        # Kill the entire launcher/worker group on timeout, not just mpiexec.
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            import signal
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    if code:
        raise RuntimeError(f"command failed ({code}); see {log}")
    return log.read_text()


def parse_measurements(text, ranks, reps, threads=None, require_parallel=True, require_native_blas=False):
    worlds = re.findall(r"^WORLD (\d+) (\d+)$", text, re.MULTILINE)
    if sorted((int(r), int(n)) for r, n in worlds) != [(r, ranks) for r in range(ranks)]:
        raise ValueError("actual MPI ranks/world sizes do not match requested world")
    if threads is not None:
        actual = re.findall(r"^THREADS (\d+) (\d+)$", text, re.MULTILINE)
        if sorted((int(r), int(n)) for r, n in actual) != [(r, threads) for r in range(ranks)]:
            raise ValueError("actual computation thread counts do not match request")
        blas = re.findall(r"^BLAS_THREADS (\d+) (\d+)$", text, re.MULTILINE)
        if sorted((int(r), int(n)) for r, n in blas) != [(r, 1) for r in range(ranks)]:
            raise ValueError("actual BLAS threads are not one per process")
        if require_native_blas:
            native = re.findall(r"^NATIVE_BLAS_THREADS (\d+) (\d+)$", text, re.MULTILINE)
            if sorted((int(r), int(n)) for r, n in native) != [(r, 1) for r in range(ranks)]:
                raise ValueError("native-helper BLAS threads are not one per process")
        evidence = re.findall(r"^EXECUTION (\d+) (\d+) (\d+)$", text, re.MULTILINE)
        if sorted(int(r) for r, _, _ in evidence) != list(range(ranks)):
            raise ValueError("missing actual kernel execution evidence")
        if require_parallel and threads > 1 and any(int(calls) == 0 or int(workers) < 2 for _, calls, workers in evidence):
            raise ValueError("internal kernels did not execute on multiple workers")
    matches = re.findall(r"^BENCH (\d+) (\S+) (\S+)$", text, re.MULTILINE)
    values = [(int(rep), float(seconds), float(energy)) for rep, seconds, energy in matches]
    if [rep for rep, _, _ in values] != list(range(1, reps + 1)):
        raise ValueError("missing or duplicate benchmark repetitions")
    if any(not math.isfinite(t) or t <= 0 or not math.isfinite(e) for _, t, e in values):
        raise ValueError("invalid timing/energy")
    return values


def prepare_project(output, julia, env, prefix, source):
    project = output / "julia-project"
    project.mkdir()
    shutil.copyfile(source / "Project.toml", project / "Project.toml")
    lock = ROOT / "benchmark/julia_comparison/Manifest-v1.13.toml"
    shutil.copyfile(lock, project / "Manifest-v1.13.toml")
    # Preserve the exact lock and its relative local-package paths. Preferences
    # live in the result directory; the reference checkout stays unmodified.
    for package in ("MVMCOptimizers.jl", "MVMCExpertModeParsers.jl", "SFMT.jl", "PfaPack.jl"):
        (project / package).symlink_to(source / package, target_is_directory=True)
    setup = output / "julia-setup.jl"
    setup.write_text('''using Pkg, Libdl
@assert VERSION == v"1.13.1"
Pkg.instantiate()
using MPIPreferences
MPIPreferences.use_system_binary(; library_names=[ARGS[1]], mpiexec=ARGS[2])
root = dirname(Base.active_project())
for (pkg, path) in (("SFMT", joinpath(root, "SFMT.jl", "deps", "sfmt", "libsfmt." * Libdl.dlext)),
                    ("PfaPack", joinpath(root, "PfaPack.jl", "deps", "libltl2inv." * Libdl.dlext)))
    isfile(path) || Pkg.build(pkg)
end
''')
    run([julia, f"--project={project}", "--startup-file=no", setup,
         prefix / "lib/libmpi.so", prefix / "bin/mpiexec"], output / "julia-setup.log", env)
    if (project / "Manifest-v1.13.toml").read_bytes() != lock.read_bytes():
        raise RuntimeError("Julia modified the pinned lock")
    run([julia, f"--project={project}", "--startup-file=no", "-e",
         'using MPI, LinearAlgebra, Libdl, PfaPack; BLAS.set_num_threads(1); '
         'println("Julia: ", VERSION); println("MPI: ", MPI.Get_library_version()); '
         'println("BLAS: ", BLAS.get_config()); println("BLAS threads: ", BLAS.get_num_threads()); '
         'library = first(BLAS.get_config().loaded_libs); '
         'println("BLAS path: ", Libdl.dlpath(library.handle)); '
         'native = Libdl.dlopen(PfaPack.libltl2inv); '
         'native_threads = ccall(Libdl.dlsym(native, :openblas_get_num_threads), Cint, ()); '
         '@assert native_threads == 1; println("Native BLAS threads: ", native_threads); '
         'println("Native BLAS version: ", unsafe_string(ccall(Libdl.dlsym(native, :openblas_get_config), Cstring, ()))); '
         'println("BLAS version: ", unsafe_string(ccall(Libdl.dlsym(library.handle, '
         'Symbol("openblas_get_config", library.suffix)), Cstring, ())))'],
        output / "julia-libraries.log", env)
    return project


def write_report(output, rows, policy, warmups, rust_threshold=None):
    medians = {}
    for row in rows:
        key = (row["model"], row["implementation"], row["ranks"], row.get("threads", 1))
        medians.setdefault(key, []).append(row["seconds"])
    repetitions = sorted({len(values) for values in medians.values()})
    medians = {key: statistics.median(values) for key, values in medians.items()}
    steps = sorted({row["steps"] for row in rows})
    lines = ["# Julia versus Rust MPI benchmark", "",
             f"Sample policy: `{policy}`. SR steps: {', '.join(map(str, steps))}. "
             f"Measured repetitions/cell: {', '.join(map(str, repetitions))}. Warmups: {warmups}.",
             "Times are medians in seconds. Computation threads vary by configuration; BLAS is fixed to one thread per rank.",
             "Rust dispatch: " + ("production automatic work gates." if rust_threshold is None
                                   else f"explicit item threshold {rust_threshold}."),
             "Internal timings use the slowest rank, with barriers before/after the production runner.",
             "They include input parsing, initialization, optimization and output; exclude process startup,",
             "package setup, warmup and separate warmed execution diagnostics (at most 20 steps).",
             "Energies are recorded for diagnosis, not used as parity proof.", "",
             "| Model | Ranks | Threads/rank | Samples/rank/step | Rust seconds | Julia seconds | Julia/Rust | Rust samples/s | Julia samples/s | Rust pooled workers/rank | Julia MVMC threads/rank |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for model, ranks, threads in sorted({(row["model"], row["ranks"], row.get("threads", 1)) for row in rows}):
        rust, julia = (medians[model, impl, ranks, threads] for impl in ("rust", "julia"))
        samples = next(row["samples_per_rank"] for row in rows
                       if row["model"] == model and row["ranks"] == ranks and row.get("threads", 1) == threads)
        steps = next(row["steps"] for row in rows
                     if row["model"] == model and row["ranks"] == ranks and row.get("threads", 1) == threads)
        work = samples * ranks * steps
        workers = []
        for impl in ("rust", "julia"):
            cell = next(row for row in rows if row["model"] == model and row["ranks"] == ranks
                        and row.get("threads", 1) == threads and row["implementation"] == impl)
            workers.append(f"{cell.get('observed_workers_min', '?')}–{cell.get('observed_workers_max', '?')}")
        lines.append(f"| {model} | {ranks} | {threads} | {samples} | {rust:.6f} | {julia:.6f} | {julia/rust:.3f} "
                     f"| {work/rust:.1f} | {work/julia:.1f} | {workers[0]} | {workers[1]} |")
    scaling_rows = []
    for (model, impl, ranks, threads), seconds in sorted(medians.items()):
        if (model, impl, 1, 1) in medians:
            speedup = medians[model, impl, 1, 1] / seconds
            throughput = speedup * (ranks if policy == "per-rank" else 1)
            scaling_rows.append(f"| {model} | {impl} | {ranks}x{threads} | {speedup:.3f} | {throughput:.3f} |")
    if scaling_rows:
        lines.extend(["", "| Model | Implementation | Ranks | Time speedup vs MPI rank 1 | Sample throughput gain |",
                      "|---|---|---:|---:|---:|"] + scaling_rows)
    else:
        lines.extend(["", "A 1-process/1-thread baseline was not measured in this run; scaling ratios are omitted."])
    if policy == "per-rank":
        lines.extend(["", "Per-rank samples are fixed: four ranks process four times as many samples.",
                      "The time ratio is not strong-scaling speedup; throughput counts the extra samples."])
    else:
        lines.extend(["", "Total samples per SR step are fixed. Rank counts change independent chains/seeds,",
                      "warmup work and trajectories; this measures workload scaling, not identical trajectories."])
    lines.extend(["", "Worker ranges are min–max across ranks in the untimed observation.",
                  "Rust counts actual entries on parallel inner-pool workers (zero means serial dispatch);",
                  "Julia counts threads with sampled MVMC frames, including the driver."])
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new result directory")
    parser.add_argument("--model", type=int, choices=(16, 24, 32), action="append")
    parser.add_argument("--layout", nargs="+", help="processes x computation threads, e.g. 1x4 2x2 4x1")
    parser.add_argument("--rust-inner-threshold", type=int,
                        help="explicit item threshold; unset uses production work gates")
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--sample-policy", choices=("total", "per-rank"), default="total")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=1800, help="seconds per command")
    parser.add_argument("--prepare-only", action="store_true", help="build/setup without timing workloads")
    parser.add_argument("--mpi-prefix", type=Path, default=Path("/opt/mpich"))
    parser.add_argument("--julia-source", type=Path, default=ROOT / "extern/Julia-mVMC",
                        help="reference checkout or separately validated optimization worktree")
    parser.add_argument("--julia-bin", type=Path,
                        default=Path("/home/vscode/.cache/mvmc/tools/julia-1.13.1/bin/julia"))
    args = parser.parse_args()
    try:
        layouts = [tuple(map(int, item.split("x"))) for item in args.layout] if args.layout else [(r, 1) for r in args.ranks]
        if any(len(item) != 2 or min(item) < 1 for item in layouts) or len(set(layouts)) != len(layouts):
            raise ValueError()
    except ValueError:
        parser.error("layouts must be unique positive process/thread pairs, e.g. 1x4")
    args.ranks = [r for r, _ in layouts]
    models = args.model or [16]
    if (min(args.steps, args.samples, args.reps, args.timeout, *args.ranks) < 1
            or args.warmups < 1):
        parser.error("positive counts, unique ranks and at least one warmup required")
    if len(set(models)) != len(models):
        parser.error("model sizes must be unique")
    if args.sample_policy == "total" and any(args.samples % r for r in args.ranks):
        parser.error("total samples must be divisible by every rank count")
    if args.output.exists():
        parser.error("output directory must be new")
    prefix = args.mpi_prefix.resolve()
    args.julia_source = args.julia_source.resolve()
    args.julia_bin = args.julia_bin.resolve()
    for tool in (prefix / "bin/mpiexec", prefix / "bin/mpicc", args.julia_bin):
        if not tool.is_file():
            parser.error(f"missing {tool}; use the Dev Container and install-julia.sh")
    output = args.output.resolve()
    output.mkdir(parents=True)
    env = os.environ.copy()
    env.update({name: "1" for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
               "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "JULIA_NUM_THREADS", "MVMC_RS_INNER_THREADS")})
    env.update(MPICC=str(prefix / "bin/mpicc"), MVMC_BLAS_PROVIDER="openblas",
               JULIA_MVMC_MPI="1", JULIA_MVMC_INNER_THREADS="1", JULIA_MVMC_PFAPACK_THREADS="0", MVMC_RS_SR_BACKEND="c-order", MVMC_RS_MEASURE_PF_BACKEND="c-order")
    # UCX reads this while libmpi is loaded, before MPI.jl's __init__.
    # Julia uses SIGSEGV internally for threaded GC safepoints.
    env.update(UCX_ERROR_SIGNALS="SIGILL,SIGBUS,SIGFPE", UCX_MEMTYPE_CACHE="no")
    env["JULIA_NUM_GC_THREADS"] = "1"
    env.pop("MVMC_RS_INNER_THRESHOLD", None)
    if args.rust_inner_threshold is not None:
        if args.rust_inner_threshold < 1:
            parser.error("Rust inner threshold must be positive")
        env["MVMC_RS_INNER_THRESHOLD"] = str(args.rust_inner_threshold)
    env["LD_LIBRARY_PATH"] = str(prefix / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    provenance = {"arguments": {key: str(value) if isinstance(value, Path) else value
                               for key, value in vars(args).items()},
                  "environment": {key: value for key, value in env.items()
                                  if key in {"CARGO_TARGET_DIR", "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS",
                                             "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER", "KACHE_CONFIG",
                                             "MPICC", "LIBCLANG_PATH", "LD_LIBRARY_PATH", "MVMC_BLAS_PROVIDER",
                                             "MVMC_RS_SR_BACKEND", "MVMC_RS_MEASURE_PF_BACKEND",
                                             "MVMC_RS_INNER_THREADS", "MVMC_RS_INNER_THRESHOLD", "JULIA_DEPOT_PATH", "JULIA_PROJECT", "JULIA_NUM_GC_THREADS",
                                             "JULIA_MVMC_MPI", "JULIA_MVMC_INNER_THREADS", "JULIA_MVMC_PFAPACK_THREADS"}
                                  or key.startswith(("CARGO_PROFILE_RELEASE_", "UCX_"))
                                  or key.endswith("_NUM_THREADS")}, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (output / "environment.json").write_text(json.dumps(provenance, indent=2) + "\n")
    for name, command in {"host": ["uname", "-a"], "cpu": ["lscpu"], "rust": ["rustc", "-Vv"],
                          "source": ["git", "rev-parse", "HEAD"], "dirty": ["git", "status", "--short"],
                          "submodules": ["git", "submodule", "status", "--recursive"],
                          "mpi": [prefix / "bin/mpichversion"], "compiler": [prefix / "bin/mpicc", "-show"],
                          "blas": ["pkg-config", "--modversion", "openblas"]}.items():
        run(command, output / f"{name}.log", env, args.timeout)
    build = run(["cargo", "build", "--locked", "--release", "-p", "mvmc-core", "--features", "mpi",
                 "--example", "mpi_benchmark", "--message-format=json"], output / "rust-build.log", env, args.timeout)
    artifacts = [json.loads(line) for line in build.splitlines() if line.startswith("{")]
    binary = next(Path(item["executable"]) for item in artifacts
                  if item.get("reason") == "compiler-artifact" and item.get("executable")
                  and item["target"]["name"] == "mpi_benchmark")
    linkage = run(["ldd", binary], output / "rust-libraries.log", env)
    libraries = re.findall(r"^\s*libmpi\.so\S* => (\S+)", linkage, re.MULTILINE)
    if len(libraries) != 1 or not Path(libraries[0]).resolve().is_relative_to(prefix):
        raise RuntimeError("Rust MPI library is not the selected provider")
    project = prepare_project(output, args.julia_bin, env, prefix, args.julia_source)
    run(["git", "-C", args.julia_source, "rev-parse", "HEAD"], output / "julia-source.log", env)
    run(["git", "-C", args.julia_source, "diff", "HEAD"], output / "julia-source-diff.log", env)
    run(["ldd", project / "PfaPack.jl/deps/libltl2inv.so"],
        output / "julia-native-libraries.log", env)
    hashes = {}
    for path in (binary, ROOT / "benchmark/mpi_comparison/worker.jl",
                 ROOT / "crates/mvmc-core/examples/mpi_benchmark.rs",
                 Path(__file__), project / "Manifest-v1.13.toml", project / "LocalPreferences.toml",
                 project / "PfaPack.jl/deps/libltl2inv.so", prefix / "lib/libmpi.so",
                 prefix / "bin/mpiexec", prefix / "bin/mpicc"):
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted((args.julia_source / "MVMCOptimizers.jl/src").glob("*.jl")):
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in re.findall(r"^\s*lib(?:openblas|lapack)\S* => (\S+)", linkage, re.MULTILINE):
        hashes[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    (output / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    snapshot = output / "executed-sources"
    snapshot.mkdir()
    shutil.copytree(args.julia_source / "MVMCOptimizers.jl/src", snapshot / "MVMCOptimizers-src")
    for path in (Path(__file__), ROOT / "crates/mvmc-core/examples/mpi_benchmark.rs",
                 ROOT / "benchmark/mpi_comparison/worker.jl"):
        shutil.copyfile(path, snapshot / path.name)
    if args.prepare_only:
        print(f"Environment prepared: {output}", flush=True)
        return
    rows = []
    with (output / "measurements.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("model", "implementation", "ranks", "threads", "blas_threads", "rep", "steps",
                               "samples_per_rank", "total_samples_per_step", "seconds", "final_energy_per_site",
                               "observed_workers_min", "observed_workers_max"))
        writer.writeheader()
        for size in models:
            for ranks, threads in layouts:
                cell_env = env.copy()
                # Keep the driver in the compute pool and worker IDs contiguous;
                # reference workspaces are indexed by threadid().
                cell_env.update(JULIA_NUM_THREADS=f"{threads},0", MVMC_RS_INNER_THREADS=str(threads))
                samples = args.samples // ranks if args.sample_policy == "total" else args.samples
                cell = output / f"L{size}-ranks{ranks}-threads{threads}"
                cell.mkdir()
                (cell / "environment.json").write_text(json.dumps({key: value for key, value in cell_env.items() if key.endswith("_THREADS") or key.startswith("JULIA_MVMC")}, indent=2) + "\n")
                inputs = cell / "inputs"
                shutil.copytree(ROOT / f"benchmark/hubbard_chain/inputs/hubbard_chain_L{size}", inputs)
                modpara = inputs / "modpara.def"
                text, count = re.subn(r"(?m)^NVMCSample[ \t]+\d+[ \t]*$", f"NVMCSample {samples}", modpara.read_text())
                if count != 1:
                    raise RuntimeError("expected exactly one NVMCSample")
                modpara.write_text(text)
                (cell / "input-sha256.json").write_text(json.dumps({
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(inputs.iterdir()) if path.is_file()
                }, indent=2) + "\n")
                common = [inputs / "namelist.def", str(args.steps), str(args.warmups), str(args.reps)]
                for impl in ("rust", "julia"):
                    (cell / f"{impl}-loadavg.txt").write_text(Path("/proc/loadavg").read_text())
                    command = [prefix / "bin/mpiexec", "-n", str(ranks)]
                    command += ([binary] if impl == "rust" else [args.julia_bin, f"--project={project}",
                                "--startup-file=no", ROOT / "benchmark/mpi_comparison/worker.jl"])
                    command += common + [cell / impl, str(ranks)]
                    text = run(command, cell / f"{impl}.log", cell_env, args.timeout)
                    measurements = parse_measurements(text, ranks, args.reps, threads,
                                                      require_parallel=impl == "julia" or args.rust_inner_threshold is not None,
                                                      require_native_blas=impl == "julia")
                    workers = [int(n) for _, _, n in re.findall(r"^EXECUTION (\d+) (\d+) (\d+)$", text, re.MULTILINE)]
                    for rep, seconds, energy in measurements:
                        out = cell / impl / f"run-{args.warmups + rep - 1}"
                        values = (out / "zvo_out.dat").read_text().splitlines()
                        if len(values) != args.steps or not (out / "zqp_opt.dat").is_file():
                            raise RuntimeError(f"incomplete outputs: {out}")
                        row = dict(model=f"L{size}", implementation=impl, ranks=ranks, threads=threads, blas_threads=1, rep=rep,
                                   steps=args.steps, samples_per_rank=samples, total_samples_per_step=samples*ranks,
                                   seconds=seconds, final_energy_per_site=energy,
                                   observed_workers_min=min(workers), observed_workers_max=max(workers))
                        writer.writerow(row)
                        rows.append(row)
                    stream.flush()
                    print(f"{impl} L{size} ranks={ranks} threads={threads}: median "
                          f"{statistics.median(t for _, t, _ in measurements):.3f}s", flush=True)
    write_report(output, rows, args.sample_policy, args.warmups, args.rust_inner_threshold)
    print(f"Results: {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()

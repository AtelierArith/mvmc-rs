#!/usr/bin/env python3
"""Run matching C/Rust inputs locally; retain outputs and numerical comparisons (#492).

Run inside the Linux Dev Container with uv. C is an explicitly built optional
reference executable; Cargo builds/tests never invoke this script or C.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import statistics
import time
from pathlib import Path

import bench_cpu_round as workloads

INTEGER = re.compile(r"[+-]?\d+\Z")


def compare_file(reference, actual, absolute, relative):
    """Check shape, exact text/integer fields, and finite numeric fields separately."""
    expected = reference.read_text().splitlines()
    observed = actual.read_text().splitlines()
    result = dict(fields=0, max_absolute=0.0, max_scaled=0.0, failures=0, first=None)

    def fail(where, left, right, reason):
        result["failures"] += 1
        if result["first"] is None:
            result["first"] = dict(
                location=where, expected=left, actual=right, reason=reason
            )

    if len(expected) != len(observed):
        fail("lines", len(expected), len(observed), "shape")
    for line, (left, right) in enumerate(zip(expected, observed), 1):
        a, b = left.split(), right.split()
        if len(a) != len(b):
            fail(f"{line}", len(a), len(b), "shape")
        for column, (x, y) in enumerate(zip(a, b), 1):
            where = f"{line}:{column}"
            if INTEGER.fullmatch(x) and INTEGER.fullmatch(y):
                if int(x) != int(y):
                    fail(where, x, y, "integer")
                continue
            try:
                xf, yf = float(x), float(y)
            except ValueError:
                if x != y:
                    fail(where, x, y, "text")
                continue
            result["fields"] += 1
            if not math.isfinite(xf) or not math.isfinite(yf):
                fail(where, x, y, "nonfinite")
                continue
            delta = abs(xf - yf)
            scale = max(abs(xf), abs(yf))
            result["max_absolute"] = max(result["max_absolute"], delta)
            result["max_scaled"] = max(result["max_scaled"], delta / (1.0 + scale))
            if delta > absolute + relative * scale:
                fail(where, x, y, "numerical")
    return result


def output_files(directory):
    # Clock records and timing-dependent SR diagnostics are not numerical results.
    files = {
        p.name: p
        for p in (directory / "output").iterdir()
        if p.is_file()
        and p.name.startswith(("zvo_", "zqp_"))
        and "Timer" not in p.name
        and "_time_" not in p.name
        and p.name != "zvo_SRinfo.dat"
    }
    settings = {
        parts[0]: parts[1]
        for line in (directory / "modpara.def").read_text().splitlines()
        if len(parts := line.split()) >= 2
    }
    if settings.get("NVMCCalMode") == "0":
        index = int(settings.get("NDataIdxStart", "0"))
        for stem in ("zvo_out", "zvo_var"):
            name = stem + ".dat"
            if name in files:
                files[f"{stem}_{index:03}.dat"] = files.pop(name)
    return files


def compare_outputs(reference, actual, absolute, relative):
    a, b = output_files(reference), output_files(actual)

    def bounds(name):
        # Fixed-input Green operators and local-energy outputs use the budgets
        # documented in docs/NUMERICAL_COMPARISONS.md. Parameter/SR checkpoints
        # use the short-prefix budget. Overrides must be explicit in provenance.
        default = (
            1e-13
            if "_cis" in name
            else (1e-12 if name.startswith("zvo_out") else 1e-11)
        )
        return (
            default if absolute is None else absolute,
            default if relative is None else relative,
        )

    return dict(
        missing=sorted(a.keys() - b.keys()),
        extra=sorted(b.keys() - a.keys()),
        files={
            name: dict(
                bounds=dict(zip(("absolute", "relative"), bounds(name))),
                **compare_file(a[name], b[name], *bounds(name)),
            )
            for name in sorted(a.keys() & b.keys())
        },
    )


def execute(
    binary,
    destination,
    flags,
    parameter,
    threads,
    implementation,
    timeout,
    observe=False,
):
    env = dict(
        os.environ,
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS=str(threads),
        BLIS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        MVMC_C_TIMER="1",
        MVMC_RS_INNER_THREADS=str(threads),
    )
    # A benchmark must not inherit profiling or gate overrides from the shell.
    for key in list(env):
        if key.startswith("MVMC_RS_INNER_") and key != "MVMC_RS_INNER_THREADS":
            del env[key]
    if observe and implementation == "rust":
        env["MVMC_RS_INNER_OBSERVE"] = "1"
    command = [str(binary), *flags, "namelist.def", *([parameter] if parameter else [])]
    if implementation == "rust" and parameter is None:
        # C does not auto-load a neighboring initial.def. The Rust Julia-style
        # convenience default must be disabled to compare the same lifecycle.
        command[1:1] = ["--initial-def", "none"]
    if implementation == "c":
        command = ["/opt/mpich/bin/mpirun", "-np", "1", *command]
    load_before = os.getloadavg()[0]
    start = time.perf_counter()
    with (destination / "run.log").open("w") as log:
        subprocess.run(
            command,
            cwd=destination,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=timeout,
        )
    wall = time.perf_counter() - start
    log = (destination / "run.log").read_text()
    observed = re.search(r"inner-execution: (.*)", log)
    return dict(
        wall_s=wall,
        execution=observed.group(1) if observed else None,
        timers=workloads.parse_timer(destination / "zvo_CalcTimer.dat"),
        load_before=load_before,
        load_after=os.getloadavg()[0],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rust", type=Path, required=True)
    parser.add_argument("--c", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, required=True, help="new artifact directory"
    )
    parser.add_argument("--workloads", default="opt_hubbard_L32,phys_hubbard_L32")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--steps", type=int, help="override optimization prefix length")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--data-count", type=int)
    # Short SR checkpoints use this bound in NUMERICAL_COMPARISONS.md. Failures
    # remain failures; long-run trajectory differences require a separate audit.
    parser.add_argument(
        "--absolute", type=float, help="explicit override of per-output budget"
    )
    parser.add_argument(
        "--relative", type=float, help="explicit override of per-output budget"
    )
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument(
        "--observe",
        action="store_true",
        help="record actual workers in a separate diagnostic run; adds overhead",
    )
    args = parser.parse_args()
    if args.reps < 1 or any(t < 1 for t in args.threads):
        parser.error("repetitions and thread counts must be positive")
    if any(
        value is not None and value < 1
        for value in (args.steps, args.samples, args.data_count)
    ):
        parser.error("steps, samples and data-count must be positive")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    for value in (args.absolute, args.relative):
        if value is not None and (not math.isfinite(value) or value < 0):
            parser.error("tolerances must be finite and nonnegative")
    args.rust, args.c = args.rust.resolve(strict=True), args.c.resolve(strict=True)
    args.out.mkdir(parents=True, exist_ok=False)
    records, comparisons = [], []
    metadata = dict(
        arguments={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        binary_sha256={
            impl: hashlib.sha256(p.read_bytes()).hexdigest()
            for impl, p in [("rust", args.rust), ("c", args.c)]
        },
    )
    metadata["platform"] = platform.platform()
    root = Path(__file__).resolve().parent.parent
    source_paths = [
        root / path
        for path in (
            "crates/mvmc-core/src/threading.rs",
            "crates/mvmc-core/src/sr.rs",
            "crates/mvmc-core/src/observables.rs",
            "crates/mvmc-cli/src/main.rs",
            "crates/mvmc-expert-parsers/src/utils/opt_flag.rs",
            "extern/mVMC-1.3.0/src/mVMC/vmccal.c",
            "scripts/bench_native_c.py",
        )
    ]
    metadata["source_sha256"] = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source_paths
    }
    metadata["cpu"] = (
        Path("/proc/cpuinfo").read_text().split("model name", 1)[-1].splitlines()[0]
    )
    metadata["tools"] = {}
    for name, command in [
        ("rust", ["rustc", "-Vv"]),
        ("gcc", ["gcc", "--version"]),
        ("blas", ["pkg-config", "--modversion", "openblas"]),
        ("mpi", ["/opt/mpich/bin/mpichversion"]),
        ("rust_libraries", ["ldd", str(args.rust)]),
        ("c_libraries", ["ldd", str(args.c)]),
        ("commit", ["git", "rev-parse", "HEAD"]),
        ("dirty", ["git", "status", "--short"]),
    ]:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        metadata["tools"][name] = dict(
            returncode=result.returncode, stdout=result.stdout, stderr=result.stderr
        )
    (args.out / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for name in args.workloads.split(","):
        if name not in workloads.WORKLOADS:
            parser.error(f"unknown workload {name}")
        for threads in args.threads:
            for rep in range(args.reps + 1):
                destinations = {}
                for impl, binary in [("rust", args.rust), ("c", args.c)]:
                    base = args.out / f"{name}-{threads}t-{rep}-{impl}"
                    dest, flags, parameter = workloads.prepare(name, base)
                    overrides = {}
                    for key, value in [
                        ("NSROptItrStep", args.steps),
                        ("NVMCSample", args.samples),
                        ("NDataQtySmp", args.data_count),
                    ]:
                        if value is not None:
                            overrides[key] = str(value)
                    if args.steps is not None:
                        overrides["NSROptItrSmp"] = str(min(args.steps, 10))
                    workloads.patch_modpara(dest / "modpara.def", overrides)
                    timing = execute(
                        binary,
                        dest,
                        flags,
                        parameter,
                        threads,
                        impl,
                        args.timeout,
                        args.observe,
                    )
                    inputs = {
                        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in dest.iterdir()
                        if p.is_file()
                        and (p.suffix == ".def" or p.name == "zqp_opt.dat")
                    }
                    record = dict(
                        workload=name,
                        threads=threads,
                        rep=rep,
                        impl=impl,
                        input_sha256=inputs,
                        **timing,
                    )
                    records.append(record)
                    destinations[impl] = dest
                    print(name, threads, rep, impl, timing["wall_s"], flush=True)
                comparisons.append(
                    dict(
                        workload=name,
                        threads=threads,
                        rep=rep,
                        **compare_outputs(
                            destinations["c"],
                            destinations["rust"],
                            args.absolute,
                            args.relative,
                        ),
                    )
                )
                # Preserve evidence even when a later workload fails.
                (args.out / "runs.json").write_text(
                    json.dumps(records, indent=2) + "\n"
                )
                (args.out / "numerical.json").write_text(
                    json.dumps(comparisons, indent=2) + "\n"
                )
    failed = any(
        c["missing"]
        or c["extra"]
        or not c["files"]
        or any(f["failures"] for f in c["files"].values())
        for c in comparisons
    )
    summary = []
    for name in args.workloads.split(","):
        for threads in args.threads:
            row = dict(workload=name, threads=threads)
            for impl in ("rust", "c"):
                measured = [
                    r
                    for r in records
                    if r["workload"] == name
                    and r["threads"] == threads
                    and r["impl"] == impl
                    and r["rep"] > 0
                ]
                row[impl] = dict(
                    internal_s=statistics.median(r["timers"][0] for r in measured),
                    wall_s=statistics.median(r["wall_s"] for r in measured),
                )
            summary.append(row)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        "numerical comparison:", "FAIL (inspect numerical.json)" if failed else "PASS"
    )
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())

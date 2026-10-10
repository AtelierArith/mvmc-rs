#!/usr/bin/env python3
"""Add explicit native C measurements to a completed fresh MPI comparison (#495).

Run through uv in the Linux Dev Container. C uses rank-zero internal All;
Rust/Julia use the maximum warmed production-API time over ranks. Preserve
these timing boundaries in the report instead of equating launcher wall time.
"""

import argparse
import csv
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
import time

from bench_cpu_round import patch_modpara, parse_timer


def validate_world(text, ranks, threads=1):
    for key, count in (("WORLD", ranks), ("THREADS", threads), ("BLAS_THREADS", 1)):
        observed = re.findall(rf"^{key} (\d+) (\d+)$", text, re.MULTILINE)
        if sorted((int(a), int(b)) for a, b in observed) != [
            (r, count) for r in range(ranks)
        ]:
            raise ValueError(f"invalid native {key} evidence")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rj", type=Path, required=True, help="completed bench_mpi output"
    )
    parser.add_argument("--c", type=Path, required=True)
    parser.add_argument(
        "--c-source",
        type=Path,
        required=True,
        help="unmodified upstream copy used for this C build",
    )
    parser.add_argument("--observer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new result directory")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.rj = args.rj.resolve(strict=True)
    args.c = args.c.resolve(strict=True)
    args.c_source = args.c_source.resolve(strict=True)
    args.observer = args.observer.resolve(strict=True)
    if not (args.rj / "report.md").is_file():
        parser.error("Rust/Julia measurement must finish before timing C")
    metadata = json.loads((args.rj / "environment.json").read_text())
    counts = metadata["arguments"]
    with (args.rj / "measurements.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    cells = sorted({(row["model"], int(row["ranks"]), int(row["threads"])) for row in rows})
    args.out.mkdir(parents=True, exist_ok=False)
    prefix = Path(counts["mpi_prefix"])
    env = dict(
        os.environ,
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        BLIS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        LD_PRELOAD=str(args.observer),
        UCX_MEMTYPE_CACHE="no",
    )
    env["LD_LIBRARY_PATH"] = str(prefix / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    provenance = dict(
        rust_julia=metadata,
        omp_threads_per_cell=[dict(model=model, ranks=ranks, threads=threads)
                              for model, ranks, threads in cells],
        native_environment={
            key: env[key]
            for key in (
                "OPENBLAS_NUM_THREADS",
                "BLIS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "LD_PRELOAD",
                "LD_LIBRARY_PATH",
                "UCX_MEMTYPE_CACHE",
            )
        },
    )
    sources = [
        args.c,
        args.observer,
        Path(__file__),
        root / "c_toolbox/mpi_comparison_495/world_observer.c",
        root / "extern/mVMC-1.3.0/src/mVMC/vmcmain.c",
    ]
    provenance["sha256"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
    }
    upstream = root / "extern/mVMC-1.3.0"
    tracked = subprocess.check_output(
        ["git", "-C", str(upstream), "ls-files"], text=True
    ).splitlines()
    verified_sources = {}
    for name in tracked:
        original = upstream / name
        if original.is_file() and (
            original.suffix in (".c", ".h", ".f", ".f90", ".cmake")
            or original.name == "CMakeLists.txt"
        ):
            digest = hashlib.sha256(original.read_bytes()).hexdigest()
            if (
                hashlib.sha256((args.c_source / name).read_bytes()).hexdigest()
                != digest
            ):
                raise ValueError(f"C build copy differs from current reference: {name}")
            verified_sources[name] = digest
    provenance["verified_c_source_sha256"] = verified_sources
    flags = args.c.parent / "CMakeFiles/vmc.out.dir/flags.make"
    if flags.is_file():
        provenance["c_flags"] = flags.read_text()
    for name, command in (
        ("main", ["git", "rev-parse", "HEAD"]),
        (
            "c_commit",
            ["git", "-C", str(root / "extern/mVMC-1.3.0"), "rev-parse", "HEAD"],
        ),
        ("c_libraries", ["ldd", str(args.c)]),
        ("observer_libraries", ["ldd", str(args.observer)]),
        ("compiler", ["gcc", "--version"]),
    ):
        provenance[name] = subprocess.check_output(command, text=True)
    (args.out / "environment.json").write_text(json.dumps(provenance, indent=2) + "\n")
    native = []
    for model, ranks, threads in cells:
        env["OMP_NUM_THREADS"] = str(threads)
        matching = [r for r in rows if r["model"] == model and int(r["ranks"]) == ranks
                    and int(r["threads"]) == threads]
        samples = int(matching[0]["samples_per_rank"])
        for rep in range(counts["warmups"] + counts["reps"]):
            dest = args.out / f"{model}-{ranks}r-{threads}t-{rep}"
            shutil.copytree(args.rj / f"{model}-ranks{ranks}-threads{threads}/inputs", dest)
            (dest / "output").mkdir()
            patch_modpara(
                dest / "modpara.def",
                {
                    # readdef.c adds output/ itself; a directory here doubles it.
                    "CDataFileHead": "zvo",
                    "CParaFileHead": "zqp",
                    "NSROptItrStep": str(counts["steps"]),
                    "NSROptItrSmp": str(counts["steps"]),
                    "NVMCSample": str(samples),
                    "NVMCCalMode": "0",
                    "NLanczosMode": "0",
                },
            )
            hashes = {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in dest.iterdir()
                if p.is_file()
            }
            (dest / "input-sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
            command = [
                str(prefix / "bin/mpiexec"),
                "-n",
                str(ranks),
                str(args.c),
                "namelist.def",
            ]
            start = time.perf_counter()
            with (dest / "run.log").open("w") as log:
                process = subprocess.Popen(
                    command,
                    cwd=dest,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    code = process.wait(timeout=counts["timeout"])
                except (subprocess.TimeoutExpired, KeyboardInterrupt):
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    raise
            wall = time.perf_counter() - start
            if code:
                raise RuntimeError(f"native C failed ({code}): {dest}")
            validate_world((dest / "run.log").read_text(), ranks, threads)
            timer = parse_timer(dest / "zvo_CalcTimer.dat")
            seconds = timer[0]
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("invalid C internal time")
            energies = (
                next((dest / "output").glob("zvo_out_*.dat")).read_text().splitlines()
            )
            if (
                len(energies) != counts["steps"]
                or not (dest / "output/zqp_opt.dat").is_file()
            ):
                raise ValueError("incomplete C optimization")
            if any(
                not math.isfinite(float(token))
                for line in energies
                for token in line.split()
            ):
                raise ValueError("nonfinite C energy output")
            record = dict(
                model=model,
                implementation="c",
                ranks=ranks,
                threads=threads,
                rep=rep - counts["warmups"] + 1,
                seconds=seconds,
                wall_s=wall,
                steps=counts["steps"],
                samples_per_rank=samples,
                total_samples_per_step=samples * ranks,
                timers=timer,
                load=os.getloadavg()[0],
                command=command,
            )
            native.append(record)
            (args.out / "native-runs.json").write_text(
                json.dumps(native, indent=2) + "\n"
            )
            print(model, ranks, threads, rep, seconds, flush=True)
    summary = []
    for model, ranks, threads in cells:
        row = dict(model=model, ranks=ranks, threads=threads)
        for impl in ("c", "julia", "rust"):
            selected = [
                r
                for r in (native if impl == "c" else rows)
                if r["model"] == model
                and int(r["ranks"]) == ranks
                and int(r["threads"]) == threads
                and r["implementation"] == impl
                and int(r["rep"]) > 0
            ]
            row[impl] = statistics.median(float(r["seconds"]) for r in selected)
        summary.append(row)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Before/after A/B of two Rust CLI binaries on the CPU-round workloads (issue #448).

Interleaves `--before` and `--after` (warm-up + `--reps` repetitions each), records wall time and
the C-compatible section timers, and checks that the numerical outputs of both binaries are
byte-identical (all `output/zvo_*` and `zqp_*` files except the wall-clock `zvo_time*` and
`zvo_CalcTimer.dat`).

    uv run --no-project python scripts/bench_cpu_ab.py --before /tmp/mvmc_base \
        --after target/release/mvmc --workloads opt_heisenberg_fsz,phys_lanczos --reps 3 \
        --out benchmark/cpu_round/results/ab.csv
"""
import argparse
import csv
import filecmp
import os
import shutil
import statistics
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_cpu_round as b  # noqa: E402


def identical(dir_a: Path, dir_b: Path):
    out_a, out_b = dir_a / "output", dir_b / "output"
    diffs = []
    names = sorted(p.name for p in out_a.iterdir())
    for name in names:
        if name.startswith("zvo_time") or name == "zvo_CalcTimer.dat":
            continue
        pb = out_b / name
        if not pb.exists() or not filecmp.cmp(out_a / name, pb, shallow=False):
            diffs.append(name)
    extra = sorted(set(p.name for p in out_b.iterdir()) - set(names))
    return diffs + [f"extra:{e}" for e in extra if not e.startswith("zvo_time")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--workloads", default="all")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--inner", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    names = list(b.WORKLOADS) if args.workloads == "all" else args.workloads.split(",")
    work = Path(tempfile.mkdtemp(prefix="ab448-", dir="/tmp/claude-1000"))
    rows = []
    status = {}
    for name in names:
        da, flags, para = b.prepare(name, work / "a")
        db, _, _ = b.prepare(name, work / "b")
        for rep in range(args.reps + 1):
            res = {}
            for tag, dest, binary in (("before", da, args.before), ("after", db, args.after)):
                l0 = b.load()
                wall, secs = b.run_rust(binary, dest, flags, para, args.inner, {})
                res[tag] = wall
                if rep > 0:
                    rows.append({"workload": name, "variant": tag, "rep": rep, "wall_s": wall,
                                 "sec3": secs.get(3), "sec4": secs.get(4),
                                 "load_before": l0, "load_after": b.load()})
            if rep > 0:
                print(f"{name:28s} rep{rep}  before {res['before']:8.3f}s  after {res['after']:8.3f}s",
                      flush=True)
        status[name] = identical(da, db)
        print(f"{name}: outputs byte-identical = {not status[name]} {status[name]}")
    print("\n=== medians (before -> after) ===")
    for name in names:
        bef = statistics.median(r["wall_s"] for r in rows if r["workload"] == name and r["variant"] == "before")
        aft = statistics.median(r["wall_s"] for r in rows if r["workload"] == name and r["variant"] == "after")
        print(f"{name:28s} {bef:8.3f}s -> {aft:8.3f}s  ({bef / aft:5.2f}x)  identical={not status[name]}")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    shutil.rmtree(work, ignore_errors=True)
    if any(status.values()):
        sys.exit("OUTPUTS DIFFER")


if __name__ == "__main__":
    main()

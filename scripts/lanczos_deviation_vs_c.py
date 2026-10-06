#!/usr/bin/env python3
"""Deviation of two Rust binaries from native C on the Lanczos outputs (issue #478).

Runs the `phys_lanczos` workload of `bench_cpu_round.py` once with native C and once with each
Rust binary (same inputs and seed), then reports the largest relative deviation (scaled by the
largest magnitude in the file) of every `zvo_ls_*` file from the C file.

    uv run --no-project python scripts/lanczos_deviation_vs_c.py --before OLD --after NEW \
        [--workload phys_lanczos] [--out FILE]
"""
import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_cpu_round as b  # noqa: E402


def read(path):
    rows = []
    for line in path.read_text().split("\n"):
        vals = []
        for tok in line.split():
            try:
                vals.append(float(tok))
            except ValueError:
                pass
        if vals:
            rows.append(vals)
    return rows


def deviation(ref, other):
    scale = max((abs(v) for r in ref for v in r), default=1.0) or 1.0
    worst = 0.0
    for ra, rb in zip(ref, other):
        for x, y in zip(ra, rb):
            worst = max(worst, abs(x - y) / scale)
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--workload", default="phys_lanczos")
    ap.add_argument("--c-dir", default="/tmp/claude-1000/c448")
    ap.add_argument("--image", default="vsc-mvmc-rs-741feed754ed4827a429135e5fcc0e7623f1f847d228698b066da094a8819e32-uid:latest")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    work = Path(tempfile.mkdtemp(prefix="dev478-", dir="/tmp/claude-1000"))
    dirs = {}
    for tag in ("c", "before", "after"):
        dest, flags, para = b.prepare(args.workload, work / tag)
        if tag == "c":
            b.run_c(Path(args.c_dir), args.image, dest, flags, para, 1)
        else:
            b.run_rust(args.before if tag == "before" else args.after, dest, flags, para, 1, {})
        dirs[tag] = dest / "output"
    lines = ["file                     dev(before)  dev(after)"]
    worst = {"before": 0.0, "after": 0.0}
    for f in sorted(p.name for p in dirs["c"].glob("zvo_ls_*")):
        ref = read(dirs["c"] / f)
        d = {t: deviation(ref, read(dirs[t] / f)) for t in ("before", "after") if (dirs[t] / f).exists()}
        for t, v in d.items():
            worst[t] = max(worst[t], v)
        lines.append(f"{f:24s} {d.get('before', float('nan')):.3e}   {d.get('after', float('nan')):.3e}")
    lines.append(f"max over files           {worst['before']:.3e}   {worst['after']:.3e}")
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()

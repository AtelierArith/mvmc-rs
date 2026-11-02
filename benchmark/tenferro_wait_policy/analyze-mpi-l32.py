#!/usr/bin/env python3
"""Summarize mpi-l32.sh's measured repetitions; time files contain wall clocks."""
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

root = Path(sys.argv[1])
summary = {}
for mode in ("opt", "phys"):
    dirs = [root / "results" / f"{mode}-t{t}-r{r}" for t in (1, 4) for r in (1, 2, 3)]
    names = sorted(p.name for p in dirs[0].iterdir() if p.is_file() and "_time_" not in p.name)
    if not names:
        raise RuntimeError(f"{mode}: no output files")
    if any(sorted(p.name for p in d.iterdir() if p.is_file() and "_time_" not in p.name) != names for d in dirs):
        raise RuntimeError(f"{mode}: mismatched output files")
    def numbers(path):
        values = [float(x) for line in path.read_text().splitlines() if not line.lstrip().startswith("#") for x in line.split()]
        if not all(map(math.isfinite, values)):
            raise RuntimeError(f"{path}: nonfinite value")
        return values
    maximum = 0.0
    hashes = {}
    for directory in dirs:
        hashes[directory.name] = {}
        for name in names:
            path = directory / name
            hashes[directory.name][name] = hashlib.sha256(path.read_bytes()).hexdigest()
            reference, values = numbers(dirs[0] / name), numbers(path)
            if len(reference) != len(values):
                raise RuntimeError(f"{path}: mismatched value count")
            maximum = max(maximum, max((abs(a - b) for a, b in zip(reference, values)), default=0.0))
    timings = {}
    for threads in (1, 4):
        samples = []
        for rep in (1, 2, 3):
            log = (root / f"{mode}-t{threads}-r{rep}.txt").read_text()
            match = re.search(r"wall=([0-9.]+) user=([0-9.]+) sys=([0-9.]+)", log)
            if match is None:
                raise RuntimeError(f"{mode} threads={threads} rep={rep}: missing time")
            wall, user, system = map(float, match.groups())
            samples.append({"wall": wall, "user": user, "system": system})
        timings[str(threads)] = {"samples": samples, "wall_median": statistics.median(v["wall"] for v in samples)}
    summary[mode] = {"output_files": names, "max_abs_from_first": maximum, "timings": timings, "sha256": hashes}
print(json.dumps(summary, indent=2))

#!/usr/bin/env python3
"""Summarize alternating MPI L32 results, comparing formatted output numbers."""
import json
import math
from pathlib import Path
import re
import statistics
import sys

root = Path(sys.argv[1])
summary = {}
for mode in ("opt", "phys"):
    dirs = [root / "results" / f"{mode}-{policy}-t{t}-r{r}" for policy in ("default", "zero") for t in (1, 4) for r in (1, 2, 3)]
    names = sorted(p.name for p in dirs[0].iterdir() if p.is_file() and "_time_" not in p.name)
    if not names or any(sorted(p.name for p in d.iterdir() if p.is_file() and "_time_" not in p.name) != names for d in dirs):
        raise RuntimeError(f"{mode}: missing or mismatched output files")
    def numbers(path):
        values = [float(x) for line in path.read_text().splitlines() if not line.lstrip().startswith("#") for x in line.split()]
        if not all(map(math.isfinite, values)):
            raise RuntimeError(f"{path}: nonfinite value")
        return values
    maximum = 0.0
    for name in names:
        reference = numbers(dirs[0] / name)
        for directory in dirs[1:]:
            values = numbers(directory / name)
            if len(reference) != len(values):
                raise RuntimeError(f"{directory / name}: mismatched count")
            maximum = max(maximum, max((abs(a-b) for a,b in zip(reference, values)), default=0.0))
    timings = {}
    for policy in ("default", "zero"):
        for threads in (1, 4):
            records = []
            for rep in (1, 2, 3):
                log = (root / f"{mode}-{policy}-t{threads}-r{rep}.txt").read_text()
                match = re.search(r"wall=([0-9.]+) user=([0-9.]+) sys=([0-9.]+)", log)
                if match is None:
                    raise RuntimeError(f"{mode}, {policy}, {threads}, {rep}: missing timing")
                wall, user, system = map(float, match.groups())
                records.append({"wall": wall, "user": user, "system": system})
            timings[f"{policy}-{threads}"] = {"samples": records, "wall_median": statistics.median(r["wall"] for r in records)}
    summary[mode] = {"files_compared": len(names), "max_abs_from_first": maximum, "timings": timings}
print(json.dumps(summary, indent=2))

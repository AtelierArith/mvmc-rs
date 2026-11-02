#!/usr/bin/env python3
"""Summarize formatted standalone SR stage timings and numerical checks."""
import json
import math
from pathlib import Path
import re
import statistics
import sys

root = Path(sys.argv[1])
summary = {}
for inner in (1, 4):
    summary[str(inner)] = {}
    for version in ("before", "after"):
        cases = {}
        process = []
        for rep in (1, 2, 3):
            path = root / f"{version}-i{inner}-r{rep}.txt"
            text = path.read_text()
            match = re.search(r"wall=([0-9.]+) user=([0-9.]+) sys=([0-9.]+)", text)
            if match is None:
                raise RuntimeError(f"{path}: missing timing")
            process.append(dict(zip(("wall", "user", "system"), map(float, match.groups()))))
            found = set()
            for line in text.splitlines():
                parts = [part.strip() for part in line.split("|")]
                if len(parts) < 10 or not parts[1].isdigit() or not parts[2].isdigit():
                    continue
                key = f"{parts[1]}-{parts[2]}"
                if key in found:
                    raise RuntimeError(f"{path}: duplicate case {key}")
                found.add(key)
                operations = [float(parts[index].split("/")[1].strip()) for index in range(3, 7)]
                differences = [float(value) for value in parts[7].split()]
                residuals = [float(value.strip()) for value in parts[8].split("/")]
                if len(differences) != 4 or len(residuals) != 2 or not all(map(math.isfinite, operations + differences + residuals)):
                    raise RuntimeError(f"{path}: invalid numerical fields")
                cases.setdefault(key, []).append({"tenferro_ms": operations, "relative_differences": differences, "residuals": residuals})
            if found != {"64-300", "64-128", "128-300", "128-256"}:
                raise RuntimeError(f"{path}: missing or unexpected cases")
        summary[str(inner)][version] = {
            "process_samples": process,
            "process_wall_median": statistics.median(value["wall"] for value in process),
            "cases": {
                key: {
                    "tenferro_ms_samples": [record["tenferro_ms"] for record in records],
                    "tenferro_ms_medians": [statistics.median(record["tenferro_ms"][index] for record in records) for index in range(4)],
                    "max_relative_difference": max(max(record["relative_differences"]) for record in records),
                    "max_scaled_residual": max(max(record["residuals"]) for record in records),
                }
                for key, records in cases.items()
            },
        }
print(json.dumps(summary, indent=2))

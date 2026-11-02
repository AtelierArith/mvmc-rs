#!/usr/bin/env python3
"""Alternate warm runs of two resident_opt_worker binaries on one GPU.

Usage: uv run --no-project python compare.py BEFORE AFTER NAMELIST OUTPUT
Set CUDA_VISIBLE_DEVICES and LD_LIBRARY_PATH outside this script. The two workers
receive the same input, seed and default L128/320-walker/20-step configuration.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

before, after, namelist, out = map(Path, sys.argv[1:])
out.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", RAYON_NUM_THREADS="1")
if "CUDA_VISIBLE_DEVICES" not in env:
    raise SystemExit("set CUDA_VISIBLE_DEVICES explicitly")
workers = {}
logs = {}
samples = {"before": [], "after": []}

def await_line(name, marker):
    while True:
        line = workers[name].stdout.readline()
        if not line:
            raise RuntimeError(f"{name} terminated before {marker}")
        logs[name].write(line)
        logs[name].flush()
        match = re.search(r"rep=(\d+) seconds=([0-9.eE+-]+)", line)
        if match:
            samples[name].append({"rep": int(match[1]), "seconds": float(match[2])})
        if line.rstrip() == marker:
            return

def run(name, rep):
    # Linux process CPU time covers this worker only, not the other idle worker.
    def cpu_seconds():
        stat = Path(f"/proc/{workers[name].pid}/stat").read_text()
        fields = stat[stat.rfind(")") + 2:].split()
        return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")

    cpu_before = cpu_seconds()
    workers[name].stdin.write(f"{rep}\n")
    workers[name].stdin.flush()
    await_line(name, f"DONE {rep}")
    samples[name][-1]["cpu_seconds"] = cpu_seconds() - cpu_before

def receipt(path):
    words = path.read_text().split()
    i = 0
    assert words[i] == "seconds"
    i += 2
    result = {}
    for label in ("energies", "parameters"):
        assert words[i] == label
        n = int(words[i + 1])
        i += 2
        result[label] = [float(x) for x in words[i : i + n]]
        assert len(result[label]) == n and all(map(math.isfinite, result[label]))
        i += n
    assert i == len(words)
    return result

try:
    for name, binary in (("before", before), ("after", after)):
        logs[name] = (out / f"{name}.log").open("w")
        workers[name] = subprocess.Popen(
            [str(binary.resolve()), str(namelist.resolve()), str((out / name).resolve())],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
        )
        await_line(name, "READY")
        run(name, 0)
    for rep in range(1, 6):
        for name in (("before", "after") if rep % 2 else ("after", "before")):
            run(name, rep)
finally:
    for name, proc in workers.items():
        if proc.poll() is None:
            proc.stdin.write("quit\n")
            proc.stdin.flush()
            proc.wait(timeout=60)
        logs[name].close()

receipts = {name: [receipt(out / name / f"rep-{i}.txt") for i in range(6)] for name in workers}
comparisons = {}
for field in ("energies", "parameters"):
    base, migrated = receipts["before"][1][field], receipts["after"][1][field]
    if len(base) != len(migrated):
        raise RuntimeError(f"{field}: different lengths")
    if any(len(run[field]) != len(base) for runs in receipts.values() for run in runs):
        raise RuntimeError(f"{field}: inconsistent repetition lengths")
    comparisons[field] = {
        "cross_version_max_abs": max((abs(a-b) for old,new in zip(receipts["before"][1:],receipts["after"][1:]) for a,b in zip(old[field],new[field])), default=0),
        "before_repeat_max_abs": max((abs(a-b) for run in receipts["before"][2:] for a,b in zip(base,run[field])), default=0),
        "after_repeat_max_abs": max((abs(a-b) for run in receipts["after"][2:] for a,b in zip(migrated,run[field])), default=0),
    }
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
report = {
    "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"], "ld_library_path": env.get("LD_LIBRARY_PATH", ""),
    "input_hashes": {p.name: sha(p) for p in sorted(namelist.parent.glob("*.def"))},
    "binary_hashes": {"before": sha(before), "after": sha(after)},
    "samples": samples, "comparison": comparisons,
}
(out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))

#!/usr/bin/env python3
"""Sample per-thread CPU ticks of one short four-rank Opt run; not a timing run."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

image, volume, inputs, output, threads = sys.argv[1:]
inputs, output = str(Path(inputs).resolve()), Path(output).resolve()
output.mkdir(parents=True, exist_ok=True)
name = f"mvmc-585-observe-{os.getpid()}"
cpu_set = os.environ.get("MVMC_BENCH_CPUSET", "0-3")
inner_threads = os.environ.get("MVMC_BENCH_INNER_THREADS", "1")
cmd = ["docker", "run", "--rm", "--name", name, "--platform=linux/amd64", f"--cpuset-cpus={cpu_set}", "--shm-size=1g", "-v", f"{inputs}:/input:ro", "-v", f"{output}:/results", "-v", f"{volume}:/target", "-w", "/input/opt", "-e", "OPENBLAS_NUM_THREADS=1", "-e", "OMP_NUM_THREADS=1", "-e", f"MVMC_RS_INNER_THREADS={inner_threads}", "-e", "RAYON_NUM_THREADS=4", "-e", f"MVMC_RS_TENFERRO_THREADS={threads}", "-e", "MVMC_RS_SR_BACKEND=tenferro", "--entrypoint", "bash", image, "-lc", "timeout --kill-after=5s 120s /opt/mpich/bin/mpirun -n 4 /target/release/mvmc /input/opt/namelist.def --nsteps 20 --nsmp 1 --initial-def none --out-dir /results/opt"]
previous = {}
seen = {}
samples = []
clock_ticks = os.sysconf("SC_CLK_TCK")
with (output / "run.txt").open("w") as log:
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        while proc.poll() is None:
            listing = subprocess.run(["docker", "top", name, "-eo", "pid,args"], capture_output=True, text=True)
            now = time.monotonic()
            for line in listing.stdout.splitlines()[1:]:
                fields = line.strip().split(None, 1)
                if len(fields) != 2 or not fields[1].startswith("/target/release/mvmc "):
                    continue
                pid = int(fields[0])
                taskdir = Path(f"/proc/{pid}/task")
                try:
                    tasks = list(taskdir.iterdir())
                except FileNotFoundError:
                    continue
                count = 0
                active = []
                for task in tasks:
                    try:
                        stat = (task / "stat").read_text()
                        fields = stat[stat.rfind(")") + 2:].split()
                        ticks = int(fields[11]) + int(fields[12])
                    except (FileNotFoundError, ProcessLookupError):
                        continue
                    tid = int(task.name)
                    key = (pid, tid)
                    count += 1
                    if key in previous and ticks > previous[key]:
                        active.append(tid)
                        seen.setdefault(pid, set()).add(tid)
                    previous[key] = ticks
                samples.append({"monotonic": now, "pid": pid, "thread_count": count, "active_tids": active})
            time.sleep(0.1)
    finally:
        if proc.poll() is None:
            subprocess.run(["docker", "stop", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            proc.wait()
if proc.returncode != 0:
    raise SystemExit(f"container failed: {proc.returncode}; see {output / 'run.txt'}")
if len(seen) != 4:
    raise SystemExit(f"observed CPU ticks for {len(seen)} of 4 ranks; sampling was incomplete")
report = {"threads_per_engine": int(threads), "cpu_set": cpu_set, "inner_threads": int(inner_threads), "clock_ticks_per_second": clock_ticks, "distinct_active_tids_by_pid": {str(pid): sorted(tids) for pid, tids in seen.items()}, "samples": samples}
(output / "thread-samples.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"threads_per_engine": int(threads), "ranks": len(seen), "sample_count": len(samples), "distinct_active_counts": {str(pid): len(tids) for pid, tids in seen.items()}}, indent=2))

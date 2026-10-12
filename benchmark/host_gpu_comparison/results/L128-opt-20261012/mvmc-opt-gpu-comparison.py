import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

rust_binary, julia_source, input_file, output = map(Path, sys.argv[1:])
output.mkdir(parents=True, exist_ok=True)
environment = dict(os.environ, CUDA_VISIBLE_DEVICES="1", OPENBLAS_NUM_THREADS="1",
                   OMP_NUM_THREADS="1", JULIA_NUM_THREADS="1", JULIA_NUM_GC_THREADS="1",
                   MVMC_RS_SAMPLER_BACKEND="cuda", MVMC_RS_SR_BACKEND="cuda",
                   MVMC_RS_MEASURE_BACKEND="cuda", MVMC_RS_MEASURE_PF_BACKEND="cuda",
                   MVMC_C_TIMER="1")
environment["LD_LIBRARY_PATH"] = ":".join([
    "/home/terasaki/.julia/artifacts/369f5472daa95aec77a8de76aa620f1b8f13293f/lib",
    "/home/terasaki/.julia/artifacts/6b0fef1d16bb8733a19f60c6efec768f99e5f90d/lib",
])
commands = {
    "rust": [str(rust_binary), str(input_file), str(output / "rust-output")],
    "julia": ["julia", "+1.13.1", "--startup-file=no",
              f"--project={julia_source / 'test/gpu'}",
              str(Path(__file__).with_suffix(".jl")), str(input_file), str(output / "julia-output")],
}
processes, logs, samples, startup = {}, {}, {}, {}

def until(name, marker):
    while True:
        line = processes[name].stdout.readline()
        if not line:
            raise RuntimeError(f"{name} exited {processes[name].poll()} before {marker}")
        logs[name].write(line)
        logs[name].flush()
        print(f"{name}: {line}", end="", flush=True)
        match = re.search(r"rep=(\d+) seconds=([0-9.eE+-]+)", line)
        if match:
            samples[name].append({"rep": int(match[1]), "seconds": float(match[2])})
        if line.rstrip() == marker:
            return

def run(name, rep):
    processes[name].stdin.write(f"{rep}\n")
    processes[name].stdin.flush()
    until(name, f"DONE {rep}")
    (output / "timings.json").write_text(json.dumps({"startup": startup, "samples": samples}, indent=2)+"\n")

try:
    for name in ("rust", "julia"):
        logs[name] = (output / f"{name}.log").open("w")
        samples[name] = []
        start = time.monotonic()
        processes[name] = subprocess.Popen(commands[name], env=environment,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)
        until(name, "READY")
        startup[name] = time.monotonic()-start
        run(name, 0)
    for rep in range(1, 6):
        for name in (("rust", "julia") if rep % 2 else ("julia", "rust")):
            run(name, rep)
finally:
    for process in processes.values():
        if process.poll() is None:
            process.stdin.write("quit\n")
            process.stdin.flush()
            process.wait(timeout=60)
    for log in logs.values():
        log.close()

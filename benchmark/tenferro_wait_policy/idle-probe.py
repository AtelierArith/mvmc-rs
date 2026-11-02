#!/usr/bin/env python3
"""Sample Linux process CPU ticks for two seconds after one Gram; not a throughput benchmark."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

binary = str(Path(sys.argv[1]).resolve())
hz = os.sysconf("SC_CLK_TCK")
def ticks(pid):
    text = Path(f"/proc/{pid}/stat").read_text()
    fields = text[text.rfind(")") + 2:].split()
    return int(fields[11]) + int(fields[12])

records = []
for rep in range(6):
    policies = ("default", "zero") if rep % 2 == 0 else ("zero", "default")
    for policy in policies:
        env = dict(os.environ, RAYON_NUM_THREADS="4", MVMC_RS_TENFERRO_THREADS="4", OPENBLAS_NUM_THREADS="1")
        env.pop("MVMC_RS_TENFERRO_SPIN_POLLS", None)
        if policy == "zero":
            env["MVMC_RS_TENFERRO_SPIN_POLLS"] = "0"
        proc = subprocess.Popen(["taskset", "-c", "0-3", binary], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            line = proc.stdout.readline().split()
            if len(line) != 2 or line[0] != "READY" or int(line[1]) != proc.pid:
                raise RuntimeError(f"unexpected probe output: {line}")
            start = time.monotonic()
            initial = ticks(proc.pid)
            samples = []
            while time.monotonic() - start < 2.0:
                time.sleep(0.02)
                samples.append({"elapsed": time.monotonic() - start, "cpu_seconds": (ticks(proc.pid) - initial) / hz})
            if proc.wait(timeout=5) != 0:
                raise RuntimeError(proc.stderr.read())
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        records.append({"rep": rep, "policy": policy, "warmup": rep == 0, "samples": samples})
print(json.dumps({"clock_ticks_per_second": hz, "records": records}, indent=2))

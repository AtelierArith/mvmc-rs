"""Paired warmed resident_physcal_worker calls; run through uv, no dependencies."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("before")
    p.add_argument("after")
    p.add_argument("output", type=Path)
    p.add_argument("--container", required=True)
    p.add_argument("--l16-input", required=True)
    p.add_argument("--l16-params", required=True)
    p.add_argument("--root", default="/work")
    p.add_argument("--cpu", default="17")
    p.add_argument("--reps", type=int, default=9)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    root = Path(a.root)
    cases = {
        "L16": (a.l16_input, a.l16_params, 64, 20),
        "L128": ("benchmark/cpu_scaling_l128/inputs/phys/namelist.def", "benchmark/cpu_scaling_l128/inputs/phys/zqp_opt.dat", 64, 20),
        "L128-api": ("benchmark/cpu_scaling_l128/inputs/phys/namelist.def", "benchmark/cpu_scaling_l128/inputs/phys/zqp_opt.dat", 320, 1),
    }
    summary = {"reps": a.reps, "excluded_warmup_calls_per_worker": 1, "cpu": a.cpu, "cases": {}}

    def numeric(path):
        tokens = iter(path.read_text().split())
        assert next(tokens) == "seconds"
        seconds = float(next(tokens))
        assert next(tokens) == "energy"
        values = [float(next(tokens))]
        counts = []
        for name in ("one", "two", "two_ex"):
            assert next(tokens) == name
            n = int(next(tokens))
            counts.append(n)
            values.extend(float(next(tokens)) for _ in range(n))
        assert next(tokens, None) is None
        return seconds, counts, values

    for case, (input_path, params, walkers, windows) in cases.items():
        processes, logs, traffic = {}, {}, {}

        def until(kind, marker):
            while True:
                line = processes[kind].stdout.readline()
                if not line:
                    raise RuntimeError(f"{case}/{kind} stopped before {marker}")
                logs[kind].write(line)
                logs[kind].flush()
                if "traffic=RawExecCounters" in line:
                    traffic[kind] = {k: int(v) for k, v in re.findall(r"(\w+): (\d+)", line)}
                if line.strip() == marker:
                    return

        try:
            for kind, binary in (("before", a.before), ("after", a.after)):
                output = a.output / f"{case}-{kind}"
                logs[kind] = (a.output / f"{case}-{kind}.log").open("w")
                command = ["docker", "exec", "-i", a.container, "taskset", "-c", a.cpu, binary, str(root / input_path), str(root / params), str(root / output), str(walkers), str(windows), "128", "128"]
                processes[kind] = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                until(kind, "READY")
            for rep in range(a.reps + 1):
                order = ("before", "after") if rep % 2 else ("after", "before")
                for kind in order:
                    processes[kind].stdin.write(f"{rep}\n")
                    processes[kind].stdin.flush()
                    until(kind, f"DONE {rep}")
            times = {"before": [], "after": []}
            max_diff = 0.0
            for rep in range(1, a.reps + 1):
                before = numeric(a.output / f"{case}-before/rep-{rep}.txt")
                after = numeric(a.output / f"{case}-after/rep-{rep}.txt")
                assert before[1] == after[1]
                assert len(before[2]) == len(after[2])
                for x, y in zip(before[2], after[2]):
                    assert math.isfinite(x) and math.isfinite(y)
                    assert abs(x-y) <= 1e-12 + 1e-12 * max(abs(x), abs(y))
                    max_diff = max(max_diff, abs(x-y))
                times["before"].append(before[0])
                times["after"].append(after[0])
            ratios = [y/x for x, y in zip(times["before"], times["after"])]
            hashes = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(Path(input_path).parent.iterdir()) if f.is_file()}
            hashes[params] = hashlib.sha256(Path(params).read_bytes()).hexdigest()
            summary["cases"][case] = {"input": input_path, "params": params, "walkers": walkers, "windows": windows, "warmup": 128, "interval": 128, "seed": 79, "seconds": times, "medians": {k: statistics.median(v) for k, v in times.items()}, "paired_ratios": ratios, "median_paired_ratio": statistics.median(ratios), "traffic": traffic, "mean_max_abs_difference": max_diff, "input_hashes": hashes}
            (a.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(case, summary["cases"][case]["medians"], "paired ratio", statistics.median(ratios), flush=True)
        finally:
            for proc in processes.values():
                if proc.poll() is None:
                    proc.stdin.write("quit\n")
                    proc.stdin.flush()
                    proc.wait(timeout=30)
            for log in logs.values():
                log.close()


if __name__ == "__main__":
    main()

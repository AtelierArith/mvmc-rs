"""Alternate matched before/after public API calls on one physical CUDA device.

Run through uv run --no-project python. Build the same pfaffian_e2e_worker.rs
in each checkout/target volume first. No Julia or C reference runtime is used.
"""
import argparse
import hashlib
import json
import math
import queue
import statistics
import subprocess
import threading
import time
from pathlib import Path


def command(*args):
    return subprocess.check_output([str(a) for a in args], text=True).strip()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest(checkout):
    """Record compiled sources even when a measured checkout has local edits."""
    roots = ("gpu/mvmc-gpu-cuda/src", "gpu/tenferro-ext/src", "crates")
    return {str(p.relative_to(checkout)): digest(p)
            for root in roots for p in sorted((checkout / root).rglob("*"))
            if p.is_file() and p.suffix in (".rs", ".cu", ".cuh", ".toml")}


def results(path):
    lines = iter(path.read_text().splitlines())
    discrete, sections = [next(lines)], {}
    for line in lines:
        if line.startswith(("cg_iterations", "cg_fallback")):
            discrete.append(line)
        else:
            name, count = line.split()
            sections[name] = [float(next(lines)) for _ in range(int(count))]
    return discrete, sections


def compare(left, right, repeat=False):
    ld, ls = results(left)
    rd, rs = results(right)
    assert ld == rd, f"discrete output differs: {left} vs {right}"
    assert ls.keys() == rs.keys(), "output section mismatch"
    report = {}
    for name, values in ls.items():
        other = rs[name]
        assert len(values) == len(other), name
        absolute, scaled, first = 0.0, 0.0, None
        for i, (a, b) in enumerate(zip(values, other)):
            assert math.isfinite(a) and math.isfinite(b), (name, i)
            error = abs(a - b)
            scale = max(1.0, abs(a), abs(b))
            absolute = max(absolute, error)
            scaled = max(scaled, error / scale)
            if first is None and error > 64 * math.ulp(1.0) * scale:
                first = {"index": i, "before": a, "after": b, "absolute": error}
            if repeat:
                # Same implementation/input/seed, fixed reduction geometry.
                # This bound is deliberately separate from cross-provider error.
                assert error <= 1e-14 + 1e-13 * max(abs(a), abs(b)), (
                    "repeatability", name, i, a, b
                )
        report[name] = {"max_absolute": absolute, "max_unit_scaled": scaled,
                        "first_over_64_epsilon": first}
    return report


class Worker:
    def __init__(self, label, checkout, volume, args):
        self.label = label
        self.queue = queue.Queue()
        self.stdout_log = (args.output / f"{label}.stdout.log").open("w")
        self.stderr_log = (args.output / f"{label}.stderr.log").open("w")
        argv = [
            "docker", "run", "--rm", "-i", "--name",
            "mvmc731-e2e-" + hashlib.sha256(str(args.output).encode()).hexdigest()[:12] + "-" + label,
            "--gpus", f"device={args.device}",
            "-e", "OPENBLAS_NUM_THREADS=1", "-e", "OMP_NUM_THREADS=1",
            "-e", "RAYON_NUM_THREADS=1", "-e", "MVMC_RS_INNER_THREADS=1",
            "-v", f"{checkout}:/work:ro", "-v", f"{volume}:/cuda-target:ro",
            "-v", f"{args.input.parent}:/inputs:ro", "-v", f"{args.output}:/results",
            "-w", "/work", args.image,
            "taskset", "-c", str(args.cpu), "/cuda-target/release/examples/pfaffian_e2e_worker",
            args.mode, f"/inputs/{args.input.name}", f"/results/{label}",
            str(args.walkers), str(args.steps), str(args.warmup), str(args.interval),
        ]
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.stderr_log, text=True, bufsize=1)

        def consume():
            for line in self.proc.stdout:
                self.stdout_log.write(line)
                self.stdout_log.flush()
                self.queue.put(line.strip())
            self.queue.put(None)

        threading.Thread(target=consume, daemon=True).start()
        self.read_until(lambda line: line.startswith("rep,seconds,"), args.timeout)

    def read_until(self, predicate, timeout):
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"{self.label}: inspect its live Docker worker and logs")
            line = self.queue.get(timeout=remaining)
            if line is None:
                raise RuntimeError(f"{self.label} exited; inspect stderr log")
            if predicate(line):
                return line

    def run(self, rep, timeout):
        self.proc.stdin.write(f"{rep}\n")
        self.proc.stdin.flush()
        row = self.read_until(lambda line: line.startswith(f"{rep},"), timeout).split(",")
        assert len(row) == 8, row
        seconds = float(row[1])
        assert math.isfinite(seconds) and seconds > 0, row
        return dict(zip(("rep", "seconds", "upload_calls", "upload_bytes", "download_calls",
                         "download_bytes", "synchronizations", "scopes"),
                        [int(row[0]), seconds, *map(int, row[2:])]))

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.write("quit\n")
            self.proc.stdin.flush()
        status = self.proc.wait(timeout=60)
        assert status == 0, (self.label, status)
        self.stdout_log.close()
        self.stderr_log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--before-volume", required=True)
    parser.add_argument("--after-volume", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("opt", "phys"), required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="1")
    parser.add_argument("--cpu", type=int, default=17)
    parser.add_argument("--walkers", type=int, default=320)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--warmup", type=int, default=128)
    parser.add_argument("--interval", type=int, default=128)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    args.steps = args.steps if args.steps is not None else (20 if args.mode == "opt" else 1)
    assert args.reps >= 2 and args.steps > 0 and args.walkers > 0
    for name in ("before", "after", "input", "output"):
        setattr(args, name, getattr(args, name).resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    worker_path = Path("gpu/mvmc-gpu-cuda/examples/pfaffian_e2e_worker.rs")
    lock_path = Path("gpu/mvmc-gpu-cuda/Cargo.lock")
    assert digest(args.before / worker_path) == digest(args.after / worker_path)
    assert digest(args.before / lock_path) == digest(args.after / lock_path)
    metadata = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "before_revision": command("git", "-C", args.before, "rev-parse", "HEAD"),
        "after_revision": command("git", "-C", args.after, "rev-parse", "HEAD"),
        "source_sha256": {"before": source_manifest(args.before),
                          "after": source_manifest(args.after)},
        "worker_sha256": digest(args.after / worker_path),
        "cuda_lock_sha256": digest(args.after / lock_path),
        "image_id": command("docker", "image", "inspect", args.image, "--format", "{{.Id}}"),
        "gpu": command("nvidia-smi", "--id", args.device,
                       "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader"),
        "input_sha256": {p.name: digest(p) for p in sorted(args.input.parent.iterdir()) if p.is_file()},
        "scope": "warm public API including parsing, fresh initialization, sampling, final downloads and internal teardown; excludes worker startup/JIT warmup and post-call serialization",
        "counter_scope": "explicit tenferro-ext raw-execution calls only; library-internal operations not counted",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    workers = {}
    records = []
    try:
        workers["before"] = Worker("before", args.before, args.before_volume, args)
        workers["after"] = Worker("after", args.after, args.after_volume, args)
        for rep in range(args.reps + 1):
            order = ("before", "after") if rep % 2 == 0 else ("after", "before")
            for label in order:
                stamp = {"started_unix": time.time(), "host_loadavg": Path("/proc/loadavg").read_text().strip()}
                result = workers[label].run(rep, args.timeout)
                result.update(label=label, warmup=rep == 0, **stamp)
                records.append(result)
                (args.output / "timings.json").write_text(json.dumps(records, indent=2) + "\n")
                print(json.dumps(result), flush=True)
        comparisons = {}
        for label in workers:
            for rep in range(1, args.reps + 1):
                comparisons[f"{label}_repeat_{rep}"] = compare(
                    args.output / label / "rep-0.txt", args.output / label / f"rep-{rep}.txt", repeat=True)
        for rep in range(args.reps + 1):
            comparisons[f"cross_revision_{rep}"] = compare(
                args.output / "before" / f"rep-{rep}.txt", args.output / "after" / f"rep-{rep}.txt")
            traffic = [r for r in records if r["rep"] == rep]
            for field in ("upload_calls", "upload_bytes", "download_calls", "download_bytes",
                          "synchronizations", "scopes"):
                assert traffic[0][field] == traffic[1][field], ("traffic regression", rep, field, traffic)
        (args.output / "numerical.json").write_text(json.dumps(comparisons, indent=2) + "\n")
        medians = {label: statistics.median(r["seconds"] for r in records if r["label"] == label and not r["warmup"])
                   for label in workers}
        summary = {"median_seconds": medians, "speedup": medians["before"] / medians["after"],
                   "cross_revision_numerics": "observed errors only; independent numerical/trajectory validation required"}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
    finally:
        # Never terminate a live CUDA call on an observation timeout. Leave its
        # process/logs for inspection; normal completed runs exit via stdin.
        if len(records) == 2 * (args.reps + 1):
            for worker in workers.values():
                worker.close()


if __name__ == "__main__":
    main()

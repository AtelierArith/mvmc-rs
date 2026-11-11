"""Matched persistent CPU workers. Run with uv run --no-project python."""
import argparse
import json
import math
import queue
import statistics
import subprocess
import threading
import time
from pathlib import Path

from run_gpu_e2e import command, digest


def compare(left, right, repeat):
    def read(path):
        lines = iter(path.read_text().splitlines())
        header, sections = next(lines), {}
        for line in lines:
            kind, name, count = line.split()
            parse = int if kind == "i" else float
            sections[name] = (kind, [parse(next(lines)) for _ in range(int(count))])
        return header, sections

    lh, ls = read(left)
    rh, rs = read(right)
    report = {"rng_header_equal": lh == rh, "sections": {}}
    assert ls.keys() == rs.keys()
    if repeat:
        assert lh == rh, "same-implementation RNG count/index differs"
    for name, (kind, values) in ls.items():
        other = rs[name][1]
        assert len(values) == len(other)
        if kind == "i":
            mismatch = next((i for i, (a, b) in enumerate(zip(values, other)) if a != b), None)
            report["sections"][name] = {"first_mismatch": mismatch}
            if repeat:
                assert mismatch is None, ("same-implementation discrete output", name, mismatch)
        else:
            absolute, scaled, first = 0.0, 0.0, None
            for i, (a, b) in enumerate(zip(values, other)):
                assert math.isfinite(a) and math.isfinite(b), (name, i)
                error = abs(a-b)
                scale = max(1.0, abs(a), abs(b))
                absolute = max(absolute, error)
                scaled = max(scaled, error/scale)
                if first is None and error > 64 * math.ulp(1.0) * scale:
                    first = {"index": i, "before": a, "after": b}
                if repeat:
                    assert error <= 1e-14 + 1e-13 * max(abs(a), abs(b)), (name, i, a, b)
            report["sections"][name] = {"max_absolute": absolute, "max_unit_scaled": scaled,
                                        "first_over_64_epsilon": first}
    return report


class Worker:
    def __init__(self, label, volume, args):
        self.label = label
        self.queue = queue.Queue()
        self.out = (args.output / f"{label}.stdout.log").open("w")
        self.err = (args.output / f"{label}.stderr.log").open("w")
        argv = ["docker", "run", "--rm", "-i", "--user", "1000:1000",
                "-e", "OPENBLAS_NUM_THREADS=1", "-e", "OMP_NUM_THREADS=1",
                "-e", "BLIS_NUM_THREADS=1", "-e", "RAYON_NUM_THREADS=1",
                "-e", "MVMC_RS_INNER_THREADS=1",
                "-v", f"{volume}:/target:ro", "-v", f"{args.input.parent}:/inputs:ro",
                "-v", f"{args.output}:/results", args.image,
                "taskset", "-c", str(args.cpu), "/target/release/examples/pfaffian_e2e_worker",
                args.mode, f"/inputs/{args.input.name}", f"/results/{label}", str(args.steps), args.dtype]
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.err, text=True, bufsize=1)

        def consume():
            for line in self.proc.stdout:
                self.out.write(line)
                self.out.flush()
                self.queue.put(line.strip())
            self.queue.put(None)

        threading.Thread(target=consume, daemon=True).start()
        self.until("READY", args.timeout)

    def until(self, prefix, timeout):
        deadline = time.monotonic() + timeout
        while True:
            line = self.queue.get(timeout=max(0.01, deadline-time.monotonic()))
            if line is None:
                raise RuntimeError(f"{self.label}: inspect stderr")
            if line.startswith(prefix):
                return line

    def run(self, rep, timeout):
        self.proc.stdin.write(f"{rep}\n")
        self.proc.stdin.flush()
        seconds = float(self.until(f"RESULT,{rep},", timeout).split(",")[2])
        assert math.isfinite(seconds) and seconds > 0
        return seconds

    def close(self):
        self.proc.stdin.write("quit\n")
        self.proc.stdin.flush()
        assert self.proc.wait(timeout=60) == 0
        self.out.close()
        self.err.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("before", "after", "input", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    for name in ("before-volume", "after-volume", "image"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--mode", choices=("opt", "phys"), required=True)
    p.add_argument("--dtype", choices=("real", "cmp"), required=True)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--cpu", type=int, default=17)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--timeout", type=int, default=1800)
    args = p.parse_args()
    assert args.reps >= 2 and args.steps > 0
    for name in ("before", "after", "input", "output"):
        setattr(args, name, getattr(args, name).resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    worker = Path("crates/mvmc-cli/examples/pfaffian_e2e_worker.rs")
    assert digest(args.before/worker) == digest(args.after/worker)
    paths = [p for p in (args.after/"crates").rglob("*") if p.suffix in (".rs", ".toml")]
    metadata = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "revisions": {s: command("git", "-C", getattr(args, s), "rev-parse", "HEAD") for s in ("before", "after")},
        "source_sha256": {s: {str(p.relative_to(args.after)): digest(getattr(args, s)/p.relative_to(args.after))
                              for p in paths if (getattr(args, s)/p.relative_to(args.after)).exists()}
                          for s in ("before", "after")},
        "lock_sha256": {s: digest(getattr(args, s)/"Cargo.lock") for s in ("before", "after")},
        "input_sha256": {p.name: digest(p) for p in sorted(args.input.parent.iterdir()) if p.is_file()},
        "image_id": command("docker", "image", "inspect", args.image, "--format", "{{.Id}}"),
        "scope": "public CPU API, parsing, initialization, sampling, optimizer and normal file IO; excludes worker startup, serialization and returned-state teardown; seed79, opt nsmp=steps",
        "numerics": "same-implementation repeat bounded; cross-provider trajectory differences require first-divergence analysis, no blanket tolerance",
    }
    (args.output/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    workers, records = {}, []
    try:
        for label in ("before", "after"):
            workers[label] = Worker(label, getattr(args, label+"_volume"), args)
        for rep in range(args.reps+1):
            for label in (("before", "after") if rep % 2 == 0 else ("after", "before")):
                row = {"label": label, "rep": rep, "warmup": rep == 0, "started_unix": time.time(),
                       "loadavg": Path("/proc/loadavg").read_text().strip()}
                row["seconds"] = workers[label].run(rep, args.timeout)
                records.append(row)
                (args.output/"timings.json").write_text(json.dumps(records, indent=2)+"\n")
                print(json.dumps(row), flush=True)
        numerics = {f"{label}_repeat_{rep}": compare(args.output/label/"rep-0.txt", args.output/label/f"rep-{rep}.txt", True)
                    for label in workers for rep in range(1, args.reps+1)}
        numerics.update({f"cross_{rep}": compare(args.output/"before"/f"rep-{rep}.txt", args.output/"after"/f"rep-{rep}.txt", False)
                         for rep in range(args.reps+1)})
        (args.output/"numerical.json").write_text(json.dumps(numerics, indent=2)+"\n")
        medians = {s: statistics.median(r["seconds"] for r in records if r["label"] == s and not r["warmup"]) for s in workers}
        summary = {"median_seconds": medians, "speedup": medians["before"]/medians["after"]}
        (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        print(json.dumps(summary), flush=True)
    finally:
        if len(records) == 2*(args.reps+1):
            for worker in workers.values():
                worker.close()


if __name__ == "__main__":
    main()

"""Strict retained evidence for a rank-local callback failure after step zero."""
import argparse
import math
from pathlib import Path

from compare_mpi_issue179 import read, validate_trace


def validate(directory, ranks, failure_rank):
    if not 0 <= failure_rank < ranks:
        raise ValueError("injected callback rank outside world")
    for rank in range(ranks):
        rows = read(directory / f"rank-{rank}.txt")
        if rows.get("d:status") != ["1"] or rows.get("d:steps") != ["3"]:
            raise ValueError(f"rank {rank}: missing coordinated callback failure")
        if (rows.get("d:callback-injected-rank") != [str(failure_rank)] or
                rows.get("d:callback-calls") != ["1"] or
                rows.get("d:callback-local-errors") != [str(int(rank == failure_rank))]):
            raise ValueError(f"rank {rank}: callback identity/count mismatch")
        reason = (directory / f"callback-result-rank-{rank}.txt").read_text()
        if "callback" not in reason or (rank == failure_rank and
                f"issue179 injected rank {failure_rank} callback failure" not in reason):
            raise ValueError(f"rank {rank}: returned error was not the actual injected callback")
        if "n:sr-step-0" not in rows or any(key.startswith("n:sr-step-") and
                key != "n:sr-step-0" for key in rows):
            raise ValueError(f"rank {rank}: callback was not the first-step terminal boundary")
        events = rows.get("d:trace-events", [])
        if len(events) != 1 or not events[0].isdecimal() or int(events[0]) <= 0:
            raise ValueError(f"rank {rank}: missing actual sampling trace")
        expected = {f"d:trace-{i:06}" for i in range(int(events[0]))}
        actual = {key for key in rows if key.startswith("d:trace-") and key != "d:trace-events"}
        if actual != expected or sum(rows[key][0] == "8" for key in expected) != 1:
            raise ValueError(f"rank {rank}: another sampler step occurred or trace is incomplete")
        validate_trace(rows, rank, expected_checkpoints=1)
        for key in ("d:initial-rng", "d:rng"):
            if len(rows.get(key, [])) != 624:
                raise ValueError(f"rank {rank}: missing RNG state")
    for filename in ("zvo_out.dat", "zvo_var.dat"):
        lines = (directory / filename).read_text().splitlines()
        if len(lines) != 1 or not lines[0].split() or any(
                not math.isfinite(float(value)) for value in lines[0].split()):
            raise ValueError(f"{filename}: first/second-step output boundary mismatch")
    if (directory / "zqp_opt.dat").exists():
        raise ValueError("final-success parameter output written after callback failure")
    return f"COORDINATED_CALLBACK_FAILURE ranks={ranks} completed_steps=1 requested_steps=3"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--ranks", type=int, choices=(2, 4), required=True)
    parser.add_argument("--failure-rank", type=int, required=True)
    args = parser.parse_args()
    print(validate(args.directory, args.ranks, args.failure_rank))

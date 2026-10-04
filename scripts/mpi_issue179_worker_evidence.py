"""Validate actual MPI worker observations, not post-run capacity alone."""
import argparse
from pathlib import Path
import re


def validate(directory, ranks, workers):
    eligible = parallel = 0
    labels = []
    for rank in range(ranks):
        path = directory / f"workers-rank-{rank}.txt"
        lines = path.read_text().splitlines()
        if len(lines) != 2:
            raise ValueError(f"{path}: missing actual snapshot")
        values = {}
        for name, value in re.findall(r"(\w+)=(\d+)(?=\s|$)", " ".join(lines)):
            if name in values:
                raise ValueError(f"{path}: duplicate field {name}")
            values[name] = int(value)
        required = {"requested", "configured", "observed_pool", "threshold", "qp_work",
                    "kernel_parallel_calls", "kernel_serial_calls", "kernel_worker_entries",
                    "kernel_workers_seen"}
        required |= {f"kernel_{prefix}{kind}_items" for kind in ("qp", "term")
                     for prefix in ("", "parallel_", "serial_")}
        allowed = required | {"effective_qp_workers", "capacity_probe_after_run"}
        numeric_text = re.sub(r"kernel_worker_ids=\[[\d, ]*\]", "", " ".join(lines))
        if (not required <= values.keys() or values.keys() - allowed or
                any(not re.fullmatch(r"\w+=\d+", token) for token in numeric_text.split())):
            raise ValueError(f"{path}: incomplete snapshot")
        if any(values[key] != workers for key in ("requested", "configured", "observed_pool")):
            raise ValueError(f"{path}: worker configuration mismatch")
        ids = re.search(r"kernel_worker_ids=\[([\d, ]*)\]$", lines[1])
        if ids is None or " ".join(lines).count("kernel_worker_ids=") != 1:
            raise ValueError(f"{path}: malformed actual worker IDs")
        if ids[1].strip() and not re.fullmatch(r"\s*\d+(?:\s*,\s*\d+)*\s*", ids[1]):
            raise ValueError(f"{path}: malformed actual worker ID list")
        worker_ids = [int(item.strip()) for item in ids[1].split(",") if item.strip()]
        if (len(set(worker_ids)) != len(worker_ids) or
                len(worker_ids) != values["kernel_workers_seen"] or
                any(item >= workers for item in worker_ids)):
            raise ValueError(f"{path}: invalid actual worker IDs")
        for kind in ("qp", "term"):
            if values[f"kernel_{kind}_items"] != sum(
                    values[f"kernel_{prefix}_{kind}_items"] for prefix in ("parallel", "serial")):
                raise ValueError(f"{path}: inconsistent {kind} item counts")
        entries = values["kernel_worker_entries"]
        if bool(entries) != bool(worker_ids) or entries < len(worker_ids):
            raise ValueError(f"{path}: inconsistent actual worker entries")
        parallel_qp = values["kernel_parallel_qp_items"]
        parallel_items = parallel_qp + values["kernel_parallel_term_items"]
        serial_items = values["kernel_serial_qp_items"] + values["kernel_serial_term_items"]
        if parallel_items and (workers == 1 or entries < parallel_items or
                               not values["kernel_parallel_calls"]):
            raise ValueError(f"{path}: unsupported parallel kernel observation")
        if values["kernel_parallel_calls"] and (workers == 1 or not entries):
            raise ValueError(f"{path}: parallel calls lack actual worker entries")
        if serial_items and not values["kernel_serial_calls"]:
            raise ValueError(f"{path}: serial items lack actual serial calls")
        if workers == 1 and (entries or values["kernel_parallel_calls"]):
            raise ValueError(f"{path}: unexpected parallel worker-1 execution")
        has_qp = values["kernel_qp_items"] > 0
        rank_eligible = workers > 1 and values["qp_work"] > values["threshold"] and has_qp
        eligible += rank_eligible
        parallel += parallel_qp
        if rank_eligible and not parallel_qp:
            raise ValueError(f"{path}: eligible QP work never entered parallel kernel")
        label = ("ACTUAL_PARALLEL_QP" if parallel_qp else
                 "NO_QP_KERNEL_WORK" if not has_qp else
                 "SERIAL_WORKER1" if workers == 1 else "SERIAL_SMALL_QP")
        labels.append(f"rank={rank} domain={label} actual_worker_entries={entries}")
    if eligible and not parallel:
        raise ValueError("eligible MPI worker cell has no actual parallel QP work")
    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--ranks", type=int, choices=(2, 4), required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2, 4), required=True)
    args = parser.parse_args()
    print("\n".join(validate(args.directory, args.ranks, args.workers)))

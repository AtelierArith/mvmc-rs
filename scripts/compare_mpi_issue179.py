"""Strict optional evidence comparison; execute with uv run --no-project python.

No default numerical tolerance: supply measured bounds and a justification file.
Discrete records are always exact, before any computed-number comparison.
"""
import argparse
import math
from pathlib import Path
import re


def read(path):
    records = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) < 2 or fields[0] in records:
            raise ValueError(f"{path}: empty/duplicate/malformed record: {line}")
        records[fields[0]] = fields[1:]
    if not records:
        raise ValueError(f"{path}: empty evidence")
    return records


def validate_trace(rows, rank, expected_checkpoints=None):
    counts = rows.get("d:trace-events", [])
    if len(counts) != 1 or not counts[0].isdecimal() or int(counts[0]) == 0:
        raise ValueError(f"rank {rank}: missing/malformed/empty actual sampling trace")
    expected = {f"d:trace-{i:06}" for i in range(int(counts[0]))}
    actual = {key for key in rows if key.startswith("d:trace-") and key != "d:trace-events"}
    if actual != expected:
        raise ValueError(f"rank {rank}: missing/extra sampling events")
    shapes = {0: 2, 1: 6, 2: 8, 3: 7, 4: 7, 5: 7, 6: 6, 7: 3, 9: 2}
    checkpoints = 0
    for index, key in enumerate(sorted(expected)):
        values = [int(value) for value in rows[key]]
        if values and values[0] == 8:
            pos = 1
            for field in range(7):
                if pos >= len(values) or values[pos] < 0:
                    raise ValueError(f"rank {rank}: malformed checkpoint {key}")
                size = values[pos]
                pos += 1
                if pos + size > len(values) or (field == 5 and size != 10) or (field == 6 and size != 624):
                    raise ValueError(f"rank {rank}: malformed checkpoint field {key}")
                if field == 6 and any(not 0 <= word < 2**32 for word in values[pos:pos+size]):
                    raise ValueError(f"rank {rank}: malformed checkpoint RNG {key}")
                pos += size
            if pos != len(values):
                raise ValueError(f"rank {rank}: trailing checkpoint data {key}")
            checkpoints += 1
            continue
        if not values or len(values) != shapes.get(values[0]):
            raise ValueError(f"rank {rank}: malformed event {key}")
        if values[0] == 7 and (values[1] not in (0, 1) or not 0 <= values[2] < 2**32):
            raise ValueError(f"rank {rank}: malformed decision/draw {key}")
        if values[0] == 7 and (index == 0 or rows[f"d:trace-{index-1:06}"] != ["9", str(values[2])]):
            raise ValueError(f"rank {rank}: decision lacks its actual consumed word {key}")
        if values[0] == 9 and not 0 <= values[1] < 2**32:
            raise ValueError(f"rank {rank}: malformed consumed word {key}")
    if "d:steps" in rows and (len(rows["d:steps"]) != 1 or not 1 <= int(rows["d:steps"][0]) <= 3 or checkpoints != (int(rows["d:steps"][0]) if expected_checkpoints is None else expected_checkpoints)):
        raise ValueError(f"rank {rank}: missing per-step sampling checkpoints")
    if "d:sampling-draw-count" in rows and rows["d:sampling-draw-count"] != [str(sum(rows[key][0] == "9" for key in expected))]:
        raise ValueError(f"rank {rank}: incorrect actual sampling draw count")
    if "d:acceptance-events" in rows:
        decisions = [rows[key] for key in expected if rows[key][0] == "7"]
        counts = [str(sum(event[1] == value for event in decisions)) for value in ("1", "0")]
        if rows["d:acceptance-events"] != counts:
            raise ValueError(f"rank {rank}: acceptance metadata disagrees with actual decisions")


def compare_root(a, b, atol, rtol):
    aa, bb = [line.split() for line in a.read_text().splitlines()], [line.split() for line in b.read_text().splitlines()]
    if not aa or not bb or any(not row for row in aa + bb) or [len(row) for row in aa] != [len(row) for row in bb]:
        raise ValueError("root output missing/empty/shape mismatch")
    compare({"n:root-output": [v for row in aa for v in row]}, {"n:root-output": [v for row in bb for v in row]}, atol, rtol)


def read_c_flags(path):
    rows = read(path)
    if rows.keys() != {"flags", "written", "C_source_sha256", "enumerator_sha256", "contract_sha256"}:
        raise ValueError("incomplete C-written flags provenance")
    for key in ("C_source_sha256", "enumerator_sha256", "contract_sha256"):
        if len(rows[key]) != 1 or not re.fullmatch(r"[0-9a-f]{64}", rows[key][0]):
            raise ValueError("malformed C flags provenance")
    if rows["C_source_sha256"] != ["6c53cb832f93d6cbfd7cea955fbb693738af5536b913d36af32b98eed38c32d9"]:
        raise ValueError("C flags reader changed; review written-mask contract")
    flags, written = ([int(x) for x in rows[key]] for key in ("flags", "written"))
    if not flags or len(flags) != len(written) or not any(written) or any(x not in (0, 1) for x in written):
        raise ValueError("malformed C-written flags mask")
    return flags, written


def compare(a, b, atol, rtol, discrete_only=False, parameter_atol=None, parameter_rtol=None, c_flags=None):
    keys_a = {key for key in a if key.startswith("d:")} if discrete_only else a.keys()
    keys_b = {key for key in b if key.startswith("d:")} if discrete_only else b.keys()
    # All integer/discrete contracts are checked before computed floats.
    priority = {key: i for i, key in enumerate(("d:group", "d:seed", "d:configured-samples", "d:initial-rng", "d:chain-samples", "d:ele_idx", "d:ele_cfg", "d:ele_num", "d:local-counter", "d:reduced-counter", "d:counter", "d:rng"))}
    for key in sorted(keys_a & keys_b, key=lambda key: (4 if key.startswith("d:trace-") and key != "d:trace-events" else priority.get(key, 100), key)):
        if len(a[key]) != len(b[key]):
            raise ValueError(f"{key}: shape mismatch")
        if key.startswith("d:"):
            aa, bb = [int(x) for x in a[key]], [int(x) for x in b[key]]
            if re.fullmatch(r"d:sr-system-\d{6}-flags", key) and c_flags is not None:
                flags, written = c_flags
                if len(aa) != len(flags):
                    raise ValueError(f"{key}: C flags shape mismatch")
                for label, raw in (("Rust", aa), ("Julia", bb)):
                    for i, known in enumerate(written):
                        if known and raw[i] != flags[i]:
                            raise ValueError(f"{key}[{i}]: {label} C-written flag mismatch")
                    active_key = key.removesuffix("-flags") + "-active"
                    records = a if label == "Rust" else b
                    stem = key.removesuffix("-flags")
                    if active_key not in records and not (
                            records.get(stem + "-not-solved") == ["2"] and
                            records.get(stem + "-dimension") == ["0"]):
                        raise ValueError(f"{key}: missing actual active indices")
                    active = [int(x) for x in records.get(active_key, [])]
                    if len(active) != len(set(active)) or any(i < 0 or i >= len(raw) or raw[i] != 1 for i in active):
                        raise ValueError(f"{key}: malformed actual active flags")
                # Original arrays remain untouched. Native malloc-backed unwritten
                # cells are not portable integer expectations; actual active
                # indices still undergo their ordinary exact comparison.
                continue
            if aa != bb:
                i = next(i for i, (x, y) in enumerate(zip(aa, bb)) if x != y)
                raise ValueError(f"{key}[{i}]: first discrete mismatch {aa[i]} != {bb[i]}")
        elif not key.startswith("n:"):
            raise ValueError(f"unknown record type: {key}")
    if keys_a != keys_b:
        raise ValueError(f"record mismatch: {keys_a ^ keys_b}")
    if discrete_only:
        return
    for key in a:
        if key.startswith("n:"):
            abs_bound, rel_bound = atol, rtol
            if key == "n:parameters":
                abs_bound = atol if parameter_atol is None else parameter_atol
                rel_bound = rtol if parameter_rtol is None else parameter_rtol
            for i, (x, y) in enumerate(zip(a[key], b[key])):
                x, y = float(x), float(y)
                if not math.isfinite(x) or not math.isfinite(y):
                    raise ValueError(f"{key}[{i}]: nonfinite computation")
                if abs(x - y) > abs_bound + rel_bound * max(abs(x), abs(y)):
                    raise ValueError(f"{key}[{i}]: {x} != {y}; delta={abs(x-y)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rust", type=Path)
    parser.add_argument("julia", type=Path)
    parser.add_argument("--ranks", type=int, required=True, choices=(2, 4))
    parser.add_argument("--atol", type=float, required=True)
    parser.add_argument("--rtol", type=float, required=True)
    parser.add_argument("--justification", type=Path, required=True)
    parser.add_argument("--discrete-only", action="store_true")
    parser.add_argument("--parameter-atol", type=float)
    parser.add_argument("--parameter-rtol", type=float)
    parser.add_argument("--root-output", nargs=2, action="append", type=Path, metavar=("RUST", "REFERENCE"),
                        help="repeat for each independently captured root output")
    parser.add_argument("--c-flags", type=Path, help="independent C-written flag values/mask; raw arrays retained")
    args = parser.parse_args()
    if not args.justification.read_text().strip():
        raise ValueError("missing numerical justification")
    if not all(math.isfinite(x) and x >= 0 for x in (args.atol, args.rtol, args.parameter_atol, args.parameter_rtol) if x is not None):
        raise ValueError("invalid tolerance")
    for rank in range(args.ranks):
        a = read(args.rust / f"rank-{rank}.txt")
        b = read(args.julia / f"rank-{rank}.txt")
        required = {"d:group", "d:requested-width", "d:steps", "d:initial-draw-count", "d:total-draw-count", "d:sampling-draw-count", "d:configured-samples", "d:chain-samples", "d:local-counter", "d:reduced-counter", "d:seed", "d:initial-rng", "d:rng", "d:status", "d:ele_idx", "d:ele_cfg", "d:ele_num", "d:counter"}
        if not args.discrete_only:
            required |= {"n:initial-parameters", "n:parameters", "n:energy", "n:local-energy", "n:reduced-energy"}
        if not required <= a.keys() or not required <= b.keys() or any(len(rows.get(key, [])) != 624 for rows in (a,b) for key in ("d:rng", "d:initial-rng")):
            raise ValueError(f"rank {rank}: missing required evidence/624 RNG words")
        if a["d:status"] != ["0"] or b["d:status"] != ["0"]:
            raise ValueError(f"rank {rank}: unsuccessful worker status")
        validate_trace(a, rank)
        validate_trace(b, rank)
        for rows in (a, b):
            initial, total, sampled = (rows[key] for key in ("d:initial-draw-count", "d:total-draw-count", "d:sampling-draw-count"))
            if any(len(values) != 1 for values in (initial, total, sampled)) or int(initial[0]) < 0 or int(total[0]) != int(initial[0]) + int(sampled[0]):
                raise ValueError(f"rank {rank}: inconsistent actual draw counts")
            if not args.discrete_only:
                for step in range(int(rows["d:steps"][0])):
                    prefix = "" if step == 0 else f"step-{step}-"
                    for key in (f"n:sr-step-{step}", f"n:{prefix}local-energy", f"n:{prefix}reduced-energy"):
                        if key not in rows:
                            raise ValueError(f"rank {rank}: missing per-step evidence {key}")
        compare(a, b, args.atol, args.rtol, args.discrete_only, args.parameter_atol, args.parameter_rtol,
                read_c_flags(args.c_flags) if args.c_flags else None)
    if args.root_output:
        if args.discrete_only:
            raise ValueError("root computed output cannot be checked in discrete-only mode")
        for pair in args.root_output:
            compare_root(*pair, args.atol, args.rtol)
    print("Exact per-step states/proposals/decisions/actual draw counts and sampler-word streams agree; computed numbers NOT CHECKED." if args.discrete_only else "Captured per-step states/proposals/decisions/actual draw counts, sampler-word streams and bounded numerical records agree.")


if __name__ == "__main__":
    main()

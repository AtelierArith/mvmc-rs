"""Strict optional evidence comparison; execute with uv run --no-project python.

No default numerical tolerance: supply measured bounds and a justification file.
Discrete records are always exact, before any computed-number comparison.
"""
import argparse
import math
from pathlib import Path


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


def compare(a, b, atol, rtol):
    if a.keys() != b.keys():
        raise ValueError(f"record mismatch: {a.keys() ^ b.keys()}")
    # All integer/discrete contracts are checked before computed floats.
    for key in a:
        if len(a[key]) != len(b[key]):
            raise ValueError(f"{key}: shape mismatch")
        if key.startswith("d:"):
            aa, bb = [int(x) for x in a[key]], [int(x) for x in b[key]]
            if aa != bb:
                i = next(i for i, (x, y) in enumerate(zip(aa, bb)) if x != y)
                raise ValueError(f"{key}[{i}]: first discrete mismatch {aa[i]} != {bb[i]}")
        elif not key.startswith("n:"):
            raise ValueError(f"unknown record type: {key}")
    for key in a:
        if key.startswith("n:"):
            for i, (x, y) in enumerate(zip(a[key], b[key])):
                x, y = float(x), float(y)
                if not math.isfinite(x) or not math.isfinite(y):
                    raise ValueError(f"{key}[{i}]: nonfinite computation")
                if abs(x - y) > atol + rtol * max(abs(x), abs(y)):
                    raise ValueError(f"{key}[{i}]: {x} != {y}; delta={abs(x-y)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rust", type=Path)
    parser.add_argument("julia", type=Path)
    parser.add_argument("--ranks", type=int, required=True, choices=(2, 4))
    parser.add_argument("--atol", type=float, required=True)
    parser.add_argument("--rtol", type=float, required=True)
    parser.add_argument("--justification", type=Path, required=True)
    args = parser.parse_args()
    if not args.justification.read_text().strip():
        raise ValueError("missing numerical justification")
    if not all(math.isfinite(x) and x >= 0 for x in (args.atol, args.rtol)):
        raise ValueError("invalid tolerance")
    for rank in range(args.ranks):
        a = read(args.rust / f"rank-{rank}.txt")
        b = read(args.julia / f"rank-{rank}.txt")
        required = {"d:seed", "d:rng", "d:ele_idx", "d:ele_cfg", "d:ele_num", "d:counter", "n:parameters", "n:energy"}
        if not required <= a.keys() or len(a.get("d:rng", [])) != 624:
            raise ValueError(f"rank {rank}: missing required evidence/624 RNG words")
        compare(a, b, args.atol, args.rtol)
    print("Captured state records agree; per-move acceptance/proposal parity is a separate gate.")


if __name__ == "__main__":
    main()

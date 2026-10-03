"""Read actual paired CG observations; report differences, never select a tolerance."""
import argparse
import json
import math
from pathlib import Path
import re


def read(path):
    rows = {}
    for line in path.read_text().splitlines():
        key, *values = line.split()
        if key in rows or not key.startswith(("d:", "n:")) or not values:
            raise ValueError(f"{path}: duplicate/unknown/empty record {key}")
        rows[key] = [int(v) if key.startswith("d:") else float(v) for v in values]
        if any(not math.isfinite(v) for v in rows[key]):
            raise ValueError(f"{path}: nonfinite {key}")
    count = rows.get("d:events", [])
    if len(count) != 1 or count[0] <= 0:
        raise ValueError(f"{path}: missing events")
    for key in rows:
        if key == "d:events":
            continue
        match = re.fullmatch(r"[dn]:event-(\d{6})-[a-z-]+", key)
        if match is None or int(match[1]) >= count[0]:
            raise ValueError(f"{path}: malformed/out-of-range event key {key}")
    dimension = None
    open_solve = False
    for event in range(count[0]):
        stem = f"event-{event:06d}-"
        kind = rows.get("d:" + stem + "kind")
        required = {0: ("d:mapping", "n:mean", "n:diagonal", "n:gradient", "n:real-samples"),
                    1: ("d:phase", "n:search"),
                    2: ("d:iteration", "n:solution", "n:residual", "n:direction", "n:delta"),
                    3: ("d:iterations", "n:solution", "n:residual", "n:direction")}
        if kind not in ([0], [1], [2], [3]):
            raise ValueError(f"{path}: malformed event kind {event}")
        for field in required[kind[0]]:
            prefix, suffix = field.split(":")
            if f"{prefix}:{stem}{suffix}" not in rows:
                raise ValueError(f"{path}: missing {stem}{suffix}")
        allowed = set(required[kind[0]]) | {"d:kind"}
        allowed |= {0: {"n:imag-samples"}, 1: {"n:product"},
                    2: {"n:alpha"}, 3: set()}[kind[0]]
        actual = {key[:2] + key.split(stem, 1)[1] for key in rows if stem in key}
        if actual - allowed:
            raise ValueError(f"{path}: unexpected event fields {actual - allowed}")
        if kind == [0]:
            mapping = rows["d:" + stem + "mapping"]
            if open_solve or len(set(mapping)) != len(mapping) or any(i < 0 for i in mapping):
                raise ValueError(f"{path}: invalid mapping/solve boundary")
            dimension = len(mapping)
            open_solve = True
        elif not open_solve:
            raise ValueError(f"{path}: event outside prepared solve")
        if kind == [1]:
            phase = rows["d:" + stem + "phase"]
            if phase not in ([0], [1], [2], [3]) or (
                    phase != [0] and "n:" + stem + "product" not in rows):
                raise ValueError(f"{path}: malformed/missing product phase")
        for field in actual:
            prefix, suffix = field.split(":")
            values = rows[f"{prefix}:{stem}{suffix}"]
            if suffix in ("kind", "phase", "iteration", "iterations", "delta", "alpha"):
                if len(values) != 1 or (prefix == "d" and values[0] < 0):
                    raise ValueError(f"{path}: invalid scalar {suffix}")
            elif suffix.endswith("samples"):
                if len(values) % dimension:
                    raise ValueError(f"{path}: malformed sampled operator plane")
                if suffix == "imag-samples" and len(values) != len(rows["n:" + stem + "real-samples"]):
                    raise ValueError(f"{path}: mismatched real/imaginary planes")
            elif len(values) != dimension:
                raise ValueError(f"{path}: invalid vector dimension {suffix}")
        if kind == [3]:
            open_solve = False
    if open_solve:
        raise ValueError(f"{path}: missing actual finish event")
    return rows


def diagnose(a, b):
    if a.keys() != b.keys():
        raise ValueError(f"CG record keyset mismatch: {a.keys() ^ b.keys()}")
    first = None
    metrics = {}
    for key, left in a.items():
        right = b[key]
        if len(left) != len(right):
            raise ValueError(f"CG shape mismatch {key}")
        if key.startswith("d:"):
            if left != right:
                raise ValueError(f"CG discrete mismatch {key}: {left} != {right}")
            continue
        delta = max((abs(x-y) for x, y in zip(left, right)), default=0.0)
        scale = max((abs(x) for x in left + right), default=0.0)
        if delta and first is None:
            index = next(i for i, (x, y) in enumerate(zip(left, right)) if x != y)
            first = dict(key=key, index=index, rust=left[index], julia=right[index],
                         absolute_difference=abs(left[index]-right[index]), scale=scale)
        field = key.rsplit("-", 1)[-1]
        if field not in metrics or delta > metrics[field]["max_absolute_difference"]:
            metrics[field] = dict(key=key, max_absolute_difference=delta, scale=scale)
    return dict(first_numeric_difference=first, metrics=metrics,
                scope="DIAGNOSTIC_ONLY_NO_NUMERICAL_ACCEPTANCE")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rust", type=Path)
    parser.add_argument("julia", type=Path)
    args = parser.parse_args()
    print(json.dumps(diagnose(read(args.rust), read(args.julia)), sort_keys=True))


if __name__ == "__main__":
    main()

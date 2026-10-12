"""Verify archived C-order CPU / CUDA interval receipts and all 20 Opt outputs."""
import gzip
import json
import math
from pathlib import Path

root = Path(__file__).resolve().parent
maximum = {}
def compare(a, b, field):
    if isinstance(a, list):
        assert len(a) == len(b), field
        for x, y in zip(a, b):
            compare(x, y, field)
    else:
        assert math.isfinite(a) and math.isfinite(b), (field, a, b)
        maximum[field] = max(maximum.get(field, 0), abs(a-b))
        assert abs(a-b) <= 1e-12 + 1e-12*max(abs(a), abs(b)), (field, a, b)

def read(path):
    with gzip.open(path, "rt") as source:
        return source.read()

rows = [[json.loads(line) for line in read(root/f"{label}-receipts.jsonl.gz").splitlines()]
        for label in ["host", "device"]]
assert len(rows[0]) == len(rows[1]) == 64
for a, b in zip(*rows):
    assert a.keys() == b.keys()
    for key in a:
        if key in ("energy", "parameters"):
            compare(a[key], b[key], key)
        else:
            assert a[key] == b[key], (a["walker"], key)
for name in ("zvo_out.dat", "zvo_var.dat", "zqp_opt.dat"):
    baseline = read(root/"host"/(name+".gz"))
    if name == "zvo_out.dat":
        assert len(baseline.splitlines()) == 20
    a = [float(x) for x in baseline.split()]
    for label in ["device"] + [f"release-{i}" for i in range(4)]:
        b = [float(x) for x in read(root/label/(name+".gz")).split()]
        compare(a, b, f"{label}/{name}")
print(json.dumps({"walkers": 64, "steps": 20,
    "exact_fields": ["walker", "rng", "rng_index", "rng_words", "ele", "occ", "proj", "counter"],
    "tolerance": {"atol": 1e-12, "rtol": 1e-12}, "max_absolute_differences": maximum}, indent=2))

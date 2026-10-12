import argparse
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser(description="Compare L128 interval baseline/candidate receipts and outputs")
parser.add_argument("baseline", type=Path)
parser.add_argument("candidate", type=Path)
parser.add_argument("baseline_output", type=Path)
parser.add_argument("candidate_output", type=Path)
args = parser.parse_args()
baseline, candidate = args.baseline, args.candidate
rows = [[json.loads(line) for line in path.read_text().splitlines()] for path in (baseline, candidate)]
assert len(rows[0]) == len(rows[1]) == 64
maximum = {}
def compare(a, b, field):
    if isinstance(a, list):
        assert len(a) == len(b), field
        for x, y in zip(a, b):
            compare(x, y, field)
    else:
        assert math.isfinite(a) and math.isfinite(b), (field, a, b)
        maximum[field] = max(maximum.get(field, 0), abs(a - b))
        assert math.isclose(a, b, abs_tol=1e-12, rel_tol=1e-12), (field, a, b)
for a, b in zip(*rows):
    assert a.keys() == b.keys()
    for key in a:
        if key in ('energy', 'parameters'):
            compare(a[key], b[key], key)
        else:
            assert a[key] == b[key], (a['walker'], key)
for name in ('zvo_out.dat', 'zvo_var.dat', 'zqp_opt.dat'):
    a = (args.baseline_output / name).read_text().splitlines()
    b = (args.candidate_output / name).read_text().splitlines()
    assert len(a) == len(b), (name, len(a), len(b))
    for x, y in zip(a, b):
        compare([float(v) for v in x.split()], [float(v) for v in y.split()], name)
print(json.dumps({'walkers': 64, 'exact_fields': ['walker', 'rng', 'rng_index', 'rng_words', 'ele', 'occ', 'proj', 'counter'], 'tolerance': {'atol': 1e-12, 'rtol': 1e-12}, 'max_absolute_differences': maximum}, indent=2))

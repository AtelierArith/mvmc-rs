"""Compare placement unset/1/2 CLI outputs, excluding timing and header records."""
import json
import math
from pathlib import Path
import sys

root = Path(sys.argv[1])
summary = []


def numbers(path):
    values = [float(token) for line in path.read_text().splitlines()
              if not line.lstrip().startswith('#') for token in line.split()]
    assert all(math.isfinite(v) for v in values), path
    return values


for ranks in (2, 4):
    for mode in ('opt', 'phys'):
        reference = root / f'{mode}-r{ranks}-cunset'
        files = {p.name for p in reference.iterdir()
                 if p.is_file() and '_time_' not in p.name}
        assert files
        maximum = 0.0
        for cpus in (1, 2):
            candidate = root / f'{mode}-r{ranks}-c{cpus}'
            assert files == {p.name for p in candidate.iterdir()
                             if p.is_file() and '_time_' not in p.name}
            for name in files:
                a, b = numbers(reference / name), numbers(candidate / name)
                assert len(a) == len(b), name
                maximum = max(maximum, max((abs(x-y) for x, y in zip(a, b)), default=0.0))
        summary.append(dict(ranks=ranks, mode=mode, files=len(files),
                            max_absolute_difference=maximum))
print(json.dumps(summary, indent=2))

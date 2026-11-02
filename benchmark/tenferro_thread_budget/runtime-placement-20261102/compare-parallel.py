"""Compare finite formatted output tokens, excluding timing/header records."""
import json
import math
from pathlib import Path
import sys

root = Path(sys.argv[1])
summary = []


def values(path):
    result = []
    for line in path.read_text().splitlines():
        if line.lstrip().startswith("#"):
            continue
        for token in line.split():
            value = float(token)
            assert math.isfinite(value), (path, token)
            result.append(value)
    return result


for ranks in (2, 4):
    for mode in ("opt", "phys"):
        for pool in ("engine", "rayon"):
            base = root / f"{mode}-r{ranks}-{pool}-cunset"
            placed = root / f"{mode}-r{ranks}-{pool}-c2"
            names = lambda directory: {
                p.name for p in directory.iterdir()
                if p.is_file() and "_time_" not in p.name
            }
            files = names(base)
            assert files and files == names(placed), (base, placed)
            differences = []
            for name in sorted(files):
                a, b = values(base / name), values(placed / name)
                assert len(a) == len(b), name
                differences.extend(abs(x - y) for x, y in zip(a, b))
            summary.append(dict(ranks=ranks, mode=mode, pool=pool,
                                files=len(files),
                                max_absolute_difference=max(differences)))
print(json.dumps(summary, indent=2))

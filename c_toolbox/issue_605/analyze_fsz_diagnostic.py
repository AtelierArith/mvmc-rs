#!/usr/bin/env python3
"""Compare 200 observational C/Linux and Rust/Linux/macOS FSZ sample records."""
import gzip
import re
from pathlib import Path

root = Path(__file__).resolve().parent
pattern = re.compile(r"(?:C)?DIAG605 (\d+) ((?:[-+0-9.eE]+\s*){6})")

def records(name):
    with gzip.open(root / name, "rt") as file:
        rows = []
        for line in file:
            match = pattern.search(line)
            if match:
                rows.append((int(match[1]), *map(float, match[2].split())))
    assert len(rows) == 200 and [x[0] for x in rows] == list(range(200))
    return rows

c = records("fsz-c-linux.log.gz")
linux = records("fsz-rust-linux.log.gz")
mac = records("fsz-rust-macos.log.gz")
for label, other in [("C/Linux", c), ("macOS/Linux", mac)]:
    print(label)
    for index, name in enumerate(("overlap.re", "overlap.im", "energy.re", "energy.im", "derivative.re", "derivative.im"), 1):
        differences = [abs(a[index] - b[index]) for a, b in zip(linux, other)]
        largest = max(range(len(differences)), key=differences.__getitem__)
        first = next((i for i, value in enumerate(differences) if value != 0.0), None)
        print(f"  {name}: max={differences[largest]:.17g}, sample={largest}, first_nonzero={first}")
contributions = []
for a, b in zip(linux, mac):
    ea, eb = complex(a[3], a[4]), complex(b[3], b[4])
    da, db = complex(a[5], a[6]), complex(b[5], b[6])
    contributions.append((eb * db - ea * da) / 200.0)
print(f"macOS/Linux HO[8] reconstructed difference: {sum(contributions)}")
print(f"sample 59 contribution: {contributions[59]}")

full_path = root / "fsz-rust-macos-full.log.gz"
for tag in ("ho", "parameters.txt", "energy.txt"):
    row_pattern = re.compile(r"DIAG605 " + re.escape(tag) + r" (\d+) ([-+0-9.eE]+) ([-+0-9.eE]+)")
    rows = []
    with gzip.open(full_path, "rt") as file:
        for line in file:
            match = row_pattern.search(line)
            if match:
                index = int(match[1])
                actual, expected = float(match[2]), float(match[3])
                rows.append((abs(actual - expected), index))
    assert rows
    maximum, index = max(rows)
    print(f"macOS historical {tag}: {len(rows)} components, max_abs={maximum:.17g}, index={index}")

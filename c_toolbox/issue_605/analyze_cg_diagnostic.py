#!/usr/bin/env python3
"""Compare recorded C/Linux and Rust/Linux/macOS RBM CG diagnostics."""
import gzip
import math
from pathlib import Path
import re

root = Path(__file__).resolve().parent
with gzip.open(root / "cg-c-linux.log.gz", "rt") as file:
    c = {line.split()[0]: line.split()[1:] for line in file}
n, samples = int(c["n"][0]), int(c["samples"][0])
assert (n, samples) == (194, 100)
c = {key: [float(x) for x in values] for key, values in c.items()}
pattern = re.compile(r"DIAGCG (\d+) ((?:[-+0-9.eE]+\s*){4})")

def rust_rows(name):
    rows = []
    with gzip.open(root / name, "rt") as file:
        for line in file:
            match = pattern.search(line)
            if match:
                rows.append([int(match[1]), *map(float, match[2].split())])
    rows = rows[:n]
    assert len(rows) == n and [x[0] for x in rows] == list(range(n))
    return rows

linux = rust_rows("cg-rust-linux.log.gz")
mac = rust_rows("cg-rust-macos.log.gz")
for column, name in enumerate(("gradient", "mean", "variance", "solution"), 1):
    differences = [abs(a[column] - b[column]) for a, b in zip(linux, mac)]
    index = max(range(n), key=differences.__getitem__)
    print(f"Linux/macOS {name}: max_abs={differences[index]:.17g}, component={index}")

# Recompute the operator with the independently captured C sample matrices.
# Substituting these for Rust's very-close samples is a cross-reference probe,
# not the exact macOS residual printed by the test's own sampled operator.
def residual(rows):
    g = [row[1] for row in rows]
    mean = [row[2] for row in rows]
    diag = [row[3] for row in rows]
    x = [row[4] for row in rows]
    mean_dot = sum(a*b for a, b in zip(mean, x))
    product = [0.0] * n
    for sample in range(samples):
        r = c["Oreal"][sample*n:(sample+1)*n]
        im = c["Oimag"][sample*n:(sample+1)*n]
        yr = sum(a*b for a, b in zip(r, x))
        yi = sum(a*b for a, b in zip(im, x))
        for i in range(n):
            product[i] += r[i]*yr + im[i]*yi
    vector = [product[i]/100.0 - mean[i]*mean_dot + .001*diag[i]*x[i] - g[i] for i in range(n)]
    return math.sqrt(sum(v*v for v in vector)/sum(v*v for v in g))

print(f"Linux solution, C sample matrices: relative residual {residual(linux):.17g}")
print(f"macOS solution, C sample matrices: relative residual {residual(mac):.17g}")

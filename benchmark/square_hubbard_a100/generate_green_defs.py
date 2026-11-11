#!/usr/bin/env python3
"""Generate the measurement definitions greenone.def and greentwo.def.

The files reproduce, byte for byte, the measurement requests used by the
Fugaku C-mVMC reference runs of the square-lattice Hubbard benchmark
(``cases.json`` records their SHA-256).  Sites are indexed x-fastest,
``site = x + L*y``.

Requested operators, for ordered site pairs (i, j) with i an origin:

- one-body ``c†_{i s} c_{j s}`` for both spins (and all on-site i == j);
- density ``n_{i s} n_{j t}`` for all spin pairs (and all on-site i == j);
- transverse spin ``c†_{i↑} c_{j↑} c†_{j↓} c_{i↓}`` and its spin flip, i != j.

``--origins all`` requests every site as origin (6N^2 - 2N two-body rows);
``--origins sublattice`` uses only the 2x2 unit-cell sites {0, 1, L, L+1}.

Usage:
    uv run --no-project python generate_green_defs.py --L 16 --origins sublattice OUTDIR
"""

from __future__ import annotations

import argparse
from pathlib import Path

BAR = "=" * 32


def origins_for(L: int, policy: str) -> list[int]:
    if policy == "all":
        return list(range(L * L))
    if policy == "sublattice":
        return sorted({x + L * y for y in range(2) for x in range(2)})
    raise ValueError(f"unknown origin policy: {policy}")


def header(label: str, count: int, columns: str) -> list[str]:
    return [BAR, f"{label}  {count:10d}", BAR, columns, BAR]


def build(L: int, policy: str) -> tuple[str, str]:
    n = L * L
    origins = origins_for(L, policy)
    origin_set = set(origins)
    one = [
        (i, s, j, s)
        for i in range(n)
        for j in range(n)
        if i in origin_set or i == j
        for s in (0, 1)
    ]
    density = [
        (i, si, i, si, j, sj, j, sj)
        for i in range(n)
        for j in range(n)
        if i in origin_set or i == j
        for si in (0, 1)
        for sj in (0, 1)
    ]
    transverse = [
        row
        for i in origins
        for j in range(n)
        if i != j
        for row in ((i, 0, j, 0, j, 1, i, 1), (i, 1, j, 1, j, 0, i, 0))
    ]
    two = density + transverse
    one_lines = header("NCisAjs", len(one), "======== i s j t ========")
    one_lines += [" ".join(f"{v:5d}" for v in row) for row in one]
    two_lines = header("NCisAjsCktAltDC", len(two), "======== i s j t k u l v ========")
    two_lines += [" ".join(f"{v:5d}" for v in row) for row in two]
    return "\n".join(one_lines) + "\n", "\n".join(two_lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--L", type=int, required=True, help="linear size of the L x L lattice")
    parser.add_argument("--origins", choices=("all", "sublattice"), required=True)
    parser.add_argument("outdir", type=Path)
    args = parser.parse_args()
    if args.L < 2 or args.L % 2:
        parser.error("--L must be an even integer >= 2 (2x2 sublattice)")
    one, two = build(args.L, args.origins)
    args.outdir.mkdir(parents=True, exist_ok=True)
    for name, text in (("greenone.def", one), ("greentwo.def", two)):
        (args.outdir / name).write_text(text, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

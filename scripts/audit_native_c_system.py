#!/usr/bin/env python3
# /// script
# dependencies = ["numpy==2.3.4"]
# ///
"""Assess saved native SR operands, without changing comparison tolerances.

uv run scripts/audit_native_c_system.py C-system.json Rust-diagnostics.json
Matrices are column-major unfactored systems captured before LAPACK.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("c", type=Path)
    parser.add_argument("rust", type=Path)
    args = parser.parse_args()
    c = json.loads(args.c.read_text())
    r = json.loads(args.rust.read_text())["systems"][0]
    if c["direct_map"] != r["active_indices"]:
        raise SystemExit(
            "Active component maps differ; conditioning cannot excuse this defect"
        )
    n = len(c["direct_map"])
    matrices = [
        np.array(values).reshape((n, n), order="F")
        for values in (c["direct_S"], r["matrix"])
    ]
    rhs = [np.array(values) for values in (c["direct_g"], r["rhs"])]
    solutions = [np.linalg.solve(a, b) for a, b in zip(matrices, rhs)]

    def residual(a, b, x):
        return float(
            np.linalg.norm(a @ x - b, np.inf)
            / (
                np.linalg.norm(a, np.inf) * np.linalg.norm(x, np.inf)
                + np.linalg.norm(b, np.inf)
            )
        )

    report = dict(
        numpy=np.__version__,
        numpy_config=np.__config__.show(mode="dicts"),
        dimension=n,
        matrix_max_absolute=float(np.max(np.abs(matrices[0] - matrices[1]))),
        rhs_max_absolute=float(np.max(np.abs(rhs[0] - rhs[1]))),
        condition_inf=[float(np.linalg.cond(a, np.inf)) for a in matrices],
        solution_max_absolute=float(np.max(np.abs(solutions[0] - solutions[1]))),
        backward_error=[residual(a, b, x) for a, b, x in zip(matrices, rhs, solutions)],
        rust_actual_backward_error=residual(
            matrices[1], rhs[1], np.array(r["increment"])
        ),
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

# Reproduce the 2026-10-08 Rust vs Julia comparison

From the repository root:

```sh
scripts/run_julia_comparison.sh
# Optional new output directory; existing directories are never overwritten:
scripts/run_julia_comparison.sh --output target/bench/my-comparison
# Inspect the four benchmark commands without running them:
scripts/run_julia_comparison.sh --dry-run
```

The script runs all cases sequentially with three measured repetitions, one
warmup, one BLAS/OpenMP/Julia thread and one Rust inner worker. Rust uses the
release profile, OpenBLAS and the C-order CPU backend. No MPI or GPU is used.

- Small optimization: all four models, 50 SR steps.
- Hubbard optimization: L16/L24/L32, 300 SR steps (L64 is excluded).
- Small PhysCal: all four reference fixtures, including Lanczos.
- Hubbard PhysCal: L16/L24/L32, 100 samples, with per-run outputs retained.

CSV files, Markdown reports, logs, environment metadata and the Julia lock are
saved in a new directory under `target/bench/` by default. Retained PhysCal
outputs are under `target/bench-output/`; their path is printed in the log.
PhysCal also checks the measured observables using the existing tolerances.
Command failures stop the script and propagate a nonzero exit code.

## Prerequisites and reference lock

Use Linux x86_64 with the setup described in
[DEVELOPMENT.md](../../docs/DEVELOPMENT.md#reference-runner-setup-julia-and-c).
Initialize the nested submodules with `git submodule update --init --recursive`
and install Julia 1.13.1. Missing Julia native libraries are built during setup;
this requires `make`, `g++`, `gfortran` and BLAS/LAPACK development packages.
Already-built libraries are reused, as in the original measurement.

The original measurement used Rust commit
`5a01cc8ab24abadd47bf2733e26f79eb2d93b081` and Julia upstream commit
`8d815db0eec0ba12dea88aa9edb3d5f74d7b5ccd`. The script records the current
commits and dirty state without changing either checkout. To reproduce the
original source versions, run this script against those checkouts; later code
is a new comparison under the same workload and settings.

The upstream Julia commit does not contain the previously prescribed
`Manifest-v1.13.toml`. The lock alongside this README is copied verbatim from
the earlier reference commit `c0788c34a6a5753c611633a97cd1ea233203320c`, exactly
as used in the original measurement. Its local package paths resolve against
the currently pinned Julia checkout. SHA-256:
`09ebd06dab244510094b99fe7c6efa2fe7a3d22221d1336b951123a5a6e8befc`.

The script temporarily copies this lock into the Julia project and removes it
on exit. An existing identical lock is preserved; a differing lock causes an
error before measurement. The script does not replace local source files.
Package downloads, precompilation and native builds happen before measurement.

## Reading results

Hubbard reports compare medians of internal timings, excluding process startup
and initial Julia JIT compilation. A `julia/rust` ratio above one means Rust
is faster. The small-model tasks include Rust process startup but measure the
Julia function in an already-running process; millisecond PhysCal results
therefore do not isolate kernel performance. Rust's Hubbard internal timer
prints seconds to two decimal places.

The original machine was a Ryzen 9 PRO 8945HS with rustc 1.99.0, Julia 1.13.1,
system OpenBLAS 0.3.26 pthread for Rust and Julia's native PfaPack helper, and
Julia OpenBLAS 0.3.30 ILP64 for `LinearAlgebra`. CPU, library versions, compiler
flags and host load can change timings; recorded metadata helps explain those
differences. The script preserves Cargo configuration and records release
profile/flag overrides rather than changing the user's compiler setup.

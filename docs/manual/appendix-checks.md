# Appendix: how the citations were checked

[Contents](README.md) · Previous: [12. Accelerated and GPU backends](12-accelerated-backends.md)

## A.1 Citation check

Every "Implementation" box lists a C function and a Rust function with `file:line`. To keep them honest,

```bash
uv run --no-project scripts/check_manual_citations.py --c-root extern/mVMC-1.3.0
```

parses `docs/manual/*.md` and `docs/manual/ja/*.md` and verifies for every backticked `path:line` or `path:line-line` citation that the file exists and the line is inside the file; for the bullets
``- C: `symbol` — `path:line` `` and ``- Rust: `symbol` — `path:line` `` it also verifies that the first identifier of `symbol` occurs on the cited line. The script is an optional developer
tool (standard library only); it is not run by the Rust build or tests. `extern/mVMC-1.3.0` is a git submodule and must be checked out (or passed with `--c-root`).

At the time of writing (repository commit `3e9024ee`, C reference commit `d73d06bd`) it reported **430 citations checked, 0 problems**. The checker now also covers the Japanese translation in `docs/manual/ja/` and verifies relative links and heading anchors; the combined run reports 860 citations (430 per language), 0 problems.

What the check does **not** cover: whether the cited function really implements the equation next to it, ranges such as `driver.rs:271-278, 532-552` listed inside one backtick span, test names, and prose
claims about operation order. Those were checked by reading the C and Rust sources while writing; the places where a claim rests only on reading are marked **(code reading)** or **(unverified)**.
After a refactoring, rerun the script and update drifting line numbers (the symbol names are the stable key).

## A.2 Provenance of the observed output

All **(observed)** statements and the [tutorial](10-tutorial.md) output come from the same build:

| Item | Value |
|------|-------|
| Repository | `AtelierArith/mvmc-rs`, `main` at commit `3e9024ee` |
| Build | `cargo build --release -p mvmc-cli` (default features; no `mpi`) |
| Platform | Linux x86_64 (`uname -sm`), 36 logical CPUs |
| Compiler | `rustc 1.99.0 (b940084d7 2026-09-28)` |
| BLAS/LAPACK | system OpenBLAS 0.3.26 (`libopenblas.so.0`), default threading |
| Inputs | `benchmark/hubbard_chain/inputs/hubbard_chain_L16` with the `modpara.def` changes shown in the tutorial |

The numbers in the tutorial are results of one build and platform; other BLAS providers, thread counts or platforms may change low-order digits and, through Metropolis decisions, later steps
([11.4](11-compatibility.md#114-numerical-comparison-policy)). The commands were run from a scratch directory (not committed); nothing under `output/` is part of the repository.

## A.3 Not verified

- Anything that needs MPI: the `mpi` feature build, `mpirun` launches, grouped execution (`NSplitSize > 1`), collective error agreement. The description in [8.4](08-running.md#84-mpi-and-grouped-execution) comes from the code and from the tests in `crates/mvmc-core/src/run_mpi_tests.rs`.
- macOS behaviour (Homebrew OpenBLAS path, Accelerate in the FSZ reference path).
- `simd-backend` for `mvmc-cli` (it is a `pfapack` feature that the CLI does not forward).
- The example programs in `crates/mvmc-cli/examples` (they need the `extern/Julia-mVMC` submodule inputs).
- Runs with FSZ/general orbitals, complex wave functions, RBM, OptTrans (`-o`), `InterAll`, local spins, `NExUpdatePath` other than 0, and `NSRCG = 1` beyond the 5-step demonstration.
- Numerical parity of any run with C or Julia; the tutorial compares nothing against a reference.
- The Japanese manual `extern/mVMC-1.3.0/doc/ja` was not read; the English manual (`doc/en`) is the source for all paraphrases.
- [Chapter 12](12-accelerated-backends.md): the CUDA gates, the CUDA kernels and every GPU or device-resident number were **not run** for this manual; they are copied from `docs/design/gpu-readiness.md` and `docs/NUMERICAL_COMPARISONS.md`. Only the CPU-side statements marked **(observed)** (the `tenferro` and `cuda` selections of the stock `mvmc` binary on `hubbard_chain_L16`) were reproduced here. The benchmark suite of issue #450 did not exist yet.

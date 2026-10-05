# Issue #176: native real-FSZ setup and initializer probe

Optional developer tool; Cargo builds and tests never compile, run or read this
directory. They read the checked-in fixture `tests/fixtures/real_fsz/c_setup.txt`
(provenance and source hashes in its header).

Scope: standalone native kernel check of the real-FSZ setup path, NOT a full `vmc.out`
or MPI run. `probe.c` links, without modification:

- `matrix.c` `CalculateMAll_fsz_real` / `calculateMAll_child_fsz_real` (spin-indexed assembly
  `invM[msj][msi] = -sltE[rsi][rsj]`, `M_DSKTRF("U","N")` with `INFO` return, `utu2pfa_d`,
  finite-Pfaffian guard, `utu2inv_d`, sign flip),
- `projection.c` `MakeProjCnt`, `sfmt/SFMT.c`,
- `makeInitialSample_fsz_real` extracted unmodified from `vmcmake_fsz_real.c`
  (`extract_initializer.pl`, source SHA-256 pinned in the fixture header),
- the real pfapack Fortran (`dsktrf`, `dsktf2`, ...) and `ltl2inv` `utu2*` C++ (CMake source list,
  no BLIS) with system OpenBLAS.

Only passive observation is added: `CalculateMAll_fsz_real` is wrapped to log its `INFO`, and the
101-attempt `MPI_Abort` becomes a recorded `longjmp`. MPI is replaced by a single-rank shim
(`mpi_single_rank_stub.c`) when no MPI runtime is installed; set `MPICC` to use a real wrapper.
Matrix cases use the literal operands of `tests/fixtures/real_fsz/setup.txt` (only `creal` of each
Slater element enters the native kernel, as in `vmcmain.c:372`); the five initializer cases
(retry, failure, zero, localspin, magnetized) use the Slater tables built in
`crates/mvmc-core/tests/real_fsz_setup.rs`, seed 1, and print the complete next 624 SFMT words.

```bash
MPI_PREFIX=<dir with include/mpi.h> c_toolbox/issue176_real_fsz/run_probe.sh <repo> <empty-out-dir>
```

Fixture regeneration is explicit and optional; do not fill it from Rust output.

## Findings (Linux x86_64, GCC 13.3.0, OpenBLAS 0.3.26)

- All 16 matrix cases return `INFO=0`; the Rust real kernels reproduce the native pivoted
  factorization, Pfaffian and inverse (difference 0 on this platform).
- retry/failure/localspin/magnetized: native configurations, spins and the next 624 RNG words
  equal the Julia fixture and Rust exactly.
- zero matrix: native `DSKTRF` reports `INFO=1`, so C retries 101 times and aborts. Julia ignores the
  zero-pivot status and accepts the zero matrix on the first attempt (a Julia-only behavior,
  not copied); Rust now follows C (`CalcMAllError::ZeroPivot`, 101 attempts).
- C stores per-QP results at `qp - qpStart`; Rust keeps absolute QP indices (layout, not numerics).

# Issue #176: retained 2×2 inverse replay

Optional developer-only probe; no Cargo build, test, FFI, or oracle dependency.
Run from any directory with `bash c_toolbox/issue176_inverse/replay.sh`.
It builds unchanged vendored routines in a fresh `/tmp` directory and retains
compiler versions, source hashes, linked libraries, and numerical output there.
The shell trace records actual compilation commands. Requirements: C++11,
gfortran, and LP64 OpenBLAS/LAPACK. No Julia or MPI is involved.

## Operand and kernel boundary

The existing `tests/fixtures/real_fsz/setup.txt` has SHA-256
`92f52b3f5f70f1bd00d5500bf7db736a4b463df5b5f5cc33615f02c87d6cfac9`.
Its case header `2 1 1 0` supplies electron indices `[0,1]`, spins `[1,1]`,
and QP1 Slater entry `S[2,3]=0.5535714285714286` (binary64 input
`3fe1b6db6db6db6e`). The driver preserves this literal operand; it does not
derive an expectation from Rust. Assembly gives column-major `[0,-a,a,0]`.
The supplied **already-factorized upper** 2×2 matrix has pivots `[1,2]`.

The probe invokes native `utu2inv_d` and `utu2inv_z`, then reports the final
minus-sign convention used by the mVMC matrix child. It does **not** execute
DSKTRF/ZSKTRF, matrix assembly, sampling, retry, RNG, QP publication, or the
full public FSZ runner. For n=2, the retained factor is the trivial upper
skew factor; this does not validate general factorization or pivoting.
It also cannot exercise the n>2 poisoned-workspace issue investigated under
#184. No claim of full FSZ correctness follows from this check.

## Primary-source boundaries and hashes

Files are compiled in place, not extracted or repaired. Upstream files retain
their MPL-2.0 notices. In `extern/mVMC-1.3.0/src/ltl2inv/`:

- `ltl2inv.cc`: lines 18 and 20 export the real and complex wrappers;
  SHA `d2b954765728a864a3729ac62016eec20c03f75b706b2c19eb8497774a3ea5dd`.
- `invert.tcc`: lines 12–28 `sktdsmx` contain scalar real/complex divisions;
  lines 76–120 implement `utu2inv`, including M initialization, vT extraction,
  triangular products and permutations;
  SHA `d477c7d2bb6d2b0c29872d06ec75fd7e804e1e30ace00888b77d31d2c74b821c`.
- `ilaenv_lauum.cc`: unchanged block-size dispatch;
  SHA `2e2f764cb9315088e397a649bd4487f1ff342e8c54549c0bc9695d0b63be3a5e`.
- `ilaenv_wrap.f90`: unchanged Fortran ILAENV bridge;
  SHA `13292b4a29358488f5a4bbac300bc09bd4b500c9af6e5579fc7a0b95d5e7b2ca`.

Each replay additionally hashes all listed directly included common headers,
`pfaffian.tcc`, `trmmt.tcc`, and ILAENV headers. Compiler flags are
`-O0 -ffp-contract=off`; C++ uses `-std=c++11 -DBLAS_EXTERNAL` for upstream
objects. Linking uses `-lopenblas -lgfortran`. Fortran module output is isolated
with `-J`, so the replay does not write `wrapper.mod` into the checkout.

## Observed Linux result, 2026-10-03

Fresh rebuild: `/tmp/mvmc-issue176-inverse.DFzBU8`, exit **0**.
Linux x86_64, GCC/G++ and gfortran 13.3.0, LP64 system OpenBLAS,
`OPENBLAS_NUM_THREADS=1`. See retained `libraries.txt` for actual library paths.
The unchanged upstream header emits a `diag2char` return-type warning; the
probe does not suppress or repair it.

Both native routines returned post-sign `(row1,col0)`
`-1.8064516129032258`, with measured `a*inverse[1]-1 = 0`.
Both extracted vT values were `-0.5535714285714286`.
Diagnostic output bits coincide (`bffce739ce739ce7`); these are observations,
**not** a cross-platform computed-bit acceptance requirement. The standalone
driver checks finite values and an analytic 2×2 residual bounded by `8e-16`
(a few binary64 rounding units at unit scale); no Rust test tolerance changes.

Executable SHA: `c2b0bb73496c94b350ade5a8e019ec11bc24e4e87e4a0494c2e418cf881178b9`.
Output SHA: `3e2cc583aa776d4ca65aed4ccba30b634e68e31a619893311e43fab8da136e05`.
The earlier borrowed-library `/tmp` replay agreed, but this fresh build is the
reusable provenance baseline. Native macOS has not been replayed here; that
limits platform conclusions, not the validity of this bounded Linux check.

No new checked fixture is necessary for this diagnostic: the independent
retained operand already exists, and normal Rust tests must not read toolbox
artifacts. #176 remains open pending the remaining algorithm/input-contract
and public-lifecycle evidence.

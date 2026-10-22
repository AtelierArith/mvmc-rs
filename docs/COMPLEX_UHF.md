# ComplexUHF initial-orbital tool (`mvmc uhf`)

Issue #350, related to #185. Pure-Rust port of the C tool in
`extern/mVMC-1.3.0/src/ComplexUHF/` (documented in
`extern/mVMC-1.3.0/doc/en/source/appendix.rst`). The C source is the
authority for input contract, operation order, SFMT draw order and output
format. The implementation is the crate `crates/mvmc-uhf`; the command line is

```text
mvmc uhf namelist.def [OptParaFile]
```

Like the C `UHF namelist.def` executable, definition files and outputs are
resolved against the current directory, and the optional second argument is
accepted but unused (C `UHFmain.c` never reads `argv[2]`). Exit status is 0
after convergence, 255 when `IterationMax` is exhausted (the files are still
written, C `return -1`) and 1 for input errors.

## Algorithm (C operation order preserved)

1. `initial.c`: with `NInitial == 0`, `G[a][b] = 0.01*(genrand_real2()-0.5)`
   for all `(2*Nsite)^2` entries in row-major order from
   `init_gen_rand(RndSeed)`; otherwise `G` is zero except the `Initial`
   records (no draws).
2. Loop `step = 0 .. IterationMax-1`: `makeham` (Transfer, CoulombIntra,
   CoulombInter, Hund, Exchange, PairHop, InterAll Hartree-Fock decoupling),
   `diag` (LAPACK `zheev('V','U')`, `lwork = 4*N`), `green`
   (`zgemm('N','N')` of the conjugated occupied eigenvectors), `cal_energy`
   (energy from the un-mixed new `G`, residual
   `sqrt(sum|G_old-G|^2)/(2 Nsite^2)`, particle number, then linear mixing with
   `Mix`). Stop when `rest < 0.1^EPS` (`EPS` successive multiplications by
   `0.1`).
3. `output.c`: `_result`, `_eigen`, `_gap`, `_UHF_cisajs` and, when an orbital
   definition is present, `{CParaFileHead}_AP_Fij.dat`/`_P_Fij.dat`/
   `_General_Fij.dat` and `*Orbital_opt.dat` (AP-only mode re-diagonalises the
   up and down blocks of the last `Ham`). Every optimizer parameter gets
   `+ genrand_real2() * 10^-EpsSlater` on its real part, drawn in index order
   after the iteration (AP first, then P in mode `Orbital`+`OrbitalParallel`).

The `*_opt.dat` files are read by the Rust optimizer through the
`InOrbital` namelist key, as in `samples/tutorial_1.3/run_uhf.sh`.

## Large normal Slater matrices and initialization retries (#526)

Normal sampling validates each initial electron placement using the complex
Slater factorization, including for a real-valued run. `Nsize = 2 * Ne` is
the matrix dimension; it need not equal `Nsite`. Positive factorization INFO
causes another placement. After **101 completed calls**, initialization fails,
even if call 101 returns a nonpositive INFO. This matches
`src/mVMC/vmcmake.c::makeInitialSample` in the C reference: increment `loop`,
abort if `loop > 100`, then test whether the reduced INFO is positive.
Rust returns a typed error with `Nsize` and the local/comm1 INFO instead of
calling `MPI_Abort`. Increasing the retry limit changes the C contract.

The investigation recorded in [issue #526](https://github.com/AtelierArith/mvmc-rs/issues/526)
found that the default random Slater initialization failed at half-filled
Hubbard-chain sizes 464, 500 and 512, while sizes 448 through 460 and
`Nsite=512, Nsize=256` started. These are observations for that input and
reference environment, **not a universal dimension limit**. The issue reports
the same failure in the C executable. A positive INFO alone does not establish
the cause of singularity or justify changing the factorization.

An external starting orbital can be supplied without changing the default
initialization algorithm:

1. Run `mvmc uhf namelist.def` with matching sites, particle number and orbital
   index definitions, using the UHF input contract below.
2. Check convergence and the generated `zqp_APOrbital_opt.dat` (or the matching
   parallel/general orbital output).
3. Add `InOrbital zqp_APOrbital_opt.dat` to the optimizer's Expert-mode
   namelist and run `mvmc namelist.def`. The filename prefix follows
   `CParaFileHead`; copy the file or use the appropriate input path.

This is an input workflow, not a demonstrated L=512 workaround. The #526
investigation did not obtain a converged L=512 UHF orbital within ten minutes
and did not establish that such an orbital starts the sampler. An arbitrary
scalar rescaling of `initial.def` is insufficient: parameter synchronization
normalizes the Slater block to `D_AMP_MAX=4.0`. Replacing the random default
would require an independently validated orbital with the same parameter map,
followed by a C/Rust initialization comparison. Neither an unconverged UHF
output nor a differently indexed orbital supplies that evidence.

The checked-in C-derived normal-initialization fixtures exercised by
`sampling::normal_initial::tests` and `native274_boundaries` pin the retry,
INFO reduction and RNG behavior independently of this size observation.
Ordinary Rust tests do not invoke the C executable or UHF.

## Reader contract (`readdef.c`) and why it is a separate reader

The UHF reader is not the Expert-mode reader used for `vmc.out`: it has its
own ModPara keys (`Mix`, `EPS`, `Print`, `IterationMax`, `EpsSlater`,
`Ncond`, `2Sz`, `Nelectron`/`Ne`; mVMC-only keys are skipped with a warning),
positional `CDataFileHead`/`CParaFileHead` on lines 6 and 7, its own orbital
mode judgement (`Orbital` only = AP, `Orbital`+`OrbitalParallel` = AP+P,
`OrbitalGeneral` alone = General), `sscanf`/`fscanf` prefix semantics and an
`Initial` Green-function file. `mvmc-expert-parsers` enforces different
contracts (strict diagnostics, different `Ne`/`Ncond` rules), so reusing it
would change what the C tool accepts. The crate therefore has a dedicated
reader (`crates/mvmc-uhf/src/definition.rs`) that mirrors `readdef.c`.

Notable C input rules that are kept:

* `Ncond` (default 0) determines the electron number: with an orbital file
  `Ne = (NLocalSpin + Ncond)/2` (overriding any `Ne`) and `Nsize = 2*Ne`,
  otherwise `Nsize = NLocalSpin + Ncond`. The documentation lists `Ne`, but
  the code does not use it for `Nsize`.
* `2Sz` defaults to the sentinel `-1`; with an orbital file and no General
  orbital, C requires an explicit `2Sz 0`.
* `PairHop` records are stored twice (`(i,j)` and `(j,i)`); orbital maps are
  `OrbitalIdx[2N][2N]` with antisymmetric signs; parallel indices are
  `NOrbitalAP + 2*idx + spin`.
* All fixed-keyword definition files named in the namelist must exist and
  have at least five header lines, even when UHF does not use them.

## C defects and deliberate deviations

Clear C bugs are not reproduced. The Rust behaviour is listed here and was
reported with the PR for issue #350.

| C source | C behaviour | Rust behaviour |
| --- | --- | --- |
| `readdef.c:186-188`, `:438-440` | `fclose(NULL)` after a failed `fopen` of a definition file: segmentation fault | error naming the file, exit 1, no partial outputs |
| `readdef.c:975` (`GetFileName`) | an unknown namelist keyword leaves `itmpKWidx` at `-1` (or stale) and then writes `cFileNameList[itmpKWidx]`: heap underflow / overwritten previous keyword | warning, keyword ignored |
| `readdef.c:70-74` `ReadDefFileError` | returns 0, so every `info = ReadDefFileError(...)` record-count mismatch is only printed and never fails | the message is printed; short files leave zero records (as C); extra records are dropped (C overflows its heap arrays) |
| `readdef.c:609` and the index uses in `readdef.c`, `initial.c`, `makeham.c` | the AP-only `Orbital` reader writes `OrbitalIdx` before `CheckPairSite`; `LocSpn[x0]`, `Initial`, Transfer, InterAll and orbital indices are never range checked | range errors for sites, spins and orbital indices |
| `readdef.c:356-366`, `output.c` gap | `Ncond` not set (`Ne` only, as documented), or `Nsize = 2*Nsite`, gives `Nsize = 0` or no gap index: `EigenValues[-1]`/`[2*Nsite]` is read | error requiring `0 < Nsize < 2*Nsite` |
| `matrixlapack.c:235-275` `ZHEEVall` | a non-zero LAPACK `info` returns 0 and the caller keeps stale arrays | error |
| orbital readers | a short line reuses stale `i`, `j` and index variables | error |

## Verification

* `scripts/check_complex_uhf_c_parity.py --write` builds the unmodified C
  executable and stores inputs, C outputs and provenance (source SHA-256,
  compiler options, OpenBLAS version) under `tests/fixtures/complex_uhf/`.
  Ten cases cover Hubbard chain/square/triangular, InterAll, `Initial`, all
  two-body families, AP, AP+P, General, no orbital file and a non-converged
  run.
* `crates/mvmc-uhf/tests/c_parity.rs` compares iteration counts, residuals,
  energies, particle numbers, eigenvalues, gaps and Green functions with the
  bounds recorded at the top of that file (print-quantum and conditioning
  based, `docs/NUMERICAL_COMPARISONS.md`). Orbital files are compared through
  the gauge-invariant `F F^dagger` and, where `EpsSlater` is small, the SFMT
  noise of each parameter; eigenvector phases are never compared.
* `crates/mvmc-cli/tests/issue350_complex_uhf.rs` runs the command, checks the
  exit codes and feeds `zqp_APOrbital_opt.dat` to the optimizer.

Platform of the stored C results: Linux x86_64, gcc 13.3, OpenBLAS 0.3.26
(Haswell, pthread). On this platform the Rust output files of all ten cases were
byte-identical to C's; the tests nevertheless use the explicit
bounds above so that other BLAS providers (macOS) are covered.

# Native negative INFO for normal initialization (#342)

Closes #342. Related to #274, #185 (multi-rank native trajectory remains #179).

## Conclusion

A negative INFO **is reachable** from the original C kernels for input that C's
readers admit, and C **does not treat it as a failure**. The only admitted trigger
is `Ne <= 0`; for `Ne >= 1` every SKTRF argument check passes, so a negative INFO
is unreachable. The checked native fixtures and Rust tests below make the
`Ne = 0` behaviour the contract (INFO -5, RNG untouched, empty configuration).

## Argument checks and how sizes are derived

Sizes (extern/mVMC-1.3.0/src/mVMC):

- `readdef.c:1890-1891` reads `Ne`/`Nelectron` (`(int) dtmp`), `readdef.c:589-593`
  derives `Ne = (NLocSpin + NCond) / 2` when `Ncond != -1`, `readdef.c:679`
  `Ne = bufInt[IdxNe]`, `readdef.c:757-758` `Nsize = 2*Ne`, `Nsite2 = 2*Nsite`.
- The modpara checks (`readdef.c:588-632`) constrain only NCond parity, 2Sz/orbital
  flags and, for `NLocSpin > 0`, `NLocSpin <= 2*Ne` and NExUpdatePath. There is no
  check that `Ne >= 1` (or `Ne <= Nsite`). `Ne = 0` with `NLocSpin = 0` is admitted.
- Every normal kernel passes `m = n = lda = Nsize`, `nsq = n*n` and calls
  `M_{D,Z}SKTRF("U", "N", &n, invM, &lda, iwork, bufM, &nsq, &info)`:
  `matrix.c:351-352,372` (`calculateMAll_child_fcmp`, ZSKTRF) and
  `matrix.c:577-578,599` (`calculateMAll_child_real`, DSKTRF); FSZ variants
  `matrix.c:145-146,165` and `243-244,264`. A nonzero INFO is returned immediately
  (`if(info!=0) return info;`), before `utu2pfa_*`/`utu2inv_*`.

SKTRF argument validation (`extern/mVMC-1.3.0/src/pfapack/fortran/dsktrf.f`
lines 191-204 and `zsktrf.f` lines 191-204; identical conditions; blocked routine
calls `XERBLA` at `dsktrf.f:218` / `zsktrf.f:219` and returns):

| Check | INFO | C arguments | Satisfied when |
| --- | --- | --- | --- |
| `UPLO` in {U,L} (193) | -1 | `"U"` | always |
| `MODE` in {N,P} (195) | -2 | `"N"` | always |
| `N < 0` (197) | -3 | `n = 2*Ne` | `Ne >= 0` |
| mode P and odd N (199-200) | -3 | mode N | always |
| `LDA < MAX(1,N)` (201-202) | -5 | `lda = n` | `Nsize >= 1`, i.e. **fails for `Ne = 0`** |
| `LWORK < 1` (203-204) | -8 | `lwork = n*n` | `Nsize >= 1` (checked after -5) |

So: `Ne < 0` gives -3, `Ne = 0` gives -5 (the `N = 0` quick return comes after the
checks), and `Ne >= 1` satisfies every check (`n >= 2` even, `lda = n >= max(1,n)`,
`lwork = n*n >= 4`). DSKTF2/DLASKTRF, called with the same `N`/`LDA`, repeat the
`N`/`LDA` checks (`dsktf2.f:185-196`). The remaining normal-path routines have no
INFO: `utu2pfa_*`, `utu2inv_*` (ltl2inv) and `M_{D,Z}SCAL`. `M_ZGETRF/M_DGETRF`
(`matrix.c:501,710`, `pfupdate*.c:541/551`) are BackFlow/two-electron-update paths,
not the normal initializer. Positive INFO (zero pivot) and `qpidx + 1`
(non-finite Pfaffian, `matrix.c:376`, `603`: `if(!isfinite(...)) return qpidx+1`) are
the only non-negative statuses for `Ne >= 1`.

`Ne < 0` is admitted textually but is undefined behaviour in C: `Nsize < 0`
reaches negative-size allocation/loops (observed: DSKTRF INFO -3, then segfault
or heap abort in `vmc.out`). It is not a parity target; Rust rejects it with a
typed precondition error (`negative Ne`).

## How C propagates it

- `vmcmain.c:259` `LapackLWork = getLWork_fcmp()` (`matrix.c:56-74`) issues
  `ZGETRI`/`ZSKPFA` workspace queries with `n = lda = 0`; OpenBLAS XERBLA reports
  arguments 3/5 and returns.
- `vmcmake.c:359-424` `makeInitialSample` loops `do { ... flag =
  CalculateMAll_fcmp(...) ... } while (flag>0)` (`vmcmake.c:410,422`). A negative
  flag leaves the loop after one attempt and returns 0. Only the 101st completed
  call aborts (`vmcmake.c:418-421`).
- `vmcmake_real.c:69`/`vmcmake.c:69-71` call this same complex initializer (not
  `makeInitialSample_real`, which is BackFlow-only), then separately
  `CalculateMAll_real` (`vmcmake_real.c:102`) / `CalculateMAll_fcmp`
  (`vmcmake.c:110`) whose status is ignored.
- Whether the process survives depends on the BLAS provider's `XERBLA`:
  OpenBLAS (the Dev Container, 0.3.26) prints and returns; reference LAPACK's
  `xerbla.f` executes `STOP` (not measured here). The Rust port follows the
  returning provider.

## Native evidence

`c_toolbox/issue342_negative_info/` runs the unmodified original
`makeInitialSample`, `matrix.c`, `projection.c`, `SFMT.c`, pfapack `dsktrf/zsktrf`
and OpenBLAS (Linux x86_64 Dev Container, gcc 13.3.0, libopenblas0 0.3.26). For
`Nsite = 2, Ne = 0, seed 1`, real and complex: `kernel_infos = -5`, 1 attempt,
return value 0, no RNG word consumed (next four words equal a fresh seed-1
stream), `eleIdx` empty, `eleCfg = -1 -1 -1 -1`, `eleNum = 0 0 0 0`, `PfM`
untouched, setup INFO -5 ignored. The `Ne = 1` control returns INFO 0 and consumes
two words. Fixtures: `tests/fixtures/issue342_negative_info/*.stdout` with
`PROVENANCE.txt`.

Whole program (unmodified `vmc.out`, `mpiexec -n 1`, `Ncond 0`, hubbard_chain_real
inputs of `extern/Julia-mVMC/examples/inputs`, `NVMCWarmUp = NVMCSample = 0`):
real prints `ZSKTRF` then `DSKTRF` "parameter number  5 had an illegal value" and
exits 0; complex (`ComplexType 1`) prints `ZSKTRF` twice and exits 0. The later
`StcOpt: r[0]=-nan` is the empty-sample SR step, not the initializer. With the
default sample counts the first proposal (`makeCandidate_hopping`, integer
`% Ne`/`Ne = 0` division) raises SIGFPE; proposals are outside this initializer
contract.

## Rust coverage

`crates/mvmc-core/tests/native342_negative_info.rs` (reads only the fixtures):

- `native342_shared_ne0_negative_info_{real,complex}`: public
  `make_initial_sample_normal_with_info` returns `attempts = 1`,
  `local_info = comm1_info = -5` (typed, not an error), collective statuses
  `[0, -5]`, RNG untouched with the same next four words, configuration and
  Pfaffian storage as C.
- `native342_shared_ne1_control_{real,complex}`: INFO 0 and C's two-word placement.
- `native342_sampler_ne0_negative_info_{real,complex}`: public sampler with zero
  proposals completes; ordinal statuses `[0, 0, -5, 0]` (sampler preflight, shared
  preflight, native INFO, separate setup), zero accepted/saved samples.
- `native342_positive_electron_counts_never_report_negative_info`: smallest
  admitted sizes `(Nsite, Ne)` in `{(1,1),(2,1),(2,2),(3,1)}` with zero, regular and
  NaN matrices produce only non-negative INFO.

The caller-control fixture `tests/fixtures/issue274_caller_control/negative.stdout`
remains the contract for generic negative-INFO retry control (a negative flag is not
a retry). `Ne = 0` is now measured native input rather than source-derived only.

Verified on Linux x86_64 (Rust `test-fast`); no tolerance, RNG algorithm, seed or
fixture value of #274 changed.

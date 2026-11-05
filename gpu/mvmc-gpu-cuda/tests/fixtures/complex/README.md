# Complex normal-orbital inputs for resident GPU Opt/PhysCal (#678, #572 milestone 6)

Julia GPU rejects complex parameters ("CUDA runner requires real parameters",
`ext/cuda_state.jl:39`, pin `7314015`), so there is no Julia GPU fixture. The
references are the Rust CPU C-order complex kernels in `mvmc-core`, evaluated
by the tests on the configurations the GPU sampled. No expected values are
stored: the inputs are the fixtures.

## Inputs

`gen_inputs.py` writes both inputs deterministically
(`uv run --no-project gpu/mvmc-gpu-cuda/tests/fixtures/complex/gen_inputs.py`).
Each directory's `inputs.sha256` lists its file hashes.

| Input | Sites | `2Ne` | `NExUpdatePath` | Planes (`NSPGaussLeg x NMPTrans`) | `NPara` (SR components) |
|---|---|---|---|---|---|
| `hubbard_complex_L16` | 16 | 12 (doped) | 1 (exchange p 0.5) | 8 x 2 = 16 | 25 (50) |
| `hubbard_complex_L128` | 128 | 128 | 0 | 16 | 193 (386) |

Both are periodic Hubbard chains with complex orbitals (`Orbital`
`ComplexType 1`, so C `AllComplexFlag=1`), a Transfer phase
`t = exp(0.25 i)` on every bond with the Hermitian reverse entry,
CoulombIntra `U=4`, PairHop 0.3, Exchange 0.25, a Hermitian pair of complex
spin-conserving InterAll terms (`0.2 ± 0.1i`), real Gutzwiller/Jastrow factors
and two `TransSym` elements: the identity (weight 1) and the affine
permutation `r -> (5r+3) mod L` with the complex weight `0.6+0.8i`. The
orbitals depend on `(j-i) mod L`, so a translation would repeat the identity
plane, and the reflection `r -> L-1-r` transposes the orbital, which the
total-spin projection maps back onto the identity planes (its weight would
only scale the overlap). The affine map gives an independent plane, so the
phase of the weight changes Metropolis decisions. OneBodyG, TwoBodyG and
TwoBodyGEx are measured. These are engine tests, not physics benchmarks.

The Rust parser, the CPU C-order validators and the release `mvmc` CLI accept
both inputs (Opt and PhysCal; see "Timing").

## Physics contract

C's complex normal path (mVMC 1.3.0): `CalculateMAll_fcmp`,
`CalculateNewPfM2`/`UpdateMAll_child_fcmp` (`pfupdate.c`), `vmcmake.c`
(`exp(2 (creal(logIpNew - logIpOld) + LogProjRatio))`, projection on
`creal(Proj)`), `calham.c` with complex coefficients, `locgrn.c`
(`GreenFunc1/2` return `conj(z/ip)`), `calgrn.c` (TwoBodyGEx
`G1[idx0] conj(G1[idx1])`), `SlaterElmDiff_fcmp`, the complex
`StochasticOpt` over `2*NPara` real components (`stcopt.c`,
`stcopt_dposv.c`) and `SyncModifiedParameter` with `cabs`. See
`docs/design/gpu-resident-opt.md` ("Complex normal orbitals").

## Contract and bounds

Tests: `opt::complex_tests` (Opt), `opt::physcal::complex_tests` (PhysCal)
and the host-only `opt::scope_tests::resident_scope_accepts_complex_normal_orbitals`
and `resident_scope_rejects_complex_mode_families_before_cuda`. GPU stage tests
use 8 walkers, seed 79, the fixed formula tape of `../localspin/README.md`,
and 8·Nsite proposals (L128: 128).

Exact:

- Projection counters, projection derivative slots (counter, 0, 0, 0),
  the d/dIm column `i O` against the d/dRe column, supplied-tape
  configurations and acceptance counts. The host replay uses the GPU tape
  contract, C `UpdateProjCnt`/`LogProjRatio`, full CPU complex Pfaffians of
  every candidate and `log|ip|` with the `1e-100` floor. It also replays every
  decision with the imaginary QP weight parts dropped (what Julia GPU would do)
  and requires that some decisions change.
- Opt and PhysCal accept the same moves on the same tape.

FP64, with the #656 condition-aware interval budget
`budget = 16 (aw+1) n eps kappa_F` (`docs/NUMERICAL_COMPARISONS.md`), where
`kappa_F = max_q |A_q|_F |A_q^-1|_F` of the CPU planes:

- Overlaps: `1e-13 + budget sum_q |W_q Pf_q|`.
- Local energies: `1e-13 + budget sum_t |c_t v_t|` (complex modulus).
- Slater derivatives (both slots): `1e-13 + budget max_k |O_k|`.
- PhysCal rows: the same budget relative to `|G|`; TwoBodyGEx products carry
  twice it (`2 |a| |b|`). The device window mean (`physcal_reduce_c`) equals
  the host mean of the same device rows within `4 W eps sum |row|`.
- SR: C's complex system rebuilt on the host from the device rows through C's
  `SROptO` slot layout, in complex arithmetic; normalized backward error of
  the device solution `< 1e-11`. The update moves imaginary Slater components.
- Gauge after one SR update: `1e-15 + 4 eps |b|` against
  `sync_modified_parameter_local` (CUDA `hypot` for C `cabs` is within one
  ulp). The rebuilt state after SR is checked again against a CPU model built
  from the updated parameters.
- Every stage requires complex local energies (`|Im e| > 1e-6`) and complex
  Green functions (`|Im G| > 1e-3`), so the imaginary parts are exercised.

Observed (RTX 3060, CUDA 12.9.2 devcontainer image `b129bd583bbe`):

| Input | max `kappa_F` (initial / final / after SR) | normalized overlap / energy / Slater derivative (bound 16) | PhysCal rows (bound 16) | decisions changed by `Im W` | smallest decision margin | SR backward error | gauge diff |
|---|---|---|---|---|---|---|---|
| `hubbard_complex_L16` | 1.4e2 / 5.0e2 / 5.1e2 | ≤0.003 / ≤0.001 / ≤0.006 | ≤0.019 | 142 | 7.2e-4 | 2.0e-17 | 0 |
| `hubbard_complex_L128` | 1.3e4 / 2.9e4 / 2.4e4 | <0.001 / <0.001 / <0.001 | <0.001 | 186 | 8.7e-4 | 2.8e-17 | 8.9e-16 |

## End-to-end timing and residency (RTX 3060)

Matched workloads as in `../dh/README.md`: seed 79, `initial_def` none, 64
walkers, warmup 128 and interval 128 proposals per walker, 20 SR steps (Opt) or
20 windows (PhysCal). The GPU time is the warm second complete namelist API
call (`resident_opt_complex_repeats_twenty_steps`,
`resident_physcal_complex_repeats_twenty_windows`). The CPU time is the release
`mvmc` CLI (C-order complex runner, default threads on a 36-core host, wall
time including startup and output files), with `NVMCSample 64`,
`NVMCWarmUp 1`, `NVMCInterval 8` (L16) or `1` (L128), and `NVMCCalMode 1`,
`NDataQtySmp 20` for PhysCal.

| Workload | GPU first / warm (s) | CPU C-order (s) |
|---|---|---|
| L16 Opt | 56.9 / 1.74 | 4.07 |
| L128 Opt | 17.9 / 17.9 | 168.5 |
| L16 PhysCal | 93.9 / 1.82 | 4.03 |
| L128 PhysCal | 17.5 / 17.6 | 167.7 |

The GPU first call includes NVRTC module compilation. Every sampling,
measurement and SR series makes 0 explicit uploads, 0 downloads and 1 stream
synchronization (RawExec counters). The complex sampler uses immediate C
rank-two updates with one block-wide update per plane, and the rebuild uses the
generic complex PFAPACK kernel; the real path's delayed queue and paired factor
have no complex counterpart yet, which is why L128 is about 13x slower than the
8-plane real DH L128 input (1.39 s).

The full CUDA gate (`scripts/run_cuda_gate.sh` command, 79 library and 13
integration tests) passes; the real DH L128 (1.38 s) and OptTrans L128
(4.04 s) warm Opt timings are unchanged.

The GPU and CPU use different random streams, so their energies are
statistical observations only. GPU `E[0], E[19]`: L16 `5.96, 0.59`, L128
`102.6, 83.7`.

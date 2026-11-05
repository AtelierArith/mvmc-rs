# RBM inputs for resident GPU Opt/PhysCal (#667, #572 milestone 3)

Julia GPU rejects RBM factors ("CUDA RBM is not supported",
`ext/cuda_state.jl:40`, pin `7314015`), so there is no Julia GPU fixture. The
references are the Rust CPU C-order RBM kernels in `mvmc-core` (C `rbm.c`),
evaluated by the tests on the configurations the GPU sampled. No expected
values are stored: the inputs are the fixtures.

## Inputs

`gen_inputs.py` writes both inputs deterministically
(`uv run --no-project gpu/mvmc-gpu-cuda/tests/fixtures/rbm/gen_inputs.py`,
SHA-256 `732ae490b9f15e9b27c43016d3e0570ac5c58c6c0e0aa7591e974576b20dc77a`).
Each directory's `inputs.sha256` lists its file hashes.

| Input | Sites | `2Ne` | `NExUpdatePath` | Neurons (charge/spin/general) | `NRBM` | Exercises |
|---|---|---|---|---|---|---|
| `hubbard_rbm_L16` | 16 | 12 (doped) | 1 (exchange p 0.5) | 4 / 4 / 2 | 76 | exchanges (two C hops, spin fields change), immediate updates |
| `hubbard_rbm_L128` | 128 | 128 | 0 | 4 / 4 / 2 | 524 | Opt delayed queue and paired inverse (`2Ne >= 128`) |

All nine RBM sections are present and completely mapped: a physical-layer row
for every site (General: every site and spin), a hidden-layer row for every
neuron and a physical-hidden row for every (site[, spin], neuron), with one
flag per declared coefficient. Indices are tied across sites and neurons (see
the generator docstring), so derivative slots accumulate several terms.
Coefficients are real (`ComplexType 0`), every flag is 1, and the `In*RBM`
overlays give binary-fraction values up to 0.25 in magnitude, so hidden fields
are of order one with both signs. Both chains are periodic with Transfer
`t=1`, CoulombIntra `U=4`, PairHop 0.3 and Exchange 0.25, real translational
orbitals, Gutzwiller/Jastrow, identity TransSym and `NSPGaussLeg=8`. There is
no InterAll: C's real InterAll Green kernel omits RBM factors. These are engine
tests, not physics benchmarks.

The tracked `tests/fixtures/rbm/run_rbm_real` stub (width 97, incomplete
mappings) is rejected by C and by the Rust parser, so it is not used.

Acceptance by the references:

- C 1.3.0 readers: every section file of both inputs passes the verbatim
  `GetInfoRBM_Layer`/`GetInfoGeneralRBM_Layer`/`GetInfoRBM_PhysHidden`/
  `GetInfoGeneralRBM_PhysHidden` with status 0, every cell assigned inside the
  declared width and every flag present
  (`c_toolbox/rbm_complete_inputs.c`; see `c_toolbox/README.md`).
- Rust CPU C-order: the release `mvmc` CLI runs both inputs, Opt and PhysCal,
  with real and with complex (`Orbital ComplexType 1`) orbitals.

## Physics contract

The resident path follows C's complex normal path (`vmcmake.c`, `calham.c`,
`calgrn.c`, `vmccal.c`, `rbm.c`) with real coefficients: the RBM factor enters
the Metropolis ratio (C `UpdateRBMCnt` + `LogRBMRatio`), every one- and
two-move operator ratio, and the SR derivatives (real part of C `RBMDiff`),
between projection and Slater. C's real path does not: `vmcmake_real.c` and
`calham_real.c` contain no RBM code, and `VMCMainCal` then passes
`RBMCnt`, which `vmcmake_real.c` never fills, to `RBMDiff`
(`vmccal.c:120,205`; reported upstream as tmisawa/Julia-mVMC#59). Since #669
(PR #671) the Rust CPU real path keeps the RBM factor in sampling and in every
Green kernel, following the same complex-path formulas, so CPU real-mode and
complex-mode RBM runs agree. The resident tests compare per-configuration
quantities with the C-order kernels and replay the tape with C's RBM ratio;
they do not compare long trajectories.

The hidden fields `theta[walker][neuron]` (C `rbmCnt[NRBM_PhysLayerIdx+h]`)
stay on the device. They are rebuilt by C `MakeRBMCnt` after initialization
and after every SR parameter update (C rebuilds at the start of every
`VMCMakeSample`), and advanced by C `UpdateRBMCnt` on every accepted hop or
exchange. A walker refactor leaves them unchanged.

## Contract and bounds

Tests: `opt::rbm_tests` (Opt), `opt::physcal::rbm_tests` (PhysCal) and the
host-only `opt::scope_tests::resident_scope_accepts_rbm_factors`. GPU tests use
8 walkers, seed 79, the fixed formula tape of `../localspin/README.md`, and
8·Nsite proposals (L128: 128).

Exact:

- Projection counters and derivatives, RBM physical-layer derivatives
  (integers), supplied-tape configurations and acceptance counts. The host
  replay uses C `UpdateRBMCnt` (twice for an exchange) and `LogRBMRatio` with
  full CPU Pfaffians. It asserts that the physical counters equal a full
  recount and that the RBM factor changes some decisions.
- Hidden fields, bit for bit: initially against a fresh `make_rbm_cnt`; after
  the tape against the replayed `UpdateRBMCnt` chain; after one SR update
  against `make_rbm_cnt` with the new coefficients. Both sides use the same
  IEEE operations in the same order (separate coupling sum, subtract-then-add
  updates; no FMA on the device). The host test also checks the device tables
  against `make_rbm_cnt` on random occupations.

FP64:

- RBM hidden/coupling derivatives against `set_rbm_diff` on the GPU fields:
  `(m^2 + 3m) eps` for a slot with `m` contributions `x tanh(theta)`
  (`|x tanh| <= 1`; CUDA `tanh` is within 2 ulp and the host within 1; the
  device sums each slot neuron by neuron as C `RBMDiff` does, the host term by
  term, which differs by at most `m(m-1) eps`).
- Local energies, Slater derivatives and PhysCal operator values: the #656
  interval bound of `../dh/README.md`, `1e-13 + 16 (aw+1) n eps kappa_F S`,
  plus an RBM log-ratio budget `H (2 dtheta + 16 eps (1 + max|theta|)) S`
  for `H` neurons. `dtheta` is the measured distance of the incremental fields
  from a fresh `MakeRBMCnt` (`|d log cosh / d theta| <= 1`), and the second
  term covers the per-neuron `exp`/`log`/division roundings. The device takes
  `log` per neuron where C takes one `log` of a block product; that is a
  rounding difference only. The tests also require that zeroing the RBM
  coefficients moves the CPU reference by more than 100 times the bound, so
  the reference includes the RBM factor.
- Overlaps: `1e-12 + 5e-11|b|` (`../opt_julia.md`). Gauge after one SR update:
  `1e-15 + 4 eps|b|` against `sync_modified_parameter_local`; C
  `SyncModifiedParameter` leaves RBM unchanged and rescales Slater only. SR
  backward error `< 1e-11`.

Observed (RTX 3060, CUDA 12.9.2 devcontainer image):

| Input | max `kappa_F` (initial / final) | normalized energy / Slater / Green error (bound 16) | RBM derivative max diff (normalized, bound 1) | decisions changed by RBM | smallest decision margin | SR backward error | gauge diff |
|---|---|---|---|---|---|---|---|
| `hubbard_rbm_L16` | 9.7e2 / 2.5e2 | 0.002 / 0.009 / 0.037 | 4.4e-16 (0.125) | 54 | 2.0e-4 | 3.8e-17 | 0 |
| `hubbard_rbm_L128` | 9.6e4 / 9.0e5 | 0.000 / 0.000 / 0.000 | 8.9e-16 (0.375) | 151 | 8.4e-4 | 1.1e-16 | 0 |

The binary-fraction overlays make every field exactly representable, so the
incremental fields equal a fresh `MakeRBMCnt` here (`dtheta = 0`); after an SR
update they do not, and the exact checks still compare with the C-order chain.
Zeroing the RBM coefficients changes the reference local energy by up to 18
(L16) and 80 (L128). The smallest `|log u - log ratio|` is 2.0e-4, far above
the ~1e-12 log-ratio error.

## End-to-end timing and residency (RTX 3060)

Matched workloads as in `../dh/README.md`: seed 79, `initial_def` none, 64
walkers, warmup 128 and interval 128 proposals per walker, 20 SR steps (Opt) or
20 windows (PhysCal). The GPU time is the warm second complete namelist API
call (`resident_opt_rbm_repeats_twenty_steps`,
`resident_physcal_rbm_repeats_twenty_windows`). The CPU time is the release
`mvmc` CLI (C-order runner, default threads on a 36-core host, wall time
including startup and output files) with `NVMCSample 64`, `NVMCWarmUp 1` and
`NVMCInterval 8` (L16) or `1` (L128). "CPU real" is the input as is; its
sampler omits the RBM factor (see above). "CPU complex" sets
`Orbital ComplexType 1`, which runs C's complex normal path with the RBM
factor in sampling, the same physics as the resident path.

| Workload | GPU first / warm (s) | CPU real (s) | CPU complex (s) |
|---|---|---|---|
| L16 Opt | 32.0 / 0.42 | 0.84 | 1.98 |
| L128 Opt | 1.45 / 1.44 | 20.7 | 71.9 |
| L16 PhysCal | 25.6 / 0.44 | 0.79 | 2.03 |
| L128 PhysCal | 1.39 / 1.39 | 19.9 | 66.4 |

The GPU first call includes NVRTC module compilation (the first test in a
process). Each sampling, measurement and SR series uses 0 explicit uploads,
0 downloads and 1 stream synchronization (RawExec counters). The RBM tables
and coefficients are uploaded once at setup; the Opt gauge refreshes the
device coefficient copy and the hidden fields on the device.

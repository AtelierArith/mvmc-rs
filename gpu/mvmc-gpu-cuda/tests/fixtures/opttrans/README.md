# OptTrans inputs for resident GPU Opt/PhysCal (#674, #572 milestone 5)

Julia GPU rejects variational OptTrans ("CUDA variational OptTrans is not
supported", `ext/cuda_state.jl:43`, pin `7314015`), so there is no Julia GPU
fixture. The references are the Rust CPU C-order kernels in `mvmc-core`,
evaluated by the tests on the configurations the GPU sampled. No expected
values are stored: the inputs are the fixtures.

## Inputs

`gen_inputs.py` writes both inputs deterministically
(`uv run --no-project gpu/mvmc-gpu-cuda/tests/fixtures/opttrans/gen_inputs.py`,
SHA-256 `72671cecbf8f396a2f748e045cb54a6c4810f5ef7f3716cefcafc6a075f60bb5`).
Each directory's `inputs.sha256` lists its file hashes.

| Input | Sites | `2Ne` | `NExUpdatePath` | `NQPOptTrans` | Planes (`NSPGaussLeg x NMPTrans x NQPOptTrans`) | Exercises |
|---|---|---|---|---|---|---|
| `hubbard_opttrans_L16` | 16 | 12 (doped) | 1 (exchange p 0.5) | 3 | 8 x 1 x 3 = 24 | exchanges, immediate updates |
| `hubbard_opttrans_L128` | 128 | 128 | 0 | 3 | 24 | Opt delayed queue and paired inverse (`2Ne >= 128`) |

The three sectors are the identity, the reflection `r -> L-1-r` and the affine
permutation `r -> (5r+3) mod L`, with every sign +1 (C forces +1 for periodic
TransSym, `readdef.c` `GetInfoOptTrans` with `APFlag=0`). The initial weights
are 0.75, 0.5 and -0.375, which setup normalizes to 1, 2/3 and -1/2. The
orbitals depend on `(j-i) mod L` only, so a pure translation sector would
repeat the identity plane; the reflection and the affine map give distinct
planes and distinct derivative columns. Both chains are periodic with Transfer
`t=1`, CoulombIntra `U=4`, PairHop 0.3 and Exchange 0.25, real translational
orbitals, Gutzwiller/Jastrow and identity TransSym. These are engine tests,
not physics benchmarks.

Acceptance by the references: the Rust parser and the CPU C-order validators
accept both inputs. The release `mvmc` CLI runs them with `-o`, Opt and
PhysCal (see "Timing").

## Physics contract

C `qp.c`, `slater.c`, `vmccal.c` and `parameter.c` (mVMC 1.3.0):

- The OptTrans weights follow projection, RBM and Slater
  (`setmemory.c:302`). Plane `q = opt*NQPFix + NSPGaussLeg*mp + sp` composes
  the OptTrans map before QPTrans (`slater.c:52-90`).
- `QPFullWeight[i*NQPFix+j] = OptTrans[i] * QPFixWeight[j]` (C
  `UpdateQPWeight`). The device rebuilds the full weights in `opt_gauge`
  after every SR update and before the rebuild.
- `O_i = sum_j QPFixWeight[j] Pf[i*NQPFix+j] / ip` (C `calculateOptTransDiff`).
  The Slater derivatives sum every plane with its full weight.
- `SyncModifiedParameter` rescales the Slater block only and then sets
  `OptTrans *= 1/max|OptTrans|`.

Two C behaviours are not reproduced:

- C's real-mode derivative slot shift (#370, tmisawa/Julia-mVMC#55). The
  tests check `sum_i OptTrans_i O_i = 1` for each sample, which C's layout
  breaks.
- C's OptTrans component flags. `GetInfoOptTrans` writes `OptFlag[fidx]` with
  an undoubled index (`readdef.c:2400`), so the flags of the OptTrans
  components are never written. The resident parser activates OptTrans from
  its definition and optimizes the real components, as Julia does. The Rust
  C-order CLI (`-o`) keeps them inactive, and its weights stay at their
  normalized input values. A resident run with those flags cleared keeps them
  too, bit for bit.

## Contract and bounds

Tests: `opt::opttrans_tests` (Opt), `opt::physcal::opttrans_tests` (PhysCal)
and the host-only `opt::scope_tests::resident_scope_accepts_opttrans`. GPU
tests use 8 walkers, seed 79, the fixed formula tape of
`../localspin/README.md`, and 8·Nsite proposals (L128: 128).

Exact:

- Projection counters and derivatives, supplied-tape configurations and
  acceptance counts. The host replay uses C `UpdateProjCnt`/`LogProjRatio`
  and full CPU overlaps over all 24 planes. It also replays every decision
  with identity-only weights (OptTrans = 1, 0, 0) and requires that some
  decisions change.
- Device full QP weights, bit for bit, against `update_qp_weight_for`
  (one IEEE product each), initially, after the tape and after one SR update.
- OptTrans normalization after one SR update, bit for bit, against
  `sync_modified_parameter_local`. The update moves every OptTrans weight and
  leaves the maximum amplitude different from one, so the normalization runs.

FP64:

- Local energies, Slater derivatives and PhysCal operator values: the #656
  interval bound of `../dh/README.md`, `1e-13 + 16 (aw+1) n eps kappa_F S`.
- OptTrans derivatives against `opt_trans_diff`. With per-plane relative
  Pfaffian errors bounded by the same `budget = 16 (aw+1) n eps kappa_F`,
  `|dO_i| <= budget (s_i + |O_i| c) + 4 (NQPFix+2) eps (s_i + |O_i| c) + 1e-14`,
  where `s_i = sum_j |w_j Pf_ij| / |ip|` and `c = sum_q |W_q Pf_q| / |ip|` is the
  cancellation factor of the total overlap. The second term covers the
  fixed-order sums and the division: the device divides as C does (`A/ip`),
  and the CPU uses Julia's complex division (`A * (1/ip)`), at most one ulp
  apart.
- `sum_i OptTrans_i O_i = 1` within `1e-10 max(c, 1)`.
- Overlaps: `1e-12 + 5e-11|b|` (`../opt_julia.md`). Gauge after one SR update:
  `1e-15 + 4 eps|b|` against `sync_modified_parameter_local`; SR backward
  error `< 1e-11`.
- After the SR update, the rebuilt device state is compared with a CPU model
  built from the updated parameters, so the refreshed weights reach sampling
  and measurement.
- Identity-only weights must move the CPU reference energy by more than 100
  times the bound, and the sector derivatives must differ by more than 1e-3,
  so the reference includes every sector.

Observed (RTX 3060, CUDA 12.9.2 devcontainer image `b129bd583bbe`):

| Input | max `kappa_F` (initial / final / after SR) | normalized energy / Slater / Green error (bound 16) | OptTrans derivative max diff | decisions changed by OptTrans | smallest decision margin | SR backward error | gauge diff |
|---|---|---|---|---|---|---|---|
| `hubbard_opttrans_L16` | 6.6e4 / 1.0e5 / 3.6e3 | 0.002 / 0.050 / 0.145 | 6.1e-14 (0.002 of the bound) | 100 | 8.5e-3 | 1.6e-17 | 0 |
| `hubbard_opttrans_L128` | 4.1e5 / 3.0e5 / 1.9e6 | <0.001 / <0.001 / 0.001 | 2.6e-12 (<0.001 of the bound) | 99 | 5.7e-3 | 1.5e-17 | 0 |

Every bound holds with a wide margin. The derivative spread between sectors is
0.7-2.0, so the three columns are far from degenerate. Per stage, the
largest local-energy difference between the reference and identity-only
weights is 2.0 to 6.0e3 (L128) and 4.6 to 30 (L16). Opt and PhysCal accept exactly the same moves on the same tape, so the delayed (Opt)
and immediate (PhysCal) inverse updates make the same decisions.

## End-to-end timing and residency (RTX 3060)

Matched workloads as in `../dh/README.md`: seed 79, `initial_def` none, 64
walkers, warmup 128 and interval 128 proposals per walker, 20 SR steps (Opt) or
20 windows (PhysCal). The GPU time is the warm second complete namelist API
call (`resident_opt_opttrans_repeats_twenty_steps`,
`resident_physcal_opttrans_repeats_twenty_windows`). The CPU time is the
release `mvmc -o` CLI (C-order runner, default threads on a 36-core host, wall
time including startup and output files), with `NVMCSample 64`,
`NVMCWarmUp 1`, `NVMCInterval 8` (L16) or `1` (L128), and `NDataQtySmp 20`
for PhysCal. Following C's flags, the CPU run keeps the OptTrans weights fixed,
while the GPU run optimizes them. The sampling and measurement cost (24 planes)
is the same.

| Workload | GPU first / warm (s) | CPU C-order `-o` (s) |
|---|---|---|
| L16 Opt | 32.4 / 1.09 | 1.90 |
| L128 Opt | 4.07 / 4.08 | 47.7 |
| L16 PhysCal | 25.9 / 1.19 | 1.92 |
| L128 PhysCal | 4.54 / 4.54 | 47.0 |

The GPU first call includes NVRTC module compilation (the first test in a
process). Every sampling, measurement and SR series makes 0 explicit uploads,
0 downloads and 1 stream synchronization (RawExec counters). The fixed QP
weights are uploaded once at setup. The full weights are refreshed on the
device. The 24 planes cost about three times the 8-plane DH/RBM inputs (L128
Opt 1.39-1.44 s).

The two implementations use different random streams, so their energies are
statistical observations only. L128 Opt `E[0], E[19]`: GPU `94.83, 76.86`
(final OptTrans `1, 0.871, -0.104`), GPU with the OptTrans flags cleared
`E[19] = 76.18`, CPU `-o` `94.95, 71.95`. L16 GPU final OptTrans
`1, 0.636, -0.575`.

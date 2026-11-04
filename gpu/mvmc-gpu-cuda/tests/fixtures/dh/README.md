# Doublon-holon (DH2/DH4) inputs for resident GPU Opt/PhysCal (#665, #572 milestone 2)

Julia GPU rejects DH factors (`ext/cuda_state.jl:41-42`, `:48-49`, pin
`7314015`), so there is no Julia GPU fixture. The references are the Rust CPU
C-order kernels in `mvmc-core`, evaluated by the tests on the configurations
the GPU sampled. No expected values are stored: the inputs are the fixtures.

## Inputs

`gen_inputs.py` writes both inputs deterministically
(`uv run --no-project gpu/mvmc-gpu-cuda/tests/fixtures/dh/gen_inputs.py`,
SHA-256 `b66f0dc3b6fc88fb54486c4e6862fc449cb735533c16ceef3c31bd2b779cfd3b`).
Each directory's `inputs.sha256` lists its file hashes.

| Input | Sites | `2Ne` | `NExUpdatePath` | DH2 tables | DH4 tables | Exercises |
|---|---|---|---|---|---|---|
| `hubbard_dh_L16` | 16 | 12 (doped) | 1 (exchange p 0.5) | 2 (shells 1, 2) | 1 (shells 1+2) | two DH2 tables, exchanges (no DH change), immediate updates |
| `hubbard_dh_L128` | 128 | 128 | 0 | 1 (shell 1) | 1 (shells 1+2) | Opt delayed queue and paired inverse (`2Ne >= 128`) |

Both are periodic chains with Transfer `t=1`, CoulombIntra `U=4`, and
nearest-neighbour PairHop (0.3) and Exchange (0.25), so the local energy
contains one- and two-electron DH ratios. They use real translational
orbitals, Gutzwiller/Jastrow, identity TransSym and `NSPGaussLeg=8`.
InGutzwiller/InDH2/InDH4 give nonzero initial projection factors. All
optimization flags are 1, so C `SetFlagShift` enables the GJ, DH2 and DH4
shifts. The six-site `tests/fixtures/dh2/production_real` and
`dh4/production_dh4_real` inputs are also tested. These are engine tests, not
physics benchmarks.

## Contract and bounds

Tests: `opt::dh_tests` (Opt) and `opt::physcal::dh_tests` (PhysCal), 8 walkers,
seed 79, the fixed formula tape of `../localspin/README.md`, 8·Nsite proposals
(L128: 128).

Exact (integers):

- DH/Gutzwiller/Jastrow counters of every walker equal `make_proj_cnt`
  (C `MakeProjCnt`), initially and after the tape. Projection derivatives
  equal the counters.
- Supplied-tape configurations and acceptance counts equal a host replay of
  the resident tape contract (`interval_kernels.cu`). The replay uses C
  `UpdateProjCnt`/`LogProjRatio` and full CPU Pfaffians (`calc_m_all_real`)
  for every candidate. It asserts that `UpdateProjCnt` agrees with a full
  recount, that an exchange changes no counter, and that some proposals change
  a DH counter.

FP64:

- Overlaps: `1e-12 + 5e-11|b|`, the resident stage bound (`../opt_julia.md`).
- Local energies, Slater derivatives (`slater_elm_diff`) and PhysCal operator
  values (`calculate_local_energy`, `green_func1`, `green_func2`):
  `1e-13 + 16 (aw+1) n eps kappa_F S`. This is the #656 interval bound
  (`docs/NUMERICAL_COMPARISONS.md`). `kappa_F = |A|_F |A^-1|_F` is the largest
  QP-plane condition of the walker, from the CPU factorization. `aw` is the
  number of accepted updates since the walker's last refactor. `S` is
  `sum_t |c_t r_t|` (energy), the walker's largest derivative, or `|value|`
  (Green). The first difference is the device inverse (paired/blocked factor
  plus rank-two or queued updates) against CPU PFAPACK, and every ratio
  inherits its forward error. The random initial L128 configurations have
  `kappa_F` up to 4e5. That gives energy differences of 8e-9 at `|E| ~ 1e2`,
  which a fixed `1e-10` relative bound would not justify.
- Device gauge after one SR update: compared with
  `sync_modified_parameter_local(data, true)` on the same pre-gauge
  parameters, `1e-15 + 4 eps|b|`. The two use the same IEEE operations in the
  same order, without FMA. The SR solve has a CPU-reconstructed backward
  error `< 1e-11`.

Observed (RTX 3060, CUDA 12.9.2 devcontainer image):

| Input | max `kappa_F` (initial / final) | normalized energy / derivative / Green error (bound 16) | smallest decision margin | SR backward error | gauge diff |
|---|---|---|---|---|---|
| DH2 production (6 sites) | 9.3e3 / 8.7e1 | 0.007 / 0.013 / n/a | 2.3e-2 | 1.9e-17 | 0 |
| DH4 production (6 sites) | 9.3e3 / 8.7e1 | 0.003 / 0.013 / n/a | 8.4e-3 | 5.9e-18 | 0 |
| `hubbard_dh_L16` | 6.6e4 / 3.5e3 | 0.001 / 0.001 / 0.027 | 3.3e-3 | 2.1e-17 | 0 |
| `hubbard_dh_L128` | 4.1e5 / 4.9e4 | 0.000 / 0.001 / 0.012 | 1.9e-3 | 5.3e-16 | 0 |

The smallest `|log u - log ratio|` is at least 1.9e-3. The log-ratio error is
about 1e-12, so no decision is near a numerical tie. After the gauge, the bin
mean of every DH generator is zero to 3e-17.

## End-to-end timing and residency (RTX 3060)

Matched workloads: seed 79, `initial_def` none, 64 walkers, warmup 128 and
interval 128 proposals per walker, 20 SR steps (Opt) or 20 windows (PhysCal).
The GPU time is the warm second complete namelist API call
(`resident_opt_dh_repeats_twenty_steps`,
`resident_physcal_dh_repeats_twenty_windows`). The CPU time is the release
`mvmc` CLI (C-order runner, wall time including startup and output files) on
the same input. It uses `NVMCSample 64`, `NVMCWarmUp 1`, and `NVMCInterval 8`
(L16) or `1` (L128), so each step or window samples `64 x 128` proposals plus
one interval of warmup. The time is the same with 1 and 36 threads.

| Workload | GPU first / warm (s) | CPU C-order (s) |
|---|---|---|
| L16 Opt | 28.0 / 0.39 | 0.82 |
| L128 Opt | 1.39 / 1.39 | 16.8 |
| L16 PhysCal | 20.7 / 0.40 | 0.83 |
| L128 PhysCal | 1.58 / 1.59 | 16.2 |

The GPU first call includes NVRTC module compilation (the first test in a
process). Each sampling, measurement and SR series uses 0 explicit uploads,
0 downloads and 1 stream synchronization (RawExec counters). The DH tables are
uploaded once at setup. The two implementations use different random streams,
so their energies are statistical observations only. L128 Opt `E[0], E[19]`:
GPU `94.27, 69.47`, CPU `95.65, 66.45`.

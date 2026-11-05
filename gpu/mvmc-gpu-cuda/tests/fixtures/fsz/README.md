# General-orbital / FSZ inputs for resident GPU Opt/PhysCal (#714, #572 milestone 7)

Julia GPU rejects FSZ ("CUDA requires sz-conserved orbitals",
`ext/cuda_state.jl:38`, pin `7314015`), so there is no Julia GPU fixture. The
references are the Rust CPU C-order FSZ kernels in `mvmc-core`, evaluated by
the tests on the configurations the GPU sampled. No expected values are
stored: the inputs are the fixtures.

## Inputs

`gen_inputs.py` writes the three inputs deterministically
(`uv run --no-project gpu/mvmc-gpu-cuda/tests/fixtures/fsz/gen_inputs.py`).
Each directory's `inputs.sha256` lists its file hashes.

| Input | Sites | `2Ne` | `2Sz` | `NExUpdatePath` | Orbitals | Planes | `NPara` |
|---|---|---|---|---|---|---|---|
| `hubbard_fsz_L16` | 16 | 12 (doped) | not set (C default -1) | 1 (exchange p 0.5) | `OrbitalGeneral`, 46 classes | 2 | 55 |
| `kondo_fsz_2sz2` | 8 conduction + 8 local | 14 | 2 (8 up, 6 down) | 1 | `Orbital` + `OrbitalParallel` (C selects general) | 2 | 55 |
| `hubbard_fsz_L128` | 128 | 128 | not set | 0 | `OrbitalGeneral`, 382 classes | 2 | 447 |

- C reads `2Sz=-1` only as the default; an explicit `2Sz -1` is an error
  (`readdef.c:2215-2218`), so the Sz-free inputs omit the line.
- The Hubbard chains are periodic, with Transfer `t=1`, a real spin-flip
  hopping `0.35 (c+_{j dn} c_{i up} - c+_{j up} c_{i dn}) + h.c.` on every
  bond, CoulombIntra `U=4`, PairHop 0.3, Exchange 0.25, a Hermitian pair of
  spin-changing InterAll terms (`S+_0 S-_1` type, 0.2) and a spin-conserving
  pair (0.15).
- The Kondo chain has a conduction ring with `t=1`, `U=4`, and Exchange 0.5,
  Hund 0.25 and CoulombInter -0.125 between conduction site `i` and local
  spin `i+8`. Its Hamiltonian conserves spin, as C requires for a fixed
  `2Sz` (GetInfoInterAll).
- OneBodyG, TwoBodyG (and TwoBodyGEx on the Hubbard chains) include
  spin-flip pairs.
- General orbitals are indexed by the class `(spin_i, spin_j, (j - i) mod L)`
  of each pair `rsi < rsj`. The Kondo AP/P orbitals depend on `(j - i) mod 16`.
- Two TransSym elements: the identity (weight 1) and the affine permutation
  `r -> (5r+3) mod L` (weight 0.5), as in `../complex`. `NSPGaussLeg=1`:
  C forces it for general orbitals (`readdef.c:703-707`).
- Real Gutzwiller/Jastrow factors with a nonzero InGutzwiller.

These are engine tests, not physics benchmarks. The Rust parser, the CPU
C-order validators and the release `mvmc` CLI accept all three inputs (Opt
and PhysCal; see "Timing").

## Physics and sampler contract

C 1.4.0's real FSZ path (`iFlgOrbitalGeneral=1`, `AllComplexFlag=0`):
`UpdateSlaterElm_fsz`, `CalculateMAll_fsz_real`, `CalculateNewPfM2_fsz_real`
and `UpdateMAll_fsz_real`, `calham_fsz_real.c`, `locgrn_fsz_real.c`,
`SlaterElmDiff_fsz`, `calgrn_fsz.c` and `UpdateProjCnt_fsz`. The sampler
extends the Julia GPU five-word tape; see `docs/design/gpu-resident-opt.md`
("General orbitals / FSZ") and the header of `src/interval_kernels.cu`.

## Contract and bounds

Tests: `opt::fsz_tests` (Opt), `opt::physcal::fsz_tests` (PhysCal) and the
host-only `opt::scope_tests::resident_scope_accepts_fsz_general_orbitals`
and `resident_scope_rejects_fsz_families_before_cuda`. GPU stage tests use 8
walkers, seed 79, the fixed formula tape of `../localspin/README.md`, and
8·Nsite proposals (L128: 128).

Host-only (no CUDA):

- The device Slater table (`opt_slater_f` on the model tables) against
  `update_slater_elm_fsz`, entry by entry, within `eps |b|` (one subtraction
  of `±1` multiples; observed 0).
- The sparse derivative basis reproduces the table, `S = sum_p F_p dS/dF_p`,
  within `4 eps |S|`.
- Operators are `rs`-encoded, and the Sz-free inputs contain spin flips.

Exact (integers):

- Initial configurations: every slot holds a distinct `rs`, the labels below
  `Ne + 2Sz/2` are up (C `makeInitialSample_fsz`), and every local-spin site
  holds one electron.
- Projection counters equal `make_proj_cnt`; projection derivatives equal
  the counters.
- Supplied-tape configurations and acceptance counts equal a host replay of
  the FSZ tape contract. The replay uses C `UpdateProjCnt_fsz` (Rust
  `update_proj_cnt`), `LogProjRatio`, and full CPU FSZ Pfaffians
  (`calc_m_all_fsz_real`) of every candidate. It asserts that an exchange
  changes no counter and that a fixed `2Sz` keeps its up count. It requires
  accepted spin-changing moves for `2Sz=-1` and accepted exchanges for
  `NExUpdatePath=1`.
- Opt and PhysCal accept the same moves on the same tape.

FP64, with the #656 condition-aware interval budget
`budget = 16 (aw+1) n eps kappa_F` (`docs/NUMERICAL_COMPARISONS.md`), where
`kappa_F = max_q |A_q|_F |A_q^-1|_F` of the CPU planes:

- Overlaps: `1e-13 + budget sum_q |W_q Pf_q|`.
- Local energies (`calculate_local_energy_fsz`): `1e-13 + budget sum_t
  |c_t v_t|`.
- Slater derivatives (`slater_elm_diff_fsz`): `1e-13 + budget max_k |O_k|`.
- PhysCal rows (C PhysCal's FSZ Green kernels, `green_func1_fsz_complex`
  and `green_func2_fsz_complex` on the complex shadow, as in C
  `calgrn_fsz.c`): the same budget relative to `|G|`; TwoBodyGEx products
  carry twice it. The device window mean equals the host mean of the same
  device rows within `4 W eps sum |row|`.
- SR: the normalized backward error of the device solution against the
  host-rebuilt system, `< 1e-11`.
- Gauge after one SR update: `1e-15 + 4 eps |b|` against
  `sync_modified_parameter_local`. The rebuilt state after SR is checked
  again against a CPU model built from the updated parameters.
- #573: one invalid plane per walker routes energies through
  `opt_energy_fallback` and derivatives through `opt_cofactors`. Both match
  the CPU, the supplied-tape trajectory with full-Pfaffian proposals and
  in-kernel refactors equals the replay, and the refactor restores
  validity.

Observed (RTX 3060, CUDA 12.9.2 devcontainer image `b129bd583bbe`), as the
largest normalized error (bound 16):

| Input | max `kappa_F` (initial / final / after SR) | overlap / energy / Slater derivative | PhysCal rows | accepted spin moves / exchanges | smallest decision margin | SR backward error | gauge diff |
|---|---|---|---|---|---|---|---|
| `hubbard_fsz_L16` | 2.2e3 / 1.7e2 / 1.5e2 | ≤0.009 / ≤0.005 / ≤0.011 | ≤0.96 | 120 / 39 | 1.2e-4 | 5.2e-18 | 0 |
| `kondo_fsz_2sz2` | 4.1e2 / 1.8e2 / 1.9e2 | ≤0.005 / ≤0.005 / ≤0.011 | ≤0.069 | 0 / 96 | 5.3e-3 | 1.5e-17 | 0 |
| `hubbard_fsz_L128` | 4.0e4 / 1.4e5 / 2.5e4 | <0.001 / <0.001 / ≤0.001 | ≤0.005 | 156 / 0 | 1.1e-3 | 1.5e-17 | 0 |

The invalid-plane test (L16) is within 0.011 of its budget. The first
floating difference is in the independent factorization: the device inverse
(PFAPACK rebuild, paired factor at `2Ne=128`, rank-two or queued updates)
against the CPU PFAPACK inverse. Every supplied-tape decision still matched.
The smallest decision margins above bound the rounding a decision could
tolerate.

## End-to-end timing and residency (RTX 3060)

Matched workloads as in `../dh/README.md`: seed 79, `initial_def` none, 64
walkers, warmup 128 and interval 128 proposals per walker, 20 SR steps (Opt)
or 20 windows (PhysCal). The GPU time is the warm second complete namelist API
call (`resident_opt_fsz_repeats_twenty_steps`,
`resident_physcal_fsz_repeats_twenty_windows`). The CPU time is the release
`mvmc` CLI (C-order FSZ runner, default threads on a 36-core host, wall time
including startup and output files), with `NVMCSample 64`, `NVMCWarmUp 1`,
`NVMCInterval 8` (16 sites) or `1` (L128), and `NVMCCalMode 1`,
`NDataQtySmp 20` for PhysCal.

| Workload | GPU first / warm (s) | CPU C-order (s) |
|---|---|---|
| `hubbard_fsz_L16` Opt | 44.0 / 0.087 | 0.51 |
| `kondo_fsz_2sz2` Opt | 0.092 / 0.092 | 0.56 |
| `hubbard_fsz_L128` Opt | 0.50 / 0.50 | 16.4 |
| `hubbard_fsz_L16` PhysCal | 33.3 / 0.092 | 0.48 |
| `kondo_fsz_2sz2` PhysCal | 0.083 / 0.083 | 0.55 |
| `hubbard_fsz_L128` PhysCal | 0.51 / 0.51 | 15.4 |

The GPU first call of a process includes NVRTC module compilation. Every
sampling, measurement and SR series makes 0 explicit uploads, 0 downloads and
1 stream synchronization (RawExec counters). The FSZ walkers reuse the real
sampler unchanged, including the Opt delayed queue and the paired factor at
`2Ne=128`; these inputs have 2 QP planes, against 8 for the normal-orbital
L128 fixtures, which is why L128 runs faster here than the DH L128 input
(1.39 s).

The GPU and CPU use different random streams, so their energies are
statistical observations only. `E[0], E[19]` on the GPU / CPU: L16
`7.10, 3.34` / `8.64, 3.03`; Kondo `-0.011, -0.485` / `0.722, -1.084`; L128
`108.5, 90.8` / `108.9, 93.4`.

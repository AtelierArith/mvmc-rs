# 4. Theory III: Markov-chain sampling

[Contents](README.md) · Previous: [3. The variational wave function](03-theory-wavefunction.md) · Next: [5. Stochastic reconfiguration](05-theory-sr.md)

The configurations $x$ of [chapter 2](02-theory-vmc-hamiltonian.md) are drawn
from $\rho(x)\propto|\psi(x)|^2$ with the Metropolis algorithm. All moves are
*local*: one electron hops, or two electrons of opposite spin exchange sites, or
(FSZ path) a spin is flipped. After each accepted move the Pfaffian and its
inverse are updated by low-rank formulas instead of being recomputed.

This chapter documents the algorithm **together with its random-number
contract**. The sequence of `gen_rand32()` and `genrand_real2()` draws on a fixed
control path is part of the C behaviour and is preserved exactly in Rust
(random-number state is compared exactly, floating-point values with
tolerances; see [11.4](11-compatibility.md#114-numerical-comparison-policy)).

## 4.1 The sampler loop

For one SR step (or one PhysCal sample) `VMCMakeSample` runs this procedure:

1. **Start configuration.** On the very first call (`BurnFlag == 0`) a random
   configuration is generated ([4.2](#42-initial-configuration)). On later calls the last
   configuration of the previous step is reused (`copyFromBurnSample` restores `eleIdx`, `eleCfg`, `eleNum` and the projection counters; the RBM counters are rebuilt by `MakeRBMCnt`); this is the "burn-in carry-over".
2. Compute all Pfaffians and inverses, $\ln\mathrm{IP}$. If $\ln\mathrm{IP}$ is
   not finite, draw a new initial configuration and recompute.
3. **Outer steps.** The number of outer steps is
   $$
   n_{\rm out}=\begin{cases}\texttt{NVMCWarmUp}+\texttt{NVMCSample}&\text{first call},\\ \texttt{NVMCSample}+1&\text{later calls}.\end{cases}
   $$
   Each outer step performs $n_{\rm in}=\texttt{NVMCInterval}\times N_s$ proposals
   ([4.3](#43-proposals)). At the end of an outer step $o\ge n_{\rm out}-\texttt{NVMCSample}$ the current
   configuration, its projection counters (and RBM counters) and $\ln\mathrm{IP}$ are saved as
   sample number $o-(n_{\rm out}-\texttt{NVMCSample})$.
4. **Refresh.** Whenever more than $N_s$ proposals have been accepted since the last
   full evaluation (`nAccept > Nsite`), the Pfaffians/inverses are recomputed from
   scratch and $\ln\mathrm{IP}$ is re-evaluated, then the counter is reset. This bounds
   the accumulation of round-off in the rank-2 updates.
5. Store the final configuration for the next call and set `BurnFlag = 1`.

> **Implementation**
> - C: `VMCMakeSample` — `extern/mVMC-1.3.0/src/mVMC/vmcmake.c:45`
> - C: `VMCMakeSample_real` — `extern/mVMC-1.3.0/src/mVMC/vmcmake_real.c:45`
> - C: `VMCMakeSample_fsz` — `extern/mVMC-1.3.0/src/mVMC/vmcmake_fsz.c:42`
> - C: `saveEleConfig` — `extern/mVMC-1.3.0/src/mVMC/vmcmake.c:445`
> - Rust: `vmc_make_sample_with_reducer_timed` — `crates/mvmc-core/src/sampling/driver.rs:623`
> - Rust: `vmc_make_sample_real_with_reducer_timed` — `crates/mvmc-core/src/sampling/driver.rs:122`
> - Rust: `vmc_make_sample_fsz_with_reducer_timed` — `crates/mvmc-core/src/sampling/driver.rs:1175`
> - Rust: `vmc_make_sample` — `crates/mvmc-core/src/sampling/driver.rs:604`
> - Parity: the number of outer steps, the saved-sample index and the `nAccept > Nsite` refresh follow `vmcmake.c:141, 309, 334-349` (Rust `driver.rs:271-278, 532-552, 558-562`). The refresh test uses a strict `>`. `copyToBurnSample` after the last step is reproduced by the "burn" buffers of `ElectronConfiguration`; the carry-over is detected by `counter[9] != 0` (`driver.rs:167`).

## 4.2 Initial configuration

`makeInitialSample` builds a random configuration:

1. For every site with a localized spin (`LocSpn[ri] == 1`), repeatedly draw
   `mi = gen_rand32() % Ne`, `si = (genrand_real2() < 0.5) ? 0 : 1` until the pair `(mi, si)`
   is unused, and place that electron on the site.
2. For each spin $s=0,1$ and each remaining electron $m$, draw
   `ri = gen_rand32() % Nsite` repeatedly until the site is empty for spin $s$ and
   not a local-spin site.
3. Set `eleNum`, build the projection counters (`MakeProjCnt`) and compute the
   Pfaffians. If any rank reports a singular matrix, repeat (at most 100 attempts,
   then abort).

> **Implementation**
> - C: `makeInitialSample` — `extern/mVMC-1.3.0/src/mVMC/vmcmake.c:359`
> - Rust: `make_initial_sample` — `crates/mvmc-core/src/sampling/initial.rs:76`
> - Rust: `make_initial_sample_normal_with_info` — `crates/mvmc-core/src/sampling/normal_initial.rs:251`
> - Rust: `make_initial_sample_fsz` — `crates/mvmc-core/src/sampling/initial.rs:273`
> - Rust: `generate_initial_fsz_configuration` — `crates/mvmc-core/src/sampling/initial.rs:198`
> - Rust: `init_loc_spn` — `crates/mvmc-core/src/sampling/projection.rs:159`
> - Parity: the draw order (local spins first, then spin 0 electrons, then spin 1 electrons), the retry limit of 100 and the collective maximum of the Pfaffian status across ranks (`MPI_Allreduce(..., MPI_MAX)`) are reproduced; the typed status/coordination tests are in `normal_initial.rs`.

## 4.3 Proposals

Every inner step first selects an **update type** from `NExUpdatePath`
(`getUpdateType`):

| `NExUpdatePath` | Draws | Result |
|-----------------|-------|--------|
| 0 | none | hopping |
| 1 | one `genrand_real2`; $<0.5$ | exchange, else hopping |
| 2, sz-conserved (`iFlgOrbitalGeneral == 0`) | none | exchange |
| 2, general orbital with `2Sz = -1` (not fixed) | one `genrand_real2` | exchange ($<0.5$) or local spin flip |
| 2, general orbital with `2Sz` fixed | none | exchange |
| 3 (Kondo-type) | one, and a second if the first $\ge0.5$ | hopping ($<0.5$), else exchange or local spin flip by the second draw |
| other | none | no move |

The C manual's rule of thumb: `0` hopping only, `1` hopping + exchange for
electron systems, `2` for spin systems (`NLocSpin = 2Ne`, enforced by the C
reader).

**Hopping candidate** (`makeCandidate_hopping`):

1. Repeat `mi = gen_rand32() % Ne; s = (genrand_real2() < 0.5) ? 0 : 1; ri = eleIdx[mi + s*Ne]` while site `ri` carries a localized spin.
2. Repeat `rj = gen_rand32() % Nsite` while `eleCfg[rj + s*Nsite] != -1` or `LocSpn[rj] == 1`, giving up after more than `Nsite*Nsite` attempts (then `rejectFlag = 1`).

**Exchange candidate** (`makeCandidate_exchange`): reject immediately (no draw) if no
site is singly occupied. Otherwise repeat `(mi, s, ri)` as above until the opposite-spin
site of `ri` is empty, then repeat `mj = gen_rand32() % Ne` with spin $t=1-s$ until the same
holds for `rj = eleIdx[mj + t*Ne]`. The move swaps the two electrons: electron
`mi` ($s$) hops $r_i\to r_j$ and electron `mj` ($t$) hops $r_j\to r_i$.

**Local spin flip** (FSZ only): `makeCandidate_LocalSpinFlip_localspin` and
`makeCandidate_LocalSpinFlip_conduction` flip the spin label of an electron on a
localized-spin site or of a conduction electron, respectively.

A candidate with `rejectFlag` set causes the loop to `continue` **before** the
Metropolis draw, so no `genrand_real2` is consumed; every other proposal consumes
exactly one `genrand_real2` in the acceptance test, even when the weight is
non-finite or zero.

> **Implementation**
> - C: `getUpdateType` — `extern/mVMC-1.3.0/src/mVMC/vmcmake.c:606`
> - C: `makeCandidate_hopping` — `extern/mVMC-1.3.0/src/mVMC/vmcmake.c:515`
> - C: `makeCandidate_exchange` — `extern/mVMC-1.3.0/src/mVMC/vmcmake.c:548`
> - C: `updateEleConfig` — `extern/mVMC-1.3.0/src/mVMC/vmcmake.c:585`
> - C: `makeCandidate_hopping_fsz` — `extern/mVMC-1.3.0/src/mVMC/vmcmake_fsz.c:594`
> - C: `makeCandidate_exchange_fsz` — `extern/mVMC-1.3.0/src/mVMC/vmcmake_fsz.c:670`
> - C: `makeCandidate_LocalSpinFlip_localspin` — `extern/mVMC-1.3.0/src/mVMC/vmcmake_fsz.c:728`
> - C: `makeCandidate_LocalSpinFlip_conduction` — `extern/mVMC-1.3.0/src/mVMC/vmcmake_fsz.c:756`
> - Rust: `get_update_type` — `crates/mvmc-core/src/sampling/candidate.rs:306`
> - Rust: `make_candidate_hopping` — `crates/mvmc-core/src/sampling/candidate.rs:136`
> - Rust: `make_candidate_exchange` — `crates/mvmc-core/src/sampling/candidate.rs:216`
> - Rust: `make_candidate_hopping_fsz` — `crates/mvmc-core/src/sampling/candidate.rs:373`
> - Rust: `make_candidate_exchange_fsz` — `crates/mvmc-core/src/sampling/candidate.rs:606`
> - Rust: `make_candidate_local_spin_flip_localspin` — `crates/mvmc-core/src/sampling/candidate.rs:555`
> - Rust: `make_candidate_local_spin_flip_conduction` — `crates/mvmc-core/src/sampling/candidate.rs:480`
> - Rust: `update_ele_config` — `crates/mvmc-core/src/sampling/projection.rs:416`
> - Rust: `revert_ele_config` — `crates/mvmc-core/src/sampling/projection.rs:441`
> - Parity: draw counts and order are exact: `gen_rand32() % n` (`Sfmt19937Rng::gen_rand32`, `crates/sfmt19937/src/lib.rs:138`) and `genrand_real2` (`crates/sfmt19937/src/lib.rs:182`) are consumed in the same sequence, including draws on rejected moves and in retry loops. The test `rejected_candidate_consumes_no_rng_and_mutates_nothing` (`one_move.rs:640`) pins the "no draw on `rejectFlag`" rule.

## 4.4 Acceptance test

For a hop of electron $m$ ($r_i\to r_j$) with new counters $c'$, the Metropolis weight is the
squared amplitude ratio,

$$
w=\Big|\frac{\psi(x')}{\psi(x)}\Big|^2
=\exp\Big\{2\operatorname{Re}\big[\Delta\ln P+\Delta\ln\mathcal N_{\rm RBM}+\ln\mathrm{IP}(x')-\ln\mathrm{IP}(x)\big]\Big\},
$$

with $\Delta\ln P=\sum_k\operatorname{Re}\alpha_k\,(c'_k-c_k)$ (`LogProjRatio`). The move is
accepted iff $w>r$ with $r=\texttt{genrand\_real2()}\in[0,1)$; a non-finite $w$ is replaced by $-1$ (always rejected). On
acceptance the Pfaffians/inverses, the projection counters, the RBM counters and
$\ln\mathrm{IP}$ are updated; on rejection the occupation arrays are reverted
(`revertEleConfig`) and nothing else changes.

> **Implementation**
> - C: `VMCMakeSample` (acceptance block `w = exp(2.0*(creal(x+logIpNew-logIpOld)))`) — `extern/mVMC-1.3.0/src/mVMC/vmcmake.c:194`
> - Rust: `metropolis_weight` — `crates/mvmc-core/src/sampling/metropolis.rs:41`
> - Rust: `metropolis_decision` — `crates/mvmc-core/src/sampling/metropolis.rs:59`
> - Rust: `attempt_hopping_move` — `crates/mvmc-core/src/sampling/one_move.rs:199`
> - Rust: `attempt_exchange_move` — `crates/mvmc-core/src/sampling/one_move.rs:282`
> - Parity: C forms `x = LogProjRatio(...)`, `x += LogRBMRatio(...)` (complex), then `exp(2.0*(creal(x + logIpNew - logIpOld)))`; Rust evaluates `2.0 * ((proj + rbm.re + ip_new.re) - ip_old.re)`, the same real-part sums in the same order, with `julia_exp::exp` instead of libm `exp`. `-1.0` is substituted for a non-finite weight, and the draw is always consumed (`metropolis_decision` draws before comparing). The libm-versus-Julia `exp` difference is one of the few places where a last-bit difference can flip an acceptance; this is expected and handled by [11.4](11-compatibility.md#114-numerical-comparison-policy).

## 4.5 Pfaffian ratio and inverse updates

Let $W=\texttt{InvM}=X^{-1}$ ([3.2](03-theory-wavefunction.md#32-the-pfaffian-pair-product-part)),
$a$ the index of the moved electron and $u_j=F(r'_a,r_j)$ the new row of the
Slater table evaluated at the *new* occupied spin-orbitals ($r_j$ are the
current positions of all electrons, $r'_a$ the new position of electron $a$).

**One-electron move (hopping).** Replacing row/column $a$ of the skew-symmetric
matrix changes the Pfaffian by

$$
\frac{\mathrm{Pf}\,X'}{\mathrm{Pf}\,X}=-\sum_{j}W_{aj}\,u_j ,
$$

(`CalculateNewPfM2`, one value per sector $q$; the sums over $j\le N_e$ use
spin-orbitals $r_j$, the remaining $r_j+N_s$). If the move is accepted
`updateMAll_child` performs the rank-2 update with $v=Wu$ (`vec1`),
$\mathrm{Pf}\,X\leftarrow-v_a\mathrm{Pf}\,X$, $s_i=-W_{ai}/v_a$ (`vec2`):

$$
W_{ij}\leftarrow W_{ij}+v_is_j-v_js_i,\qquad
W_{ia}\leftarrow W_{ia}-s_i,\qquad W_{aj}\leftarrow W_{aj}+s_j .
$$

**Two-electron move (exchange).** For electrons $a,b$ with rows $u_i=F(r'_a,r_i)$,
$v_i=F(r'_b,r_i)$ (both at their new positions, so $u_b=F(r'_a,r'_b)$ and $v_a=F(r'_b,r'_a)$):

$$
\frac{\mathrm{Pf}\,X'}{\mathrm{Pf}\,X}
= W_{ab}\,v_a + W_{ab}\,(v^{\!T}Wu)+p_aq_b-p_bq_a,
\qquad
p_c=\sum_iW_{ci}u_i,\ \ q_c=\sum_iW_{ci}v_i\ (c=a,b),
$$

(`calculateNewPfMTwo_child_fcmp`; $v^{T}Wu=\sum_iv_i\sum_jW_{ij}u_j$). The accepted-move update
(`updateMAllTwo_child_fcmp`) is the corresponding rank-4 Woodbury-type update
with six scalar coefficients $a,b,c,d,e,f$ built from $p,q$, the $2\times2$
determinant $\det=ad-bc-ef$ and the vectors $s=\det^{-1}W_{a\cdot}$,
$t=\det^{-1}W_{b\cdot}$ (see `update_two_complex`).

The same update machinery is used by the Green-function ratios of
[2.5](02-theory-vmc-hamiltonian.md#25-green-function-ratios) on a *copy* of the
configuration; `calHCA`/`calHCACA` of the Lanczos path temporarily update and then
restore the stored `InvM`/`PfM` (`copyMAll`).

> **Implementation**
> - C: `CalculateNewPfM2` — `extern/mVMC-1.3.0/src/mVMC/pfupdate.c:78`
> - C: `UpdateMAll` — `extern/mVMC-1.3.0/src/mVMC/pfupdate.c:119`
> - C: `updateMAll_child` — `extern/mVMC-1.3.0/src/mVMC/pfupdate.c:143`
> - C: `CalculateNewPfMTwo2_fcmp` — `extern/mVMC-1.3.0/src/mVMC/pfupdate_two_fcmp.c:73`
> - C: `calculateNewPfMTwo_child_fcmp` — `extern/mVMC-1.3.0/src/mVMC/pfupdate_two_fcmp.c:107`
> - C: `UpdateMAllTwo_fcmp` — `extern/mVMC-1.3.0/src/mVMC/pfupdate_two_fcmp.c:189`
> - C: `updateMAllTwo_child_fcmp` — `extern/mVMC-1.3.0/src/mVMC/pfupdate_two_fcmp.c:217`
> - C: `UpdateMAll_fsz` — `extern/mVMC-1.3.0/src/mVMC/pfupdate_fsz.c:114`
> - Rust: `calculate_new_pf_m2_complex_flat` — `crates/mvmc-core/src/sampling/updates.rs:26`
> - Rust: `calculate_new_pf_m2_real_flat` — `crates/mvmc-core/src/sampling/updates.rs:69`
> - Rust: `update_m_all_complex_flat` — `crates/mvmc-core/src/sampling/updates.rs:393`
> - Rust: `update_one_complex` — `crates/mvmc-core/src/sampling/updates.rs:868`
> - Rust: `calculate_new_pf_m_two2_complex_flat` — `crates/mvmc-core/src/sampling/updates.rs:186`
> - Rust: `two_ratio_complex` — `crates/mvmc-core/src/sampling/updates.rs:1132`
> - Rust: `update_m_all_two_complex_flat` — `crates/mvmc-core/src/sampling/updates.rs:548`
> - Rust: `update_two_complex` — `crates/mvmc-core/src/sampling/updates.rs:980`
> - Rust: `update_m_all_two_real_flat` — `crates/mvmc-core/src/sampling/updates.rs:605`
> - Parity: the C `invM` accumulation order inside `updateMAll_child` is `vec1[msi] += -invM_j[msi] * sltE_aj` over $j$ then $i$-major rank-2 update; the Rust kernels keep this loop nest and scalar order so that results agree to round-off. **Known quirk (observed):** in the two-electron update C defines `rsbOld = raOld + t*Nsite` (it uses `raOld`, not `rbOld`) in all four variants (`pfupdate_two_fcmp.c:227`, `pfupdate_two_real.c:227`, and the FSZ files). The Rust *real* update reproduces this on purpose (`update_m_all_two_real_flat`, `updates.rs:565`, `rsb_old = ra_old + ...` at `updates.rs:588`), but the Rust *complex normal* update uses `rb_old` (`updates.rs:537`). See [11.5](11-compatibility.md#115-open-observations).

## 4.6 Parallelism inside the sampler

For `NSplitSize = 1` each MPI rank runs its own chain with RNG seed
`RndSeed + rank` (C: `init_gen_rand(RndSeed + group1)` with
`group1 = rank0/NSplitSize`, `vmcmain.c:257`) and all sector sums are local. With
`NSplitSize > 1` the ranks of one group share one chain: the sector range
`[qpStart, qpEnd)` is divided among them (`SplitLoop`) and $\mathrm{IP}$ is summed with an
`MPI_Allreduce` over the group communicator in `CalculateIP_fcmp`; the other draws are
made identically on every rank of the group. See [8.4](08-running.md#84-mpi-and-grouped-execution).

> **Implementation**
> - C: `SplitLoop` — `extern/mVMC-1.3.0/src/mVMC/splitloop.c:31`
> - C: `CalculateIP_fcmp` (group `MPI_Allreduce`) — `extern/mVMC-1.3.0/src/mVMC/qp.c:110`
> - Rust: `partition_range` — `crates/mvmc-core/src/parallel.rs:88`
> - Rust: `assign_group` — `crates/mvmc-core/src/parallel.rs:67`
> - Rust: `resolve_rnd_seed` — `crates/mvmc-core/src/run.rs:1976`
> - Parity: `partition_range` reproduces `SplitLoop` including the "remainder to the last ranks" rule and the small-work branch. `resolve_rnd_seed` adds the group index to the base seed with wrapping `i64` arithmetic and then requires the result to fit in `u32` (`seeded_rng`, `run.rs:1799`); a negative `RndSeed` uses one clock value read on the output root and broadcast (Julia lifecycle).

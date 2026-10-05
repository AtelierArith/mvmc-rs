# 2. Theory I: variational Monte Carlo and the Hamiltonian

[Contents](README.md) · Previous: [1. Overview](01-overview-install.md) · Next: [3. The variational wave function](03-theory-wavefunction.md)

This chapter fixes the notation, derives the Monte Carlo estimator that every
other chapter uses, lists the Hamiltonian terms that can be defined in Expert
mode, and explains how each term is evaluated as a ratio of wave-function
amplitudes (the *local energy* and the *Green function ratios*).

## 2.1 Notation

| Symbol | Meaning | In the code |
|--------|---------|-------------|
| $N_s$ | number of sites | `Nsite` (`NSite`) |
| $N_e$ | number of electrons *per spin* in the sz-conserved ("normal") path; $N = 2N_e$ is the size of the Pfaffian matrix | `Ne`/`Nelectron` in C input (`NElec`/`Nelec` in Rust input, field `nelec`), `Nsize = 2*Ne` |
| $N_{\rm QP}$ | number of quantum-projection sectors (`NQPFull`) | `NQPFull` |
| $x$ | a real-space configuration | `eleIdx`, `eleCfg`, `eleNum` |
| $\psi(x)=\langle x\vert\psi\rangle$ | amplitude of the trial state | `ip`, `logIp` |
| $\alpha_k$ | the $k$-th variational parameter (complex) | `Para[k]` |

`Ne` is read from `modpara.def` either directly (`Nelec`/`Nelectron`) or derived
from `Ncond`: if `Ncond` is given, $N_e=(N_{\rm locspin}+N_{\rm cond})/2$, and
`Ncond` must be even (`readdef.c`, *CalcNCond*, lines 589-595; Rust
`crates/mvmc-expert-parsers/src/lib.rs:308-313`). Localized spins count as
electrons: a Heisenberg model has `NLocSpin = Nsite`, `Ncond = 0`.

The configuration $x$ is the basis state

$$
|x\rangle=\prod_{n=1}^{N_e}c^\dagger_{r_{n\uparrow}\uparrow}\prod_{n=1}^{N_e}c^\dagger_{r_{n\downarrow}\downarrow}|0\rangle ,
$$

where $r_{n\sigma}$ is the site of the $n$-th electron with spin $\sigma$
([C manual, *Algorithm*](../../extern/mVMC-1.3.0/doc/en/source/algorithm.rst)).
The code stores it as three arrays (C names; Rust uses the same names in
snake case):

- `eleIdx[n + s*Ne]` – site of the $n$-th electron with spin $s$ (0 = up, 1 = down),
- `eleCfg[r + s*Nsite]` – index $n$ of the electron at site $r$ with spin $s$, or $-1$,
- `eleNum[r + s*Nsite]` – occupation (0 or 1).

In the FSZ/general-orbital path the electrons carry an explicit spin label
(`eleSpn`) and $N_\uparrow\neq N_\downarrow$ is allowed.

## 2.2 The variational Monte Carlo estimator

For a trial state $|\psi\rangle$ the expectation value of an operator $A$ is

$$
\langle A\rangle=\frac{\langle\psi|A|\psi\rangle}{\langle\psi|\psi\rangle}
=\sum_x\rho(x)\,\frac{\langle\psi|A|x\rangle}{\langle\psi|x\rangle},
\qquad
\rho(x)=\frac{|\langle x|\psi\rangle|^2}{\langle\psi|\psi\rangle}.
$$

The Markov chain of [chapter 4](04-theory-sampling.md) samples $x\sim\rho(x)$, so
every sample carries the same weight. The C code states this explicitly
(`w = 1.0`, `vmccal.c:153`; the commented-out reweighting formula is unused) and
the Rust port keeps a weight variable equal to 1. With the local estimator

$$
F(x,A)=\frac{\langle\psi|A|x\rangle}{\langle\psi|x\rangle},
$$

the code accumulates over the $N_{\rm smp}$ = `NVMCSample` saved configurations

$$
\langle H\rangle\simeq\frac{1}{W}\sum_x w\,F(x,H),\qquad
\langle H^2\rangle_{\rm est}\equiv\frac{1}{W}\sum_x w\,\overline{F(x,H)}\,F(x,H),
\qquad W=\sum_x w .
$$

Note that the second moment stored as `Etot2` (the column called $\langle H^2\rangle$
in `zvo_out.dat`) is $\langle F^\dagger F\rangle=\langle\lvert E_{\rm loc}\rvert^2\rangle$, **not**
$\langle F(x,H^2)\rangle$. The C manual explains why: the product form is
numerically more stable and gives a non-negative variance for a finite sample
([C manual, *Power Lanczos method*](../../extern/mVMC-1.3.0/doc/en/source/algorithm.rst)).
The relative variance written to `zvo_out.dat` is

$$
\text{variance}=\operatorname{Re}\frac{\langle H^2\rangle_{\rm est}-\langle H\rangle^2}{\langle H\rangle^2}.
$$

Averaging over independent Markov chains (MPI ranks, `NSplitSize` groups) sums
$W$, $\sum wF$, $\sum w|F|^2$ and the SR accumulators before the division by $W$
([chapter 8](08-running.md)).

> **Implementation**
> - C: `VMCMainCal` — `extern/mVMC-1.3.0/src/mVMC/vmccal.c:82`
> - C: `WeightAverageWE` — `extern/mVMC-1.3.0/src/mVMC/average.c:41`
> - C: `outputData` — `extern/mVMC-1.3.0/src/mVMC/vmcmain.c:640`
> - Rust: `accumulate_observables_local` — `crates/mvmc-core/src/run.rs:2716`
> - Rust: `reduce_accumulators` — `crates/mvmc-core/src/run.rs:1638`
> - Rust: `weight_average_we` — `crates/mvmc-core/src/average.rs:18`
> - Rust: `output_data` — `crates/mvmc-core/src/io.rs:93`
> - Parity: `Etot2 += w * conj(e) * e` (`vmccal.c:191`) is `etot2 += w * e.conj() * e` in `run.rs:2771`, the same product order. In `output_data` the relative variance is computed only when $|\langle H\rangle|>10^{-14}$ and is written as `0.0` otherwise (Julia's guard); C divides unconditionally. The complex division in the optimization output uses `julia_complex::divide`, while the PhysCal output (`output_phys_data`, `io.rs:161`) uses the C99-style `c_complex::divide`.
> - Parity: samples with a failed Pfaffian setup or a non-finite energy are skipped in both implementations (C prints a warning and `continue`s; Rust `continue`s).

## 2.3 Hamiltonian terms

Expert mode defines the Hamiltonian as a sum of seven families
([C manual, *Input files for Expert mode*](../../extern/mVMC-1.3.0/doc/en/source/expert.rst), lines 24-67):

$$
\begin{aligned}
\mathcal H_T&=-\sum_{ij}\sum_{\sigma_1\sigma_2}t_{ij\sigma_1\sigma_2}\,c^\dagger_{i\sigma_1}c_{j\sigma_2}, &
\mathcal H_U&=\sum_i U_i\,n_{i\uparrow}n_{i\downarrow},\\
\mathcal H_V&=\sum_{ij}V_{ij}\,n_in_j, &
\mathcal H_H&=-\sum_{ij}J^{\rm Hund}_{ij}\,(n_{i\uparrow}n_{j\uparrow}+n_{i\downarrow}n_{j\downarrow}),\\
\mathcal H_E&=\sum_{ij}J^{\rm Ex}_{ij}\,(c^\dagger_{i\uparrow}c_{j\uparrow}c^\dagger_{j\downarrow}c_{i\downarrow}+c^\dagger_{i\downarrow}c_{j\downarrow}c^\dagger_{j\uparrow}c_{i\uparrow}), &
\mathcal H_P&=\sum_{ij}J^{\rm Pair}_{ij}\,c^\dagger_{i\uparrow}c_{j\uparrow}c^\dagger_{i\downarrow}c_{j\downarrow},\\
\mathcal H_I&=\sum_{ijkl}\sum_{\sigma_1\ldots\sigma_4}I_{ijkl\sigma_1\sigma_2\sigma_3\sigma_4}\,c^\dagger_{i\sigma_1}c_{j\sigma_2}c^\dagger_{k\sigma_3}c_{l\sigma_4},
\end{aligned}
$$

with $n_{i\sigma}=c^\dagger_{i\sigma}c_{i\sigma}$ and $n_i=n_{i\uparrow}+n_{i\downarrow}$.
Each row of a definition file contributes **one** term of the sum; the program
does not symmetrize $(i,j)$ and $(j,i)$ (see `CalculateHamiltonian`, which loops
over the stored rows). The single exception is `PairHop`, whose rows are
expanded to both $(i,j)$ and $(j,i)$ with the same coefficient at read time
(`ReadPairHopValue`, `NPairHopping = 2*NPairHop`).

| Family | Namelist keyword / file | Operator evaluated as | Rust parsed field |
|--------|------------------------|-----------------------|-------------------|
| $\mathcal H_T$ | `Trans` (`trans.def`) | $-t\,G^{(1)}$ | `transfer_terms` |
| $\mathcal H_U$ | `CoulombIntra` | diagonal $U_i n_{i\uparrow}n_{i\downarrow}$ | `coulomb_intra_terms` |
| $\mathcal H_V$ | `CoulombInter` | diagonal $V_{ij}(n_{i\uparrow}+n_{i\downarrow})(n_{j\uparrow}+n_{j\downarrow})$ | `coulomb_inter_terms` |
| $\mathcal H_H$ | `Hund` | diagonal $-J(n_{i\uparrow}n_{j\uparrow}+n_{i\downarrow}n_{j\downarrow})$ | `hund_terms` |
| $\mathcal H_E$ | `Exchange` | $J\,[G^{(2)}_{\uparrow\downarrow}+G^{(2)}_{\downarrow\uparrow}]$ | `exchange_terms` |
| $\mathcal H_P$ | `PairHop` | $J\,G^{(2)}(i,j,i,j;\uparrow,\downarrow)$ | `pair_hop_terms` |
| $\mathcal H_I$ | `InterAll` | $I\,G^{(2)}$ | `inter_all_terms` (owned separately; not documented here beyond its evaluation) |

The diagonal families ($U$, $V$, Hund) only need the occupations; the others
change the configuration and are evaluated as ratios of amplitudes, see
[2.5](#25-green-function-ratios). Spin models use the Bogoliubov representation
$S_z=\tfrac12(n_\uparrow-n_\downarrow)$, $S^+=c^\dagger_\uparrow c_\downarrow$,
$S^-=c^\dagger_\downarrow c_\uparrow$ for $S=1/2$ (only $S=1/2$ is supported),
and sites with localized spins are marked in `locspn.def`; the sampler never
moves their electrons by hopping ([chapter 4](04-theory-sampling.md)).

## 2.4 The local energy

Inserting $H$ into the estimator of [2.2](#22-the-variational-monte-carlo-estimator),

$$
E_{\rm loc}(x)=F(x,H)=\sum_{x'}H_{x'x}\,\overline{\frac{\psi(x')}{\psi(x)}} ,
$$

so that the diagonal terms contribute their value on $x$ and every off-diagonal
term $c^\dagger_ic_j\cdots$ contributes its coefficient times the conjugated
amplitude ratio between the moved configuration and $x$. The code evaluates, in
this order,

$$
E_{\rm loc}=\sum_i U_i n_{i\uparrow}n_{i\downarrow}
+\sum V_{ij}n_in_j-\sum J^{\rm H}_{ij}(\ldots)
-\sum t_{ij}G^{(1)}_{ij\sigma}
+\sum J^{\rm P}_{ij}G^{(2)}_{\rm pair}
+\sum J^{\rm E}_{ij}(G^{(2)}_{\uparrow\downarrow}+G^{(2)}_{\downarrow\uparrow})
+\sum I\,G^{(2)} .
$$

> **Implementation**
> - C: `CalculateHamiltonian` — `extern/mVMC-1.3.0/src/mVMC/calham.c:59`
> - C: `CalculateHamiltonian0` (diagonal part, used by Lanczos) — `extern/mVMC-1.3.0/src/mVMC/calham.c:198`
> - C: `CalculateHamiltonian1` (transfer part, Lanczos) — `extern/mVMC-1.3.0/src/mVMC/calham.c:244`
> - C: `CalculateHamiltonian2` (two-body part, Lanczos) — `extern/mVMC-1.3.0/src/mVMC/calham.c:299`
> - C: `CalculateHamiltonian_fsz` — `extern/mVMC-1.3.0/src/mVMC/calham_fsz.c:49`
> - Rust: `calculate_local_energy_timed` — `crates/mvmc-core/src/observables.rs:2881`
> - Rust: `calculate_local_energy` — `crates/mvmc-core/src/observables.rs:1913`
> - Rust: `calculate_hamiltonian_diagonal` — `crates/mvmc-core/src/observables.rs:551`
> - Rust: `calculate_local_energy_fsz` — `crates/mvmc-core/src/observables.rs:1246`
> - Parity: the accumulation order is CoulombIntra, CoulombInter, Hund (minus sign), Transfer (minus sign), PairHop, Exchange (`tmp = G(0,1) + G(1,0)`, then `J*tmp`), InterAll in file order. The C code runs these loops under OpenMP `reduction(+:e)` with `schedule(dynamic)`, so the C summation order over terms is not fixed across thread counts; Rust sums sequentially in file order. For *real* transfer coefficients Rust accumulates the whole Transfer section into a separate sum (`transfer_energy`) and adds it to the diagonal part afterwards (`observables.rs:2807-2889`); combining the two sums changes the SR gradient at round-off level, so this order must be kept.
> - Parity: for real wave functions C calls `CalculateHamiltonian_real(creal(ip), ...)` (`calham_real.c`), whose accumulator is a `double`; the Rust real path therefore drops the imaginary part of an `InterAll` coefficient (comment at `observables.rs:2984`; C `calham_real.c:52` declares `double myEnergy` and `calham_real.c:136` uses `creal(ParaTransfer[idx])`, so the real path also drops imaginary `Trans` coefficients).
> - Parity: terms whose site indices fall outside $0\ldots N_s-1$ are skipped in Rust (they are rejected earlier by validation for `InterAll`, see [7.5](07-input-files.md#75-supported-and-rejected-inputs)).

## 2.5 Green function ratios

The one- and two-body *local Green functions* are

$$
G^{(1)}_{ij\sigma}(x)=\frac{\langle\psi|c^\dagger_{i\sigma}c_{j\sigma}|x\rangle}{\langle\psi|x\rangle},\qquad
G^{(2)}_{ijkl\sigma\tau}(x)=\frac{\langle\psi|c^\dagger_{i\sigma}c_{j\sigma}c^\dagger_{k\tau}c_{l\tau}|x\rangle}{\langle\psi|x\rangle}.
$$

If $x'_1=c^\dagger_{i\sigma}c_{j\sigma}|x\rangle$ is the configuration obtained by
moving the electron at site $j$ to the empty site $i$, then
$G^{(1)}=\overline{\psi(x'_1)/\psi(x)}$, and the code computes

$$
\frac{\psi(x')}{\psi(x)}=
\underbrace{e^{\Delta\ln P}}_{\text{ProjRatio}}\;
\underbrace{R_{\rm RBM}}_{\text{RBMRatio}}\;
\frac{\mathrm{IP}(x')}{\mathrm{IP}(x)},\qquad
\mathrm{IP}(x)=\sum_q w_q\,\mathrm{Pf}\,X_q(x),
$$

where $\mathrm{IP}$ is the projected Pfaffian inner product defined in
[3.2](03-theory-wavefunction.md#32-the-pfaffian-pair-product-part) and
[3.4](03-theory-wavefunction.md#34-quantum-number-projection). `GreenFunc1` sets
the moved electron, updates the projection counters, evaluates $\mathrm{IP}(x')$
with a one-electron Pfaffian update ([4.5](04-theory-sampling.md#45-pfaffian-ratio-and-inverse-updates))
without modifying the stored inverse, restores the configuration and returns
`conj(z/ip)`.

Selection rules implemented before any amplitude is evaluated
(`GreenFunc1`): $G^{(1)}_{ii\sigma}=n_{i\sigma}$, and $G^{(1)}_{ij\sigma}=0$ if
site $i$ is already occupied or site $j$ is empty ($i\ne j$).

For $G^{(2)}$ the cases in which indices coincide reduce to $G^{(1)}$ or to
occupations (the number operator identities listed in `GreenFunc2`, e.g.
$c^\dagger_ic_jc^\dagger_kc_l$ with $k=l$ is $G^{(1)}_{ij}n_k$ when $\sigma=\tau$
and $n_k$ is occupied). In the generic case $c^\dagger_{k\tau}c_{l\tau}$ is applied
first (moving the $\tau$ electron from $l$ to $k$), then $c^\dagger_{i\sigma}c_{j\sigma}$
(moving the $\sigma$ electron from $j$ to $i$); the Pfaffian of the doubly
moved configuration is obtained with the two-electron update of
[4.5](04-theory-sampling.md#45-pfaffian-ratio-and-inverse-updates).

> **Implementation**
> - C: `GreenFunc1` — `extern/mVMC-1.3.0/src/mVMC/locgrn.c:41`
> - C: `GreenFunc2` — `extern/mVMC-1.3.0/src/mVMC/locgrn.c:86`
> - C: `GreenFunc1_fsz` — `extern/mVMC-1.3.0/src/mVMC/locgrn_fsz.c:29`
> - C: `CalculateIP_fcmp` — `extern/mVMC-1.3.0/src/mVMC/qp.c:110`
> - Rust: `green_func1` — `crates/mvmc-core/src/observables.rs:1463`
> - Rust: `green_func1_impl` — `crates/mvmc-core/src/observables.rs:1742`
> - Rust: `green_func2` — `crates/mvmc-core/src/observables.rs:626`
> - Rust: `green_func2_impl` — `crates/mvmc-core/src/observables.rs:731`
> - Rust: `green_func2_fsz` — `crates/mvmc-core/src/observables/fsz_green.rs:20`
> - Parity: the final amplitude ratio is conjugated (`conj(z/ip)`); in Rust `divide(proj_ratio * new_ip, ip).conj()` selects `c_complex::divide` (C99 semantics) in the C-kernel instantiation (`C_KERNEL = true`) and `julia_complex::divide` otherwise (`observables.rs:1874-1877`). `ProjRatio` uses $\exp$ of the *real part* of the projection exponent only, because C declares Gutzwiller/Jastrow/DH parameters real (`projection.c:41-56`, "we assume gutzwiller and jastrow is real").
> - Parity: C `GreenFunc1` calls `UpdateProjCnt` a second time in the non-RBM branch (`locgrn.c:68`, a redundant duplicate with identical arguments); this does not change the result.
> - Parity: a transfer-coefficient cache (`refresh_transfer_cache`, `observables.rs:1533`) and an optional multi-threaded transfer loop (`MVMC_RS_INNER_THREADS`) exist in the Rust real path; both preserve the result of the sequential term order for each term (the sum order is unchanged).

## 2.6 Quantities that need only the diagonal

`Sz` accumulators (`sztot`, `sztot2`, the last two columns of `zvo_out.dat`) are
computed from `eleNum` by `calculate_sz` (`crates/mvmc-core/src/observables.rs:192`).
For the sz-conserved path they are identically zero.

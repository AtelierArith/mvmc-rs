# 3. Theory II: the variational wave function

[Contents](README.md) · Previous: [2. VMC and the Hamiltonian](02-theory-vmc-hamiltonian.md) · Next: [4. Markov-chain sampling](04-theory-sampling.md)

## 3.1 The trial state

The variational state of mVMC is

$$
|\psi\rangle=\mathcal N_{\rm RBM}\;\mathcal P_G\,\mathcal P_J\,\mathcal P^{(2)}_{d\text{-}h}\,\mathcal P^{(4)}_{d\text{-}h}\;
\mathcal L^{S}\mathcal L^{K}\mathcal L^{P}\;|\phi_{\rm pair}\rangle
$$

([C manual, *Input files for Expert mode*](../../extern/mVMC-1.3.0/doc/en/source/expert.rst), lines 75-110) with

| Factor | Definition | Section |
|--------|------------|---------|
| $\lvert\phi_{\rm pair}\rangle$ | Pfaffian pair product $\big[\sum_{ij\sigma\sigma'}f_{i\sigma j\sigma'}c^\dagger_{i\sigma}c^\dagger_{j\sigma'}\big]^{N/2}\lvert0\rangle$ | [3.2](#32-the-pfaffian-pair-product-part) |
| $\mathcal P_G$ | $\exp\big[\sum_i g_i\,n_{i\uparrow}n_{i\downarrow}\big]$ | [3.5](#35-gutzwiller-jastrow-and-doublon-holon-factors) |
| $\mathcal P_J$ | $\exp\big[\tfrac12\sum_{i\ne j}v_{ij}(n_i-1)(n_j-1)\big]$ | [3.5](#35-gutzwiller-jastrow-and-doublon-holon-factors) |
| $\mathcal P^{(2)}_{d\text{-}h}$, $\mathcal P^{(4)}_{d\text{-}h}$ | 2- and 4-site doublon-holon correlation factors | [3.5](#35-gutzwiller-jastrow-and-doublon-holon-factors) |
| $\mathcal N_{\rm RBM}$ | restricted Boltzmann machine factor | [3.6](#36-restricted-boltzmann-machine-factor) |
| $\mathcal L^S,\mathcal L^K,\mathcal L^P$ | spin, momentum and lattice-symmetry projections | [3.4](#34-quantum-number-projection) |

In real space, for a configuration $x$ the code evaluates the **amplitude**

$$
\boxed{\;\psi(x)=\langle x|\psi\rangle
= \exp\Big[\sum_{k\in{\rm proj}}\alpha_k\,c_k(x)\Big]\;
\exp\big[\ln\mathcal N_{\rm RBM}(x)\big]\;
\underbrace{\sum_{q=1}^{N_{\rm QP}}w_q\,\mathrm{Pf}\,X_q(x)}_{\mathrm{IP}(x)}\;}
$$

where $c_k(x)$ are the integer *projection counters* ($\texttt{eleProjCnt}$), $X_q(x)$
is the skew-symmetric matrix of [3.2](#32-the-pfaffian-pair-product-part) in the
$q$-th quantum-projection sector and $w_q$ is the sector weight of
[3.4](#34-quantum-number-projection). Only $\ln|\psi|$ differences enter the
Metropolis test, so the code keeps $\ln\mathrm{IP}$ (`logIp`) and the counters
separately and never forms $\psi$ itself.

### Parameter layout

All variational parameters live in one complex vector (`Para` in C, packed by
`pack_parameters` in Rust) in this order:

$$
\texttt{Para}=\big[\underbrace{g_i}_{N_G}\;\underbrace{v_{ij}}_{N_J}\;\underbrace{\alpha^{(2)}}_{6N_{\rm DH2}}\;\underbrace{\alpha^{(4)}}_{10N_{\rm DH4}}\;\big|\;\text{RBM}\;\big|\;\underbrace{f}_{N_{\rm Slater}}\;\big|\;\underbrace{p^{\rm opt}}_{N_{\rm OptTrans}}\big].
$$

The first group has $N_{\rm proj}=N_G+N_J+6N_{\rm DH2}+10N_{\rm DH4}$ entries
(`NProj`). The offsets in the C reader are visible in `ReadInputParameters`
(DH2 starts at `NGutzwillerIdx + NJastrowIdx`, DH4 at
`NGutzwillerIdx + NJastrowIdx + 2*3*NDoublonHolon2siteIdx`, `readdef.c`); the
Rust layout is `ProjectionLayout` (`projection_layout`,
`crates/mvmc-expert-parsers/src/types.rs:1311`) with `dh2_offset`,
`dh4_offset`, `n_proj`. Every parameter has two *optimization flags* (real part,
imaginary part) stored consecutively (`OptFlag[2k]`, `OptFlag[2k+1]`). A
parameter component is optimized only if its flag equals 1.

The SR vector $O$ ([chapter 5](05-theory-sr.md)) uses the same order with two
slots (real, imaginary derivative) per parameter and an extra leading pair for
the identity: $[1,\ \text{proj}\ (2N_{\rm proj}),\ \text{RBM}\ (2N_{\rm RBM}),\ \text{Slater}\ (2N_{\rm Slater}),\ \text{OptTrans}]$
(`VMCMainCal`, `vmccal.c:197-215`).

> **Implementation**
> - C: `ReadInputParameters` — `extern/mVMC-1.3.0/src/mVMC/readdef.c:1183`
> - Rust: `projection_layout` — `crates/mvmc-expert-parsers/src/types.rs:1317`
> - Rust: `accumulate_observables_local` — `crates/mvmc-core/src/run.rs:2896`
> - Parity: Rust reserves the *declared* widths for each block even when the definition file lists fewer rows (the "sparse projection" rule, `ProjectionLayout` docs). The FSZ main-calculation places the Slater derivatives immediately after the projection block (no RBM slot), the normal path reserves all RBM slots first (`run.rs:2887-2900`).

## 3.2 The Pfaffian pair-product part

The one-body part is a generalized (BCS-like) pair product,

$$
|\phi_{\rm pair}\rangle=\Big(\sum_{I,J=1}^{2N_s}F_{IJ}\,c^\dagger_Ic^\dagger_J\Big)^{N/2}|0\rangle ,
\qquad F_{IJ}=-F_{JI},
$$

where $I=(i,\sigma)$ is a spin-orbital. For the usual spin-singlet-pairing case
only $f_{ij}\equiv F_{(i\uparrow)(j\downarrow)}$ is nonzero and
$|\phi_{\rm pair}\rangle=(\sum_{ij}f_{ij}c^\dagger_{i\uparrow}c^\dagger_{j\downarrow})^{N_e}|0\rangle$.
A single Slater determinant is the special case in which the $N_e$ nonzero
singular values of $F$ equal 1 ([C manual, *Properties of the Pfaffian-Slater determinant*](../../extern/mVMC-1.3.0/doc/en/source/algorithm.rst)):

$$
f_{ij}=\sum_{n=1}^{N_e}\Phi_{in\uparrow}\Phi_{jn\downarrow},\qquad
F_{IJ}=\sum_{n=1}^{N/2}\big(\Phi_{I,2n-1}\Phi_{J,2n}-\Phi_{J,2n-1}\Phi_{I,2n}\big).
$$

Its overlap with a configuration is a Pfaffian. Number the $N=2N_e$ electrons
of $x$ with $m=1\ldots N$ (up electrons first in the normal path) and let
$r_m$ be the spin-orbital of electron $m$. Then

$$
\langle x|\phi_{\rm pair}\rangle=\mathrm{Pf}\,X(x),\qquad X_{mn}(x)=F_{r_mr_n},
$$

a skew-symmetric $N\times N$ matrix. The code builds, once per SR step and for each
projection sector $q$ ([3.4](#34-quantum-number-projection)), a *Slater-element table*
$\texttt{SlaterElm}[q][I][J]$ of size $2N_s\times2N_s$ (the translated and spin-rotated
$F^{(q)}_{IJ}$ derived from the parameters $f_{ij}$) and gathers the rows and
columns of the occupied spin-orbitals into $X_q$ at every Pfaffian
recomputation. C stores the gathered block as
$\texttt{invM}[m\cdot N+n]=-\texttt{sltE}[r_m][r_n]$; read as a column-major
matrix this buffer is exactly $X$, because the table is skew-symmetric
($X_{nm}=F_{r_nr_m}=-F_{r_mr_n}$), and that is what the factorization routine receives.

The Pfaffian and the inverse are obtained from a skew-symmetric LTL
factorization (`M_ZSKTRF` + `utu2pfa_z` + `utu2inv_z` in C; `zsktf2`,
`utu2pfa_complex`, `utu2inv_complex` in `crates/pfapack`). After the final
`M_ZSCAL(..., -1)` the stored array satisfies, as a row-major matrix,
$\texttt{InvM}_{mn}=(X^{-1})_{mn}$. This is the quantity used by every update
formula in [4.5](04-theory-sampling.md#45-pfaffian-ratio-and-inverse-updates).

> **Implementation**
> - C: `CalculateMAll_fcmp` — `extern/mVMC-1.3.0/src/mVMC/matrix.c:285`
> - C: `calculateMAll_child_fcmp` — `extern/mVMC-1.3.0/src/mVMC/matrix.c:332`
> - C: `CalculateMAll_real` — `extern/mVMC-1.3.0/src/mVMC/matrix.c:516`
> - C: `calculateMAll_child_real` — `extern/mVMC-1.3.0/src/mVMC/matrix.c:557`
> - C: `CalculateMAll_fsz` — `extern/mVMC-1.3.0/src/mVMC/matrix.c:78`
> - Rust: `calc_m_all_complex` — `crates/mvmc-core/src/pfaffian.rs:280`
> - Rust: `calc_m_all_real` — `crates/mvmc-core/src/pfaffian.rs:134`
> - Rust: `calc_m_all_fsz_complex` — `crates/mvmc-core/src/pfaffian.rs:555`
> - Rust: `calc_m_all_fsz_real` — `crates/mvmc-core/src/pfaffian.rs:677`
> - Rust: `calc_m_all_child_complex` — `crates/mvmc-core/src/pfaffian.rs:891`
> - Rust: `zsktf2_c_compat` — `crates/pfapack/src/ltl.rs:54`
> - Rust: `utu2pfa_complex` — `crates/pfapack/src/utu2.rs:57`
> - Rust: `utu2inv_complex` — `crates/pfapack/src/utu2.rs:339`
> - Parity: C fails the sample (`info != 0`) when the factorization reports a zero pivot or the Pfaffian is not finite (`matrix.c:371-373`); Rust returns `CalcMAllError::{ZeroPivot, NonFinitePfaffian, AllZero}`. The ordinary complex optimizer follows Julia's `zsktf2_turbo` operation order and the `c_compat` variant (`calc_m_all_complex_c_compat`, `crates/mvmc-core/src/pfaffian.rs:301`) follows the C PFAPACK kernel; the real path uses BLAS `dger`/`dtrtri`/`dtrmm`. Pfaffian and inverse results are compared with explicit tolerances, not bitwise (see [11.4](11-compatibility.md#114-numerical-comparison-policy)).

### Orbital modes

The set of non-zero $F_{IJ}$ and how they are parametrized depends on which
orbital definition files are present. The parser records them in
`i_flg_orbital_anti_parallel`, `i_flg_orbital_parallel` and
`i_flg_orbital_general` (`crates/mvmc-expert-parsers/src/orbital_mode.rs`):

| Mode | Namelist keywords | Non-zero pair amplitudes | Pfaffian path |
|------|-------------------|--------------------------|---------------|
| normal (sz-conserved) | `Orbital` or `OrbitalAntiParallel` | $f_{ij}$ between $\uparrow$ at $i$ and $\downarrow$ at $j$ | `*_fcmp`/`*_real` kernels, $N_\uparrow=N_\downarrow=N_e$ |
| parallel-spin pairing | `OrbitalAntiParallel` **and** `OrbitalParallel` | additionally $F_{i\sigma j\sigma}$ | treated as *general* (the pair $AP=P=1$ sets `General=1`, `judge_orbital_mode`) |
| general (FSZ) | `OrbitalGeneral` | all $F_{I J}$ with $I,J\in\{1..2N_s\}$ | `*_fsz` kernels with explicit spin label `eleSpn` |

The C reader requires `OrbitalParallel`/`OrbitalGeneral` whenever $2S_z\neq0$,
and forces `NSPGaussLeg = 1` for general orbitals (spin projection is only
implemented for the normal path); Lanczos is rejected for general orbitals
(`readdef.c:597-617`). C `JudgeOrbitalMode` is mirrored by Rust's
`judge_orbital_mode`, which reports (but does not impose) the C admission result.

For FSZ/general orbitals the table is built without spin rotation,
$\texttt{sltE}[I][J]=F_{IJ}-F_{JI}$ (each of the four spin blocks;
`UpdateSlaterElm_fsz`, `slater_fsz.c:32`). Translation signs and the
`OrbitalSgn` matrix are applied exactly as in the normal path.

## 3.3 Real and complex modes

mVMC has separate real and complex kernels (`*_real`, `*_fcmp`). The mode is
**not** inferred from the numerical values: C sets

$$
\texttt{AllComplexFlag}=\texttt{iComplexFlgGutzwiller}+\texttt{iComplexFlgJastrow}+\texttt{iComplexFlgDH2}+\texttt{iComplexFlgDH4}+\texttt{iComplexFlgOrbital},
$$

where each flag is the `ComplexType` header value of the corresponding index
file (`readdef.c:641-642`), and all-zero selects the real kernels. The RBM
files do not contribute to this sum. Rust mirrors this with
`get_all_complex_flag` (`crates/mvmc-core/src/run.rs:1713`) and
`all_complex_flag` (`crates/mvmc-expert-parsers/src/utils/parameter_init.rs:23`), which
take the same header declarations; an imaginary value loaded from an
`In*` file does not change the mode, and a loaded declaration that is modified
without clearing its stored header is an error.

In the real mode only the real part of every parameter is optimized and the SR
matrix is the $N_{\rm para}\times N_{\rm para}$ real matrix; in the complex mode
real and imaginary parts are $2N_{\rm para}$ independent real variables
([5.2](05-theory-sr.md#52-the-sr-equations)).

> **Implementation**
> - C: `ReadInputParameters` (sets `AllComplexFlag`) — `extern/mVMC-1.3.0/src/mVMC/readdef.c:1183`
> - Rust: `get_all_complex_flag` — `crates/mvmc-core/src/run.rs:2018`
> - Rust: `all_complex_flag` — `crates/mvmc-expert-parsers/src/utils/parameter_init.rs:23`
> - Parity: real-mode runs hold `SlaterElm_real`/`InvM_real`/`PfM_real` copies in C; Rust keeps real buffers (`pf_m_real`, `sr_opt_oo_real`, ...) and a complex shadow where the shared code needs it.

## 3.4 Quantum-number projection

The projectors $\mathcal L^S\mathcal L^K\mathcal L^P$ are evaluated by
quadrature/summation over *sectors* $q=(o,K,s)$:

- $s=1\ldots N_{\rm GL}$ with $N_{\rm GL}=\texttt{NSPGaussLeg}$: Gauss–Legendre nodes
  $\beta_s\in[0,\pi]$ and weights $\omega_s$ for the spin projection (rotation about the $y$ axis);
  $S=\texttt{NSPStot}$ selects the Legendre polynomial $P_S$.
- $K=1\ldots\lvert N_{\rm MP}\rvert$ with $N_{\rm MP}=\texttt{NMPTrans}$: lattice translations (or any site
  permutation group) given in `TransSym`/`qptransidx.def` with weights $p_K$
  (complex phases select the momentum). A negative `NMPTrans` turns on
  anti-periodic boundary conditions (sign factors on wrapped sites). $N_{\rm MP}=1$
  means "no translation projection".
- $o=1\ldots N_{\rm opt}$ with $N_{\rm opt}=\texttt{NQPOptTrans}$: *optimized* translations
  (`OptTrans`, enabled by the `-o` flag, whose weights $p^{\rm opt}_o$ are
  variational parameters; usually $N_{\rm opt}=1$).

The sector index is $q=o\,N_{\rm fix}+K\,N_{\rm GL}+s$ (0-based, $N_{\rm fix}=N_{\rm GL}\lvert N_{\rm MP}\rvert$),
so $N_{\rm QP}=N_{\rm GL}\,\lvert N_{\rm MP}\rvert\,N_{\rm opt}$. The weights are

$$
w_q=p^{\rm opt}_o\;p_K\;\tfrac12\sin\beta_s\;\omega_s\;P_S(\cos\beta_s)
\qquad(N_{\rm GL}>1),\qquad
w_q=p^{\rm opt}_o\,p_K\quad(N_{\rm GL}=1,\ \beta=0).
$$

The Slater-element table of sector $q$ uses the translated, signed pair
amplitudes and the spin rotation by $\beta_s$. With
$f^{(q)}_{ij}=\mathrm{sgn}_i\,\mathrm{sgn}_j\,f_{\,T_q(i)\,T_q(j)}$ (site map
$T_q$ and sign from `QPTrans`, `QPOptTrans` and `OrbitalSgn`) and
$c=\cos\tfrac{\beta_s}2$, $s=\sin\tfrac{\beta_s}2$:

$$
\begin{aligned}
F^{(q)}(i\uparrow,j\uparrow)&=-(f_{ij}-f_{ji})\,cs, &
F^{(q)}(i\uparrow,j\downarrow)&=f_{ij}c^2+f_{ji}s^2,\\
F^{(q)}(i\downarrow,j\uparrow)&=-f_{ij}s^2-f_{ji}c^2, &
F^{(q)}(i\downarrow,j\downarrow)&=(f_{ij}-f_{ji})\,cs .
\end{aligned}
$$

At $\beta=0$ ($N_{\rm GL}=1$) only $F(i\uparrow,j\downarrow)=f_{ij}$ and
$F(i\downarrow,j\uparrow)=-f_{ji}$ survive. The projected inner product that
enters every amplitude and every Metropolis ratio is

$$
\mathrm{IP}(x)=\sum_{q}w_q\,\mathrm{Pf}\,X_q(x),\qquad
\ln\mathrm{IP}(x)\ \text{is stored as}\ \texttt{logIp}.
$$

> **Implementation**
> - C: `InitQPWeight` — `extern/mVMC-1.3.0/src/mVMC/qp.c:38`
> - C: `UpdateQPWeight` — `extern/mVMC-1.3.0/src/mVMC/qp.c:129`
> - C: `GaussLeg` — `extern/mVMC-1.3.0/src/mVMC/gauleg.c:32`
> - C: `LegendrePoly` — `extern/mVMC-1.3.0/src/mVMC/legendrepoly.c:32`
> - C: `UpdateSlaterElm_fcmp` — `extern/mVMC-1.3.0/src/mVMC/slater.c:37`
> - C: `UpdateSlaterElm_fsz` — `extern/mVMC-1.3.0/src/mVMC/slater_fsz.c:32`
> - C: `CalculateIP_fcmp` — `extern/mVMC-1.3.0/src/mVMC/qp.c:110`
> - C: `CalculateLogIP_fcmp` — `extern/mVMC-1.3.0/src/mVMC/qp.c:90`
> - Rust: `init_qp_weight` — `crates/mvmc-core/src/qp.rs:12`
> - Rust: `init_qp_weight_inplace` — `crates/mvmc-expert-parsers/src/utils/qp_weight.rs:86`
> - Rust: `update_qp_weight` — `crates/mvmc-expert-parsers/src/utils/qp_weight.rs:150`
> - Rust: `gauss_legendre` — `crates/mvmc-expert-parsers/src/utils/qp_weight.rs:24`
> - Rust: `legendre_poly` — `crates/mvmc-expert-parsers/src/utils/qp_weight.rs:65`
> - Rust: `update_slater_elm` — `crates/mvmc-core/src/slater_update.rs:17`
> - Rust: `update_slater_elm_fsz` — `crates/mvmc-core/src/slater_update.rs:128`
> - Rust: `calculate_ip_complex` — `crates/mvmc-core/src/observables.rs:148`
> - Rust: `calculate_log_ip_complex` — `crates/mvmc-core/src/observables.rs:179`
> - Rust: `translated_site` — `crates/mvmc-core/src/qp.rs:33`
> - Parity: C evaluates `w = 0.5*sin(beta[i])*weight[i]*LegendrePoly(cos(beta[i]), NSPStot)` and `QPFixWeight[i + j*NSPGaussLeg] = w*ParaQPTrans[j]` (`qp.c:73-77`); `QPFullWeight = OptTrans[i]*QPFixWeight` when `FlagOptTrans > 0` (`qp.c:129-146`). The sum `ip += QPFullWeight[q]*pfM[q]` is accumulated in sector order in `CalculateIP_fcmp`; `CalculateLogIP_fcmp` returns `clog(ip)`. Rust follows the same order; the Slater table is rebuilt from the *declared* coefficient without Julia's historical $10^{-14}$ amplitude cutoff (`slater_update.rs:46`). The translation sign (anti-periodic mode) is applied only when `NMPTrans < 0` (`qp.rs:67`).
> - Parity: `OptTrans` has two flag layouts; the C driver flag `-o` selects "consecutive" flag writes (see `c_opt_trans_flags`, `crates/mvmc-core/src/sr.rs:23`). C's `calculateOptTransDiff` (`vmccal.c:639`) writes the derivative at consecutive complex indices ("this part will not be used" in the C comment), while Rust's `opt_trans_diff` (`observables.rs:957`) writes (value, i·value) pairs like Julia. OptTrans is covered by fixtures in `tests/fixtures/opttrans`; this manual does not map it further **(unverified)**.

## 3.5 Gutzwiller, Jastrow and doublon-holon factors

All four factors are exponentials of *integer counters* times (real) coefficients,

$$
\ln P(x)=\sum_{k=1}^{N_{\rm proj}}\operatorname{Re}\alpha_k\;c_k(x).
$$

C takes only $\operatorname{Re}\,\texttt{Proj}[k]$ (`LogProjVal`, `LogProjRatio`: "we assume gutzwiller and jastrow is real"). The counters are computed by
`MakeProjCnt` and updated incrementally when an electron hops by `UpdateProjCnt`:

- **Gutzwiller** (index map $\texttt{GutzwillerIdx}[i]$): $c_{\rm idx(i)}\mathrel{+}=n_{i\uparrow}n_{i\downarrow}$, i.e. $\mathcal P_G=\exp[\sum_ig_{{\rm idx}(i)}n_{i\uparrow}n_{i\downarrow}]$.
- **Jastrow** (index map $\texttt{JastrowIdx}[i][j]$): for $i<j$, $c_{{\rm idx}(i,j)}\mathrel{+}=(n_i-1)(n_j-1)$, i.e. $\mathcal P_J=\exp[\sum_{i<j}v_{{\rm idx}(i,j)}(n_i-1)(n_j-1)]$ with $n_i=n_{i\uparrow}+n_{i\downarrow}$. This equals the manual's $\tfrac12\sum_{i\ne j}$ when the index map is symmetric. Sites with $n_i=1$ contribute nothing.
- **2-site doublon–holon** (`DH2`, index table $\texttt{DoublonHolon2siteIdx}[t][2i..2i+1]$ = the two partner sites $r_0,r_1$ of site $i$ in pattern $t$): for every site with $n_i\ne1$ let $\xi=n_i/2\in\{0\ (\text{holon}),1\ (\text{doublon})\}$ and let $m$ be the number of *opposite* defects among the partners (doublons if site $i$ is a holon, holons if it is a doublon), $m=0,1,2$. Then $c_{\,{\rm offset}+t+(\xi+2m)N_{\rm DH2}}\mathrel{+}=1$, giving $2\times3=6$ coefficients per pattern.
- **4-site doublon–holon** (`DH4`): same with four partner sites, $m=0\ldots4$, $2\times5=10$ coefficients per pattern.

> **Implementation**
> - C: `MakeProjCnt` — `extern/mVMC-1.3.0/src/mVMC/projection.c:58`
> - C: `UpdateProjCnt` — `extern/mVMC-1.3.0/src/mVMC/projection.c:155`
> - C: `LogProjVal` — `extern/mVMC-1.3.0/src/mVMC/projection.c:32`
> - C: `LogProjRatio` — `extern/mVMC-1.3.0/src/mVMC/projection.c:41`
> - C: `ProjRatio` — `extern/mVMC-1.3.0/src/mVMC/projection.c:50`
> - Rust: `make_proj_cnt` — `crates/mvmc-core/src/sampling/projection.rs:179`
> - Rust: `update_proj_cnt` — `crates/mvmc-core/src/sampling/projection.rs:278`
> - Rust: `log_proj_val` — `crates/mvmc-core/src/sampling/projection.rs:391`
> - Rust: `log_proj_ratio` — `crates/mvmc-core/src/sampling/projection.rs:401`
> - Rust: `recompute_dh_counts` — `crates/mvmc-core/src/sampling/projection.rs:20`
> - Parity: `LogProjRatio` accumulates `z += creal(Proj[idx]) * (double)(projCntNew[idx]-projCntOld[idx])` over $k=0..N_{\rm proj}-1$ in index order; the Rust `log_proj_ratio` sums in the same order and precision (tests `log_proj_val_and_ratio_align`, `update_proj_cnt_matches_make_proj_cnt_after_hop`). `ProjRatio` is $\exp$ of the sum: C uses libm `exp`, Rust uses `julia_exp::exp` (a port of Julia's `exp`), a documented source of last-bit differences that can flip a Metropolis decision ([11.3](11-compatibility.md#113-known-differences-from-the-c-reference)).

## 3.6 Restricted Boltzmann machine factor

The RBM factor multiplies the amplitude by

$$
\mathcal N_{\rm RBM}(x)=\exp\Big[\sum_{p}a_p\,m_p(x)\Big]\;\prod_{h=1}^{N_h}\cosh\theta_h(x),\qquad
\theta_h(x)=b_h+\sum_{p}W_{ph}\,m_p(x),
$$

with visible features $m_p$ of three kinds (the manual's single "General RBM" is the third):

| Kind | visible features $m_p$ | hidden neurons |
|------|------------------------|----------------|
| charge | $n_i-1$ (sites $i=0..N_s-1$) | `NneuronCharge` |
| spin | $n_{i\uparrow}-n_{i\downarrow}$ | `NneuronSpin` |
| general | $2n_I-1$ over the $2N_s$ spin-orbitals $I$ | `NneuronGeneral` |

Parameters are grouped in **nine** index blocks — `{Charge,Spin,General}RBM_{PhysLayer, HiddenLayer, PhysHidden}` —
holding $a$ (physical-layer bias), $b$ (hidden-layer bias) and $W$ (couplings);
the order in `Para` is: all `PhysLayer` blocks (charge, spin, general), then all `HiddenLayer`
blocks, then all `PhysHidden` blocks. The code keeps the
vector $\texttt{rbmCnt}=[\,\sum_pa\text{-weighted counts}\ (N^{\rm phys}_{\rm RBM}),\ \theta_h\ (N_h)\,]$ and
updates it incrementally on a hop (`UpdateRBMCnt`). The log amplitude is

$$
\ln\mathcal N_{\rm RBM}=\sum_{p}\texttt{RBM}[p]\,\texttt{rbmCnt}[p]+\sum_{h}\ln\cosh\texttt{rbmCnt}[N^{\rm phys}+h].
$$

Only the **real part** of $\Delta\ln\mathcal N_{\rm RBM}$ enters the Metropolis
exponent ([4.4](04-theory-sampling.md#44-acceptance-test)); for Green functions
the full complex ratio `RBMRatio` multiplies the amplitude ratio. `NBlockSize_RBMRatio`
(default 200) is the C block size for the vectorized ratio evaluation.

> **Implementation**
> - C: `MakeRBMCnt` — `extern/mVMC-1.3.0/src/mVMC/rbm.c:187`
> - C: `UpdateRBMCnt` — `extern/mVMC-1.3.0/src/mVMC/rbm.c:386`
> - C: `LogRBMRatio` — `extern/mVMC-1.3.0/src/mVMC/rbm.c:125`
> - C: `RBMRatio` — `extern/mVMC-1.3.0/src/mVMC/rbm.c:64`
> - C: `WeightRBM` — `extern/mVMC-1.3.0/src/mVMC/rbm.c:30`
> - Rust: `make_rbm_cnt` — `crates/mvmc-core/src/sampling/rbm.rs:160`
> - Rust: `update_rbm_cnt_hopping` — `crates/mvmc-core/src/sampling/rbm.rs:311`
> - Rust: `log_rbm_ratio` — `crates/mvmc-core/src/sampling/rbm.rs:460`
> - Rust: `log_rbm_val` — `crates/mvmc-core/src/sampling/rbm.rs:553`
> - Rust: `log_cosh_stable` — `crates/mvmc-core/src/sampling/rbm.rs:454`
> - Parity: the C code evaluates `clog(ccosh(theta))` and `cexp` per hidden neuron (`rbm.c:30-60`); Rust uses a numerically stable `log_cosh_stable`, so large $|\operatorname{Re}\theta|$ can differ from `clog(ccosh)` at round-off level. The Metropolis exponent adds the RBM term in the left-associative order `(proj + rbm.re + ip_new.re) - ip_old.re` (`metropolis.rs:41`, test `rbm_acceptance_preserves_julia_left_associative_log_additions`). The RBM block parsers reject "archived sparse" RBM definitions before output (`crates/mvmc-core/src/run.rs:2074` test `public_runner_rejects_archived_sparse_rbm_definitions_before_output`).

## 3.7 Initial values and synchronization

**Initialization** (`InitParameter`, run once on every rank with the *same* RNG
state so that the draw order is identical everywhere): all projection
parameters start at 0; RBM and Slater coefficients are random **only if their
flag is positive**, otherwise 0 and no draw is consumed:

| Block | Real mode | Complex mode | Draws per parameter |
|-------|-----------|--------------|---------------------|
| RBM | $0.01\,(r-\tfrac12)/N_{\rm neuron}$ | $10^{-2}\,r_1\,e^{2\pi i r_2}$ | 1 / 2 |
| Slater | $2(r-\tfrac12)$ | $[2(r_1-\tfrac12)+2i(r_2-\tfrac12)]/\sqrt2$ | 1 / 2 |
| OptTrans | `ParaQPOptTrans` | same | 0 |

with $r$ drawn by `genrand_real2`. RBM draws precede Slater draws. Afterwards the
optional *initial parameter file* (second positional argument of the C
driver, `--initial-def` in Rust) and the `In*` definition files overwrite
values, in that order.

**Synchronization** (`SyncModifiedParameter`, after initialization and after
every SR update):

1. Broadcast `Para` from the root (MPI).
2. Doublon–holon shift: if all DH2 (resp. DH4) coefficients are optimized,
   subtract from each group of three (resp. five) bins their mean and add the
   removed amount to the Gutzwiller parameters (`shiftDH2`, `shiftDH4`).
3. Gutzwiller–Jastrow shift: if all Gutzwiller *and* Jastrow coefficients are
   optimized, subtract the common mean of all $N_G+N_J$ values (`shiftGJ`).
   These shifts do not change $|\psi|^2$ up to normalization; the flags are set
   once by `SetFlagShift` (`parameter.c:255`, called at `readdef.c:1170`).
4. Rescale the Slater parameters so that $\max_k|f_k|=4$ (`D_AmpMax`).
5. If OptTrans is active, rescale the OptTrans weights to unit maximum modulus.

> **Implementation**
> - C: `InitParameter` — `extern/mVMC-1.3.0/src/mVMC/parameter.c:35`
> - C: `ReadInitParameter` — `extern/mVMC-1.3.0/src/mVMC/parameter.c:95`
> - C: `SyncModifiedParameter` — `extern/mVMC-1.3.0/src/mVMC/parameter.c:134`
> - C: `shiftGJ` — `extern/mVMC-1.3.0/src/mVMC/parameter.c:181`
> - C: `shiftDH2` — `extern/mVMC-1.3.0/src/mVMC/parameter.c:202`
> - C: `shiftDH4` — `extern/mVMC-1.3.0/src/mVMC/parameter.c:228`
> - C: `D_AmpMax` (4.0) — `extern/mVMC-1.3.0/src/mVMC/parameter.c:32`
> - Rust: `init_parameter` — `crates/mvmc-expert-parsers/src/utils/parameter_init.rs:63`
> - Rust: `sync_modified_parameter` — `crates/mvmc-expert-parsers/src/utils/parameter_init.rs:167`
> - Rust: `sync_modified_parameter` (reducer-aware wrapper) — `crates/mvmc-core/src/sync.rs:18`
> - Rust: `sync_modified_parameter_local` — `crates/mvmc-core/src/sync.rs:87`
> - Rust: `read_initial_def` — `crates/mvmc-core/src/initial_params.rs:117`
> - Rust: `read_opt_para_file` — `crates/mvmc-core/src/initial_params.rs:141`
> - Parity: the RNG **draw count and order** (RBM block first in canonical section/index order, then Slater; real: one `genrand_real2` per active parameter, complex: radius then phase / real then imaginary) must be preserved; this is exact, not toleranced ([11.4](11-compatibility.md#114-numerical-comparison-policy)). Complex Slater values are divided by `sqrt(2.0)` after both draws. The Rust complex RBM phase uses Julia-compatible `sin`/`cos` (`julia_trig`). Rust's Slater amplitude cap uses `julia_hypot::hypot` for $|f|$ (`parameter_init.rs:252`) where C uses `cabs`, and OptTrans amplitudes in `sync_modified_parameter_local` also use `hypot` (`sync.rs:91`). C normalizes by `1.0/xmax` over `cabs`; the shifted-DH accumulation order of DH4 is summed in bin order then divided by 5 (`parameter_init.rs:197-207`).
> - Parity: the correlation shifts in `sync_modified_parameter_local(data, shift_correlations)` can be switched off by the caller (Julia lifecycle); the CLI always passes the default.

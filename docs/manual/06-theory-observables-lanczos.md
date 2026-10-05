# 6. Theory V: physical quantities and the Lanczos correction

[Contents](README.md) · Previous: [5. Stochastic reconfiguration](05-theory-sr.md) · Next: [7. Input files](07-input-files.md)

In fixed-parameter mode (`NVMCCalMode = 1`, *PhysCal*) the program reads optimized
parameters (normally `zqp_opt.dat`, see [8.2](08-running.md#82-physical-quantities-with-fixed-parameters)),
samples configurations as in [chapter 4](04-theory-sampling.md) and measures the energy and the
correlation functions requested in `greenone.def`, `greentwo.def` and `greentwoex.def`. This
chapter gives the estimators, the weighted averaging, and the single-step Lanczos
correction (`NLanczosMode = 1, 2`).

## 6.1 Green functions

For each requested index set the *local* value on a sample $x$ is a Green-function ratio
([2.5](02-theory-vmc-hamiltonian.md#25-green-function-ratios)):

| Namelist keyword | File | Observable | Local value | Output |
|------------------|------|------------|-------------|--------|
| `OneBodyG` | `greenone.def` | $\langle c^\dagger_{i\sigma_1}c_{j\sigma_2}\rangle$ | $G^{(1)}$ via `GreenFunc1` | `zvo_cisajs_NNN.dat` |
| `TwoBodyG` | `greentwo.def` | $\langle c^\dagger_{i\sigma_1}c_{j\sigma_2}c^\dagger_{k\sigma_3}c_{l\sigma_4}\rangle$ | $G^{(2)}$ via `GreenFunc2` | `zvo_cisajscktalt_NNN.dat` |
| `TwoBodyGEx` | `greentwoex.def` | factored product of two one-body entries | $G^{(1)}_a\,\overline{G^{(1)}_b}$ | `zvo_cisajscktaltex_NNN.dat` |

The measured value is the weighted average over all saved samples of all (merged) chains,

$$
\langle A\rangle=\frac{1}{W}\sum_xw\,A_{\rm loc}(x),\qquad
\langle G_{\rm ex}\rangle=\frac{1}{W}\sum_xw\,G^{(1)}_a(x)\,\overline{G^{(1)}_b(x)}
$$

(with $w=1$). The factored (`TwoBodyGEx`) quantity is the product-form estimator
$\langle F^\dagger(x,A)F(x,B)\rangle$ of the C manual ([*Power Lanczos method*](../../extern/mVMC-1.3.0/doc/en/source/algorithm.rst)):
it is built from the **one-body** local values of the *same* sample (each `TwoBodyGEx` row stores the two positions in the one-body list: `CisAjsCktAltIdx[idx][0..1]` in C and `green_two_ex_indices` in Rust), so it needs no two-electron
amplitude. All local values are first stored per
sample (`LocalCisAjs`, `LocalCisAjsCktAltDC`) and then accumulated.

PhysCal repeats `NDataQtySmp` times: sample (the first call includes the `NVMCWarmUp` burn-in, later calls continue from the previous configuration, [4.1](04-theory-sampling.md#41-the-sampler-loop)), measure, average over ranks,
and write one numbered set of output files (`NDataIdxStart`, `NDataIdxStart+1`, ...).

> **Implementation**
> - C: `VMCPhysCal` — `extern/mVMC-1.3.0/src/mVMC/vmcmain.c:531`
> - C: `CalculateGreenFunc` — `extern/mVMC-1.3.0/src/mVMC/calgrn.c:32`
> - C: `WeightAverageGreenFunc` — `extern/mVMC-1.3.0/src/mVMC/average.c:209`
> - C: `GreenFunc1` — `extern/mVMC-1.3.0/src/mVMC/locgrn.c:41`
> - C: `GreenFunc2` — `extern/mVMC-1.3.0/src/mVMC/locgrn.c:86`
> - C: `CalculateGreenFunc_fsz` — `extern/mVMC-1.3.0/src/mVMC/calgrn_fsz.c:33`
> - Rust: `vmc_phys_cal_in_place_timed` — `crates/mvmc-core/src/run.rs:775`
> - Rust: `prepare_phys_cal_from_namelist` — `crates/mvmc-core/src/run.rs:512`
> - Rust: `ordinary_green_values` — `crates/mvmc-core/src/observables/green_measurements.rs:232`
> - Rust: `accumulate_two_body_gex_sample` — `crates/mvmc-core/src/observables.rs:109`
> - Rust: `normalize_physcal_green` — `crates/mvmc-core/src/run.rs:249`
> - Rust: `calculate_green_func_fsz` — `crates/mvmc-core/src/observables/fsz_measurements.rs:17`
> - Rust: `output_phys_data` — `crates/mvmc-core/src/io.rs:161`
> - Parity: C accumulates `PhysCisAjsCktAlt[idx] += w*LocalCisAjs[idx0]*conj(LocalCisAjs[idx1])` (`calgrn.c:110`); Rust `accumulate_two_body_gex_sample` forms `weight * one_body[first] * one_body[second].conj()` in the same order. The normalization multiplies by a precomputed reciprocal of $W$ (`const double complex invW = 1.0/Wc`, `average.c:278`, applied after `SafeMpiReduce_fcmp` on rank 0, `average.c:287-290`), and Rust uses the C99-style `c_complex::divide(1, wc)` and multiplies each value (`run.rs:249-278`), not a per-element division. In C the averaged Green functions exist only on the root rank (`weightAverageReduce` reduces to rank 0) and only the root writes the files. Per-sample results go to the *same* numbered file set; files are re-created (`"w"`) for each `NDataIdxStart + sample`.

## 6.2 The single-step Lanczos wave function

The power-Lanczos state $|\phi\rangle=(1+\alpha H)|\psi\rangle$ improves $|\psi\rangle$ with one
Hamiltonian application. All required moments are evaluated *on the original*
$\rho(x)$ with local operators:

$$
h_1=\langle F^\dagger(H)\rangle,\quad
h_{2(11)}=\langle F^\dagger(H)F(H)\rangle,\quad
h_{2(20)}=\langle F^\dagger(H^2)\rangle,\quad
h_{3(12)}=\langle F^\dagger(H)F(H^2)\rangle,\quad
h_4=\langle F^\dagger(H^2)F(H^2)\rangle .
$$

Because $\langle\phi|\phi\rangle=1+2\alpha h_1+\alpha^2h_{2(11)}$, the Lanczos energy is

$$
E_{\rm LS}(\alpha)=\frac{h_1+\alpha\,(h_{2(11)}+h_{2(20)})+\alpha^2h_{3(12)}}{1+2\alpha h_1+\alpha^2h_{2(11)}}
$$

([C manual, *Determination of $\alpha$*](../../extern/mVMC-1.3.0/doc/en/source/algorithm.rst)). Setting $dE_{\rm LS}/d\alpha=0$ gives a quadratic
equation (derived directly from the quotient rule):

$$
-A\,\alpha^2+2B\,\alpha+C_0=0,\quad
A=h_{2(11)}(h_{2(11)}+h_{2(20)})-2h_1h_{3(12)},\quad
B=h_{3(12)}-h_1h_{2(11)},\quad
C_0=h_{2(11)}+h_{2(20)}-2h_1^2,
$$

$$
\alpha_\pm=\frac{B\pm\sqrt{\Delta}}{A},\qquad
\Delta=B^2+A\,C_0 .
$$

The code evaluates $\Delta$ in the expanded form
$h_{2(11)}(h_{2(11)}+h_{2(20)})^2-h_1^2h_{2(11)}(h_{2(11)}+2h_{2(20)})+4h_1^3h_{3(12)}-2h_1(2h_{2(11)}+h_{2(20)})h_{3(12)}+h_{3(12)}^2$
(equal to $B^2+AC_0$; the expansion was checked by hand while writing this manual). The root with the lower energy is used, and

$$
\frac{\sigma^2_{\rm LS}}{E^2_{\rm LS}}=\frac{\dfrac{h_{2(11)}+2\alpha h_{3(12)}+\alpha^2h_4}{D}-E_{\rm LS}^2}{E_{\rm LS}^2},\qquad D=1+2\alpha h_1+\alpha^2h_{2(11)},
$$

is the relative variance written to `zvo_ls_out_NNN.dat` together with $E_{\rm LS}$ and $\alpha$. The estimate fails (an error is reported)
if $\Delta<0$ or $|D/h_1|<10^{-12}$ for either root.

**How the moments are measured.** With $\nu=2$ (`NLSHam`) the code stores, per sample, the $2\times2$ array of local values

$$
\mathrm{LSLQ}=\begin{pmatrix}1 & F(x,H)\\ F(x,H) & F(x,H^2)\end{pmatrix}
\quad(\text{flat order }[1,\ E_{\rm loc},\ E_{\rm loc},\ F(x,H^2)]),
$$

and accumulates the 16-component tensor

$$
\mathrm{QQQQ}[r_q][r_p][r_i][r_j]\mathrel{+}=w\,\overline{\mathrm{LSLQ}[r_q][r_i]}\;\mathrm{LSLQ}[r_p][r_j]\qquad(\text{flat index }8r_q+4r_p+2r_i+r_j),
$$

whose entries $2,3,10,11,15$ (0-based) are $h_1,\ h_{2(11)},\ h_{2(20)},\ h_{3(12)},\ h_4$ (real parts). The local square
$F(x,H^2)=\langle\psi|HH|x\rangle/\langle\psi|x\rangle$ is built from the diagonal part $V$ of $H$ and
one-step hops (`LSLocalQ`):

$$
F(x,H^2)=E_{\rm loc}(x)\,V(x)\;-\;\sum_{(ij\sigma)}t_{ij}\,\frac{\langle\psi|H\,c^\dagger_{i\sigma}c_{j\sigma}|x\rangle}{\langle\psi|x\rangle}
\;+\;\sum_{\rm 2\text{-}body}(\ldots),
$$

where each $\langle\psi|Hc^\dagger_ic_j|x\rangle/\langle\psi|x\rangle=E_{\rm loc}(x')\,\overline{\psi(x')/\psi(x)}$ is the local energy of the
hopped configuration $x'$ times the conjugated amplitude ratio (`calHCA1`), with a separate branch
(`calHCA2`) when $\psi(x')=0$ so that the zero amplitude cannot be divided out. Exchange, pair-hop and `InterAll` terms use the
two-body counterparts `calHCACA`.

> **Implementation**
> - C: `LSLocalQ` — `extern/mVMC-1.3.0/src/mVMC/lslocgrn.c:76`
> - C: `calculateHK` — `extern/mVMC-1.3.0/src/mVMC/lslocgrn.c:122`
> - C: `calculateHW` — `extern/mVMC-1.3.0/src/mVMC/lslocgrn.c:139`
> - C: `calHCA` — `extern/mVMC-1.3.0/src/mVMC/lslocgrn.c:180`
> - C: `calHCACA` — `extern/mVMC-1.3.0/src/mVMC/lslocgrn.c:449`
> - C: `calculateQQQQ` — `extern/mVMC-1.3.0/src/mVMC/vmccal.c:832`
> - C: `CalculateEne` — `extern/mVMC-1.3.0/src/mVMC/physcal_lanczos.c:270`
> - C: `CalculateEneByAlpha` — `extern/mVMC-1.3.0/src/mVMC/physcal_lanczos.c:298`
> - C: `PhysCalLanczos_fcmp` — `extern/mVMC-1.3.0/src/mVMC/physcal_lanczos.c:149`
> - Rust: `accumulate_lanczos_qqqq` — `crates/mvmc-core/src/lanczos.rs:44`
> - Rust: `lanczos_energy` — `crates/mvmc-core/src/lanczos.rs:73`
> - Rust: `energy_by_alpha` — `crates/mvmc-core/src/lanczos.rs:128`
> - Rust: `calculate_lanczos_h2_transfer` — `crates/mvmc-core/src/observables.rs:2512`
> - Parity: the discriminant expression, the choice `if (ene_p > ene_m) alpha = alpha_m` and the tolerance `fabs(dnorm/H1) < pow(10.0,-12)` are reproduced literally (`lanczos.rs:61-114`, `lanczos.rs:117-143`); `QQQQ` uses the conjugate on the *left* factor for complex runs only (`all_complex`) and no conjugate for real runs, as `calculateQQQQ` vs `calculateQQQQ_real`. The Rust operator application follows Julia's order (PairHop applies its down-spin hop first, Exchange applies each spin channel in the order up-down then down-up; tests `lanczos_pair_hop_applies_down_then_up_like_julia`, `lanczos_exchange_applies_each_spin_channel_in_julia_order`). **Rust restriction:** Lanczos is accumulated only when there is no `InterAll` and the path is not FSZ (`run.rs:2779`), and validation additionally rejects spin-changing `Trans`, `NSplitSize > 1` and general orbitals (`crates/mvmc-core/src/validation.rs:255-291`), a subset of what C supports. **Failure behaviour differs:** when the quadratic has no admissible root C prints an error and writes nothing to the `zvo_ls_*` files; Rust still writes `zvo_ls_qqqq_NNN.dat` and writes `NaN, NaN, NaN` to `zvo_ls_out_NNN.dat` (`io.rs:311-316`).

## 6.3 Physical quantities after the Lanczos step

For a Hermitian operator $A$ the Lanczos expectation value is

$$
A_{\rm LS}(\alpha)=\frac{\langle\phi|A|\phi\rangle}{\langle\phi|\phi\rangle}
=\frac{A_0+\alpha\,(A_{1(10)}+A_{1(01)})+\alpha^2A_{2(11)}}{1+2\alpha h_1+\alpha^2h_{2(11)}},
$$

with $A_0=\langle F(A)\rangle$, $A_{1(10)}=\langle F^\dagger(H)F(A)\rangle$, $A_{1(01)}=\langle F(AH)\rangle$,
$A_{2(11)}=\langle F^\dagger(H)F(AH)\rangle$ ([C manual](../../extern/mVMC-1.3.0/doc/en/source/algorithm.rst)). For
$A=c^\dagger_ic_j$ and $A=c^\dagger_ic_jc^\dagger_kc_l$ the local values
$F(A)$ and $F(HA)=\langle\psi|HA|x\rangle/\langle\psi|x\rangle$ are `LSLCisAjs` and `LSLCisAjsCktAlt`
(`LSLocalCisAjs`, `calHCA`, `calHCACA`); the moment tensors `QCisAjsQ`, `QCisAjsCktAltQ` and
`QCisAjsCktAltQDC` have shape $\nu\times\nu\times N_{\rm phys}$ and are normalized by $W$ together with `QQQQ`.
`CalculatePhysVal_fcmp` assembles, for each quantity $i$ (`NPhys` entries),

$$
A_{\rm LS,i}=\frac{Q[i]+\alpha\,(Q[N_{\rm phys}+i]+Q[\nu N_{\rm phys}+i])+\alpha^2\,Q[\nu N_{\rm phys}+N_{\rm phys}+i]}{\operatorname{Re}\big(1+2\alpha h_1+\alpha^2h_{2(11)}\big)} .
$$

`NLanczosMode = 1` writes only the energy file and the `QQQQ` moments; `NLanczosMode = 2`
additionally writes the Lanczos one-body, direct two-body and factored two-body Green functions
(`zvo_ls_cisajs_NNN.dat`, `zvo_ls_cisajscktalt_NNN.dat`, `zvo_ls_cisajscktaltex_NNN.dat`, see [chapter 9](09-output-files.md)).

> **Implementation**
> - C: `CalculatePhysVal_fcmp` — `extern/mVMC-1.3.0/src/mVMC/physcal_lanczos.c:336`
> - C: `CalculatePhysVal_real` — `extern/mVMC-1.3.0/src/mVMC/physcal_lanczos.c:313`
> - C: `LSLocalCisAjs` — `extern/mVMC-1.3.0/src/mVMC/lslocgrn.c:98`
> - C: `calculateQCAQ` — `extern/mVMC-1.3.0/src/mVMC/vmccal.c:852`
> - C: `calculateQCACAQ` — `extern/mVMC-1.3.0/src/mVMC/vmccal.c:871`
> - Rust: `lanczos_phys_values` — `crates/mvmc-core/src/io.rs:436`
> - Rust: `calculate_lanczos_green` — `crates/mvmc-core/src/observables.rs:2316`
> - Rust: `output_phys_data` — `crates/mvmc-core/src/io.rs:161`
> - Parity: `lanczos_phys_values` evaluates `(Q[i] + alpha*(Q[n+i] + Q[2n+i]) + alpha*alpha*Q[3n+i]) / dnorm` with `dnorm = (1 + 2*alpha*h1 + alpha*alpha*h2_1).re`, the same expression and association as C `CalculatePhysVal_fcmp` (`physcal_lanczos.c:336-358`). In real mode the imaginary parts are written as the literal `0.0` (as in C, which has a separate `_real` writer). The symbols $A_{1(01)}$/$A_{1(10)}$ in the C manual correspond to the two cross slots `Q[n+i]` and `Q[2n+i]`; the assignment of slots to the manual symbols follows `calculateQCAQ` (conjugated left factor) and was not re-derived here **(unverified)**.

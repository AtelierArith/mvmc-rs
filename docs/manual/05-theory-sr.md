# 5. Theory IV: stochastic reconfiguration

[Contents](README.md) · Previous: [4. Markov-chain sampling](04-theory-sampling.md) · Next: [6. Physical quantities and Lanczos](06-theory-observables-lanczos.md)

Parameter optimization (`NVMCCalMode = 0`) minimizes $E(\alpha)=\langle\psi_\alpha|H|\psi_\alpha\rangle/\langle\psi_\alpha|\psi_\alpha\rangle$
with the **stochastic reconfiguration** (SR) method: an imaginary-time step
$e^{-\Delta t H}|\psi_\alpha\rangle$ is projected onto the tangent space of the variational
manifold, which yields a linear system with the quantum-geometric metric $S$.
mVMC uses the formulation of [Tahara and Imada, J. Phys. Soc. Jpn. 77, 114701 (2008)].

## 5.1 Log-derivatives $O_k$

For every real component $x_a$ of the parameters ($\operatorname{Re}\alpha_k$ and $\operatorname{Im}\alpha_k$) the code needs
$O_a(x)=\partial\ln\psi(x)/\partial x_a$. Because the amplitude is holomorphic in $\alpha_k$,
$\partial_{\operatorname{Im}\alpha_k}\ln\psi=i\,\partial_{\operatorname{Re}\alpha_k}\ln\psi$, so each parameter contributes a pair
$(O_k,\ iO_k)$ and the vector of the SR step has the layout of
[3.1](03-theory-wavefunction.md#parameter-layout): $O=[\,1\ (\text{energy slot}),\ \text{proj},\ \text{RBM},\ \text{Slater},\ \text{OptTrans}\,]$
with two entries per parameter. The pieces are:

| Block | $O_k(x)$ |
|-------|----------|
| Gutzwiller, Jastrow, DH2, DH4 | the integer counter $c_k(x)$ (pair $(c_k,0)$, since only $\operatorname{Re}\alpha_k$ enters $\ln P$) |
| RBM, visible bias $a_p$ | $m_p(x)$ |
| RBM, hidden bias $b_h$ | $\tanh\theta_h(x)$ |
| RBM, coupling $W_{ph}$ | $m_p(x)\tanh\theta_h(x)$ |
| Slater $f_k$ | $\displaystyle\frac{1}{\mathrm{IP}(x)}\sum_qw_q\,\mathrm{Pf}X_q\;\tfrac12\,\mathrm{Tr}\!\Big[X_q^{-1}\frac{\partial X_q}{\partial f_k}\Big]$ |
| OptTrans $p^{\rm opt}_o$ | $\displaystyle\frac{1}{\mathrm{IP}(x)}\sum_{K,s}p_K\tfrac12\sin\beta_s\omega_sP_S\,\mathrm{Pf}X_{o,K,s}$ |

For the Slater block the code evaluates the trace analytically from the stored
inverse: every pair $(m,n)$ of electrons whose spin-orbitals map to orbital
index $k$ adds the entry $\texttt{InvM}[m][n]$ times the sector's spin-rotation factor
($cs$, $-cc$, $ss$, $-cs$ for the up–up, up–down, down–up, down–down blocks of
[3.4](03-theory-wavefunction.md#34-quantum-number-projection)) times the translation sign; the
$\mathrm{Pf}X_q$-weighted, $w_q$-weighted sum over sectors is then divided by
$\mathrm{IP}$. At $\beta=0$ this reduces to $O_k=-\sum_{i\uparrow,j\downarrow}(X^{-1})_{ij}\,\sigma^{(k)}_{ij}$.

> **Implementation**
> - C: `VMCMainCal` (assembles `SROptO`) — `extern/mVMC-1.3.0/src/mVMC/vmccal.c:82`
> - C: `SlaterElmDiff_fcmp` — `extern/mVMC-1.3.0/src/mVMC/slater.c:100`
> - C: `SlaterElmDiff_fsz` — `extern/mVMC-1.3.0/src/mVMC/slater_fsz.c:119`
> - C: `RBMDiff` — `extern/mVMC-1.3.0/src/mVMC/rbm.c:323`
> - C: `calculateOptTransDiff` — `extern/mVMC-1.3.0/src/mVMC/vmccal.c:639`
> - Rust: `set_projection_diff` — `crates/mvmc-core/src/observables.rs:201`
> - Rust: `set_rbm_diff` — `crates/mvmc-core/src/sampling/rbm.rs:693`
> - Rust: `slater_elm_diff_with_scratch_timed` — `crates/mvmc-core/src/slater_derivative.rs:176`
> - Rust: `slater_elm_diff_fsz_with_scratch` — `crates/mvmc-core/src/slater_derivative.rs:473`
> - Rust: `opt_trans_diff` — `crates/mvmc-core/src/observables.rs:998`
> - Parity: in C the per-sector buffer is accumulated with `buf[orbidx] += invM_i[msj]*cs*tOrbSgn_i[msj]` etc., summed over sectors with `QPFullWeight`, and finally multiplied by $1/\mathrm{IP}$ (`slater.c:194-238`). The Rust accumulation preserves this order (a tensor contraction helper is used for the sector sum, `qp_weighted_orbital_sum_einsum`, `observables.rs:537`; test `qp_weighted_orbital_sum_einsum_matches_manual_complex_reference`) and multiplies by `julia_complex::reciprocal(ip)` rather than dividing; results are compared with tolerances. The OptTrans layout differs between C and Rust, see [3.4](03-theory-wavefunction.md#34-quantum-number-projection).

## 5.2 The SR equations

Let $\langle\cdot\rangle$ be the Monte Carlo average over the saved samples (weight 1).
The code accumulates, per SR step,

$$
\mathrm{OO}_{ab}=\big\langle O_b\,\overline{O_a}\big\rangle\ (a\ge\text{first parameter}),\qquad
\mathrm{OO}_{0b}=\langle O_b\rangle,\qquad
\mathrm{HO}_b=\langle E_{\rm loc}\,O_b\rangle,\ \ \mathrm{HO}_0=\langle E_{\rm loc}\rangle .
$$

The metric and force use only real parts, treating real and imaginary parts of the
parameters as independent real variables:

$$
\boxed{\;S_{ab}=\operatorname{Re}\mathrm{OO}_{ab}-\operatorname{Re}\mathrm{OO}_{0a}\,\operatorname{Re}\mathrm{OO}_{0b},\qquad
g_a=-2\,\Delta t\,\big(\operatorname{Re}\mathrm{HO}_a-\operatorname{Re}\mathrm{HO}_0\operatorname{Re}\mathrm{OO}_{0a}\big)\;}
$$

with $\Delta t=\texttt{DSROptStepDt}$. $g_a$ is $-\Delta t$ times the energy gradient
$\partial E/\partial x_a=2\operatorname{Re}\big(\langle E_{\rm loc}O_a\rangle-\langle E_{\rm loc}\rangle\langle O_a\rangle\big)$.
Two stabilizations are applied before solving:

1. **Redundant directions are cut.** Let $S_{\max}=\max_aS_{aa}$. Components with $S_{aa}<S_{\max}\cdot\texttt{DSROptRedCut}$ are
   removed from the solve and not updated ($\texttt{diagCut}$ in `zvo_SRinfo.dat`). Components with optimization flag $\ne1$ are fixed
   ($\texttt{optCut}$). The remaining $n_S$ components, listed by `smatToParaIdx`, form the system.
2. **Diagonal shift.** $S_{aa}\leftarrow S_{aa}\,(1+\texttt{DSROptStaDel})$ (a multiplicative shift, the $\varepsilon$ of Tahara–Imada).

The parameter change solves

$$
\sum_b S'_{ab}\,\delta x_b=g_a,\qquad
x_a\leftarrow x_a+\delta x_a,
$$

where the component $a=2k$ updates $\operatorname{Re}\alpha_k$ and $a=2k+1$ updates $\operatorname{Im}\alpha_k$
(complex mode), or $a=k$ in real mode. If any $\delta x_a$ is not finite, or the factorization/solver reports an error,
no parameter is updated and the run stops with an error. Afterwards the parameters
are synchronized ([3.7](03-theory-wavefunction.md#37-initial-values-and-synchronization)).

> **Implementation**
> - C: `StochasticOpt` (cut, diag shift via `stcOptInit`, solve, update) — `extern/mVMC-1.3.0/src/mVMC/stcopt.c:33`
> - C: `stcOptInit` (builds $S$ and $g$) — `extern/mVMC-1.3.0/src/mVMC/stcopt_dposv.c:53`
> - C: `stcOptMain` (LAPACK `dposv`) — `extern/mVMC-1.3.0/src/mVMC/stcopt_dposv.c:33`
> - Rust: `stochastic_opt_real_timed` — `crates/mvmc-core/src/sr.rs:78`
> - Rust: `stochastic_opt_complex_timed` — `crates/mvmc-core/src/sr.rs:172`
> - Rust: `assemble_s_g` (C-order backend, real and complex layouts) — `crates/mvmc-core/src/sr_backend.rs:336`
> - Rust: `collect_active_real` — `crates/mvmc-core/src/sr.rs:357`
> - Rust: `component_is_optimized` — `crates/mvmc-core/src/sr.rs:25`
> - Rust: `cholesky_solve` — `crates/mvmc-core/src/sr.rs:856`
> - Rust: `update_parameter_value` — `crates/mvmc-core/src/sr.rs:760`
> - Parity: `S[idx] = OO[(pi+2)*(2*size)+(pj+2)].re - OO[pi+2].re*OO[pj+2].re` and `S[ii] *= 1+DSROptStaDel` (`stcopt_dposv.c:69-74`); `g[si] = -DSROptStepDt*2.0*(HO[pi+2].re - HO[0].re*OO[pi+2].re)` (`stcopt_dposv.c:81`) — Rust `build_s_g_complex` evaluates the same expressions in the same order (real mode uses index offset 1 instead of 2). In real mode C embeds the real matrices into the complex layout with zero imaginary entries, so $S_{\max}$ includes the zero imaginary variances; Rust's `collect_active_real` folds from `0.0` for the same reason. Rust solves with LAPACK `dpotrf_`/`dpotrs_` (Cholesky of the upper triangle, declared in `sr.rs:45-52`), C with `dposv('U')`. A non-positive-definite $S$ rejects the update in Rust (`cholesky_solve` returns `Err`) and C returns the `dposv` `info`. The $\texttt{zvo\_SRinfo.dat}$ line that C writes for the direct solver (`stcopt.c:157`) is **not** written by Rust ([9](09-output-files.md)).

## 5.3 Accumulating $\mathrm{OO}$ and $\mathrm{HO}$; the `NStore` option

For `NSRCG = 0` and `NStore = 0` each sample performs a rank-1 update of the full
matrix: `calculateOO` (complex, explicit loops) or `calculateOO_real` (`DGER`),
$\mathrm{OO}\mathrel{+}=w\,O\,O^{\dagger}$ and $\mathrm{HO}\mathrel{+}=wE_{\rm loc}O$ (cost $O(N_p^2)$ per sample).
With `NStore = 1` (the default) or `NSRCG = 1` the scaled samples $\sqrt wO_s$ are stored
(`SROptO_Store`) and the matrix is formed once after the sampling loop by a
matrix–matrix product (`calculateOO_Store`, `calculateOO_Store_real`); this costs
$O(N_pN_{\rm smp})$ extra memory but is much faster, and is required by the CG solver. Across MPI ranks the accumulators
are summed and divided by the total weight $W$ (`WeightAverageSROpt`).

> **Implementation**
> - C: `calculateOO` — `extern/mVMC-1.3.0/src/mVMC/vmccal.c:769`
> - C: `calculateOO_real` — `extern/mVMC-1.3.0/src/mVMC/vmccal.c:796`
> - C: `calculateOO_Store` — `extern/mVMC-1.3.0/src/mVMC/vmccal.c:692`
> - C: `calculateOO_Store_real` — `extern/mVMC-1.3.0/src/mVMC/vmccal.c:656`
> - C: `WeightAverageSROpt` — `extern/mVMC-1.3.0/src/mVMC/average.c:78`
> - C: `WeightAverageSROpt_real` — `extern/mVMC-1.3.0/src/mVMC/average.c:115`
> - Rust: `calculate_oo` — `crates/mvmc-core/src/observables.rs:263`
> - Rust: `calculate_oo_real` — `crates/mvmc-core/src/observables.rs:219`
> - Rust: `calculate_oo_store` — `crates/mvmc-core/src/observables.rs:425`
> - Rust: `calculate_oo_store_real` — `crates/mvmc-core/src/observables.rs:322`
> - Rust: `finalize_oo_store` — `crates/mvmc-core/src/observables.rs:447`
> - Rust: `weight_average_sr_opt` — `crates/mvmc-core/src/average.rs:31`
> - Rust: `weight_average_sr_opt_real` — `crates/mvmc-core/src/average.rs:50`
> - Rust: `reduce_accumulators` — `crates/mvmc-core/src/run.rs:1750`
> - Parity: C scales stored samples by `sqrt(w)` (`SROptO_Store[...] = sqrtw*SROptO[...]`, `vmccal.c:241,248`); the Gram product is formed once per step. For the real stored matrix Rust follows Julia's SYRK dispatch and upper-triangle copy; the complex stored product preserves a sequential sample sum (`sr_store_gram_julia`, `observables.rs:497`; tests `stored_direct_sr_gram_matches_sampled_julia_values`, `real_gram_matches_julia_generic_and_syrk_dispatch_boundary`). These floating-point summation orders differ between BLAS providers and are compared with tolerances. Only the active branch (real or complex) is reduced over MPI, exactly as `vmcmain.c` selects one `WeightAverageSROpt` branch.

## 5.4 Conjugate-gradient solver (`NSRCG = 1`)

Instead of forming and factorizing $S'$, the CG solver applies it to a vector using the stored
samples:

$$
S'x=\frac1W\sum_s O^{(s)}\big(O^{(s)T}x\big)\;-\;\langle O\rangle\,\big(\langle O\rangle\!\cdot\!x\big)\;+\;\texttt{DSROptStaDel}\;\mathrm{diag}(S)\,x ,
$$

(in complex mode the sample matrix has separate real and imaginary parts, giving two `dgemv` pairs; the vector has $2N_{\rm para}$ real entries).
It solves $S'x=g$ with the standard CG recurrence

$$
r_0=d_0=g,\ \ \delta_0=r_0\!\cdot\!r_0;\qquad
\alpha_i=\frac{\delta_i}{d_i\!\cdot\!S'd_i},\ \ x_{i+1}=x_i+\alpha_id_i,\ \ r_{i+1}=r_i-\alpha_iS'd_i,\ \
\beta_i=\frac{r_{i+1}\!\cdot\!r_{i+1}}{\delta_i},\ \ \delta_{i+1}=\beta_i\delta_i,\ \ d_{i+1}=r_{i+1}+\beta_id_i ,
$$

with three rules that matter for reproducing C:

- stop when $\delta<\texttt{DSROptCGTol}^2\,n_S^2$ (checked at the *start* of an iteration),
- run at most `NSROptCGMaxIter` iterations (`n_S` if it is $\le 0$),
- **every 20th iteration** the residual is recomputed exactly, $r=g-S'x$, instead of by the recurrence.

The cut and the force are identical to the direct solver ([5.2](#52-the-sr-equations)); the diagonal elements used for the
cut are $\mathrm{OO}_{aa}-\mathrm{OO}_{0a}^2$. The number of CG iterations is printed as the last column of `zvo_SRinfo.dat`.
Only `NSplitSize = 1` is supported for CG in Rust (grouped CG is undefined in C) ([7.5](07-input-files.md#75-supported-and-rejected-inputs)).

> **Implementation**
> - C: `StochasticOptCG` (wrapper selecting real/complex) — `extern/mVMC-1.3.0/src/mVMC/stcopt_cg.c:42`
> - C: `fn_StochasticOptCG` — `extern/mVMC-1.3.0/src/mVMC/stcopt_cg_impl.c:73`
> - C: `fn_StochasticOptCG_Main` — `extern/mVMC-1.3.0/src/mVMC/stcopt_cg_impl.c:258`
> - C: `fn_operate_by_S` — `extern/mVMC-1.3.0/src/mVMC/stcopt_cg_impl.c:356`
> - C: `fn_StochasticOptCG_Init` — `extern/mVMC-1.3.0/src/mVMC/stcopt_cg_impl.c:428`
> - Rust: `stochastic_opt_cg_with_reducer` — `crates/mvmc-core/src/sr_cg.rs:101`
> - Rust: `solve_with_reducer` — `crates/mvmc-core/src/sr_cg.rs:328`
> - Rust: `apply_with_reducer` — `crates/mvmc-core/src/sr_cg.rs:442`
> - Rust: `sequential_dot` — `crates/mvmc-core/src/sr_cg.rs:264`
> - Rust: `SampledSrOperator` — `crates/mvmc-core/src/sr_cg.rs:288`
> - Parity: this is where operation order matters most. C computes `cg_thresh = DSROptCGTol*DSROptCGTol * (double)nSmat * (double)nSmat` (`stcopt_cg_impl.c:265`), and the recurrence `beta = xdot(r,r)/delta; delta = beta*delta;` — the old $\delta$ is *not* replaced by $r\!\cdot\!r$ (`stcopt_cg_impl.c:333-336`). Rust reproduces both (`crates/mvmc-core/src/sr_cg.rs:346`, `crates/mvmc-core/src/sr_cg.rs:387`: "C:336 rounds the quotient and then multiplies it by the old norm"). The dot products are sequential (`sequential_dot`) rather than BLAS `ddot`, and the products use the same `dgemv` pairs as C, which is why CG is sensitive to FMA and reduction order; truncated CG results therefore use a tolerance gate, not bit parity.
> - Parity: the diagonal shift appears as `z += sdiag[si]*DSROptStaDel*x[si]` (`stcopt_cg_impl.c:420`) rather than a modified matrix; $\langle O\rangle\cdot x$ uses `xdot`. The MPI reduction of the sampled product is performed before the global weight, mean and shift corrections (`apply_with_reducer` docs).

## 5.5 The optimization loop and the final averaging window

`VMCParaOpt` repeats, for $t=0\ldots\texttt{NSROptItrStep}-1$:

1. `UpdateSlaterElm` (rebuild the Slater tables from the current parameters), `UpdateQPWeight`.
2. Sample (`VMCMakeSample`, [chapter 4](04-theory-sampling.md)), measure (`VMCMainCal`): $E_{\rm loc}$, $O$, $\mathrm{OO}$, $\mathrm{HO}$.
3. Weighted averages over ranks (`WeightAverageWE`, `WeightAverageSROpt`).
4. **Write** the step's energy and parameters (`zvo_out`, `zvo_var`) — the parameters are those *before* this step's update.
5. SR update (direct or CG), then `SyncModifiedParameter`.
6. If $t\ge\texttt{NSROptItrStep}-\texttt{NSROptItrSmp}$, store $(\langle H\rangle,\langle H^2\rangle,\text{all parameters})$ for the final window (`StoreOptData`).

After the last step `OutputOptData` writes `zqp_opt.dat`: for each of $\langle H\rangle$, $\langle H^2\rangle$ and every parameter the window
mean (real, imaginary) and the sample standard deviation $\sqrt{\sum|x_s-\bar x|^2/(n-1)}$ (three numbers per quantity); then one file per parameter block with the window means
only. If `NSROptItrSmp = 1` only the last values are written, as (value, 0) pairs, and no per-block files. The optimized parameters in `zqp_opt.dat` are
the input of a later PhysCal run ([chapter 6](06-theory-observables-lanczos.md), [8.2](08-running.md#82-physical-quantities-with-fixed-parameters)).

> **Implementation**
> - C: `VMCParaOpt` — `extern/mVMC-1.3.0/src/mVMC/vmcmain.c:331`
> - C: `StoreOptData` — `extern/mVMC-1.3.0/src/mVMC/avevar.c:82`
> - C: `OutputOptData` — `extern/mVMC-1.3.0/src/mVMC/avevar.c:94`
> - Rust: `vmc_para_opt_timed` — `crates/mvmc-core/src/run.rs:1113`
> - Rust: `vmc_para_opt` — `crates/mvmc-core/src/run.rs:1092`
> - Rust: `run_para_opt_from_namelist` — `crates/mvmc-core/src/run.rs:1444`
> - Rust: `store_opt_data` — `crates/mvmc-core/src/io.rs:21`
> - Rust: `output_opt_data` — `crates/mvmc-core/src/io.rs:561`
> - Parity: the window is `step >= NSROptItrStep - NSROptItrSmp`; Rust rejects `NSROptItrSmp > NSROptItrStep` up front ("nsteps must be >= nsmp; C leaves oversized-window rows unwritten", `validate_optimization_window`, `run.rs:1306`). The standard-deviation formula is `sqrt(var/(n-1))` of `creal(data*conj(data))` in both. The `--nsteps`/`--nsmp` command-line overrides change `NSROptItrStep`/`NSROptItrSmp` for the window.

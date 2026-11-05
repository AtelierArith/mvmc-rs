//! Sampled SR-CG: the C 1.4.0 `StochasticOptCG` step and its CG solvers.

use crate::output_files::{RunFiles, SrInfoRow};
use crate::{reducer::Reducer, ExpertModeData, VmcOptimizationState};
use std::io;
use std::{cell::RefCell, rc::Rc};

/// Actual sampled-product boundary within the production CG operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CgProductPhase {
    /// Root search vector after its in-place broadcast, before local BLAS.
    RootSearch,
    /// Local sampled Gram product before the global reduction.
    Local,
    /// Global raw product, before weight/mean/diagonal corrections.
    Global,
    /// Corrected covariance product used by the solver.
    Corrected,
}

/// Borrowed actual CG iteration state; observers cannot mutate solver buffers.
pub struct CgIterationView<'a> {
    /// Zero for initialization, otherwise the actual iteration number.
    pub iteration: usize,
    /// Actual accumulated solution.
    pub solution: &'a [f64],
    /// Actual residual after initialization or the iteration update.
    pub residual: &'a [f64],
    /// Actual search direction.
    pub direction: &'a [f64],
    /// Sequential residual dot product used by the solver.
    pub delta: f64,
    /// Actual step length, absent at initialization.
    pub alpha: Option<f64>,
}

/// Read-only diagnostics at actual production CG calculation boundaries.
pub trait CgObserver {
    /// Observe actual active mapping, saved samples, means, variances and gradient.
    fn prepared(&self, _mapping: &[usize], _operator: &SampledSrOperator, _gradient: &[f64]) {}
    /// Observe actual search vector and product at the named operator boundary.
    fn product(&self, _phase: CgProductPhase, _search: &[f64], _product: &[f64]) {}
    /// Observe actual initialization or completed iteration state.
    fn iteration(&self, _state: CgIterationView<'_>) {}
    /// Observe the actual returned solution, residual, direction and iteration count.
    fn finished(&self, _result: &CgSolution) {}
}

thread_local! {
    static CG_OBSERVER: RefCell<Option<Rc<dyn CgObserver>>> = const { RefCell::new(None) };
}

/// Clears its thread-local observer on drop, including panic unwinding.
pub struct CgObserverGuard(Rc<dyn CgObserver>);

impl Drop for CgObserverGuard {
    fn drop(&mut self) {
        CG_OBSERVER.with(|slot| {
            let mut active = slot.borrow_mut();
            if active
                .as_ref()
                .is_some_and(|observer| Rc::ptr_eq(observer, &self.0))
            {
                *active = None;
            }
        });
    }
}

/// Install read-only diagnostics on the calling runner thread only.
pub fn install_cg_observer(observer: Rc<dyn CgObserver>) -> Result<CgObserverGuard, String> {
    CG_OBSERVER.with(|slot| {
        let mut active = slot.borrow_mut();
        if active.is_some() {
            return Err("CG observer already installed on this thread".into());
        }
        *active = Some(observer.clone());
        Ok(CgObserverGuard(observer))
    })
}

fn observe(f: impl FnOnce(&dyn CgObserver)) {
    CG_OBSERVER.with(|slot| {
        if let Some(observer) = slot.borrow().clone() {
            f(observer.as_ref());
        }
    });
}

/// Apply the C 1.4.0 sampled SR-CG step (`StochasticOptCG`), writing the
/// `zqp_SRinfo.dat` row before the update.
///
/// Returns 0 after a finite update and 1 when the increment is not finite with
/// `NSRCGAbortOnFail = 0` (the update is skipped, as in C). A failed CG solve
/// is an error when `NSRCGAbortOnFail` is set (C calls `MPI_Abort`), and so is
/// a non-finite increment.
pub fn stochastic_opt_cg(
    data: &mut ExpertModeData,
    state: &VmcOptimizationState,
    files: Option<&mut RunFiles>,
) -> io::Result<i32> {
    stochastic_opt_cg_with_reducer(data, state, files, &crate::reducer::SingleProcessReducer)
}

/// Which C 1.4.0 CG solver runs (`useDiagScale`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CgVariant {
    /// `StochasticOptCG_Main`: unpreconditioned CG.
    Plain,
    /// `StochasticOptPreconCG_DiagScale_Main`: Jacobi (DiagScale) preconditioned CG.
    DiagScale,
}

impl CgVariant {
    fn name(self) -> &'static str {
        match self {
            Self::Plain => "CG",
            Self::DiagScale => "CG-diag",
        }
    }

    fn other(self) -> Self {
        match self {
            Self::Plain => Self::DiagScale,
            Self::DiagScale => Self::Plain,
        }
    }
}

/// C `get_absmax` (stcopt_cg.c:36-46): the largest `|p[i]|` in `start..end`,
/// keeping the earlier value when a comparison fails (NaN).
fn get_absmax(start: usize, end: usize, p: &[f64]) -> f64 {
    if end <= start {
        return 0.0;
    }
    let mut max = p[start].abs();
    for value in &p[start + 1..end] {
        if max < value.abs() {
            max = value.abs();
        }
    }
    max
}

/// Slater-block factors of C `Rescale4SRCG` (stcopt_cg_impl.c:772-875).
struct SlaterRescale {
    /// `OFFSET*sltStart .. OFFSET*sltEnd`: the rescaled O components.
    components: std::ops::Range<usize>,
    /// `rescaleOO_SlaterRatio`.
    oo_ratio: f64,
    /// `rescaleO_SlaterRatio = sqrt(rescaleOO_SlaterRatio)`.
    o_ratio: f64,
}

/// `None` where C `Rescale4SRCG` returns without rescaling.
fn slater_rescale(
    data: &ExpertModeData,
    oo: &[f64],
    size: usize,
    offset: usize,
    n_para: usize,
) -> Option<SlaterRescale> {
    let n_proj = data.projection_layout().n_proj;
    let n_slater = mvmc_expert_parsers::utils::parameter_init::n_slater(data);
    // Para = Proj | RBM | ProjBF (always empty here) | Slater | OptTrans.
    let slt_start = n_proj + data.count_rbm_parameters();
    let slt_end = slt_start + n_slater;
    if n_proj == 0 || n_proj >= n_para || n_slater == 0 {
        return None;
    }
    // r[pi] = OO[pi+OFFSET] - O[pi+OFFSET]^2: the diagonal of S, before the cut.
    let r: Vec<f64> = (0..offset * n_para)
        .map(|pi| oo[size + pi + offset] - oo[pi + offset] * oo[pi + offset])
        .collect();
    let proj_max = get_absmax(0, offset * n_proj, &r);
    let mut slater_max = get_absmax(offset * slt_start, offset * slt_end, &r);
    if proj_max < 1e-10 {
        return None;
    }
    if slater_max < 1e-10 {
        slater_max = 1e-10;
    }
    let oo_ratio = proj_max / slater_max;
    Some(SlaterRescale {
        components: offset * slt_start..offset * slt_end,
        oo_ratio,
        o_ratio: oo_ratio.sqrt(),
    })
}

/// C's `"error"`/`"warning"` prefix and action of a failed SR-CG solve.
fn failure_words(abort: bool, continued: &'static str) -> (&'static str, &'static str) {
    if abort {
        ("error", "Abort")
    } else {
        ("warning", continued)
    }
}

/// Apply sampled SR-CG with the global accumulator communicator.
pub fn stochastic_opt_cg_with_reducer<R: Reducer + ?Sized>(
    data: &mut ExpertModeData,
    state: &VmcOptimizationState,
    files: Option<&mut RunFiles>,
    reducer: &R,
) -> io::Result<i32> {
    let n_proj = data.projection_layout().n_proj;
    let n_para = data.count_variational_parameters();
    let complex = crate::run::get_all_complex_flag(data).map_err(io::Error::other)?;
    let offset = if complex { 2 } else { 1 };
    let full = offset * n_para;
    if full == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "SR-CG requires a nonempty parameter variance array",
        ));
    }
    if data.optimization_flags.is_empty() {
        data.optimization_flags = vec![1; 2 * n_para];
    }
    let root = reducer.rank() == 0;
    let size = offset * state.sr_opt.sr_opt_size;
    let mut oo: Vec<f64> = if complex {
        state.sr_opt.sr_opt_oo.iter().map(|z| z.re).collect()
    } else {
        state.sr_opt.sr_opt_oo_real.clone()
    };
    let mut ho: Vec<f64> = if complex {
        state.sr_opt.sr_opt_ho.iter().map(|z| z.re).collect()
    } else {
        state.sr_opt.sr_opt_ho_real.clone()
    };
    // stcopt_cg.c:58-65: RescaleSmat rescales the averaged operands, the stored
    // samples (below) and the Slater parameters before StochasticOptCG.
    let rescale = if data.modpara.rescale_smat {
        slater_rescale(data, &oo, size, offset, n_para)
    } else {
        None
    };
    if let Some(rescale) = &rescale {
        for i in rescale.components.clone() {
            oo[size + i + offset] *= rescale.oo_ratio;
            oo[i + offset] *= rescale.o_ratio;
            ho[i + offset] *= rescale.o_ratio;
        }
        for value in &mut data.slater_params {
            *value /= rescale.o_ratio;
        }
    }
    let variance: Vec<f64> = (0..full)
        .map(|pi| {
            let idx = pi + offset;
            oo.get(size + idx).copied().unwrap_or(0.0) - oo.get(idx).copied().unwrap_or(0.0).powi(2)
        })
        .collect();
    let (maximum, minimum) = if variance.iter().any(|v| v.is_nan()) {
        (f64::NAN, f64::NAN)
    } else {
        (
            variance.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            variance.iter().copied().fold(f64::INFINITY, f64::min),
        )
    };
    let threshold = maximum * data.modpara.dsr_opt_red_cut;
    let mut mapping = Vec::new();
    let mut opt_cut = 0;
    let mut diag_cut = 0;
    for (pi, &v) in variance.iter().enumerate() {
        let flag = if complex { pi } else { 2 * pi };
        if !crate::sr::component_is_optimized(data, flag) {
            opt_cut += 1;
        } else if v < threshold {
            diag_cut += 1;
        } else {
            mapping.push(pi);
        }
    }
    // With nSmat = 0, C 1.4.0 runs the solver anyway, returns -(0+1), aborts by
    // default and reads r[0] out of bounds for zqp_SRinfo.dat. An empty active
    // set is an exact no-op here, as before (#682).
    if mapping.is_empty() {
        return Ok(0);
    }
    let samples = data.modpara.nvmc_sample.max(0) as usize;
    let mut operator = SampledSrOperator::new(mapping.len(), samples, complex);
    let mut gradient = vec![0.0; mapping.len()];
    let dt = 2.0 * data.modpara.dsr_opt_step_dt;
    // stcopt_cg_impl.c:733-744 `omp parallel for` over the active components `si`.
    crate::threading::for_each_mut(&mut operator.mean, 2, |si, value| {
        *value = oo[mapping[si] + offset]
    });
    crate::threading::for_each_mut(&mut operator.diagonal, 2, |si, value| {
        *value = variance[mapping[si]]
    });
    crate::threading::for_each_mut(&mut gradient, 4, |si, value| {
        let idx = mapping[si] + offset;
        *value = -dt * (ho[idx] - ho[0] * oo[idx]);
    });
    let n_active = mapping.len();
    if samples > 0 {
        // RescaleSmat multiplies the stored Slater components in place
        // (stcopt_cg_impl.c:856-862); the factor is applied while copying.
        let factor = |pi: usize| match &rescale {
            Some(r) if r.components.contains(&pi) => Some(r.o_ratio),
            _ => None,
        };
        // Sample `s` owns the contiguous window `s * n_active ..` of the sample matrices.
        let fill = |s: usize, real: &mut [f64], imag: &mut [f64]| {
            for (si, &pi) in mapping.iter().enumerate() {
                let src = s * size + pi + offset;
                let scale = factor(pi);
                if complex {
                    if let Some(&o) = state.sr_opt.sr_opt_o_store.get(src) {
                        let o = scale.map_or(o, |f| o * f);
                        real[si] = o.re;
                        imag[si] = o.im;
                    }
                } else if let Some(&o) = state.sr_opt.sr_opt_o_store_real.get(src) {
                    real[si] = scale.map_or(o, |f| o * f);
                }
            }
        };
        if complex {
            crate::threading::for_each_chunk_pair_mut(
                &mut operator.real_samples,
                n_active,
                &mut operator.imag_samples,
                n_active,
                6 * n_active,
                fill,
            );
        } else {
            crate::threading::for_each_chunk_mut(
                &mut operator.real_samples,
                n_active,
                3 * n_active,
                |s, real| fill(s, real, &mut []),
            );
        }
    }
    let max_iterations = if data.modpara.nsr_opt_cg_max_iter > 0 {
        data.modpara.nsr_opt_cg_max_iter as usize
    } else {
        mapping.len()
    };
    observe(|observer| observer.prepared(&mapping, &operator, &gradient));
    let variant = if data.modpara.use_diag_scale {
        CgVariant::DiagScale
    } else {
        CgVariant::Plain
    };
    let inv_weight = 1.0 / state.energy.wc.re;
    let (sta_del, tolerance) = (data.modpara.dsr_opt_sta_del, data.modpara.dsr_opt_cg_tol);
    let abort = data.modpara.nsrcg_abort_on_fail;
    let solve = |operator: &mut SampledSrOperator, variant| {
        operator
            .solve_variant_with_reducer(
                variant,
                &gradient,
                inv_weight,
                sta_del,
                tolerance,
                max_iterations,
                reducer,
            )
            .map_err(io::Error::other)
    };
    let mut result = solve(&mut operator, variant)?;
    let mut info = result.status;
    // stcopt_cg_impl.c:203-269: failure report, optional fallback and abort.
    if result.status < 0 {
        let fallback_iter = (-result.status - 1).max(0) as usize;
        let reason = |iter: usize| {
            if iter >= max_iterations {
                "did not converge"
            } else {
                "became numerically unstable"
            }
        };
        if root {
            eprintln!(
                "warning: SR-{} {} at iter={fallback_iter} (max_iter={max_iterations}).",
                variant.name(),
                reason(fallback_iter)
            );
        }
        if data.modpara.nsrcg_fallback {
            let alternative = variant.other();
            if root {
                if !FALLBACK_INFO_PRINTED.with(|printed| printed.replace(true)) {
                    eprintln!(
                        "remark: in zqp_SRinfo.dat, negative info means SR-CG fallback at \
                         iteration (info=-(iter+1))."
                    );
                }
                eprintln!(
                    "warning: fallback from SR-{} to SR-{}.",
                    variant.name(),
                    alternative.name()
                );
            }
            // C reinitializes VecCG (x = 0) before the alternative solve.
            result = solve(&mut operator, alternative)?;
            if root && result.status >= 0 {
                eprintln!(
                    "remark: SR-{} after fallback converged at iter={}.",
                    alternative.name(),
                    result.status
                );
            }
            info = -(fallback_iter as i64 + 1);
            if result.status < 0 {
                let alt_iter = (-result.status - 1).max(0) as usize;
                let (level, action) =
                    failure_words(abort, "Continue with the approximate solution");
                let message = format!(
                    "{level}: fallback SR-{} {} at iter={alt_iter} (max_iter={max_iterations}). \
                     {action}.",
                    alternative.name(),
                    reason(alt_iter)
                );
                if abort {
                    return Err(io::Error::other(message));
                }
                if root {
                    eprintln!("{message}");
                }
            }
        } else {
            let (level, action) = failure_words(abort, "Continue with the approximate solution");
            let message = format!("{level}: SR-CG fallback is disabled. {action}.");
            if abort {
                return Err(io::Error::other(message));
            }
            if root {
                eprintln!("{message}");
            }
            info = -(fallback_iter as i64 + 1);
        }
    }
    if let Some(files) = files {
        let mut imax = 0;
        for i in 1..result.solution.len() {
            if result.solution[imax].abs() < result.solution[i].abs() {
                imax = i;
            }
        }
        // C stcopt_cg_impl.c:285 prints the global NPara, not OFFSET*NPara.
        files.write_sr_info(&SrInfoRow {
            n_para: n_para as i64,
            n_smat: mapping.len() as i64,
            opt_num: opt_cut,
            cut_num: diag_cut,
            s_diag_max: maximum,
            s_diag_min: minimum,
            r_max: result.solution[imax],
            i_max: mapping[imax] as i64,
            cg_info: Some(info),
        })?;
    }
    // stcopt_cg_impl.c:291-310: a non-finite increment skips the update and
    // returns 1 (the run then stops), or aborts with NSRCGAbortOnFail.
    if result.solution.iter().any(|x| !x.is_finite()) {
        let (level, action) = failure_words(abort, "Skip this parameter update");
        let message = format!("{level}: SR-CG produced non-finite parameter updates. {action}.");
        if abort {
            return Err(io::Error::other(message));
        }
        if root {
            eprintln!("{message}");
        }
        return Ok(1);
    }
    for (&pi, &x) in mapping.iter().zip(&result.solution) {
        let para = if complex { pi / 2 } else { pi };
        let (re, im) = if complex && pi % 2 != 0 {
            (0.0, x)
        } else {
            (x, 0.0)
        };
        crate::sr::update_parameter_value(data, para, re, im, n_proj);
    }
    Ok(0)
}

thread_local! {
    /// C's function-static `fallback_info_printed` (stcopt_cg_impl.c:119).
    static FALLBACK_INFO_PRINTED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Dot product with Julia/C's explicit sequential accumulation order.
pub fn sequential_dot(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len());
    let mut sum = 0.0;
    for i in 0..p.len() {
        sum += p[i] * q[i];
    }
    sum
}

/// Result of one C 1.4.0 CG solve (`StochasticOptCG_Main` or
/// `StochasticOptPreconCG_DiagScale_Main`).
#[derive(Debug)]
pub struct CgSolution {
    /// Solution increment `x`, initialized to zero for every solve. After a
    /// failure it holds the iterate at the point where C returned.
    pub solution: Vec<f64>,
    /// C return value: the iteration count `iter` on convergence, or
    /// `-(iter+1)` when iteration `iter` broke down (non-finite or tiny
    /// `delta`, `d.q`, `alpha` or `beta`) and `-(max_iter+1)` when the
    /// iteration limit was reached without convergence.
    pub status: i64,
    /// C loop index `iter` where the solve stopped: `status` on success, the
    /// failing iteration otherwise (`max_iter` for non-convergence).
    pub iterations: usize,
    /// Final residual, including the periodic explicit recomputation.
    pub residual: Vec<f64>,
    /// Final search direction, retained for deterministic solver diagnostics.
    pub direction: Vec<f64>,
}

impl CgSolution {
    /// `true` when C reports convergence (`status >= 0`).
    pub fn converged(&self) -> bool {
        self.status >= 0
    }
}

type LocalCgProduct<'a> = dyn FnMut(
        &crate::sr_backend::CgSamples<'_>,
        &[f64],
        &mut [f64],
    ) -> Result<(), crate::stage_backend::StageError>
    + 'a;

/// Apply the covariance matrix from saved samples without materializing S.
/// Sample matrices have shape [active component, sample] in column-major order.
pub struct SampledSrOperator {
    /// Real part of the normalized component means.
    pub mean: Vec<f64>,
    /// Normalized component variances before diagonal regularization.
    pub diagonal: Vec<f64>,
    /// Raw real parts of sqrt(weight)*O for active components.
    pub real_samples: Vec<f64>,
    /// Raw imaginary parts; empty for a real parameter layout.
    pub imag_samples: Vec<f64>,
    components: usize,
    samples: usize,
    complex: bool,
}

/// C 1.4.0 breakdown guard for `d.q` and the old `delta` (stcopt_cg_impl.c:540,567).
const CG_TINY: f64 = 1.0e-20;
/// C `precon_diag_eps` of the DiagScale preconditioner (stcopt_cg_impl.c:363).
const PRECON_DIAG_EPS: f64 = 1.0e-12;

impl SampledSrOperator {
    /// Solve with C 1.4.0 `StochasticOptCG_Main`: residual-norm stop test,
    /// 20-iteration residual refresh and the breakdown/non-convergence status.
    pub fn solve(
        &mut self,
        gradient: &[f64],
        inv_weight: f64,
        shift: f64,
        tolerance: f64,
        max_iterations: usize,
    ) -> CgSolution {
        self.solve_variant(
            CgVariant::Plain,
            gradient,
            inv_weight,
            shift,
            tolerance,
            max_iterations,
        )
    }

    /// Serial solve with the selected C 1.4.0 CG variant.
    pub fn solve_variant(
        &mut self,
        variant: CgVariant,
        gradient: &[f64],
        inv_weight: f64,
        shift: f64,
        tolerance: f64,
        max_iterations: usize,
    ) -> CgSolution {
        self.solve_variant_with_reducer(
            variant,
            gradient,
            inv_weight,
            shift,
            tolerance,
            max_iterations,
            &crate::reducer::SingleProcessReducer,
        )
        .expect("serial CG communication cannot fail")
    }

    /// Plain CG using global sampled products.
    pub fn solve_with_reducer<R: Reducer + ?Sized>(
        &mut self,
        gradient: &[f64],
        inv_weight: f64,
        shift: f64,
        tolerance: f64,
        max_iterations: usize,
        reducer: &R,
    ) -> Result<CgSolution, String> {
        self.solve_variant_with_reducer(
            CgVariant::Plain,
            gradient,
            inv_weight,
            shift,
            tolerance,
            max_iterations,
            reducer,
        )
    }

    /// Solve using global sampled products, including periodic residual refresh.
    #[allow(clippy::too_many_arguments)]
    pub fn solve_variant_with_reducer<R: Reducer + ?Sized>(
        &mut self,
        variant: CgVariant,
        gradient: &[f64],
        inv_weight: f64,
        shift: f64,
        tolerance: f64,
        max_iterations: usize,
        reducer: &R,
    ) -> Result<CgSolution, String> {
        // One operand version per solve (issue #421) identifies its constant sampled
        // matrices. Shared backends are acquired for each product so collective waits
        // cannot block other walkers from accessing the same stage.
        let kind = crate::stage_backend::selected_stage_backend();
        let _team = crate::stage_backend::retain_cpu_team(kind).map_err(|e| e.to_string())?;
        let mut backend = crate::stage_backend::acquire_kind(kind);
        if backend.is_owned() {
            return self.solve_with_product(
                variant,
                &mut |samples, x, z| backend.sr().cg_local_product(samples, x, z),
                gradient,
                inv_weight,
                shift,
                tolerance,
                max_iterations,
                reducer,
            );
        }
        // A shared stage must not be held while another walker enters a
        // collective. Lock it only around the local product, retaining the
        // solve's operand version for the backend's cache.
        drop(backend);
        self.solve_with_product(
            variant,
            &mut |samples, x, z| {
                crate::stage_backend::acquire_kind(kind)
                    .sr()
                    .cg_local_product(samples, x, z)
            },
            gradient,
            inv_weight,
            shift,
            tolerance,
            max_iterations,
            reducer,
        )
    }

    /// Plain CG on an explicit SR stage implementation (the default `cg_step`
    /// and the validation of resident implementations use it).
    #[allow(clippy::too_many_arguments)]
    pub fn solve_with_stages<R: Reducer + ?Sized>(
        &mut self,
        stages: &mut dyn crate::sr_backend::SrStages,
        gradient: &[f64],
        inv_weight: f64,
        shift: f64,
        tolerance: f64,
        max_iterations: usize,
        reducer: &R,
    ) -> Result<CgSolution, String> {
        self.solve_with_product(
            CgVariant::Plain,
            &mut |samples, x, z| stages.cg_local_product(samples, x, z),
            gradient,
            inv_weight,
            shift,
            tolerance,
            max_iterations,
            reducer,
        )
    }

    /// C `StochasticOptCG_Main` (stcopt_cg_impl.c:484-591) and
    /// `StochasticOptPreconCG_DiagScale_Main` (:352-480), in C operation order.
    /// `x` starts at zero; DiagScale forms `r = g - S x` with an operator call.
    #[allow(clippy::too_many_arguments)]
    fn solve_with_product<R: Reducer + ?Sized>(
        &mut self,
        variant: CgVariant,
        local_product: &mut LocalCgProduct<'_>,
        gradient: &[f64],
        inv_weight: f64,
        shift: f64,
        tolerance: f64,
        max_iterations: usize,
        reducer: &R,
    ) -> Result<CgSolution, String> {
        let n = self.components;
        assert_eq!(gradient.len(), n);
        let version = crate::stage_backend::next_operand_version();
        let diag_scale = variant == CgVariant::DiagScale;
        // stcopt_cg_impl.c:492 evaluates these four factors left to right.
        let threshold = tolerance * tolerance * n as f64 * n as f64;
        // :362 precon_factor and the clamped preconditioner (:383-389, :450-456).
        let precon_factor = 1.0 + shift;
        let diagonal = self.diagonal.clone();
        let precondition = move |i: usize, r: f64| {
            let mut dq = precon_factor * diagonal[i];
            if !dq.is_finite() || dq.abs() < PRECON_DIAG_EPS {
                dq = if dq < 0.0 {
                    -PRECON_DIAG_EPS
                } else {
                    PRECON_DIAG_EPS
                };
            }
            r / dq
        };
        let mut solution = vec![0.0; n];
        let mut residual = gradient.to_vec();
        let mut product = vec![0.0; n];
        let mut direction;
        let mut delta;
        if diag_scale {
            // r = g - S x, d = M^-1 r, delta = r.d.
            self.apply_with_product(
                local_product,
                version,
                &mut residual,
                &mut solution,
                inv_weight,
                shift,
                reducer,
            )?;
            crate::threading::for_each_mut(&mut residual, 2, |i, value| {
                *value = gradient[i] - *value
            });
            direction = vec![0.0; n];
            crate::threading::for_each_mut(&mut direction, 2, |i, value| {
                *value = precondition(i, residual[i])
            });
            delta = sequential_dot(&residual, &direction);
        } else {
            direction = gradient.to_vec();
            delta = sequential_dot(&residual, &residual);
        }
        observe(|observer| {
            observer.iteration(CgIterationView {
                iteration: 0,
                solution: &solution,
                residual: &residual,
                direction: &direction,
                delta,
                alpha: None,
            })
        });
        let finish = |solution, status: i64, iterations, residual, direction| {
            let result = CgSolution {
                solution,
                status,
                iterations,
                residual,
                direction,
            };
            observe(|observer| observer.finished(&result));
            Ok(result)
        };
        if !delta.is_finite() {
            return finish(solution, -1, 0, residual, direction);
        }
        let mut iter = 0;
        while iter < max_iterations {
            let failed = -(iter as i64 + 1);
            if !delta.is_finite() {
                return finish(solution, failed, iter, residual, direction);
            }
            if delta.abs() < threshold {
                break;
            }
            // q = S d.
            self.apply_with_product(
                local_product,
                version,
                &mut product,
                &mut direction,
                inv_weight,
                shift,
                reducer,
            )?;
            let dq = sequential_dot(&direction, &product);
            if !dq.is_finite() || dq.abs() < CG_TINY {
                return finish(solution, failed, iter, residual, direction);
            }
            let alpha = delta / dq;
            if !alpha.is_finite() {
                return finish(solution, failed, iter, residual, direction);
            }
            // stcopt_cg_impl.c:545 `omp parallel for`: elementwise, one producer per entry.
            crate::threading::for_each_mut(&mut solution, 2, |i, value| {
                *value += alpha * direction[i]
            });
            if (iter + 1) % 20 == 0 {
                self.apply_with_product(
                    local_product,
                    version,
                    &mut residual,
                    &mut solution,
                    inv_weight,
                    shift,
                    reducer,
                )?;
                crate::threading::for_each_mut(&mut residual, 2, |i, value| {
                    *value = gradient[i] - *value
                });
            } else {
                crate::threading::for_each_mut(&mut residual, 2, |i, value| {
                    *value -= alpha * product[i]
                });
            }
            let delta_new = if diag_scale {
                // q is reused for M^-1 r.
                crate::threading::for_each_mut(&mut product, 2, |i, value| {
                    *value = precondition(i, residual[i])
                });
                sequential_dot(&residual, &product)
            } else {
                sequential_dot(&residual, &residual)
            };
            if !delta_new.is_finite() || delta.abs() < CG_TINY {
                return finish(solution, failed, iter, residual, direction);
            }
            let beta = delta_new / delta;
            if !beta.is_finite() {
                return finish(solution, failed, iter, residual, direction);
            }
            if diag_scale {
                delta = delta_new;
                crate::threading::for_each_mut(&mut direction, 2, |i, value| {
                    *value = product[i] + beta * *value
                });
            } else {
                // C:575 keeps delta = beta*delta (mathematically delta_new) to
                // preserve the legacy rounding; assigning delta_new would change
                // later alpha/stop decisions.
                delta *= beta; // IEEE multiplication commutes: C `beta*delta`
                crate::threading::for_each_mut(&mut direction, 2, |i, value| {
                    *value = residual[i] + beta * *value
                });
            }
            iter += 1;
            observe(|observer| {
                observer.iteration(CgIterationView {
                    iteration: iter,
                    solution: &solution,
                    residual: &residual,
                    direction: &direction,
                    delta,
                    alpha: Some(alpha),
                })
            });
        }
        // :583 (and :469): the limit was reached without meeting the threshold.
        if iter == max_iterations && delta.abs() >= threshold {
            let status = -(max_iterations as i64 + 1);
            return finish(solution, status, max_iterations, residual, direction);
        }
        finish(solution, iter as i64, iter, residual, direction)
    }

    /// Allocate the sampled operator and its reusable intermediate vectors.
    pub fn new(components: usize, samples: usize, complex: bool) -> Self {
        Self {
            mean: vec![0.0; components],
            diagonal: vec![0.0; components],
            real_samples: vec![0.0; components * samples],
            imag_samples: vec![0.0; if complex { components * samples } else { 0 }],
            components,
            samples,
            complex,
        }
    }

    /// Compute z = inv_weight*Re(O Oᴴ)*x - mean*(meanᵀx) + shift*diag*x.
    /// The sampled products use Julia's GEMV order; corrections use its
    /// sequential dot product. MPI reduction belongs before the corrections.
    pub fn apply(&mut self, z: &mut [f64], x: &[f64], inv_weight: f64, shift: f64) {
        let mut search = x.to_vec();
        self.apply_with_reducer(
            z,
            &mut search,
            inv_weight,
            shift,
            &crate::reducer::SingleProcessReducer,
        )
        .expect("serial CG communication cannot fail");
    }

    /// Broadcast the root search vector and sum local sampled products before
    /// applying global weight, mean and diagonal corrections (C operate_by_S).
    pub fn apply_with_reducer<R: Reducer + ?Sized>(
        &mut self,
        z: &mut [f64],
        x: &mut [f64],
        inv_weight: f64,
        shift: f64,
        reducer: &R,
    ) -> Result<(), String> {
        let kind = crate::stage_backend::selected_stage_backend();
        let mut backend = crate::stage_backend::acquire_kind(kind);
        let version = crate::stage_backend::next_operand_version();
        if backend.is_owned() {
            return self.apply_with_product(
                &mut |samples, x, z| backend.sr().cg_local_product(samples, x, z),
                version,
                z,
                x,
                inv_weight,
                shift,
                reducer,
            );
        }
        drop(backend);
        self.apply_with_product(
            &mut |samples, x, z| {
                crate::stage_backend::acquire_kind(kind)
                    .sr()
                    .cg_local_product(samples, x, z)
            },
            version,
            z,
            x,
            inv_weight,
            shift,
            reducer,
        )
    }

    /// [`Self::apply_with_reducer`] with an explicit local product and operand version.
    #[allow(clippy::too_many_arguments)]
    fn apply_with_product<R: Reducer + ?Sized>(
        &mut self,
        product: &mut LocalCgProduct<'_>,
        version: u64,
        z: &mut [f64],
        x: &mut [f64],
        inv_weight: f64,
        shift: f64,
        reducer: &R,
    ) -> Result<(), String> {
        let n = self.components;
        assert_eq!(z.len(), n);
        assert_eq!(x.len(), n);
        assert_eq!(self.mean.len(), n);
        assert_eq!(self.diagonal.len(), n);
        assert_eq!(self.real_samples.len(), n * self.samples);
        if self.complex {
            assert_eq!(self.imag_samples.len(), n * self.samples);
        }
        if n == 0 {
            return Ok(());
        }
        reducer.broadcast_f64(0, x)?;
        observe(|observer| observer.product(CgProductPhase::RootSearch, x, &[]));
        if self.samples == 0 {
            z.fill(0.0);
        } else {
            product(
                &crate::sr_backend::CgSamples {
                    real: &self.real_samples,
                    imag: &self.imag_samples,
                    components: n,
                    samples: self.samples,
                    version,
                },
                x,
                z,
            )
            .map_err(|e| e.to_string())?;
        }
        observe(|observer| observer.product(CgProductPhase::Local, x, z));
        reducer.barrier();
        reducer.allreduce_sum_f64(z);
        observe(|observer| observer.product(CgProductPhase::Global, x, z));
        let coef = sequential_dot(&self.mean, x);
        let (mean, diagonal) = (&self.mean, &self.diagonal);
        let x = &*x;
        crate::threading::for_each_mut(z, 4, |i, value| {
            *value = inv_weight * *value - coef * mean[i] + shift * diagonal[i] * x[i]
        });
        observe(|observer| observer.product(CgProductPhase::Corrected, x, z));
        Ok(())
    }
}

#[cfg(test)]
mod collective_tests {
    use super::*;
    use std::cell::RefCell;

    struct UnimplementedMultiRank;

    impl Reducer for UnimplementedMultiRank {
        fn world_size(&self) -> usize {
            2
        }
        fn allreduce_sum_f64(&self, _: &mut [f64]) {
            panic!("failed broadcast must not enter product reduction");
        }
        fn allreduce_sum_c64(&self, _: &mut [num_complex::Complex64]) {
            panic!("unexpected complex reduction");
        }
        fn allreduce_sum_i64(&self, _: &mut [i64]) {
            panic!("unexpected counter reduction");
        }
    }

    #[test]
    fn missing_multi_rank_broadcast_fails_before_search_product_or_sample_mutation() {
        let mut operator = SampledSrOperator::new(2, 2, false);
        operator.real_samples.copy_from_slice(&[1.0, 3.0, 2.0, 4.0]);
        let samples = operator.real_samples.clone();
        let mut search = [0.5, -1.0];
        let mut product = [123.0, 456.0];
        let error = operator
            .apply_with_reducer(
                &mut product,
                &mut search,
                0.25,
                0.1,
                &UnimplementedMultiRank,
            )
            .unwrap_err();
        assert!(error.contains("real vector broadcast"));
        assert_eq!(search, [0.5, -1.0]);
        assert_eq!(product, [123.0, 456.0]);
        assert_eq!(operator.real_samples, samples);
    }

    #[derive(Default)]
    struct ActualObserver {
        products: RefCell<Vec<(CgProductPhase, Vec<f64>)>>,
        iterations: RefCell<Vec<usize>>,
    }

    impl CgObserver for ActualObserver {
        fn product(&self, phase: CgProductPhase, _: &[f64], product: &[f64]) {
            self.products.borrow_mut().push((phase, product.to_vec()));
        }
        fn iteration(&self, state: CgIterationView<'_>) {
            assert_eq!(state.delta, sequential_dot(state.residual, state.residual));
            self.iterations.borrow_mut().push(state.iteration);
        }
    }

    #[test]
    fn actual_cg_observer_records_boundaries_without_changing_the_solve() {
        let mut operator = SampledSrOperator::new(2, 2, false);
        operator.real_samples.copy_from_slice(&[2.0, 1.0, 0.0, 1.0]);
        let baseline = operator.solve(&[6.0, 4.0], 1.0, 0.0, 1e-14, 10);
        let observer = Rc::new(ActualObserver::default());
        let guard = install_cg_observer(observer.clone()).unwrap();
        let observed = operator.solve(&[6.0, 4.0], 1.0, 0.0, 1e-14, 10);
        drop(guard);
        assert_eq!(observed.solution, baseline.solution);
        assert_eq!(observed.residual, baseline.residual);
        assert_eq!(observed.direction, baseline.direction);
        assert_eq!(observed.iterations, baseline.iterations);
        assert_eq!(*observer.iterations.borrow(), [0, 1, 2]);
        let products = observer.products.borrow();
        assert_eq!(products[1], (CgProductPhase::Local, vec![32.0, 20.0]));
        assert_eq!(products.len() % 4, 0);
        for chunk in products.as_chunks::<4>().0 {
            assert_eq!(
                chunk.iter().map(|entry| entry.0).collect::<Vec<_>>(),
                [
                    CgProductPhase::RootSearch,
                    CgProductPhase::Local,
                    CgProductPhase::Global,
                    CgProductPhase::Corrected
                ]
            );
        }
    }

    #[test]
    fn nested_observer_is_rejected_without_replacing_outer_and_threads_are_isolated() {
        let outer = Rc::new(ActualObserver::default());
        let guard = install_cg_observer(outer.clone()).unwrap();
        assert!(install_cg_observer(Rc::new(ActualObserver::default()))
            .err()
            .unwrap()
            .contains("already installed"));
        std::thread::spawn(|| {
            let inner = install_cg_observer(Rc::new(ActualObserver::default())).unwrap();
            drop(inner);
            assert!(CG_OBSERVER.with(|slot| slot.borrow().is_none()));
        })
        .join()
        .unwrap();
        let mut operator = SampledSrOperator::new(1, 1, false);
        operator.real_samples[0] = 2.0;
        operator.solve(&[4.0], 1.0, 0.0, 1e-14, 2);
        assert_eq!(*outer.iterations.borrow(), [0, 1]);
        drop(guard);
        assert!(CG_OBSERVER.with(|slot| slot.borrow().is_none()));
    }

    #[test]
    fn observer_guard_cleans_up_during_panic_unwind() {
        let panic = std::panic::catch_unwind(|| {
            let _guard = install_cg_observer(Rc::new(ActualObserver::default())).unwrap();
            panic!("observer client panicked");
        });
        assert!(panic.is_err());
        assert!(CG_OBSERVER.with(|slot| slot.borrow().is_none()));
        let guard = install_cg_observer(Rc::new(ActualObserver::default())).unwrap();
        drop(guard);
    }

    struct RemoteSamples {
        events: RefCell<Vec<&'static str>>,
        expected_local: Vec<f64>,
        remote: Vec<f64>,
    }

    impl Reducer for RemoteSamples {
        fn broadcast_f64(&self, root: usize, values: &mut [f64]) -> Result<(), String> {
            assert_eq!(root, 0);
            self.events.borrow_mut().push("root search vector");
            for (value, root_value) in values.iter_mut().zip([0.5, -1.0]) {
                *value = root_value;
            }
            Ok(())
        }

        fn barrier(&self) {
            self.events.borrow_mut().push("global barrier");
        }

        fn allreduce_sum_f64(&self, values: &mut [f64]) {
            self.events.borrow_mut().push("raw sampled product");
            assert_eq!(values, self.expected_local);
            for (value, remote) in values.iter_mut().zip(&self.remote) {
                *value += remote;
            }
        }

        fn allreduce_sum_c64(&self, _: &mut [num_complex::Complex64]) {
            panic!("CG sampled product is real");
        }

        fn allreduce_sum_i64(&self, _: &mut [i64]) {
            panic!("unexpected counter reduction");
        }
    }

    #[test]
    fn sampled_product_is_globally_summed_before_weight_mean_and_shift() {
        let mut operator = SampledSrOperator::new(2, 2, false);
        operator.real_samples.copy_from_slice(&[1.0, 3.0, 2.0, 4.0]);
        operator.mean.copy_from_slice(&[0.25, -0.5]);
        operator.diagonal.copy_from_slice(&[2.0, 3.0]);
        let reducer = RemoteSamples {
            events: RefCell::new(Vec::new()),
            expected_local: vec![-8.5, -19.5],
            // Independent other-rank sample [3,-1] has dot(x,O)=2.5.
            remote: vec![7.5, -2.5],
        };
        let mut product = [0.0; 2];
        let mut search = [99.0, 99.0];
        operator
            .apply_with_reducer(&mut product, &mut search, 0.25, 0.1, &reducer)
            .unwrap();
        assert_eq!(search, [0.5, -1.0]);
        for (actual, expected) in product.into_iter().zip([-0.30625, -5.4875]) {
            assert!((actual - expected).abs() < 1e-14);
        }
        assert_eq!(
            *reducer.events.borrow(),
            [
                "root search vector",
                "global barrier",
                "raw sampled product"
            ]
        );
    }

    #[test]
    fn empty_local_sample_partition_still_contributes_to_global_operator() {
        let mut operator = SampledSrOperator::new(2, 0, true);
        let reducer = RemoteSamples {
            events: RefCell::new(Vec::new()),
            expected_local: vec![0.0, 0.0],
            remote: vec![7.5, -2.5],
        };
        let mut product = [0.0; 2];
        operator
            .apply_with_reducer(&mut product, &mut [99.0, 99.0], 0.25, 0.0, &reducer)
            .unwrap();
        assert_eq!(product, [1.875, -0.625]);
        assert_eq!(reducer.events.borrow().len(), 3);
    }
}

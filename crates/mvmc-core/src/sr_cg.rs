//! Sampled matrix operator for Julia's standard stochastic-reconfiguration CG.

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

/// Apply Julia's standard sampled SR-CG step, writing diagnostics before updates.
/// Returns zero for finite increments, including iteration-limit/breakdown exits.
pub fn stochastic_opt_cg(
    data: &mut ExpertModeData,
    state: &VmcOptimizationState,
    files: Option<&mut RunFiles>,
) -> io::Result<i32> {
    stochastic_opt_cg_with_reducer(data, state, files, &crate::reducer::SingleProcessReducer)
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
    let size = offset * state.sr_opt.sr_opt_size;
    let oo: Vec<f64> = if complex {
        state.sr_opt.sr_opt_oo.iter().map(|z| z.re).collect()
    } else {
        state.sr_opt.sr_opt_oo_real.clone()
    };
    let ho: Vec<f64> = if complex {
        state.sr_opt.sr_opt_ho.iter().map(|z| z.re).collect()
    } else {
        state.sr_opt.sr_opt_ho_real.clone()
    };
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
    if mapping.is_empty() {
        return Ok(0);
    }
    let samples = data.modpara.nvmc_sample.max(0) as usize;
    let mut operator = SampledSrOperator::new(mapping.len(), samples, complex);
    let mut gradient = vec![0.0; mapping.len()];
    let dt = 2.0 * data.modpara.dsr_opt_step_dt;
    // stcopt_cg_impl.c:471-492 `omp parallel for` over the active components `si`.
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
        // Sample `s` owns the contiguous window `s * n_active ..` of the sample matrices.
        let fill = |s: usize, real: &mut [f64], imag: &mut [f64]| {
            for (si, &pi) in mapping.iter().enumerate() {
                let src = s * size + pi + offset;
                if complex {
                    if let Some(o) = state.sr_opt.sr_opt_o_store.get(src) {
                        real[si] = o.re;
                        imag[si] = o.im;
                    }
                } else if let Some(&o) = state.sr_opt.sr_opt_o_store_real.get(src) {
                    real[si] = o;
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
    let result = operator
        .solve_with_reducer(
            &gradient,
            1.0 / state.energy.wc.re,
            data.modpara.dsr_opt_sta_del,
            data.modpara.dsr_opt_cg_tol,
            max_iterations,
            reducer,
        )
        .map_err(io::Error::other)?;
    let info = i32::from(result.solution.iter().any(|x| !x.is_finite()));
    if let Some(files) = files {
        let mut imax = 0;
        for i in 1..result.solution.len() {
            if result.solution[imax].abs() < result.solution[i].abs() {
                imax = i;
            }
        }
        // C stcopt_cg_impl.c:202 prints the global NPara, not OFFSET*NPara.
        files.write_sr_info(&SrInfoRow {
            n_para: n_para as i64,
            n_smat: mapping.len() as i64,
            opt_num: opt_cut,
            cut_num: diag_cut,
            s_diag_max: maximum,
            s_diag_min: minimum,
            r_max: result.solution[imax],
            i_max: mapping[imax] as i64,
            cg_info: Some(result.iterations as i64),
        })?;
    }
    if info == 0 {
        for (&pi, &x) in mapping.iter().zip(&result.solution) {
            let para = if complex { pi / 2 } else { pi };
            let (re, im) = if complex && pi % 2 != 0 {
                (0.0, x)
            } else {
                (x, 0.0)
            };
            crate::sr::update_parameter_value(data, para, re, im, n_proj);
        }
    }
    Ok(info)
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

/// Result of the standard CG iteration loop (not a convergence status).
#[derive(Debug)]
pub struct CgSolution {
    /// Solution increment, initialized to zero for every solve.
    pub solution: Vec<f64>,
    /// Julia iteration count, including the iteration that detects breakdown.
    pub iterations: usize,
    /// Final residual, including Julia's periodic explicit recomputation.
    pub residual: Vec<f64>,
    /// Final search direction, retained for deterministic solver diagnostics.
    pub direction: Vec<f64>,
}

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
    y_real: Vec<f64>,
    y_imag: Vec<f64>,
}

impl SampledSrOperator {
    /// Solve with Julia's convergence threshold and 20-iteration residual refresh.
    /// A small search-direction product ends the loop without an error, as in
    /// Julia. The caller must check the returned increments for finiteness.
    pub fn solve(
        &mut self,
        gradient: &[f64],
        inv_weight: f64,
        shift: f64,
        tolerance: f64,
        max_iterations: usize,
    ) -> CgSolution {
        self.solve_with_reducer(
            gradient,
            inv_weight,
            shift,
            tolerance,
            max_iterations,
            &crate::reducer::SingleProcessReducer,
        )
        .expect("serial CG communication cannot fail")
    }

    /// Solve using global sampled products, including periodic residual refresh.
    pub fn solve_with_reducer<R: Reducer + ?Sized>(
        &mut self,
        gradient: &[f64],
        inv_weight: f64,
        shift: f64,
        tolerance: f64,
        max_iterations: usize,
        reducer: &R,
    ) -> Result<CgSolution, String> {
        let n = self.components;
        assert_eq!(gradient.len(), n);
        // stcopt_cg_impl.c:265 evaluates these four factors left to right.
        let threshold = tolerance * tolerance * n as f64 * n as f64;
        let mut solution = vec![0.0; n];
        let mut direction = gradient.to_vec();
        let mut residual = gradient.to_vec();
        let mut product = vec![0.0; n];
        let mut delta = sequential_dot(&residual, &residual);
        let mut iterations = 0;
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
        for iteration in 1..=max_iterations {
            iterations = iteration;
            if delta < threshold {
                iterations = iteration - 1;
                break;
            }
            self.apply_with_reducer(&mut product, &mut direction, inv_weight, shift, reducer)?;
            let dq = sequential_dot(&direction, &product);
            let alpha = delta / dq;
            // stcopt_cg_impl.c:313 `omp parallel for`: elementwise, one producer per entry.
            crate::threading::for_each_mut(&mut solution, 2, |i, value| {
                *value += alpha * direction[i]
            });
            if iteration % 20 == 0 {
                self.apply_with_reducer(&mut residual, &mut solution, inv_weight, shift, reducer)?;
                crate::threading::for_each_mut(&mut residual, 2, |i, value| {
                    *value = gradient[i] - *value
                });
            } else {
                crate::threading::for_each_mut(&mut residual, 2, |i, value| {
                    *value -= alpha * product[i]
                });
            }
            let delta_new = sequential_dot(&residual, &residual);
            let beta = delta_new / delta;
            // C:336 rounds the quotient and then multiplies it by the old
            // norm. Assigning delta_new changes subsequent alpha/stop tests.
            let recurrent_norm = beta * delta;
            delta = recurrent_norm;
            crate::threading::for_each_mut(&mut direction, 2, |i, value| {
                *value = residual[i] + beta * *value
            });
            observe(|observer| {
                observer.iteration(CgIterationView {
                    iteration,
                    solution: &solution,
                    residual: &residual,
                    direction: &direction,
                    delta,
                    alpha: Some(alpha),
                })
            });
        }
        let result = CgSolution {
            solution,
            iterations,
            residual,
            direction,
        };
        observe(|observer| observer.finished(&result));
        Ok(result)
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
            y_real: vec![0.0; samples],
            y_imag: vec![0.0; if complex { samples } else { 0 }],
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
            crate::serial_blas::initialize();
            let rows = i32::try_from(n).expect("CG component count must fit BLAS LP64");
            let cols = i32::try_from(self.samples).expect("CG sample count must fit BLAS LP64");
            // SAFETY: matrix buffers have exactly rows*cols entries, leading
            // dimensions are rows, and every input/output vector has the
            // required length. Mutable outputs never alias matrix/input views.
            unsafe {
                blas::dgemv(
                    b'T',
                    rows,
                    cols,
                    1.0,
                    &self.real_samples,
                    rows,
                    x,
                    1,
                    0.0,
                    &mut self.y_real,
                    1,
                );
                if self.complex {
                    blas::dgemv(
                        b'T',
                        rows,
                        cols,
                        1.0,
                        &self.imag_samples,
                        rows,
                        x,
                        1,
                        0.0,
                        &mut self.y_imag,
                        1,
                    );
                }
                blas::dgemv(
                    b'N',
                    rows,
                    cols,
                    1.0,
                    &self.real_samples,
                    rows,
                    &self.y_real,
                    1,
                    0.0,
                    z,
                    1,
                );
                if self.complex {
                    blas::dgemv(
                        b'N',
                        rows,
                        cols,
                        1.0,
                        &self.imag_samples,
                        rows,
                        &self.y_imag,
                        1,
                        1.0,
                        z,
                        1,
                    );
                }
            }
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

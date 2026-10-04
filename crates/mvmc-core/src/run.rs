//! Top-level entry points.
//!
//! Port targets:
//! * `vmc_para_opt.jl` -> `vmc_para_opt`
//! * `vmc_phys_cal.jl` -> `vmc_phys_cal`
//! * `run_para_opt_from_namelist.jl` -> `run_para_opt_from_namelist`
//! * `initial_params.jl` -> `read_initial_def`
//!
//! Preserve the deterministic initialization phase order documented in
//! `run_para_opt_from_namelist.jl:65-100`:
//!     init_gen_rand -> InitParameter -> ReadInitParameter
//!         -> ReadInputParameters -> SyncModifiedParameter -> InitQPWeight.

#[cfg(test)]
#[path = "../../../tests/support/reference_slater.rs"]
mod reference_slater;

#[cfg(test)]
#[path = "issue179_native_same_input.rs"]
mod issue179_native_same_input;

#[cfg(test)]
#[path = "../../../tests/support/native_fsz_fixture.rs"]
mod native_fsz_fixture;

#[cfg(test)]
use std::fs;
use std::path::Path;

#[cfg(test)]
use mvmc_expert_parsers::parse_expert_mode_files;
use mvmc_expert_parsers::utils::parameter_init::{init_parameter, n_slater};
use mvmc_expert_parsers::utils::qp_weight::init_qp_weight;
use mvmc_expert_parsers::utils::read_input_parameters::read_input_parameters;
use mvmc_expert_parsers::ExpertModeData;
use num_complex::Complex64;
use sfmt19937::Sfmt19937Rng;

use crate::average::{weight_average_sr_opt, weight_average_sr_opt_real, weight_average_we};
use crate::c_timer::{CTimer, TimerEnv};
use crate::counter::reduce_counter;
use crate::initial_params::{read_initial_def, read_opt_para_file};
use crate::io::{output_data, output_opt_data, store_opt_data};
use crate::observables::clear_phys_quantity;
use crate::reducer::{Reducer, SingleProcessReducer};
use crate::slater_update::{update_slater_elm, update_slater_elm_fsz};
use crate::state::{ThreadedPfaPackWorkspace, VmcOptimizationState};
use crate::sync::sync_modified_parameter as sync_modified;
use crate::sync::sync_modified_parameter_local as sync_modified_parameter;

#[cfg(all(test, feature = "mpi"))]
#[path = "run_mpi_tests.rs"]
mod mpi_runtime_tests;

#[cfg(test)]
#[path = "issue180_boundary_dev.rs"]
mod issue180_boundary_dev;

// Every rank must reach this boundary even when its local operation failed.
fn collective_result<T, R: Reducer + ?Sized>(
    result: Result<T, String>,
    reducer: &R,
    operation: &str,
) -> Result<T, String> {
    if reducer.any_failure(result.is_err()) {
        return Err(result
            .err()
            .unwrap_or_else(|| format!("{operation} failed on another MPI rank")));
    }
    result
}

/// C-compatible parser default when `RndSeed` is omitted. Zero is a
/// valid seed, and negative ModPara seeds request the current Unix time.
pub const FALLBACK_SEED: i64 = 11272;

/// Actual optimization walker immediately before its SR contribution is added.
/// Values are local, unaveraged and borrowed; this is not a sampler replay.
pub struct OptimizationMeasurementView<'a> {
    /// Current parameters and ordered input descriptors.
    pub data: &'a ExpertModeData,
    /// Actual walker matrix scratch and derivative vector before accumulation.
    pub state: &'a VmcOptimizationState,
    /// Zero-based saved configuration index within the current frame.
    pub sample: usize,
    /// Actual projected Pfaffian overlap.
    pub overlap: Complex64,
    /// Actual Hamiltonian evaluation used in the following accumulation.
    pub local_energy: Complex64,
    /// Actual unaveraged walker weight.
    pub weight: f64,
}

/// Read-only observation of overlap, energy, Pfaffians and `state.sr_opt.sr_opt_o`.
pub trait OptimizationMeasurementObserver {
    /// Opt into caller-thread Real factor observation for this saved walker.
    /// Default false preserves existing observers. A collector must bound frames.
    fn begin_real_factor(
        &self,
        _data: &ExpertModeData,
        _state: &VmcOptimizationState,
        _sample: usize,
    ) -> bool {
        false
    }
    /// Whether the collector's first-walker factor scope is currently active.
    fn real_factor_active(&self) -> bool {
        false
    }
    /// Borrow actual QP0 Real assembly/factor/PF/inverse storage boundaries.
    fn real_factor(&self, _view: RealFactorView<'_>) {}
    /// Borrow actual Step5 RHS and signed tridiagonal operands, never reconstructed.
    fn real_step5(&self, _view: pfapack::utu2::InverseStep5View<'_, f64>) {}
    /// Close the borrowed factor scope, including errors or panic unwinding.
    fn end_real_factor(&self) {}
    /// Borrow the production values without changing state or consuming RNG.
    fn measured(&self, view: OptimizationMeasurementView<'_>);
    /// Borrow the actual local O/HO/OO prefix immediately after its accumulation.
    fn accumulated(&self, _view: OptimizationMeasurementView<'_>) {}
    /// Test-only actual runner RNG boundary; no production API or live draws.
    #[cfg(test)]
    fn rng_boundary(&self, _phase: &'static str, _rng: &Sfmt19937Rng) {}
}

/// Actual Real QP boundary; matrix may include its existing public padding.
pub struct RealFactorView<'a> {
    /// Chronological boundary name, not a replayed operation.
    pub stage: &'static str,
    /// Actual QP index, restricted to zero by this diagnostic scope.
    pub qp: usize,
    /// Actual occupied-matrix dimension.
    pub dimension: usize,
    /// Actual column-major matrix slice; padding, if any, is not synthesized.
    pub matrix: &'a [f64],
    /// Actual one-based pivot swap targets, not a permutation.
    pub pivots: &'a [pfapack::PivotIndex1Based],
    /// Actual factor error index, if factorization failed.
    pub factor_error: Option<usize>,
    /// Actual PF when already evaluated by the original kernel.
    pub pf: Option<f64>,
}

struct FirstRealFactorScope(std::rc::Rc<dyn OptimizationMeasurementObserver>);
impl Drop for FirstRealFactorScope {
    fn drop(&mut self) {
        self.0.end_real_factor();
    }
}

fn begin_first_real_factor(
    data: &ExpertModeData,
    state: &VmcOptimizationState,
    sample: usize,
) -> Option<FirstRealFactorScope> {
    if sample != 0 || data.modpara.vmc_calc_mode != 0 {
        return None;
    }
    OPTIMIZATION_MEASUREMENT_OBSERVER.with(|slot| {
        let observer = slot.borrow().clone()?;
        observer
            .begin_real_factor(data, state, sample)
            .then(|| FirstRealFactorScope(observer))
    })
}

pub(crate) fn first_real_factor_observer(
    qp: usize,
) -> Option<std::rc::Rc<dyn OptimizationMeasurementObserver>> {
    if qp != 0 {
        return None;
    }
    OPTIMIZATION_MEASUREMENT_OBSERVER.with(|slot| {
        let observer = slot.borrow().clone()?;
        observer.real_factor_active().then_some(observer)
    })
}

thread_local! {
    static OPTIMIZATION_MEASUREMENT_OBSERVER: std::cell::RefCell<Option<std::rc::Rc<dyn OptimizationMeasurementObserver>>> = const { std::cell::RefCell::new(None) };
}

/// Calling-thread scope; nested installations fail without replacing the owner.
pub struct OptimizationMeasurementObserverGuard {
    _thread_bound: std::marker::PhantomData<std::rc::Rc<()>>,
}

/// Install on the calling thread; disabled observation allocates no buffers.
pub fn install_optimization_measurement_observer(
    observer: std::rc::Rc<dyn OptimizationMeasurementObserver>,
) -> Result<OptimizationMeasurementObserverGuard, &'static str> {
    OPTIMIZATION_MEASUREMENT_OBSERVER.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_some() {
            return Err("Optimization measurement observer already active on this thread");
        }
        *slot = Some(observer);
        Ok(OptimizationMeasurementObserverGuard {
            _thread_bound: std::marker::PhantomData,
        })
    })
}

impl Drop for OptimizationMeasurementObserverGuard {
    fn drop(&mut self) {
        OPTIMIZATION_MEASUREMENT_OBSERVER.with(|slot| *slot.borrow_mut() = None);
    }
}

fn observe_optimization_measurement(view: OptimizationMeasurementView<'_>) {
    if view.data.modpara.vmc_calc_mode != 0 {
        return;
    }
    OPTIMIZATION_MEASUREMENT_OBSERVER.with(|slot| {
        // No allocation/copy when disabled. Release TLS borrow before callback.
        let observer = slot.borrow().clone();
        if let Some(observer) = observer {
            observer.measured(view);
        }
    });
}

#[cfg(test)]
fn observe_optimization_rng(phase: &'static str, rng: &Sfmt19937Rng) {
    OPTIMIZATION_MEASUREMENT_OBSERVER.with(|slot| {
        let observer = slot.borrow().clone();
        if let Some(observer) = observer {
            observer.rng_boundary(phase, rng);
        }
    });
}

/// Borrowed local PhysCal accumulation immediately before Green normalization.
/// This is not a globally reduced mean; MPI callers observe each rank separately.
pub struct PhysCalGreenView<'a> {
    /// Actual fixed-parameter input, including the ordered Green descriptors.
    pub data: &'a ExpertModeData,
    /// Zero-based completed measurement-frame index.
    pub sample: usize,
    /// Whether the General/FSZ measurement path was used.
    pub use_fsz: bool,
    /// Actual local accumulated correlation-sampling weight before averaging.
    pub weight: Complex64,
    /// Raw ordered one-body sums, before division by the weight.
    pub one_body: &'a [Complex64],
    /// Raw factored two-body sums, in descriptor order.
    pub factored_two_body: &'a [Complex64],
    /// Raw direct two-body sums, in descriptor order.
    pub direct_two_body: &'a [Complex64],
}

/// Read-only diagnostic observation of the actual production accumulation.
pub trait PhysCalGreenObserver {
    /// Actual initialization boundary; diagnostic only, with a borrowed RNG.
    /// Fixed-loaded/overlaid/synchronized precede seeding; initialized-clone
    /// describes the real initialization scratch copy, not the fixed parameters.
    fn lifecycle(
        &self,
        _stage: &'static str,
        _data: &ExpertModeData,
        _rng: Option<&Sfmt19937Rng>,
        _n_para_consumed: Option<usize>,
    ) {
    }
    /// Actual saved chain and nonconsuming RNG immediately after sampling,
    /// before measurement changes matrix/configuration scratch. No replay.
    fn sample_completed(
        &self,
        _data: &ExpertModeData,
        _sample: usize,
        _state: &VmcOptimizationState,
        _rng: &Sfmt19937Rng,
    ) {
    }
    /// Observe the borrowed actual buffers without mutating runner state or RNG.
    fn accumulated(&self, view: PhysCalGreenView<'_>);
}

thread_local! {
    static PHYSCAL_GREEN_OBSERVER: std::cell::RefCell<Option<std::rc::Rc<dyn PhysCalGreenObserver>>> = const { std::cell::RefCell::new(None) };
    static PHYSCAL_GREEN_SAMPLE: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// Thread-bound scope; dropping it also clears observation during unwinding.
pub struct PhysCalGreenObserverGuard {
    _thread_bound: std::marker::PhantomData<std::rc::Rc<()>>,
}

/// Install an observer on the calling thread. Nested scopes are rejected.
/// Disabled observation does not allocate or copy numerical buffers.
pub fn install_physcal_green_observer(
    observer: std::rc::Rc<dyn PhysCalGreenObserver>,
) -> Result<PhysCalGreenObserverGuard, &'static str> {
    PHYSCAL_GREEN_OBSERVER.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_some() {
            return Err("PhysCal Green observer already active on this thread");
        }
        *slot = Some(observer);
        Ok(PhysCalGreenObserverGuard {
            _thread_bound: std::marker::PhantomData,
        })
    })
}

impl Drop for PhysCalGreenObserverGuard {
    fn drop(&mut self) {
        PHYSCAL_GREEN_OBSERVER.with(|slot| *slot.borrow_mut() = None);
    }
}

fn with_physcal_green_sample<T>(sample: usize, operation: impl FnOnce() -> T) -> T {
    struct Restore(Option<usize>);
    impl Drop for Restore {
        fn drop(&mut self) {
            PHYSCAL_GREEN_SAMPLE.with(|slot| slot.set(self.0));
        }
    }
    let _restore = Restore(PHYSCAL_GREEN_SAMPLE.with(|slot| slot.replace(Some(sample))));
    operation()
}

fn observe_physcal_lifecycle(
    stage: &'static str,
    data: &ExpertModeData,
    rng: Option<&Sfmt19937Rng>,
    n_para_consumed: Option<usize>,
) {
    PHYSCAL_GREEN_OBSERVER.with(|slot| {
        let observer = slot.borrow().clone();
        if let Some(observer) = observer {
            observer.lifecycle(stage, data, rng, n_para_consumed);
        }
    });
}

fn scale_physcal_lanczos(value: &mut Complex64, inverse_weight: Complex64, all_complex: bool) {
    if all_complex {
        *value *= inverse_weight;
    } else {
        // C uses a double array; the complex shadow has no imaginary observable.
        *value = Complex64::new(value.re * inverse_weight.re, 0.0);
    }
}

fn normalize_physcal_green(state: &mut VmcOptimizationState, use_fsz: bool, all_complex: bool) {
    if state.energy.wc.re != 0.0 {
        let inverse = crate::c_complex::divide(Complex64::new(1.0, 0.0), state.energy.wc);
        if let Some(phys) = state.phys_quantities.as_mut() {
            for values in [
                &mut phys.phys_lanczos_qqqq,
                &mut phys.phys_lanczos_qcisajsq,
                &mut phys.phys_lanczos_qcisajscktaltq,
                &mut phys.phys_lanczos_qcisajscktaltq_dc,
            ] {
                for value in values {
                    scale_physcal_lanczos(value, inverse, all_complex);
                }
            }
            if !use_fsz {
                for values in [
                    &mut phys.phys_cis_ajs,
                    &mut phys.phys_cis_ajs_ckt_alt,
                    &mut phys.phys_cis_ajs_ckt_alt_dc,
                ] {
                    for value in values {
                        *value *= inverse;
                    }
                }
            }
        }
    }
    if use_fsz {
        crate::observables::weight_average_green_func_fsz(state);
    }
}

#[cfg(test)]
mod physcal_normalization_literal_tests {
    use super::*;
    fn close(actual: f64, expected: f64) {
        assert!(actual.is_finite() && expected.is_finite());
        if expected == 0.0 {
            assert_eq!(actual, 0.0);
            return;
        }
        // For these nine positive-real-weight literals only, quotient scales,
        // squares/sums and numerator are exact; reciprocal division and final
        // component product are the two rounding operations. Not generic complex
        // division. |R-C| <= 2gamma2/(1-gamma2)*|C|, since C itself is rounded.
        // Evaluate a conservative upper endpoint rather than rounding it inward.
        fn up(x: f64) -> f64 {
            assert!(x > 0.0 && x.is_finite());
            f64::from_bits(x.to_bits() + 1)
        }
        fn down(x: f64) -> f64 {
            assert!(x > 0.0 && x.is_finite());
            f64::from_bits(x.to_bits() - 1)
        }
        let u = f64::EPSILON / 2.0;
        let gamma2 = up(2.0 * u / (1.0 - 2.0 * u));
        let factor = up((2.0 * gamma2) / down(1.0 - gamma2));
        let bound = up(factor * expected.abs());
        assert!((actual - expected).abs() <= bound);
    }
    #[test]
    fn native_literals_cover_production_all_seven_planes_and_zero_guard() {
        let literals = include_str!("../tests/fixtures/normalization_c_literals.tsv");
        let mut lines = literals.lines();
        assert_eq!(
            lines.next(),
            Some("weight_index\tvalue_index\treal_bits\tcomplex_re_bits\tcomplex_im_bits")
        );
        let rows: Vec<_> = lines.collect();
        assert_eq!(rows.len(), 9, "exact Cartesian fixture required");
        let mut seen = [[false; 3]; 3];
        let weights = [3.0, 10.0, 100.0];
        let inputs = [
            Complex64::new(0.3, -0.7),
            Complex64::new(-0.7, 0.3),
            Complex64::new(13.486979846002933, 0.0),
        ];
        for line in rows {
            let cols: Vec<_> = line.split('\t').collect();
            assert_eq!(cols.len(), 5);
            let wi: usize = cols[0].parse().unwrap();
            let vi: usize = cols[1].parse().unwrap();
            assert!(wi < 3 && vi < 3, "fixture index out of range");
            assert!(!seen[wi][vi], "duplicate fixture key");
            seen[wi][vi] = true;
            for text in &cols[2..] {
                assert_eq!(text.len(), 16, "exact binary64 hex width");
                assert!(text.bytes().all(|c| c.is_ascii_hexdigit()));
            }
            let real = f64::from_bits(u64::from_str_radix(cols[2], 16).unwrap());
            let expected = Complex64::new(
                f64::from_bits(u64::from_str_radix(cols[3], 16).unwrap()),
                f64::from_bits(u64::from_str_radix(cols[4], 16).unwrap()),
            );
            assert!(real.is_finite() && expected.re.is_finite() && expected.im.is_finite());
            for fsz in [false, true] {
                for complex in [false, true] {
                    for mode in 0..=2 {
                        let mut state = VmcOptimizationState::zeros(2, 1, 0, 1, 1, 1, complex, fsz);
                        let mut p = crate::state::PhysicalQuantities::zeros(1, 1, 1);
                        for values in [
                            &mut p.phys_lanczos_qqqq,
                            &mut p.phys_lanczos_qcisajsq,
                            &mut p.phys_lanczos_qcisajscktaltq,
                            &mut p.phys_lanczos_qcisajscktaltq_dc,
                        ] {
                            values.fill(if mode == 0 {
                                Complex64::new(0.0, 0.0)
                            } else {
                                Complex64::new(
                                    inputs[vi].re,
                                    if complex { inputs[vi].im } else { 0.0 },
                                )
                            });
                        }
                        for values in [
                            &mut p.phys_cis_ajs,
                            &mut p.phys_cis_ajs_ckt_alt,
                            &mut p.phys_cis_ajs_ckt_alt_dc,
                        ] {
                            values.fill(inputs[vi]);
                        }
                        state.phys_quantities = Some(p);
                        state.energy.wc = Complex64::new(weights[wi], 0.0);
                        normalize_physcal_green(&mut state, fsz, complex);
                        let p = state.phys_quantities.as_ref().unwrap();
                        for values in [
                            &p.phys_lanczos_qqqq,
                            &p.phys_lanczos_qcisajsq,
                            &p.phys_lanczos_qcisajscktaltq,
                            &p.phys_lanczos_qcisajscktaltq_dc,
                        ] {
                            for v in values {
                                close(v.re, if mode == 0 { 0.0 } else { real });
                                close(
                                    v.im,
                                    if mode == 0 || !complex {
                                        0.0
                                    } else {
                                        expected.im
                                    },
                                );
                            }
                        }
                        for values in [
                            &p.phys_cis_ajs,
                            &p.phys_cis_ajs_ckt_alt,
                            &p.phys_cis_ajs_ckt_alt_dc,
                        ] {
                            for v in values {
                                close(v.re, expected.re);
                                close(v.im, expected.im);
                            }
                        }
                        let before = state.phys_quantities.clone();
                        state.energy.wc = Complex64::new(0.0, 0.0);
                        normalize_physcal_green(&mut state, fsz, complex);
                        assert_eq!(before, state.phys_quantities);
                    }
                }
            }
        }
        assert!(seen.into_iter().flatten().all(|value| value));
    }
    #[test]
    fn production_real_scaling_preserves_reciprocal_then_multiply_order() {
        // Structural SAME-implementation operation-order contract, NOT
        // cross-language computed-bit acceptance nor independent golden.
        let mut distinct = false;
        for weight in [3.0, 10.0, 100.0] {
            for raw in [0.3, -0.7, 13.486979846002933] {
                let inverse =
                    crate::c_complex::divide(Complex64::new(1.0, 0.0), Complex64::new(weight, 0.0));
                let multiplied = raw * inverse.re;
                let divided = raw / weight;
                if multiplied != divided {
                    distinct = true;
                    let mut value = Complex64::new(raw, 0.0);
                    scale_physcal_lanczos(&mut value, inverse, false);
                    assert_eq!(value.re, multiplied);
                    assert_ne!(value.re, divided);
                }
            }
        }
        assert!(
            distinct,
            "fixed literal suite must distinguish operation orders"
        );
    }
}

fn observe_physcal_green(data: &ExpertModeData, state: &VmcOptimizationState, use_fsz: bool) {
    if data.modpara.vmc_calc_mode != 1 {
        return;
    }
    let Some(sample) = PHYSCAL_GREEN_SAMPLE.with(std::cell::Cell::get) else {
        return;
    };
    PHYSCAL_GREEN_OBSERVER.with(|slot| {
        // Clone only the observer handle, never the numerical data. Release the
        // TLS borrow before invoking user diagnostics so nested installs reject cleanly.
        let observer = slot.borrow().clone();
        if let (Some(observer), Some(phys)) = (observer, state.phys_quantities.as_ref()) {
            observer.accumulated(PhysCalGreenView {
                data,
                sample,
                use_fsz,
                weight: state.energy.wc,
                one_body: &phys.phys_cis_ajs,
                factored_two_body: &phys.phys_cis_ajs_ckt_alt,
                direct_two_body: &phys.phys_cis_ajs_ckt_alt_dc,
            });
        }
    });
}

/// Per-step callback; errors propagate to the caller like Julia exceptions.
/// Arguments are zero-based step, post-sync parameters, measured energy and status.
pub type StepCallback<'a> =
    dyn FnMut(usize, &mut ExpertModeData, Complex64, i32) -> Result<(), String> + 'a;

/// Per-sample PhysCal callback.
///
/// The callback receives the zero-based sample index, the fixed parameter data,
/// the post-average total energy, and a zero status. PhysCal invokes it on
/// every rank after that sample's reductions and output have completed. The
/// data is shared read-only so observation cannot change the fixed parameters,
/// QP weights, RNG trajectory, or subsequent output.
pub type PhysCalCallback<'a> =
    dyn FnMut(usize, &ExpertModeData, Complex64, i32) -> Result<(), String> + 'a;

/// Fixed-parameter PhysCal preparation, before the sampling loop owns the
/// internal initialization and QP-weight setup.
#[derive(Debug)]
pub struct PhysCalPreparation {
    /// Parsed and overlaid Expert data.
    pub data: ExpertModeData,
    /// RNG positioned immediately before PhysCal's internal initialization.
    pub rng: Sfmt19937Rng,
    /// Number of fixed parameter slots consumed by the record.
    pub n_para_consumed: usize,
}

/// Result of a serial PhysCal core run.
#[derive(Debug)]
pub struct PhysCalResult {
    /// Fixed parameters after the measurement loop; unchanged from loading.
    pub data: ExpertModeData,
    /// Sampling and observable state.
    pub state: VmcOptimizationState,
    /// Actual rank-local RNG after initialization and every completed sample.
    /// Observing this owned state does not advance the runner's sampling stream.
    pub final_rng: Sfmt19937Rng,
    /// Number of measurement iterations completed.
    pub iterations: usize,
}

/// Prepare a fixed-parameter PhysCal run using Julia's phase order.
///
/// The function deliberately does not call `init_parameter` or
/// `init_qp_weight`; the PhysCal sampling driver must own those calls so the
/// fixed values are restored after the single C-compatible initialization RNG
/// consumption.
pub fn prepare_phys_cal_from_namelist(
    namelist_path: impl AsRef<Path>,
    opt_para_path: impl AsRef<Path>,
    mode: &str,
    seed: Option<i64>,
) -> Result<PhysCalPreparation, String> {
    prepare_phys_cal_from_namelist_with_seed_offset(
        namelist_path,
        opt_para_path,
        mode,
        seed,
        true,
        0,
        &SingleProcessReducer,
    )
}

/// Prepare PhysCal with an explicit MPI group seed offset.
///
/// Julia adds the group index before initializing each independent sampling
/// chain. The ordinary serial entry point uses offset zero; MPI callers should
/// pass [`Reducer::seed_offset`] so ranks do not replay the same SFMT stream.
pub fn prepare_phys_cal_from_namelist_with_reducer<R: Reducer + ?Sized>(
    namelist_path: impl AsRef<Path>,
    opt_para_path: impl AsRef<Path>,
    mode: &str,
    seed: Option<i64>,
    reducer: &R,
) -> Result<PhysCalPreparation, String> {
    prepare_phys_cal_from_namelist_with_seed_offset(
        namelist_path,
        opt_para_path,
        mode,
        seed,
        true,
        reducer.seed_offset(),
        reducer,
    )
}

/// Prepare PhysCal with an explicit C-compatible OptTrans selection.
pub fn prepare_phys_cal_from_namelist_with_reducer_and_opt_trans<R: Reducer + ?Sized>(
    namelist_path: impl AsRef<Path>,
    opt_para_path: impl AsRef<Path>,
    mode: &str,
    seed: Option<i64>,
    reducer: &R,
    enable_opt_trans: bool,
) -> Result<PhysCalPreparation, String> {
    prepare_phys_cal_from_namelist_with_seed_offset(
        namelist_path,
        opt_para_path,
        mode,
        seed,
        enable_opt_trans,
        reducer.seed_offset(),
        reducer,
    )
}

fn prepare_phys_cal_from_namelist_with_seed_offset<R: Reducer + ?Sized>(
    namelist_path: impl AsRef<Path>,
    opt_para_path: impl AsRef<Path>,
    mode: &str,
    seed: Option<i64>,
    enable_opt_trans: bool,
    seed_offset: usize,
    reducer: &R,
) -> Result<PhysCalPreparation, String> {
    let loaded = (|| {
        if !matches!(mode, "real" | "cmp" | "fsz") {
            return Err(format!("mode must be :real, :cmp, or :fsz; got :{mode}"));
        }
        let namelist_path = namelist_path.as_ref();
        let mut data = mvmc_expert_parsers::parse_expert_mode_files_with_c_opt_trans(
            namelist_path,
            enable_opt_trans,
        )
        .map_err(|error| error.to_string())?;
        let n_para_consumed = read_opt_para_file(&mut data, opt_para_path)?;
        observe_physcal_lifecycle("fixed-loaded", &data, None, Some(n_para_consumed));
        read_input_parameters(&mut data, namelist_path)?;
        observe_physcal_lifecycle("overlaid", &data, None, Some(n_para_consumed));
        sync_modified_parameter(&mut data, false);
        observe_physcal_lifecycle("synchronized", &data, None, Some(n_para_consumed));
        crate::validation::validate_phys_cal(&data)?;
        crate::validation::validate_reducer_rank(&data, reducer)?;
        Ok((data, n_para_consumed))
    })();
    let (data, n_para_consumed) =
        collective_result(loaded, reducer, "PhysCal parse/load/validation")?;
    let actual_seed = resolve_rnd_seed(data.modpara.rnd_seed, seed, seed_offset, reducer)?;
    let rng = seeded_rng_with_reducer(actual_seed, reducer)?;
    observe_physcal_lifecycle("seeded", &data, Some(&rng), Some(n_para_consumed));
    Ok(PhysCalPreparation {
        data,
        rng,
        n_para_consumed,
    })
}

/// Run the serial PhysCal sampling and main-calculation loop.
///
/// Parameter optimization and SR are intentionally absent. The preparation's
/// cloned data consumes the one C-compatible initialization draw block, while
/// the returned `data` remains the fixed loaded parameter set.
pub fn vmc_phys_cal(preparation: PhysCalPreparation) -> Result<PhysCalResult, String> {
    vmc_phys_cal_with_reducer(preparation, None, &SingleProcessReducer)
}

/// Run serial PhysCal and invoke `callback` once per completed sample.
pub fn vmc_phys_cal_with_callback(
    preparation: PhysCalPreparation,
    callback: &mut PhysCalCallback<'_>,
) -> Result<PhysCalResult, String> {
    vmc_phys_cal_with_reducer_and_callback(preparation, None, &SingleProcessReducer, Some(callback))
}

/// Run PhysCal and write indexed Green files under `output_dir`.
pub fn vmc_phys_cal_to_dir(
    preparation: PhysCalPreparation,
    output_dir: impl AsRef<Path>,
) -> Result<PhysCalResult, String> {
    let output_dir = output_dir.as_ref();
    vmc_phys_cal_with_reducer(preparation, Some(output_dir), &SingleProcessReducer)
}

/// Run serial PhysCal, write indexed Green files, and invoke `callback` once
/// per completed sample.
pub fn vmc_phys_cal_to_dir_with_callback(
    preparation: PhysCalPreparation,
    output_dir: impl AsRef<Path>,
    callback: &mut PhysCalCallback<'_>,
) -> Result<PhysCalResult, String> {
    let output_dir = output_dir.as_ref();
    vmc_phys_cal_with_reducer_and_callback(
        preparation,
        Some(output_dir),
        &SingleProcessReducer,
        Some(callback),
    )
}

/// Run PhysCal with a caller-provided reducer.
///
/// Each rank performs the same number of independent samples. Accumulators
/// are reduced before Julia's weight-average boundary, and only the reducer's
/// output root writes indexed Green files. This keeps the serial API unchanged
/// while making MPI PhysCal use the same comm0/group reduction contract as
/// parameter optimization.
pub fn vmc_phys_cal_with_reducer<R: Reducer + ?Sized>(
    preparation: PhysCalPreparation,
    output_dir: Option<&Path>,
    reducer: &R,
) -> Result<PhysCalResult, String> {
    vmc_phys_cal_with_reducer_and_callback(preparation, output_dir, reducer, None)
}

/// Run PhysCal with a caller-provided reducer and C-compatible section timer.
///
/// `[20]` UpdateSlaterElm, `[3]` VMCMakeSample, `[4]` VMCMainCal,
/// `[21]` WeightAverage and `[22]` outputData are started/stopped inside; the
/// caller owns `[0]`/`[1]`/`[2]` and report writing. A disabled timer is a
/// no-op (the const generic erases the clock reads).
pub fn vmc_phys_cal_with_reducer_timed<const TIMED: bool, R: Reducer + ?Sized>(
    preparation: PhysCalPreparation,
    output_dir: Option<&Path>,
    reducer: &R,
    timer: &mut CTimer<TIMED>,
) -> Result<PhysCalResult, String> {
    vmc_phys_cal_with_reducer_and_callback_timed::<TIMED, R>(
        preparation,
        output_dir,
        reducer,
        None,
        timer,
    )
}

/// Run PhysCal with a caller-provided reducer and per-sample callback.
///
/// The callback runs on every rank, after reductions and rank-root output for
/// the sample. Every rank participates in the callback-failure agreement,
/// including ranks whose callback succeeded, so a rank-local error is returned
/// consistently without leaving MPI peers in different loop states.
pub fn vmc_phys_cal_with_reducer_and_callback<R: Reducer + ?Sized>(
    preparation: PhysCalPreparation,
    output_dir: Option<&Path>,
    reducer: &R,
    callback: Option<&mut PhysCalCallback<'_>>,
) -> Result<PhysCalResult, String> {
    vmc_phys_cal_with_reducer_and_callback_timed::<false, R>(
        preparation,
        output_dir,
        reducer,
        callback,
        &mut CTimer::<false>::new(),
    )
}

/// Run PhysCal with caller-provided reducer, callback and section timer.
///
/// Behaves like [`vmc_phys_cal_with_reducer_and_callback`] and additionally
/// records the `NVMCCalMode=1` sections into `timer`.
pub fn vmc_phys_cal_with_reducer_and_callback_timed<const TIMED: bool, R: Reducer + ?Sized>(
    mut preparation: PhysCalPreparation,
    output_dir: Option<&Path>,
    reducer: &R,
    callback: Option<&mut PhysCalCallback<'_>>,
    timer: &mut CTimer<TIMED>,
) -> Result<PhysCalResult, String> {
    // The in-place core allocates the correctly sized state after validation
    // and QP setup. Avoid allocating a second full saved chain here.
    let mut state = VmcOptimizationState::zeros(0, 0, 0, 0, 0, 0, false, false);
    let iterations = vmc_phys_cal_in_place_timed::<TIMED, R>(
        &mut preparation.data,
        &mut state,
        &mut preparation.rng,
        output_dir,
        reducer,
        callback,
        timer,
    )?;
    Ok(PhysCalResult {
        data: preparation.data,
        state,
        final_rng: preparation.rng,
        iterations,
    })
}

/// Run fixed-parameter PhysCal using caller-owned data, state and RNG.
///
/// Like Julia's vmc_phys_cal!, this consumes the initialization draw block
/// without replacing fixed coefficients, initializes state, then measures.
/// State is initialized only after validation/output setup; on any later error
/// the caller retains the actual sampled state and RNG at that error boundary.
/// Callback errors return before any subsequent sampling or RNG draw.
/// Existing state contents are replaced at initialization, not continued.
pub fn vmc_phys_cal_in_place<R: Reducer + ?Sized>(
    data: &mut ExpertModeData,
    state: &mut VmcOptimizationState,
    rng: &mut Sfmt19937Rng,
    output_dir: Option<&Path>,
    reducer: &R,
    callback: Option<&mut PhysCalCallback<'_>>,
) -> Result<usize, String> {
    vmc_phys_cal_in_place_timed::<false, R>(
        data,
        state,
        rng,
        output_dir,
        reducer,
        callback,
        &mut CTimer::<false>::new(),
    )
}

/// Run fixed-parameter PhysCal with a caller-provided section timer.
///
/// Records `[20]` UpdateSlaterElm, `[3]` VMCMakeSample, `[4]` VMCMainCal,
/// `[21]` WeightAverage and `[22]` outputData (the caller owns
/// `[0]`/`[1]`/`[2]`).
pub fn vmc_phys_cal_in_place_timed<const TIMED: bool, R: Reducer + ?Sized>(
    data: &mut ExpertModeData,
    state: &mut VmcOptimizationState,
    rng: &mut Sfmt19937Rng,
    output_dir: Option<&Path>,
    reducer: &R,
    mut callback: Option<&mut PhysCalCallback<'_>>,
    timer: &mut CTimer<TIMED>,
) -> Result<usize, String> {
    let all_complex = collective_result(
        crate::validation::validate_phys_cal(data)
            .and_then(|()| crate::validation::validate_reducer_rank(data, reducer))
            .and_then(|()| get_all_complex_flag(data)),
        reducer,
        "PhysCal validation",
    )?;
    let output_setup = if reducer.is_output_root() {
        output_dir.map_or(Ok(()), |path| {
            std::fs::create_dir_all(path).map_err(|error| error.to_string())
        })
    } else {
        Ok(())
    };
    collective_result(output_setup, reducer, "PhysCal output directory")?;
    let mut init_data = data.clone();
    collective_result(
        init_parameter(&mut init_data, rng).map_err(str::to_string),
        reducer,
        "PhysCal complex declaration/initialization",
    )?;
    observe_physcal_lifecycle("initialized-clone", &init_data, Some(rng), None);
    if data.modpara.nmp_trans == 0 {
        data.modpara.nmp_trans = 1;
    } else if data.modpara.nmp_trans < 0 {
        data.modpara.nmp_trans = data.modpara.nmp_trans.abs();
    }
    data.modpara.vmc_calc_mode = 1;
    init_qp_weight(data);
    let use_fsz = data.i_flg_orbital_general != 0;
    *state = collective_result(state_from_data(data), reducer, "PhysCal state construction")?;
    timer.start(20);
    if use_fsz {
        update_slater_elm_fsz(data, state);
    } else {
        update_slater_elm(data, state);
    }
    timer.stop(20);
    let iterations = data.modpara.n_data_qty_smp.max(0) as usize;
    for sample in 0..iterations {
        timer.start(3);
        let sample_result = if use_fsz {
            if all_complex {
                crate::sampling::driver::vmc_make_sample_fsz_with_reducer_timed(
                    data, state, rng, timer, reducer,
                );
                Ok(())
            } else {
                crate::sampling::driver::vmc_make_sample_fsz_real_with_reducer(
                    data, state, rng, reducer,
                )
                .map(|_| ())
                .map_err(|error| error.to_string())
            }
        } else if all_complex {
            crate::sampling::driver::vmc_make_sample_with_reducer_timed(
                data, state, rng, timer, reducer,
            )
            .map(|_| ())
            .map_err(|error| error.to_string())
        } else {
            crate::sampling::driver::vmc_make_sample_real_with_reducer_timed(
                data, state, rng, timer, reducer,
            )
            .map(|_| ())
            .map_err(|error| error.to_string())
        };
        timer.stop(3);
        let sample_error = sample_result.as_ref().err().cloned();
        if reducer.any_failure(sample_error.is_some()) {
            return Err(sample_error
                .unwrap_or_else(|| format!("sample {sample} failed on another MPI rank")));
        }
        sample_result?;
        if use_fsz && !all_complex {
            sync_real_fsz_shadow(state);
        }
        PHYSCAL_GREEN_OBSERVER.with(|slot| {
            let observer = slot.borrow().clone();
            if let Some(observer) = observer {
                observer.sample_completed(data, sample, state, rng);
            }
        });
        clear_phys_quantity(state);
        timer.start(4);
        with_physcal_green_sample(sample, || {
            accumulate_observables(data, state, all_complex, use_fsz, timer, reducer);
        });
        timer.stop(4);
        timer.start(21);
        reduce_accumulators(state, reducer, all_complex);
        weight_average_we(state);
        average_physcal_rank_contributions(state, reducer);
        reduce_counter(state, reducer);
        timer.stop(21);
        timer.start(22);
        let output_error = if reducer.is_output_root() {
            if let Some(output_dir) = output_dir {
                crate::io::output_phys_data(data, state, sample, Some(output_dir))
                    .err()
                    .map(|error| error.to_string())
            } else {
                None
            }
        } else {
            None
        };
        timer.stop(22);
        if reducer.any_failure(output_error.is_some()) {
            return Err(output_error
                .unwrap_or_else(|| format!("output sample {sample} failed on another MPI rank")));
        }

        {
            let callback_error = callback
                .as_deref_mut()
                .and_then(|callback| callback(sample, data, state.energy.etot, 0).err());
            let any_callback_error = reducer.any_failure(callback_error.is_some());
            if any_callback_error {
                return Err(callback_error
                    .unwrap_or_else(|| "PhysCal callback failed on another rank".into()));
            }
        }
    }
    Ok(iterations)
}

fn average_physcal_rank_contributions<R: Reducer + ?Sized>(
    state: &mut VmcOptimizationState,
    reducer: &R,
) {
    let Some(phys) = state.phys_quantities.as_mut() else {
        return;
    };
    let count = reducer.reduction_size();
    if count <= 1 {
        return;
    }
    let inv = 1.0 / count as f64;
    for values in [
        &mut phys.phys_cis_ajs,
        &mut phys.phys_cis_ajs_ckt_alt,
        &mut phys.phys_cis_ajs_ckt_alt_dc,
        &mut phys.phys_lanczos_qqqq,
        &mut phys.phys_lanczos_qcisajsq,
        &mut phys.phys_lanczos_qcisajscktaltq,
        &mut phys.phys_lanczos_qcisajscktaltq_dc,
    ] {
        for value in values.iter_mut() {
            *value *= inv;
        }
    }
}

/// Optional controls for the direct optimization loop.
#[derive(Default)]
pub struct OptimizationOptions<'a> {
    /// Called after successful SR and synchronization, or after sampling-only output.
    pub callback: Option<&'a mut StepCallback<'a>>,
    /// Stop after the first sample/output step, before SR, sync and final output.
    pub skip_sr: bool,
}

/// Run `nsteps` SR steps starting from `data` and the seeded `rng`.
///
/// Real and complex sz-conserved models and complex AP/P FSZ models use
/// their corresponding drivers. Runtime validation rejects unported
/// features before sampling; a failed SR step returns an error before
/// parameter synchronization or any subsequent iteration.
pub fn vmc_para_opt<R: Reducer + ?Sized>(
    data: &mut ExpertModeData,
    state: &mut VmcOptimizationState,
    rng: &mut Sfmt19937Rng,
    output_dir: Option<&Path>,
    reducer: &R,
    options: OptimizationOptions<'_>,
) -> Result<(), String> {
    vmc_para_opt_timed(
        data,
        state,
        rng,
        output_dir,
        reducer,
        options,
        &mut CTimer::<false>::new(),
    )
}

/// Run the same numerical loop with a caller-owned, optionally enabled timer.
/// Disabled instantiations compile clock reads and slot operations out of kernels.
pub fn vmc_para_opt_timed<const TIMED: bool, R: Reducer + ?Sized>(
    data: &mut ExpertModeData,
    state: &mut VmcOptimizationState,
    rng: &mut Sfmt19937Rng,
    output_dir: Option<&Path>,
    reducer: &R,
    mut options: OptimizationOptions<'_>,
    timer: &mut CTimer<TIMED>,
) -> Result<(), String> {
    let all_complex = collective_result(
        crate::validation::validate_grouped_runtime(
            data,
            crate::validation::RuntimeEntryPoint::ParaOpt,
        )
        .and_then(|()| crate::validation::validate_reducer_rank(data, reducer))
        .and_then(|()| crate::validation::validate_para_opt(data))
        .and_then(|()| {
            validate_optimization_window(
                data.modpara.nsr_opt_itr_step,
                data.modpara.nsr_opt_itr_smp,
            )
        })
        .and_then(|()| state.validate_declared_mode(data))
        .and_then(|()| get_all_complex_flag(data)),
        reducer,
        "optimization validation",
    )?;
    let n_steps = data.modpara.nsr_opt_itr_step.max(0) as usize;
    // Validation excludes C's unwritten leading rows for oversized windows.
    // Freeze the explicitly selected supported window before callbacks run.
    let window_len = data.modpara.nsr_opt_itr_smp as usize;
    let window_start = n_steps - window_len;
    state.opt_data.clear();
    let n_para = data.count_variational_parameters();
    data.ensure_optimization_flags(n_para);
    let i_flg_general = data.i_flg_orbital_general;
    let use_fsz = i_flg_general != 0;
    // C VMCMakeSample(comm_child1) generates NVMCSample saved configurations
    // on every chain. Only VMCMainCal partitions those saved configurations
    // within comm_child1; partitioning this count changes the RNG trajectory.
    timer.start(2);
    #[cfg(test)]
    observe_optimization_rng("initialized", rng);
    for step in 0..n_steps {
        timer.start(20);
        // 1. Slater table refresh.
        if use_fsz {
            update_slater_elm_fsz(data, state);
        } else {
            update_slater_elm(data, state);
        }
        crate::qp::update_qp_weight_for(data);
        timer.stop(20);
        timer.start(3);
        // 2. Sampler.
        let sample_result: Result<crate::sampling::SampleStats, String> = if use_fsz {
            if all_complex {
                Ok(
                    crate::sampling::driver::vmc_make_sample_fsz_with_reducer_timed(
                        data, state, rng, timer, reducer,
                    ),
                )
            } else {
                crate::sampling::driver::vmc_make_sample_fsz_real_with_reducer(
                    data, state, rng, reducer,
                )
                .map_err(|error| error.to_string())
            }
        } else if !all_complex {
            crate::sampling::driver::vmc_make_sample_real_with_reducer_timed(
                data, state, rng, timer, reducer,
            )
            .map_err(|error| error.to_string())
        } else {
            crate::sampling::driver::vmc_make_sample_with_reducer_timed(
                data, state, rng, timer, reducer,
            )
            .map_err(|error| error.to_string())
        };
        let sample_error = sample_result.as_ref().err().cloned();
        if reducer.any_failure(sample_error.is_some()) {
            return Err(sample_error
                .unwrap_or_else(|| format!("sample step {step} failed on another MPI rank")));
        }
        let _sample_stats = sample_result?;
        #[cfg(test)]
        if step == 0 {
            observe_optimization_rng("sampling-return", rng);
        }
        if use_fsz && !all_complex {
            sync_real_fsz_shadow(state);
        }
        timer.stop(3);

        // Julia proceeds after a void sampler early return, retaining the saved
        // configurations. Any nonfinite SR result then stops before mutation.
        // 3. Main accumulator.
        timer.start(4);
        timer.start(24);
        clear_phys_quantity(state);
        timer.stop(24);
        accumulate_observables(data, state, all_complex, use_fsz, timer, reducer);

        timer.stop(4);
        timer.start(21);
        reduce_accumulators(state, reducer, all_complex);
        timer.start_diag(960, timer.diagnostics.weightavg);
        // 4. Weighted averages + counter reduction.
        timer.start_diag(962, timer.diagnostics.weightavg);
        weight_average_we(state);
        timer.stop_diag(962, timer.diagnostics.weightavg);
        timer.start(25);
        timer.start_diag(965, timer.diagnostics.weightavg);
        if all_complex {
            weight_average_sr_opt(state);
        } else {
            weight_average_sr_opt_real(state, data.modpara.nsrcg != 0);
        }
        timer.stop_diag(965, timer.diagnostics.weightavg);
        timer.stop(25);
        timer.start_diag(966, timer.diagnostics.weightavg);
        reduce_counter(state, reducer);
        timer.stop_diag(966, timer.diagnostics.weightavg);
        timer.stop_diag(960, timer.diagnostics.weightavg);
        timer.stop(21);

        crate::sr::observer::normalized(step, state, all_complex);
        // 5. Output.
        timer.start(22);
        let output_error = if reducer.is_output_root() {
            output_data(data, state, step, output_dir)
                .err()
                .map(|error| error.to_string())
        } else {
            None
        };
        if reducer.any_failure(output_error.is_some()) {
            return Err(output_error
                .unwrap_or_else(|| format!("output step {step} failed on another MPI rank")));
        }
        timer.stop(22);

        if options.skip_sr {
            timer.stop(2);
            let callback_result = options.callback.as_mut().map_or(Ok(()), |callback| {
                callback(step, data, state.energy.etot, 0)
            });
            collective_result(
                callback_result.and_then(|()| state.validate_declared_mode(data)),
                reducer,
                "optimization callback/declaration",
            )?;
            return Ok(());
        }

        // 6. SR update.
        timer.start(5);
        // SR kernels apply their update locally. Restore successful ranks too
        // when a peer fails, before synchronization can publish that update.
        let before_sr = (reducer.reduction_size() > 1).then(|| data.clone());
        let sr_result = if data.modpara.nsrcg != 0 {
            let sr_output = if reducer.is_output_root() {
                output_dir
            } else {
                None
            };
            crate::sr_cg::stochastic_opt_cg_with_reducer(data, state, sr_output, reducer)
                .map_err(|e| e.to_string())
        } else if all_complex {
            Ok(crate::sr::stochastic_opt_complex_timed(data, state, timer))
        } else {
            Ok(crate::sr::stochastic_opt_real_timed(data, state, timer))
        };
        let sr_error = sr_result.as_ref().err().cloned();
        let info = sr_result.unwrap_or(1);
        timer.stop(5);
        let mut sr_failure = [i64::from(sr_error.is_some() || info != 0)];
        reducer.allreduce_sum_i64(&mut sr_failure);
        if sr_failure[0] != 0 {
            if let Some(before_sr) = before_sr {
                *data = before_sr;
            }
            timer.stop(2);
            return Err(sr_error.unwrap_or_else(|| {
                format!(
                    "vmc_para_opt: {} SR failed at step {step} (local status {info}); parameters were not updated",
                    if data.modpara.nsrcg != 0 { "CG" } else { "direct" }
                )
            }));
        }

        // 7. Sync modified parameters.
        timer.start(23);
        sync_modified(data, reducer);
        timer.stop(23);
        #[cfg(test)]
        issue180_boundary_dev::record_step(step, data, state, rng);
        if step >= window_start {
            store_opt_data(data, state, step - window_start);
        }
        let callback_result = options.callback.as_mut().map_or(Ok(()), |callback| {
            callback(step, data, state.energy.etot, info)
        });
        collective_result(
            callback_result.and_then(|()| state.validate_declared_mode(data)),
            reducer,
            "optimization callback/declaration",
        )?;
    }

    let output_error = if reducer.is_output_root() {
        output_opt_data(data, state, output_dir)
            .err()
            .map(|error| error.to_string())
    } else {
        None
    };
    if reducer.any_failure(output_error.is_some()) {
        return Err(
            output_error.unwrap_or_else(|| "final output failed on another MPI rank".into())
        );
    }
    timer.stop(2);
    Ok(())
}

/// Initial parameter overlay policy, matching Julia's `:auto`, `:none` and path.
#[derive(Debug, Clone, Default)]
pub enum InitialDef {
    /// Load a neighboring initial.def if present; malformed records fail.
    #[default]
    Auto,
    /// Skip initial.def even when present.
    None,
    /// Require this file, resolved relative to the current working directory.
    Path(std::path::PathBuf),
}

/// Configuration for the namelist runner. Actual numerical mode comes from inputs.
#[derive(Debug, Clone)]
pub struct RunConfig {
    /// Positive number of optimization steps.
    pub nsteps: i64,
    /// Sanity label: `real`, `cmp` or `fsz`; does not override input mode.
    pub mode: String,
    /// Final averaging window; None preserves NSROptItrSmp from modpara.def.
    pub nsmp: Option<i64>,
    /// Output directory; None creates a fresh directory in the system temp area.
    /// Explicit paths may be rank-local. Only the output root reads output files.
    pub output_dir: Option<std::path::PathBuf>,
    /// Explicit seed override.
    pub seed: Option<i64>,
    /// Initial parameter file selection.
    pub initial_def: InitialDef,
    /// C-style OptTrans activation. `None` preserves the library's Julia
    /// definition-file default; `Some(false)` ignores OptTrans definitions.
    pub enable_opt_trans: Option<bool>,
}

impl RunConfig {
    /// Construct a configuration with Julia's optional argument defaults.
    pub fn new(nsteps: i64, mode: impl Into<String>) -> Self {
        Self {
            nsteps,
            mode: mode.into(),
            nsmp: None,
            output_dir: None,
            seed: None,
            initial_def: InitialDef::Auto,
            enable_opt_trans: None,
        }
    }
}

/// Run optimization with Julia's configuration, phase order and summary contract.
pub fn run_para_opt_from_namelist(
    namelist_path: impl AsRef<Path>,
    config: RunConfig,
) -> Result<RunSummary, String> {
    run_para_opt_from_namelist_with_reducer(namelist_path, config, &SingleProcessReducer)
}

/// Run optimization with a caller-provided all-reduce backend.
///
/// The MPI-enabled application initializes [`crate::mpi::MpiContext`] and
/// passes it here.  Keeping initialization outside the library preserves MPI
/// ownership and allows ordinary Rust callers to continue using the serial
/// entry point without linking an MPI implementation.
pub fn run_para_opt_from_namelist_with_reducer<R: Reducer + ?Sized>(
    namelist_path: impl AsRef<Path>,
    config: RunConfig,
    reducer: &R,
) -> Result<RunSummary, String> {
    collective_result(
        validate_run_options(&config),
        reducer,
        "optimization configuration",
    )?;
    let flags = TimerEnv::from_env();
    if flags.legacy_warning() {
        eprintln!("warning: MVMC_TIMER is deprecated; use MVMC_C_TIMER=1 for the C-compatible zvo_CalcTimer.dat timer.");
    }
    if flags.enabled() {
        run_para_opt_timed(
            namelist_path,
            config,
            reducer,
            &mut CTimer::<true>::new(),
            flags,
        )
    } else {
        run_para_opt_timed(
            namelist_path,
            config,
            reducer,
            &mut CTimer::<false>::new(),
            flags,
        )
    }
}

fn validate_run_options(config: &RunConfig) -> Result<(), String> {
    if !matches!(config.mode.as_str(), "real" | "cmp" | "fsz") {
        return Err(format!(
            "mode must be :real, :cmp, or :fsz; got {}",
            config.mode
        ));
    }
    if config.nsteps <= 0 {
        return Err(format!("nsteps must be positive; got {}", config.nsteps));
    }
    if config.nsmp.is_some_and(|value| value <= 0) {
        return Err(format!(
            "nsmp must be positive when provided; got {}",
            config.nsmp.unwrap()
        ));
    }
    Ok(())
}

fn validate_optimization_window(nsteps: i64, nsmp: i64) -> Result<(), String> {
    if nsmp <= 0 {
        return Err(format!("effective nsmp must be positive; got {nsmp}"));
    }
    if nsteps < nsmp {
        return Err(format!(
            "nsteps ({nsteps}) must be >= nsmp ({nsmp}); C leaves oversized-window rows unwritten"
        ));
    }
    Ok(())
}

fn run_para_opt_timed<const TIMED: bool, R: Reducer + ?Sized>(
    namelist_path: impl AsRef<Path>,
    config: RunConfig,
    reducer: &R,
    timer: &mut CTimer<TIMED>,
    flags: TimerEnv,
) -> Result<RunSummary, String> {
    timer.reset();
    timer.diagnostics = flags;
    timer.start(0);
    timer.start(1);
    timer.start(11);
    let parsed = (|| {
        let data = match config.enable_opt_trans {
            Some(enabled) => mvmc_expert_parsers::parse_expert_mode_files_with_c_opt_trans(
                &namelist_path,
                enabled,
            ),
            // The production runner follows C's FlagOptTrans contract: a
            // definition file alone does not activate optimized translation.
            None => {
                mvmc_expert_parsers::parse_expert_mode_files_with_c_opt_trans(&namelist_path, false)
            }
        }
        .map_err(|e| e.to_string())?;
        timer.stop(11);
        crate::validation::validate_para_opt(&data)?;
        crate::validation::validate_reducer_rank(&data, reducer)?;
        let effective_nsmp = config.nsmp.unwrap_or(data.modpara.nsr_opt_itr_smp);
        if effective_nsmp <= 0 {
            return Err(format!(
                "effective nsmp must be positive; got {effective_nsmp}"
            ));
        }
        validate_optimization_window(config.nsteps, effective_nsmp)?;
        Ok((data, effective_nsmp))
    })();
    let (mut data, effective_nsmp) =
        collective_result(parsed, reducer, "optimization parse/validation")?;

    // Preserve Julia's init -> initial.def -> In*.def -> sync -> QP phase order.
    let actual_seed = resolve_rnd_seed(
        data.modpara.rnd_seed,
        config.seed,
        reducer.seed_offset(),
        reducer,
    )?;
    let mut rng = seeded_rng_with_reducer(actual_seed, reducer)?;
    timer.start(13);
    collective_result(
        init_parameter(&mut data, &mut rng).map_err(str::to_string),
        reducer,
        "optimization complex declaration/initialization",
    )?;
    let (initial_path, auto) = match config.initial_def {
        InitialDef::Auto => {
            let path = namelist_path
                .as_ref()
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join("initial.def");
            (path.is_file().then_some(path), true)
        }
        InitialDef::None => (None, false),
        InitialDef::Path(path) => (Some(path), false),
    };
    let loaded = (|| {
        if let Some(path) = initial_path {
            if !read_initial_def(&mut data, &path).map_err(|error| error.to_string())? {
                return Err(if auto {
                    format!("read_initial_def! failed on auto-detected initial.def at {}; pass initial_def=None to skip explicitly", path.display())
                } else {
                    format!(
                        "read_initial_def! failed for explicitly requested path: {}",
                        path.display()
                    )
                });
            }
        }
        read_input_parameters(&mut data, &namelist_path)?;
        Ok(())
    })();
    collective_result(loaded, reducer, "optimization parameter load")?;
    // C SyncModifiedParameter(comm0): publish root parameters before local gauge fixing.
    sync_modified(&mut data, reducer);
    timer.stop(13);
    init_qp_weight(&mut data);
    timer.stop(1);
    data.modpara.nsr_opt_itr_step = config.nsteps;
    data.modpara.nsr_opt_itr_smp = effective_nsmp;
    let mut state = collective_result(
        state_from_data(&data),
        reducer,
        "optimization state construction",
    )?;
    let output_result = (|| {
        let output_dir = match config.output_dir {
            Some(path) => {
                std::fs::create_dir_all(&path).map_err(|e| e.to_string())?;
                path
            }
            None => {
                let path = if reducer.is_output_root() {
                    fresh_output_directory().and_then(|path| {
                        path.into_os_string()
                            .into_string()
                            .map(Some)
                            .map_err(|_| "generated output path is not UTF-8".into())
                    })
                } else {
                    Ok(None)
                };
                let path = collective_result(path, reducer, "default output directory")?;
                let mut bytes: Vec<i64> = path
                    .as_ref()
                    .map_or_else(Vec::new, |path| path.bytes().map(i64::from).collect());
                let mut length = [bytes.len() as i64];
                collective_result(
                    reducer.broadcast_i64(0, &mut length),
                    reducer,
                    "output path length broadcast",
                )?;
                bytes.resize(length[0] as usize, 0);
                collective_result(
                    reducer.broadcast_i64(0, &mut bytes),
                    reducer,
                    "output path broadcast",
                )?;
                let path = String::from_utf8(bytes.into_iter().map(|value| value as u8).collect())
                    .map(std::path::PathBuf::from)
                    .map_err(|error| error.to_string());
                collective_result(path, reducer, "output path decode")?
            }
        };
        Ok(output_dir)
    })();
    let output_dir = collective_result(output_result, reducer, "optimization output directory")?;
    #[cfg(test)]
    issue180_boundary_dev::record("initialized", &data, &state, &rng);
    vmc_para_opt_timed(
        &mut data,
        &mut state,
        &mut rng,
        Some(&output_dir),
        reducer,
        OptimizationOptions::default(),
        timer,
    )?;
    timer.stop(0);
    #[cfg(test)]
    issue180_boundary_dev::record("final", &data, &state, &rng);
    let final_result = (|| {
        if TIMED && reducer.is_output_root() {
            timer
                .write_para_opt(&output_dir, "zvo")
                .map_err(|error| error.to_string())?;
            if flags.any_diag() {
                timer
                    .write_diag(&output_dir, "zvo")
                    .map_err(|error| error.to_string())?;
            }
        }
        if reducer.is_output_root() {
            read_run_summary(&data, &output_dir)
        } else {
            // Julia's non-output ranks return metadata without file readback.
            Ok(RunSummary {
                status: 0,
                output_dir: std::fs::canonicalize(&output_dir)
                    .map_err(|error| error.to_string())?,
                zvo_first_n: Vec::new(),
                ctest_values: Vec::new(),
                final_energy_per_site: f64::NAN,
                effective_nsteps: data.modpara.nsr_opt_itr_step as usize,
                effective_nsmp: data.modpara.nsr_opt_itr_smp as usize,
            })
        }
    })();
    collective_result(final_result, reducer, "optimization final output/summary")
}

fn fresh_output_directory() -> Result<std::path::PathBuf, String> {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    loop {
        let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("mvmc-run-{}-{id}", std::process::id()));
        match std::fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.to_string()),
        }
    }
}

/// Sum the per-rank sample accumulators before normalization and SR.
///
/// Sampling buffers remain rank-local; only quantities that feed the global
/// weighted averages are reduced.  This mirrors Julia's `WeightAverage!`
/// boundary and keeps rank-local configurations available for diagnostics.
fn reduce_accumulators<R: Reducer + ?Sized>(
    state: &mut VmcOptimizationState,
    reducer: &R,
    all_complex: bool,
) {
    let mut energy = [
        state.energy.wc,
        state.energy.etot,
        state.energy.etot2,
        state.energy.sztot,
        state.energy.sztot2,
    ];
    reducer.allreduce_sum_c64(&mut energy);
    [
        &mut state.energy.wc,
        &mut state.energy.etot,
        &mut state.energy.etot2,
        &mut state.energy.sztot,
        &mut state.energy.sztot2,
    ]
    .into_iter()
    .zip(energy)
    .for_each(|(dst, value)| *dst = value);

    // vmcmain.c selects exactly one WeightAverageSROpt branch by AllComplexFlag.
    // Allocated inactive shadow buffers must not introduce extra collectives.
    if all_complex {
        reducer.allreduce_sum_c64(&mut state.sr_opt.sr_opt_oo);
        reducer.allreduce_sum_c64(&mut state.sr_opt.sr_opt_ho);
    } else {
        reducer.allreduce_sum_f64(&mut state.sr_opt.sr_opt_oo_real);
        reducer.allreduce_sum_f64(&mut state.sr_opt.sr_opt_ho_real);
    }
    if let Some(phys) = state.phys_quantities.as_mut() {
        reducer.allreduce_sum_c64(&mut phys.phys_cis_ajs);
        reducer.allreduce_sum_c64(&mut phys.phys_cis_ajs_ckt_alt);
        reducer.allreduce_sum_c64(&mut phys.phys_cis_ajs_ckt_alt_dc);
        reducer.allreduce_sum_c64(&mut phys.phys_lanczos_qqqq);
        reducer.allreduce_sum_c64(&mut phys.phys_lanczos_qcisajsq);
        reducer.allreduce_sum_c64(&mut phys.phys_lanczos_qcisajscktaltq);
        reducer.allreduce_sum_c64(&mut phys.phys_lanczos_qcisajscktaltq_dc);
    }
}

#[cfg(test)]
mod mpi_tests {
    use super::*;

    #[derive(Debug)]
    struct ScalingReducer {
        world: usize,
        rank: usize,
    }

    impl Reducer for ScalingReducer {
        fn allreduce_sum_f64(&self, values: &mut [f64]) {
            for value in values {
                *value *= self.world as f64;
            }
        }

        fn allreduce_sum_c64(&self, values: &mut [Complex64]) {
            for value in values {
                *value *= self.world as f64;
            }
        }

        fn allreduce_sum_i64(&self, values: &mut [i64]) {
            for value in values {
                *value *= self.world as i64;
            }
        }

        fn world_size(&self) -> usize {
            self.world
        }

        fn rank(&self) -> usize {
            self.rank
        }
    }

    #[test]
    fn sample_parallel_reduction_sums_energy_and_sr_buffers() {
        let mut state = VmcOptimizationState::zeros(2, 2, 0, 1, 1, 2, true, false);
        state.energy.wc = Complex64::new(3.0, 0.0);
        state.sr_opt.sr_opt_oo[0] = Complex64::new(4.0, -1.0);
        state.sr_opt.sr_opt_ho[0] = Complex64::new(-2.0, 0.5);
        reduce_accumulators(&mut state, &ScalingReducer { world: 2, rank: 1 }, true);
        assert_eq!(state.energy.wc, Complex64::new(6.0, 0.0));
        assert_eq!(state.sr_opt.sr_opt_oo[0], Complex64::new(8.0, -2.0));
        assert_eq!(state.sr_opt.sr_opt_ho[0], Complex64::new(-4.0, 1.0));
    }

    #[test]
    fn accumulator_reduction_selects_only_active_sr_branch_in_exact_order() {
        use std::cell::RefCell;
        #[derive(Default)]
        struct RecordingReducer(RefCell<Vec<(&'static str, usize)>>);
        impl Reducer for RecordingReducer {
            fn allreduce_sum_f64(&self, values: &mut [f64]) {
                self.0.borrow_mut().push(("real", values.len()));
                values.iter_mut().for_each(|value| *value *= 2.0);
            }
            fn allreduce_sum_c64(&self, values: &mut [Complex64]) {
                self.0.borrow_mut().push(("complex", values.len()));
                values.iter_mut().for_each(|value| *value *= 2.0);
            }
            fn allreduce_sum_i64(&self, _: &mut [i64]) {
                panic!("no integer accumulator reduction");
            }
        }
        for all_complex in [false, true] {
            let mut state = VmcOptimizationState::zeros(2, 2, 0, 1, 1, 2, true, false);
            state.sr_opt.sr_opt_oo.fill(Complex64::new(3.0, -4.0));
            state.sr_opt.sr_opt_ho.fill(Complex64::new(-5.0, 6.0));
            state.sr_opt.sr_opt_oo_real.fill(7.0);
            state.sr_opt.sr_opt_ho_real.fill(-8.0);
            let before = state.sr_opt.clone();
            let reducer = RecordingReducer::default();
            reduce_accumulators(&mut state, &reducer, all_complex);
            let expected = if all_complex {
                assert_eq!(state.sr_opt.sr_opt_oo_real, before.sr_opt_oo_real);
                assert_eq!(state.sr_opt.sr_opt_ho_real, before.sr_opt_ho_real);
                vec![
                    ("complex", 5),
                    ("complex", before.sr_opt_oo.len()),
                    ("complex", before.sr_opt_ho.len()),
                ]
            } else {
                assert_eq!(state.sr_opt.sr_opt_oo, before.sr_opt_oo);
                assert_eq!(state.sr_opt.sr_opt_ho, before.sr_opt_ho);
                vec![
                    ("complex", 5),
                    ("real", before.sr_opt_oo_real.len()),
                    ("real", before.sr_opt_ho_real.len()),
                ]
            };
            assert_eq!(*reducer.0.borrow(), expected);
        }
    }
}

fn read_run_summary(data: &ExpertModeData, output_dir: &Path) -> Result<RunSummary, String> {
    let nsteps = data.modpara.nsr_opt_itr_step as usize;
    let nsmp = data.modpara.nsr_opt_itr_smp as usize;
    // Issue #48 requires configured-head readback; Julia v0.5.0 hardcodes
    // zvo here despite honoring CDataFileHead in its writer.
    let head = if data.modpara.c_data_file_head.is_empty() {
        "zvo"
    } else {
        &data.modpara.c_data_file_head
    };
    let path = output_dir.join(format!("{head}_out.dat"));
    let text = std::fs::read_to_string(&path).map_err(|error| error.to_string())?;
    let lines: Vec<_> = text.lines().map(|line| line.trim().to_owned()).collect();
    if lines.len() < nsteps {
        return Err(format!(
            "expected at least {nsteps} output rows, found {} at {}",
            lines.len(),
            path.display()
        ));
    }
    let zvo_first_n = lines[..nsteps].to_vec();
    let rows: Vec<Vec<f64>> = zvo_first_n
        .iter()
        .map(|line| {
            line.split_whitespace()
                .map(|token| token.parse::<f64>().map_err(|error| error.to_string()))
                .collect()
        })
        .collect::<Result<_, _>>()?;
    if rows.iter().any(|row| row.len() < 2) {
        return Err(format!(
            "output must contain at least two columns for C ctest comparison: {}",
            path.display()
        ));
    }
    if data.modpara.nsite <= 0 {
        return Err("modpara.nsite must be positive to compute energy per site".into());
    }
    let window = &rows[nsteps - nsmp..nsteps];
    let ctest_values = (0..2)
        .map(|column| window.iter().map(|row| row[column]).sum::<f64>() / nsmp as f64)
        .collect();
    Ok(RunSummary {
        status: 0,
        output_dir: std::fs::canonicalize(output_dir).map_err(|error| error.to_string())?,
        zvo_first_n,
        ctest_values,
        final_energy_per_site: rows[nsteps - 1][0] / data.modpara.nsite as f64,
        effective_nsteps: nsteps,
        effective_nsmp: nsmp,
    })
}

/// Select the execution mode from definition-level flags.
///
/// C does not infer complex mode from loaded imaginary coefficients. Both
/// initialization and execution must honor the model's declarations.
pub fn get_all_complex_flag(data: &ExpertModeData) -> Result<bool, String> {
    let declared = mvmc_expert_parsers::utils::parameter_init::all_complex_flag(data)
        .map_err(str::to_string)?;
    if !data.complex_flags.is_empty() {
        return Ok(data.complex_flags.iter().any(|&flag| flag != 0));
    }
    Ok(declared)
}

/// Resolve the seed integer before the runner's separate SFMT UInt32 conversion.
///
/// An explicit override takes precedence, including negative overrides. Without
/// an override, nonnegative input is used directly (zero remains zero); negative
/// input uses one output-root Unix clock value broadcast through the reducer.
/// The caller-supplied group offset is then added with existing wrapping i64
/// arithmetic. Passing the parser's default11272 preserves that default.
///
/// This retains Julia's public resolution lifecycle and C's base-plus-group
/// seeding order without initializing or drawing from an RNG. A returned value
/// is not necessarily a valid SFMT seed: runners subsequently check UInt32
/// conversion. No absolute-value conversion or reseeding repair is performed.
///
/// Group-offset conversion, root-clock and broadcast errors propagate
/// collectively through the existing reducer contract. Every participating rank
/// must enter this function with consistent override/input policy. The clock is
/// not queried for explicit overrides or nonnegative input.
pub fn resolve_rnd_seed<R: Reducer + ?Sized>(
    rnd_seed: i64,
    seed_override: Option<i64>,
    group1: usize,
    reducer: &R,
) -> Result<i64, String> {
    resolve_seed_with_reducer(rnd_seed, seed_override, group1, reducer, || {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|error| format!("cannot resolve time-based RndSeed: {error}"))
            .and_then(|duration| {
                i64::try_from(duration.as_secs())
                    .map_err(|error| format!("cannot convert time-based RndSeed: {error}"))
            })
    })
}

fn resolve_seed_with_reducer<R: Reducer + ?Sized>(
    rnd_seed: i64,
    seed_override: Option<i64>,
    group1: usize,
    reducer: &R,
    unix_seconds: impl FnOnce() -> Result<i64, String>,
) -> Result<i64, String> {
    let group1_result = i64::try_from(group1)
        .map_err(|_| format!("MPI seed offset {group1} does not fit in the seed offset"));
    if reducer.any_failure(group1_result.is_err()) {
        return Err(group1_result
            .err()
            .unwrap_or_else(|| "MPI seed offset conversion failed collectively".into()));
    }
    let group1 = group1_result.expect("seed offset conversion succeeded collectively");
    let base = if let Some(seed) = seed_override {
        seed
    } else if rnd_seed < 0 {
        // Keep every rank in the collective even when the root clock fails.
        // [status, base], with status 0 for success and 1 for a clock error.
        let mut payload = if reducer.is_output_root() {
            match unix_seconds() {
                Ok(seed) => [0, seed],
                Err(_) => [1, 0],
            }
        } else {
            [0, 0]
        };
        collective_result(
            reducer.broadcast_i64(0, &mut payload),
            reducer,
            "seed broadcast",
        )?;
        if payload[0] != 0 {
            return Err("cannot resolve time-based RndSeed collectively".into());
        }
        payload[1]
    } else {
        rnd_seed
    };
    Ok(base.wrapping_add(group1))
}

fn seeded_rng(seed: i64) -> Result<Sfmt19937Rng, String> {
    let seed = u32::try_from(seed)
        .map_err(|_| format!("resolved SFMT seed {seed} is outside the UInt32 range"))?;
    Ok(Sfmt19937Rng::new(seed))
}

fn seeded_rng_with_reducer<R: Reducer + ?Sized>(
    seed: i64,
    reducer: &R,
) -> Result<Sfmt19937Rng, String> {
    let result = seeded_rng(seed);
    if reducer.any_failure(result.is_err()) {
        return Err(result
            .err()
            .unwrap_or_else(|| "resolved SFMT seed conversion failed collectively".into()));
    }
    result
}

#[cfg(test)]
mod seed_tests {
    use std::cell::RefCell;

    use super::*;

    #[derive(Debug)]
    struct MockSeedReducer {
        root: bool,
        broadcast: RefCell<Result<[i64; 2], String>>,
    }

    impl Reducer for MockSeedReducer {
        fn broadcast_i64(&self, _root: usize, values: &mut [i64]) -> Result<(), String> {
            let broadcast = self.broadcast.borrow().clone()?;
            values.copy_from_slice(&broadcast);
            Ok(())
        }

        fn allreduce_sum_f64(&self, _buf: &mut [f64]) {}
        fn allreduce_sum_c64(&self, _buf: &mut [Complex64]) {}
        fn allreduce_sum_i64(&self, _buf: &mut [i64]) {}

        fn is_output_root(&self) -> bool {
            self.root
        }
    }

    #[test]
    fn seed_conversion_rejects_values_outside_julia_uint32_range() {
        for seed in [-1, i64::from(u32::MAX) + 1] {
            assert!(seeded_rng(seed).is_err(), "SFMT.jl UInt32 rejects {seed}");
        }
        assert!(seeded_rng(0).is_ok());
        assert!(seeded_rng(i64::from(u32::MAX)).is_ok());
    }

    // Julia test_unit_parallel.jl: resolve_rnd_seed C parity.
    #[test]
    fn resolve_seed_matches_julia_serial_and_group_policy() {
        for (input, explicit, group, expected) in [
            (FALLBACK_SEED, None, 0, 11272),
            (0, None, 0, 0),
            (123, None, 0, 123),
            (-1, None, 0, 1700000000),
            (123, Some(777), 0, 777),
            (-1, Some(777), 0, 777),
            (100, None, 3, 103),
            (-1, None, 3, 1700000003),
        ] {
            assert_eq!(
                resolve_seed_with_reducer(input, explicit, group, &SingleProcessReducer, || {
                    Ok(1700000000)
                })
                .unwrap(),
                expected
            );
        }
    }

    #[test]
    fn clock_is_only_read_for_negative_modpara_without_override() {
        assert_eq!(
            resolve_seed_with_reducer(0, None, 0, &SingleProcessReducer, || {
                panic!("unneeded clock read")
            })
            .unwrap(),
            0
        );
        assert_eq!(
            resolve_seed_with_reducer(-1, Some(7), 0, &SingleProcessReducer, || {
                panic!("override reads clock")
            })
            .unwrap(),
            7
        );
        assert!(
            resolve_seed_with_reducer(-1, None, 0, &SingleProcessReducer, || {
                Err("clock failed".into())
            })
            .is_err()
        );
    }

    #[test]
    fn negative_seed_uses_one_root_clock_even_when_ranks_start_late() {
        let root = MockSeedReducer {
            root: true,
            broadcast: RefCell::new(Ok([0, 1_700_000_000])),
        };
        let non_root = MockSeedReducer {
            root: false,
            broadcast: RefCell::new(Ok([0, 1_700_000_000])),
        };
        assert_eq!(
            resolve_seed_with_reducer(-1, None, 0, &root, || Ok(1_700_000_000)).unwrap(),
            1_700_000_000
        );
        assert_eq!(
            resolve_seed_with_reducer(-1, None, 0, &non_root, || {
                panic!("a late rank must not read its local clock")
            })
            .unwrap(),
            1_700_000_000
        );
    }

    #[test]
    fn broadcast_base_seed_precedes_group_offset() {
        let reducer = MockSeedReducer {
            root: false,
            broadcast: RefCell::new(Ok([0, 1_700_000_000])),
        };
        assert_eq!(
            resolve_seed_with_reducer(-1, None, 3, &reducer, || unreachable!()).unwrap(),
            1_700_000_003
        );
    }

    #[test]
    fn clock_and_broadcast_failures_are_returned_before_rng_initialization() {
        let clock_failure = MockSeedReducer {
            root: false,
            broadcast: RefCell::new(Ok([1, 0])),
        };
        assert_eq!(
            resolve_seed_with_reducer(-1, None, 0, &clock_failure, || unreachable!()).unwrap_err(),
            "cannot resolve time-based RndSeed collectively"
        );

        let broadcast_failure = MockSeedReducer {
            root: true,
            broadcast: RefCell::new(Err("collective conversion failed".into())),
        };
        assert_eq!(
            resolve_seed_with_reducer(-1, None, 0, &broadcast_failure, || Ok(1)).unwrap_err(),
            "collective conversion failed"
        );
    }

    #[test]
    fn invalid_seed_conversion_is_checked_collectively() {
        assert!(seeded_rng_with_reducer(-1, &SingleProcessReducer).is_err());
        assert!(seeded_rng_with_reducer(i64::from(u32::MAX) + 1, &SingleProcessReducer).is_err());
    }

    #[test]
    fn resolved_seed_stream_matches_julia_sfmt_full_block() {
        // SFMT.jl at the pinned Julia v0.5.0 revision. Hash every word
        // of a full 624-word SFMT block; do not compare sample averages.
        for (input, explicit, group, expected_hash) in [
            (11272, None, 0, 6236248514720008343_u64),
            (0, None, 0, 859227577111010836),
            (123, None, 0, 3014287863449549136),
            (-1, None, 0, 2984471086139650888),
            (-1, Some(777), 0, 18056312496568059157),
            (100, None, 3, 8720158099669028788),
            (-1, None, 3, 8638024564356521936),
        ] {
            let seed =
                resolve_seed_with_reducer(input, explicit, group, &SingleProcessReducer, || {
                    Ok(1700000000)
                })
                .unwrap();
            let mut rng = seeded_rng(seed).unwrap();
            let mut hash = 0xcbf29ce484222325_u64;
            for _ in 0..624 {
                hash = (hash ^ u64::from(rng.gen_rand32())).wrapping_mul(0x100000001b3);
            }
            assert_eq!(hash, expected_hash, "resolved seed {seed}");
        }
    }
}

fn state_from_data(data: &ExpertModeData) -> Result<VmcOptimizationState, String> {
    let n_site = data.modpara.nsite.max(0) as usize;
    let n_elec = data.modpara.nelec.max(0) as usize;
    let n_proj = data.projection_layout().n_proj;
    let n_para = data.count_variational_parameters();
    let n_sp = data.modpara.nsp_gauss_leg.max(1) as usize;
    let n_mp = data.modpara.nmp_trans.unsigned_abs() as usize;
    let n_opt = data.n_qp_opt_trans.max(1) as usize;
    let n_qp_full = n_sp * n_mp * n_opt;
    let n_vmc_sample = data.modpara.nvmc_sample.max(0) as usize;
    let all_complex = get_all_complex_flag(data)?;
    let mut state = VmcOptimizationState::zeros(
        n_site,
        n_elec,
        n_proj,
        n_para,
        n_qp_full,
        n_vmc_sample,
        all_complex,
        data.i_flg_orbital_general != 0,
    );
    if data.modpara.vmc_calc_mode != 0 {
        state.phys_quantities = Some(crate::state::PhysicalQuantities::zeros(
            data.green_one_terms.len(),
            data.green_two_ex_terms.len(),
            data.green_two_terms.len(),
        ));
    }
    Ok(state)
}

/// Make real-FSZ sampling results visible to the shared observable kernels.
fn sync_real_fsz_shadow(state: &mut VmcOptimizationState) {
    crate::threading::copy_real_to_complex(
        state.slater_matrix.slater_elm.as_mut_slice(),
        state.slater_matrix.slater_elm_real.as_slice(),
    );
    crate::threading::copy_real_to_complex(
        state.slater_matrix.inv_m.as_mut_slice(),
        state.slater_matrix.inv_m_real.as_slice(),
    );
    crate::threading::copy_real_to_complex(
        &mut state.slater_matrix.pf_m,
        &state.slater_matrix.pf_m_real,
    );
}

#[cfg(test)]
mod mode_tests {
    use super::*;
    use mvmc_expert_parsers::{
        GreenOneTerm, GreenTwoExTerm, GreenTwoTerm, GutzwillerTerm, JastrowTerm, OrbitalTerm, Spin,
    };

    #[test]
    fn opttrans_state_reserves_active_width_after_projection_rbm_and_slater() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/opttrans/namelist_layout.def");
        let mut data = parse_expert_mode_files(path).unwrap();
        let base = data.projection_layout().n_proj + data.count_rbm_parameters() + n_slater(&data);
        for width in [0, 1, 2, 3] {
            data.opt_trans.resize(width, Complex64::new(1.0, 0.0));
            let state = state_from_data(&data).unwrap();
            assert_eq!(state.sr_opt.sr_opt_size, 1 + base + width);
            assert_eq!(
                state.slater_matrix.pf_m.len(),
                data.modpara.nsp_gauss_leg.max(1) as usize
                    * data.modpara.nmp_trans.unsigned_abs().max(1) as usize
                    * data.n_qp_opt_trans.max(1) as usize
            );
        }
    }

    #[test]
    fn rbm_state_places_all_nine_blocks_between_projection_and_slater() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/rbm/namelist_all.def");
        let data = crate::historical_orbital_model::historical_kernel_model(path).unwrap();
        assert_eq!(state_from_data(&data).unwrap().sr_opt.sr_opt_size, 37);
    }

    #[test]
    fn public_runner_rejects_archived_sparse_rbm_definitions_before_output() {
        // These archived Julia inputs are used by test-only programmatic
        // models, not accepted native C definitions. Never manufacture the
        // missing declared flag pairs to turn them into C parity inputs.
        let fixtures =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures");
        for (index, relative) in [
            "rbm/run_rbm_general_cmp/namelist.def",
            "opttrans/run_opt_dh24_rbm_cmp/namelist.def",
        ]
        .iter()
        .enumerate()
        {
            let output = std::env::temp_dir().join(format!(
                "mvmc-rejected-sparse-rbm-{}-{index}",
                std::process::id()
            ));
            assert!(!output.exists());
            let mut config = RunConfig::new(1, "cmp");
            config.nsmp = Some(1);
            config.seed = Some(12395);
            config.output_dir = Some(output.clone());
            let error = run_para_opt_from_namelist(fixtures.join(relative), config)
                .expect_err("incomplete native RBM definitions must not execute");
            assert!(
                error.contains(
                    "RBM requires complete spatial mappings and exactly the declared flag pairs"
                ),
                "{relative}: {error}"
            );
            assert!(!output.exists(), "rejection must precede output setup");
        }
    }

    fn data() -> ExpertModeData {
        let mut data = ExpertModeData::new();
        data.modpara.nsite = 2;
        data.modpara.nelec = 1;
        data.modpara.nmp_trans = 1;
        data.modpara.nvmc_sample = 1;
        data.modpara.n_orbital_idx = 1;
        data.slater_params = vec![Complex64::new(1.0, 0.0)];
        data.orbital_terms.push(OrbitalTerm {
            site1: 0,
            site2: 1,
            idx: 0,
            is_complex: false,
            sign: 1,
        });
        data
    }

    #[test]
    fn real_fsz_shadow_copies_inverse_pads_and_preserves_unmatched_tail() {
        let mut state = VmcOptimizationState::zeros(2, 1, 0, 0, 1, 1, false, true);
        state.slater_matrix.inv_m_real.as_mut_slice().fill(3.0);
        state.slater_matrix.pf_m_real.fill(5.0);
        let tail = Complex64::new(7.0, -9.0);
        state.slater_matrix.pf_m.push(tail);
        sync_real_fsz_shadow(&mut state);
        assert!(state
            .slater_matrix
            .inv_m
            .as_slice()
            .iter()
            .all(|&value| value == Complex64::new(3.0, 0.0)));
        assert_eq!(state.slater_matrix.pf_m[0], Complex64::new(5.0, 0.0));
        assert_eq!(state.slater_matrix.pf_m.last(), Some(&tail));
    }

    #[test]
    fn real_fsz_observation_refresh_populates_real_shadow() {
        let mut data = data();
        data.modpara.nsite = 2;
        data.modpara.nelec = 1;
        let mut state = VmcOptimizationState::zeros(2, 1, 0, 0, 1, 1, false, true);
        state
            .slater_matrix
            .slater_elm
            .set(0, 0, 3, Complex64::new(1.0, 0.0));
        state
            .slater_matrix
            .slater_elm
            .set(0, 3, 0, Complex64::new(-1.0, 0.0));
        let pool = ThreadedPfaPackWorkspace::new(2, 1);
        refresh_fsz_observation_matrix(&data, &mut state, false, &[0, 1], &[0, 1], &pool).unwrap();
        assert_ne!(state.slater_matrix.pf_m_real[0], 0.0);
        assert_eq!(state.slater_matrix.pf_m[0].im, 0.0);
        assert_eq!(
            state.slater_matrix.pf_m[0].re,
            state.slater_matrix.pf_m_real[0]
        );
        assert!(state
            .slater_matrix
            .inv_m_real
            .qp_matrix_slice(0)
            .iter()
            .any(|value| *value != 0.0));
    }

    // Julia test_unit_types.jl checks that complex SROptData has no real
    // buffers. Exercise the actual runner allocation for each factor family.
    #[test]
    fn complex_gutzwiller_with_real_orbitals_allocates_complex_state() {
        let mut data = data();
        data.gutzwiller_terms.push(GutzwillerTerm {
            site: 0,
            value: Complex64::new(0.0, 0.0),
            is_complex: true,
        });
        let state = state_from_data(&data).unwrap();
        assert!(state.sr_opt.sr_opt_oo_real.is_empty());
        assert_eq!(state.slater_matrix.slater_elm_real.n_qp_full(), 0);
    }

    #[test]
    fn complex_jastrow_with_real_orbitals_allocates_complex_state() {
        let mut data = data();
        data.jastrow_terms.push(JastrowTerm {
            site1: 0,
            site2: 1,
            value: Complex64::new(0.0, 0.0),
            is_complex: true,
        });
        assert!(state_from_data(&data)
            .unwrap()
            .sr_opt
            .sr_opt_oo_real
            .is_empty());
    }

    #[test]
    fn imaginary_loaded_parameter_does_not_change_c_header_mode() {
        let mut data = data();
        data.slater_params[data.orbital_terms[0].idx as usize].im = 0.5;
        let state = state_from_data(&data).unwrap();
        // C's AllComplexFlag is decided from definition headers before
        // loading values; an imaginary overlay cannot change the buffers.
        assert!(!state.sr_opt.sr_opt_oo_real.is_empty());
    }

    #[test]
    fn all_real_parameters_allocate_real_state() {
        assert!(!state_from_data(&data())
            .unwrap()
            .sr_opt
            .sr_opt_oo_real
            .is_empty());
    }

    #[test]
    fn physcal_state_allocates_green_measurement_buffers() {
        let mut data = data();
        data.modpara.vmc_calc_mode = 1;
        data.green_one_terms.push(GreenOneTerm {
            site1: 0,
            spin1: Spin::Up,
            site2: 1,
            spin2: Spin::Up,
        });
        data.green_two_terms.push(GreenTwoTerm {
            site1: 0,
            spin1: Spin::Up,
            site2: 1,
            spin2: Spin::Up,
            site3: 1,
            spin3: Spin::Down,
            site4: 0,
            spin4: Spin::Down,
        });
        data.green_two_ex_terms.push(GreenTwoExTerm {
            site1: 0,
            spin1: Spin::Up,
            site2: 1,
            spin2: Spin::Up,
            site3: 1,
            spin3: Spin::Down,
            site4: 0,
            spin4: Spin::Down,
        });
        let state = state_from_data(&data).unwrap();
        let phys = state.phys_quantities.expect("PhysCal buffers");
        assert_eq!(phys.local_cis_ajs.len(), 1);
        assert_eq!(phys.phys_cis_ajs_ckt_alt.len(), 1);
        assert_eq!(phys.local_cis_ajs_ckt_alt_dc.len(), 1);
    }

    #[test]
    fn physcal_preparation_loads_fixed_parameters_before_rng_consumption() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "../../extern/Julia-mVMC/test/integration/reference/heisenberg_chain_real/physcal_ref",
        );
        // Decode the reference record independently of the production loader:
        // six summary fields, two projection triples, then twelve Slater triples.
        let fields: Vec<f64> = fs::read_to_string(root.join("zqp_opt.dat"))
            .unwrap()
            .split_whitespace()
            .map(|field| field.parse().unwrap())
            .collect();
        assert_eq!(fields.len(), 6 + 3 * 14);
        let raw_slater: Vec<f64> = fields[12..]
            .as_chunks::<3>()
            .0
            .iter()
            .map(|triple| {
                assert_eq!(triple[1], 0.0);
                triple[0]
            })
            .collect();
        let max = raw_slater
            .iter()
            .map(|value| value.abs())
            .fold(0.0, f64::max);
        let expected_slater: Vec<f64> =
            raw_slater.iter().map(|value| value * (4.0 / max)).collect();
        let prepared = prepare_phys_cal_from_namelist(
            root.join("inputs/namelist.def"),
            root.join("zqp_opt.dat"),
            "real",
            Some(11272),
        )
        .expect("fixed PhysCal preparation");
        assert_eq!(prepared.n_para_consumed, 14);
        assert_eq!(prepared.data.modpara.vmc_calc_mode, 1);
        assert_eq!(
            prepared.data.projection_parameters(),
            vec![
                Complex64::new(fields[6], fields[7]),
                Complex64::new(fields[9], fields[10])
            ]
        );
        let actual_slater: Vec<f64> = prepared
            .data
            .slater_params
            .iter()
            .map(|value| {
                assert_eq!(value.im, 0.0);
                value.re
            })
            .collect();
        // One positive rescaling of the independently decoded fixed values.
        crate::numerical_comparison::assert_values_close(
            actual_slater,
            expected_slater,
            32.0 * f64::EPSILON,
            32.0 * f64::EPSILON,
            "fixed PhysCal Slater normalization",
        );
        let mut actual_rng = prepared.rng;
        let mut expected_rng = Sfmt19937Rng::new(11272);
        for _ in 0..624 {
            assert_eq!(actual_rng.gen_rand32(), expected_rng.gen_rand32());
        }
    }

    #[test]
    fn physcal_iteration_keeps_fixed_parameters_unchanged() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "../../extern/Julia-mVMC/test/integration/reference/heisenberg_chain_real/physcal_ref",
        );
        let mut preparation = prepare_phys_cal_from_namelist(
            root.join("inputs/namelist.def"),
            root.join("zqp_opt.dat"),
            "real",
            Some(11272),
        )
        .unwrap();
        preparation.data.modpara.n_data_qty_smp = 1;
        let before = preparation.data.slater_params.clone();
        let projection_before = preparation.data.projection_parameters();
        let result = vmc_phys_cal(preparation).unwrap();
        assert_eq!(result.iterations, 1);
        assert_eq!(result.data.slater_params, before);
        assert_eq!(result.data.projection_parameters(), projection_before);
    }

    #[test]
    fn zero_translation_count_is_rejected_before_mutation_rng_or_output() {
        let mut data = data();
        data.modpara.nmp_trans = 0;
        data.modpara.nsp_gauss_leg = 3;
        let mut state = state_from_data(&data).unwrap();
        assert_eq!(state.slater_matrix.slater_elm.n_qp_full(), 0);
        assert_eq!(state.slater_matrix.slater_elm_real.n_qp_full(), 0);
        let parameters = data.slater_params.clone();
        let mut rng = Sfmt19937Rng::new(1);
        let mut expected_rng = rng.clone();
        let dir = std::env::temp_dir().join(format!("mvmc-c-zero-count-{}", std::process::id()));
        assert!(!dir.exists());
        let err = vmc_para_opt(
            &mut data,
            &mut state,
            &mut rng,
            Some(&dir),
            &SingleProcessReducer,
            OptimizationOptions::default(),
        )
        .unwrap_err();
        assert!(err.contains("NMPTrans"), "{err}");
        assert_eq!(data.modpara.nmp_trans, 0);
        assert_eq!(data.slater_params, parameters);
        for _ in 0..624 {
            assert_eq!(rng.gen_rand32(), expected_rng.gen_rand32());
        }
        assert!(!dir.exists());
    }

    #[test]
    fn pure_general_and_ap_parallel_runs_have_identical_chain_and_updates() {
        let namelist = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../extern/Julia-mVMC/examples/inputs/heisenberg_chain_fsz/namelist.def");
        let template = parse_expert_mode_files(namelist).unwrap();
        let run = |pure_general| {
            let mut data = template.clone();
            data.modpara.nsr_opt_itr_step = 1;
            data.modpara.nsr_opt_itr_smp = 1;
            if pure_general {
                let nsite = data.modpara.nsite;
                let ap = data.n_orbital_anti_parallel;
                for term in &mut data.orbital_terms {
                    if term.idx < ap {
                        term.site2 += nsite;
                    } else if (term.idx - ap) % 2 == 1 {
                        term.site1 += nsite;
                        term.site2 += nsite;
                    }
                }
                data.i_flg_orbital_anti_parallel = 0;
                data.i_flg_orbital_parallel = 0;
                data.orbital_idx_matrix = None;
                data.orbital_sgn_matrix = None;
            }
            let mut rng = Sfmt19937Rng::new(1);
            init_parameter(&mut data, &mut rng).unwrap();
            sync_modified_parameter(&mut data, true);
            init_qp_weight(&mut data);
            let mut state = state_from_data(&data).unwrap();
            let output = std::env::temp_dir().join(format!(
                "mvmc-general-trajectory-{}-{pure_general}",
                std::process::id()
            ));
            vmc_para_opt(
                &mut data,
                &mut state,
                &mut rng,
                Some(&output),
                &SingleProcessReducer,
                OptimizationOptions::default(),
            )
            .unwrap();
            fs::remove_dir_all(output).unwrap();
            let words: Vec<_> = (0..624).map(|_| rng.gen_rand32()).collect();
            let values: Vec<_> = data
                .orbital_terms
                .iter()
                .map(|term| data.slater_params[term.idx as usize])
                .collect();
            (state, words, values)
        };
        let (general_state, general_words, general_values) = run(true);
        let (ap_state, ap_words, ap_values) = run(false);
        assert_eq!(general_values, ap_values);
        assert_eq!(
            general_state.electron_config.ele_idx,
            ap_state.electron_config.ele_idx
        );
        assert_eq!(
            general_state.electron_config.ele_spn,
            ap_state.electron_config.ele_spn
        );
        assert_eq!(general_state.energy.etot, ap_state.energy.etot);
        assert_eq!(general_words, ap_words);
    }

    #[test]
    fn explicit_zero_complex_flags_override_imaginary_values() {
        let mut data = data();
        data.slater_params[data.orbital_terms[0].idx as usize].im = 0.5;
        data.complex_flags = vec![0, 0];
        assert!(!state_from_data(&data)
            .unwrap()
            .sr_opt
            .sr_opt_oo_real
            .is_empty());
    }

    #[test]
    fn explicit_nonzero_complex_flags_select_complex_state() {
        let mut data = data();
        data.complex_flags = vec![0, -1];
        assert!(state_from_data(&data)
            .unwrap()
            .sr_opt
            .sr_opt_oo_real
            .is_empty());
    }

    #[test]
    fn loaded_signed_cancellation_reaches_state_and_explicit_override_remains_distinct() {
        let mut data = data();
        data.doublon_holon_2site_complex = true;
        data.doublon_holon_4site_complex = true;
        data.native_complex_headers.insert("DH2".to_string(), 1);
        data.native_complex_headers.insert("DH4".to_string(), -1);
        data.native_complex_declarations
            .insert("DH2".to_string(), true);
        data.native_complex_declarations
            .insert("DH4".to_string(), true);
        assert_eq!(get_all_complex_flag(&data), Ok(false));
        assert!(!state_from_data(&data)
            .unwrap()
            .sr_opt
            .sr_opt_oo_real
            .is_empty());
        data.complex_flags = vec![1];
        assert_eq!(get_all_complex_flag(&data), Ok(true));
        assert!(state_from_data(&data)
            .unwrap()
            .sr_opt
            .sr_opt_oo_real
            .is_empty());
    }

    #[test]
    fn stale_loaded_declaration_is_an_error_in_validation_and_state_construction() {
        let mut data = data();
        data.native_complex_headers.insert("DH2".to_string(), 0);
        data.native_complex_declarations
            .insert("DH2".to_string(), false);
        data.doublon_holon_2site_complex = true;
        assert!(get_all_complex_flag(&data).is_err());
        assert!(state_from_data(&data).is_err());
        assert!(crate::validation::validate_para_opt(&data).is_err());
        assert!(crate::validation::validate_phys_cal(&data).is_err());
    }

    #[test]
    fn modpara_flag_does_not_override_runtime_factor_inference() {
        let mut data = data();
        data.modpara.complex_flag = 1;
        assert!(!state_from_data(&data)
            .unwrap()
            .sr_opt
            .sr_opt_oo_real
            .is_empty());
    }

    #[test]
    fn imaginary_projection_values_do_not_change_c_header_mode() {
        for gutzwiller in [true, false] {
            let mut data = data();
            if gutzwiller {
                data.gutzwiller_terms.push(GutzwillerTerm {
                    site: 0,
                    value: Complex64::new(0.0, 0.5),
                    is_complex: false,
                });
            } else {
                data.jastrow_terms.push(JastrowTerm {
                    site1: 0,
                    site2: 1,
                    value: Complex64::new(0.0, 0.5),
                    is_complex: false,
                });
            }
            let state = state_from_data(&data).unwrap();
            assert!(!state.sr_opt.sr_opt_oo_real.is_empty());
            assert!(!state.sr_opt.sr_opt_ho_real.is_empty());
            assert!(!state.sr_opt.sr_opt_o_real.is_empty());
            assert!(!state.sr_opt.sr_opt_o_store_real.is_empty());
        }
    }
}

/// Summary returned by [`run_para_opt_from_namelist`].
///
/// Mirrors the `NamedTuple` returned by Julia's `run_para_opt_from_namelist`.
/// Non-output ranks return empty rows/means and NaN energy, without file readback.
/// Root readback errors still propagate collectively to every rank.
#[derive(Debug, Clone)]
pub struct RunSummary {
    /// Successful optimizer status (failures return an error).
    pub status: i32,
    /// Absolute output directory.
    pub output_dir: std::path::PathBuf,
    /// First effective_nsteps output rows, trimmed as in Julia.
    pub zvo_first_n: Vec<String>,
    /// Means of the first two columns over the final effective_nsmp rows.
    pub ctest_values: Vec<f64>,
    /// Last step's total energy divided by Nsite.
    pub final_energy_per_site: f64,
    /// NSROptItrStep used for this run.
    pub effective_nsteps: usize,
    /// NSROptItrSmp used for this run.
    pub effective_nsmp: usize,
}

fn accumulate_observables<const TIMED: bool, R: Reducer + ?Sized>(
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    all_complex: bool,
    use_fsz: bool,
    timer: &mut CTimer<TIMED>,
    reducer: &R,
) {
    if data.modpara.vmc_calc_mode == 1 {
        // PhysCal does not need an owned SR cache. Retain its existing
        // in-place publication arithmetic without allocating a local shape.
        accumulate_observables_local(data, state, all_complex, use_fsz, timer, reducer);
        crate::sr_accumulator::publish_physcal_in_place(&mut state.sr_opt);
        return;
    }
    let mut local = crate::sr_accumulator::SrMeasurement::begin(state);
    accumulate_observables_local(data, local.state(), all_complex, use_fsz, timer, reducer);
    local.finish();
}

fn accumulate_observables_local<const TIMED: bool, R: Reducer + ?Sized>(
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    all_complex: bool,
    use_fsz: bool,
    timer: &mut CTimer<TIMED>,
    reducer: &R,
) {
    let diag = timer.diagnostics.maincal && !use_fsz;
    timer.start_diag(940, diag);
    timer.start_diag(941, diag);
    let n_site = data.modpara.nsite.max(0) as usize;
    let n_elec = data.modpara.nelec.max(0) as usize;
    let n_size = 2 * n_elec;
    let n_qp_full = state.slater_matrix.pf_m.len();
    let n_vmc_sample = data.modpara.nvmc_sample.max(0) as usize;
    let n_proj = data.projection_layout().n_proj;
    let sr_opt_size = state.sr_opt.sr_opt_size;
    let use_store = data.modpara.nstore_o != 0 || data.modpara.nsrcg != 0;
    state.sr_opt.sr_opt_o_store.fill(Complex64::new(0.0, 0.0));
    state.sr_opt.sr_opt_o_store_real.fill(0.0);
    let n_rbm = data.count_rbm_parameters();
    let n_orb_total = n_slater(data);
    let pool = crate::state::ThreadedPfaPackWorkspace::new(n_size, 1);
    let mut slater_derivative_scratch = crate::slater_derivative::SlaterDerivativeScratch::new();

    timer.stop_diag(941, diag);
    timer.stop_diag(940, diag);
    // C VMCMainCal(comm_child1), vmccal.c:103–114. The saved chain and its
    // allocation remain full length even for ranks with no measurement work.
    let measurement_range = if reducer.supports_grouped_sampling() {
        crate::parallel::partition_range(n_vmc_sample, reducer.world_size(), reducer.rank())
    } else {
        // NSplitSize=1 makes comm_child1 MPI_COMM_SELF, not comm0/world.
        // Every independent chain measures all its own saved configurations.
        0..n_vmc_sample
    };
    for sample in measurement_range {
        timer.start_diag(940, diag);
        timer.start_diag(942, diag);
        let ele_idx = state.electron_config.ele_idx_slice(sample).to_vec();
        if ele_idx.iter().all(|&v| v == 0) || ele_idx.iter().all(|&v| v < 0) {
            timer.stop_diag(942, diag);
            timer.stop_diag(940, diag);
            continue;
        }
        let ele_cfg = state.electron_config.ele_cfg_slice(sample).to_vec();
        let ele_num = state.electron_config.ele_num_slice(sample).to_vec();
        let ele_spn = if use_fsz {
            state.electron_config.ele_spn_slice(sample).to_vec()
        } else {
            Vec::new()
        };
        let ele_proj_cnt = if n_proj > 0 {
            state.electron_config.ele_proj_cnt_slice(sample).to_vec()
        } else {
            Vec::new()
        };

        timer.stop_diag(942, diag);
        timer.stop_diag(940, diag);
        timer.start(40);
        // Refresh Pfaffian for the saved walker.
        let factor_scope = if !use_fsz && !all_complex {
            begin_first_real_factor(data, state, sample)
        } else {
            None
        };
        let info = if use_fsz {
            refresh_fsz_observation_matrix(data, state, all_complex, &ele_idx, &ele_spn, &pool)
                .err()
        } else if all_complex {
            crate::pfaffian::calc_m_all_complex(
                &ele_idx,
                &state.slater_matrix.slater_elm,
                &mut state.slater_matrix.inv_m,
                &mut state.slater_matrix.pf_m,
                0,
                n_qp_full,
                n_site,
                n_elec,
                &pool,
            )
            .err()
        } else {
            crate::pfaffian::calc_m_all_real(
                &ele_idx,
                &state.slater_matrix.slater_elm_real,
                &mut state.slater_matrix.inv_m_real,
                &mut state.slater_matrix.pf_m_real,
                0,
                n_qp_full,
                n_site,
                n_elec,
                &pool,
            )
            .err()
        };
        drop(factor_scope);
        timer.stop(40);
        timer.start_diag(940, diag);
        timer.start_diag(943, diag);
        if info.is_some() {
            timer.stop_diag(943, diag);
            timer.stop_diag(940, diag);
            continue;
        }
        if !all_complex {
            for qp in 0..n_qp_full {
                let real_plane = state.slater_matrix.inv_m_real.qp_matrix_slice(qp);
                // Copy only the matrix, leaving the per-QP inverse pad untouched.
                crate::threading::copy_real_to_complex(
                    &mut state.slater_matrix.inv_m.qp_matrix_slice_mut(qp)[..n_size * n_size],
                    &real_plane[..n_size * n_size],
                );
            }
            crate::threading::copy_real_to_complex(
                &mut state.slater_matrix.pf_m[..n_qp_full],
                &state.slater_matrix.pf_m_real[..n_qp_full],
            );
        }
        timer.stop_diag(943, diag);
        timer.stop_diag(940, diag);
        timer.start_diag(940, diag);
        timer.start_diag(944, diag);
        let ip = if all_complex {
            crate::observables::calculate_ip_complex(&state.slater_matrix.pf_m, 0, n_qp_full, data)
        } else {
            Complex64::new(
                crate::observables::calculate_ip_real(
                    &state.slater_matrix.pf_m_real,
                    0,
                    n_qp_full,
                    data,
                ),
                0.0,
            )
        };
        timer.stop_diag(944, diag);
        timer.stop_diag(940, diag);
        timer.start_diag(940, diag);
        timer.start_diag(945, diag);
        if ip.norm() < 1.0e-100 {
            timer.stop_diag(945, diag);
            timer.stop_diag(940, diag);
            continue;
        }
        let w = 1.0;
        timer.stop_diag(945, diag);
        timer.stop_diag(940, diag);
        timer.start(41);
        let e = if use_fsz {
            crate::observables::calculate_local_energy_fsz_timed(
                ip,
                data,
                state,
                &ele_idx,
                &ele_cfg,
                &ele_num,
                &ele_proj_cnt,
                &ele_spn,
                timer,
            )
        } else {
            crate::observables::calculate_local_energy_timed(
                ip,
                data,
                state,
                &ele_idx,
                &ele_cfg,
                &ele_num,
                &ele_proj_cnt,
                timer,
            )
        };
        timer.stop(41);
        // Julia rejects the sum, including overflow of otherwise finite parts.
        if !(e.re + e.im).is_finite() {
            continue;
        }
        timer.start_diag(940, diag);
        timer.start_diag(946, diag);
        let sz = crate::observables::calculate_sz(&ele_num, n_site);

        state.energy.wc += Complex64::new(w, 0.0);
        state.energy.etot += Complex64::new(w, 0.0) * e;
        state.energy.etot2 += Complex64::new(w, 0.0) * e.conj() * e;
        state.energy.sztot += Complex64::new(w * sz, 0.0);
        state.energy.sztot2 += Complex64::new(w * sz * sz, 0.0);

        // The Lanczos path evaluates H on each moved configuration. Transfer
        // Transfer, PairHop and Exchange use the Julia operator order;
        // InterAll and FSZ remain gated until their operator moves are
        // ported.
        if data.modpara.lanczos_mode > 0 && data.inter_all_terms.is_empty() && !use_fsz {
            let h2 = crate::observables::calculate_lanczos_h2_transfer(
                e,
                ip,
                data,
                state,
                &ele_idx,
                &ele_cfg,
                &ele_num,
                &ele_proj_cnt,
                all_complex,
            );
            if let Some(phys) = state.phys_quantities.as_mut() {
                let _ = crate::lanczos::accumulate_lanczos_qqqq(
                    &mut phys.phys_lanczos_qqqq,
                    w,
                    e,
                    h2,
                    all_complex,
                );
            }
        }

        if state.phys_quantities.is_some() && use_fsz {
            crate::observables::calculate_green_func_fsz_timed(
                data,
                state,
                w,
                ip,
                &ele_idx,
                &ele_cfg,
                &ele_num,
                &ele_proj_cnt,
                &ele_spn,
                timer,
            );
        } else if state.phys_quantities.is_some() {
            let (one_body, direct) = crate::observables::ordinary_green_values(
                data,
                state,
                ip,
                &ele_idx,
                &ele_cfg,
                &ele_num,
                &ele_proj_cnt,
            );
            let lanczos_green = if data.modpara.lanczos_mode > 1 {
                Some(crate::observables::calculate_lanczos_green(
                    e,
                    ip,
                    data,
                    state,
                    &ele_idx,
                    &ele_cfg,
                    &ele_num,
                    &ele_proj_cnt,
                    &one_body,
                    &direct,
                    all_complex,
                ))
            } else {
                None
            };
            let phys = state.phys_quantities.as_mut().expect("checked above");
            for (index, value) in one_body.iter().copied().enumerate() {
                phys.local_cis_ajs[index] = value;
                phys.phys_cis_ajs[index] += value;
            }
            crate::observables::accumulate_two_body_gex_sample(
                &mut phys.phys_cis_ajs_ckt_alt,
                &one_body,
                &data.green_two_ex_indices,
                Complex64::new(w, 0.0),
            );
            for (index, value) in direct.into_iter().enumerate() {
                phys.local_cis_ajs_ckt_alt_dc[index] = value;
                phys.phys_cis_ajs_ckt_alt_dc[index] += value;
            }
            if let Some(values) = lanczos_green {
                for (dst, src) in phys.phys_lanczos_qcisajsq.iter_mut().zip(values.one_body) {
                    *dst += src;
                }
                for (dst, src) in phys
                    .phys_lanczos_qcisajscktaltq
                    .iter_mut()
                    .zip(values.factored_two_body)
                {
                    *dst += src;
                }
                for (dst, src) in phys
                    .phys_lanczos_qcisajscktaltq_dc
                    .iter_mut()
                    .zip(values.direct_two_body)
                {
                    *dst += src;
                }
            }
        }

        timer.stop_diag(946, diag);
        timer.stop_diag(940, diag);
        timer.start_diag(940, diag);
        timer.start_diag(948, diag);
        // SR `O` vector — projection diff fills the leading block.
        for slot in state.sr_opt.sr_opt_o.iter_mut() {
            *slot = Complex64::new(0.0, 0.0);
        }
        crate::observables::set_projection_diff(&mut state.sr_opt.sr_opt_o, &ele_proj_cnt, n_proj);
        // Normal Julia main-calculation reserves all RBM derivative slots.
        // Its FSZ main-calculation places Slater immediately after projection.
        if !use_fsz && n_rbm > 0 {
            let cfg = crate::sampling::rbm::RbmConfig::from(data);
            let cnt = crate::sampling::rbm::make_rbm_cnt(&ele_num, &cfg);
            let offset = 2 * (1 + n_proj);
            crate::sampling::rbm::set_rbm_diff(
                &mut state.sr_opt.sr_opt_o[offset..offset + 2 * n_rbm],
                &cnt,
                &ele_num,
                &cfg,
            );
        }
        let slater_offset = 2 * (1 + n_proj + if use_fsz { 0 } else { n_rbm });
        timer.stop_diag(948, diag);
        timer.stop_diag(940, diag);
        if n_orb_total > 0 && slater_offset < state.sr_opt.sr_opt_o.len() {
            timer.start(42);
            let n_copy = (2 * n_orb_total).min(state.sr_opt.sr_opt_o.len() - slater_offset);
            let slater_o = &mut state.sr_opt.sr_opt_o[slater_offset..slater_offset + n_copy];
            if use_fsz {
                crate::slater_derivative::slater_elm_diff_fsz_with_scratch(
                    slater_o,
                    ip,
                    &ele_idx,
                    &ele_spn,
                    data,
                    &state.slater_matrix,
                    &mut slater_derivative_scratch,
                );
            } else {
                timer.start_diag(930, timer.diagnostics.slater);
                crate::slater_derivative::slater_elm_diff_with_scratch_timed(
                    slater_o,
                    ip,
                    &ele_idx,
                    data,
                    &state.slater_matrix,
                    &mut slater_derivative_scratch,
                    timer,
                );
                timer.stop_diag(930, timer.diagnostics.slater);
            }
            timer.stop(42);
        }
        let n_opt = data.count_opt_trans_parameters();
        let opt_offset = slater_offset + 2 * n_orb_total;
        let opt_end = opt_offset + 2 * n_opt;
        if n_opt > 0 && opt_end <= state.sr_opt.sr_opt_o.len() {
            let diag = !use_fsz && timer.diagnostics.maincal;
            timer.start_diag(940, diag);
            timer.start_diag(949, diag);
            crate::observables::opt_trans_diff(
                &mut state.sr_opt.sr_opt_o[opt_offset..opt_end],
                ip,
                data,
                &state.slater_matrix.pf_m,
            );
            timer.stop_diag(949, diag);
            timer.stop_diag(940, diag);
        }
        observe_optimization_measurement(OptimizationMeasurementView {
            data,
            state,
            sample,
            overlap: ip,
            local_energy: e,
            weight: w,
        });
        timer.start(43);
        if all_complex && use_store {
            crate::observables::calculate_oo_store(
                &mut state.sr_opt.sr_opt_ho,
                &mut state.sr_opt.sr_opt_o_store,
                &state.sr_opt.sr_opt_o,
                w,
                e,
                sample,
                sr_opt_size,
            );
        } else if all_complex {
            crate::observables::calculate_oo(
                &mut state.sr_opt.sr_opt_oo,
                &mut state.sr_opt.sr_opt_ho,
                &state.sr_opt.sr_opt_o,
                w,
                e,
                sr_opt_size,
            );
        } else {
            for i in 0..sr_opt_size {
                state.sr_opt.sr_opt_o_real[i] = state.sr_opt.sr_opt_o[2 * i].re;
            }
            if use_store {
                crate::observables::calculate_oo_store_real(
                    &mut state.sr_opt.sr_opt_ho_real,
                    &mut state.sr_opt.sr_opt_o_store_real,
                    &state.sr_opt.sr_opt_o_real,
                    w,
                    e.re,
                    sample,
                    sr_opt_size,
                );
            } else {
                crate::observables::calculate_oo_real(
                    &mut state.sr_opt.sr_opt_oo_real,
                    &mut state.sr_opt.sr_opt_ho_real,
                    &state.sr_opt.sr_opt_o_real,
                    w,
                    e.re,
                    sr_opt_size,
                );
            }
        }
        timer.stop(43);
        OPTIMIZATION_MEASUREMENT_OBSERVER.with(|slot| {
            let observer = slot.borrow().clone();
            if let Some(observer) = observer {
                observer.accumulated(OptimizationMeasurementView {
                    data,
                    state,
                    sample,
                    overlap: ip,
                    local_energy: e,
                    weight: w,
                });
            }
        });
    }
    observe_physcal_green(data, state, use_fsz);
    normalize_physcal_green(state, use_fsz, all_complex);
    if use_store {
        timer.start(45);
        let options = crate::observables::StoreFinalization {
            sample_start: 0,
            diagonal_only: data.modpara.nsrcg != 0,
        };
        if all_complex {
            crate::observables::finalize_oo_store(
                &mut state.sr_opt.sr_opt_oo,
                &state.sr_opt.sr_opt_o_store,
                sr_opt_size,
                n_vmc_sample,
                options,
            );
        } else {
            crate::observables::finalize_oo_store_real(
                &mut state.sr_opt.sr_opt_oo_real,
                &state.sr_opt.sr_opt_o_store_real,
                sr_opt_size,
                n_vmc_sample,
                options,
            );
        }
        timer.stop(45);
    }
}

/// Rebuild the saved-walker Pfaffian using the mode selected by C's
/// `AllComplexFlag`. Real-FSZ must refresh its real shadows before the shared
/// observable kernels consume them.
fn refresh_fsz_observation_matrix(
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    all_complex: bool,
    ele_idx: &[i64],
    ele_spn: &[i64],
    pool: &ThreadedPfaPackWorkspace,
) -> Result<(), crate::pfaffian::CalcMAllError> {
    let n_site = data.modpara.nsite.max(0) as usize;
    let n_elec = data.modpara.nelec.max(0) as usize;
    let n_qp_full = state.slater_matrix.pf_m.len();
    if all_complex {
        crate::pfaffian::calc_m_all_fsz_complex(
            ele_idx,
            ele_spn,
            &state.slater_matrix.slater_elm,
            &mut state.slater_matrix.inv_m,
            &mut state.slater_matrix.pf_m,
            0,
            n_qp_full,
            n_site,
            n_elec,
            pool,
        )
    } else {
        crate::pfaffian::calc_m_all_fsz_real(
            ele_idx,
            ele_spn,
            &mut state.slater_matrix,
            0,
            n_qp_full,
            n_site,
            n_elec,
            pool,
        )?;
        sync_real_fsz_shadow(state);
        Ok(())
    }
}

#[cfg(test)]
use crate::julia_fixture;

#[cfg(test)]
mod callback_tests {
    mod historical_overlay_stage {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/support/historical_overlay_stage.rs"
        ));
    }
    use super::julia_fixture;
    use super::*;

    use super::reference_slater::{declared_output, declared_slater_rows};

    fn declared_history_values(data: &ExpertModeData, historical: Vec<f64>) -> Vec<f64> {
        let prefix = 2 * data.projection_layout().n_proj;
        let mapped: Vec<[f64; 2]> = historical[prefix..]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|row| [row[0], row[1]])
            .collect();
        historical[..prefix]
            .iter()
            .copied()
            .chain(declared_slater_rows(data, &mapped).into_iter().flatten())
            .collect()
    }

    fn assert_complete_dh_history(
        data: &ExpertModeData,
        fixtures: &Path,
        case: &str,
        step: usize,
        point: &crate::state::OptDataPoint,
        archived: Vec<f64>,
    ) {
        let read = |suffix: &str| {
            let relative = format!("sr_direct/{case}_runner/step-{step}-{suffix}");
            let path = if case.ends_with("fsz") && native_fsz_fixture::directory(fixtures).is_some()
            {
                native_fsz_fixture::resolve(
                    native_fsz_fixture::directory(fixtures)
                        .unwrap()
                        .join(relative),
                )
            } else {
                julia_fixture::fixture_path(fixtures, relative)
            };
            julia_fixture::read_text(&path).unwrap_or_else(|error| {
                panic!(
                    "{case} history prefix{step} {suffix} at {}: {error}",
                    path.display()
                )
            })
        };
        // Independent prefix records contain DH slots omitted by the old
        // history observer. They are not reconstructed from the Rust run.
        let complete: Vec<f64> = read("parameters.txt")
            .split_whitespace()
            .map(|s| f64::from_bits(u64::from_str_radix(s, 16).unwrap()))
            .collect();
        let old_prefix = 2 * (data.gutzwiller_terms.len() + data.jastrow_terms.len());
        let projection = 2 * data.projection_layout().n_proj;
        crate::numerical_comparison::assert_values_close(
            complete[..old_prefix]
                .iter()
                .chain(&complete[projection..])
                .copied(),
            archived,
            1e-11,
            1e-11,
            format!("{case} step {step}: archived history projection"),
        );
        let expected = declared_history_values(data, complete);
        assert_eq!(point.parameters.len(), data.count_variational_parameters());
        assert_eq!(expected.len(), 2 * point.parameters.len());
        crate::numerical_comparison::assert_values_close(
            point.parameters.iter().flat_map(|v| [v.re, v.im]),
            expected,
            1e-11,
            1e-11,
            format!("{case} step {step}: complete declared history"),
        );
        // `read` resolves the checked-in reference prefix, never the Rust output directory.
        // The archived observer's pre-SR output carries the independently captured Etot2.
        let output = read("zvo_var.dat");
        let measured: Vec<f64> = output
            .lines()
            .last()
            .unwrap()
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        crate::numerical_comparison::assert_values_close(
            [point.energy_squared.re, point.energy_squared.im],
            [measured[3], measured[4]],
            1e-11,
            1e-11,
            format!("{case} step {step}: archived reference zvo_var Etot2"),
        );
    }

    // Archived Julia output omitted RBM coefficients. C-compatible output
    // includes them in declared order. Preserve every archived output field
    // and insert Rust-observed pre-SR coefficients for each row. This is only
    // a consistency check, not an independent RBM output oracle; replace it
    // with separately captured complete C-contract histories.
    fn declared_runner_output(
        data: &ExpertModeData,
        name: &str,
        historical: String,
        rbm_before_sr: &[Vec<Complex64>],
    ) -> String {
        let historical = declared_output(data, name, historical);
        if data.rbm_params.is_empty() {
            return historical;
        }
        let prefix = data.gutzwiller_terms.len() + data.jastrow_terms.len();
        let format = crate::io::format_c_double;
        match name {
            "zvo_var.dat" => historical
                .lines()
                .enumerate()
                .map(|(step, line)| {
                    let rows: Vec<_> = line.split_inclusive("0.0 ").collect();
                    let rbm: String = rbm_before_sr[step]
                        .iter()
                        .map(|value| format!("{} {} 0.0 ", format(value.re), format(value.im)))
                        .collect();
                    rows[..2 + prefix].concat() + &rbm + &rows[2 + prefix..].concat() + "\n"
                })
                .collect(),
            "zqp_opt.dat" => {
                let rows: Vec<_> = historical.split_inclusive('\n').collect();
                let rbm: String = data
                    .rbm_params
                    .iter()
                    .map(|value| format!("{} {} \n", format(value.re), format(value.im)))
                    .collect();
                rows[..prefix].concat() + &rbm + &rows[prefix..].concat()
            }
            _ => historical,
        }
    }

    fn prepared(steps: i64) -> (ExpertModeData, VmcOptimizationState, Sfmt19937Rng) {
        prepared_case(steps, "heisenberg_chain_real")
    }

    #[test]
    fn remote_sr_failure_restores_successful_local_parameter_update() {
        struct RemoteSrFailure;
        impl Reducer for RemoteSrFailure {
            fn sampling_max_info(&self, info: i32) -> Result<i32, String> {
                // This double's comm1 peers share the local initialization
                // INFO. Only the later SR status reduction injects failure.
                // Signed MAX(info, info) preserves negative INFO unchanged.
                Ok(info)
            }
            fn allreduce_sum_f64(&self, _: &mut [f64]) {}
            fn allreduce_sum_c64(&self, _: &mut [Complex64]) {}
            fn allreduce_sum_i64(&self, values: &mut [i64]) {
                if values.len() == 1 {
                    assert_eq!(values[0], 0, "local SR must succeed before peer failure");
                    values[0] = 1;
                }
            }
            fn any_failure(&self, failed: bool) -> bool {
                failed
            }
            fn reduction_size(&self) -> usize {
                2
            }
        }
        let (mut data, mut state, mut rng) = prepared(1);
        let before = data.clone();
        let output = fresh_output_directory().unwrap();
        let error = vmc_para_opt(
            &mut data,
            &mut state,
            &mut rng,
            Some(&output),
            &RemoteSrFailure,
            OptimizationOptions::default(),
        )
        .unwrap_err();
        assert!(error.contains("local status 0"), "{error}");
        assert_eq!(data.slater_params, before.slater_params);
        assert_eq!(data.projection_parameters(), before.projection_parameters());
        assert_eq!(data.qp_weights, before.qp_weights);
        assert_eq!(data.optimization_flags, before.optimization_flags);
        fs::remove_dir_all(output).unwrap();
    }

    fn prepared_case(
        steps: i64,
        name: &str,
    ) -> (ExpertModeData, VmcOptimizationState, Sfmt19937Rng) {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../extern/Julia-mVMC/examples/inputs")
            .join(name)
            .join("namelist.def");
        prepared_namelist(steps, &path)
    }

    fn prepared_namelist(
        steps: i64,
        path: &Path,
    ) -> (ExpertModeData, VmcOptimizationState, Sfmt19937Rng) {
        // These Julia mixed-DH models relied on global complex mode enabling
        // real-orbital imaginary flags. C uses only orbital headers; explicit
        // complex AP replacements preserve these historical SR/RNG workloads.
        // RBM legacy workloads also use explicit binary flags: C flag 2 is
        // fixed for SR, while Julia had converted it to true. These helpers
        // never change flags on a parsed production model.
        let input = path.to_string_lossy();
        let replacement = [
            ("dh2/production_cmp/namelist.def", "dh2_cmp"),
            ("dh4/production_dh4_cmp/namelist.def", "dh4_cmp"),
            ("dh4/production_dh24_cmp/namelist.def", "dh24_cmp"),
            ("rbm/run_rbm_real/namelist.def", "rbm_real"),
            ("rbm/run_rbm_cmp/namelist.def", "rbm_cmp"),
            ("rbm/run_rbm_general_cmp/namelist.def", "rbm_general_cmp"),
            ("rbm/run_rbm_dh24_cmp/namelist.def", "rbm_dh24_cmp"),
            ("rbm/run_rbm_fsz/namelist.def", "rbm_fsz"),
            (
                "opttrans/run_opt_dh24_rbm_cmp/namelist.def",
                "opt_dh24_rbm_cmp",
            ),
        ]
        .into_iter()
        .find(|(suffix, _)| input.ends_with(suffix))
        .map(|(_, name)| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
                "../../tests/fixtures/c_orbital_inputs/namelist_{name}.def"
            ))
        });
        let path = replacement.as_deref().unwrap_or(path);

        // C-complete RBM controls include fixed-zero padding. Reconstruct the
        // archived sparse model for these Julia trajectory regressions, just
        // as the kernel tests do; production keeps the full declared widths.
        let mut data = crate::historical_orbital_model::historical_kernel_model(path).unwrap();
        data.modpara.nsr_opt_itr_step = steps;
        data.modpara.nsr_opt_itr_smp = steps;
        let mut rng = Sfmt19937Rng::new(1);
        init_parameter(&mut data, &mut rng).unwrap();
        if !data.doublon_holon_2site_indices.is_empty()
            || !data.doublon_holon_4site_indices.is_empty()
            || data.has_rbm_terms()
        {
            historical_overlay_stage::read_input_parameters(&mut data, path).unwrap();
        }
        sync_modified_parameter(&mut data, true);
        init_qp_weight(&mut data);
        let state = state_from_data(&data).unwrap();
        (data, state, rng)
    }

    #[test]
    fn fsz_measurements_preserve_real_and_complex_optimization_sampling_and_rng() {
        use mvmc_expert_parsers::{GreenOneTerm, GreenTwoTerm, Spin};
        for complex in [false, true] {
            let (mut baseline, _, base_rng) = prepared_case(3, "heisenberg_chain_fsz");
            baseline.complex_flags = vec![i64::from(complex)];
            if !complex {
                for value in &mut baseline.slater_params {
                    value.im = 0.0;
                }
                sync_modified_parameter(&mut baseline, true);
            }
            assert_eq!(get_all_complex_flag(&baseline).unwrap(), complex);
            baseline.green_one_terms = vec![
                GreenOneTerm {
                    site1: 0,
                    spin1: Spin::Up,
                    site2: 0,
                    spin2: Spin::Up,
                },
                GreenOneTerm {
                    site1: 0,
                    spin1: Spin::Up,
                    site2: 1,
                    spin2: Spin::Down,
                },
                GreenOneTerm {
                    site1: 1,
                    spin1: Spin::Down,
                    site2: 0,
                    spin2: Spin::Up,
                },
            ];
            baseline.green_two_terms = vec![
                GreenTwoTerm {
                    site1: 0,
                    spin1: Spin::Up,
                    site2: 1,
                    spin2: Spin::Up,
                    site3: 1,
                    spin3: Spin::Down,
                    site4: 0,
                    spin4: Spin::Down,
                },
                GreenTwoTerm {
                    site1: 0,
                    spin1: Spin::Up,
                    site2: 0,
                    spin2: Spin::Up,
                    site3: 1,
                    spin3: Spin::Down,
                    site4: 1,
                    spin4: Spin::Down,
                },
            ];
            baseline.green_two_terms.push(baseline.green_two_terms[0]);
            baseline.green_two_ex_indices = vec![(0, 0), (1, 2), (1, 2)];
            let mut observed = baseline.clone();
            let mut base_state = state_from_data(&baseline).unwrap();
            let mut state = state_from_data(&observed).unwrap();
            state.phys_quantities = Some(crate::state::PhysicalQuantities::zeros(3, 3, 3));
            let mut base_rng = base_rng;
            let mut rng = base_rng.clone();
            let base_dir = fresh_output_directory().unwrap();
            let dir = fresh_output_directory().unwrap();
            for step in 0..3 {
                baseline.modpara.nsr_opt_itr_step = 1;
                observed.modpara.nsr_opt_itr_step = 1;
                baseline.modpara.nsr_opt_itr_smp = 1;
                observed.modpara.nsr_opt_itr_smp = 1;
                vmc_para_opt(
                    &mut baseline,
                    &mut base_state,
                    &mut base_rng,
                    Some(&base_dir),
                    &SingleProcessReducer,
                    OptimizationOptions::default(),
                )
                .unwrap();
                vmc_para_opt(
                    &mut observed,
                    &mut state,
                    &mut rng,
                    Some(&dir),
                    &SingleProcessReducer,
                    OptimizationOptions::default(),
                )
                .unwrap();
                assert_eq!(
                    observed.slater_params, baseline.slater_params,
                    "complex={complex}, step={step}"
                );
                assert_eq!(state.electron_config, base_state.electron_config);
                assert_eq!(state.slater_matrix, base_state.slater_matrix);
                assert_eq!(state.energy, base_state.energy);
                assert_eq!(state.sr_opt, base_state.sr_opt);
                assert_eq!(state.opt_data, base_state.opt_data);
                let phys = state.phys_quantities.as_ref().unwrap();
                assert!(phys.phys_cis_ajs[0].norm() > 0.0);
                // Compare a full SFMT block after each SR step without advancing
                // either live generator, so the next chain continues unchanged.
                let mut actual = rng.clone();
                let mut expected = base_rng.clone();
                for word in 0..624 {
                    assert_eq!(
                        actual.gen_rand32(),
                        expected.gen_rand32(),
                        "complex={complex}, step={step}, RNG word={word}"
                    );
                }
            }
            fs::remove_dir_all(base_dir).unwrap();
            fs::remove_dir_all(dir).unwrap();
        }
    }

    #[test]
    fn callbacks_observe_post_sync_parameters_and_do_not_change_rng_trajectory() {
        let (mut baseline, mut base_state, mut base_rng) = prepared(3);
        let base_dir = fresh_output_directory().unwrap();
        vmc_para_opt(
            &mut baseline,
            &mut base_state,
            &mut base_rng,
            Some(&base_dir),
            &SingleProcessReducer,
            OptimizationOptions::default(),
        )
        .unwrap();
        let (mut data, mut state, mut rng) = prepared(3);
        let dir = fresh_output_directory().unwrap();
        let mut records = Vec::new();
        let mut callback = |step, data: &mut ExpertModeData, energy, info| {
            records.push((step, data.slater_params.clone(), energy, info));
            Ok(())
        };
        vmc_para_opt(
            &mut data,
            &mut state,
            &mut rng,
            Some(&dir),
            &SingleProcessReducer,
            OptimizationOptions {
                callback: Some(&mut callback),
                skip_sr: false,
            },
        )
        .unwrap();
        assert_eq!(
            records.iter().map(|record| record.0).collect::<Vec<_>>(),
            [0, 1, 2]
        );
        assert!(records.iter().all(|record| record.3 == 0));
        assert_eq!(records[2].1, data.slater_params);
        assert_eq!(records[2].2, state.energy.etot);
        assert_eq!(data.orbital_terms, baseline.orbital_terms);
        assert_eq!(data.slater_params, baseline.slater_params);
        assert_eq!(
            state.electron_config.ele_idx,
            base_state.electron_config.ele_idx
        );
        assert_eq!(
            fs::read(dir.join("zvo_out.dat")).unwrap(),
            fs::read(base_dir.join("zvo_out.dat")).unwrap()
        );
        let mut hash = 0xcbf29ce484222325_u64;
        for _ in 0..624 {
            let word = rng.gen_rand32();
            assert_eq!(word, base_rng.gen_rand32());
            hash = (hash ^ u64::from(word)).wrapping_mul(0x100000001b3);
        }
        assert_eq!(hash, 382483484918994011);
        fs::remove_dir_all(dir).unwrap();
        fs::remove_dir_all(base_dir).unwrap();
    }

    #[test]
    fn mpi_measurement_partition_preserves_full_chain_count_and_rng() {
        struct MeasurementRank(usize);
        impl Reducer for MeasurementRank {
            fn sampling_max_info(&self, info: i32) -> Result<i32, String> {
                // Replicated sampling uses identical comm1 initializer INFO;
                // measurement partitioning must not consume SR collectives.
                Ok(info)
            }
            fn allreduce_sum_f64(&self, _: &mut [f64]) {}
            fn allreduce_sum_c64(&self, _: &mut [Complex64]) {}
            fn allreduce_sum_i64(&self, _: &mut [i64]) {}
            fn world_size(&self) -> usize {
                2
            }
            fn rank(&self) -> usize {
                self.0
            }
            fn supports_grouped_sampling(&self) -> bool {
                true
            }
        }
        let (mut baseline, _, initial_rng) = prepared(1);
        baseline.modpara.nvmc_sample = 3;
        let mut baseline_state = state_from_data(&baseline).unwrap();
        let mut baseline_rng = initial_rng.clone();
        let dir = fresh_output_directory().unwrap();
        vmc_para_opt(
            &mut baseline,
            &mut baseline_state,
            &mut baseline_rng,
            Some(&dir),
            &SingleProcessReducer,
            OptimizationOptions {
                skip_sr: true,
                ..OptimizationOptions::default()
            },
        )
        .unwrap();
        for rank in 0..2 {
            let (mut data, _, _) = prepared(1);
            data.modpara.nvmc_sample = 3;
            let mut state = state_from_data(&data).unwrap();
            let mut rng = initial_rng.clone();
            vmc_para_opt(
                &mut data,
                &mut state,
                &mut rng,
                Some(&dir),
                &MeasurementRank(rank),
                OptimizationOptions {
                    skip_sr: true,
                    ..OptimizationOptions::default()
                },
            )
            .unwrap();
            assert_eq!(data.modpara.nvmc_sample, 3);
            assert_eq!(state.electron_config, baseline_state.electron_config);
            let mut expected_rng = baseline_rng.clone();
            for word in 0..624 {
                assert_eq!(
                    rng.gen_rand32(),
                    expected_rng.gen_rand32(),
                    "rank {rank}, word {word}"
                );
            }
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn sampling_only_stops_after_one_output_and_before_sr_or_final_parameters() {
        let (mut data, mut state, mut rng) = prepared(3);
        data.modpara.dsr_opt_step_dt = f64::NAN; // Prove the solver is skipped.
        let before = data.slater_params.clone();
        let dir = fresh_output_directory().unwrap();
        let mut calls = Vec::new();
        let mut callback = |step, _: &mut ExpertModeData, energy, info| {
            calls.push((step, energy, info));
            Ok(())
        };
        vmc_para_opt(
            &mut data,
            &mut state,
            &mut rng,
            Some(&dir),
            &SingleProcessReducer,
            OptimizationOptions {
                callback: Some(&mut callback),
                skip_sr: true,
            },
        )
        .unwrap();
        assert_eq!(calls, [(0, state.energy.etot, 0)]);
        assert_eq!(before, data.slater_params);
        assert_eq!(
            fs::read_to_string(dir.join("zvo_out.dat"))
                .unwrap()
                .lines()
                .count(),
            1
        );
        assert!(!dir.join("zqp_opt.dat").exists());
        let (mut control, mut control_state, mut control_rng) = prepared(1);
        control.modpara.dsr_opt_step_dt = f64::NAN;
        let control_dir = fresh_output_directory().unwrap();
        assert!(vmc_para_opt(
            &mut control,
            &mut control_state,
            &mut control_rng,
            Some(&control_dir),
            &SingleProcessReducer,
            OptimizationOptions::default()
        )
        .is_err());
        let mut hash = 0xcbf29ce484222325_u64;
        for _ in 0..624 {
            let word = rng.gen_rand32();
            assert_eq!(word, control_rng.gen_rand32());
            hash = (hash ^ u64::from(word)).wrapping_mul(0x100000001b3);
        }
        assert_eq!(hash, 13510181319970448127);
        fs::remove_dir_all(dir).unwrap();
        fs::remove_dir_all(control_dir).unwrap();
    }

    #[test]
    fn callback_errors_propagate_and_failed_sr_does_not_call_callback() {
        let (mut data, mut state, mut rng) = prepared(3);
        let dir = fresh_output_directory().unwrap();
        let mut calls = 0;
        let mut callback = |_, _: &mut ExpertModeData, _, _| {
            calls += 1;
            Err("callback failed".into())
        };
        let error = vmc_para_opt(
            &mut data,
            &mut state,
            &mut rng,
            Some(&dir),
            &SingleProcessReducer,
            OptimizationOptions {
                callback: Some(&mut callback),
                skip_sr: false,
            },
        )
        .unwrap_err();
        assert_eq!(error, "callback failed");
        assert_eq!(calls, 1);
        assert!(!dir.join("zqp_opt.dat").exists());
        let (mut data, mut state, mut rng) = prepared(3);
        data.modpara.dsr_opt_step_dt = f64::NAN;
        let mut callback = |_, _: &mut ExpertModeData, _, _| {
            calls += 1;
            Ok(())
        };
        assert!(vmc_para_opt(
            &mut data,
            &mut state,
            &mut rng,
            Some(&dir),
            &SingleProcessReducer,
            OptimizationOptions {
                callback: Some(&mut callback),
                skip_sr: false
            }
        )
        .is_err());
        assert_eq!(calls, 1);
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn final_window_history_is_stored_before_callback_and_uses_initial_window() {
        for window in [2, 3] {
            let (mut data, mut state, mut rng) = prepared(3);
            data.modpara.nsr_opt_itr_smp = window;
            let dir = fresh_output_directory().unwrap();
            let mut records = Vec::new();
            let mut callback = |step, data: &mut ExpertModeData, energy, _| {
                records.push((
                    step,
                    data.gutzwiller_terms
                        .iter()
                        .map(|t| t.value)
                        .chain(data.jastrow_terms.iter().map(|t| t.value))
                        .chain(data.slater_params.iter().copied())
                        .collect::<Vec<_>>(),
                    energy,
                ));
                data.modpara.nsr_opt_itr_smp = 100; // Julia captures n_smp before the loop.
                data.slater_params[data.orbital_terms[0].idx as usize].re += 0.001;
                Ok(())
            };
            vmc_para_opt(
                &mut data,
                &mut state,
                &mut rng,
                Some(&dir),
                &SingleProcessReducer,
                OptimizationOptions {
                    callback: Some(&mut callback),
                    skip_sr: false,
                },
            )
            .unwrap();
            let effective_window = window;
            assert_eq!(state.opt_data.len(), effective_window as usize);
            for (index, point) in state.opt_data.iter().enumerate() {
                let step = index as i64 + 3 - effective_window;
                assert_eq!(point.parameters, records[step as usize].1);
                assert_eq!(point.energy, records[step as usize].2);
            }
            assert_ne!(state.opt_data.last().unwrap().parameters.last(), None);
            let snapshot = state.opt_data.last().unwrap().parameters.clone();
            data.slater_params[data.orbital_terms[0].idx as usize].re = 999.0;
            assert_eq!(state.opt_data.last().unwrap().parameters, snapshot);
            fs::remove_dir_all(dir).unwrap();
        }
    }
    #[test]
    fn oversized_optimization_window_rejects_before_mutation_output_or_rng() {
        let (mut data, mut state, mut rng) = prepared(3);
        data.modpara.nsr_opt_itr_smp = 5;
        let before_parameters = data.slater_params.clone();
        let before_configs = state.electron_config.ele_idx.clone();
        let mut before_rng = rng.clone();
        let directory = fresh_output_directory().unwrap();
        let error = vmc_para_opt(
            &mut data,
            &mut state,
            &mut rng,
            Some(&directory),
            &SingleProcessReducer,
            OptimizationOptions::default(),
        )
        .unwrap_err();
        assert!(error.contains("must be >= nsmp"), "{error}");
        assert_eq!(data.slater_params, before_parameters);
        assert_eq!(state.electron_config.ele_idx, before_configs);
        assert_eq!(fs::read_dir(&directory).unwrap().count(), 0);
        for _ in 0..624 {
            assert_eq!(rng.gen_rand32(), before_rng.gen_rand32());
        }
        fs::remove_dir(directory).unwrap();
    }

    #[test]
    fn enabled_sections_preserve_parameters_samples_energy_and_rng() {
        for case in [
            "heisenberg_chain_real",
            "heisenberg_chain_cmp",
            "heisenberg_chain_fsz",
            "hubbard_chain_real",
        ] {
            let (mut baseline, mut base_state, mut base_rng) = prepared_case(3, case);
            let base_dir = fresh_output_directory().unwrap();
            vmc_para_opt(
                &mut baseline,
                &mut base_state,
                &mut base_rng,
                Some(&base_dir),
                &SingleProcessReducer,
                OptimizationOptions::default(),
            )
            .unwrap();
            let (mut data, mut state, mut rng) = prepared_case(3, case);
            let dir = fresh_output_directory().unwrap();
            let mut timer = crate::c_timer::CTimer::<true>::new();
            timer.diagnostics = TimerEnv {
                calham1: true,
                slater: true,
                maincal: true,
                weightavg: true,
                ..TimerEnv::default()
            };
            vmc_para_opt_timed(
                &mut data,
                &mut state,
                &mut rng,
                Some(&dir),
                &SingleProcessReducer,
                OptimizationOptions::default(),
                &mut timer,
            )
            .unwrap();
            for id in [
                2, 20, 3, 4, 21, 24, 25, 22, 5, 23, 30, 31, 35, 40, 41, 42, 43, 50, 51, 52, 56, 57,
                70, 71, 72, 960, 962, 965, 966,
            ] {
                assert!(timer.elapsed_ns[id] > 0, "{case}: missing timer {id}");
            }
            if case == "hubbard_chain_real" {
                for id in [32, 60, 61, 62, 63, 920, 921, 922, 924, 927] {
                    assert!(timer.elapsed_ns[id] > 0, "{case}: missing timer {id}");
                }
                assert_eq!(timer.elapsed_ns[33], 0);
            } else {
                for id in [33, 65, 66, 67, 68] {
                    assert!(timer.elapsed_ns[id] > 0, "{case}: missing timer {id}");
                }
            }
            if case != "heisenberg_chain_fsz" {
                for id in [
                    930, 931, 932, 933, 934, 940, 941, 942, 943, 944, 945, 946, 948,
                ] {
                    assert!(timer.elapsed_ns[id] > 0, "{case}: missing diag {id}");
                }
            }
            if case == "heisenberg_chain_fsz" {
                for id in [
                    930, 931, 932, 933, 934, 940, 941, 942, 943, 944, 945, 946, 948,
                ] {
                    assert_eq!(timer.elapsed_ns[id], 0, "unexpected FSZ diagnostic {id}");
                }
            }
            assert!(
                timer.elapsed_ns[2]
                    >= [20, 3, 4, 21, 22, 5, 23]
                        .iter()
                        .map(|&id| timer.elapsed_ns[id])
                        .sum::<u64>()
            );
            assert!(
                timer.elapsed_ns[3]
                    >= [30, 31, 32, 33, 34, 35, 36]
                        .iter()
                        .map(|&id| timer.elapsed_ns[id])
                        .sum::<u64>()
            );
            assert!(
                timer.elapsed_ns[960]
                    >= [962, 965, 966]
                        .iter()
                        .map(|&id| timer.elapsed_ns[id])
                        .sum::<u64>()
            );
            assert_eq!(timer.elapsed_ns[12], 0);
            assert_eq!(timer.elapsed_ns[55], 0);
            assert_eq!(baseline.orbital_terms, data.orbital_terms);
            assert_eq!(baseline.slater_params, data.slater_params);
            assert_eq!(base_state.electron_config, state.electron_config);
            assert_eq!(base_state.energy, state.energy);
            assert_eq!(base_state.opt_data, state.opt_data);
            assert_eq!(
                fs::read(base_dir.join("zvo_out.dat")).unwrap(),
                fs::read(dir.join("zvo_out.dat")).unwrap()
            );
            for _ in 0..624 {
                assert_eq!(base_rng.gen_rand32(), rng.gen_rand32());
            }
            fs::remove_dir_all(base_dir).unwrap();
            fs::remove_dir_all(dir).unwrap();
        }
    }
    #[test]
    fn real_cg_prefixes_match_julia_parameters_samples_energy_and_full_rng_blocks() {
        check_sr_prefixes("real", true, 0);
    }

    #[test]
    fn complex_cg_prefixes_match_julia_parameters_samples_energy_and_full_rng_blocks() {
        check_sr_prefixes("cmp", true, 0);
    }

    #[test]
    fn fsz_cg_prefixes_match_julia_parameters_samples_energy_and_full_rng_blocks() {
        check_sr_prefixes("fsz", true, 0);
    }

    #[test]
    fn general_cg_prefixes_match_julia_parameters_samples_energy_and_full_rng_blocks() {
        check_sr_prefixes("general", true, 0);
    }

    #[test]
    fn interall_fsz_cg_prefixes_match_julia_parameters_spins_samples_energy_and_rng() {
        check_sr_prefixes("interall", true, 0);
    }

    #[test]
    fn dh2_loaded_values_and_rng_match_julia_with_history_in_c_declared_order() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/dh2");
        let scalars = |text: &str| -> Vec<f64> {
            text.split_whitespace()
                .map(|s| f64::from_bits(u64::from_str_radix(s, 16).unwrap()))
                .collect()
        };
        let serialize = |values: Vec<Complex64>| -> Vec<f64> {
            values.iter().flat_map(|v| [v.re, v.im]).collect()
        };
        for mode in ["real", "cmp", "fsz"] {
            let (mut data, mut state, mut rng) =
                prepared_namelist(3, &root.join(format!("production_{mode}/namelist.def")));
            let input = fs::read_to_string(root.join(format!("loaded-{mode}.txt"))).unwrap();
            let mut lines = input.lines().skip(1);
            assert_eq!(
                data.optimization_flags,
                crate::historical_optimization_flags::c_orbital_representation(
                    &data,
                    lines
                        .next()
                        .unwrap()
                        .split_whitespace()
                        .map(|s| s.parse::<i64>().unwrap())
                        .collect()
                )
            );
            let values = data
                .projection_parameters()
                .into_iter()
                .chain(
                    data.orbital_terms
                        .iter()
                        .map(|t| data.slater_params[t.idx as usize]),
                )
                .collect();
            let expected_values = scalars(lines.next().unwrap());
            let mut probe = rng.clone();
            assert_eq!(
                (0..624).map(|_| probe.gen_rand32()).collect::<Vec<_>>(),
                lines
                    .next()
                    .unwrap()
                    .split_whitespace()
                    .map(|s| s.parse::<u32>().unwrap())
                    .collect::<Vec<_>>(),
                "{mode} loaded RNG"
            );
            crate::numerical_comparison::assert_values_close(
                serialize(values),
                expected_values,
                32.0 * f64::EPSILON,
                32.0 * f64::EPSILON,
                format!("{mode} loaded values"),
            );
            assert!(lines.next().is_none());
            let dir = fresh_output_directory().unwrap();
            vmc_para_opt(
                &mut data,
                &mut state,
                &mut rng,
                Some(&dir),
                &SingleProcessReducer,
                OptimizationOptions::default(),
            )
            .unwrap();
            let fixtures = root.parent().unwrap();
            let case = format!("dh2_{mode}");
            let read_checkpoint = |kind: &str| {
                julia_fixture::read_text(
                    if mode.ends_with("fsz") && native_fsz_fixture::directory(fixtures).is_some() {
                        native_fsz_fixture::resolve(
                            native_fsz_fixture::directory(fixtures)
                                .unwrap()
                                .join(format!("sr_direct/{case}_runner/step-3-{kind}.txt")),
                        )
                    } else {
                        julia_fixture::fixture_path(
                            fixtures,
                            format!("sr_direct/{case}_runner/step-3-{kind}.txt"),
                        )
                    },
                )
                .unwrap()
            };
            assert_sampling_checkpoint(&case, 3, &state, &mut rng, &read_checkpoint);
            let history =
                if mode.ends_with("fsz") && native_fsz_fixture::directory(fixtures).is_some() {
                    native_fsz_fixture::directory(fixtures)
                        .unwrap()
                        .join(root.file_name().unwrap())
                        .join(format!("history-{mode}.txt"))
                } else {
                    julia_fixture::fixture_path(
                        fixtures,
                        format!(
                            "{}/history-{mode}.txt",
                            root.file_name().unwrap().to_str().unwrap()
                        ),
                    )
                };
            let text = julia_fixture::read_text(if mode.ends_with("fsz") {
                native_fsz_fixture::resolve(history)
            } else {
                history
            })
            .unwrap();
            let mut lines = text.lines().filter(|line| !line.starts_with('#'));
            assert_eq!(state.opt_data.len(), 3);
            for (index, point) in state.opt_data.iter().enumerate() {
                crate::numerical_comparison::assert_values_close(
                    serialize(vec![point.energy]),
                    scalars(lines.next().unwrap()),
                    1e-11,
                    1e-11,
                    format!("{mode} history energy"),
                );
                assert_complete_dh_history(
                    &data,
                    fixtures,
                    &case,
                    index + 1,
                    point,
                    scalars(lines.next().unwrap()),
                );
            }
            assert!(lines.next().is_none());
            fs::remove_dir_all(dir).unwrap();
        }
    }

    #[test]
    fn dh4_loaded_values_and_rng_match_julia_with_history_in_c_declared_order() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/dh4");
        let scalars = |text: &str| -> Vec<f64> {
            text.split_whitespace()
                .map(|s| f64::from_bits(u64::from_str_radix(s, 16).unwrap()))
                .collect()
        };
        let serialize = |values: Vec<Complex64>| -> Vec<f64> {
            values.iter().flat_map(|v| [v.re, v.im]).collect()
        };
        for mode in [
            "dh4_real",
            "dh4_cmp",
            "dh4_fsz",
            "dh24_real",
            "dh24_cmp",
            "dh24_fsz",
        ] {
            let (mut data, mut state, mut rng) =
                prepared_namelist(3, &root.join(format!("production_{mode}/namelist.def")));
            let input = fs::read_to_string(root.join(format!("loaded-{mode}.txt"))).unwrap();
            let mut lines = input.lines().skip(1);
            assert_eq!(
                data.optimization_flags,
                crate::historical_optimization_flags::c_orbital_representation(
                    &data,
                    lines
                        .next()
                        .unwrap()
                        .split_whitespace()
                        .map(|s| s.parse::<i64>().unwrap())
                        .collect()
                )
            );
            let values = data
                .projection_parameters()
                .into_iter()
                .chain(
                    data.orbital_terms
                        .iter()
                        .map(|t| data.slater_params[t.idx as usize]),
                )
                .collect();
            let expected_values = scalars(lines.next().unwrap());
            let mut probe = rng.clone();
            assert_eq!(
                (0..624).map(|_| probe.gen_rand32()).collect::<Vec<_>>(),
                lines
                    .next()
                    .unwrap()
                    .split_whitespace()
                    .map(|s| s.parse::<u32>().unwrap())
                    .collect::<Vec<_>>(),
                "{mode} loaded RNG"
            );
            crate::numerical_comparison::assert_values_close(
                serialize(values),
                expected_values,
                32.0 * f64::EPSILON,
                32.0 * f64::EPSILON,
                format!("{mode} loaded values"),
            );
            assert!(lines.next().is_none());
            let dir = fresh_output_directory().unwrap();
            vmc_para_opt(
                &mut data,
                &mut state,
                &mut rng,
                Some(&dir),
                &SingleProcessReducer,
                OptimizationOptions::default(),
            )
            .unwrap();
            let fixtures = root.parent().unwrap();
            let case = mode.to_owned();
            let read_checkpoint = |kind: &str| {
                julia_fixture::read_text(
                    if mode.ends_with("fsz") && native_fsz_fixture::directory(fixtures).is_some() {
                        native_fsz_fixture::resolve(
                            native_fsz_fixture::directory(fixtures)
                                .unwrap()
                                .join(format!("sr_direct/{case}_runner/step-3-{kind}.txt")),
                        )
                    } else {
                        julia_fixture::fixture_path(
                            fixtures,
                            format!("sr_direct/{case}_runner/step-3-{kind}.txt"),
                        )
                    },
                )
                .unwrap()
            };
            assert_sampling_checkpoint(&case, 3, &state, &mut rng, &read_checkpoint);
            let history =
                if mode.ends_with("fsz") && native_fsz_fixture::directory(fixtures).is_some() {
                    native_fsz_fixture::directory(fixtures)
                        .unwrap()
                        .join(root.file_name().unwrap())
                        .join(format!("history-{mode}.txt"))
                } else {
                    julia_fixture::fixture_path(
                        fixtures,
                        format!(
                            "{}/history-{mode}.txt",
                            root.file_name().unwrap().to_str().unwrap()
                        ),
                    )
                };
            let text = julia_fixture::read_text(if mode.ends_with("fsz") {
                native_fsz_fixture::resolve(history)
            } else {
                history
            })
            .unwrap();
            let mut lines = text.lines().filter(|line| !line.starts_with('#'));
            assert_eq!(state.opt_data.len(), 3);
            for (index, point) in state.opt_data.iter().enumerate() {
                crate::numerical_comparison::assert_values_close(
                    serialize(vec![point.energy]),
                    scalars(lines.next().unwrap()),
                    1e-11,
                    1e-11,
                    format!("{mode} history energy"),
                );
                assert_complete_dh_history(
                    &data,
                    fixtures,
                    &case,
                    index + 1,
                    point,
                    scalars(lines.next().unwrap()),
                );
            }
            assert!(lines.next().is_none());
            fs::remove_dir_all(dir).unwrap();
        }
    }

    #[test]
    fn dh4_and_dh24_real_complex_and_fsz_cg_prefixes_match_julia_updates_samples_energy_and_rng() {
        for case in [
            "dh4_real",
            "dh4_cmp",
            "dh4_fsz",
            "dh24_real",
            "dh24_cmp",
            "dh24_fsz",
        ] {
            check_sr_prefixes(case, true, 0);
        }
    }
    #[test]
    fn dh4_real_complex_and_fsz_direct_prefixes_match_julia_updates_samples_energy_and_rng() {
        for case in ["dh4_real", "dh4_cmp", "dh4_fsz"] {
            for store in 0..=1 {
                check_sr_prefixes(case, false, store);
            }
        }
    }
    #[test]
    fn dh24_real_complex_and_fsz_direct_prefixes_match_julia_updates_samples_energy_and_rng() {
        for case in ["dh24_real", "dh24_cmp", "dh24_fsz"] {
            for store in 0..=1 {
                check_sr_prefixes(case, false, store);
            }
        }
    }

    #[test]
    fn dh2_real_complex_and_fsz_cg_prefixes_match_julia_updates_samples_energy_and_rng() {
        for case in ["dh2_real", "dh2_cmp", "dh2_fsz"] {
            check_sr_prefixes(case, true, 0);
        }
    }

    #[test]
    fn dh2_real_complex_and_fsz_direct_prefixes_match_julia_updates_samples_energy_and_rng() {
        for case in ["dh2_real", "dh2_cmp", "dh2_fsz"] {
            for store in 0..=1 {
                check_sr_prefixes(case, false, store);
            }
        }
    }

    #[test]
    fn pairhop_real_and_fsz_cg_prefixes_match_julia_updates_samples_energy_and_rng() {
        for case in ["pairhop_real", "pairhop_fsz"] {
            check_sr_prefixes(case, true, 0);
        }
    }

    #[test]
    fn pairhop_real_and_fsz_direct_prefixes_match_julia_updates_samples_energy_and_rng() {
        for case in ["pairhop_real", "pairhop_fsz"] {
            for store in [0, 1] {
                check_sr_prefixes(case, false, store);
            }
        }
    }

    #[test]
    fn pairhop_real_and_fsz_initial_flags_parameters_and_rng_match_julia() {
        for case in ["pairhop_real", "pairhop_fsz"] {
            let input = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../extern/Julia-mVMC/test/integration/reference/hubbard_chain_{case}/inputs/namelist.def"));
            let (data, _, mut rng) = prepared_namelist(1, &input);
            assert_eq!(data.pair_hop_terms.len(), 2);
            assert_eq!(get_all_complex_flag(&data).unwrap(), case == "pairhop_fsz");
            let root = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(format!("../../tests/fixtures/sr_cg/{case}_runner"));
            check_initial_boundary(&data, &mut rng, &root);
        }
    }

    #[test]
    fn normal_interall_equivalents_preserve_sr_parameters_samples_energy_and_rng() {
        for complex in [false, true] {
            for (cg, store) in [(0, 0), (0, 1), (1, 0)] {
                let mut runs = Vec::new();
                for interall in [false, true] {
                    let (mut data, _, mut rng) = prepared_case(3, "hubbard_chain_real");
                    data.modpara.nsrcg = cg;
                    data.modpara.nstore_o = store;
                    data.exchange_terms.clear();
                    if complex {
                        data.complex_flags = vec![1];
                        for (i, value) in data.slater_params.iter_mut().enumerate() {
                            value.im = (i + 1) as f64 / 32.0;
                        }
                        // PairHop and InterAll evaluate the same complex
                        // operators in the same position after Transfer.
                        data.pair_hop_terms = [
                            (0, 1, 0.3),
                            (1, 0, 0.3),
                            (2, 3, -0.125),
                            (3, 2, -0.125),
                            (0, 1, 0.0625),
                        ]
                        .into_iter()
                        .map(|(site1, site2, value)| mvmc_expert_parsers::PairHopTerm {
                            site1,
                            site2,
                            value,
                        })
                        .collect();
                        if interall {
                            data.inter_all_terms = data
                                .pair_hop_terms
                                .iter()
                                .map(|term| mvmc_expert_parsers::InterAllTerm {
                                    site0: term.site1,
                                    spin0: 0,
                                    site1: term.site2,
                                    spin1: 0,
                                    site2: term.site1,
                                    spin2: 1,
                                    site3: term.site2,
                                    spin3: 1,
                                    value: Complex64::new(term.value, 0.0),
                                    is_complex: false,
                                })
                                .collect();
                            data.pair_hop_terms.clear();
                        }
                    } else {
                        // Density reductions give an exact real identity
                        // independently of historical Julia quotient order.
                        data.transfer_terms.clear();
                        data.coulomb_inter_terms.clear();
                        data.hund_terms.clear();
                        data.coulomb_intra_terms = [(0, 0.375), (1, -0.25), (2, 0.125)]
                            .into_iter()
                            .map(|(site, value)| mvmc_expert_parsers::CoulombIntraTerm {
                                site,
                                value,
                            })
                            .collect();
                        if interall {
                            data.inter_all_terms = data
                                .coulomb_intra_terms
                                .iter()
                                .map(|term| mvmc_expert_parsers::InterAllTerm {
                                    site0: term.site,
                                    spin0: 0,
                                    site1: term.site,
                                    spin1: 0,
                                    site2: term.site,
                                    spin2: 1,
                                    site3: term.site,
                                    spin3: 1,
                                    value: Complex64::new(term.value, 0.0),
                                    is_complex: false,
                                })
                                .collect();
                            data.coulomb_intra_terms.clear();
                        }
                    }
                    sync_modified_parameter(&mut data, true);
                    let mut state = state_from_data(&data).unwrap();
                    let mut records = Vec::new();
                    let directory = fresh_output_directory().unwrap();
                    let mut callback =
                        |step, data: &mut ExpertModeData, energy: Complex64, info| {
                            let parameters: Vec<_> = data
                                .projection_parameters()
                                .iter()
                                .chain(&data.slater_params)
                                .flat_map(|v| [v.re, v.im])
                                .collect();
                            records.push((step, energy.re, energy.im, info, parameters));
                            Ok(())
                        };
                    vmc_para_opt(
                        &mut data,
                        &mut state,
                        &mut rng,
                        Some(&directory),
                        &SingleProcessReducer,
                        OptimizationOptions {
                            callback: Some(&mut callback),
                            skip_sr: false,
                        },
                    )
                    .unwrap();
                    assert_eq!(records.len(), 3);
                    assert!(records.iter().any(|record| record.1.abs() > 1e-6));
                    let rng_words: Vec<_> = (0..624).map(|_| rng.gen_rand32()).collect();
                    runs.push((records, state.electron_config, rng_words));
                    fs::remove_dir_all(directory).unwrap();
                }
                assert_eq!(
                    runs[0].1, runs[1].1,
                    "equivalent Hamiltonian configurations"
                );
                assert_eq!(runs[0].2, runs[1].2, "equivalent Hamiltonian RNG");
                for (a, b) in runs[0].0.iter().zip(&runs[1].0) {
                    assert_eq!((a.0, a.3), (b.0, b.3), "step/status");
                    crate::numerical_comparison::assert_values_close(
                        [a.1, a.2],
                        [b.1, b.2],
                        1e-12,
                        1e-12,
                        "equivalent energy",
                    );
                    crate::numerical_comparison::assert_values_close(
                        a.4.iter().copied(),
                        b.4.iter().copied(),
                        1e-11,
                        1e-11,
                        "equivalent SR parameters",
                    );
                }
            }
        }
    }

    #[test]
    fn interall_fsz_direct_prefixes_match_julia_parameters_spins_samples_energy_and_rng() {
        for store in [0, 1] {
            check_sr_prefixes("interall", false, store);
        }
    }

    #[test]
    fn interall_fsz_initial_flags_parameters_and_rng_match_julia_before_sampling() {
        let input = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/c_orbital_inputs/namelist_interall_fsz.def");
        let (data, _, mut rng) = prepared_namelist(1, &input);
        assert_eq!(data.inter_all_terms.len(), 26);
        assert!(get_all_complex_flag(&data).unwrap());
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/sr_cg/interall_runner");
        check_initial_boundary(&data, &mut rng, &root);
    }

    fn check_initial_boundary(
        data: &ExpertModeData,
        rng: &mut sfmt19937::Sfmt19937Rng,
        root: &Path,
    ) {
        let flags: Vec<i64> = fs::read_to_string(root.join("initial-flags.txt"))
            .unwrap()
            .split_whitespace()
            .map(|v| v.parse::<i64>().unwrap())
            .collect();
        let flags = crate::historical_optimization_flags::c_orbital_representation(data, flags);
        assert_eq!(data.optimization_flags, flags);
        let expected: Vec<f64> = fs::read_to_string(root.join("initial-parameters.txt"))
            .unwrap()
            .split_whitespace()
            .map(|v| f64::from_bits(u64::from_str_radix(v, 16).unwrap()))
            .collect();
        let mut mapped = data.clone();
        let mut rbm_values = Vec::new();
        mapped.visit_rbm_terms_mut(|_, t| rbm_values.push(t.value()));
        let values = data
            .gutzwiller_terms
            .iter()
            .map(|t| t.value)
            .chain(data.jastrow_terms.iter().map(|t| t.value))
            .chain(data.doublon_holon_2site_params.iter().copied())
            .chain(data.doublon_holon_4site_params.iter().copied())
            .chain(rbm_values)
            .chain(
                data.orbital_terms
                    .iter()
                    .map(|t| data.slater_params[t.idx as usize]),
            )
            .chain(data.opt_trans.iter().copied());
        let actual: Vec<_> = values.flat_map(|v| [v.re, v.im]).collect();

        let words: Vec<u32> = fs::read_to_string(root.join("initial-rng.txt"))
            .unwrap()
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        assert_eq!(words.len(), 624);
        for expected in words {
            assert_eq!(rng.gen_rand32(), expected);
        }
        crate::numerical_comparison::assert_values_close(
            actual,
            expected,
            32.0 * f64::EPSILON,
            32.0 * f64::EPSILON,
            "initial parameters",
        );
    }

    #[test]
    fn general_direct_sr_prefixes_match_julia_parameters_samples_energy_and_rng() {
        for store in [0, 1] {
            check_sr_prefixes("general", false, store);
        }
    }

    #[test]
    fn hubbard_cg_prefixes_match_julia_parameters_samples_energy_and_full_rng_blocks() {
        check_sr_prefixes("hubbard", true, 0);
    }

    #[test]
    fn real_direct_sr_prefixes_match_julia_parameters_samples_energy_and_rng() {
        for store in [0, 1] {
            check_sr_prefixes("real", false, store);
        }
    }

    #[test]
    fn cmp_direct_sr_prefixes_match_julia_parameters_samples_energy_and_rng() {
        for store in [0, 1] {
            check_sr_prefixes("cmp", false, store);
        }
    }

    #[test]
    fn fsz_direct_sr_prefixes_match_julia_parameters_samples_energy_and_rng() {
        for store in [0, 1] {
            check_sr_prefixes("fsz", false, store);
        }
    }

    #[test]
    fn hubbard_direct_sr_prefixes_match_julia_parameters_samples_energy_and_rng() {
        for store in [0, 1] {
            check_sr_prefixes("hubbard", false, store);
        }
    }

    #[test]
    fn rbm_complex_direct_prefixes_match_source_parameters_samples_energy_and_rng() {
        check_sr_prefixes("rbm_cmp", false, 0);
    }

    #[test]
    fn opttrans_initial_flags_parameters_and_rng_match_julia_before_sampling() {
        for case in ["opt_real", "opt_cmp", "opt_fsz", "opt_dh24_rbm_cmp"] {
            let input = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
                "../../tests/fixtures/opttrans/run_{case}/namelist.def"
            ));
            let (data, _, mut rng) = prepared_namelist(1, &input);
            assert_eq!(data.opt_trans.len(), 3);
            let root = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(format!("../../tests/fixtures/sr_cg/{case}_runner"));
            check_initial_boundary(&data, &mut rng, &root);
        }
    }

    #[test]
    fn opttrans_real_direct_prefixes_match_c_kernel_reference_parameters_samples_and_rng() {
        check_sr_prefixes("opt_real", false, 0);
    }

    #[test]
    fn opttrans_complex_fsz_and_all_factor_direct_prefixes_match_source() {
        for case in ["opt_cmp", "opt_fsz", "opt_dh24_rbm_cmp"] {
            check_sr_prefixes(case, false, 0);
        }
    }

    #[test]
    fn opttrans_stored_direct_prefixes_match_explicit_kernel_references() {
        for case in ["opt_real", "opt_cmp", "opt_fsz", "opt_dh24_rbm_cmp"] {
            check_sr_prefixes(case, false, 1);
        }
    }

    #[test]
    fn opttrans_cg_prefixes_match_source() {
        for case in ["opt_real", "opt_cmp", "opt_fsz", "opt_dh24_rbm_cmp"] {
            check_sr_prefixes(case, true, 0);
        }
    }

    #[test]
    fn rbm_real_and_general_complex_direct_prefixes_match_explicit_kernel_references() {
        for case in ["rbm_real", "rbm_general_cmp"] {
            check_sr_prefixes(case, false, 0);
        }
    }
    #[test]
    fn rbm_real_and_complex_cg_prefixes_match_explicit_kernel_references() {
        for case in ["rbm_real", "rbm_cmp", "rbm_general_cmp"] {
            check_sr_prefixes(case, true, 0);
        }
    }

    #[test]
    fn rbm_dh24_direct_and_cg_prefixes_match_source() {
        check_sr_prefixes("rbm_dh24_cmp", false, 0);
        check_sr_prefixes("rbm_dh24_cmp", true, 0);
    }

    #[test]
    fn rbm_fsz_direct_prefixes_match_source_including_failure_state() {
        check_sr_prefixes("rbm_fsz", false, 0);
    }

    #[test]
    fn rbm_fsz_cg_prefixes_match_source_including_empty_weight_steps() {
        check_sr_prefixes("rbm_fsz", true, 0);
    }

    #[test]
    fn rbm_stored_direct_prefixes_match_explicit_kernel_references() {
        for case in [
            "rbm_real",
            "rbm_cmp",
            "rbm_general_cmp",
            "rbm_dh24_cmp",
            "rbm_fsz",
        ] {
            check_sr_prefixes(case, false, 1);
        }
    }

    #[test]
    fn canonical_general_rbm_complex_reference_uses_native_c_counter_order() {
        check_sr_prefixes("rbm_reference_cmp", false, 1);
        check_sr_prefixes("rbm_reference_cmp", true, 0);
    }

    #[derive(Default)]
    struct RunnerCgDiagnostics(
        std::cell::RefCell<String>,
        std::cell::Cell<bool>, // first factor claimed
        std::cell::Cell<bool>, // factor scope active
        std::cell::Cell<bool>, // first O claimed
        std::cell::Cell<bool>, // first accumulator prefix claimed
        std::cell::Cell<bool>, // quota failure, never labelled complete
        std::cell::Cell<bool>, // inverse-published boundary actually observed
        std::cell::Cell<bool>, // pre-Step5 boundary actually observed
        std::cell::Cell<bool>, // actual CG prepared operands recorded
        std::cell::Cell<bool>, // actual CG operator product recorded
        std::cell::Cell<bool>, // actual CG completion recorded
    );

    impl RunnerCgDiagnostics {
        fn metadata(&self, value: &str) {
            if self.5.get() {
                return;
            }
            if self.0.borrow().len().saturating_add(value.len()) > 4 * 1024 * 1024 - 1024 {
                self.5.set(true);
                self.0
                    .borrow_mut()
                    .push_str("CAPTURE_INCOMPLETE metadata quota\n");
                return;
            }
            self.0.borrow_mut().push_str(value);
        }

        fn cg_boundaries_complete(&self) -> bool {
            !self.5.get() && self.8.get() && self.9.get() && self.10.get()
        }

        fn direct_record_counts_complete(normalized: usize, solved: usize) -> bool {
            normalized == 1 && solved == 1
        }

        fn values(&self, label: &str, values: &[f64]) {
            use std::fmt::Write;
            if self.5.get() {
                return;
            }
            if values.len() > 32_768
                || self.0.borrow().len() + 64 * values.len() + label.len() + 64 > 4 * 1024 * 1024
            {
                self.5.set(true);
                self.0.borrow_mut().push_str("CAPTURE_INCOMPLETE quota\n");
                return;
            }
            let mut text = self.0.borrow_mut();
            writeln!(text, "{label} {}", values.len()).unwrap();
            for value in values {
                write!(text, "{value:.17e} ").unwrap();
            }
            text.push('\n');
        }

        fn components(&self, label: &str, values: &[Complex64]) {
            if values.len() > 16_384 {
                self.5.set(true);
                self.0
                    .borrow_mut()
                    .push_str("CAPTURE_INCOMPLETE component quota\n");
                return;
            }
            let components: Vec<_> = values.iter().flat_map(|v| [v.re, v.im]).collect();
            self.values(label, &components);
        }
    }

    impl OptimizationMeasurementObserver for RunnerCgDiagnostics {
        fn rng_boundary(&self, phase: &'static str, rng: &Sfmt19937Rng) {
            if self.5.get() {
                return;
            }
            let (words, cursor) = rng.state_snapshot();
            let count = rng.words_consumed();
            let mut peek = rng.clone();
            let future: Vec<u32> = (0..624).map(|_| peek.gen_rand32()).collect();
            assert_eq!(rng.state_snapshot(), (words, cursor));
            assert_eq!(rng.words_consumed(), count);
            self.metadata(&format!(
                "raw-boundary={phase} cursor={cursor} count={count} words={words:?} future={future:?}\n"
            ));
        }
        fn begin_real_factor(
            &self,
            data: &ExpertModeData,
            state: &VmcOptimizationState,
            sample: usize,
        ) -> bool {
            if sample != 0 || self.1.replace(true) {
                return false;
            }
            let n = state.electron_config.ele_idx_slice(sample).len();
            if n == 0
                || n > 16
                || state.slater_matrix.pf_m_real.len() > 128
                || state.electron_config.ele_cfg_slice(sample).len() > 256
                || state.electron_config.ele_num_slice(sample).len() > 256
                || state.electron_config.ele_proj_cnt_slice(sample).len() > 256
            {
                self.0
                    .borrow_mut()
                    .push_str("CAPTURE_INCOMPLETE unsupported factor shape\n");
                self.5.set(true);
                return false;
            }
            self.metadata(&format!("first-factor sample={sample} nsize={n} seed={} idx={:?} cfg={:?} num={:?} proj={:?}\n",
                data.modpara.rnd_seed, state.electron_config.ele_idx_slice(sample),
                state.electron_config.ele_cfg_slice(sample), state.electron_config.ele_num_slice(sample),
                state.electron_config.ele_proj_cnt_slice(sample)));
            self.2.set(true);
            self.components("first-slater-parameters", &data.slater_params);
            self.values(
                "first-slater-real-rowmajor-allqp",
                state.slater_matrix.slater_elm_real.as_slice(),
            );
            if let Some(weights) = &data.qp_weights {
                self.components("first-qp-full-weights", &weights.qp_full_weight);
            }
            true
        }
        fn real_factor_active(&self) -> bool {
            self.2.get()
        }
        fn end_real_factor(&self) {
            self.2.set(false);
        }
        fn real_factor(&self, view: RealFactorView<'_>) {
            if self.5.get() {
                return;
            }
            if view.pivots.len() > 16 || view.stage.len() > 64 {
                self.5.set(true);
                self.0
                    .borrow_mut()
                    .push_str("CAPTURE_INCOMPLETE factor metadata shape\n");
                return;
            }
            self.metadata(&format!(
                "factor stage={} qp={} n={} pivots={:?} error={:?} pf={:?}\n",
                view.stage, view.qp, view.dimension, view.pivots, view.factor_error, view.pf
            ));
            self.values(view.stage, view.matrix);
            if view.stage == "inverse-published" {
                self.6.set(true);
            }
        }
        fn real_step5(&self, view: pfapack::utu2::InverseStep5View<'_, f64>) {
            self.values("pre-step5-rhs", view.rhs);
            self.values("pre-step5-tridiagonal", view.tridiagonal);
            self.7.set(true);
        }
        fn measured(&self, view: OptimizationMeasurementView<'_>) {
            if view.sample != 0 || self.3.replace(true) {
                return;
            }
            self.components("first-O-complex", &view.state.sr_opt.sr_opt_o);
            self.components("first-HO-before", &view.state.sr_opt.sr_opt_ho);
            self.values("first-HO-real-before", &view.state.sr_opt.sr_opt_ho_real);
            self.components("first-overlap-energy", &[view.overlap, view.local_energy]);
            self.values("first-weight", &[view.weight]);
        }
        fn accumulated(&self, view: OptimizationMeasurementView<'_>) {
            if view.sample != 0 || self.4.replace(true) {
                return;
            }
            self.values("first-O-real", &view.state.sr_opt.sr_opt_o_real);
            self.values("first-HO-real-after", &view.state.sr_opt.sr_opt_ho_real);
            self.components("first-HO-after", &view.state.sr_opt.sr_opt_ho);
            self.values("first-OO-real-after", &view.state.sr_opt.sr_opt_oo_real);
            self.components("first-OO-after", &view.state.sr_opt.sr_opt_oo);
        }
    }

    #[test]
    fn small_model_factor_diagnostic_is_first_sample_qp0_and_unwind_scoped() {
        let data = ExpertModeData::default();
        let state = VmcOptimizationState::zeros(2, 1, 1, 2, 1, 3, false, false);
        let observer = std::rc::Rc::new(RunnerCgDiagnostics::default());
        let installed = install_optimization_measurement_observer(observer.clone()).unwrap();
        assert!(begin_first_real_factor(&data, &state, 1).is_none());
        assert!(first_real_factor_observer(0).is_none());
        let interrupted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _scope = begin_first_real_factor(&data, &state, 0).unwrap();
            assert!(first_real_factor_observer(0).is_some());
            assert!(first_real_factor_observer(1).is_none());
            assert!(install_optimization_measurement_observer(observer.clone()).is_err());
            panic!("controlled first-factor unwind");
        }));
        let payload = interrupted.expect_err("controlled scope must unwind");
        assert_eq!(
            payload.downcast_ref::<&str>().copied(),
            Some("controlled first-factor unwind")
        );
        assert!(first_real_factor_observer(0).is_none());
        assert!(begin_first_real_factor(&data, &state, 0).is_none());
        drop(installed);
        assert!(install_optimization_measurement_observer(observer).is_ok());
    }

    #[test]
    fn small_model_diagnostic_records_only_first_walker_prefix() {
        let data = ExpertModeData::default();
        let state = VmcOptimizationState::zeros(2, 1, 1, 2, 1, 3, false, false);
        let observer = RunnerCgDiagnostics::default();
        let view = || OptimizationMeasurementView {
            data: &data,
            state: &state,
            sample: 0,
            overlap: Complex64::new(1.0, 0.0),
            local_energy: Complex64::new(-0.25, 0.0),
            weight: 1.0,
        };
        observer.measured(view());
        observer.accumulated(view());
        let first = observer.0.borrow().clone();
        observer.measured(view());
        observer.accumulated(view());
        assert_eq!(*observer.0.borrow(), first);
        assert!(first.contains("first-HO-real-before"));
        assert!(first.contains("first-HO-real-after"));
        assert!(!observer.5.get());
    }

    #[test]
    fn small_model_diagnostic_quota_is_explicit_incomplete_not_success() {
        let observer = RunnerCgDiagnostics::default();
        observer.values("too-wide", &vec![0.0; 32_769]);
        assert!(observer.5.get());
        assert_eq!(&*observer.0.borrow(), "CAPTURE_INCOMPLETE quota\n");
        observer.values("later", &[1.0]);
        assert!(!observer.0.borrow().contains("later"));
    }

    #[test]
    fn small_model_missing_cg_boundary_never_completes() {
        let observer = RunnerCgDiagnostics::default();
        assert!(!observer.cg_boundaries_complete());
        for missing in 0..3 {
            observer.8.set(missing != 0);
            observer.9.set(missing != 1);
            observer.10.set(missing != 2);
            assert!(!observer.cg_boundaries_complete());
        }
        observer.8.set(true);
        observer.9.set(true);
        observer.10.set(false);
        assert!(!observer.cg_boundaries_complete());
        observer.10.set(true);
        assert!(observer.cg_boundaries_complete());
        observer.9.set(false);
        assert!(!observer.cg_boundaries_complete());
        observer.9.set(true);
        observer.5.set(true);
        assert!(!observer.cg_boundaries_complete());
    }

    #[test]
    fn small_model_direct_missing_or_duplicate_records_are_incomplete() {
        for (normalized, solved) in [(0, 0), (0, 1), (1, 0), (2, 1), (1, 2)] {
            assert!(!RunnerCgDiagnostics::direct_record_counts_complete(
                normalized, solved
            ));
        }
        assert!(RunnerCgDiagnostics::direct_record_counts_complete(1, 1));
    }

    #[test]
    fn small_model_raw_checkpoint_keeps_live_words_cursor_and_u128_count() {
        let mut rng = Sfmt19937Rng::new(1);
        for _ in 0..12 {
            rng.gen_rand32();
        }
        let before = rng.state_snapshot();
        let count: u128 = rng.words_consumed();
        let observer = RunnerCgDiagnostics::default();
        observer.rng_boundary("initialized", &rng);
        assert_eq!(rng.state_snapshot(), before);
        assert_eq!(rng.words_consumed(), count);
        assert!(observer
            .0
            .borrow()
            .contains("raw-boundary=initialized cursor=12 count=12"));
        assert!(observer.0.borrow().contains(" words=["));
        assert!(observer.0.borrow().contains(" future=["));
        assert!(!observer.5.get());
    }

    impl crate::sr_cg::CgObserver for RunnerCgDiagnostics {
        fn prepared(
            &self,
            mapping: &[usize],
            op: &crate::sr_cg::SampledSrOperator,
            gradient: &[f64],
        ) {
            if self.5.get() {
                return;
            }
            if mapping.len() > 128 {
                self.5.set(true);
                self.0
                    .borrow_mut()
                    .push_str("CAPTURE_INCOMPLETE active mapping quota\n");
                return;
            }
            self.metadata(&format!("mapping {mapping:?}\n"));
            for (name, values) in [
                ("mean", &op.mean),
                ("diagonal", &op.diagonal),
                ("real_samples", &op.real_samples),
                ("imag_samples", &op.imag_samples),
            ] {
                self.values(name, values);
            }
            self.values("gradient", gradient);
            self.8.set(!self.5.get());
        }
        fn product(&self, phase: crate::sr_cg::CgProductPhase, search: &[f64], product: &[f64]) {
            self.values(&format!("{phase:?}-search"), search);
            self.values(&format!("{phase:?}-product"), product);
            self.9.set(!self.5.get());
        }
        fn iteration(&self, state: crate::sr_cg::CgIterationView<'_>) {
            if self.5.get() {
                return;
            }
            self.metadata(&format!(
                "iteration={} delta={:.17e} alpha={:?}\n",
                state.iteration, state.delta, state.alpha
            ));
            self.values("solution", state.solution);
            self.values("residual", state.residual);
            self.values("direction", state.direction);
        }
        fn finished(&self, result: &crate::sr_cg::CgSolution) {
            if self.5.get() {
                return;
            }
            self.metadata(&format!("finished iterations={}\n", result.iterations));
            self.values("final-solution", &result.solution);
            self.values("final-residual", &result.residual);
            self.10.set(!self.5.get());
        }
    }

    fn assert_sampling_checkpoint(
        case: &str,
        steps: i64,
        state: &VmcOptimizationState,
        rng: &mut Sfmt19937Rng,
        read: &impl Fn(&str) -> String,
    ) {
        // The complete saved state and SFMT block are independent gates:
        // numerical comparison must never prevent detecting trajectory drift.
        let conf = read("configs");
        let mut lines = conf.lines();
        for (name, actual) in [
            ("indices", &state.electron_config.ele_idx),
            ("configuration", &state.electron_config.ele_cfg),
            ("occupancy", &state.electron_config.ele_num),
            ("projection", &state.electron_config.ele_proj_cnt),
        ] {
            let expected: Vec<i64> = lines
                .next()
                .unwrap()
                .split_whitespace()
                .map(|v| v.parse().unwrap())
                .collect();
            assert_eq!(actual, &expected, "step {steps} {name}");
        }
        if matches!(
            case,
            "interall" | "pairhop_fsz" | "dh2_fsz" | "dh4_fsz" | "dh24_fsz" | "rbm_fsz" | "opt_fsz"
        ) {
            for (name, actual) in [
                ("spins", &state.electron_config.ele_spn),
                ("burn", &state.electron_config.burn_ele_idx),
                ("counters", &state.electron_config.counter.to_vec()),
            ] {
                let expected: Vec<i64> = lines
                    .next()
                    .unwrap()
                    .split_whitespace()
                    .map(|v| v.parse().unwrap())
                    .collect();
                assert_eq!(actual, &expected, "step {steps} {name}");
            }
        }
        if (case.starts_with("rbm_") || case.starts_with("opt_"))
            && case != "rbm_fsz"
            && case != "opt_fsz"
        {
            for (name, actual) in [
                ("burn", &state.electron_config.burn_ele_idx),
                ("counters", &state.electron_config.counter.to_vec()),
            ] {
                let expected: Vec<i64> = lines
                    .next()
                    .unwrap()
                    .split_whitespace()
                    .map(|v| v.parse().unwrap())
                    .collect();
                assert_eq!(actual, &expected, "{case} step {steps} {name}");
            }
        }
        assert!(lines.next().is_none());
        let expected: Vec<u32> = read("rng")
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        let actual: Vec<u32> = (0..624).map(|_| rng.gen_rand32()).collect();
        assert_eq!(actual, expected, "step {steps} RNG block");
    }

    fn check_sr_prefixes(case: &str, cg: bool, store: i64) {
        let reference_case = if case == "general" { "fsz" } else { case };
        // C's counter grouping and subthreshold Slater retention differ from
        // Julia. These cases use a separately labelled mixed reference whose
        // counter translation is checked against all 4,994 native C cases.
        let c_kernel_order = case == "rbm_reference_cmp" || (case == "opt_real" && !cg);
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures");
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(if c_kernel_order && cg {
                "../../tests/fixtures/c_kernel_order/sr_cg"
            } else if c_kernel_order {
                "../../tests/fixtures/c_kernel_order/sr_direct"
            } else if cg {
                "../../tests/fixtures/sr_cg"
            } else {
                "../../tests/fixtures/sr_direct"
            })
            .join(format!(
                "{reference_case}{}",
                if store == 0 {
                    "_runner"
                } else {
                    "_store_runner"
                }
            ));
        // The production FSZ Hamiltonian now follows the actual C callees.
        // Independently generated mixed references retain Julia's sampler/SR
        // while calling those C bodies; archived Julia-only expectations stay
        // unchanged in their original directories.
        let native_fsz = matches!(
            reference_case,
            "fsz" | "interall" | "pairhop_fsz" | "opt_fsz" | "dh2_fsz" | "dh4_fsz" | "dh24_fsz"
        ) && native_fsz_fixture::directory(&fixtures).is_some();
        let root = if native_fsz {
            native_fsz_fixture::directory(&fixtures)
                .unwrap()
                .join(if cg { "sr_cg" } else { "sr_direct" })
                .join(format!(
                    "{reference_case}{}",
                    if store == 0 {
                        "_runner"
                    } else {
                        "_store_runner"
                    }
                ))
        } else {
            root
        };
        let reviewed_case_name = if case == "rbm_reference_cmp" {
            "canonical_general_rbm"
        } else {
            reference_case
        };
        let reviewed_case_dir = fixtures.join("reviewed_cg_62b").join(reviewed_case_name);
        let reviewed_cg = cg && reviewed_case_dir.join("acquisition-complete.txt").is_file();
        if cg {
            assert!(reviewed_cg,
                "{case}: reviewed 62b prefixes1/2/3/20 acquisition is incomplete; historical CG references are not a current numerical comparison");
            for prefix in [1, 2, 3, 20] {
                for kind in ["parameters", "configs", "energy", "rng", "SRinfo"] {
                    assert!(
                        reviewed_case_dir
                            .join(format!("step-{prefix}-{kind}.txt"))
                            .is_file(),
                        "{case}: incomplete reviewed prefix{prefix} {kind}"
                    );
                }
                if case.starts_with("dh") || case.starts_with("rbm_") || case.starts_with("opt_") {
                    for relative in [
                        format!("step-{prefix}-zvo_out.dat"),
                        format!("step-{prefix}/zvo_var.dat"),
                        format!("step-{prefix}/provenance.txt"),
                    ] {
                        assert!(
                            reviewed_case_dir.join(&relative).is_file(),
                            "{case}: incomplete reviewed output {relative}"
                        );
                    }
                }
            }
        }
        let reviewed_direct_dir = reviewed_case_dir.join(format!("direct-store{store}"));
        let reviewed_direct = !cg
            && reviewed_direct_dir
                .join("acquisition-complete.txt")
                .is_file();
        let read_fixture = |name: &str| {
            let reviewed_prefix = ["step-1-", "step-2-", "step-3-", "step-20-"]
                .iter()
                .any(|prefix| name.starts_with(prefix));
            julia_fixture::read_text(if reviewed_cg && reviewed_prefix {
                // Platform-specific independently generated references override
                // the archived Linux reviewed lineage through fixture_path, which
                // falls back to the base reviewed_cg_62b directory.
                julia_fixture::fixture_path(
                    &fixtures,
                    Path::new("reviewed_cg_62b")
                        .join(reviewed_case_name)
                        .join(name),
                )
            } else if !cg && name.starts_with("step-20-") {
                assert!(reviewed_direct, "{case}: reviewed direct/store{store} 20-step acquisition incomplete; no historical50 truncation");
                reviewed_direct_dir.join(name)
            } else if native_fsz {
                native_fsz_fixture::resolve(root.join(name))
            } else {
                julia_fixture::fixture_path(
                    &fixtures,
                    root.strip_prefix(&fixtures).unwrap().join(name),
                )
            })
            .unwrap_or_else(|error| panic!("{case}: reference {name}: {error}"))
        };
        let prefixes = if cg {
            // User-selected long-run baseline; fresh 20-step references,
            // never a truncation of historical 50-step final parameters.
            vec![1, 2, 3, 20]
        } else if case == "hubbard" {
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20]
        } else if !cg && store == 0 && case == "opt_real" {
            // Keep prefixes around the archived Julia failure at step 29.
            // Preserve purposeful historical failure-boundary regressions,
            // independently of the user-selected 20-step long baseline.
            vec![1, 2, 3, 20, 27, 28, 29]
        } else {
            vec![1, 2, 3, 20]
        };
        for steps in prefixes {
            let name = if case == "hubbard" {
                "hubbard_chain_real".into()
            } else {
                format!("heisenberg_chain_{case}")
            };
            let (mut data, mut state, mut rng) = if case == "rbm_reference_cmp" {
                let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../extern/Julia-mVMC/test/integration/reference/general_rbm_cmp/inputs/namelist.def");
                let mut data = parse_expert_mode_files(&path).unwrap();
                data.modpara.nsr_opt_itr_step = steps;
                data.modpara.nsr_opt_itr_smp = steps;
                let mut rng = Sfmt19937Rng::new(12395);
                init_parameter(&mut data, &mut rng).unwrap();
                assert!(
                    read_initial_def(&mut data, path.parent().unwrap().join("initial.def"))
                        .unwrap()
                );
                read_input_parameters(&mut data, &path).unwrap();
                sync_modified_parameter(&mut data, true);
                init_qp_weight(&mut data);
                let state = state_from_data(&data).unwrap();
                (data, state, rng)
            } else if case.starts_with("opt_") {
                prepared_namelist(
                    steps,
                    &Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
                        "../../tests/fixtures/opttrans/run_{case}/namelist.def"
                    )),
                )
            } else if case.starts_with("rbm_") {
                prepared_namelist(
                    steps,
                    &Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join(format!("../../tests/fixtures/rbm/run_{case}/namelist.def")),
                )
            } else if let Some(mode) = case.strip_prefix("dh2_") {
                prepared_namelist(
                    steps,
                    &Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
                        "../../tests/fixtures/dh2/production_{mode}/namelist.def"
                    )),
                )
            } else if case.starts_with("dh4_") || case.starts_with("dh24_") {
                prepared_namelist(
                    steps,
                    &Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
                        "../../tests/fixtures/dh4/production_{case}/namelist.def"
                    )),
                )
            } else if case.starts_with("pairhop_") {
                prepared_namelist(
                    steps,
                    &Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../extern/Julia-mVMC/test/integration/reference/hubbard_chain_{case}/inputs/namelist.def")),
                )
            } else if case == "interall" {
                prepared_namelist(
                    steps,
                    &Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join("../../tests/fixtures/c_orbital_inputs/namelist_interall_fsz.def"),
                )
            } else if case == "general" {
                prepared_namelist(
                    steps,
                    &Path::new(env!("CARGO_MANIFEST_DIR")).join(
                        "../../tests/fixtures/c_orbital_inputs/namelist_heisenberg_general.def",
                    ),
                )
            } else {
                prepared_case(steps, &name)
            };
            data.modpara.nsrcg = i64::from(cg);
            data.modpara.nstore_o = store;
            let initial_data = data.clone();
            let initial_rng = rng.clone();
            let dir = fresh_output_directory().unwrap();
            let mut rbm_before_sr = vec![data.rbm_params.clone()];
            let mut record_rbm = |_, data: &mut ExpertModeData, _, _| {
                rbm_before_sr.push(data.rbm_params.clone());
                Ok(())
            };
            // Optional diagnostics retain ACTUAL operands/events before a
            // failing forward assertion. Never used as regenerated expectations.
            let diagnostic = (steps == 1 && std::env::var_os("MVMC_CG_DIAGNOSTICS").is_some())
                .then(|| std::rc::Rc::new(RunnerCgDiagnostics::default()));
            let diagnostic_guard = diagnostic
                .as_ref()
                .filter(|_| cg)
                .map(|observer| crate::sr_cg::install_cg_observer(observer.clone()).unwrap());
            let factor_guard = diagnostic.as_ref().map(|observer| {
                install_optimization_measurement_observer(observer.clone()).unwrap()
            });
            let mut direct_guard = diagnostic
                .as_ref()
                .filter(|_| !cg)
                .map(|_| crate::sr::observer::capture_with_normalized().unwrap());
            let mut run_once = || {
                vmc_para_opt(
                    &mut data,
                    &mut state,
                    &mut rng,
                    Some(&dir),
                    &SingleProcessReducer,
                    OptimizationOptions {
                        callback: Some(&mut record_rbm),
                        ..OptimizationOptions::default()
                    },
                )
            };
            let caught = if diagnostic.is_some() {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(&mut run_once))
            } else {
                Ok(run_once())
            };
            let (result, panic_payload) = match caught {
                Ok(result) => (result, None),
                Err(payload) => (
                    Err("diagnostic model panicked (no return status)".to_owned()),
                    Some(payload),
                ),
            };
            drop(diagnostic_guard);
            drop(factor_guard);
            if let Some(diagnostic) = diagnostic {
                let path = dir.with_extension("cg-diagnostics.txt");
                let mut text = format!("case={case} steps={steps} store={store} seed={} flags={:?}\ninitial_rng={initial_rng:?}\nfinal_rng={rng:?}\nweight={:?} result={result:?}\n",
                    initial_data.modpara.rnd_seed, initial_data.optimization_flags, state.energy.wc);
                text.push_str(&diagnostic.0.borrow());
                let mut direct_complete = false;
                if let Some(mut guard) = direct_guard.take() {
                    let normalized = guard.take_normalized();
                    let solved = guard.finish();
                    // These actual direct systems are a separate residual plan;
                    // no residual/matrix product is calculated in the model.
                    let within_shape = RunnerCgDiagnostics::direct_record_counts_complete(
                        normalized.len(),
                        solved.len(),
                    ) && normalized.iter().all(|r| {
                        r.oo.len() <= 32_768
                            && r.ho.len() <= 32_768
                            && r.oo_real.len() <= 32_768
                            && r.ho_real.len() <= 32_768
                    }) && solved.iter().all(|r| {
                        r.matrix.len() <= 16_384
                            && r.rhs.len() <= 128
                            && r.increment.len() <= 128
                            && r.active_indices.len() <= 128
                            && r.flags.len() <= 256
                    });
                    if within_shape && text.len() < 2 * 1024 * 1024 {
                        use std::fmt::Write;
                        writeln!(
                            text,
                            "direct-normalized={normalized:?}\ndirect-solve={solved:?}"
                        )
                        .unwrap();
                        // Transport completeness is not numerical acceptance.
                        // Interrupted/early-return systems remain partial evidence.
                        direct_complete = solved[0].status.is_some()
                            && solved[0].factor_info.is_some()
                            && solved[0].solve_info.is_some()
                            && solved[0].not_solved.is_none();
                    } else {
                        text.push_str("CAPTURE_INCOMPLETE direct record quota\n");
                        diagnostic.5.set(true);
                    }
                }
                if !diagnostic.5.get()
                    && diagnostic.6.get()
                    && diagnostic.7.get()
                    && diagnostic.3.get()
                    && diagnostic.4.get()
                    && if cg {
                        diagnostic.cg_boundaries_complete()
                    } else {
                        direct_complete
                    }
                    && result.is_ok()
                    && panic_payload.is_none()
                    && text.len() <= 4 * 1024 * 1024
                {
                    text.push_str(
                        "CAPTURE_FINISHED actual-model-return (not numerical acceptance)\n",
                    );
                } else {
                    let mut end = text.len().min(4 * 1024 * 1024);
                    while !text.is_char_boundary(end) {
                        end -= 1;
                    }
                    text.truncate(end);
                    text.push_str(
                        "\nCAPTURE_INCOMPLETE return-error/unwind/quota/missing-boundary\n",
                    );
                }
                fs::write(&path, text).unwrap();
                eprintln!("ACTUAL CG diagnostic: {}", path.display());
            }
            if let Some(payload) = panic_payload {
                std::panic::resume_unwind(payload);
            }
            if steps > 10 {
                if case != "rbm_fsz" && !case.starts_with("opt_") {
                    assert!(result.is_ok(), "{case} {steps}: {result:?}");
                }
                // General math and reduction rounding may change an acceptance
                // branch. Compare long trajectories within this implementation;
                // fixed-input kernels and short prefixes retain oracle checks.
                match &result {
                    Ok(()) => assert_eq!(state.opt_data.len(), steps as usize),
                    Err(error) => {
                        assert!(error.contains("SR failed at step"), "{case}: {error}");
                        assert!(state.opt_data.len() < steps as usize);
                    }
                }
                let mut repeat_data = initial_data;
                let mut repeat_state = state_from_data(&repeat_data).unwrap();
                let mut repeat_rng = initial_rng;
                let repeat_dir = fresh_output_directory().unwrap();
                let repeat_result = vmc_para_opt(
                    &mut repeat_data,
                    &mut repeat_state,
                    &mut repeat_rng,
                    Some(&repeat_dir),
                    &SingleProcessReducer,
                    OptimizationOptions::default(),
                );
                assert_eq!(result, repeat_result, "{case} {steps} status");
                let a = &state.electron_config;
                let b = &repeat_state.electron_config;
                for (actual, expected) in [
                    (&a.ele_idx, &b.ele_idx),
                    (&a.ele_cfg, &b.ele_cfg),
                    (&a.ele_num, &b.ele_num),
                    (&a.ele_proj_cnt, &b.ele_proj_cnt),
                    (&a.ele_spn, &b.ele_spn),
                    (&a.burn_ele_idx, &b.burn_ele_idx),
                ] {
                    assert_eq!(actual, expected, "{case} {steps} discrete state");
                }
                assert_eq!(a.counter, b.counter);
                assert!(a.ele_num.iter().all(|&n| n == 0 || n == 1));
                for _ in 0..624 {
                    assert_eq!(rng.gen_rand32(), repeat_rng.gen_rand32());
                }
                let files = |path: &Path| {
                    let mut names: Vec<_> = fs::read_dir(path)
                        .unwrap()
                        .map(|entry| entry.unwrap().file_name())
                        .collect();
                    names.sort();
                    names
                };
                let names = files(&dir);
                assert_eq!(names, files(&repeat_dir));
                assert!(!names.is_empty(), "{case} {steps} output");
                for name in names {
                    let indexed = name.to_str().unwrap();
                    let exact_columns: &[usize] = if indexed == "zvo_SRinfo.dat" {
                        &[0, 1, 2, 3, 7, 8]
                    } else if indexed.starts_with("zqp_") && indexed != "zqp_opt.dat" {
                        &[0]
                    } else {
                        &[]
                    };
                    crate::numerical_comparison::assert_numeric_text(
                        &fs::read_to_string(dir.join(&name)).unwrap(),
                        &fs::read_to_string(repeat_dir.join(&name)).unwrap(),
                        1e-11,
                        1e-11,
                        exact_columns,
                        format!("{case} {steps} reproducible output {name:?}"),
                    );
                }
                fs::remove_dir_all(repeat_dir).unwrap();
                fs::remove_dir_all(dir).unwrap();
                continue;
            }
            let failed = if case == "rbm_fsz" || case.starts_with("opt_") {
                let status = read_fixture(&format!("step-{steps}-status.txt"));
                let mut status = status.split_whitespace();
                let info: i32 = status.next().unwrap().parse().unwrap();
                let step: i32 = status.next().unwrap().parse().unwrap();
                if info != 0 {
                    let method = if cg { "CG" } else { "direct" };
                    assert_eq!(result.unwrap_err(), format!("vmc_para_opt: {method} SR failed at step {step} (local status {info}); parameters were not updated"));
                } else {
                    result.unwrap();
                }
                info != 0
            } else {
                result.unwrap();
                false
            };
            let read = |kind: &str| read_fixture(&format!("step-{steps}-{kind}.txt"));
            assert_sampling_checkpoint(case, steps, &state, &mut rng, &read);
            let scalars = |text: &str| -> Vec<f64> {
                text.split_whitespace()
                    .map(|v| f64::from_bits(u64::from_str_radix(v, 16).unwrap()))
                    .collect()
            };
            if (case.starts_with("rbm_") || case.starts_with("opt_")) && steps == 1 && !cg {
                let fixture = read_fixture("fixed-input.txt");
                let lines: Vec<&str> = fixture.lines().filter(|l| !l.starts_with('#')).collect();
                let components = |values: &[Complex64]| -> Vec<f64> {
                    values.iter().flat_map(|z| [z.re, z.im]).collect()
                };
                let (oo, ho) = if get_all_complex_flag(&data).unwrap() {
                    (
                        components(&state.sr_opt.sr_opt_oo),
                        components(&state.sr_opt.sr_opt_ho),
                    )
                } else {
                    (
                        state.sr_opt.sr_opt_oo_real.to_vec(),
                        state.sr_opt.sr_opt_ho_real.to_vec(),
                    )
                };
                crate::numerical_comparison::assert_values_close(
                    oo,
                    scalars(lines[2]),
                    1e-12,
                    1e-12,
                    format!("{case} sampled SR OO"),
                );
                crate::numerical_comparison::assert_values_close(
                    ho,
                    scalars(lines[3]),
                    1e-12,
                    1e-12,
                    format!("{case} sampled SR HO"),
                );
                if store == 1 {
                    let fixture = read_fixture("gram.txt");
                    let expected = scalars(fixture.lines().nth(1).unwrap());
                    let actual = if get_all_complex_flag(&data).unwrap() {
                        components(&state.sr_opt.sr_opt_o_store)
                    } else {
                        state.sr_opt.sr_opt_o_store_real.to_vec()
                    };
                    assert_eq!(actual.len(), expected.len());
                    for (i, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                        crate::numerical_comparison::assert_close(
                            *actual,
                            *expected,
                            1e-12,
                            1e-12,
                            format!("{case} sampled SR O store component {i}"),
                        );
                    }
                }
            }
            let mut rbm_values = Vec::new();
            data.visit_rbm_terms_mut(|_, t| rbm_values.push(t.value()));
            let values = data
                .gutzwiller_terms
                .iter()
                .map(|t| t.value)
                .chain(data.jastrow_terms.iter().map(|t| t.value))
                .chain(data.doublon_holon_2site_params.iter().copied())
                .chain(data.doublon_holon_4site_params.iter().copied())
                .chain(rbm_values)
                .chain(
                    data.orbital_terms
                        .iter()
                        .map(|t| data.slater_params[t.idx as usize]),
                )
                .chain(data.opt_trans.iter().copied());
            let actual: Vec<f64> = values.flat_map(|v| [v.re, v.im]).collect();
            crate::numerical_comparison::assert_values_close(
                actual,
                scalars(&read("parameters")),
                1e-11,
                1e-11,
                format!("step {steps} parameters"),
            );
            crate::numerical_comparison::assert_values_close(
                [state.energy.etot.re, state.energy.etot.im]
                    .as_slice()
                    .iter()
                    .copied(),
                scalars(&read("energy")),
                1e-11,
                1e-11,
                format!("step {steps} energy"),
            );
            if cg {
                // Diagnostics print only five digits after the decimal point.
                // One final printed quantum can differ at a rounding boundary.
                // Dimensions, cuts, index and iteration count remain exact.
                crate::numerical_comparison::assert_numeric_text(
                    &fs::read_to_string(dir.join("zvo_SRinfo.dat")).unwrap(),
                    &read("SRinfo"),
                    1e-12,
                    1e-5,
                    &[0, 1, 2, 3, 7, 8],
                    format!("{case} step {steps} SR diagnostics"),
                );
            }
            if (reviewed_cg || (steps == 20 && reviewed_direct))
                && (case.starts_with("dh") || case.starts_with("rbm_") || case.starts_with("opt_"))
                || (matches!(
                    case,
                    "dh2_real"
                        | "dh2_cmp"
                        | "dh4_real"
                        | "dh4_cmp"
                        | "dh24_real"
                        | "dh24_cmp"
                        | "rbm_real"
                        | "rbm_cmp"
                        | "rbm_general_cmp"
                        | "rbm_dh24_cmp"
                        | "opt_real"
                        | "opt_cmp"
                        | "opt_dh24_rbm_cmp"
                ) && (store == 0 || !cg))
                || (matches!(case, "rbm_fsz" | "rbm_reference_cmp") && (store == 0 || !cg))
                || (matches!(case, "dh2_fsz" | "dh4_fsz") && (store == 0 || !cg))
                || (case == "dh24_fsz" && (store == 0 || !cg))
                || (case == "opt_fsz" && !cg)
            {
                let historical_expected_dir = fixtures
                    .join(format!(
                        "runner_opt_windows/{case}/{}-store{store}",
                        if cg { "cg" } else { "direct" }
                    ))
                    .join(format!("step-{steps}"));
                // Platform-specific reviewed windows override the archived Linux
                // directory as a whole, so the manifest, c-window-input and
                // zqp_*.dat files come from the same generated stage.
                let reviewed_expected_dir = julia_fixture::arm_directory(&fixtures)
                    .map(|root| {
                        root.join("reviewed_cg_62b")
                            .join(reviewed_case_name)
                            .join(format!("step-{steps}"))
                    })
                    .filter(|dir| dir.is_dir())
                    .unwrap_or_else(|| reviewed_case_dir.join(format!("step-{steps}")));
                let expected_dir = if reviewed_cg {
                    reviewed_expected_dir
                } else if steps == 20 && reviewed_direct {
                    reviewed_direct_dir.join("step-20")
                } else {
                    historical_expected_dir
                };
                let output_names = |directory: &Path| {
                    let mut names: Vec<_> = fs::read_dir(directory)
                        .unwrap()
                        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
                        .filter(|name| name.starts_with("zqp_") && name.ends_with(".dat"))
                        .collect();
                    names.sort();
                    names
                };
                let expected_names = output_names(&expected_dir);
                if failed {
                    assert!(expected_names.is_empty(), "{case} {steps}: failed SR must not aggregate partial history as final output");
                    assert!(expected_dir
                        .join("successful-history-not-final-output.txt")
                        .exists());
                    assert!(!expected_dir.join("c-window-input.txt").exists());
                } else {
                    assert!(expected_dir.join("c-window-input.txt").exists());
                }
                // C window=1 emits only the contiguous main row. Larger windows
                // emit every active block, including both DH sections.
                assert_eq!(
                    output_names(&dir),
                    expected_names,
                    "{case} {steps} C output manifest"
                );
                for name in expected_names.into_iter().chain(["zvo_var.dat".into()]) {
                    crate::numerical_comparison::assert_numeric_text(
                        &fs::read_to_string(dir.join(&name)).unwrap(),
                        &fs::read_to_string(expected_dir.join(&name)).unwrap(),
                        1e-11,
                        1e-11,
                        if name.starts_with("zqp_") && name != "zqp_opt.dat" {
                            &[0]
                        } else {
                            &[]
                        },
                        format!("{case} {steps} independent declared C window {name}"),
                    );
                }
                crate::numerical_comparison::assert_numeric_text(
                    &fs::read_to_string(dir.join("zvo_out.dat")).unwrap(),
                    &read_fixture(&format!("step-{steps}-zvo_out.dat")),
                    1e-11,
                    1e-11,
                    &[],
                    format!("{case} {steps} archived energy output"),
                );
            } else if case.starts_with("dh2_")
                || case.starts_with("dh4_")
                || case.starts_with("dh24_")
                || case.starts_with("rbm_")
                || case.starts_with("opt_")
            {
                for name in [
                    "zvo_out.dat",
                    "zvo_var.dat",
                    "zqp_opt.dat",
                    "zqp_gutzwiller_opt.dat",
                    "zqp_jastrow_opt.dat",
                    "zqp_orbital_opt.dat",
                ] {
                    if failed && name.starts_with("zqp_") {
                        assert!(!dir.join(name).exists(), "{case} {steps} {name}");
                        assert_eq!(
                            read_fixture(&format!("step-{steps}-{name}")),
                            "# absent after source SR failure\n"
                        );
                    } else {
                        crate::numerical_comparison::assert_numeric_text(
                            &fs::read_to_string(dir.join(name)).unwrap(),
                            &declared_runner_output(
                                &data,
                                name,
                                read_fixture(&format!("step-{steps}-{name}")),
                                &rbm_before_sr,
                            ),
                            1e-11,
                            1e-11,
                            if name.starts_with("zqp_") && name != "zqp_opt.dat" {
                                &[0]
                            } else {
                                &[]
                            },
                            format!("{case} {steps} {name}"),
                        );
                    }
                }
                assert!(!dir.join("zqp_dh2_opt.dat").exists());
                assert!(!dir.join("zqp_dh4_opt.dat").exists());
            }
            fs::remove_dir_all(dir).unwrap();
        }
    }

    #[test]
    fn dh24_real_direct_output_matches_independent_complete_c_windows() {
        check_sr_prefixes("dh24_real", false, 0);
    }

    #[test]
    fn canonical_cg_fixed_seed_same_configuration_is_repeatable() {
        let run = || {
            let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../extern/Julia-mVMC/test/integration/reference/general_rbm_cmp/inputs/namelist.def");
            let mut data = parse_expert_mode_files(&path).unwrap();
            data.modpara.nsr_opt_itr_step = 3;
            data.modpara.nsr_opt_itr_smp = 3;
            data.modpara.nsrcg = 1;
            data.modpara.nstore_o = 0;
            let mut rng = Sfmt19937Rng::new(12395);
            init_parameter(&mut data, &mut rng).unwrap();
            assert!(
                read_initial_def(&mut data, path.parent().unwrap().join("initial.def")).unwrap()
            );
            read_input_parameters(&mut data, &path).unwrap();
            sync_modified_parameter(&mut data, true);
            init_qp_weight(&mut data);
            let mut state = state_from_data(&data).unwrap();
            let dir = fresh_output_directory().unwrap();
            vmc_para_opt(
                &mut data,
                &mut state,
                &mut rng,
                Some(&dir),
                &SingleProcessReducer,
                OptimizationOptions::default(),
            )
            .unwrap();
            (data, state, rng, dir)
        };
        let (a, sa, mut ra, da) = run();
        let (b, sb, mut rb, db) = run();
        // Same Rust implementation/configuration: exact identity is appropriate,
        // not a cross-language computed-floating comparison policy.
        assert_eq!(a.projection_parameters(), b.projection_parameters());
        assert_eq!(a.rbm_parameters(), b.rbm_parameters());
        assert_eq!(a.slater_params, b.slater_params);
        assert_eq!(a.opt_trans, b.opt_trans);
        assert_eq!(a.optimization_flags, b.optimization_flags);
        assert_eq!(sa.energy.wc, sb.energy.wc);
        assert_eq!(sa.energy.etot, sb.energy.etot);
        assert_eq!(sa.energy.etot2, sb.energy.etot2);
        assert_eq!(sa.sr_opt.sr_opt_oo, sb.sr_opt.sr_opt_oo);
        assert_eq!(sa.sr_opt.sr_opt_ho, sb.sr_opt.sr_opt_ho);
        assert_eq!(sa.sr_opt.sr_opt_o_store, sb.sr_opt.sr_opt_o_store);
        assert_eq!(sa.electron_config.ele_idx, sb.electron_config.ele_idx);
        assert_eq!(sa.electron_config.ele_cfg, sb.electron_config.ele_cfg);
        assert_eq!(sa.electron_config.ele_num, sb.electron_config.ele_num);
        assert_eq!(sa.electron_config.ele_spn, sb.electron_config.ele_spn);
        assert_eq!(
            sa.electron_config.ele_proj_cnt,
            sb.electron_config.ele_proj_cnt
        );
        assert_eq!(
            sa.electron_config.burn_ele_idx,
            sb.electron_config.burn_ele_idx
        );
        assert_eq!(sa.electron_config.counter, sb.electron_config.counter);
        assert_eq!(ra.words_consumed(), rb.words_consumed());
        for _ in 0..624 {
            assert_eq!(ra.gen_rand32(), rb.gen_rand32());
        }
        let files = |dir: &Path| {
            let mut files: Vec<_> = fs::read_dir(dir)
                .unwrap()
                .map(|e| e.unwrap().file_name())
                .filter(|name| !name.to_string_lossy().contains("time"))
                .collect();
            files.sort();
            files
        };
        let names = files(&da);
        assert_eq!(names, files(&db));
        for name in names {
            assert_eq!(
                fs::read(da.join(&name)).unwrap(),
                fs::read(db.join(&name)).unwrap(),
                "{name:?}"
            );
        }
        fs::remove_dir_all(da).unwrap();
        fs::remove_dir_all(db).unwrap();
    }

    #[test]
    fn optimization_measurement_observer_rejects_nesting_and_cleans_up_unwind() {
        use std::{cell::Cell, rc::Rc};
        struct Counter(Cell<usize>);
        impl OptimizationMeasurementObserver for Counter {
            fn measured(&self, _: OptimizationMeasurementView<'_>) {
                self.0.set(self.0.get() + 1);
            }
        }
        let outer = Rc::new(Counter(Cell::new(0)));
        let guard = install_optimization_measurement_observer(outer.clone()).unwrap();
        assert!(install_optimization_measurement_observer(Rc::new(Counter(Cell::new(0)))).is_err());
        std::thread::spawn(|| {
            let guard =
                install_optimization_measurement_observer(Rc::new(Counter(Cell::new(0)))).unwrap();
            drop(guard);
            assert!(OPTIMIZATION_MEASUREMENT_OBSERVER.with(|slot| slot.borrow().is_none()));
        })
        .join()
        .unwrap();
        let (data, state, _) = prepared_case(1, "heisenberg_chain_real");
        observe_optimization_measurement(OptimizationMeasurementView {
            data: &data,
            state: &state,
            sample: 0,
            overlap: Complex64::new(1.0, 0.0),
            local_energy: Complex64::new(2.0, 0.0),
            weight: 1.0,
        });
        assert_eq!(outer.0.get(), 1);
        let mut physcal = data.clone();
        physcal.modpara.vmc_calc_mode = 1;
        observe_optimization_measurement(OptimizationMeasurementView {
            data: &physcal,
            state: &state,
            sample: 0,
            overlap: Complex64::new(1.0, 0.0),
            local_energy: Complex64::new(2.0, 0.0),
            weight: 1.0,
        });
        assert_eq!(outer.0.get(), 1);
        drop(guard);
        assert!(std::panic::catch_unwind(|| {
            let _guard =
                install_optimization_measurement_observer(Rc::new(Counter(Cell::new(0)))).unwrap();
            panic!("observer scope unwinds");
        })
        .is_err());
        assert!(OPTIMIZATION_MEASUREMENT_OBSERVER.with(|slot| slot.borrow().is_none()));
    }

    #[test]
    fn actual_cg_sampling_observer_preserves_configs_rng_count_and_next624() {
        use crate::sr_cg::{CgObserver, CgProductPhase};
        use std::{cell::Cell, rc::Rc};
        #[derive(Default)]
        struct CountProducts(Cell<usize>);
        impl CgObserver for CountProducts {
            fn product(&self, _: CgProductPhase, _: &[f64], _: &[f64]) {
                self.0.set(self.0.get() + 1);
            }
        }
        #[derive(Default)]
        struct CountMeasurements(Cell<usize>);
        impl OptimizationMeasurementObserver for CountMeasurements {
            fn measured(&self, view: OptimizationMeasurementView<'_>) {
                assert_eq!(view.data.modpara.vmc_calc_mode, 0);
                assert!(view.sample < view.data.modpara.nvmc_sample as usize);
                assert!(view.overlap.norm() > 0.0);
                assert!(view.local_energy.re.is_finite());
                assert!(!view.state.sr_opt.sr_opt_o.is_empty());
                self.0.set(self.0.get() + 1);
            }
        }
        for case in [
            "heisenberg_chain_real",
            "heisenberg_chain_cmp",
            "heisenberg_chain_fsz",
        ] {
            for (cg, store) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                let (mut baseline, mut base_state, mut base_rng) = prepared_case(3, case);
                baseline.modpara.nsrcg = cg;
                baseline.modpara.nstore_o = store;
                let (mut data, mut state, mut rng) = prepared_case(3, case);
                data.modpara.nsrcg = cg;
                data.modpara.nstore_o = store;
                let baseline_dir = fresh_output_directory().unwrap();
                let observed_dir = fresh_output_directory().unwrap();
                vmc_para_opt(
                    &mut baseline,
                    &mut base_state,
                    &mut base_rng,
                    Some(&baseline_dir),
                    &SingleProcessReducer,
                    OptimizationOptions::default(),
                )
                .unwrap();
                let observer = Rc::new(CountProducts::default());
                let guard = crate::sr_cg::install_cg_observer(observer.clone()).unwrap();
                let measurements = Rc::new(CountMeasurements::default());
                let measurement_guard =
                    install_optimization_measurement_observer(measurements.clone()).unwrap();
                vmc_para_opt(
                    &mut data,
                    &mut state,
                    &mut rng,
                    Some(&observed_dir),
                    &SingleProcessReducer,
                    OptimizationOptions::default(),
                )
                .unwrap();
                drop(guard);
                drop(measurement_guard);
                assert_eq!(
                    observer.0.get() > 0,
                    cg != 0,
                    "{case} CG={cg} store={store}"
                );
                assert_eq!(measurements.0.get(), 3 * data.modpara.nvmc_sample as usize);
                // Exact Rust-to-Rust observer identity, not a floating reference policy.
                let parameters = |data: &ExpertModeData| {
                    data.projection_parameters()
                        .into_iter()
                        .chain(data.rbm_parameters())
                        .chain(data.slater_params.iter().copied())
                        .chain(data.opt_trans.iter().copied())
                        .collect::<Vec<_>>()
                };
                assert_eq!(parameters(&data), parameters(&baseline));
                assert_eq!(data.optimization_flags, baseline.optimization_flags);
                assert_eq!(state.energy.wc, base_state.energy.wc);
                assert_eq!(state.energy.etot, base_state.energy.etot);
                assert_eq!(state.energy.etot2, base_state.energy.etot2);
                assert_eq!(
                    state.sr_opt.sr_opt_oo_real,
                    base_state.sr_opt.sr_opt_oo_real
                );
                assert_eq!(
                    state.sr_opt.sr_opt_ho_real,
                    base_state.sr_opt.sr_opt_ho_real
                );
                assert_eq!(state.sr_opt.sr_opt_oo, base_state.sr_opt.sr_opt_oo);
                assert_eq!(state.sr_opt.sr_opt_ho, base_state.sr_opt.sr_opt_ho);
                assert_eq!(
                    state.sr_opt.sr_opt_o_store_real,
                    base_state.sr_opt.sr_opt_o_store_real
                );
                assert_eq!(
                    state.sr_opt.sr_opt_o_store,
                    base_state.sr_opt.sr_opt_o_store
                );
                let files = |directory: &Path| {
                    let mut files: Vec<_> = fs::read_dir(directory)
                        .unwrap()
                        .map(|entry| entry.unwrap().file_name())
                        .collect();
                    files.sort();
                    files
                };
                let expected_files = files(&baseline_dir);
                assert_eq!(files(&observed_dir), expected_files);
                for file in expected_files {
                    assert_eq!(
                        fs::read(observed_dir.join(&file)).unwrap(),
                        fs::read(baseline_dir.join(&file)).unwrap(),
                        "observer identity: {file:?}"
                    );
                }
                assert_eq!(
                    state.electron_config.ele_cfg,
                    base_state.electron_config.ele_cfg
                );
                assert_eq!(
                    state.electron_config.ele_idx,
                    base_state.electron_config.ele_idx
                );
                assert_eq!(
                    state.electron_config.ele_num,
                    base_state.electron_config.ele_num
                );
                assert_eq!(
                    state.electron_config.ele_spn,
                    base_state.electron_config.ele_spn
                );
                assert_eq!(
                    state.electron_config.counter,
                    base_state.electron_config.counter
                );
                assert_eq!(rng.words_consumed(), base_rng.words_consumed());
                assert_eq!(format!("{rng:?}"), format!("{base_rng:?}"));
                for _ in 0..624 {
                    assert_eq!(rng.gen_rand32(), base_rng.gen_rand32());
                }
                fs::remove_dir_all(baseline_dir).unwrap();
                fs::remove_dir_all(observed_dir).unwrap();
            }
        }
    }

    #[test]
    fn standard_cg_runs_three_steps_and_writes_srinfo() {
        for case in [
            "heisenberg_chain_real",
            "heisenberg_chain_cmp",
            "heisenberg_chain_fsz",
        ] {
            let (mut data, mut state, mut rng) = prepared_case(3, case);
            data.modpara.nsrcg = 1;
            data.modpara.nstore_o = 0;
            let dir = fresh_output_directory().unwrap();
            vmc_para_opt(
                &mut data,
                &mut state,
                &mut rng,
                Some(&dir),
                &SingleProcessReducer,
                OptimizationOptions::default(),
            )
            .unwrap();
            let info = fs::read_to_string(dir.join("zvo_SRinfo.dat")).unwrap();
            assert_eq!(info.lines().count(), 4, "{case}");
            assert_eq!(state.opt_data.len(), 3);
            assert!(data
                .slater_params
                .iter()
                .all(|t| t.re.is_finite() && t.im.is_finite()));
            let mut hash = 0xcbf29ce484222325_u64;
            for _ in 0..624 {
                hash = (hash ^ u64::from(rng.gen_rand32())).wrapping_mul(0x100000001b3);
            }
            let julia_hash = match case {
                "heisenberg_chain_real" => 382483484918994011,
                "heisenberg_chain_cmp" => 5883921295860317420,
                "heisenberg_chain_fsz" => 6705941385670463079,
                _ => unreachable!(),
            };
            assert_eq!(hash, julia_hash, "{case}: Julia SFMT block mismatch");
            fs::remove_dir_all(dir).unwrap();
        }
    }

    #[test]
    fn cg_accumulation_forces_store_even_when_nstore_is_zero() {
        for case in [
            "heisenberg_chain_real",
            "heisenberg_chain_cmp",
            "heisenberg_chain_fsz",
        ] {
            let (mut data, mut state, mut rng) = prepared_case(1, case);
            let dir = fresh_output_directory().unwrap();
            vmc_para_opt(
                &mut data,
                &mut state,
                &mut rng,
                Some(&dir),
                &SingleProcessReducer,
                OptimizationOptions {
                    skip_sr: true,
                    ..OptimizationOptions::default()
                },
            )
            .unwrap();
            // Exercise the CG accumulator with the already sampled walkers.
            data.modpara.nstore_o = 0;
            data.modpara.nsrcg = 1;
            clear_phys_quantity(&mut state);
            let complex = get_all_complex_flag(&data).unwrap();
            let fsz = data.i_flg_orbital_general != 0;
            let mut timer = CTimer::<true>::new();
            accumulate_observables(
                &data,
                &mut state,
                complex,
                fsz,
                &mut timer,
                &SingleProcessReducer,
            );
            assert!(timer.elapsed_ns[45] > 0);
            let samples = data.modpara.nvmc_sample as usize;
            let n = state.sr_opt.sr_opt_size;
            if complex {
                assert_eq!(
                    state.sr_opt.sr_opt_oo[0],
                    Complex64::new(samples as f64, 0.0)
                );
                let n = 2 * n;
                for i in 0..n {
                    let mut mean = Complex64::new(0.0, 0.0);
                    let mut diagonal = 0.0;
                    for s in 0..samples {
                        let o = state.sr_opt.sr_opt_o_store[i + s * n];
                        mean += o;
                        diagonal += o.norm_sqr();
                    }
                    assert_eq!(state.sr_opt.sr_opt_oo[i], mean);
                    assert_eq!(state.sr_opt.sr_opt_oo[i + n], Complex64::new(diagonal, 0.0));
                }
                assert!(state.sr_opt.sr_opt_oo[2 * n..]
                    .iter()
                    .all(|&z| z == Complex64::new(0.0, 0.0)));
            } else {
                assert_eq!(state.sr_opt.sr_opt_oo_real[0], samples as f64);
                for i in 0..n {
                    let mut mean = 0.0;
                    let mut diagonal = 0.0;
                    for s in 0..samples {
                        let o = state.sr_opt.sr_opt_o_store_real[i + s * n];
                        mean += o;
                        diagonal += o * o;
                    }
                    assert_eq!(state.sr_opt.sr_opt_oo_real[i], mean);
                    assert_eq!(state.sr_opt.sr_opt_oo_real[i + n], diagonal);
                }
                assert!(state.sr_opt.sr_opt_oo_real[2 * n..]
                    .iter()
                    .all(|&v| v == 0.0));
            }
            fs::remove_dir_all(dir).unwrap();
        }
    }

    #[test]
    fn nonfinite_local_energy_skips_energy_sr_and_sample_store() {
        for case in [
            "heisenberg_chain_real",
            "heisenberg_chain_cmp",
            "heisenberg_chain_fsz",
        ] {
            let (mut data, mut state, mut rng) = prepared_case(1, case);
            data.modpara.nstore_o = 1;
            let dir = fresh_output_directory().unwrap();
            vmc_para_opt(
                &mut data,
                &mut state,
                &mut rng,
                Some(&dir),
                &SingleProcessReducer,
                OptimizationOptions {
                    skip_sr: true,
                    ..OptimizationOptions::default()
                },
            )
            .unwrap();
            assert!(state.energy.wc.re > 0.0);
            data.coulomb_intra_terms = vec![mvmc_expert_parsers::CoulombIntraTerm {
                site: 0,
                value: f64::NAN,
            }];
            clear_phys_quantity(&mut state);
            let complex = get_all_complex_flag(&data).unwrap();
            let fsz = data.i_flg_orbital_general != 0;
            accumulate_observables(
                &data,
                &mut state,
                complex,
                fsz,
                &mut CTimer::<false>::new(),
                &SingleProcessReducer,
            );
            assert_eq!(state.energy.wc, Complex64::new(0.0, 0.0), "{case}");
            assert_eq!(state.energy.etot, Complex64::new(0.0, 0.0), "{case}");
            assert!(state
                .sr_opt
                .sr_opt_oo
                .iter()
                .all(|&z| z == Complex64::new(0.0, 0.0)));
            assert!(state
                .sr_opt
                .sr_opt_ho
                .iter()
                .all(|&z| z == Complex64::new(0.0, 0.0)));
            assert!(state
                .sr_opt
                .sr_opt_o_store
                .iter()
                .all(|&z| z == Complex64::new(0.0, 0.0)));
            assert!(state.sr_opt.sr_opt_oo_real.iter().all(|&v| v == 0.0));
            assert!(state.sr_opt.sr_opt_ho_real.iter().all(|&v| v == 0.0));
            assert!(state.sr_opt.sr_opt_o_store_real.iter().all(|&v| v == 0.0));
            fs::remove_dir_all(dir).unwrap();
        }
    }

    #[test]
    fn invalid_saved_walkers_clear_previous_sample_store() {
        for case in [
            "heisenberg_chain_real",
            "heisenberg_chain_cmp",
            "heisenberg_chain_fsz",
        ] {
            let (mut data, mut state, _) = prepared_case(1, case);
            data.modpara.nstore_o = 1;
            state.electron_config.ele_idx.fill(-1);
            state
                .sr_opt
                .sr_opt_o_store
                .fill(Complex64::new(123.0, -456.0));
            state.sr_opt.sr_opt_o_store_real.fill(123.0);
            clear_phys_quantity(&mut state);
            let complex = get_all_complex_flag(&data).unwrap();
            let fsz = data.i_flg_orbital_general != 0;
            let mut timer = CTimer::<true>::new();
            accumulate_observables(
                &data,
                &mut state,
                complex,
                fsz,
                &mut timer,
                &SingleProcessReducer,
            );
            assert!(timer.elapsed_ns[45] > 0);
            assert!(state
                .sr_opt
                .sr_opt_o_store
                .iter()
                .all(|z| *z == Complex64::new(0.0, 0.0)));
            assert!(state.sr_opt.sr_opt_o_store_real.iter().all(|&v| v == 0.0));
            assert!(state
                .sr_opt
                .sr_opt_oo
                .iter()
                .all(|z| *z == Complex64::new(0.0, 0.0)));
            assert!(state.sr_opt.sr_opt_oo_real.iter().all(|&v| v == 0.0));
        }
    }

    #[test]
    fn nstore_preserves_three_step_optimization_trajectory() {
        for case in [
            "heisenberg_chain_real",
            "heisenberg_chain_cmp",
            "heisenberg_chain_fsz",
            "hubbard_chain_real",
        ] {
            let (mut direct, mut direct_state, mut direct_rng) = prepared_case(3, case);
            direct.modpara.nstore_o = 0;
            let direct_dir = fresh_output_directory().unwrap();
            vmc_para_opt(
                &mut direct,
                &mut direct_state,
                &mut direct_rng,
                Some(&direct_dir),
                &SingleProcessReducer,
                OptimizationOptions::default(),
            )
            .unwrap();
            let (mut stored, mut stored_state, mut stored_rng) = prepared_case(3, case);
            stored.modpara.nstore_o = 1;
            let stored_dir = fresh_output_directory().unwrap();
            vmc_para_opt(
                &mut stored,
                &mut stored_state,
                &mut stored_rng,
                Some(&stored_dir),
                &SingleProcessReducer,
                OptimizationOptions::default(),
            )
            .unwrap();
            // Julia also produces different optimized parameters for NStore=0
            // and 1 (different OO reduction paths). Compare the discrete
            // trajectory exactly rather than asserting equality of those APIs.
            assert_eq!(
                direct_state.electron_config, stored_state.electron_config,
                "{case}"
            );
            let mut hash = 0xcbf29ce484222325_u64;
            for _ in 0..624 {
                let word = direct_rng.gen_rand32();
                assert_eq!(word, stored_rng.gen_rand32(), "{case}");
                hash = (hash ^ u64::from(word)).wrapping_mul(0x100000001b3);
            }
            // Julia v0.5.0, seed=1, three SR steps, both NStore settings.
            // Regenerate via scripts/check_sample_store_parity.jl.
            let julia_hash = match case {
                "heisenberg_chain_real" => 382483484918994011,
                "heisenberg_chain_cmp" => 5883921295860317420,
                "heisenberg_chain_fsz" => 6705941385670463079,
                "hubbard_chain_real" => 5863593240845525434,
                _ => unreachable!(),
            };
            assert_eq!(hash, julia_hash, "{case}: Julia SFMT block mismatch");
            fs::remove_dir_all(direct_dir).unwrap();
            fs::remove_dir_all(stored_dir).unwrap();
        }
    }

    #[test]
    fn nstore_controls_production_buffers_without_changing_sampling() {
        for case in [
            "heisenberg_chain_real",
            "heisenberg_chain_cmp",
            "heisenberg_chain_fsz",
            "hubbard_chain_real",
        ] {
            let (mut direct, mut direct_state, mut direct_rng) = prepared_case(1, case);
            direct.modpara.nstore_o = 0;
            let direct_dir = fresh_output_directory().unwrap();
            vmc_para_opt(
                &mut direct,
                &mut direct_state,
                &mut direct_rng,
                Some(&direct_dir),
                &SingleProcessReducer,
                OptimizationOptions {
                    skip_sr: true,
                    ..OptimizationOptions::default()
                },
            )
            .unwrap();
            let (mut stored, mut stored_state, mut stored_rng) = prepared_case(1, case);
            stored.modpara.nstore_o = 1;
            // Only current valid samples may survive a new main-calculation call.
            stored_state
                .sr_opt
                .sr_opt_o_store
                .fill(Complex64::new(123.0, -456.0));
            stored_state.sr_opt.sr_opt_o_store_real.fill(123.0);
            let stored_dir = fresh_output_directory().unwrap();
            let mut timer = CTimer::<true>::new();
            vmc_para_opt_timed(
                &mut stored,
                &mut stored_state,
                &mut stored_rng,
                Some(&stored_dir),
                &SingleProcessReducer,
                OptimizationOptions {
                    skip_sr: true,
                    ..OptimizationOptions::default()
                },
                &mut timer,
            )
            .unwrap();
            assert!(timer.elapsed_ns[45] > 0, "{case}: no Gram finalization");
            if get_all_complex_flag(&stored).unwrap() {
                assert_eq!(
                    stored_state.sr_opt.sr_opt_o_store[0],
                    Complex64::new(1.0, 0.0)
                );
                assert!(direct_state
                    .sr_opt
                    .sr_opt_o_store
                    .iter()
                    .all(|value| value.norm() == 0.0));
            } else {
                assert_eq!(stored_state.sr_opt.sr_opt_o_store_real[0], 1.0);
                assert!(direct_state
                    .sr_opt
                    .sr_opt_o_store_real
                    .iter()
                    .all(|&value| value == 0.0));
            }
            for (left, right) in direct_state
                .sr_opt
                .sr_opt_oo
                .iter()
                .zip(&stored_state.sr_opt.sr_opt_oo)
            {
                // Julia's direct complex OO uses conj(O_i)*O_j; the store
                // finalizer uses O_i*conj(O_j), including the mean row.
                assert!(
                    (left.conj() - *right).norm() < 1e-12,
                    "{case}: complex OO mismatch"
                );
            }
            for (left, right) in direct_state
                .sr_opt
                .sr_opt_oo_real
                .iter()
                .zip(&stored_state.sr_opt.sr_opt_oo_real)
            {
                assert!((left - right).abs() < 1e-12, "{case}: real OO mismatch");
            }
            assert_eq!(direct_state.sr_opt.sr_opt_ho, stored_state.sr_opt.sr_opt_ho);
            assert_eq!(
                direct_state.sr_opt.sr_opt_ho_real,
                stored_state.sr_opt.sr_opt_ho_real
            );
            assert_eq!(direct_state.electron_config, stored_state.electron_config);
            assert_eq!(direct_state.energy, stored_state.energy);
            assert_eq!(
                fs::read(direct_dir.join("zvo_out.dat")).unwrap(),
                fs::read(stored_dir.join("zvo_out.dat")).unwrap()
            );
            for _ in 0..624 {
                assert_eq!(direct_rng.gen_rand32(), stored_rng.gen_rand32());
            }
            fs::remove_dir_all(direct_dir).unwrap();
            fs::remove_dir_all(stored_dir).unwrap();
        }
    }
}

#[cfg(test)]
mod physcal_green_observer_tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    struct RawFrame {
        sample: usize,
        weight: Complex64,
        one_body: Vec<Complex64>,
    }
    #[derive(Default)]
    struct Capture(RefCell<Vec<RawFrame>>);
    impl PhysCalGreenObserver for Capture {
        fn accumulated(&self, view: PhysCalGreenView<'_>) {
            assert_eq!(view.data.modpara.vmc_calc_mode, 1);
            self.0.borrow_mut().push(RawFrame {
                sample: view.sample,
                weight: view.weight,
                one_body: view.one_body.to_vec(),
            });
        }
    }

    #[test]
    fn physcal_green_observer_nested_scope_and_unwind_cleanup() {
        let outer = install_physcal_green_observer(Rc::new(Capture::default())).unwrap();
        assert!(install_physcal_green_observer(Rc::new(Capture::default())).is_err());
        std::thread::spawn(|| {
            let _guard = install_physcal_green_observer(Rc::new(Capture::default())).unwrap();
        })
        .join()
        .unwrap();
        drop(outer);
        let result = std::panic::catch_unwind(|| {
            let _guard = install_physcal_green_observer(Rc::new(Capture::default())).unwrap();
            with_physcal_green_sample(9, || panic!("intentional observer scope unwind"));
        });
        assert!(result.is_err());
        assert_eq!(PHYSCAL_GREEN_SAMPLE.with(std::cell::Cell::get), None);
        let _guard = install_physcal_green_observer(Rc::new(Capture::default())).unwrap();
    }

    #[test]
    fn physcal_green_observer_actual_runner_identity_and_raw_boundary() {
        check_runner_identity("heisenberg_chain_real");
    }

    #[test]
    fn physcal_green_observer_fsz_runner_identity_and_raw_boundary() {
        check_runner_identity("heisenberg_chain_fsz");
    }

    fn check_runner_identity(case: &str) {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
            "../../extern/Julia-mVMC/test/integration/reference/{case}/physcal_ref/inputs/namelist.def"));
        let prepare = || {
            let mut prepared = prepare_phys_cal_from_namelist(
                &path,
                path.parent().unwrap().parent().unwrap().join("zqp_opt.dat"),
                "real",
                Some(1),
            )
            .unwrap();
            prepared.data.modpara.n_data_qty_smp = 2;
            prepared
        };
        let baseline_dir = fresh_output_directory().unwrap();
        let observed_dir = fresh_output_directory().unwrap();
        let before = std::time::Instant::now();
        let baseline = vmc_phys_cal_to_dir(prepare(), &baseline_dir).unwrap();
        let disabled_elapsed = before.elapsed();
        let capture = Rc::new(Capture::default());
        let guard = install_physcal_green_observer(capture.clone()).unwrap();
        let before = std::time::Instant::now();
        let observed = vmc_phys_cal_to_dir(prepare(), &observed_dir).unwrap();
        let enabled_elapsed = before.elapsed();
        drop(guard);
        eprintln!("PhysCal observational timing only: disabled={disabled_elapsed:?} enabled={enabled_elapsed:?}");
        assert_eq!(baseline.data.slater_params, observed.data.slater_params);
        assert_eq!(baseline.state.energy, observed.state.energy);
        assert_eq!(
            baseline.state.phys_quantities,
            observed.state.phys_quantities
        );
        assert_eq!(
            baseline.state.electron_config,
            observed.state.electron_config
        );
        assert_eq!(
            observed.data.i_flg_orbital_general != 0,
            case.ends_with("fsz")
        );
        let mut baseline_rng = baseline.final_rng;
        let mut observed_rng = observed.final_rng;
        assert_eq!(baseline_rng.words_consumed(), observed_rng.words_consumed());
        // Debug includes all internal SFMT words, current index and draw count.
        // Compare before peeking/consuming any future word.
        assert_eq!(format!("{baseline_rng:?}"), format!("{observed_rng:?}"));
        for word in 0..624 {
            assert_eq!(
                baseline_rng.gen_rand32(),
                observed_rng.gen_rand32(),
                "observer RNG word{word}"
            );
        }
        let rows = capture.0.borrow();
        assert_eq!(
            rows.iter().map(|row| row.sample).collect::<Vec<_>>(),
            [0, 1]
        );
        let weight = rows[1].weight;
        let raw = &rows[1].one_body;
        assert!(weight.re > 0.0);
        let normalized = &observed
            .state
            .phys_quantities
            .as_ref()
            .unwrap()
            .phys_cis_ajs;
        assert_eq!(raw.len(), normalized.len());
        assert!(!raw.is_empty());
        // SAME-implementation raw-boundary operation-order assertion, not an
        // independent golden: C averages by reciprocal once then multiplication.
        // Independent numerical authority is the separate nine-native-literal test.
        let inverse_weight = crate::c_complex::divide(Complex64::new(1.0, 0.0), weight);
        for (raw, normalized) in raw.iter().zip(normalized) {
            assert_eq!(*raw * inverse_weight, *normalized);
        }
        let files = |directory: &Path| {
            let mut files: Vec<_> = fs::read_dir(directory)
                .unwrap()
                .map(|entry| {
                    let entry = entry.unwrap();
                    (entry.file_name(), fs::read(entry.path()).unwrap())
                })
                .collect();
            files.sort();
            files
        };
        assert_eq!(files(&baseline_dir), files(&observed_dir));
        fs::remove_dir_all(baseline_dir).unwrap();
        fs::remove_dir_all(observed_dir).unwrap();
    }

    #[test]
    fn physcal_green_observer_does_not_attribute_optimization_to_outer_sample() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../extern/Julia-mVMC/examples/inputs/heisenberg_chain_real/namelist.def");
        let mut data = parse_expert_mode_files(&path).unwrap();
        let mut rng = Sfmt19937Rng::new(1);
        init_parameter(&mut data, &mut rng).unwrap();
        read_input_parameters(&mut data, &path).unwrap();
        sync_modified_parameter(&mut data, true);
        init_qp_weight(&mut data);
        data.modpara.vmc_calc_mode = 1;
        let mut state = state_from_data(&data).unwrap();
        assert!(state.phys_quantities.is_some());
        data.modpara.vmc_calc_mode = 0;
        data.modpara.nsr_opt_itr_step = 1;
        data.modpara.nsr_opt_itr_smp = 1;
        let capture = Rc::new(Capture::default());
        let _guard = install_physcal_green_observer(capture.clone()).unwrap();
        with_physcal_green_sample(7, || {
            vmc_para_opt(
                &mut data,
                &mut state,
                &mut rng,
                None,
                &SingleProcessReducer,
                OptimizationOptions::default(),
            )
        })
        .unwrap();
        assert!(capture.0.borrow().is_empty());
    }
}

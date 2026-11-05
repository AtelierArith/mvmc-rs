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
use crate::measurement_batch::SlotStatus;
use crate::observables::clear_phys_quantity;
use crate::reducer::{Reducer, ReducerAsDyn, SingleProcessReducer};
use crate::slater_update::{update_slater_elm, update_slater_elm_fsz};
use crate::state::{ThreadedPfaPackWorkspace, VmcOptimizationState};
use crate::sync::sync_modified_parameter as sync_modified;
use crate::sync::sync_modified_parameter_local as sync_modified_parameter;

/// The production real, non-FSZ normal-mode sampler of one chain (`VMCMakeSample`): the C-order
/// driver [`vmc_make_sample_real_with_reducer_timed`](crate::sampling::driver::vmc_make_sample_real_with_reducer_timed).
///
/// The driver runs against a fresh internal [`CTimer`] whose per-section counters are merged
/// into `timing`, and a thread-safe reducer runs it inside the QP region team.
fn sample_real_c_order(
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    rng: &mut Sfmt19937Rng,
    timing: &mut dyn crate::stage_backend::MeasurementTiming,
    reducer: &dyn Reducer,
) -> Result<crate::sampling::SampleStats, String> {
    fn run<const TIMED: bool>(
        data: &ExpertModeData,
        state: &mut VmcOptimizationState,
        rng: &mut Sfmt19937Rng,
        timer: &mut CTimer<TIMED>,
        reducer: &dyn Reducer,
    ) -> Result<crate::sampling::SampleStats, String> {
        use crate::sampling::driver::vmc_make_sample_real_with_reducer_timed;
        let result = if let Some(shared) = reducer.host_thread_safe() {
            let width = state.slater_matrix.pf_m_real.len();
            let cost =
                crate::pfaffian::pfaffian_qp_cost_ns(state.slater_matrix.inv_m_real.n_size());
            crate::threading::with_qp_region_team(width, cost, || {
                vmc_make_sample_real_with_reducer_timed(data, state, rng, timer, shared)
            })
        } else {
            vmc_make_sample_real_with_reducer_timed(data, state, rng, timer, reducer)
        };
        result.map_err(|error| format!("failed: {error}"))
    }
    if timing.enabled() {
        let mut timer = CTimer::<true>::new();
        timer.diagnostics = timing.diagnostics();
        let result = run(data, state, rng, &mut timer, reducer);
        timing.merge_elapsed(&timer.elapsed_ns);
        result
    } else {
        let mut timer = CTimer::<false>::new();
        timer.diagnostics = timing.diagnostics();
        run(data, state, rng, &mut timer, reducer)
    }
}

#[cfg(all(test, feature = "mpi"))]
#[path = "run_mpi_tests.rs"]
mod mpi_runtime_tests;

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
    /// Borrow the production values without changing state or consuming RNG.
    fn measured(&self, view: OptimizationMeasurementView<'_>);
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
    /// C `-b` (`FlagBinary`): write `_varbin_` files instead of `_var_` text files.
    pub binary_output: bool,
    /// True when the C `InitParameter` draw block was already consumed by
    /// preparation (the no-parameter-file path), so the driver must not draw again.
    pub initialization_consumed: bool,
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
        Some(opt_para_path.as_ref()),
        mode,
        seed,
        true,
        0,
        &SingleProcessReducer,
    )
}

/// Prepare one PhysCal walker with an explicit seed offset (`RndSeed + offset`, C
/// `init_gen_rand(RndSeed + group1)`), single-process. Used by the multi-chain runner (#425).
pub fn prepare_phys_cal_with_seed_offset(
    namelist_path: impl AsRef<Path>,
    opt_para_path: Option<&Path>,
    mode: &str,
    seed: Option<i64>,
    enable_opt_trans: bool,
    seed_offset: usize,
) -> Result<PhysCalPreparation, String> {
    prepare_phys_cal_from_namelist_with_seed_offset(
        namelist_path,
        opt_para_path,
        mode,
        seed,
        enable_opt_trans,
        seed_offset,
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
        Some(opt_para_path.as_ref()),
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
        Some(opt_para_path.as_ref()),
        mode,
        seed,
        enable_opt_trans,
        reducer.seed_offset(),
        reducer,
    )
}

/// Prepare PhysCal without a fixed parameter file, as C `vmc.out namelist.def`
/// does for `NVMCCalMode=1` (`vmcmain.c:260-268`): `InitParameter` draws, then
/// the `In*` overlays and `SyncModifiedParameter`. The draw block is consumed
/// here and recorded in [`PhysCalPreparation::initialization_consumed`].
pub fn prepare_phys_cal_without_parameter_file_with_reducer_and_opt_trans<R: Reducer + ?Sized>(
    namelist_path: impl AsRef<Path>,
    mode: &str,
    seed: Option<i64>,
    reducer: &R,
    enable_opt_trans: bool,
) -> Result<PhysCalPreparation, String> {
    prepare_phys_cal_from_namelist_with_seed_offset(
        namelist_path,
        None,
        mode,
        seed,
        enable_opt_trans,
        reducer.seed_offset(),
        reducer,
    )
}

fn prepare_phys_cal_from_namelist_with_seed_offset<R: Reducer + ?Sized>(
    namelist_path: impl AsRef<Path>,
    opt_para_path: Option<&Path>,
    mode: &str,
    seed: Option<i64>,
    enable_opt_trans: bool,
    seed_offset: usize,
    reducer: &R,
) -> Result<PhysCalPreparation, String> {
    let Some(opt_para_path) = opt_para_path else {
        // C order: seed -> InitParameter -> (no ReadInitParameter) -> ReadInputParameters
        // -> SyncModifiedParameter. The draws are real, not a discarded clone.
        let parsed = (|| {
            if !matches!(mode, "real" | "cmp" | "fsz") {
                return Err(format!("mode must be :real, :cmp, or :fsz; got :{mode}"));
            }
            mvmc_expert_parsers::parse_expert_mode_files_with_opt_trans(
                namelist_path.as_ref(),
                enable_opt_trans,
            )
            .map_err(|error| error.to_string())
        })();
        let mut data = collective_result(parsed, reducer, "PhysCal parse")?;
        let actual_seed = resolve_rnd_seed(data.modpara.rnd_seed, seed, seed_offset, reducer)?;
        let mut rng = seeded_rng_with_reducer(actual_seed, reducer)?;
        let loaded = (|| {
            init_parameter(&mut data, &mut rng).map_err(str::to_string)?;
            read_input_parameters(&mut data, namelist_path.as_ref())?;
            sync_modified_parameter(&mut data, false);
            crate::validation::validate_phys_cal(&data)?;
            crate::validation::validate_reducer_rank(&data, reducer)
        })();
        collective_result(loaded, reducer, "PhysCal initialization/load/validation")?;
        return Ok(PhysCalPreparation {
            data,
            rng,
            n_para_consumed: 0,
            binary_output: false,
            initialization_consumed: true,
        });
    };
    let loaded = (|| {
        if !matches!(mode, "real" | "cmp" | "fsz") {
            return Err(format!("mode must be :real, :cmp, or :fsz; got :{mode}"));
        }
        let namelist_path = namelist_path.as_ref();
        let mut data = mvmc_expert_parsers::parse_expert_mode_files_with_opt_trans(
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
        binary_output: false,
        initialization_consumed: false,
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
pub fn vmc_phys_cal_with_reducer<R: Reducer + ReducerAsDyn + ?Sized>(
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
pub fn vmc_phys_cal_with_reducer_timed<const TIMED: bool, R: Reducer + ReducerAsDyn + ?Sized>(
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
pub fn vmc_phys_cal_with_reducer_and_callback<R: Reducer + ReducerAsDyn + ?Sized>(
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
pub fn vmc_phys_cal_with_reducer_and_callback_timed<
    const TIMED: bool,
    R: Reducer + ReducerAsDyn + ?Sized,
>(
    mut preparation: PhysCalPreparation,
    output_dir: Option<&Path>,
    reducer: &R,
    callback: Option<&mut PhysCalCallback<'_>>,
    timer: &mut CTimer<TIMED>,
) -> Result<PhysCalResult, String> {
    // The in-place core allocates the correctly sized state after validation
    // and QP setup. Avoid allocating a second full saved chain here.
    let mut state = VmcOptimizationState::zeros(0, 0, 0, 0, 0, 0, false, false);
    let iterations = vmc_phys_cal_in_place_with_options_timed::<TIMED, R>(
        &mut preparation.data,
        &mut state,
        &mut preparation.rng,
        output_dir,
        reducer,
        callback,
        timer,
        PhysCalRunOptions {
            binary_output: preparation.binary_output,
            initialization_consumed: preparation.initialization_consumed,
        },
    )?;
    Ok(PhysCalResult {
        data: preparation.data,
        state,
        final_rng: preparation.rng,
        iterations,
    })
}

/// A walker ready for the sampling stage: data, initialized state and the RNG positioned as
/// `vmc_phys_cal` leaves it immediately before the first `vmc_make_sample` (issue #434).
#[derive(Debug)]
pub struct SamplingWalker {
    /// Expert data after PhysCal's QP-weight setup.
    pub data: ExpertModeData,
    /// State with the Slater table filled.
    pub state: VmcOptimizationState,
    /// Host SFMT stream.
    pub rng: Sfmt19937Rng,
}

/// Run PhysCal's initialization (the C `InitParameter` draw block unless already consumed,
/// QP weights, state construction, `UpdateSlaterElm`) and stop before the first sample. Real
/// normal mode only; the following `vmc_make_sample_real` continues exactly as
/// `vmc_phys_cal` would.
///
/// # Errors
///
/// Validation errors, or `Err` for complex/FSZ inputs.
pub fn prepare_sampling_walker(mut prep: PhysCalPreparation) -> Result<SamplingWalker, String> {
    let data = &mut prep.data;
    crate::validation::validate_phys_cal(data)?;
    if get_all_complex_flag(data)? || data.i_flg_orbital_general != 0 {
        return Err("device sampler supports real normal mode only".to_string());
    }
    if !prep.initialization_consumed {
        let mut init_data = data.clone();
        init_parameter(&mut init_data, &mut prep.rng).map_err(str::to_string)?;
    }
    if data.modpara.nmp_trans == 0 {
        data.modpara.nmp_trans = 1;
    } else if data.modpara.nmp_trans < 0 {
        data.modpara.nmp_trans = data.modpara.nmp_trans.abs();
    }
    data.modpara.vmc_calc_mode = 1;
    init_qp_weight(data);
    let mut state = state_from_data(data)?;
    update_slater_elm(data, &mut state);
    Ok(SamplingWalker {
        data: prep.data,
        state,
        rng: prep.rng,
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
pub fn vmc_phys_cal_in_place<R: Reducer + ReducerAsDyn + ?Sized>(
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
pub fn vmc_phys_cal_in_place_timed<const TIMED: bool, R: Reducer + ReducerAsDyn + ?Sized>(
    data: &mut ExpertModeData,
    state: &mut VmcOptimizationState,
    rng: &mut Sfmt19937Rng,
    output_dir: Option<&Path>,
    reducer: &R,
    callback: Option<&mut PhysCalCallback<'_>>,
    timer: &mut CTimer<TIMED>,
) -> Result<usize, String> {
    vmc_phys_cal_in_place_with_options_timed::<TIMED, R>(
        data,
        state,
        rng,
        output_dir,
        reducer,
        callback,
        timer,
        PhysCalRunOptions::default(),
    )
}

/// Output and initialization controls for the PhysCal driver.
#[derive(Debug, Clone, Copy, Default)]
pub struct PhysCalRunOptions {
    /// C `-b`: write `_varbin_` files instead of `_var_` text files.
    pub binary_output: bool,
    /// The C `InitParameter` draw block was already consumed by the caller.
    pub initialization_consumed: bool,
}

/// [`vmc_phys_cal_in_place_timed`] with explicit [`PhysCalRunOptions`].
#[allow(clippy::too_many_arguments)]
pub fn vmc_phys_cal_in_place_with_options_timed<
    const TIMED: bool,
    R: Reducer + ReducerAsDyn + ?Sized,
>(
    data: &mut ExpertModeData,
    state: &mut VmcOptimizationState,
    rng: &mut Sfmt19937Rng,
    output_dir: Option<&Path>,
    reducer: &R,
    mut callback: Option<&mut PhysCalCallback<'_>>,
    timer: &mut CTimer<TIMED>,
    options: PhysCalRunOptions,
) -> Result<usize, String> {
    // Pin BLAS to one thread for the deterministic serial policy, as `vmc_para_opt_timed` does
    // (#524). Without this the host measurement would run with whatever OpenBLAS threading
    // happens to be active: the C-order path leaves the default pool, while the CUDA sampler
    // stage initializes the pin itself, so selecting the device backend would change the
    // measurement's summation order and its last bits (issue #517).
    crate::serial_blas::initialize();
    let all_complex = collective_result(
        crate::validation::validate_phys_cal(data)
            .and_then(|()| crate::validation::validate_reducer_rank(data, reducer))
            .and_then(|()| get_all_complex_flag(data)),
        reducer,
        "PhysCal validation",
    )?;
    // C InitFile (NVMCCalMode==1) creates only the `_time_` file on rank 0; its
    // failure joins the directory-setup agreement so no extra collective is added.
    let mut run_files = None;
    let mut io_error: Option<String> = None;
    let output_setup = if reducer.is_output_root() {
        output_dir.map_or(Ok(()), |path| {
            std::fs::create_dir_all(path).map_err(|error| error.to_string())?;
            crate::output_files::RunFiles::init(
                data,
                path,
                crate::output_files::RunKind::PhysCal,
                None,
            )
            .map(|files| run_files = Some(files))
            .map_err(|error| error.to_string())
        })
    } else {
        Ok(())
    };
    collective_result(output_setup, reducer, "PhysCal output directory")?;
    if !options.initialization_consumed {
        let mut init_data = data.clone();
        collective_result(
            init_parameter(&mut init_data, rng).map_err(str::to_string),
            reducer,
            "PhysCal complex declaration/initialization",
        )?;
        observe_physcal_lifecycle("initialized-clone", &init_data, Some(rng), None);
    }
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
        // C: OutputTime(ismp); FlushFile(0) (always a flush) before sampling.
        if let Some(files) = run_files.as_mut() {
            if let Err(error) = files
                .output_time(sample as i64, &state.electron_config.counter)
                .and_then(|()| files.flush_file(0))
            {
                io_error.get_or_insert(error.to_string());
            }
        }
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
            sample_real_c_order(data, state, rng, timer, reducer.reducer_dyn()).map(|_| ())
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
                crate::io::output_phys_data(
                    data,
                    state,
                    sample,
                    Some(output_dir),
                    options.binary_output,
                )
                .err()
                .map(|error| error.to_string())
            } else {
                None
            }
        } else {
            None
        };
        let output_error = output_error.or_else(|| io_error.take());
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
    if let Some(mut files) = run_files.take() {
        files
            .output_time(iterations as i64, &state.electron_config.counter)
            .and_then(|()| files.close())
            .map_err(|error| error.to_string())?;
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
    /// C `NFileFlushInterval` (`-F`, default 1): flush the `_time_` and
    /// `_SRinfo` files when `step % interval == 0`. Values below 1 are rejected.
    pub file_flush_interval: Option<i64>,
    /// C `-b` (`FlagBinary`): write `_varbin_` files instead of `_var` text files.
    pub binary_output: bool,
}

/// Run `nsteps` SR steps starting from `data` and the seeded `rng`.
///
/// Real and complex sz-conserved models and complex AP/P FSZ models use
/// their corresponding drivers. Runtime validation rejects unported
/// features before sampling; a failed SR step returns an error before
/// parameter synchronization or any subsequent iteration.
pub fn vmc_para_opt<R: Reducer + ReducerAsDyn + ?Sized>(
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
pub fn vmc_para_opt_timed<const TIMED: bool, R: Reducer + ReducerAsDyn + ?Sized>(
    data: &mut ExpertModeData,
    state: &mut VmcOptimizationState,
    rng: &mut Sfmt19937Rng,
    output_dir: Option<&Path>,
    reducer: &R,
    mut options: OptimizationOptions<'_>,
    timer: &mut CTimer<TIMED>,
) -> Result<(), String> {
    // Pin BLAS to one thread for the deterministic serial policy. The C-order path initialises
    // this on its first Cholesky solve; the resident SR path (#452) never reaches it, so OpenBLAS
    // would otherwise keep its default pool busy-waiting for the whole run (issue #517).
    crate::serial_blas::initialize();
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
    let mut run_files: Option<crate::output_files::RunFiles> = None;
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
    // C InitFile: rank 0 creates the `_time_`/`_SRinfo` files before the loop.
    // A creation failure is reported through the first output agreement.
    if options
        .file_flush_interval
        .is_some_and(|interval| interval < 1)
    {
        return Err("FileFlushInterval should be natural number".into());
    }
    let mut io_error: Option<String> = None;
    if reducer.is_output_root() {
        if let Some(dir) = output_dir {
            match crate::output_files::RunFiles::init(
                data,
                dir,
                crate::output_files::RunKind::ParaOpt,
                options.file_flush_interval,
            ) {
                Ok(files) => run_files = Some(files),
                Err(error) => io_error = Some(error.to_string()),
            }
        }
    }
    timer.start(2);
    for step in 0..n_steps {
        // C OutputTime(step) precedes the sample; Counter still holds the
        // previous step's reduced statistics (step 0 prints the title row).
        if let Some(files) = run_files.as_mut() {
            if let Err(error) = files.output_time(step as i64, &state.electron_config.counter) {
                io_error.get_or_insert(error.to_string());
            }
        }
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
            sample_real_c_order(data, state, rng, timer, reducer.reducer_dyn())
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
            output_data(data, state, step, output_dir, options.binary_output)
                .err()
                .map(|error| error.to_string())
                .or_else(|| io_error.take())
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
        let mut direct_sr_info = None;
        let sr_result = if data.modpara.nsrcg != 0 {
            crate::sr_cg::stochastic_opt_cg_with_reducer(data, state, run_files.as_mut(), reducer)
                .map_err(|e| e.to_string())
        } else if all_complex {
            Ok(crate::sr::stochastic_opt_complex_with_sr_info_timed(
                data,
                state,
                timer,
                &mut direct_sr_info,
            ))
        } else {
            Ok(crate::sr::stochastic_opt_real_with_sr_info_timed(
                data,
                state,
                timer,
                &mut direct_sr_info,
            ))
        };
        if let (Some(files), Some(row)) = (run_files.as_mut(), direct_sr_info.as_ref()) {
            files
                .write_sr_info(row)
                .map_err(|error| error.to_string())?;
        }
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
        if step >= window_start {
            store_opt_data(data, state, step - window_start);
        }
        // C FlushFile(step) closes each VMCParaOpt iteration.
        if let Some(files) = run_files.as_mut() {
            if let Err(error) = files.flush_file(step as i64) {
                io_error.get_or_insert(error.to_string());
            }
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

    if let Some(files) = run_files.as_mut() {
        if let Err(error) = files.output_time(n_steps as i64, &state.electron_config.counter) {
            io_error.get_or_insert(error.to_string());
        }
    }
    let output_error = if reducer.is_output_root() {
        output_opt_data(data, state, output_dir)
            .err()
            .map(|error| error.to_string())
            .or_else(|| io_error.take())
    } else {
        None
    };
    if reducer.any_failure(output_error.is_some()) {
        return Err(
            output_error.unwrap_or_else(|| "final output failed on another MPI rank".into())
        );
    }
    // C CloseFile after the optimization returns.
    if let Some(files) = run_files.take() {
        files.close().map_err(|error| error.to_string())?;
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
    /// C `-F` / `NFileFlushInterval`; `None` is C's default of 1.
    pub file_flush_interval: Option<i64>,
    /// C `-b` (`FlagBinary`): write `_varbin_` files instead of `_var` text files.
    pub binary_output: bool,
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
            file_flush_interval: None,
            binary_output: false,
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
pub fn run_para_opt_from_namelist_with_reducer<R: Reducer + ReducerAsDyn + ?Sized>(
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
    .map(|(summary, _, _)| summary)
}

/// Run the production optimization entry point and also return this rank's final
/// sampler state and RNG, for rank-wise verification (#179). The numerical run
/// is identical to [`run_para_opt_from_namelist_with_reducer`].
pub fn run_para_opt_from_namelist_observed<R: Reducer + ReducerAsDyn + ?Sized>(
    namelist_path: impl AsRef<Path>,
    config: RunConfig,
    reducer: &R,
) -> Result<(RunSummary, VmcOptimizationState, Sfmt19937Rng), String> {
    collective_result(
        validate_run_options(&config),
        reducer,
        "optimization configuration",
    )?;
    run_para_opt_timed(
        namelist_path,
        config,
        reducer,
        &mut CTimer::<false>::new(),
        TimerEnv::from_env(),
    )
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
    if config.file_flush_interval.is_some_and(|value| value < 1) {
        return Err(format!(
            "file_flush_interval must be a natural number when provided; got {}",
            config.file_flush_interval.unwrap()
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

fn run_para_opt_timed<const TIMED: bool, R: Reducer + ReducerAsDyn + ?Sized>(
    namelist_path: impl AsRef<Path>,
    config: RunConfig,
    reducer: &R,
    timer: &mut CTimer<TIMED>,
    flags: TimerEnv,
) -> Result<(RunSummary, VmcOptimizationState, Sfmt19937Rng), String> {
    timer.reset();
    timer.diagnostics = flags;
    timer.start(0);
    timer.start(1);
    timer.start(11);
    let parsed = (|| {
        let data = match config.enable_opt_trans {
            Some(enabled) => {
                mvmc_expert_parsers::parse_expert_mode_files_with_opt_trans(&namelist_path, enabled)
            }
            // The production runner follows C's FlagOptTrans contract: a
            // definition file alone does not activate optimized translation.
            None => {
                mvmc_expert_parsers::parse_expert_mode_files_with_opt_trans(&namelist_path, false)
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
    vmc_para_opt_timed(
        &mut data,
        &mut state,
        &mut rng,
        Some(&output_dir),
        reducer,
        OptimizationOptions {
            file_flush_interval: config.file_flush_interval,
            binary_output: config.binary_output,
            ..OptimizationOptions::default()
        },
        timer,
    )?;
    timer.stop(0);
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
    let summary = collective_result(final_result, reducer, "optimization final output/summary")?;
    Ok((summary, state, rng))
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
    // Both runner lifecycles clear aggregates/scratch before entering this
    // scope. Samples are processed in their original order into the uniquely
    // owned rank-local buffers, just as in C VMCMainCal. A second cache would
    // duplicate the clear and copy the completed aggregates back into zeros.
    accumulate_observables_local(data, state, all_complex, use_fsz, timer, reducer);
    crate::sr_accumulator::publish_measurement_in_place(&mut state.sr_opt);
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
    // C setmemory.c:428: SROptO storage exists iff NSRCG == 1 || NStoreO != 0.
    // NSRCG >= 2 with NStore == 0 is rejected by validation (undefined in C).
    let use_store = data.modpara.nstore_o != 0 || data.modpara.nsrcg == 1;
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
    // Issue #422: stage A (Pfaffian/inverse tables) runs for a batch of samples, stage B
    // (the unchanged per-sample consumers) then runs in sample order, so every accumulation
    // keeps the C order. Batch size 1 is the former serial loop. See `measurement_batch`.
    let mut batch = std::mem::take(&mut state.measurement_batch);
    let samples: Vec<usize> = measurement_range.collect();
    // Stage A source (issue #422/#437): C-order `calc_m_all_*` per sample by default, or one
    // batched `PfaffianStages` call of the selected stage backend (real, non-FSZ only).
    let stage_kind = match crate::measurement_batch::selected_measurement_pfaffian() {
        crate::measurement_batch::MeasurementPfaffian::CalcMAll => None,
        crate::measurement_batch::MeasurementPfaffian::Stage(kind) => {
            assert!(
                !all_complex && !use_fsz,
                "{}: the stage-backend measurement supports only real, non-FSZ mode",
                crate::measurement_batch::MEASURE_PF_VARIABLE
            );
            Some(kind)
        }
    };
    // Stage B consumer (issue #519): the local energy and the Slater derivative of each real,
    // non-FSZ sample run through the selected backend's measurement stage (default: the
    // C-order oracle, byte-identical to the inline kernels). Complex and FSZ modes have no
    // stage implementation: the default C-order consumer keeps the inline kernels, while any
    // other selected backend is a hard error, never a silent CPU fallback (issue #518).
    let measure_stage = match crate::measurement_batch::selected_measurement_consumer() {
        crate::measurement_batch::MeasurementConsumer::Direct => None,
        crate::measurement_batch::MeasurementConsumer::Stage(kind) => {
            if !all_complex && !use_fsz {
                Some(kind)
            } else {
                assert!(
                    kind == crate::stage_backend::StageBackendKind::COrder,
                    "{}: the stage-backend measurement supports only real, non-FSZ mode",
                    crate::measurement_batch::MEASURE_CONSUMER_VARIABLE
                );
                None
            }
        }
    };
    let batch_size = if stage_kind == Some(crate::stage_backend::StageBackendKind::COrder)
        && (measure_stage.is_none()
            || measure_stage == Some(crate::stage_backend::StageBackendKind::COrder))
    {
        crate::measurement_batch::resolve_c_order_host_batch_size(samples.len(), n_qp_full)
    } else {
        crate::measurement_batch::resolve_batch_size(samples.len())
    };
    batch.configure(n_size, n_qp_full, batch_size, all_complex);
    let slater_offset_stage_b = 2 * (1 + n_proj + if use_fsz { 0 } else { n_rbm });
    let slater_len_stage_b = if measure_stage.is_some()
        && data.modpara.vmc_calc_mode == 0
        && n_orb_total > 0
        && slater_offset_stage_b < state.sr_opt.sr_opt_o.len()
    {
        (2 * n_orb_total).min(state.sr_opt.sr_opt_o.len() - slater_offset_stage_b)
    } else {
        0
    };
    let mut calculate_samples = || {
        for chunk in samples.chunks(batch_size) {
            if let Some(kind) = stage_kind {
                // ---- stage A through the stage backend: one batched call per chunk ------------
                let mut active = Vec::with_capacity(chunk.len());
                for (slot, &sample) in chunk.iter().enumerate() {
                    timer.start_diag(940, diag);
                    timer.start_diag(942, diag);
                    let ele_idx = state.electron_config.ele_idx_slice(sample);
                    if ele_idx.iter().all(|&v| v == 0) || ele_idx.iter().all(|&v| v < 0) {
                        batch.status[slot] = SlotStatus::Skipped;
                    } else {
                        batch.ele_idx[slot * n_size..(slot + 1) * n_size].copy_from_slice(ele_idx);
                        batch.status[slot] = SlotStatus::Ready;
                        active.push(slot);
                    }
                    timer.stop_diag(942, diag);
                    timer.stop_diag(940, diag);
                }
                timer.start(40);
                let mut handle = crate::stage_backend::acquire_kind(kind);
                crate::measurement_batch::stage_a_pfaffian(
                    handle.backend().pfaffian(),
                    &state.slater_matrix.slater_elm_real,
                    &mut batch,
                    &active,
                    n_site,
                    n_elec,
                    n_qp_full,
                )
                .unwrap_or_else(|e| {
                    panic!(
                        "{}: measurement Pfaffian stage failed: {e}",
                        crate::measurement_batch::MEASURE_PF_VARIABLE
                    )
                });
                timer.stop(40);
                drop(handle);
            } else {
                // ---- stage A: tables for every sample of the batch -------------------------------
                for (slot, &sample) in chunk.iter().enumerate() {
                    timer.start_diag(940, diag);
                    timer.start_diag(942, diag);
                    let ele_idx = state.electron_config.ele_idx_slice(sample);
                    if ele_idx.iter().all(|&v| v == 0) || ele_idx.iter().all(|&v| v < 0) {
                        batch.status[slot] = SlotStatus::Skipped;
                        timer.stop_diag(942, diag);
                        timer.stop_diag(940, diag);
                        continue;
                    }
                    batch.ele_idx[slot * n_size..(slot + 1) * n_size].copy_from_slice(ele_idx);
                    timer.stop_diag(942, diag);
                    timer.stop_diag(940, diag);
                    // The batch slot's tables become the working tables while the sample is built
                    // (pointer swap, no copy) and are swapped back afterwards.
                    swap_batch_tables(
                        state,
                        &mut batch.real,
                        &mut batch.complex,
                        slot,
                        all_complex,
                    );
                    timer.start(40);
                    // Refresh Pfaffian for the saved walker.
                    let ele_idx = &batch.ele_idx[slot * n_size..(slot + 1) * n_size];
                    let info = if use_fsz {
                        let ele_spn = state.electron_config.ele_spn_slice(sample).to_vec();
                        refresh_fsz_observation_matrix(
                            data,
                            state,
                            all_complex,
                            ele_idx,
                            &ele_spn,
                            &pool,
                        )
                        .err()
                    } else if all_complex {
                        // C `CalculateMAll` operation order (#449), as the other measurement paths.
                        crate::pfaffian::calc_m_all_complex_production(
                            ele_idx,
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
                            ele_idx,
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
                    timer.stop(40);
                    swap_batch_tables(
                        state,
                        &mut batch.real,
                        &mut batch.complex,
                        slot,
                        all_complex,
                    );
                    if info.is_some() {
                        batch.status[slot] = SlotStatus::Failed;
                        continue;
                    }
                    batch.status[slot] = SlotStatus::Ready;
                }
            }
            // ---- stage B: per-sample consumers, in sample order -------------------------------
            // One backend handle per chunk for the routed measurement stage (issue #519);
            // `None` keeps the inline C-order calls for this chunk.
            let mut measure_handle = measure_stage.map(crate::stage_backend::acquire_kind);
            for (slot, &sample) in chunk.iter().enumerate() {
                if batch.status[slot] != SlotStatus::Ready {
                    continue;
                }
                timer.start_diag(940, diag);
                timer.start_diag(942, diag);
                let ele_idx = &batch.ele_idx[slot * n_size..(slot + 1) * n_size];
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
                timer.start_diag(940, diag);
                timer.start_diag(943, diag);
                swap_batch_tables(
                    state,
                    &mut batch.real,
                    &mut batch.complex,
                    slot,
                    all_complex,
                );
                if !all_complex {
                    // The real Slater derivative reads the real inverse; refresh the complex
                    // inverse planes only for consumers that still need them.
                    // PhysCal without SR (no SlaterElmDiff) and without Lanczos reads only
                    // the real tables, so skip the full real->complex inverse copy there.
                    let physcal_without_sr =
                        data.modpara.vmc_calc_mode != 0 && data.modpara.lanczos_mode == 0;
                    if !physcal_without_sr
                        && (use_fsz
                            || data.modpara.vmc_calc_mode != 0
                            || !crate::slater_derivative::real_slater_derivative_available(
                                data, n_qp_full,
                            ))
                    {
                        for qp in 0..n_qp_full {
                            let real_plane = state.slater_matrix.inv_m_real.qp_matrix_slice(qp);
                            // Copy only the matrix, leaving the per-QP inverse pad untouched.
                            crate::threading::copy_real_to_complex(
                                &mut state.slater_matrix.inv_m.qp_matrix_slice_mut(qp)
                                    [..n_size * n_size],
                                &real_plane[..n_size * n_size],
                            );
                        }
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
                    crate::observables::calculate_ip_complex(
                        &state.slater_matrix.pf_m,
                        0,
                        n_qp_full,
                        data,
                    )
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
                let mut slater_diff_buf: Option<Vec<Complex64>> = None;
                let e = if use_fsz {
                    crate::observables::calculate_local_energy_fsz_timed(
                        ip,
                        data,
                        state,
                        ele_idx,
                        &ele_cfg,
                        &ele_num,
                        &ele_proj_cnt,
                        &ele_spn,
                        timer,
                    )
                } else if let Some(handle) = measure_handle.as_mut() {
                    // Routed through the selected backend's measurement stage (issue #519).
                    // The Slater derivative is computed here with the local energy and its
                    // block is stored below, at the same point in the sample order as the
                    // inline call. Byte-identical for the C-order stage.
                    let mut input = crate::stage_backend::MeasurementInput {
                        data,
                        state,
                        ip,
                        ele_idx,
                        ele_cfg: &ele_cfg,
                        ele_num: &ele_num,
                        ele_proj_cnt: &ele_proj_cnt,
                    };
                    let out = handle
                        .backend()
                        .measurement()
                        .measure_sample(&mut input, slater_len_stage_b, timer)
                        .unwrap_or_else(|error| {
                            panic!(
                                "{}: measurement stage failed: {error}",
                                crate::measurement_batch::MEASURE_CONSUMER_VARIABLE
                            )
                        });
                    if !out.slater_diff.is_empty() {
                        slater_diff_buf = Some(out.slater_diff);
                    }
                    out.local_energy
                } else {
                    crate::observables::calculate_local_energy_timed(
                        ip,
                        data,
                        state,
                        ele_idx,
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

                // The Lanczos path evaluates H on each moved configuration.
                // General-orbital/FSZ configurations remain gated.
                if data.modpara.lanczos_mode > 0 && !use_fsz {
                    let h2 = crate::observables::calculate_lanczos_h2_transfer(
                        e,
                        ip,
                        data,
                        state,
                        ele_idx,
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
                        ele_idx,
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
                        ele_idx,
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
                            ele_idx,
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
                        for (dst, src) in phys.phys_lanczos_qcisajsq.iter_mut().zip(values.one_body)
                        {
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
                // C `VMCMainCal` builds the SR `O` vector, `SlaterElmDiff` and the
                // OO/HO accumulators only for NVMCCalMode==0 (`vmccal.c:204`). PhysCal
                // (NVMCCalMode=1) computes the Green functions only, so skip this
                // wasted per-sample work.
                if data.modpara.vmc_calc_mode == 0 {
                    timer.start_diag(940, diag);
                    timer.start_diag(948, diag);
                    // SR `O` vector — projection diff fills the leading block.
                    for slot in state.sr_opt.sr_opt_o.iter_mut() {
                        *slot = Complex64::new(0.0, 0.0);
                    }
                    crate::observables::set_projection_diff(
                        &mut state.sr_opt.sr_opt_o,
                        &ele_proj_cnt,
                        n_proj,
                    );
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
                        let n_copy =
                            (2 * n_orb_total).min(state.sr_opt.sr_opt_o.len() - slater_offset);
                        let slater_o =
                            &mut state.sr_opt.sr_opt_o[slater_offset..slater_offset + n_copy];
                        if let Some(buffer) = slater_diff_buf.take() {
                            // Computed by the routed measurement stage with the local energy.
                            slater_o.copy_from_slice(&buffer);
                        } else if use_fsz {
                            crate::slater_derivative::slater_elm_diff_fsz_with_scratch(
                                slater_o,
                                ip,
                                ele_idx,
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
                                ele_idx,
                                data,
                                &state.slater_matrix,
                                !all_complex,
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
                }
            }
        }
    };
    let host_pfaffian =
        stage_kind.is_none() || stage_kind == Some(crate::stage_backend::StageBackendKind::COrder);
    let host_consumer = measure_stage.is_none()
        || measure_stage == Some(crate::stage_backend::StageBackendKind::COrder);
    if !all_complex && !use_fsz && host_pfaffian && host_consumer {
        // Batched stage A owns sample/QP planes, not just a single sample's
        // QP width. Retain its workers for the following host consumers too.
        // Device/tenferro consumers keep their own execution lifecycle.
        let work_width = if stage_kind.is_some() {
            n_qp_full.saturating_mul(batch_size)
        } else {
            n_qp_full
        };
        crate::threading::with_qp_region_team(
            work_width,
            crate::pfaffian::pfaffian_qp_cost_ns(n_size),
            calculate_samples,
        );
    } else {
        calculate_samples();
    }
    state.measurement_batch = batch;
    observe_physcal_green(data, state, use_fsz);
    normalize_physcal_green(state, use_fsz, all_complex);
    // C finalizes the stored OO only for NVMCCalMode==0 (`vmccal.c:318`).
    if use_store && data.modpara.vmc_calc_mode == 0 {
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

/// Exchange batch slot `slot` with the working inverse/Pfaffian tables of the active mode.
fn swap_batch_tables(
    state: &mut VmcOptimizationState,
    real: &mut [crate::measurement_batch::TableSet<f64>],
    complex: &mut [crate::measurement_batch::TableSet<Complex64>],
    slot: usize,
    all_complex: bool,
) {
    if all_complex {
        complex[slot].swap_with(
            &mut state.slater_matrix.inv_m,
            &mut state.slater_matrix.pf_m,
        );
    } else {
        real[slot].swap_with(
            &mut state.slater_matrix.inv_m_real,
            &mut state.slater_matrix.pf_m_real,
        );
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
mod callback_tests {
    mod historical_overlay_stage {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/support/historical_overlay_stage.rs"
        ));
    }
    use super::*;

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
                file_flush_interval: None,
                binary_output: false,
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
                file_flush_interval: None,
                binary_output: false,
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
                    file_flush_interval: None,
                    binary_output: false,
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
                file_flush_interval: None,
                binary_output: false,
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
                file_flush_interval: None,
                binary_output: false,
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
                skip_sr: false,
                file_flush_interval: None,
                binary_output: false,
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
                    file_flush_interval: None,
                    binary_output: false,
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
                // Serial real Transfer runs Julia's CalH1 fast path (#207): prep, term loop,
                // combine, direct projection ratio and the fused PfM2+IP kernel.
                for id in [32, 60, 61, 62, 63, 925, 926, 929, 935, 936] {
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
    fn normal_interall_equivalents_preserve_sr_parameters_samples_energy_and_rng() {
        for complex in [false, true] {
            for (cg, store) in [(0, 0), (0, 1), (1, 0)] {
                let mut runs = Vec::new();
                for interall in [false, true] {
                    let (mut data, _, mut rng) = prepared_case(3, "hubbard_chain_real");
                    data.modpara.nsrcg = cg;
                    data.modpara.nsrcg_abort_on_fail = false; // short CG solves may hit their limit (NSRCGAbortOnFail 0)
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
                            file_flush_interval: None,
                            binary_output: false,
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

    /// Real-mode OptTrans derivatives occupy their own slots (#370).
    ///
    /// Each sample has `ipAll = sum_i w_i * ip_i` and derivative `O_i = ip_i / ipAll`,
    /// so `sum_i w_i * O_i = 1` for every sample and for the weighted mean. Native C
    /// writes these through a complex pointer offset in real mode (`vmccal.c`
    /// `calculateOptTransDiff`), which drops derivative 1, stores derivative 2 in
    /// slot 1 and leaves the last slot zero; that layout violates this identity.
    /// Rust keeps the mathematically correct layout (tmisawa/Julia-mVMC#55).
    #[test]
    fn real_mode_opttrans_derivatives_use_their_own_slots() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/opttrans/run_opt_real/namelist.def");
        let (mut data, mut state, mut rng) = prepared_namelist(1, &path);
        data.modpara.nsrcg = 1;
        data.modpara.nsrcg_abort_on_fail = false; // short CG solves may hit their limit (NSRCGAbortOnFail 0)
        data.modpara.nstore_o = 0;
        assert!(!get_all_complex_flag(&data).unwrap());
        let weights: Vec<f64> = data.opt_trans.iter().map(|w| w.re).collect();
        assert_eq!(weights.len(), 3);
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
        fs::remove_dir_all(dir).unwrap();
        let size = state.sr_opt.sr_opt_size;
        let first = size - weights.len();
        let derivatives = &state.sr_opt.sr_opt_o_real[first..size];
        assert!(derivatives.iter().all(|d| *d != 0.0), "{derivatives:?}");
        let sum: f64 = weights.iter().zip(derivatives).map(|(w, d)| w * d).sum();
        assert!((sum - 1.0).abs() < 1e-12, "sum w_i O_i = {sum}");
    }

    #[test]
    fn canonical_cg_fixed_seed_same_configuration_is_repeatable() {
        let run = || {
            let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../extern/Julia-mVMC/test/integration/reference/general_rbm_cmp/inputs/namelist.def");
            let mut data = parse_expert_mode_files(&path).unwrap();
            data.modpara.nsr_opt_itr_step = 3;
            data.modpara.nsr_opt_itr_smp = 3;
            data.modpara.nsrcg = 1;
            data.modpara.nsrcg_abort_on_fail = false; // short CG solves may hit their limit (NSRCGAbortOnFail 0)
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
                baseline.modpara.nsrcg_abort_on_fail = false; // short CG solves may hit their limit (NSRCGAbortOnFail 0)
                baseline.modpara.nstore_o = store;
                let (mut data, mut state, mut rng) = prepared_case(3, case);
                data.modpara.nsrcg = cg;
                data.modpara.nsrcg_abort_on_fail = false; // short CG solves may hit their limit (NSRCGAbortOnFail 0)
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
                        .filter(|name| !name.to_string_lossy().contains("_time_"))
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
            data.modpara.nsrcg_abort_on_fail = false; // short CG solves may hit their limit (NSRCGAbortOnFail 0)
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
            assert!(info.starts_with("#Npara") && info.lines().next().unwrap().ends_with(" info"));
            for row in info.lines().skip(1) {
                // C 1.4.0 CG rows end with " info" ("%5d %d") and print the
                // global NPara (not 2*NPara).
                let fields: Vec<&str> = row.split_whitespace().collect();
                assert_eq!(fields.len(), 9, "{case}: {row}");
                assert!(!row.contains(','), "{case}: {row}");
                assert_eq!(
                    fields[0].parse::<usize>().unwrap(),
                    data.count_variational_parameters(),
                    "{case}: {row}"
                );
            }
            let time = fs::read_to_string(
                dir.join(format!("zvo_time_{:03}.dat", data.modpara.n_data_idx_start)),
            )
            .unwrap();
            assert_eq!(time.lines().count(), 4, "{case}\n{time}");
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
                    file_flush_interval: None,
                    binary_output: false,
                    ..OptimizationOptions::default()
                },
            )
            .unwrap();
            // Exercise the CG accumulator with the already sampled walkers.
            data.modpara.nstore_o = 0;
            data.modpara.nsrcg = 1;
            data.modpara.nsrcg_abort_on_fail = false; // short CG solves may hit their limit (NSRCGAbortOnFail 0)
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
                    file_flush_interval: None,
                    binary_output: false,
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
    fn nsrcg_three_with_store_follows_nsrcg_one_control_path() {
        // C treats every nonzero NSRCG as CG (vmcmain.c:463,487; vmccal.c:591,602)
        // and allocates SROptO storage for NSRCG==1 || NStoreO!=0 (setmemory.c:428),
        // so NSRCG=3,NStore=1 and NSRCG=1,NStore=1 execute identical code in C
        // (1.4.0 maps only NSRCG=2 to DiagScale CG, readdef.c:927-930).
        // This is a same-implementation equivalence, not a C numerical reference.
        for case in [
            "heisenberg_chain_real",
            "heisenberg_chain_cmp",
            "hubbard_chain_real",
        ] {
            let mut runs = Vec::new();
            for cg in [1, 3] {
                let (mut data, mut state, mut rng) = prepared_case(3, case);
                data.modpara.nsrcg = cg;
                data.modpara.nsrcg_abort_on_fail = false; // short CG solves may hit their limit (NSRCGAbortOnFail 0)
                data.modpara.nstore_o = 1;
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
                runs.push((data, state, rng, dir));
            }
            let (a, b) = runs.split_at_mut(1);
            let (da, sa, ra, dira) = &mut a[0];
            let (db, sb, rb, dirb) = &mut b[0];
            assert_eq!(da.slater_params, db.slater_params, "{case}");
            assert_eq!(da.optimization_flags, db.optimization_flags, "{case}");
            assert_eq!(sa.energy, sb.energy, "{case}");
            assert_eq!(sa.sr_opt, sb.sr_opt, "{case}");
            assert_eq!(sa.electron_config, sb.electron_config, "{case}");
            assert_eq!(ra.words_consumed(), rb.words_consumed(), "{case}");
            for _ in 0..624 {
                assert_eq!(ra.gen_rand32(), rb.gen_rand32(), "{case}");
            }
            let names = |d: &Path| {
                let mut v: Vec<_> = fs::read_dir(d)
                    .unwrap()
                    .map(|e| e.unwrap().file_name())
                    .collect();
                v.sort();
                v
            };
            let files = names(dira);
            assert_eq!(files, names(dirb), "{case}");
            // `_time_` rows end in a wall-clock ctime string (": Mon Oct  5 14:18:39 2026"),
            // which differs between two runs that straddle a second boundary; compare the rest.
            let strip_ctime = |bytes: Vec<u8>, name: &std::ffi::OsStr| -> Vec<u8> {
                if !name.to_string_lossy().contains("_time_") {
                    return bytes;
                }
                String::from_utf8(bytes)
                    .unwrap()
                    .lines()
                    .map(|line| line.rsplit_once(": ").map_or(line, |(body, _)| body))
                    .collect::<Vec<_>>()
                    .join("\n")
                    .into_bytes()
            };
            for f in files {
                // Wall-clock timer files differ between any two runs.
                if f.to_string_lossy().contains("_time_") {
                    continue;
                }
                assert_eq!(
                    strip_ctime(fs::read(dira.join(&f)).unwrap(), &f),
                    strip_ctime(fs::read(dirb.join(&f)).unwrap(), &f),
                    "{case}: {f:?}"
                );
            }
            fs::remove_dir_all(dira).unwrap();
            fs::remove_dir_all(dirb).unwrap();
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
                    file_flush_interval: None,
                    binary_output: false,
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
                    file_flush_interval: None,
                    binary_output: false,
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
                .filter(|entry| {
                    !entry
                        .as_ref()
                        .unwrap()
                        .file_name()
                        .to_string_lossy()
                        .contains("_time_")
                })
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
        let output_dir = fresh_output_directory().unwrap();
        let capture = Rc::new(Capture::default());
        let _guard = install_physcal_green_observer(capture.clone()).unwrap();
        with_physcal_green_sample(7, || {
            vmc_para_opt(
                &mut data,
                &mut state,
                &mut rng,
                Some(&output_dir),
                &SingleProcessReducer,
                OptimizationOptions::default(),
            )
        })
        .unwrap();
        assert!(capture.0.borrow().is_empty());
        fs::remove_dir_all(output_dir).unwrap();
    }
}

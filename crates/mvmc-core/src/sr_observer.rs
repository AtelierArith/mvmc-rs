//! Explicit scoped observation of actual direct SR solves, never an oracle.
//! Disabled hooks copy/allocate nothing. Capture on each MPI rank's caller thread.
use mvmc_expert_parsers::ExpertModeData;
use std::cell::{Cell, RefCell};
use std::marker::PhantomData;
use std::rc::Rc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Storage path used by the original direct solver.
pub enum DirectMode {
    /// Dedicated real SR moment buffers.
    Real,
    /// Complex SR moment buffers, represented as real/imaginary components.
    Complex,
}

/// Original early-return branch, not an invented solve of an empty matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotSolvedReason {
    /// No declared variational parameters.
    NoParameters,
    /// All components were fixed or removed by the original diagonal cut.
    NoActiveComponents,
}

#[derive(Debug, Clone, PartialEq)]
/// Original input/regularization settings at the observation point.
pub struct DirectSolveSettings {
    /// Input seed; the harness must separately record MPI rank/seed overrides.
    pub input_seed: i64,
    /// Effective optimization iteration count.
    pub steps: i64,
    /// Requested optimization averaging window.
    pub window: i64,
    /// Diagonal stabilization multiplier minus one.
    pub diagonal_shift: f64,
    /// Redundant-direction cutoff relative to the maximum diagonal.
    pub redundant_cut: f64,
    /// Gradient step factor.
    pub step_dt: f64,
    /// Input direct/CG selector.
    pub nsrcg: i64,
    /// Input derivative-storage selector.
    pub nstore: i64,
    /// Whether C's consecutive OptTrans flag layout is active.
    pub c_opt_trans_flags: bool,
}

#[derive(Debug, Clone, PartialEq)]
/// Owned pre-factorization system and actual original solve result.
pub struct DirectSolveObservation {
    /// Real or complex moment-storage path.
    pub mode: DirectMode,
    /// Active square matrix dimension.
    pub dimension: usize,
    /// Authoritative original LAPACK triangle, always 'U'.
    pub triangle: char,
    /// Explicit early-return reason; matrix/RHS are absent in that case.
    pub not_solved: Option<NotSolvedReason>,
    /// Original regularized matrix, column-major. The solve uses UPLO=U.
    pub matrix: Vec<f64>,
    /// Original energy gradient, before it is overwritten by substitution.
    pub rhs: Vec<f64>,
    /// Zero-based C real/imaginary component indices (real parameters are doubled).
    pub active_indices: Vec<usize>,
    /// Complete original component flags, including unused declared slots.
    pub flags: Vec<i64>,
    /// Original effective input settings.
    pub settings: DirectSolveSettings,
    /// Actual overwritten RHS before parameter updates, including failure values.
    pub increment: Vec<f64>,
    /// Original direct-SR solve result, 0 or 1; None only if interrupted by panic.
    pub status: Option<i32>,
    /// Actual LAPACK statuses, not reinterpreted by the observer.
    pub factor_info: Option<i32>,
    /// Original substitution INFO, absent if substitution was not reached.
    pub solve_info: Option<i32>,
}

/// Actual post-average, pre-output/pre-SR buffers, not reconstructed operands.
#[derive(Debug, Clone, PartialEq)]
pub struct NormalizedObservation {
    /// Zero-based optimization step.
    pub step: usize,
    /// Active moment-buffer branch.
    pub mode: DirectMode,
    /// Actual weight and normalized energy moments.
    pub energy: Vec<num_complex::Complex64>,
    /// Active complex moment buffers; empty in real mode.
    pub oo: Vec<num_complex::Complex64>,
    /// Active complex gradient buffers; empty in real mode.
    pub ho: Vec<num_complex::Complex64>,
    /// Active real moment buffers; empty in complex mode.
    pub oo_real: Vec<f64>,
    /// Active real gradient buffers; empty in complex mode.
    pub ho_real: Vec<f64>,
}

thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(false) };
    static RECORDS: RefCell<Option<Vec<DirectSolveObservation>>> = const { RefCell::new(None) };
    static NORMALIZED: RefCell<Option<Vec<NormalizedObservation>>> = const { RefCell::new(None) };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Nested captures on the same thread are rejected without disturbing the first.
pub struct CaptureAlreadyActive;
impl std::fmt::Display for CaptureAlreadyActive {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("direct SR capture already active on this thread")
    }
}
impl std::error::Error for CaptureAlreadyActive {}

/// Thread-bound RAII scope; dropping it clears capture even during unwinding.
#[derive(Debug)]
pub struct CaptureGuard {
    active: bool,
    _thread_bound: PhantomData<Rc<()>>,
}

/// Start observation on the current thread; MPI callers start it on every rank.
pub fn capture() -> Result<CaptureGuard, CaptureAlreadyActive> {
    if ENABLED.with(Cell::get) {
        return Err(CaptureAlreadyActive);
    }
    RECORDS.with(|records| *records.borrow_mut() = Some(Vec::new()));
    NORMALIZED.with(|records| *records.borrow_mut() = None);
    ENABLED.with(|enabled| enabled.set(true));
    Ok(CaptureGuard {
        active: true,
        _thread_bound: PhantomData,
    })
}

/// Opt into normalized-boundary copies separately from legacy solve capture.
pub fn capture_with_normalized() -> Result<CaptureGuard, CaptureAlreadyActive> {
    let guard = capture()?;
    NORMALIZED.with(|records| *records.borrow_mut() = Some(Vec::new()));
    Ok(guard)
}

impl CaptureGuard {
    /// Retrieve normalized boundary records without ending solve observation.
    pub fn take_normalized(&mut self) -> Vec<NormalizedObservation> {
        NORMALIZED.with(|records| {
            records
                .borrow_mut()
                .as_mut()
                .map(std::mem::take)
                .unwrap_or_default()
        })
    }

    /// Disable observation and return the owned records in actual solve order.
    pub fn finish(mut self) -> Vec<DirectSolveObservation> {
        ENABLED.with(|enabled| enabled.set(false));
        self.active = false;
        NORMALIZED.with(|records| *records.borrow_mut() = None);
        RECORDS.with(|records| records.borrow_mut().take().unwrap_or_default())
    }
}
impl Drop for CaptureGuard {
    fn drop(&mut self) {
        if self.active {
            ENABLED.with(|enabled| enabled.set(false));
            RECORDS.with(|records| *records.borrow_mut() = None);
            NORMALIZED.with(|records| *records.borrow_mut() = None);
        }
    }
}

/// Whether a capture is active on this thread (the resident SR step downloads `S` and `g` for
/// the observer only in that case).
#[inline]
pub(super) fn is_enabled() -> bool {
    ENABLED.with(Cell::get)
}

/// Internal observation route; disabled capture does not inspect buffers.
#[inline]
pub(crate) fn normalized(
    step: usize,
    state: &crate::state::VmcOptimizationState,
    all_complex: bool,
) {
    if !ENABLED.with(Cell::get) {
        return;
    }
    NORMALIZED.with(|records| {
        let mut records = records.borrow_mut();
        let Some(records) = records.as_mut() else {
            return;
        };
        records.push(NormalizedObservation {
            step,
            mode: if all_complex {
                DirectMode::Complex
            } else {
                DirectMode::Real
            },
            energy: vec![
                state.energy.wc,
                state.energy.etot,
                state.energy.etot2,
                state.energy.sztot,
                state.energy.sztot2,
            ],
            oo: if all_complex {
                state.sr_opt.sr_opt_oo.clone()
            } else {
                Vec::new()
            },
            ho: if all_complex {
                state.sr_opt.sr_opt_ho.clone()
            } else {
                Vec::new()
            },
            // A deferred Gram (issue #452) was never materialized on the host.
            oo_real: if all_complex || state.sr_oo_deferred() {
                Vec::new()
            } else {
                state.sr_opt.sr_opt_oo_real.clone()
            },
            ho_real: if all_complex {
                Vec::new()
            } else {
                state.sr_opt.sr_opt_ho_real.clone()
            },
        });
    });
}

#[inline]
pub(super) fn before_solve(
    data: &ExpertModeData,
    matrix: &[f64],
    rhs: &[f64],
    mapping: &[usize],
    mode: DirectMode,
) -> Option<usize> {
    if !ENABLED.with(Cell::get) {
        return None;
    }
    RECORDS.with(|records| {
        let mut records = records.borrow_mut();
        let records = records.as_mut()?;
        let index = records.len();
        records.push(DirectSolveObservation {
            mode,
            dimension: rhs.len(),
            triangle: 'U',
            not_solved: None,
            matrix: matrix.to_vec(),
            rhs: rhs.to_vec(),
            active_indices: mapping
                .iter()
                .map(|&index| {
                    if mode == DirectMode::Real {
                        2 * index
                    } else {
                        index
                    }
                })
                .collect(),
            flags: data.optimization_flags.clone(),
            settings: DirectSolveSettings {
                input_seed: data.modpara.rnd_seed,
                steps: data.modpara.nsr_opt_itr_step,
                window: data.modpara.nsr_opt_itr_smp,
                diagonal_shift: data.modpara.dsr_opt_sta_del,
                redundant_cut: data.modpara.dsr_opt_red_cut,
                step_dt: data.modpara.dsr_opt_step_dt,
                nsrcg: data.modpara.nsrcg,
                nstore: data.modpara.nstore_o,
                c_opt_trans_flags: data.c_opt_trans_flags,
            },
            increment: Vec::new(),
            status: None,
            factor_info: None,
            solve_info: None,
        });
        Some(index)
    })
}

#[inline]
pub(super) fn not_solved(data: &ExpertModeData, mode: DirectMode, reason: NotSolvedReason) {
    let Some(index) = before_solve(data, &[], &[], &[], mode) else {
        return;
    };
    RECORDS.with(|records| {
        let mut records = records.borrow_mut();
        if let Some(record) = records.as_mut().and_then(|records| records.get_mut(index)) {
            record.not_solved = Some(reason);
            record.status = Some(0);
        }
    });
}

#[inline]
pub(super) fn lapack_status(factorization: bool, info: i32) {
    if !ENABLED.with(Cell::get) {
        return;
    }
    RECORDS.with(|records| {
        let mut records = records.borrow_mut();
        if let Some(record) = records.as_mut().and_then(|records| records.last_mut()) {
            if record.status.is_none() {
                if factorization {
                    record.factor_info = Some(info);
                } else {
                    record.solve_info = Some(info);
                }
            }
        }
    });
}

#[inline]
pub(super) fn after_solve(index: Option<usize>, increment: &[f64], result: &Result<(), ()>) {
    let Some(index) = index else { return };
    RECORDS.with(|records| {
        let mut records = records.borrow_mut();
        if let Some(record) = records.as_mut().and_then(|records| records.get_mut(index)) {
            record.increment = increment.to_vec();
            record.status = Some(i32::from(result.is_err()));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn normalized_frame() -> crate::state::VmcOptimizationState {
        crate::state::VmcOptimizationState::zeros(2, 1, 0, 1, 1, 1, false, false)
    }

    #[test]
    fn legacy_solve_capture_does_not_enable_normalized_copies() {
        let mut legacy = capture().unwrap();
        normalized(0, &normalized_frame(), false);
        assert!(NORMALIZED.with(|records| records.borrow().is_none()));
        assert!(legacy.take_normalized().is_empty());
        assert!(capture_with_normalized().is_err());
        assert!(NORMALIZED.with(|records| records.borrow().is_none()));
        legacy.finish();
    }

    #[test]
    fn normalized_capture_is_thread_local_and_owned_after_state_changes() {
        let mut state = normalized_frame();
        state.energy.etot.re = 2.0;
        let mut guard = capture_with_normalized().unwrap();
        normalized(0, &state, false);
        std::thread::spawn(|| {
            assert!(!ENABLED.with(Cell::get));
            normalized(99, &normalized_frame(), false);
            assert!(NORMALIZED.with(|records| records.borrow().is_none()));
            let mut independent = capture_with_normalized().unwrap();
            normalized(1, &normalized_frame(), false);
            assert_eq!(independent.take_normalized().len(), 1);
            independent.finish();
        })
        .join()
        .unwrap();
        state.energy.etot.re = 7.0;
        let records = guard.take_normalized();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].step, 0);
        assert_eq!(records[0].energy[1].re, 2.0);
        guard.finish();
    }

    #[test]
    fn normalized_disabled_nested_finish_drop_and_unwind() {
        let state = normalized_frame();
        normalized(0, &state, false);
        assert!(NORMALIZED.with(|records| records.borrow().is_none()));
        let mut guard = capture_with_normalized().unwrap();
        normalized(0, &state, false);
        assert!(capture_with_normalized().is_err());
        let records = guard.take_normalized();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].step, 0);
        assert_eq!(records[0].mode, DirectMode::Real);
        assert_eq!(records[0].oo_real, state.sr_opt.sr_opt_oo_real);
        assert!(records[0].oo.is_empty());
        assert!(guard.take_normalized().is_empty());
        assert!(guard.finish().is_empty());
        assert!(NORMALIZED.with(|records| records.borrow().is_none()));
        {
            let _guard = capture_with_normalized().unwrap();
            normalized(1, &state, false);
        }
        assert!(NORMALIZED.with(|records| records.borrow().is_none()));
        assert!(std::panic::catch_unwind(|| {
            let _guard = capture_with_normalized().unwrap();
            let state = normalized_frame();
            normalized(2, &state, false);
            panic!("test normalized observation unwind");
        })
        .is_err());
        assert!(!ENABLED.with(Cell::get));
        assert!(NORMALIZED.with(|records| records.borrow().is_none()));
        let mut fresh = capture_with_normalized().unwrap();
        assert!(fresh.take_normalized().is_empty());
        fresh.finish();
    }

    #[test]
    fn normalized_copies_active_branch_without_rng_or_configuration_changes() {
        let mut state = normalized_frame();
        state.energy.etot = num_complex::Complex64::new(-0.0, 3.0);
        let mut rng = sfmt19937::Sfmt19937Rng::new(1);
        rng.gen_rand32();
        let raw = rng.state_snapshot();
        let count = rng.words_consumed();
        let config = (
            &state.electron_config.ele_idx,
            &state.electron_config.ele_cfg,
            &state.electron_config.ele_num,
            &state.electron_config.ele_proj_cnt,
            &state.electron_config.ele_spn,
            &state.electron_config.counter,
        );
        let expected = (
            config.0.clone(),
            config.1.clone(),
            config.2.clone(),
            config.3.clone(),
            config.4.clone(),
            *config.5,
        );
        let mut guard = capture_with_normalized().unwrap();
        normalized(0, &state, false);
        let records = guard.take_normalized();
        assert_eq!(records[0].energy.len(), 5);
        assert_eq!(records[0].energy[1].re.to_bits(), (-0.0_f64).to_bits());
        assert_eq!(records[0].ho_real, state.sr_opt.sr_opt_ho_real);
        assert_eq!(rng.state_snapshot(), raw);
        assert_eq!(rng.words_consumed(), count);
        assert_eq!(*config.0, expected.0);
        assert_eq!(*config.1, expected.1);
        assert_eq!(*config.2, expected.2);
        assert_eq!(*config.3, expected.3);
        assert_eq!(*config.4, expected.4);
        assert_eq!(*config.5, expected.5);
        guard.finish();
        let complex = crate::state::VmcOptimizationState::zeros(2, 1, 0, 1, 1, 1, true, false);
        let mut guard = capture_with_normalized().unwrap();
        normalized(1, &complex, true);
        let records = guard.take_normalized();
        assert_eq!(records[0].mode, DirectMode::Complex);
        assert_eq!(records[0].oo, complex.sr_opt.sr_opt_oo);
        assert!(records[0].oo_real.is_empty());
        guard.finish();
    }

    #[test]
    fn disabled_hooks_do_not_access_or_copy_payload_or_initialize_records() {
        assert!(!ENABLED.with(Cell::get));
        let data = ExpertModeData::new();
        // Enabled real-index conversion would overflow; disabled hooks must
        // return before inspecting indices or creating any snapshot payload.
        assert_eq!(
            before_solve(&data, &[], &[], &[usize::MAX], DirectMode::Real),
            None
        );
        lapack_status(true, 0);
        after_solve(None, &[f64::NAN], &Err(()));
        assert!(RECORDS.with(|records| records.borrow().is_none()));
    }
}

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

thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(false) };
    static RECORDS: RefCell<Option<Vec<DirectSolveObservation>>> = const { RefCell::new(None) };
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
    ENABLED.with(|enabled| enabled.set(true));
    Ok(CaptureGuard {
        active: true,
        _thread_bound: PhantomData,
    })
}

impl CaptureGuard {
    /// Disable observation and return the owned records in actual solve order.
    pub fn finish(mut self) -> Vec<DirectSolveObservation> {
        ENABLED.with(|enabled| enabled.set(false));
        self.active = false;
        RECORDS.with(|records| records.borrow_mut().take().unwrap_or_default())
    }
}
impl Drop for CaptureGuard {
    fn drop(&mut self) {
        if self.active {
            ENABLED.with(|enabled| enabled.set(false));
            RECORDS.with(|records| *records.borrow_mut() = None);
        }
    }
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

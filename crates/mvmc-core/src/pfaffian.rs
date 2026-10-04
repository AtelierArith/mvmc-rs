//! Pfaffian + inverse-matrix construction from Slater elements.
//!
//! Port target: `MVMCOptimizers.jl/src/calculate_m_all.jl` (~930 LOC,
//! `fcmp` / `real` / `fsz` variants). Re-uses `pfapack::{ltl, utu2}`.
//!
//! The QP loop defaults to the sequential path and can use the explicit
//! `MVMC_RS_INNER_THREADS`/`MVMC_RS_INNER_THRESHOLD` controls for independent
//! chunks. Each parallel chunk owns its temporary inverse and Pfaffian output;
//! chunks are copied back in QP order so the sampling/RNG contract is unchanged.
//!
//! The kernels are direct ports of `calculate_m_all_child_{fcmp,real}!`
//! and reproduce the upstream pipeline exactly:
//!
//! 1. Build `inv_m[msj, msi] = -slater_elm[ rsi*n_site2 + rsj ]`
//!    (column-major store; the explicit transpose mirrors the
//!    "store column-major" comment in `calculate_m_all.jl`).
//! 2. Bail with [`CalcMAllError::AllZero`] if the assembled matrix is
//!    numerically zero (matches upstream's `max_abs2 < 1e-28` guard).
//! 3. LTL-decompose via [`pfapack::dsktf2`] / [`pfapack::zsktf2`].
//! 4. Pull the Pfaffian out with [`pfapack::utu2pfa_real`] /
//!    [`pfapack::utu2pfa_complex`].
//! 5. Solve the inverse with [`pfapack::utu2inv_real`] /
//!    [`pfapack::utu2inv_complex`].
//! 6. Apply the `rmul!(inv_m, -1.0)` sign flip (`M_DSCAL` / `M_ZSCAL`
//!    in the C reference).
//!
//! See `matrix.c:285-387` for the C reference the upstream Julia port
//! traces back to.

#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

use num_complex::Complex64;
use pfapack::{
    dsktf2, utu2inv_complex, utu2inv_complex_fsz, utu2inv_real, utu2pfa_complex, utu2pfa_real,
    zsktf2, zsktf2_c_compat, zsktf2_turbo, SqMat,
};
use rayon::prelude::*;

use crate::state::{
    InvMColMajor, PfaPackMode, PfaPackWorkspace, SlaterElmFlat, SlaterMatrixData,
    ThreadedPfaPackWorkspace,
};

/// Lower bound on the squared Frobenius norm before we treat the
/// reconstructed `inv_m` as numerically zero. Matches the upstream
/// `max_abs2 < 1e-14^2` guard in `calculate_m_all_child_fcmp!`.
const MIN_ABS2: f64 = 1.0e-28;

/// Per-QP outcome surfaced by [`calc_m_all_real`] / [`calc_m_all_complex`].
///
/// The Julia kernels return a single `Int` info code; we keep the same
/// information but split it into a typed enum so the caller does not
/// have to memorise the numeric meaning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CalcMAllError {
    /// Checked native-normal input/storage precondition, not numeric INFO.
    InputShape {
        /// Invalid shape or range description.
        reason: &'static str,
    },
    /// `ele_idx` produced an `rsi` (or `rsj`) outside `[0, 2*n_site)`.
    SiteOutOfRange {
        /// QP plane that hit the error.
        qp: usize,
        /// Offending `rsi` / `rsj` site index.
        site: i64,
    },
    /// Reconstructed `inv_m` was numerically zero before LTL
    /// (matches upstream `return 2`).
    AllZero {
        /// QP plane that hit the error.
        qp: usize,
    },
    /// LTL decomposition encountered a zero pivot (`info > 0`).
    /// The wrapped value is the 1-based pivot row reported by
    /// [`pfapack::dsktf2`] / [`pfapack::zsktf2`].
    ZeroPivot {
        /// QP plane that hit the error.
        qp: usize,
        /// 1-based pivot row.
        info: usize,
    },
    /// The computed Pfaffian was `NaN` or `±∞`.
    NonFinitePfaffian {
        /// QP plane that hit the error.
        qp: usize,
    },
}

impl core::fmt::Display for CalcMAllError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CalcMAllError::InputShape { reason } => {
                write!(f, "calc_m_all: invalid native-normal shape: {reason}")
            }
            CalcMAllError::SiteOutOfRange { qp, site } => {
                write!(
                    f,
                    "calc_m_all: rsi/rsj out of range at qp={qp}, site={site}"
                )
            }
            CalcMAllError::AllZero { qp } => {
                write!(f, "calc_m_all: assembled inv_m all zero at qp={qp}")
            }
            CalcMAllError::ZeroPivot { qp, info } => write!(
                f,
                "calc_m_all: LTL zero pivot at qp={qp}, info={info} (1-based)"
            ),
            CalcMAllError::NonFinitePfaffian { qp } => {
                write!(f, "calc_m_all: non-finite Pfaffian at qp={qp}")
            }
        }
    }
}

impl std::error::Error for CalcMAllError {}

/// Real `calculate_m_all` over the half-open QP range `[qp_start, qp_end)`.
///
/// `ele_idx` is the upstream electron-index buffer: 0-based site
/// indices indexed as `mi + si * n_elec` (so `ele_idx.len() ==
/// 2 * n_elec`, matching the upstream `n_size`). `slater_elm`,
/// `inv_m`, and `pf_m` are expected to be sized for the full QP range
/// (`slater_elm.n_qp_full() >= qp_end`, etc.); the kernel only touches
/// indices in `[qp_start, qp_end)`.
///
/// `n_elec` is the number of electrons **per spin** (i.e. `Ne` in the
/// C reference, *not* the doubled `Nsize`). The matrix side length is
/// `2 * n_elec`.
///
/// On success the per-QP slot `pf_m[qp]` holds the Pfaffian and the
/// QP plane `inv_m[qp]` holds the (sign-flipped) inverse, ready for
/// the sampler hot path.
pub fn calc_m_all_real(
    ele_idx: &[i64],
    slater_elm: &SlaterElmFlat<f64>,
    inv_m: &mut InvMColMajor<f64>,
    pf_m: &mut [f64],
    qp_start: usize,
    qp_end: usize,
    n_site: usize,
    n_elec: usize,
    pool: &ThreadedPfaPackWorkspace,
) -> Result<(), CalcMAllError> {
    calc_m_all_real_with_status::<false>(
        ele_idx, slater_elm, inv_m, pf_m, qp_start, qp_end, n_site, n_elec, pool,
    )
}

/// C's distinct real setup, including scalar partial publication on failure.
/// Its caller ignores numeric INFO, not typed invalid-input errors.
pub(crate) fn calc_m_all_real_native_info(
    ele_idx: &[i64],
    slater_elm: &SlaterElmFlat<f64>,
    inv_m: &mut InvMColMajor<f64>,
    pf_m: &mut [f64],
    qp_start: usize,
    qp_end: usize,
    n_site: usize,
    n_elec: usize,
    pool: &ThreadedPfaPackWorkspace,
) -> Result<i32, CalcMAllError> {
    native_normal_preflight(
        ele_idx.len(),
        slater_elm.n_site2(),
        slater_elm.n_qp_full(),
        inv_m.n_size(),
        inv_m.n_qp_full(),
        pf_m.len(),
        qp_start,
        qp_end,
        n_site,
        n_elec,
        pool.n_size(),
    )?;
    // C passes N=LDA=0 to DSKTRF for an empty electron matrix. Its argument
    // validation reports INFO=-5 before the N==0 quick return (and before
    // utu2pfa/inverse). Do not call Rust inverse slices with n_size-1.
    if n_elec == 0 {
        return Ok(if qp_start < qp_end { -5 } else { 0 });
    }
    native_normal_info(
        calc_m_all_real_with_status::<true>(
            ele_idx, slater_elm, inv_m, pf_m, qp_start, qp_end, n_site, n_elec, pool,
        ),
        qp_start,
    )
}

fn calc_m_all_real_with_status<const NATIVE_STATUS: bool>(
    ele_idx: &[i64],
    slater_elm: &SlaterElmFlat<f64>,
    inv_m: &mut InvMColMajor<f64>,
    pf_m: &mut [f64],
    qp_start: usize,
    qp_end: usize,
    n_site: usize,
    n_elec: usize,
    pool: &ThreadedPfaPackWorkspace,
) -> Result<(), CalcMAllError> {
    let n_size = 2 * n_elec;
    debug_assert!(qp_start <= qp_end);
    debug_assert!(qp_end <= slater_elm.n_qp_full());
    debug_assert!(qp_end <= inv_m.n_qp_full());
    debug_assert!(pf_m.len() >= qp_end);
    debug_assert_eq!(ele_idx.len(), n_size);
    debug_assert_eq!(inv_m.n_size(), n_size);
    debug_assert_eq!(slater_elm.n_site2(), 2 * n_site);

    let observed = crate::threading::observe_kernel(
        crate::threading::ObservedWork::Qp,
        !NATIVE_STATUS && crate::threading::inner_parallel_enabled(qp_end - qp_start),
    );
    if NATIVE_STATUS || !crate::threading::inner_parallel_enabled(qp_end - qp_start) {
        let mut ws = pool.take();
        let result = (qp_start..qp_end).try_for_each(|qp| {
            let _entry = observed.enter_item();
            calc_m_all_child_real::<NATIVE_STATUS>(
                qp,
                ele_idx,
                slater_elm,
                inv_m,
                &mut pf_m[qp],
                n_site,
                n_elec,
                &mut ws,
            )
        });
        pool.release(ws);
        return result;
    }

    let workers = crate::threading::inner_worker_count(qp_end - qp_start);
    pool.ensure_capacity(workers);
    let chunk = (qp_end - qp_start).div_ceil(workers);
    let ranges: Vec<_> = (qp_start..qp_end)
        .step_by(chunk)
        .map(|start| (start, (start + chunk).min(qp_end)))
        .collect();
    crate::threading::install(|| {
        let chunks: Vec<Result<_, CalcMAllError>> = ranges
            .into_par_iter()
            .map(|(start, end)| {
                let mut ws = pool.take();
                let mut local_inv = InvMColMajor::zeros(end, n_elec);
                let mut local_pf = vec![0.0_f64; end];
                let result = (start..end).try_for_each(|qp| {
                    let _entry = observed.enter_item();
                    calc_m_all_child_real::<NATIVE_STATUS>(
                        qp,
                        ele_idx,
                        slater_elm,
                        &mut local_inv,
                        &mut local_pf[qp],
                        n_site,
                        n_elec,
                        &mut ws,
                    )
                });
                pool.release(ws);
                result.map(|()| (start, end, local_inv, local_pf))
            })
            .collect();
        let chunks: Result<Vec<_>, CalcMAllError> = chunks.into_iter().collect();
        let chunks = chunks?;
        for (start, end, local_inv, local_pf) in chunks {
            pf_m[start..end].copy_from_slice(&local_pf[start..end]);
            for qp in start..end {
                let source = local_inv.qp_matrix_slice(qp);
                inv_m.qp_matrix_slice_mut(qp).copy_from_slice(source);
            }
        }
        Ok(())
    })
}

/// Complex `calculate_m_all` over the half-open QP range `[qp_start, qp_end)`.
///
/// Mirrors [`calc_m_all_real`] but operates on the complex Slater /
/// inverse blocks. Same `n_elec` (= per-spin) convention.
pub fn calc_m_all_complex(
    ele_idx: &[i64],
    slater_elm: &SlaterElmFlat<Complex64>,
    inv_m: &mut InvMColMajor<Complex64>,
    pf_m: &mut [Complex64],
    qp_start: usize,
    qp_end: usize,
    n_site: usize,
    n_elec: usize,
    pool: &ThreadedPfaPackWorkspace,
) -> Result<(), CalcMAllError> {
    calc_m_all_complex_with_kernel::<false, false>(
        ele_idx, slater_elm, inv_m, pf_m, qp_start, qp_end, n_site, n_elec, pool,
    )
}

/// Complex `calculate_m_all` using the operation order from C's `ZSKTRF` path.
///
/// This is reserved for C-authoritative numerical paths such as Full Lanczos;
/// ordinary sampling keeps the Julia-compatible production kernel above.
pub(crate) fn calc_m_all_complex_c_compat(
    ele_idx: &[i64],
    slater_elm: &SlaterElmFlat<Complex64>,
    inv_m: &mut InvMColMajor<Complex64>,
    pf_m: &mut [Complex64],
    qp_start: usize,
    qp_end: usize,
    n_site: usize,
    n_elec: usize,
    pool: &ThreadedPfaPackWorkspace,
) -> Result<(), CalcMAllError> {
    calc_m_all_complex_with_kernel::<true, false>(
        ele_idx, slater_elm, inv_m, pf_m, qp_start, qp_end, n_site, n_elec, pool,
    )
}

/// Native normal-complex validation status, without a Julia-only norm cutoff.
///
/// INFO is the factorization's code, or owned-range-relative QP+1 for a
/// nonfinite Pfaffian. Invalid Rust/input preconditions remain typed errors.
/// This does not promise C's timing-dependent choice among concurrent errors.
/// Validation uses the scalar QP path so failed calls retain the same ordered
/// partial publication as C with OMP_NUM_THREADS=1, not all-chunk rollback.
pub(crate) fn calc_m_all_complex_native_info(
    ele_idx: &[i64],
    slater_elm: &SlaterElmFlat<Complex64>,
    inv_m: &mut InvMColMajor<Complex64>,
    pf_m: &mut [Complex64],
    qp_start: usize,
    qp_end: usize,
    n_site: usize,
    n_elec: usize,
    pool: &ThreadedPfaPackWorkspace,
) -> Result<i32, CalcMAllError> {
    native_normal_preflight(
        ele_idx.len(),
        slater_elm.n_site2(),
        slater_elm.n_qp_full(),
        inv_m.n_size(),
        inv_m.n_qp_full(),
        pf_m.len(),
        qp_start,
        qp_end,
        n_site,
        n_elec,
        pool.n_size(),
    )?;
    // Same ZSKTRF argument validation: LDA=0 is illegal even when N=0.
    // This is a native signed status, not a successful zero-size Pfaffian.
    if n_elec == 0 {
        return Ok(if qp_start < qp_end { -5 } else { 0 });
    }
    native_normal_info(
        calc_m_all_complex_with_kernel::<true, true>(
            ele_idx, slater_elm, inv_m, pf_m, qp_start, qp_end, n_site, n_elec, pool,
        ),
        qp_start,
    )
}

/// Validate before handling empty owned ranges or the N=0 native INFO.
/// All failures leave Pf/Inv untouched and remain typed, not numeric status.
#[allow(clippy::too_many_arguments)]
fn native_normal_preflight(
    ele_len: usize,
    site2: usize,
    slater_qp: usize,
    inv_size: usize,
    inv_qp: usize,
    pf_len: usize,
    qp_start: usize,
    qp_end: usize,
    n_site: usize,
    n_elec: usize,
    pool_size: usize,
) -> Result<(), CalcMAllError> {
    let error = |reason| CalcMAllError::InputShape { reason };
    let n_size = n_elec
        .checked_mul(2)
        .ok_or_else(|| error("Nsize overflow"))?;
    let n_site2 = n_site
        .checked_mul(2)
        .ok_or_else(|| error("Nsite2 overflow"))?;
    if n_elec > n_site {
        return Err(error("normal Ne exceeds Nsite"));
    }
    if n_size >= i32::MAX as usize || slater_qp >= i32::MAX as usize || inv_qp >= i32::MAX as usize
    {
        return Err(error("native INFO dimension domain"));
    }
    if qp_start > qp_end || qp_end > slater_qp || qp_end > inv_qp || qp_end > pf_len {
        return Err(error("owned QP range"));
    }
    if ele_len != n_size || inv_size != n_size || pool_size != n_size || site2 != n_site2 {
        return Err(error("electron/slater/inverse/workspace dimensions"));
    }
    Ok(())
}

fn native_normal_info(
    result: Result<(), CalcMAllError>,
    qp_start: usize,
) -> Result<i32, CalcMAllError> {
    match result {
        Ok(()) => Ok(0),
        Err(error @ CalcMAllError::ZeroPivot { info, .. }) => {
            i32::try_from(info).map_err(|_| error)
        }
        Err(error @ CalcMAllError::NonFinitePfaffian { qp }) => qp
            .checked_sub(qp_start)
            .and_then(|relative| relative.checked_add(1))
            .and_then(|info| i32::try_from(info).ok())
            .ok_or(error),
        Err(error) => Err(error),
    }
}

fn calc_m_all_complex_with_kernel<const C_COMPAT: bool, const NATIVE_STATUS: bool>(
    ele_idx: &[i64],
    slater_elm: &SlaterElmFlat<Complex64>,
    inv_m: &mut InvMColMajor<Complex64>,
    pf_m: &mut [Complex64],
    qp_start: usize,
    qp_end: usize,
    n_site: usize,
    n_elec: usize,
    pool: &ThreadedPfaPackWorkspace,
) -> Result<(), CalcMAllError> {
    let n_size = 2 * n_elec;
    debug_assert!(qp_start <= qp_end);
    debug_assert!(qp_end <= slater_elm.n_qp_full());
    debug_assert!(qp_end <= inv_m.n_qp_full());
    debug_assert!(pf_m.len() >= qp_end);
    debug_assert_eq!(ele_idx.len(), n_size);
    debug_assert_eq!(inv_m.n_size(), n_size);
    debug_assert_eq!(slater_elm.n_site2(), 2 * n_site);

    let observed = crate::threading::observe_kernel(
        crate::threading::ObservedWork::Qp,
        !NATIVE_STATUS && crate::threading::inner_parallel_enabled(qp_end - qp_start),
    );
    if NATIVE_STATUS || !crate::threading::inner_parallel_enabled(qp_end - qp_start) {
        let mut ws = pool.take();
        let result = (qp_start..qp_end).try_for_each(|qp| {
            let _entry = observed.enter_item();
            calc_m_all_child_complex::<C_COMPAT, NATIVE_STATUS>(
                qp,
                ele_idx,
                slater_elm,
                inv_m,
                &mut pf_m[qp],
                n_site,
                n_elec,
                &mut ws,
            )
        });
        pool.release(ws);
        return result;
    }

    let workers = crate::threading::inner_worker_count(qp_end - qp_start);
    pool.ensure_capacity(workers);
    let chunk = (qp_end - qp_start).div_ceil(workers);
    let ranges: Vec<_> = (qp_start..qp_end)
        .step_by(chunk)
        .map(|start| (start, (start + chunk).min(qp_end)))
        .collect();
    crate::threading::install(|| {
        let chunks: Vec<Result<_, CalcMAllError>> = ranges
            .into_par_iter()
            .map(|(start, end)| {
                let mut ws = pool.take();
                let mut local_inv = InvMColMajor::zeros(end, n_elec);
                let mut local_pf = vec![Complex64::default(); end];
                let result = (start..end).try_for_each(|qp| {
                    let _entry = observed.enter_item();
                    calc_m_all_child_complex::<C_COMPAT, NATIVE_STATUS>(
                        qp,
                        ele_idx,
                        slater_elm,
                        &mut local_inv,
                        &mut local_pf[qp],
                        n_site,
                        n_elec,
                        &mut ws,
                    )
                });
                pool.release(ws);
                result.map(|()| (start, end, local_inv, local_pf))
            })
            .collect();
        let chunks: Result<Vec<_>, CalcMAllError> = chunks.into_iter().collect();
        let chunks = chunks?;
        for (start, end, local_inv, local_pf) in chunks {
            pf_m[start..end].copy_from_slice(&local_pf[start..end]);
            for qp in start..end {
                let source = local_inv.qp_matrix_slice(qp);
                inv_m.qp_matrix_slice_mut(qp).copy_from_slice(source);
            }
        }
        Ok(())
    })
}

/// Complex FSZ `calculate_m_all_fsz!` over `[qp_start, qp_end)`.
///
/// Unlike [`calc_m_all_complex`], electron spin is read from `ele_spn[mi]`
/// instead of deriving it from `mi / n_elec`.
pub fn calc_m_all_fsz_complex(
    ele_idx: &[i64],
    ele_spn: &[i64],
    slater_elm: &SlaterElmFlat<Complex64>,
    inv_m: &mut InvMColMajor<Complex64>,
    pf_m: &mut [Complex64],
    qp_start: usize,
    qp_end: usize,
    n_site: usize,
    n_elec: usize,
    pool: &ThreadedPfaPackWorkspace,
) -> Result<(), CalcMAllError> {
    let n_size = 2 * n_elec;
    debug_assert!(qp_start <= qp_end);
    debug_assert!(qp_end <= slater_elm.n_qp_full());
    debug_assert!(qp_end <= inv_m.n_qp_full());
    debug_assert!(pf_m.len() >= qp_end);
    debug_assert_eq!(ele_idx.len(), n_size);
    debug_assert_eq!(ele_spn.len(), n_size);
    debug_assert_eq!(inv_m.n_size(), n_size);
    debug_assert_eq!(slater_elm.n_site2(), 2 * n_site);

    // Julia computes into workspace planes, then publishes the entire range
    // only after every Pfaffian is finite. Failed initialization retries must
    // not publish a partly calculated inverse or Pfaffian table.
    if qp_start == qp_end {
        return Ok(());
    }
    let mut inv_temp = InvMColMajor::zeros(qp_end, n_elec);
    let mut pf_temp = vec![Complex64::default(); qp_end];
    let observed = crate::threading::observe_kernel(
        crate::threading::ObservedWork::Qp,
        crate::threading::inner_parallel_enabled(qp_end - qp_start),
    );
    if crate::threading::inner_parallel_enabled(qp_end - qp_start) {
        let workers = crate::threading::inner_worker_count(qp_end - qp_start);
        pool.ensure_capacity(workers);
        let chunk = (qp_end - qp_start).div_ceil(workers);
        let ranges: Vec<_> = (qp_start..qp_end)
            .step_by(chunk)
            .map(|start| (start, (start + chunk).min(qp_end)))
            .collect();
        let chunks: Vec<Result<_, CalcMAllError>> = crate::threading::install(|| {
            ranges
                .into_par_iter()
                .map(|(start, end)| {
                    let mut ws = pool.take();
                    let mut inverse = InvMColMajor::zeros(end, n_elec);
                    let mut pf = vec![Complex64::default(); end];
                    let result = (start..end).try_for_each(|qp| {
                        let _entry = observed.enter_item();
                        calc_m_all_child_fsz_complex(
                            qp,
                            ele_idx,
                            ele_spn,
                            slater_elm,
                            &mut inverse,
                            &mut pf[qp],
                            n_site,
                            n_elec,
                            &mut ws,
                        )
                    });
                    pool.release(ws);
                    result.map(|()| (start, end, inverse, pf))
                })
                .collect()
        });
        // Collect every chunk in indexed order before selecting the first
        // error serially. Rayon Result collection alone can choose any error.
        let chunks: Result<Vec<_>, CalcMAllError> = chunks.into_iter().collect();
        // Nothing is published if any worker fails.
        for (start, end, inverse, pf) in chunks? {
            pf_temp[start..end].copy_from_slice(&pf[start..end]);
            for qp in start..end {
                inv_temp
                    .qp_matrix_slice_mut(qp)
                    .copy_from_slice(inverse.qp_matrix_slice(qp));
            }
        }
    } else {
        let mut ws = pool.take();
        let result = (qp_start..qp_end).try_for_each(|qp| {
            let _entry = observed.enter_item();
            calc_m_all_child_fsz_complex(
                qp,
                ele_idx,
                ele_spn,
                slater_elm,
                &mut inv_temp,
                &mut pf_temp[qp],
                n_site,
                n_elec,
                &mut ws,
            )
        });
        pool.release(ws);
        result?;
    }
    for qp in qp_start..qp_end {
        inv_m
            .qp_matrix_slice_mut(qp)
            .copy_from_slice(inv_temp.qp_matrix_slice(qp));
        pf_m[qp] = pf_temp[qp];
    }
    Ok(())
}

/// Julia's `calculate_m_all_fsz_real!`: calculate through the complex FSZ
/// kernel, then copy real parts into the real Pfaffian and inverse shadows.
///
/// The complex Slater table is authoritative even in real mode. Only the
/// half-open QP range is copied; inverse scratch pads are left untouched.
/// Failed calculations publish neither complex nor real results.
pub fn calc_m_all_fsz_real(
    ele_idx: &[i64],
    ele_spn: &[i64],
    matrix: &mut SlaterMatrixData,
    qp_start: usize,
    qp_end: usize,
    n_site: usize,
    n_elec: usize,
    pool: &ThreadedPfaPackWorkspace,
) -> Result<(), CalcMAllError> {
    assert!(qp_start <= qp_end && qp_end <= matrix.pf_m_real.len());
    assert!(qp_end <= matrix.inv_m_real.n_qp_full());
    assert_eq!(matrix.inv_m_real.n_size(), 2 * n_elec);
    calc_m_all_fsz_complex(
        ele_idx,
        ele_spn,
        &matrix.slater_elm,
        &mut matrix.inv_m,
        &mut matrix.pf_m,
        qp_start,
        qp_end,
        n_site,
        n_elec,
        pool,
    )?;
    crate::threading::copy_complex_realpart(
        &mut matrix.pf_m_real[qp_start..qp_end],
        &matrix.pf_m[qp_start..qp_end],
    );
    let stride = (2 * n_elec).pow(2) + 1;
    let src = matrix.inv_m.as_slice();
    let start = qp_start * stride;
    crate::threading::for_each_mut(
        &mut matrix.inv_m_real.as_mut_slice()[start..qp_end * stride],
        |i, dst| {
            if i % stride != stride - 1 {
                *dst = src[start + i].re;
            }
        },
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Per-QP child kernels
// ---------------------------------------------------------------------------

fn calc_m_all_child_real<const NATIVE_STATUS: bool>(
    qp: usize,
    ele_idx: &[i64],
    slater_elm: &SlaterElmFlat<f64>,
    inv_m: &mut InvMColMajor<f64>,
    pf_slot: &mut f64,
    n_site: usize,
    n_elec: usize,
    ws: &mut PfaPackWorkspace,
) -> Result<(), CalcMAllError> {
    let ne = n_elec; // electrons per spin
    let n_size = 2 * ne;
    let n_site2 = 2 * n_site;

    assemble_inv_m_real(qp, ele_idx, slater_elm, inv_m, n_site, ne, n_size, n_site2)?;
    let observer = crate::run::first_real_factor_observer(qp);
    if let Some(observer) = &observer {
        observer.real_factor(crate::run::RealFactorView {
            stage: "assembled",
            qp,
            dimension: n_size,
            matrix: inv_m.qp_matrix_slice(qp),
            pivots: &[],
            factor_error: None,
            pf: None,
        });
    }
    if !NATIVE_STATUS && frobenius_norm_sqr_real(inv_m, qp) < MIN_ABS2 {
        return Err(CalcMAllError::AllZero { qp });
    }

    ensure_workspace_real(ws, n_size);
    let pf_value = {
        let qp_buf = inv_m.qp_matrix_slice_mut(qp);
        let factor_result = {
            let mut a = SqMat::new(qp_buf, n_size);
            dsktf2(&mut a, &mut ws.pivots[..n_size])
        };
        if let Some(observer) = &observer {
            observer.real_factor(crate::run::RealFactorView {
                stage: "factorized",
                qp,
                dimension: n_size,
                matrix: qp_buf,
                pivots: &ws.pivots[..n_size],
                factor_error: factor_result.as_ref().err().copied(),
                pf: None,
            });
        }
        factor_result.map_err(|info| CalcMAllError::ZeroPivot { qp, info })?;
        let a = SqMat::new(qp_buf, n_size);
        utu2pfa_real(&a, &ws.pivots[..n_size])
    };
    if let Some(observer) = &observer {
        observer.real_factor(crate::run::RealFactorView {
            stage: "pf",
            qp,
            dimension: n_size,
            matrix: inv_m.qp_matrix_slice(qp),
            pivots: &ws.pivots[..n_size],
            factor_error: None,
            pf: Some(pf_value),
        });
    }
    if !pf_value.is_finite() {
        return Err(CalcMAllError::NonFinitePfaffian { qp });
    }
    *pf_slot = pf_value;

    {
        let qp_buf = inv_m.qp_matrix_slice_mut(qp);
        let mut a = SqMat::new(qp_buf, n_size);
        let mut m_work = SqMat::new(&mut ws.m_work_real[..n_size * n_size], n_size);
        if let Some(observer) = &observer {
            pfapack::utu2::utu2inv_real_observed(
                &mut a,
                &ws.pivots[..n_size],
                &mut ws.v_t_real[..n_size - 1],
                &mut m_work,
                &|view| observer.real_step5(view),
            );
        } else {
            utu2inv_real(
                &mut a,
                &ws.pivots[..n_size],
                &mut ws.v_t_real[..n_size - 1],
                &mut m_work,
            );
        }
    }

    if let Some(observer) = &observer {
        observer.real_factor(crate::run::RealFactorView {
            stage: "inverse-before-sign",
            qp,
            dimension: n_size,
            matrix: inv_m.qp_matrix_slice(qp),
            pivots: &ws.pivots[..n_size],
            factor_error: None,
            pf: Some(pf_value),
        });
    }

    // `M_DSCAL(&nsq, &minus_one, invM, &one)` -- final sign flip.
    for x in inv_m.qp_matrix_slice_mut(qp) {
        *x = -*x;
    }
    if let Some(observer) = &observer {
        observer.real_factor(crate::run::RealFactorView {
            stage: "inverse-published",
            qp,
            dimension: n_size,
            matrix: inv_m.qp_matrix_slice(qp),
            pivots: &ws.pivots[..n_size],
            factor_error: None,
            pf: Some(pf_value),
        });
    }
    Ok(())
}

fn calc_m_all_child_complex<const C_COMPAT: bool, const NATIVE_STATUS: bool>(
    qp: usize,
    ele_idx: &[i64],
    slater_elm: &SlaterElmFlat<Complex64>,
    inv_m: &mut InvMColMajor<Complex64>,
    pf_slot: &mut Complex64,
    n_site: usize,
    n_elec: usize,
    ws: &mut PfaPackWorkspace,
) -> Result<(), CalcMAllError> {
    let ne = n_elec; // electrons per spin
    let n_size = 2 * ne;
    let n_site2 = 2 * n_site;

    assemble_inv_m_complex(qp, ele_idx, slater_elm, inv_m, n_site, ne, n_size, n_site2)?;
    if !NATIVE_STATUS && frobenius_norm_sqr_complex(inv_m, qp) < MIN_ABS2 {
        return Err(CalcMAllError::AllZero { qp });
    }

    ensure_workspace_complex(ws, n_size);
    let pf_value = {
        let qp_buf = inv_m.qp_matrix_slice_mut(qp);
        let mut a = SqMat::new(qp_buf, n_size);
        let result = if C_COMPAT {
            zsktf2_c_compat(&mut a, &mut ws.pivots[..n_size])
        } else {
            zsktf2_turbo(&mut a, &mut ws.pivots[..n_size])
        };
        result.map_err(|info| CalcMAllError::ZeroPivot { qp, info })?;
        utu2pfa_complex(&a, &ws.pivots[..n_size])
    };
    if if NATIVE_STATUS {
        !(pf_value.re + pf_value.im).is_finite()
    } else {
        !pf_value.re.is_finite() || !pf_value.im.is_finite()
    } {
        return Err(CalcMAllError::NonFinitePfaffian { qp });
    }
    *pf_slot = pf_value;

    {
        let qp_buf = inv_m.qp_matrix_slice_mut(qp);
        let mut a = SqMat::new(qp_buf, n_size);
        let mut m_work = SqMat::new(&mut ws.m_work_complex[..n_size * n_size], n_size);
        utu2inv_complex(
            &mut a,
            &ws.pivots[..n_size],
            &mut ws.v_t_complex[..n_size - 1],
            &mut m_work,
        );
    }

    // `M_ZSCAL(&nsq, &minus_one, invM, &one)` -- final sign flip.
    for z in inv_m.qp_matrix_slice_mut(qp) {
        *z = -*z;
    }
    Ok(())
}

fn calc_m_all_child_fsz_complex(
    qp: usize,
    ele_idx: &[i64],
    ele_spn: &[i64],
    slater_elm: &SlaterElmFlat<Complex64>,
    inv_m: &mut InvMColMajor<Complex64>,
    pf_slot: &mut Complex64,
    n_site: usize,
    n_elec: usize,
    ws: &mut PfaPackWorkspace,
) -> Result<(), CalcMAllError> {
    let n_size = 2 * n_elec;
    let n_site2 = 2 * n_site;

    assemble_inv_m_fsz_complex(
        qp, ele_idx, ele_spn, slater_elm, inv_m, n_size, n_site, n_site2,
    )?;
    ensure_workspace_complex(ws, n_size);
    let pf_value = {
        let qp_buf = inv_m.qp_matrix_slice_mut(qp);
        let mut a = SqMat::new(qp_buf, n_size);
        // The authoritative FSZ path ignores the factorization's zero-pivot
        // status and checks only whether the resulting Pfaffian is finite.
        // This includes a zero Pfaffian with a nonfinite inverse.
        let _ = zsktf2(&mut a, &mut ws.pivots[..n_size]);
        utu2pfa_complex(&a, &ws.pivots[..n_size])
    };
    if !pf_value.re.is_finite() || !pf_value.im.is_finite() {
        return Err(CalcMAllError::NonFinitePfaffian { qp });
    }
    *pf_slot = pf_value;

    {
        let qp_buf = inv_m.qp_matrix_slice_mut(qp);
        let mut a = SqMat::new(qp_buf, n_size);
        let mut m_work = SqMat::new(&mut ws.m_work_complex[..n_size * n_size], n_size);
        utu2inv_complex_fsz(
            &mut a,
            &ws.pivots[..n_size],
            &mut ws.v_t_complex[..n_size - 1],
            &mut m_work,
            fsz_inverse_divide,
        );
    }

    for z in inv_m.qp_matrix_slice_mut(qp) {
        *z = -*z;
    }
    Ok(())
}

// Julia's FSZ native C++ inverse uses the platform complex division runtime.
// GNU/Linux uses libgcc's Smith ratio; macOS ARM uses compiler-rt's FMA
// quotient. Intel macOS retains the independently verified archived arithmetic.
#[cfg(any(
    all(target_os = "linux", target_env = "gnu"),
    all(target_os = "macos", target_arch = "aarch64")
))]
fn fsz_inverse_divide(a: Complex64, b: Complex64) -> Complex64 {
    crate::c_complex::divide(a, b)
}

#[cfg(not(any(
    all(target_os = "linux", target_env = "gnu"),
    all(target_os = "macos", target_arch = "aarch64")
)))]
fn fsz_inverse_divide(a: Complex64, b: Complex64) -> Complex64 {
    a / b
}

// ---------------------------------------------------------------------------
// inv_m assembly + diagnostics
// ---------------------------------------------------------------------------

fn assemble_inv_m_real(
    qp: usize,
    ele_idx: &[i64],
    slater_elm: &SlaterElmFlat<f64>,
    inv_m: &mut InvMColMajor<f64>,
    n_site: usize,
    ne: usize,
    n_size: usize,
    n_site2: usize,
) -> Result<(), CalcMAllError> {
    for msi in 0..n_size {
        let si = msi / ne; // spin index (0 or 1)
        let ri = ele_idx[msi];
        let rsi = ri + (si as i64) * (n_site as i64);
        if rsi < 0 || rsi >= n_site2 as i64 {
            return Err(CalcMAllError::SiteOutOfRange { qp, site: rsi });
        }
        for msj in 0..n_size {
            let sj = msj / ne;
            let rj = ele_idx[msj];
            let rsj = rj + (sj as i64) * (n_site as i64);
            if rsj < 0 || rsj >= n_site2 as i64 {
                return Err(CalcMAllError::SiteOutOfRange { qp, site: rsj });
            }
            let value = slater_elm.get(qp, rsi as usize, rsj as usize);
            // Upstream stores column-major: inv_m[msj, msi] = -value.
            inv_m.set(qp, msj, msi, -value);
        }
    }
    Ok(())
}

fn assemble_inv_m_complex(
    qp: usize,
    ele_idx: &[i64],
    slater_elm: &SlaterElmFlat<Complex64>,
    inv_m: &mut InvMColMajor<Complex64>,
    n_site: usize,
    ne: usize,
    n_size: usize,
    n_site2: usize,
) -> Result<(), CalcMAllError> {
    for msi in 0..n_size {
        let si = msi / ne;
        let ri = ele_idx[msi];
        let rsi = ri + (si as i64) * (n_site as i64);
        if rsi < 0 || rsi >= n_site2 as i64 {
            return Err(CalcMAllError::SiteOutOfRange { qp, site: rsi });
        }
        for msj in 0..n_size {
            let sj = msj / ne;
            let rj = ele_idx[msj];
            let rsj = rj + (sj as i64) * (n_site as i64);
            if rsj < 0 || rsj >= n_site2 as i64 {
                return Err(CalcMAllError::SiteOutOfRange { qp, site: rsj });
            }
            let value = slater_elm.get(qp, rsi as usize, rsj as usize);
            inv_m.set(qp, msj, msi, -value);
        }
    }
    Ok(())
}

fn assemble_inv_m_fsz_complex(
    qp: usize,
    ele_idx: &[i64],
    ele_spn: &[i64],
    slater_elm: &SlaterElmFlat<Complex64>,
    inv_m: &mut InvMColMajor<Complex64>,
    n_size: usize,
    n_site: usize,
    n_site2: usize,
) -> Result<(), CalcMAllError> {
    for msi in 0..n_size {
        let ri = ele_idx[msi];
        let si = ele_spn[msi];
        let rsi = ri + si * (n_site as i64);
        if rsi < 0 || rsi >= n_site2 as i64 {
            return Err(CalcMAllError::SiteOutOfRange { qp, site: rsi });
        }
        for msj in 0..n_size {
            let rj = ele_idx[msj];
            let sj = ele_spn[msj];
            let rsj = rj + sj * (n_site as i64);
            if rsj < 0 || rsj >= n_site2 as i64 {
                return Err(CalcMAllError::SiteOutOfRange { qp, site: rsj });
            }
            let value = slater_elm.get(qp, rsi as usize, rsj as usize);
            inv_m.set(qp, msj, msi, -value);
        }
    }
    Ok(())
}

fn frobenius_norm_sqr_real(inv_m: &InvMColMajor<f64>, qp: usize) -> f64 {
    let n = inv_m.n_size();
    let mut max_abs2 = 0.0_f64;
    for col in 0..n {
        for row in 0..n {
            let v = inv_m.get(qp, row, col);
            max_abs2 = max_abs2.max(v * v);
        }
    }
    max_abs2
}

fn frobenius_norm_sqr_complex(inv_m: &InvMColMajor<Complex64>, qp: usize) -> f64 {
    let n = inv_m.n_size();
    let mut max_abs2 = 0.0_f64;
    for col in 0..n {
        for row in 0..n {
            let v = inv_m.get(qp, row, col);
            max_abs2 = max_abs2.max(v.norm_sqr());
        }
    }
    max_abs2
}

// ---------------------------------------------------------------------------
// Workspace sizing helpers (lazy grow if the pool was created for a
// smaller `n_size`).
// ---------------------------------------------------------------------------

fn ensure_workspace_real(ws: &mut PfaPackWorkspace, n_size: usize) {
    if ws.buf_m_real.len() < n_size * n_size {
        ws.buf_m_real.resize(n_size * n_size, 0.0);
        ws.m_work_real.resize(n_size * n_size, 0.0);
        ws.v_t_real.resize(n_size - 1, 0.0);
    }
    if ws.pivots.len() < n_size {
        ws.pivots.resize(n_size, pfapack::PivotIndex1Based(0));
    }
    // mode might be RealOnly so n_size grew but ComplexOnly already 0.
    debug_assert!(ws.buf_m_real.len() >= n_size * n_size);
    debug_assert!(ws.m_work_real.len() >= n_size * n_size);
    debug_assert!(ws.v_t_real.len() >= n_size - 1);
    debug_assert!(ws.pivots.len() >= n_size);
    let _ = PfaPackMode::Both; // keep import in scope
}

fn ensure_workspace_complex(ws: &mut PfaPackWorkspace, n_size: usize) {
    if ws.buf_m_complex.len() < n_size * n_size {
        ws.buf_m_complex
            .resize(n_size * n_size, Complex64::new(0.0, 0.0));
        ws.m_work_complex
            .resize(n_size * n_size, Complex64::new(0.0, 0.0));
        ws.v_t_complex.resize(n_size - 1, Complex64::new(0.0, 0.0));
    }
    if ws.pivots.len() < n_size {
        ws.pivots.resize(n_size, pfapack::PivotIndex1Based(0));
    }
    debug_assert!(ws.buf_m_complex.len() >= n_size * n_size);
    debug_assert!(ws.m_work_complex.len() >= n_size * n_size);
    debug_assert!(ws.v_t_complex.len() >= n_size - 1);
    debug_assert!(ws.pivots.len() >= n_size);
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn native_normal_preflight_rejects_unsupported_dimensions_without_allocation() {
        assert!(matches!(
            native_normal_preflight(4, 2, 1, 4, 1, 1, 0, 1, 1, 2, 4,),
            Err(CalcMAllError::InputShape {
                reason: "normal Ne exceeds Nsite"
            })
        ));
        let huge = i32::MAX as usize;
        let ne = huge / 2 + 1;
        let size = 2 * ne;
        assert!(matches!(
            native_normal_preflight(size, size, 1, size, 1, 1, 0, 0, ne, ne, size),
            Err(CalcMAllError::InputShape {
                reason: "native INFO dimension domain"
            })
        ));
        assert!(matches!(
            native_normal_preflight(0, 2, huge, 0, huge, huge, 0, 0, 1, 0, 0,),
            Err(CalcMAllError::InputShape {
                reason: "native INFO dimension domain"
            })
        ));
        assert!(matches!(
            native_normal_preflight(0, 2, 1, 0, 1, 1, 0, 0, 1, usize::MAX, 0,),
            Err(CalcMAllError::InputShape {
                reason: "Nsize overflow"
            })
        ));
    }

    #[test]
    fn native_zero_electron_status_preserves_real_and_complex_storage() {
        let pool = ThreadedPfaPackWorkspace::new(0, 1);
        let real_slater = SlaterElmFlat::<f64>::zeros(2, 1);
        let complex_slater = SlaterElmFlat::<Complex64>::zeros(2, 1);
        let mut real_inv = InvMColMajor::<f64>::zeros(2, 0);
        let mut complex_inv = InvMColMajor::<Complex64>::zeros(2, 0);
        real_inv.as_mut_slice().fill(17.0);
        complex_inv.as_mut_slice().fill(Complex64::new(17.0, -9.0));
        let real_before = real_inv.as_slice().to_vec();
        let complex_before = complex_inv.as_slice().to_vec();
        let mut real_pf = [3.0, 5.0];
        let mut complex_pf = [Complex64::new(3.0, 4.0), Complex64::new(5.0, 6.0)];
        let complex_pf_before = complex_pf;
        for (start, end, expected) in [(0, 2, -5), (1, 1, 0), (2, 2, 0)] {
            assert_eq!(
                calc_m_all_real_native_info(
                    &[],
                    &real_slater,
                    &mut real_inv,
                    &mut real_pf,
                    start,
                    end,
                    1,
                    0,
                    &pool,
                ),
                Ok(expected)
            );
            assert_eq!(
                calc_m_all_complex_native_info(
                    &[],
                    &complex_slater,
                    &mut complex_inv,
                    &mut complex_pf,
                    start,
                    end,
                    1,
                    0,
                    &pool,
                ),
                Ok(expected)
            );
            assert_eq!(real_pf, [3.0, 5.0]);
            assert_eq!(complex_pf, complex_pf_before);
            assert_eq!(real_inv.as_slice(), real_before);
            assert_eq!(complex_inv.as_slice(), complex_before);
        }
        for (start, end) in [(2, 1), (0, 3), (3, 3)] {
            assert!(matches!(
                calc_m_all_real_native_info(
                    &[],
                    &real_slater,
                    &mut real_inv,
                    &mut real_pf,
                    start,
                    end,
                    1,
                    0,
                    &pool,
                ),
                Err(CalcMAllError::InputShape { .. })
            ));
            assert!(matches!(
                calc_m_all_complex_native_info(
                    &[],
                    &complex_slater,
                    &mut complex_inv,
                    &mut complex_pf,
                    start,
                    end,
                    1,
                    0,
                    &pool,
                ),
                Err(CalcMAllError::InputShape { .. })
            ));
        }
        assert!(matches!(
            calc_m_all_real_native_info(
                &[],
                &real_slater,
                &mut real_inv,
                &mut [],
                0,
                1,
                1,
                0,
                &pool,
            ),
            Err(CalcMAllError::InputShape { .. })
        ));
        assert!(matches!(
            calc_m_all_complex_native_info(
                &[],
                &complex_slater,
                &mut complex_inv,
                &mut [],
                0,
                1,
                1,
                0,
                &pool,
            ),
            Err(CalcMAllError::InputShape { .. })
        ));
        // A malformed electron buffer must not turn into the zero-range OK.
        assert!(matches!(
            calc_m_all_real_native_info(
                &[0],
                &real_slater,
                &mut real_inv,
                &mut real_pf,
                0,
                0,
                1,
                0,
                &pool,
            ),
            Err(CalcMAllError::InputShape { .. })
        ));
        assert!(matches!(
            calc_m_all_complex_native_info(
                &[0],
                &complex_slater,
                &mut complex_inv,
                &mut complex_pf,
                0,
                0,
                1,
                0,
                &pool,
            ),
            Err(CalcMAllError::InputShape { .. })
        ));
        assert_eq!(real_pf, [3.0, 5.0]);
        assert_eq!(complex_pf, complex_pf_before);
        assert_eq!(real_inv.as_slice(), real_before);
        assert_eq!(complex_inv.as_slice(), complex_before);
    }

    /// A 2-site, 2-electron, 1-QP example crafted so the assembled
    /// `inv_m` is the simple skew block `[[0, a], [-a, 0]]`. Tests the
    /// end-to-end pipeline + sign-flip convention.
    #[test]
    fn calc_m_all_real_minimal_block() {
        // 2-site Hubbard, 1 electron per spin -> n_size = 2.
        let n_site = 2;
        let n_elec = 1; // per spin
        let n_qp_full = 1;
        let mut slater = SlaterElmFlat::<f64>::zeros(n_qp_full, n_site);
        // Place a single non-zero pair in slater_elm so the assembled
        // (column-major) matrix is the 2x2 skew block [[0, -a], [a, 0]].
        // ele_idx[0]=0 (up electron on site 0) -> rsi=0,
        // ele_idx[1]=0 (down electron on site 0) -> rsi=0+1*n_site=2.
        // So we want slater_elm[0, 2] = a (for msi=0, msj=1).
        let ele_idx = vec![0_i64, 0_i64];
        let a = 1.5;
        slater.set(0, 0, 2, a);
        slater.set(0, 2, 0, -a);

        let mut inv_m = InvMColMajor::<f64>::zeros(n_qp_full, n_elec);
        let mut pf = vec![0.0; n_qp_full];
        let pool = ThreadedPfaPackWorkspace::new(2 * n_elec, 1);
        calc_m_all_real(
            &ele_idx, &slater, &mut inv_m, &mut pf, 0, n_qp_full, n_site, n_elec, &pool,
        )
        .expect("real kernel succeeds on the 2x2 skew block");
        // The assembled column-major matrix is M = -slater[transposed].
        // For our slater_elm[0, 2] = a / [2, 0] = -a, the assembled M
        // (stored as inv_m[msj, msi] = -slater[rsi, rsj]) gives the
        // value at (row=0, col=1) = -slater_elm[2, 0] = -(-a) = a; the
        // skew structure forces M = [[0, a], [-a, 0]].
        //
        // Pf(M) = a. After utu2inv we get M^{-1} = (1/a) * [[0, 1], [-1, 0]];
        // the final sign flip yields the published inv_m = -M^{-1}.
        assert_relative_eq!(pf[0], a, max_relative = 1e-15);
        assert_relative_eq!(inv_m.get(0, 0, 0), 0.0, epsilon = 1e-15);
        assert_relative_eq!(inv_m.get(0, 0, 1), 1.0 / a, max_relative = 1e-15);
        assert_relative_eq!(inv_m.get(0, 1, 0), -1.0 / a, max_relative = 1e-15);
        assert_relative_eq!(inv_m.get(0, 1, 1), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn calc_m_all_real_all_zero_returns_err() {
        let n_site = 2;
        let n_elec = 1; // per spin -> n_size = 2
        let n_qp_full = 1;
        let slater = SlaterElmFlat::<f64>::zeros(n_qp_full, n_site);
        let mut inv_m = InvMColMajor::<f64>::zeros(n_qp_full, n_elec);
        let mut pf = vec![0.0; n_qp_full];
        let pool = ThreadedPfaPackWorkspace::new(2 * n_elec, 1);
        let result = calc_m_all_real(
            &[0_i64, 0_i64],
            &slater,
            &mut inv_m,
            &mut pf,
            0,
            n_qp_full,
            n_site,
            n_elec,
            &pool,
        );
        assert!(matches!(result, Err(CalcMAllError::AllZero { qp: 0 })));
    }

    #[test]
    fn calc_m_all_complex_matches_real_on_real_input() {
        let n_site = 3;
        let n_elec = 2; // per spin -> n_size = 4
        let n_qp_full = 2;
        let n_size = 2 * n_elec;
        let n_site2 = 2 * n_site;

        // Deterministic LCG so the test is reproducible without `rand`.
        let mut s: u64 = 0xdead_beef_dead_beef;
        let mut next_f64 = || -> f64 {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 32) as u32) as f64 / u32::MAX as f64) * 2.0 - 1.0
        };

        // Build a skew-symmetric Slater table per QP plane.
        let mut slater_r = SlaterElmFlat::<f64>::zeros(n_qp_full, n_site);
        let mut slater_c = SlaterElmFlat::<Complex64>::zeros(n_qp_full, n_site);
        for qp in 0..n_qp_full {
            for i in 0..n_site2 {
                for j in (i + 1)..n_site2 {
                    let v = next_f64();
                    slater_r.set(qp, i, j, v);
                    slater_r.set(qp, j, i, -v);
                    slater_c.set(qp, i, j, Complex64::new(v, 0.0));
                    slater_c.set(qp, j, i, Complex64::new(-v, 0.0));
                }
            }
        }
        // Choose ele_idx as the leading n_elec sites; spin layout is
        // automatic via the msi/ne split inside the kernel.
        // ele_idx[mi + si * n_elec] = ri (0-based site, < n_site).
        // Pick the first n_elec sites for spin-up and the same for spin-down.
        let ele_idx: Vec<i64> = (0..n_elec as i64).chain(0..n_elec as i64).collect();

        let mut inv_m_r = InvMColMajor::<f64>::zeros(n_qp_full, n_elec);
        let mut inv_m_c = InvMColMajor::<Complex64>::zeros(n_qp_full, n_elec);
        let mut pf_r = vec![0.0; n_qp_full];
        let mut pf_c = vec![Complex64::new(0.0, 0.0); n_qp_full];
        let pool = ThreadedPfaPackWorkspace::new(n_size, 1);

        calc_m_all_real(
            &ele_idx,
            &slater_r,
            &mut inv_m_r,
            &mut pf_r,
            0,
            n_qp_full,
            n_site,
            n_elec,
            &pool,
        )
        .unwrap();
        calc_m_all_complex(
            &ele_idx,
            &slater_c,
            &mut inv_m_c,
            &mut pf_c,
            0,
            n_qp_full,
            n_site,
            n_elec,
            &pool,
        )
        .unwrap();

        for qp in 0..n_qp_full {
            assert_relative_eq!(pf_r[qp], pf_c[qp].re, max_relative = 1e-12);
            assert!(pf_c[qp].im.abs() < 1e-12);
            for i in 0..n_size {
                for j in 0..n_size {
                    let r = inv_m_r.get(qp, i, j);
                    let c = inv_m_c.get(qp, i, j);
                    assert_relative_eq!(r, c.re, max_relative = 1e-12, epsilon = 1e-14);
                    assert!(c.im.abs() < 1e-12);
                }
            }
        }
    }
}

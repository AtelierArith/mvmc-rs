//! Pfaffian + inverse-matrix construction from Slater elements.
//!
//! Port target: `MVMCOptimizers.jl/src/calculate_m_all.jl` (~930 LOC,
//! `fcmp` / `real` / `fsz` variants). Re-uses `pfapack::{ltl, utu2}`.
//!
//! Phase 4.2 ships the scalar real + complex paths. The Julia code
//! ships both a sequential and an `@threads`-parallel wrapper over the
//! `qpidx` loop; we mirror the API but execute serially through a
//! single workspace taken from [`ThreadedPfaPackWorkspace`]. Phase 6
//! swaps the inner loop for a `rayon::par_iter` once the 10-step
//! bit-parity diff is green.
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
    zsktf2, zsktf2_turbo, SqMat,
};

use crate::state::{
    InvMColMajor, PfaPackMode, PfaPackWorkspace, SlaterElmFlat, ThreadedPfaPackWorkspace,
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
    let n_size = 2 * n_elec;
    debug_assert!(qp_start <= qp_end);
    debug_assert!(qp_end <= slater_elm.n_qp_full());
    debug_assert!(qp_end <= inv_m.n_qp_full());
    debug_assert!(pf_m.len() >= qp_end);
    debug_assert_eq!(ele_idx.len(), n_size);
    debug_assert_eq!(inv_m.n_size(), n_size);
    debug_assert_eq!(slater_elm.n_site2(), 2 * n_site);

    let mut ws = pool.take();
    let result = (qp_start..qp_end).try_for_each(|qp| {
        calc_m_all_child_real(
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
    result
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
    let n_size = 2 * n_elec;
    debug_assert!(qp_start <= qp_end);
    debug_assert!(qp_end <= slater_elm.n_qp_full());
    debug_assert!(qp_end <= inv_m.n_qp_full());
    debug_assert!(pf_m.len() >= qp_end);
    debug_assert_eq!(ele_idx.len(), n_size);
    debug_assert_eq!(inv_m.n_size(), n_size);
    debug_assert_eq!(slater_elm.n_site2(), 2 * n_site);

    let mut ws = pool.take();
    let result = (qp_start..qp_end).try_for_each(|qp| {
        calc_m_all_child_complex(
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
    result
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

    let mut ws = pool.take();
    let result = (qp_start..qp_end).try_for_each(|qp| {
        calc_m_all_child_fsz_complex(
            qp,
            ele_idx,
            ele_spn,
            slater_elm,
            inv_m,
            &mut pf_m[qp],
            n_site,
            n_elec,
            &mut ws,
        )
    });
    pool.release(ws);
    result
}

// ---------------------------------------------------------------------------
// Per-QP child kernels
// ---------------------------------------------------------------------------

fn calc_m_all_child_real(
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
    if frobenius_norm_sqr_real(inv_m, qp) < MIN_ABS2 {
        return Err(CalcMAllError::AllZero { qp });
    }

    ensure_workspace_real(ws, n_size);
    let pf_value = {
        let qp_buf = inv_m.qp_matrix_slice_mut(qp);
        let mut a = SqMat::new(qp_buf, n_size);
        dsktf2(&mut a, &mut ws.pivots[..n_size])
            .map_err(|info| CalcMAllError::ZeroPivot { qp, info })?;
        utu2pfa_real(&a, &ws.pivots[..n_size])
    };
    if !pf_value.is_finite() {
        return Err(CalcMAllError::NonFinitePfaffian { qp });
    }
    *pf_slot = pf_value;

    {
        let qp_buf = inv_m.qp_matrix_slice_mut(qp);
        let mut a = SqMat::new(qp_buf, n_size);
        let mut m_work = SqMat::new(&mut ws.m_work_real[..n_size * n_size], n_size);
        utu2inv_real(
            &mut a,
            &ws.pivots[..n_size],
            &mut ws.v_t_real[..n_size - 1],
            &mut m_work,
        );
    }

    // `M_DSCAL(&nsq, &minus_one, invM, &one)` -- final sign flip.
    for x in inv_m.qp_matrix_slice_mut(qp) {
        *x = -*x;
    }
    Ok(())
}

fn calc_m_all_child_complex(
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
    if frobenius_norm_sqr_complex(inv_m, qp) < MIN_ABS2 {
        return Err(CalcMAllError::AllZero { qp });
    }

    ensure_workspace_complex(ws, n_size);
    let pf_value = {
        let qp_buf = inv_m.qp_matrix_slice_mut(qp);
        let mut a = SqMat::new(qp_buf, n_size);
        zsktf2_turbo(&mut a, &mut ws.pivots[..n_size])
            .map_err(|info| CalcMAllError::ZeroPivot { qp, info })?;
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
    if frobenius_norm_sqr_complex(inv_m, qp) < MIN_ABS2 {
        return Err(CalcMAllError::AllZero { qp });
    }

    ensure_workspace_complex(ws, n_size);
    let pf_value = {
        let qp_buf = inv_m.qp_matrix_slice_mut(qp);
        let mut a = SqMat::new(qp_buf, n_size);
        zsktf2(&mut a, &mut ws.pivots[..n_size])
            .map_err(|info| CalcMAllError::ZeroPivot { qp, info })?;
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
        );
    }

    for z in inv_m.qp_matrix_slice_mut(qp) {
        *z = -*z;
    }
    Ok(())
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

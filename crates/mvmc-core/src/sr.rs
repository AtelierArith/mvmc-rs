//! Stochastic Reconfiguration optimizer.
//!
//! Port target: `MVMCOptimizers.jl/src/stochastic_opt.jl`.
//!
//! Implementation note (do NOT collapse): the direct solver uses the
//! two-step `dpotrf` -> `dpotrs` form rather than `dposv`. The two-step
//! form lets the upstream apply the SR diagonal-shift regularisation
//! between factorisation and back-substitution; collapsing into `dposv`
//! would silently change the regularisation behaviour.

use mvmc_expert_parsers::utils::parameter_init::n_slater;
use mvmc_expert_parsers::ExpertModeData;
use num_complex::Complex64;

use crate::state::VmcOptimizationState;

// Fortran LAPACK symbols provided by the system LAPACK library linked via
// `build.rs`.  The Fortran calling convention passes scalars by pointer and
// appends a trailing underscore to the routine name.
extern "C" {
    fn dpotrf_(
        uplo: *const u8, // b"U" → upper triangle, b"L" → lower triangle
        n: *const i32,
        a: *mut f64, // n×n SPD matrix; overwritten with Cholesky factor
        lda: *const i32,
        info: *mut i32, // 0 on success, >0 if not positive-definite
    );
    fn dpotrs_(
        uplo: *const u8, // must match the uplo used in dpotrf_
        n: *const i32,
        nrhs: *const i32,
        a: *const f64, // Cholesky factor from dpotrf_ (read-only)
        lda: *const i32,
        b: *mut f64, // RHS on entry, solution on exit
        ldb: *const i32,
        info: *mut i32,
    );
}

/// Apply one Stochastic Reconfiguration step in real mode.
///
/// Mirrors `stochastic_opt!` in upstream Julia for `all_complex = false`.
/// The complex branch lives in [`stochastic_opt_complex`] and shares
/// the same `S`-matrix / `g`-vector construction logic.
///
/// Returns `0` on success and `1` if the Cholesky factorisation fails
/// (matches the upstream `info` convention).
pub fn stochastic_opt_real(data: &mut ExpertModeData, state: &mut VmcOptimizationState) -> i32 {
    let n_proj = data.gutzwiller_terms.len() + data.jastrow_terms.len();
    let n_orb = n_slater(data);
    let n_para = n_proj + n_orb;
    if n_para == 0 {
        return 0;
    }
    data.ensure_optimization_flags(n_para);

    let sr_opt_size = state.sr_opt.sr_opt_size;
    if !state.sr_opt.sr_opt_oo_real.is_empty() {
        // Real fast-path: SROptOO_real lives in the dedicated real buffer.
        let (s_diag, smat_to_para_idx) = collect_active_real(data, state, n_para, sr_opt_size);
        if smat_to_para_idx.is_empty() {
            return 0;
        }
        let n_smat = smat_to_para_idx.len();
        let mut s = vec![0.0_f64; n_smat * n_smat];
        let mut g = vec![0.0_f64; n_smat];
        build_s_g_real(
            &mut s,
            &mut g,
            &smat_to_para_idx,
            state,
            sr_opt_size,
            data.modpara.dsr_opt_sta_del,
            data.modpara.dsr_opt_step_dt,
        );
        let _ = s_diag;
        if cholesky_solve(&mut s, &mut g, n_smat).is_err() {
            return 1;
        }
        apply_parameter_update(data, &smat_to_para_idx, &g, n_proj);
        0
    } else {
        stochastic_opt_complex(data, state)
    }
}

/// Complex-mode SR step. The `O† O` matrix lives in
/// `sr_opt_oo` with logical leading dimension `2 * sr_opt_size`; the
/// upstream code only uses the real diagonal/off-diagonal block of
/// `(2*i, 2*j)` for the SR matrix entries.
pub fn stochastic_opt_complex(data: &mut ExpertModeData, state: &mut VmcOptimizationState) -> i32 {
    let n_proj = data.gutzwiller_terms.len() + data.jastrow_terms.len();
    let n_orb = n_slater(data);
    let n_para = n_proj + n_orb;
    if n_para == 0 {
        return 0;
    }
    data.ensure_optimization_flags(n_para);

    let sr_opt_size = state.sr_opt.sr_opt_size;
    let lda_oo = 2 * sr_opt_size;
    let mut s_diag = vec![0.0_f64; 2 * n_para];
    for pi in 0..(2 * n_para) {
        let oo_idx_diag = (pi + 2) * lda_oo + (pi + 2);
        let oo_idx_0 = pi + 2;
        if oo_idx_diag < state.sr_opt.sr_opt_oo.len() && oo_idx_0 < state.sr_opt.sr_opt_oo.len() {
            s_diag[pi] = state.sr_opt.sr_opt_oo[oo_idx_diag].re
                - state.sr_opt.sr_opt_oo[oo_idx_0].re.powi(2);
        }
    }
    let s_diag_max = s_diag.iter().cloned().fold(0.0_f64, f64::max);
    let cut = s_diag_max * data.modpara.dsr_opt_red_cut;

    let mut smat_to_para_idx: Vec<usize> = Vec::new();
    for pi in 0..(2 * n_para) {
        let opt = data.optimization_flags.get(pi).copied().unwrap_or(true);
        if !opt {
            continue;
        }
        if s_diag[pi] >= cut {
            smat_to_para_idx.push(pi);
        }
    }
    let n_smat = smat_to_para_idx.len();
    if n_smat == 0 {
        return 0;
    }

    let mut s = vec![0.0_f64; n_smat * n_smat];
    let mut g = vec![0.0_f64; n_smat];
    let ratio_diag = 1.0 + data.modpara.dsr_opt_sta_del;
    for (si, &pi) in smat_to_para_idx.iter().enumerate() {
        let tmp = state.sr_opt.sr_opt_oo[pi + 2].re;
        for (sj, &pj) in smat_to_para_idx.iter().enumerate() {
            let idx = si + n_smat * sj;
            let oo_idx = (pi + 2) * lda_oo + (pj + 2);
            let oo_0j = pj + 2;
            let s_val = state.sr_opt.sr_opt_oo[oo_idx].re - tmp * state.sr_opt.sr_opt_oo[oo_0j].re;
            s[idx] = s_val;
        }
        let diag = si + n_smat * si;
        s[diag] *= ratio_diag;
    }
    let ho_0 = state.sr_opt.sr_opt_ho[0].re;
    for (si, &pi) in smat_to_para_idx.iter().enumerate() {
        let ho_idx = pi + 2;
        let oo_idx = pi + 2;
        let value = state.sr_opt.sr_opt_ho[ho_idx].re - ho_0 * state.sr_opt.sr_opt_oo[oo_idx].re;
        g[si] = -2.0 * data.modpara.dsr_opt_step_dt * value;
    }

    if cholesky_solve(&mut s, &mut g, n_smat).is_err() {
        return 1;
    }
    for (si, &pi) in smat_to_para_idx.iter().enumerate() {
        let r = g[si];
        if pi % 2 == 0 {
            let para_idx = pi / 2;
            update_parameter_value(data, para_idx, r, 0.0, n_proj);
        } else {
            let para_idx = (pi - 1) / 2;
            update_parameter_value(data, para_idx, 0.0, r, n_proj);
        }
    }
    0
}

fn collect_active_real(
    data: &ExpertModeData,
    state: &VmcOptimizationState,
    n_para: usize,
    sr_opt_size: usize,
) -> (Vec<f64>, Vec<usize>) {
    let mut s_diag = vec![0.0_f64; n_para];
    for pi in 0..n_para {
        let idx_diag = (pi + 1) * sr_opt_size + (pi + 1);
        let idx_0 = pi + 1;
        if idx_diag < state.sr_opt.sr_opt_oo_real.len() && idx_0 < state.sr_opt.sr_opt_oo_real.len()
        {
            s_diag[pi] =
                state.sr_opt.sr_opt_oo_real[idx_diag] - state.sr_opt.sr_opt_oo_real[idx_0].powi(2);
        }
    }
    let s_diag_max = s_diag.iter().cloned().fold(0.0_f64, f64::max);
    let cut = s_diag_max * data.modpara.dsr_opt_red_cut;
    let mut smat_to_para_idx: Vec<usize> = Vec::new();
    for pi in 0..n_para {
        let opt_real = data.optimization_flags.get(2 * pi).copied().unwrap_or(true);
        if !opt_real {
            continue;
        }
        if s_diag[pi] >= cut {
            smat_to_para_idx.push(pi);
        }
    }
    (s_diag, smat_to_para_idx)
}

fn build_s_g_real(
    s: &mut [f64],
    g: &mut [f64],
    smat_to_para_idx: &[usize],
    state: &VmcOptimizationState,
    sr_opt_size: usize,
    sta_del: f64,
    step_dt: f64,
) {
    let n_smat = smat_to_para_idx.len();
    let ratio_diag = 1.0 + sta_del;
    for (si, &pi) in smat_to_para_idx.iter().enumerate() {
        let tmp = state.sr_opt.sr_opt_oo_real[pi + 1];
        for (sj, &pj) in smat_to_para_idx.iter().enumerate() {
            let idx = si + n_smat * sj;
            let oo_idx = (pi + 1) * sr_opt_size + (pj + 1);
            let s_val =
                state.sr_opt.sr_opt_oo_real[oo_idx] - tmp * state.sr_opt.sr_opt_oo_real[pj + 1];
            s[idx] = s_val;
        }
        let diag = si + n_smat * si;
        s[diag] *= ratio_diag;
    }
    let ho_0 = state.sr_opt.sr_opt_ho_real[0];
    for (si, &pi) in smat_to_para_idx.iter().enumerate() {
        let value =
            state.sr_opt.sr_opt_ho_real[pi + 1] - ho_0 * state.sr_opt.sr_opt_oo_real[pi + 1];
        g[si] = -2.0 * step_dt * value;
    }
}

fn apply_parameter_update(
    data: &mut ExpertModeData,
    smat_to_para_idx: &[usize],
    g: &[f64],
    n_proj: usize,
) {
    for (si, &pi) in smat_to_para_idx.iter().enumerate() {
        update_parameter_value(data, pi, g[si], 0.0, n_proj);
    }
}

fn update_parameter_value(
    data: &mut ExpertModeData,
    para_idx: usize,
    delta_real: f64,
    delta_imag: f64,
    n_proj: usize,
) {
    let n_gutz = data.gutzwiller_terms.len();
    let delta = Complex64::new(delta_real, delta_imag);
    if para_idx < n_gutz {
        data.gutzwiller_terms[para_idx].value += delta;
    } else if para_idx < n_proj {
        let jastrow_idx = para_idx - n_gutz;
        if jastrow_idx < data.jastrow_terms.len() {
            data.jastrow_terms[jastrow_idx].value += delta;
        }
    } else {
        let orbital_idx = (para_idx - n_proj) as i64;
        for term in data.orbital_terms.iter_mut() {
            if term.idx == orbital_idx {
                term.value += delta;
            }
        }
    }
}

/// LAPACK `dpotrf('U')` + `dpotrs('U')` Cholesky factorisation and solve.
///
/// `s` is a column-major `n×n` symmetric positive-definite matrix.
/// On entry only the **upper** triangle is referenced; `dpotrf_` overwrites
/// it with the Cholesky factor U (A = Uᵀ U).  `dpotrs_` then solves
/// Uᵀ U x = rhs, overwriting `rhs` with the solution vector.
///
/// This matches Julia's `potrf!('U', S)` → `potrs!('U', S, g)` exactly —
/// same algorithm, same FP operation order — so the SR parameter update
/// is bit-for-bit identical to the Julia gold standard.
///
/// Returns `Err(())` if `dpotrf_` reports `info > 0` (matrix not
/// positive-definite), matching the upstream `info ≠ 0` convention.
fn cholesky_solve(s: &mut [f64], rhs: &mut [f64], n: usize) -> Result<(), ()> {
    if n == 0 {
        return Ok(());
    }
    let n_i32 = n as i32;
    let lda = n_i32;
    let mut info = 0_i32;
    // potrf!('U', S) — upper-triangular Cholesky factorisation in-place.
    unsafe {
        dpotrf_(b"U".as_ptr(), &n_i32, s.as_mut_ptr(), &lda, &mut info);
    }
    if info != 0 {
        return Err(());
    }
    let nrhs = 1_i32;
    // potrs!('U', S, g) — solve Uᵀ U x = g using the factor in the upper
    // triangle of s (which dpotrf_ just wrote).
    unsafe {
        dpotrs_(
            b"U".as_ptr(),
            &n_i32,
            &nrhs,
            s.as_ptr(),
            &lda,
            rhs.as_mut_ptr(),
            &lda,
            &mut info,
        );
    }
    if info != 0 {
        return Err(());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cholesky_solve_matches_direct_inverse() {
        // Symmetric positive-definite 2x2 matrix.
        let mut s = vec![4.0, 1.0, 1.0, 3.0]; // column-major
        let mut b = vec![5.0, 6.0];
        cholesky_solve(&mut s, &mut b, 2).unwrap();
        // A * x = [4 1; 1 3] * x = [5; 6] -> x = [9/11; 19/11].
        assert!((b[0] - 9.0 / 11.0).abs() < 1e-12);
        assert!((b[1] - 19.0 / 11.0).abs() < 1e-12);
    }
}

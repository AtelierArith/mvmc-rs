//! Stochastic Reconfiguration optimizer.
//!
//! Port target: `MVMCOptimizers.jl/src/stochastic_opt.jl`.
//!
//! Implementation note (do NOT collapse): the direct solver uses the
//! two-step `dpotrf` -> `dpotrs` form rather than `dposv`. The two-step
//! form mirrors the upstream call sequence. Diagonal stabilization is applied
//! during matrix construction, before factorisation.

#![allow(clippy::needless_range_loop)]

use mvmc_expert_parsers::utils::parameter_init::n_slater;
use mvmc_expert_parsers::ExpertModeData;
use num_complex::Complex64;

use crate::c_timer::CTimer;
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
/// Returns `0` on success and `1` if the solve fails or an update is nonfinite
/// (matches the upstream `info` convention).
pub fn stochastic_opt_real(data: &mut ExpertModeData, state: &mut VmcOptimizationState) -> i32 {
    stochastic_opt_real_timed(data, state, &mut CTimer::<false>::new())
}

/// Time the direct SR preprocessing, matrix construction, solve and parameter update.
pub fn stochastic_opt_real_timed<const TIMED: bool>(
    data: &mut ExpertModeData,
    state: &mut VmcOptimizationState,
    timer: &mut CTimer<TIMED>,
) -> i32 {
    let n_proj = data.projection_layout().n_proj;
    let n_orb = n_slater(data);
    let n_para = n_proj + n_orb;
    if n_para == 0 {
        return 0;
    }
    data.ensure_optimization_flags(n_para);

    let sr_opt_size = state.sr_opt.sr_opt_size;
    if !state.sr_opt.sr_opt_oo_real.is_empty() {
        // Real fast-path: SROptOO_real lives in the dedicated real buffer.
        timer.start(50);
        let (s_diag, smat_to_para_idx) = collect_active_real(data, state, n_para, sr_opt_size);
        timer.stop(50);
        if smat_to_para_idx.is_empty() {
            return 0;
        }
        let n_smat = smat_to_para_idx.len();
        timer.start(51);
        timer.start(56);
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
        timer.stop(56);
        timer.start(57);
        let result = cholesky_solve(&mut s, &mut g, n_smat);
        timer.stop(57);
        timer.stop(51);
        if result.is_err() {
            return 1;
        }
        timer.start(52);
        apply_parameter_update(data, &smat_to_para_idx, &g, n_proj);
        timer.stop(52);
        0
    } else {
        stochastic_opt_complex_timed(data, state, timer)
    }
}

/// Complex-mode SR step. The `O† O` matrix lives in
/// `sr_opt_oo` with logical leading dimension `2 * sr_opt_size`; the
/// upstream code only uses the real diagonal/off-diagonal block of
/// `(2*i, 2*j)` for the SR matrix entries.
pub fn stochastic_opt_complex(data: &mut ExpertModeData, state: &mut VmcOptimizationState) -> i32 {
    stochastic_opt_complex_timed(data, state, &mut CTimer::<false>::new())
}

/// Time the direct SR preprocessing, matrix construction, solve and parameter update.
pub fn stochastic_opt_complex_timed<const TIMED: bool>(
    data: &mut ExpertModeData,
    state: &mut VmcOptimizationState,
    timer: &mut CTimer<TIMED>,
) -> i32 {
    let n_proj = data.projection_layout().n_proj;
    let n_orb = n_slater(data);
    let n_para = n_proj + n_orb;
    if n_para == 0 {
        return 0;
    }
    data.ensure_optimization_flags(n_para);

    timer.start(50);
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
    let s_diag_max = if s_diag.iter().any(|v| v.is_nan()) {
        f64::NAN
    } else {
        s_diag.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    };
    let cut = s_diag_max * data.modpara.dsr_opt_red_cut;

    let mut smat_to_para_idx: Vec<usize> = Vec::new();
    for pi in 0..(2 * n_para) {
        let opt = data.optimization_flags.get(pi).copied().unwrap_or(false);
        if !opt {
            continue;
        }
        if s_diag[pi] < cut {
            continue;
        }
        smat_to_para_idx.push(pi);
    }
    let n_smat = smat_to_para_idx.len();
    timer.stop(50);
    if n_smat == 0 {
        return 0;
    }

    timer.start(51);
    timer.start(56);
    let mut s = vec![0.0_f64; n_smat * n_smat];
    let mut g = vec![0.0_f64; n_smat];
    build_s_g_complex(
        &mut s,
        &mut g,
        &smat_to_para_idx,
        state,
        sr_opt_size,
        data.modpara.dsr_opt_sta_del,
        data.modpara.dsr_opt_step_dt,
    );

    timer.stop(56);
    timer.start(57);
    let result = cholesky_solve(&mut s, &mut g, n_smat);
    timer.stop(57);
    timer.stop(51);
    if result.is_err() {
        return 1;
    }
    timer.start(52);
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
    timer.stop(52);
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
    // Julia's real-to-complex layout includes zero imaginary variances.
    let s_diag_max = if s_diag.iter().any(|v| v.is_nan()) {
        f64::NAN
    } else {
        s_diag.iter().copied().fold(0.0_f64, f64::max)
    };
    let cut = s_diag_max * data.modpara.dsr_opt_red_cut;
    let mut smat_to_para_idx: Vec<usize> = Vec::new();
    for pi in 0..n_para {
        let opt_real = data
            .optimization_flags
            .get(2 * pi)
            .copied()
            .unwrap_or(false);
        if !opt_real {
            continue;
        }
        if s_diag[pi] < cut {
            continue;
        }
        smat_to_para_idx.push(pi);
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

fn build_s_g_complex(
    s: &mut [f64],
    g: &mut [f64],
    smat_to_para_idx: &[usize],
    state: &VmcOptimizationState,
    sr_opt_size: usize,
    sta_del: f64,
    step_dt: f64,
) {
    let n_smat = smat_to_para_idx.len();
    let lda_oo = 2 * sr_opt_size;
    let ratio_diag = 1.0 + sta_del;
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

pub(crate) fn update_parameter_value(
    data: &mut ExpertModeData,
    para_idx: usize,
    delta_real: f64,
    delta_imag: f64,
    n_proj: usize,
) {
    let n_gutz = data.projection_layout().n_gutzwiller;
    let delta = Complex64::new(delta_real, delta_imag);
    if para_idx < n_gutz {
        if let Some(term) = data.gutzwiller_terms.get_mut(para_idx) {
            term.value += delta;
        }
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
/// Julia's `potrf!('U', S)` → `potrs!('U', S, g)` sequence. Exact numerical
/// agreement also requires matching input construction and BLAS kernels;
/// fixed sampled-input and end-to-end gates check those separately.
///
/// Returns `Err(())` on LAPACK failure or any nonfinite solved update,
/// before the caller mutates parameters, matching Julia's post-solve check.
fn cholesky_solve(s: &mut [f64], rhs: &mut [f64], n: usize) -> Result<(), ()> {
    if n == 0 {
        return Ok(());
    }
    crate::serial_blas::initialize();
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
    if rhs.iter().any(|value| !value.is_finite()) {
        return Err(());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mvmc_expert_parsers::OrbitalTerm;

    #[test]
    fn sampled_direct_sr_matrix_gradient_factor_and_solution_match_julia() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        for (case, store) in [
            "real",
            "cmp",
            "fsz",
            "hubbard",
            "pairhop_real",
            "pairhop_fsz",
        ]
        .into_iter()
        .flat_map(|case| [(case, 0), (case, 1)])
        {
            let suffix = if store == 0 {
                "_runner"
            } else {
                "_store_runner"
            };
            let fixture = std::fs::read_to_string(root.join(format!(
                "tests/fixtures/sr_direct/{case}{suffix}/fixed-input.txt"
            )))
            .unwrap();
            let mut lines = fixture.lines().filter(|line| !line.starts_with('#'));
            let sizes: Vec<usize> = lines
                .next()
                .unwrap()
                .split_whitespace()
                .map(|s| s.parse().unwrap())
                .collect();
            let (size, n) = (sizes[0], sizes[1]);
            let mut mapping: Vec<usize> = lines
                .next()
                .unwrap()
                .split_whitespace()
                .map(|s| s.parse().unwrap())
                .collect();
            let read = |line: &str| -> Vec<f64> {
                line.split_whitespace()
                    .map(|s| f64::from_bits(u64::from_str_radix(s, 16).unwrap()))
                    .collect()
            };
            let complex = matches!(case, "cmp" | "fsz" | "pairhop_fsz");
            let mut state = VmcOptimizationState::zeros(1, 1, 0, size - 1, 1, 1, complex, false);
            let oo = read(lines.next().unwrap());
            let ho = read(lines.next().unwrap());
            if complex {
                state.sr_opt.sr_opt_oo = oo
                    .chunks_exact(2)
                    .map(|z| Complex64::new(z[0], z[1]))
                    .collect();
                state.sr_opt.sr_opt_ho = ho
                    .chunks_exact(2)
                    .map(|z| Complex64::new(z[0], z[1]))
                    .collect();
            } else {
                state.sr_opt.sr_opt_oo_real = oo;
                state.sr_opt.sr_opt_ho_real = ho;
                mapping.iter_mut().for_each(|pi| *pi /= 2);
            }
            let name = if case == "hubbard" {
                "hubbard_chain_real".into()
            } else {
                format!("heisenberg_chain_{case}")
            };
            let namelist = if case.starts_with("pairhop_") {
                root.join(format!("extern/Julia-mVMC/test/integration/reference/hubbard_chain_{case}/inputs/namelist.def"))
            } else {
                root.join("extern/Julia-mVMC/examples/inputs")
                    .join(name)
                    .join("namelist.def")
            };
            let data = mvmc_expert_parsers::parse_expert_mode_files(namelist).unwrap();
            let mut s = vec![0.0; n * n];
            let mut g = vec![0.0; n];
            let build = if complex {
                build_s_g_complex
            } else {
                build_s_g_real
            };
            build(
                &mut s,
                &mut g,
                &mapping,
                &state,
                size,
                data.modpara.dsr_opt_sta_del,
                data.modpara.dsr_opt_step_dt,
            );
            let exact = |label: &str, actual: &[f64], expected: &[f64]| {
                assert_eq!(actual.len(), expected.len());
                for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
                    assert_eq!(a.to_bits(), e.to_bits(), "{case} {label}[{i}]: {a} vs {e}");
                }
            };
            exact("matrix", &s, &read(lines.next().unwrap()));
            exact("gradient", &g, &read(lines.next().unwrap()));
            cholesky_solve(&mut s, &mut g, n).unwrap();
            exact("factor", &s, &read(lines.next().unwrap()));
            exact("solution", &g, &read(lines.next().unwrap()));
        }
    }

    fn two_parameter_problem(complex: bool) -> (ExpertModeData, VmcOptimizationState) {
        let mut data = ExpertModeData::new();
        data.modpara.n_orbital_idx = 2;
        data.modpara.dsr_opt_sta_del = 0.0;
        data.optimization_flags = vec![true, false, true, false];
        data.orbital_terms = (0..2)
            .map(|idx| OrbitalTerm {
                site1: 0,
                site2: 0,
                idx,
                value: Complex64::new(2.0 + idx as f64, 0.0),
                is_complex: complex,
                sign: 1,
            })
            .collect();
        let mut state = VmcOptimizationState::zeros(1, 1, 0, 2, 1, 1, complex, false);
        if complex {
            let lda = 6;
            state.sr_opt.sr_opt_oo[2 * lda + 2] = Complex64::new(1.0, 0.0);
            state.sr_opt.sr_opt_oo[4 * lda + 4] = Complex64::new(1.0, 0.0);
            state.sr_opt.sr_opt_ho[2] = Complex64::new(1.0, 0.0);
            state.sr_opt.sr_opt_ho[4] = Complex64::new(f64::NAN, 0.0);
        } else {
            state.sr_opt.sr_opt_oo_real[4] = 1.0;
            state.sr_opt.sr_opt_oo_real[8] = 1.0;
            state.sr_opt.sr_opt_ho_real[1] = 1.0;
            state.sr_opt.sr_opt_ho_real[2] = f64::NAN;
        }
        (data, state)
    }

    #[test]
    fn real_nonfinite_update_preserves_all_parameters() {
        let (mut data, mut state) = two_parameter_problem(false);
        let before = data.orbital_terms.clone();
        assert_eq!(stochastic_opt_real(&mut data, &mut state), 1);
        assert_eq!(data.orbital_terms, before);
    }

    #[test]
    fn complex_nonfinite_update_preserves_all_parameters() {
        let (mut data, mut state) = two_parameter_problem(true);
        let before = data.orbital_terms.clone();
        assert_eq!(stochastic_opt_complex(&mut data, &mut state), 1);
        assert_eq!(data.orbital_terms, before);
    }

    #[test]
    fn nonfinite_variance_is_solved_and_rejected_without_dropping_its_parameter() {
        for complex in [false, true] {
            let (mut data, mut state) = two_parameter_problem(complex);
            if complex {
                state.sr_opt.sr_opt_oo[2 * 6 + 2] = Complex64::new(f64::NAN, 0.0);
                state.sr_opt.sr_opt_ho[4] = Complex64::new(1.0, 0.0);
            } else {
                state.sr_opt.sr_opt_oo_real[4] = f64::NAN;
                state.sr_opt.sr_opt_ho_real[2] = 1.0;
            }
            let before = data.orbital_terms.clone();
            let solve = if complex {
                stochastic_opt_complex
            } else {
                stochastic_opt_real
            };
            assert_eq!(solve(&mut data, &mut state), 1);
            assert_eq!(data.orbital_terms, before);
        }
    }

    #[test]
    fn incomplete_flags_keep_missing_components_fixed() {
        for complex in [false, true] {
            let (mut data, mut state) = two_parameter_problem(complex);
            data.optimization_flags = vec![true, false];
            let before = data.orbital_terms[1].value;
            let solve = if complex {
                stochastic_opt_complex
            } else {
                stochastic_opt_real
            };
            assert_eq!(solve(&mut data, &mut state), 0);
            assert_eq!(data.orbital_terms[1].value, before);
        }
    }

    #[test]
    fn failed_factorization_preserves_all_parameters() {
        let (mut data, mut state) = two_parameter_problem(false);
        // Positive diagonals keep both active, but the covariance is indefinite.
        state.sr_opt.sr_opt_oo_real[5] = 2.0;
        state.sr_opt.sr_opt_oo_real[7] = 2.0;
        state.sr_opt.sr_opt_ho_real[2] = 1.0;
        let before = data.orbital_terms.clone();
        assert_eq!(stochastic_opt_real(&mut data, &mut state), 1);
        assert_eq!(data.orbital_terms, before);
    }

    #[test]
    fn fixed_correlations_remain_unchanged_across_sr_steps_and_sync() {
        use mvmc_expert_parsers::{GutzwillerTerm, JastrowTerm};
        let mut data = ExpertModeData::new();
        data.gutzwiller_terms.push(GutzwillerTerm {
            site: 0,
            value: Complex64::new(2.0, 0.3),
            is_complex: false,
        });
        data.jastrow_terms.push(JastrowTerm {
            site1: 0,
            site2: 1,
            value: Complex64::new(5.0, 0.2),
            is_complex: false,
        });
        data.optimization_flags = vec![false, false, true, false];
        let before_g = data.gutzwiller_terms.clone();
        let before_j = data.jastrow_terms[0].value;
        let mut state = VmcOptimizationState::zeros(2, 1, 2, 2, 1, 1, false, false);
        state.sr_opt.sr_opt_oo_real[4] = 1.0;
        state.sr_opt.sr_opt_oo_real[8] = 1.0;
        state.sr_opt.sr_opt_ho_real[1] = 10.0;
        state.sr_opt.sr_opt_ho_real[2] = 1.0;
        for _ in 0..3 {
            assert_eq!(stochastic_opt_real(&mut data, &mut state), 0);
            crate::sync::sync_modified_parameter(&mut data, &crate::SingleProcessReducer);
            assert_eq!(data.gutzwiller_terms, before_g);
            assert_eq!(data.jastrow_terms[0].value.im, before_j.im);
        }
        // Julia's identity covariance gives -2*dt*HO=-0.02 per step.
        let mut expected = before_j.re;
        for _ in 0..3 {
            expected += -0.02;
        }
        assert_eq!(data.jastrow_terms[0].value.re, expected);
    }

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

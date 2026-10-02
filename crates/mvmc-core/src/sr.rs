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
    let n_para = data.count_variational_parameters();
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
    let n_para = data.count_variational_parameters();
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

#[cfg(test)]
mod opttrans_tests {
    use super::*;
    use std::path::PathBuf;

    fn root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/opttrans")
    }

    fn model(name: &str) -> ExpertModeData {
        let base = if matches!(name, "short_opt" | "long_opt" | "empty_opt") {
            "layout"
        } else {
            name
        };
        let mut data = mvmc_expert_parsers::parse_expert_mode_files(
            root().join(format!("namelist_{base}.def")),
        )
        .unwrap();
        match name {
            "empty_opt" => data.opt_trans.clear(),
            "short_opt" => data.opt_trans.truncate(1),
            "long_opt" => data.opt_trans.push(Complex64::new(0.5, -0.25)),
            _ => {}
        }
        let n = data.projection_layout().n_proj
            + data.count_rbm_parameters()
            + n_slater(&data)
            + data.count_opt_trans_parameters();
        let n_proj = data.projection_layout().n_proj;
        for i in 0..n {
            update_parameter_value(
                &mut data,
                i,
                (i + 1) as f64 / 64.0,
                -((i + 1) as f64) / 128.0,
                n_proj,
            );
        }
        data.modpara.nmp_trans = 2;
        data.para_qp_trans = vec![Complex64::new(1.0, 0.25), Complex64::new(-0.5, -0.125)];
        crate::qp::init_qp_weight(&mut data);
        data
    }

    fn values(data: &mut ExpertModeData) -> Vec<Complex64> {
        let mut out = data.projection_parameters();
        data.visit_rbm_terms_mut(|_, t| out.push(t.value()));
        out.extend(data.orbital_terms.iter().map(|t| t.value));
        out.extend(data.opt_trans.iter().copied());
        out
    }

    fn bits(values: impl IntoIterator<Item = Complex64>, expected: &str, label: &str) {
        let actual: Vec<_> = values
            .into_iter()
            .flat_map(|v| [v.re.to_bits(), v.im.to_bits()])
            .collect();
        let expected: Vec<_> = expected
            .split_whitespace()
            .map(|s| u64::from_str_radix(s, 16).unwrap())
            .collect();
        assert_eq!(actual.len(), expected.len(), "{label}: width");
        for (i, (a, e)) in actual.into_iter().zip(expected).enumerate() {
            assert_eq!(a, e, "{label}: component {i}");
        }
    }

    #[test]
    fn opttrans_direct_and_cg_updates_match_julia_component_flags_and_offsets() {
        let text = std::fs::read_to_string(root().join("sr.txt")).unwrap();
        let mut lines = text.lines().filter(|l| !l.starts_with('#'));
        let mut count = 0;
        while let Some(header) = lines.next() {
            let fields: Vec<_> = header.split_whitespace().collect();
            let mut data = model(fields[0]);
            let complex = fields[1] == "1";
            let cg = fields[2] == "cg";
            let n: usize = fields[4].parse().unwrap();
            let no = data.opt_trans.len();
            data.modpara.complex_flag = i64::from(complex);
            data.complex_flags = vec![i64::from(complex)];
            for t in &mut data.orbital_terms {
                t.is_complex = complex;
            }
            for t in &mut data.gutzwiller_terms {
                t.is_complex = complex;
            }
            for t in &mut data.jastrow_terms {
                t.is_complex = complex;
            }
            data.doublon_holon_2site_complex = complex;
            data.doublon_holon_4site_complex = complex;
            data.modpara.dsr_opt_sta_del = 0.0;
            data.modpara.dsr_opt_step_dt = 0.125;
            data.modpara.dsr_opt_red_cut = 1e-8;
            data.modpara.nvmc_sample = 8;
            data.optimization_flags = vec![false; 2 * n];
            for i in n - no..n {
                data.optimization_flags[2 * i] = matches!(fields[3], "real" | "both");
                data.optimization_flags[2 * i + 1] = matches!(fields[3], "imag" | "both");
            }
            let mut state = VmcOptimizationState::zeros(
                data.modpara.nsite as usize,
                data.modpara.nelec.max(1) as usize,
                data.projection_layout().n_proj,
                n,
                1,
                8,
                complex,
                false,
            );
            state.energy.wc = Complex64::new(8.0, 0.0);
            let off = if complex { 2 } else { 1 };
            let size = off * (n + 1);
            for p in 0..off * no {
                let idx = off * (n - no + 1) + p;
                let diag = if cg { size + idx } else { idx * size + idx };
                if complex {
                    state.sr_opt.sr_opt_oo[diag] = Complex64::new(1.0, 0.0);
                    state.sr_opt.sr_opt_ho[idx] = Complex64::new((p + 1) as f64 / 16.0, 0.0);
                } else {
                    state.sr_opt.sr_opt_oo_real[diag] = 1.0;
                    state.sr_opt.sr_opt_ho_real[idx] = (p + 1) as f64 / 16.0;
                }
                for s in 0_usize..8 {
                    let val = if (s & (p + 1)).count_ones() % 2 == 1 {
                        -1.0
                    } else {
                        1.0
                    };
                    if complex {
                        state.sr_opt.sr_opt_o_store[s * size + idx] = Complex64::new(val, 0.0);
                    } else {
                        state.sr_opt.sr_opt_o_store_real[s * size + idx] = val;
                    }
                }
            }
            let info = if cg {
                crate::sr_cg::stochastic_opt_cg(&mut data, &state, None).unwrap()
            } else if complex {
                stochastic_opt_complex(&mut data, &mut state)
            } else {
                stochastic_opt_real(&mut data, &mut state)
            };
            assert_eq!(info, fields[5].parse::<i32>().unwrap(), "{header}");
            bits(values(&mut data), lines.next().unwrap(), header);
            bits(
                data.qp_weights
                    .as_ref()
                    .unwrap()
                    .qp_full_weight
                    .iter()
                    .copied(),
                lines.next().unwrap(),
                header,
            );
            bits(data.para_qp_opt_trans, lines.next().unwrap(), header);
            count += 1;
        }
        assert_eq!(count, 96);
    }

    #[test]
    fn opttrans_optimizer_normalization_and_weight_refresh_match_julia() {
        let text = std::fs::read_to_string(root().join("sync.txt")).unwrap();
        let mut lines = text.lines().filter(|l| !l.starts_with('#'));
        let mut count = 0;
        while let Some(header) = lines.next() {
            let f: Vec<_> = header.split_whitespace().collect();
            let mut data = model(f[0]);
            data.opt_trans = match f[1] {
                "1" => vec![],
                "2" => vec![Complex64::new(0.0, 0.0), Complex64::new(-0.0, -0.0)],
                "3" => vec![Complex64::new(3.0, 4.0), Complex64::new(-6.0, 8.0)],
                "4" => vec![Complex64::new(0.125, -0.75), Complex64::new(1.5, -0.5)],
                "5" => vec![
                    Complex64::new(1e-300, 2e-300),
                    Complex64::new(-3e-300, 4e-300),
                ],
                "6" => vec![Complex64::new(1e300, 2e300), Complex64::new(-3e300, 4e300)],
                _ => panic!("unknown vector: {header}"),
            };
            crate::qp::update_qp_weight_for(&mut data);
            if f[3] == "parser" {
                mvmc_expert_parsers::utils::parameter_init::sync_modified_parameter(
                    &mut data, false,
                );
            } else {
                crate::sync::sync_modified_parameter_local(&mut data, f[2] == "1");
            }
            bits(values(&mut data), lines.next().unwrap(), header);
            bits(
                data.qp_weights
                    .as_ref()
                    .unwrap()
                    .qp_full_weight
                    .iter()
                    .copied(),
                lines.next().unwrap(),
                header,
            );
            bits(
                data.para_qp_opt_trans.iter().copied(),
                lines.next().unwrap(),
                header,
            );
            crate::qp::update_qp_weight_for(&mut data);
            bits(
                data.qp_weights
                    .as_ref()
                    .unwrap()
                    .qp_full_weight
                    .iter()
                    .copied(),
                lines.next().unwrap(),
                header,
            );
            count += 1;
        }
        assert_eq!(count, 72);
    }
}

pub(crate) fn update_parameter_value(
    data: &mut ExpertModeData,
    para_idx: usize,
    delta_real: f64,
    delta_imag: f64,
    n_proj: usize,
) {
    let layout = data.projection_layout();
    let n_gutz = layout.n_gutzwiller;
    let delta = Complex64::new(delta_real, delta_imag);
    if para_idx < n_gutz {
        if let Some(term) = data.gutzwiller_terms.get_mut(para_idx) {
            term.value += delta;
        }
    } else if para_idx < layout.dh2_offset {
        let jastrow_idx = para_idx - layout.jastrow_offset;
        if jastrow_idx < data.jastrow_terms.len() {
            data.jastrow_terms[jastrow_idx].value += delta;
        }
    } else if para_idx < layout.dh4_offset {
        if let Some(value) = data
            .doublon_holon_2site_params
            .get_mut(para_idx - layout.dh2_offset)
        {
            *value += delta;
        }
    } else if para_idx < n_proj {
        if let Some(value) = data
            .doublon_holon_4site_params
            .get_mut(para_idx - layout.dh4_offset)
        {
            *value += delta;
        }
    } else if para_idx < n_proj + data.count_rbm_parameters() {
        let sizes = data.rbm_section_sizes();
        let mut offsets = [n_proj; 9];
        for i in 1..9 {
            offsets[i] = offsets[i - 1] + sizes[i - 1];
        }
        data.visit_rbm_terms_mut(|section, term| {
            if term.idx() >= 0
                && (term.idx() as usize) < sizes[section]
                && offsets[section] + term.idx() as usize == para_idx
            {
                term.set_value(term.value() + delta);
            }
        });
    } else if para_idx < n_proj + data.count_rbm_parameters() + n_slater(data) {
        let orbital_idx = (para_idx - n_proj - data.count_rbm_parameters()) as i64;
        for term in data.orbital_terms.iter_mut() {
            if term.idx == orbital_idx {
                term.value += delta;
            }
        }
    } else {
        let offset = n_proj + data.count_rbm_parameters() + n_slater(data);
        if let Some(value) = data.opt_trans.get_mut(para_idx - offset) {
            *value += delta;
            crate::qp::update_qp_weight_for(data);
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
/// Returns `Err(())` on an illegal LAPACK argument or nonfinite solved update,
/// before parameter mutation. Julia's `potrf!` returns positive INFO without
/// throwing, and canonical SR discards that status before calling `potrs!`.
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
    if info < 0 {
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
            "dh2_real",
            "dh2_cmp",
            "dh2_fsz",
            "dh4_real",
            "dh4_cmp",
            "dh4_fsz",
            "dh24_real",
            "dh24_cmp",
            "dh24_fsz",
            "rbm_real",
            "rbm_cmp",
            "rbm_general_cmp",
            "rbm_dh24_cmp",
            "rbm_fsz",
            "rbm_reference_cmp",
            "opt_real",
            "opt_cmp",
            "opt_fsz",
            "opt_dh24_rbm_cmp",
        ]
        .into_iter()
        .flat_map(|case| {
            if case == "rbm_reference_cmp" {
                vec![(case, 1)]
            } else {
                vec![(case, 0), (case, 1)]
            }
        }) {
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
            let complex = matches!(
                case,
                "cmp"
                    | "fsz"
                    | "pairhop_fsz"
                    | "dh2_cmp"
                    | "dh2_fsz"
                    | "dh4_cmp"
                    | "dh4_fsz"
                    | "dh24_cmp"
                    | "dh24_fsz"
                    | "rbm_cmp"
                    | "rbm_general_cmp"
                    | "rbm_dh24_cmp"
                    | "rbm_fsz"
                    | "rbm_reference_cmp"
                    | "opt_cmp"
                    | "opt_fsz"
                    | "opt_dh24_rbm_cmp"
            );
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
            let namelist = if case == "rbm_reference_cmp" {
                root.join("extern/Julia-mVMC/test/integration/reference/general_rbm_cmp/inputs/namelist.def")
            } else if case.starts_with("opt_") {
                root.join(format!("tests/fixtures/opttrans/run_{case}/namelist.def"))
            } else if case.starts_with("rbm_") {
                root.join(format!("tests/fixtures/rbm/run_{case}/namelist.def"))
            } else if let Some(mode) = case.strip_prefix("dh2_") {
                root.join(format!("tests/fixtures/dh2/production_{mode}/namelist.def"))
            } else if case.starts_with("dh4_") || case.starts_with("dh24_") {
                root.join(format!("tests/fixtures/dh4/production_{case}/namelist.def"))
            } else if case.starts_with("pairhop_") {
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

    #[test]
    fn rbm_indexed_updates_and_sparse_values_match_original_julia() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/rbm/production");
        let text = std::fs::read_to_string(root.join("updates.txt")).unwrap();
        let mut lines = text.lines().filter(|line| !line.starts_with('#'));
        let bits = |values: &[Complex64]| {
            values
                .iter()
                .flat_map(|z| [z.re.to_bits(), z.im.to_bits()])
                .collect::<Vec<_>>()
        };
        let expected = |line: &str| {
            line.split_whitespace()
                .map(|word| u64::from_str_radix(word, 16).unwrap())
                .collect::<Vec<_>>()
        };
        while let Some(case) = lines.next() {
            let file = root.join(format!("namelist_{case}.def"));
            let mut data = mvmc_expert_parsers::parse_expert_mode_files(&file).unwrap();
            mvmc_expert_parsers::utils::read_input_parameters::read_input_parameters(
                &mut data, &file,
            )
            .unwrap();
            let counts: Vec<usize> = lines
                .next()
                .unwrap()
                .split_whitespace()
                .map(|s| s.parse().unwrap())
                .collect();
            let nproj = data.projection_layout().n_proj;
            let nrbm = data.count_rbm_parameters();
            let npara = nproj + nrbm + mvmc_expert_parsers::utils::parameter_init::n_slater(&data);
            assert_eq!([nproj, nrbm, npara], counts.as_slice(), "{case}");
            let snapshot = |d: &mut ExpertModeData| {
                let mut values = d.projection_parameters();
                d.visit_rbm_terms_mut(|_, t| values.push(t.value()));
                values.extend(d.orbital_terms.iter().map(|t| t.value));
                bits(&values)
            };
            assert_eq!(
                snapshot(&mut data),
                expected(lines.next().unwrap()),
                "{case} before"
            );
            assert_eq!(
                bits(&data.rbm_parameters()),
                expected(lines.next().unwrap()),
                "{case} packed before"
            );
            for index in 0..npara {
                update_parameter_value(
                    &mut data,
                    index,
                    (index + 1) as f64 / 128.0,
                    -((index + 1) as f64) / 256.0,
                    nproj,
                );
            }
            assert_eq!(
                snapshot(&mut data),
                expected(lines.next().unwrap()),
                "{case} after"
            );
            assert_eq!(
                bits(&data.rbm_parameters()),
                expected(lines.next().unwrap()),
                "{case} packed after"
            );
        }
    }

    #[test]
    fn rbm_parameter_updates_hit_all_shared_rows_and_leave_slater_at_final_offset() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/rbm/namelist_tied.def");
        let mut data = mvmc_expert_parsers::parse_expert_mode_files(path).unwrap();
        let n_proj = data.projection_layout().n_proj;
        let n_rbm = data.count_rbm_parameters();
        update_parameter_value(&mut data, n_proj, 0.5, -0.25, n_proj);
        assert_eq!(
            data.charge_rbm_phys_layer_terms[0].value,
            Complex64::new(0.5, -0.25)
        );
        assert_eq!(
            data.charge_rbm_phys_layer_terms[1].value,
            Complex64::new(0.5, -0.25)
        );
        assert!(data
            .orbital_terms
            .iter()
            .all(|t| t.value == Complex64::new(0.0, 0.0)));
        update_parameter_value(&mut data, n_proj + n_rbm + 1, -0.25, 0.5, n_proj);
        assert_eq!(data.orbital_terms[1].value, Complex64::new(-0.25, 0.5));
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
    fn positive_potrf_status_and_finite_check_match_canonical_julia() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/sr_failure/potrf_status.txt");
        let text = std::fs::read_to_string(path).unwrap();
        let mut lines = text.lines().filter(|l| !l.starts_with('#'));
        let mut cases = 0;
        while let Some(header) = lines.next() {
            let fields: Vec<_> = header.split_whitespace().collect();
            let complex = fields[0] == "1";
            let (mut data, mut state) = two_parameter_problem(complex);
            let covariance = if fields[1] == "indefinite" { 2.0 } else { 1.0 };
            let off = if complex { 2 } else { 1 };
            let size = off * 3;
            if complex {
                state.sr_opt.sr_opt_oo[off * size + 2 * off] = Complex64::new(covariance, 0.0);
                state.sr_opt.sr_opt_oo[2 * off * size + off] = Complex64::new(covariance, 0.0);
                state.sr_opt.sr_opt_ho[2 * off] = Complex64::new(1.0, 0.0);
            } else {
                state.sr_opt.sr_opt_oo_real[off * size + 2 * off] = covariance;
                state.sr_opt.sr_opt_oo_real[2 * off * size + off] = covariance;
                state.sr_opt.sr_opt_ho_real[2 * off] = 1.0;
            }
            let solve = if complex {
                stochastic_opt_complex
            } else {
                stochastic_opt_real
            };
            assert_eq!(
                solve(&mut data, &mut state),
                fields[2].parse::<i32>().unwrap(),
                "{header}"
            );
            let actual: Vec<_> = data
                .orbital_terms
                .iter()
                .flat_map(|t| [t.value.re.to_bits(), t.value.im.to_bits()])
                .collect();
            let expected: Vec<_> = lines
                .next()
                .unwrap()
                .split_whitespace()
                .map(|s| u64::from_str_radix(s, 16).unwrap())
                .collect();
            assert_eq!(actual, expected, "{header}");
            cases += 1;
        }
        assert_eq!(cases, 4);
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

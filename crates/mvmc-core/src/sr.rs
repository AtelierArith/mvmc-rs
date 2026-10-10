//! Stochastic Reconfiguration optimizer.
//!
//! Port target: `MVMCOptimizers.jl/src/stochastic_opt.jl`.
//!
//! Implementation note (do NOT collapse): the direct solver uses the
//! two-step `dpotrf` -> `dpotrs` form rather than `dposv`. The two-step
//! form mirrors the upstream call sequence. Diagonal stabilization is applied
//! during matrix construction, before factorisation.

#![allow(clippy::needless_range_loop)]

#[path = "sr_observer.rs"]
pub mod observer;

use mvmc_expert_parsers::utils::parameter_init::n_slater;
use mvmc_expert_parsers::ExpertModeData;
use num_complex::Complex64;

use crate::c_timer::CTimer;
use crate::output_files::SrInfoRow;
use crate::state::VmcOptimizationState;

/// Select the actual component flag, as C's stcOptInit does. Native
/// GetInfoOptTrans writes are already represented in optimization_flags;
/// remapping them here would activate components C never selects.
pub(crate) fn component_is_optimized(data: &ExpertModeData, component: usize) -> bool {
    data.optimization_flags.get(component).copied() == Some(1)
}

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
    stochastic_opt_real_with_sr_info_timed(data, state, timer, &mut None)
}

/// Direct real SR that also reports the C `_SRinfo.dat` row (`stcopt.c:157`).
///
/// `sr_info` is set only when a system is solved (C reaches the print after the
/// solve, before the finite check), and is left untouched otherwise.
pub fn stochastic_opt_real_with_sr_info_timed<const TIMED: bool>(
    data: &mut ExpertModeData,
    state: &mut VmcOptimizationState,
    timer: &mut CTimer<TIMED>,
    sr_info: &mut Option<SrInfoRow>,
) -> i32 {
    let n_proj = data.projection_layout().n_proj;
    let n_para = data.count_variational_parameters();
    if n_para == 0 {
        observer::not_solved(
            data,
            observer::DirectMode::Real,
            observer::NotSolvedReason::NoParameters,
        );
        return 0;
    }
    data.ensure_optimization_flags(n_para);

    let sr_opt_size = state.sr_opt.sr_opt_size;
    if !state.sr_opt.sr_opt_oo_real.is_empty() {
        if state.sr_oo_deferred() {
            return stochastic_opt_real_resident(data, state, timer, sr_info, n_para, n_proj);
        }
        // Real fast-path: SROptOO_real lives in the dedicated real buffer.
        timer.start(50);
        let (s_diag, smat_to_para_idx) = collect_active_real(data, state, n_para, sr_opt_size);
        timer.stop(50);
        if smat_to_para_idx.is_empty() {
            observer::not_solved(
                data,
                observer::DirectMode::Real,
                observer::NotSolvedReason::NoActiveComponents,
            );
            return 0;
        }
        let n_smat = smat_to_para_idx.len();
        timer.start(51);
        timer.start(56);
        let mut s = vec![0.0_f64; n_smat * n_smat];
        let mut g = vec![0.0_f64; n_smat];
        // One backend handle per SR solve (issue #421): C order by default.
        let mut backend = crate::stage_backend::acquire();
        backend
            .sr()
            .assemble_s_g(
                &crate::sr_backend::SrAssembleInput {
                    oo: crate::sr_backend::RealView::Real(&state.sr_opt.sr_opt_oo_real),
                    ho: crate::sr_backend::RealView::Real(&state.sr_opt.sr_opt_ho_real),
                    map: &smat_to_para_idx,
                    ld: sr_opt_size,
                    offset: 1,
                    sta_del: data.modpara.dsr_opt_sta_del,
                    step_dt: data.modpara.dsr_opt_step_dt,
                },
                &mut s,
                &mut g,
            )
            .expect("SR S/g assembly backend failed");
        timer.stop(56);
        timer.start(57);
        let observation =
            observer::before_solve(data, &s, &g, &smat_to_para_idx, observer::DirectMode::Real);
        let result = backend.sr().cholesky_solve(&mut s, &mut g, n_smat);
        observer::after_solve(observation, &g, &result);
        timer.stop(57);
        timer.stop(51);
        if result.is_err() {
            return 1;
        }
        *sr_info = Some(direct_real_sr_info(
            data,
            &s_diag,
            &smat_to_para_idx,
            &g,
            n_para,
        ));
        timer.start(52);
        apply_parameter_update(data, &smat_to_para_idx, &g, n_proj);
        timer.stop(52);
        0
    } else {
        stochastic_opt_complex_with_sr_info_timed(data, state, timer, sr_info)
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
    stochastic_opt_complex_with_sr_info_timed(data, state, timer, &mut None)
}

/// Direct complex SR that also reports the C `_SRinfo.dat` row (`stcopt.c:157`).
pub fn stochastic_opt_complex_with_sr_info_timed<const TIMED: bool>(
    data: &mut ExpertModeData,
    state: &mut VmcOptimizationState,
    timer: &mut CTimer<TIMED>,
    sr_info: &mut Option<SrInfoRow>,
) -> i32 {
    let n_proj = data.projection_layout().n_proj;
    let n_para = data.count_variational_parameters();
    if n_para == 0 {
        observer::not_solved(
            data,
            observer::DirectMode::Complex,
            observer::NotSolvedReason::NoParameters,
        );
        return 0;
    }
    data.ensure_optimization_flags(n_para);

    timer.start(50);
    let sr_opt_size = state.sr_opt.sr_opt_size;
    let lda_oo = 2 * sr_opt_size;
    let mut s_diag = vec![0.0_f64; 2 * n_para];
    // C stcopt.c:83 `omp parallel for` over `pi`: one producer per element.
    crate::threading::for_each_mut(&mut s_diag, 3, |pi, value| {
        let oo_idx_diag = (pi + 2) * lda_oo + (pi + 2);
        let oo_idx_0 = pi + 2;
        if oo_idx_diag < state.sr_opt.sr_opt_oo.len() && oo_idx_0 < state.sr_opt.sr_opt_oo.len() {
            *value = state.sr_opt.sr_opt_oo[oo_idx_diag].re
                - state.sr_opt.sr_opt_oo[oo_idx_0].re.powi(2);
        }
    });
    let s_diag_max = if s_diag.iter().any(|v| v.is_nan()) {
        f64::NAN
    } else {
        s_diag.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    };
    let cut = s_diag_max * data.modpara.dsr_opt_red_cut;

    let mut smat_to_para_idx: Vec<usize> = Vec::new();
    for pi in 0..(2 * n_para) {
        let opt = component_is_optimized(data, pi);
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
        observer::not_solved(
            data,
            observer::DirectMode::Complex,
            observer::NotSolvedReason::NoActiveComponents,
        );
        return 0;
    }

    timer.start(51);
    timer.start(56);
    let mut s = vec![0.0_f64; n_smat * n_smat];
    let mut g = vec![0.0_f64; n_smat];
    let mut backend = crate::stage_backend::acquire();
    backend
        .sr()
        .assemble_s_g(
            &crate::sr_backend::SrAssembleInput {
                oo: crate::sr_backend::RealView::ReOfComplex(&state.sr_opt.sr_opt_oo),
                ho: crate::sr_backend::RealView::ReOfComplex(&state.sr_opt.sr_opt_ho),
                map: &smat_to_para_idx,
                ld: lda_oo,
                offset: 2,
                sta_del: data.modpara.dsr_opt_sta_del,
                step_dt: data.modpara.dsr_opt_step_dt,
            },
            &mut s,
            &mut g,
        )
        .expect("SR S/g assembly backend failed");

    timer.stop(56);
    timer.start(57);
    let observation = observer::before_solve(
        data,
        &s,
        &g,
        &smat_to_para_idx,
        observer::DirectMode::Complex,
    );
    let result = backend.sr().cholesky_solve(&mut s, &mut g, n_smat);
    observer::after_solve(observation, &g, &result);
    timer.stop(57);
    timer.stop(51);
    if result.is_err() {
        return 1;
    }
    *sr_info = Some(direct_complex_sr_info(
        data,
        &s_diag,
        &smat_to_para_idx,
        &g,
        n_para,
    ));
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

/// C's search for the element of largest magnitude (strict `<` keeps the first).
fn largest_magnitude(solution: &[f64]) -> (f64, usize) {
    let mut imax = 0;
    for i in 0..solution.len() {
        if solution[imax].abs() < solution[i].abs() {
            imax = i;
        }
    }
    (solution[imax], imax)
}

/// Row for the complex layout: C index space is `0..2*NPara` (`stcopt.c:60-157`).
fn direct_complex_sr_info(
    data: &ExpertModeData,
    s_diag: &[f64],
    smat_to_para_idx: &[usize],
    solution: &[f64],
    n_para: usize,
) -> SrInfoRow {
    let s_max = s_diag
        .iter()
        .copied()
        .fold(s_diag[0], |m, v| if v > m { v } else { m });
    let s_min = s_diag
        .iter()
        .copied()
        .fold(s_diag[0], |m, v| if v < m { v } else { m });
    let cut = s_max * data.modpara.dsr_opt_red_cut;
    let (mut opt_num, mut cut_num) = (0, 0);
    for (pi, &diag) in s_diag.iter().enumerate() {
        if !component_is_optimized(data, pi) {
            opt_num += 1;
        } else if diag < cut {
            cut_num += 1;
        }
    }
    let (r_max, imax) = largest_magnitude(solution);
    SrInfoRow {
        n_para: n_para as i64,
        n_smat: smat_to_para_idx.len() as i64,
        opt_num,
        cut_num,
        s_diag_max: s_max,
        s_diag_min: s_min,
        r_max,
        i_max: smat_to_para_idx[imax] as i64,
        cg_info: None,
    }
}

/// Row for the real fast path. C expands the real matrix into the complex
/// layout (index `2*pi` real, `2*pi+1` imaginary with zero variance).
fn direct_real_sr_info(
    data: &ExpertModeData,
    s_diag: &[f64],
    smat_to_para_idx: &[usize],
    solution: &[f64],
    n_para: usize,
) -> SrInfoRow {
    let expanded: Vec<f64> = s_diag.iter().flat_map(|&v| [v, 0.0]).collect();
    let doubled: Vec<usize> = smat_to_para_idx.iter().map(|&pi| 2 * pi).collect();
    direct_complex_sr_info(data, &expanded, &doubled, solution, n_para)
}

fn collect_active_real(
    data: &ExpertModeData,
    state: &VmcOptimizationState,
    n_para: usize,
    sr_opt_size: usize,
) -> (Vec<f64>, Vec<usize>) {
    let mut s_diag = vec![0.0_f64; n_para];
    crate::threading::for_each_mut(&mut s_diag, 3, |pi, value| {
        let idx_diag = (pi + 1) * sr_opt_size + (pi + 1);
        let idx_0 = pi + 1;
        if idx_diag < state.sr_opt.sr_opt_oo_real.len() && idx_0 < state.sr_opt.sr_opt_oo_real.len()
        {
            *value =
                state.sr_opt.sr_opt_oo_real[idx_diag] - state.sr_opt.sr_opt_oo_real[idx_0].powi(2);
        }
    });
    select_active_real(data, n_para, s_diag)
}

/// The active-component selection of the real direct SR (`stcopt.c` redundancy cut) from the
/// component variances `s_diag`.
fn select_active_real(
    data: &ExpertModeData,
    n_para: usize,
    s_diag: Vec<f64>,
) -> (Vec<f64>, Vec<usize>) {
    // Julia's real-to-complex layout includes zero imaginary variances.
    let s_diag_max = if s_diag.iter().any(|v| v.is_nan()) {
        f64::NAN
    } else {
        s_diag.iter().copied().fold(0.0_f64, f64::max)
    };
    let cut = s_diag_max * data.modpara.dsr_opt_red_cut;
    let mut smat_to_para_idx: Vec<usize> = Vec::new();
    for pi in 0..n_para {
        let opt_real = component_is_optimized(data, 2 * pi);
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

/// Whether the real direct SR step of this measurement is routed to a resident SR backend
/// (issue #452): direct (not CG) real SR with a saved-sample store, one process, and a selected
/// backend whose SR stages report [`crate::sr_backend::SrStages::resident_direct`]. Then the host
/// `OO` is never materialized (`run.rs` skips the Gram) and [`stochastic_opt_real_resident`]
/// drives the step. Everything else keeps the per-stage path on the selected backend.
pub(crate) fn resident_direct_real_applies<R: crate::reducer::Reducer + ?Sized>(
    data: &ExpertModeData,
    n_vmc_sample: usize,
    sr_opt_size: usize,
    use_store: bool,
    reducer: &R,
) -> bool {
    data.modpara.nsrcg == 0
        && use_store
        && n_vmc_sample > 0
        && sr_opt_size > 0
        && reducer.reduction_size() == 1
        && !reducer.supports_grouped_sampling()
        && crate::stage_backend::selected_resident_direct()
}

/// Real direct SR on a resident backend (issue #452): the O store is uploaded once, the Gram,
/// `S`, `g` and the Cholesky factor stay on the device, and only `x` comes back. The host sees
/// the Gram diagonal and first column (for the redundancy cut) and nothing else; the formulas
/// and the `1 / wc` normalization are those of the per-stage path.
fn stochastic_opt_real_resident<const TIMED: bool>(
    data: &mut ExpertModeData,
    state: &mut VmcOptimizationState,
    timer: &mut CTimer<TIMED>,
    sr_info: &mut Option<SrInfoRow>,
    n_para: usize,
    n_proj: usize,
) -> i32 {
    use crate::sr_backend::DirectSolveInput;
    let n = state.sr_opt.sr_opt_size;
    let samples = data.modpara.nvmc_sample.max(0) as usize;
    // `weight_average_sr_opt_real` is a no-op below this weight; the same guard here.
    let wc = state.energy.wc;
    let gram_scale = if wc.norm() < 1.0e-15 {
        1.0
    } else {
        1.0 / wc.re
    };
    let mut backend = crate::stage_backend::acquire();
    let sr = backend.sr();
    timer.start(51);
    timer.start(56);
    let summary = sr
        .direct_begin(&state.sr_opt.sr_opt_o_store_real[..n * samples], n, samples)
        .expect("resident SR Gram stage failed");
    timer.start(50);
    let mut s_diag = vec![0.0_f64; n_para];
    crate::threading::for_each_mut(&mut s_diag, 3, |pi, value| {
        // the normalized `OO(pi+1, pi+1) - OO(pi+1, 0)^2` of `collect_active_real`
        let diagonal = summary.diag[pi + 1] * gram_scale;
        let mean = summary.col0[pi + 1] * gram_scale;
        *value = diagonal - mean.powi(2);
    });
    let (s_diag, smat_to_para_idx) = select_active_real(data, n_para, s_diag);
    timer.stop(50);
    if smat_to_para_idx.is_empty() {
        timer.stop(56);
        timer.stop(51);
        observer::not_solved(
            data,
            observer::DirectMode::Real,
            observer::NotSolvedReason::NoActiveComponents,
        );
        return 0;
    }
    let n_smat = smat_to_para_idx.len();
    sr.direct_assemble(&DirectSolveInput {
        ho: &state.sr_opt.sr_opt_ho_real,
        map: &smat_to_para_idx,
        offset: 1,
        sta_del: data.modpara.dsr_opt_sta_del,
        step_dt: data.modpara.dsr_opt_step_dt,
        gram_scale,
    })
    .expect("resident SR S/g assembly failed");
    timer.stop(56);
    timer.start(57);
    let observation = if observer::is_enabled() {
        let (s, g) = sr
            .direct_download_s_g(n_smat)
            .expect("resident SR S/g download failed");
        observer::before_solve(data, &s, &g, &smat_to_para_idx, observer::DirectMode::Real)
    } else {
        None
    };
    let solved = sr.direct_factor_solve(n_smat);
    let status: Result<(), ()> = solved.as_ref().map(|_| ()).map_err(|_| ());
    let g = solved.unwrap_or_default();
    observer::after_solve(observation, &g, &status);
    timer.stop(57);
    timer.stop(51);
    if status.is_err() {
        return 1;
    }
    *sr_info = Some(direct_real_sr_info(
        data,
        &s_diag,
        &smat_to_para_idx,
        &g,
        n_para,
    ));
    timer.start(52);
    apply_parameter_update(data, &smat_to_para_idx, &g, n_proj);
    timer.stop(52);
    0
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

    #[test]
    fn c_opttrans_flags_are_not_remapped_to_parameter_tail() {
        let mut data = ExpertModeData::new();
        data.n_gutzwiller_idx = 3;
        data.opt_trans = vec![Complex64::new(0.5, 0.0), Complex64::new(0.75, 0.0)];
        data.c_opt_trans_flags = true;
        data.optimization_flags = vec![0, 0, 0, 1, 1, 0, 0];
        assert!(component_is_optimized(&data, 3));
        assert!(component_is_optimized(&data, 4));
        assert!(!component_is_optimized(&data, 6));
        assert!(!component_is_optimized(&data, 7));
        assert!(!component_is_optimized(&data, 8));
        assert!(!component_is_optimized(&data, 9));
    }

    fn model(name: &str) -> ExpertModeData {
        let base = if matches!(name, "short_opt" | "long_opt" | "empty_opt") {
            "layout"
        } else {
            name
        };
        let mut data = crate::historical_orbital_model::historical_kernel_model(
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
        out.extend(
            data.orbital_terms
                .iter()
                .map(|t| data.slater_params[t.idx as usize]),
        );
        out.extend(data.opt_trans.iter().copied());
        out
    }

    fn compare_values(values: impl IntoIterator<Item = Complex64>, expected: &str, label: &str) {
        // These synthetic update/normalization systems are small and regularized;
        // budget 64 rounding errors, including the parameter update and QP sync.
        let actual: Vec<_> = values.into_iter().flat_map(|v| [v.re, v.im]).collect();
        let expected = crate::numerical_comparison::hex_values(expected);
        assert_eq!(actual.len(), expected.len(), "{label}: parameter width");
        for (index, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
            // The range fixtures include 1e-300 coefficients. An O(epsilon)
            // absolute floor would incorrectly accept losing them entirely.
            let absolute = if expected != 0.0 && expected.abs() < f64::MIN_POSITIVE.sqrt() {
                4.0 * f64::from_bits(1)
            } else {
                64.0 * f64::EPSILON
            };
            crate::numerical_comparison::assert_close(
                actual,
                expected,
                absolute,
                64.0 * f64::EPSILON,
                format!("{label}: parameter component {index}"),
            );
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
            // This fixture explicitly replaces the loaded declarations with
            // its synthetic real/complex SR system. Retire both source-bound
            // snapshots, not the guard or the independent numerical oracle.
            data.native_complex_headers.clear();
            data.native_complex_declarations.clear();
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
            assert_eq!(crate::run::get_all_complex_flag(&data).unwrap(), complex);
            data.modpara.dsr_opt_sta_del = 0.0;
            data.modpara.dsr_opt_step_dt = 0.125;
            data.modpara.dsr_opt_red_cut = 1e-8;
            data.modpara.nvmc_sample = 8;
            data.optimization_flags = vec![0; 2 * n];
            for i in n - no..n {
                data.optimization_flags[2 * i] = i64::from(matches!(fields[3], "real" | "both"));
                data.optimization_flags[2 * i + 1] =
                    i64::from(matches!(fields[3], "imag" | "both"));
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
            state.validate_declared_mode(&data).unwrap();
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
            compare_values(values(&mut data), lines.next().unwrap(), header);
            compare_values(
                data.qp_weights
                    .as_ref()
                    .unwrap()
                    .qp_full_weight
                    .iter()
                    .copied(),
                lines.next().unwrap(),
                header,
            );
            compare_values(data.para_qp_opt_trans, lines.next().unwrap(), header);
            count += 1;
        }
        assert_eq!(count, 96);
    }

    #[test]
    fn opttrans_sync_matches_julia_with_declared_slater_normalization_from_c() {
        let text = std::fs::read_to_string(root().join("sync.txt")).unwrap();
        let mut lines = text.lines().filter(|l| !l.starts_with('#'));
        let mut count = 0;
        while let Some(header) = lines.next() {
            let f: Vec<_> = header.split_whitespace().collect();
            let mut data = model(f[0]);
            let c_rows: Vec<_> =
                include_str!("../../../tests/fixtures/orbital_general/c_declared_flags.txt")
                    .lines()
                    .filter(|line| !line.starts_with('#'))
                    .collect();
            let c_sync = c_rows
                .as_chunks::<4>()
                .0
                .iter()
                .find(|row| row[0] == "layout_sync")
                .unwrap();
            if n_slater(&data) == 4 {
                compare_values(data.slater_params.iter().copied(), c_sync[1], header);
            }
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
            let historical = lines.next().unwrap();
            let actual = values(&mut data);
            if n_slater(&data) == 4 {
                // The C array includes slots 2/3, which model() also updates.
                // Slot 3 is the normalization maximum. Keep all historical
                // non-Slater values and replace Slater expectations with C values.
                compare_values(data.slater_params.iter().copied(), c_sync[2], header);
                let mut expected: Vec<_> = historical.split_whitespace().collect();
                let c_bits: Vec<_> = c_sync[2].split_whitespace().collect();
                let offset = 2 * (actual.len() - data.opt_trans.len() - data.orbital_terms.len());
                for (row, term) in data.orbital_terms.iter().enumerate() {
                    let idx = term.idx as usize;
                    expected[offset + 2 * row] = c_bits[2 * idx];
                    expected[offset + 2 * row + 1] = c_bits[2 * idx + 1];
                }
                compare_values(actual, &expected.join(" "), header);
            } else {
                compare_values(actual, historical, header);
            }
            compare_values(
                data.qp_weights
                    .as_ref()
                    .unwrap()
                    .qp_full_weight
                    .iter()
                    .copied(),
                lines.next().unwrap(),
                header,
            );
            compare_values(
                data.para_qp_opt_trans.iter().copied(),
                lines.next().unwrap(),
                header,
            );
            crate::qp::update_qp_weight_for(&mut data);
            compare_values(
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
        let index = para_idx - n_proj;
        data.set_rbm_parameter(index, data.rbm_params[index] + delta);
    } else if para_idx < n_proj + data.count_rbm_parameters() + n_slater(data) {
        let orbital_idx = para_idx - n_proj - data.count_rbm_parameters();
        if let Some(value) = data.slater_params.get_mut(orbital_idx) {
            *value += delta;
        }
    } else {
        let offset = n_proj + data.count_rbm_parameters() + n_slater(data);
        if let Some(value) = data.opt_trans.get_mut(para_idx - offset) {
            *value += delta;
            crate::qp::update_qp_weight_for(data);
        }
    }
}

#[cfg(test)]
mod original_parameter_delta_tests {
    use super::update_parameter_value;
    use crate::pack_parameters;
    use num_complex::Complex64;
    use std::path::Path;

    #[test]
    fn original_m0581_additive_helper_updates_first_and_last_parameters() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/original_heisenberg_parser_184/namelist.def");
        let mut data = mvmc_expert_parsers::parse_expert_mode_files(path).unwrap();
        assert!(data.input_errors.is_empty(), "{:?}", data.input_errors);
        let n = data.count_variational_parameters();
        assert!(n > 1);
        let n_proj = data.projection_layout().n_proj;
        // Julia's original first/last ordinals (1,n) become Rust (0,n-1).
        // Invoke the actual SR additive helper, not get/set as a replacement.
        for index in [0, n - 1] {
            let before = pack_parameters(&data).unwrap()[index];
            // Original parsed first projection and last declared Slater are zero;
            // independent literal endpoint expectations supplement the original
            // before+delta assertion, without initialization or an RNG draw.
            assert_eq!(before, Complex64::new(0.0, 0.0));
            update_parameter_value(&mut data, index, 0.25, -0.5, n_proj);
            let after = pack_parameters(&data).unwrap()[index];
            assert_eq!(after, before + Complex64::new(0.25, -0.5));
            assert_eq!(after, Complex64::new(0.25, -0.5));
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
/// Returns `Err(())` on failed factorization/substitution or nonfinite update,
/// before parameter mutation. C's stcopt_dposv.c uses DPOSV, which does not
/// substitute when POTRF returns positive INFO (a non-positive-definite matrix).
pub(crate) fn cholesky_solve(s: &mut [f64], rhs: &mut [f64], n: usize) -> Result<(), ()> {
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
    observer::lapack_status(true, info);
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
    observer::lapack_status(false, info);
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
    mod historical_overlay_stage {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/support/historical_overlay_stage.rs"
        ));
    }
    use super::*;
    use crate::julia_fixture;
    use mvmc_expert_parsers::OrbitalTerm;

    #[test]
    fn rbm_sr_updates_preserve_unmapped_declared_slots_and_following_slater_offset() {
        use mvmc_expert_parsers::ChargeRBMPhysLayerTerm;
        let mut data = ExpertModeData::new();
        data.n_gutzwiller_idx = 2;
        data.rbm_section_widths = [97, 5, 0, 0, 0, 0, 0, 0, 0];
        data.rbm_params = vec![Complex64::new(0.0, 0.0); 102];
        data.modpara.n_orbital_idx = 4;
        data.slater_params = vec![Complex64::new(0.0, 0.0); 4];
        data.charge_rbm_phys_layer_terms = [0, 2, 0]
            .into_iter()
            .enumerate()
            .map(|(site, idx)| ChargeRBMPhysLayerTerm {
                site: site as i64,
                idx,
                value: Complex64::new(0.0, 0.0),
                is_complex: true,
            })
            .collect();
        for index in [1, 96, 97, 101] {
            update_parameter_value(&mut data, 2 + index, 0.125, -0.25, 2);
            assert_eq!(data.rbm_params[index], Complex64::new(0.125, -0.25));
        }
        assert!(data
            .charge_rbm_phys_layer_terms
            .iter()
            .all(|term| term.value == Complex64::new(0.0, 0.0)));
        update_parameter_value(&mut data, 2, 0.5, -0.125, 2);
        assert_eq!(data.rbm_params[0], Complex64::new(0.5, -0.125));
        assert!(data
            .charge_rbm_phys_layer_terms
            .iter()
            .filter(|term| term.idx == 0)
            .all(|term| term.value == data.rbm_params[0]));
        update_parameter_value(&mut data, 104, 0.25, 0.5, 2);
        assert_eq!(data.slater_params[0], Complex64::new(0.25, 0.5));
        assert_eq!(data.rbm_params[101], Complex64::new(0.125, -0.25));
    }

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
            let fixtures = root.join("tests/fixtures");
            let archived = format!("sr_direct/{case}{suffix}/fixed-input.txt");
            let fixture = julia_fixture::read_text(fixtures.join(archived)).unwrap();
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
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|z| Complex64::new(z[0], z[1]))
                    .collect();
                state.sr_opt.sr_opt_ho = ho
                    .as_chunks::<2>()
                    .0
                    .iter()
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
            let data = crate::historical_orbital_model::historical_kernel_model(namelist).unwrap();
            let mut s = vec![0.0; n * n];
            let mut g = vec![0.0; n];
            let (oo, ho, ld, offset) = if complex {
                (
                    crate::sr_backend::RealView::ReOfComplex(&state.sr_opt.sr_opt_oo),
                    crate::sr_backend::RealView::ReOfComplex(&state.sr_opt.sr_opt_ho),
                    2 * size,
                    2,
                )
            } else {
                (
                    crate::sr_backend::RealView::Real(&state.sr_opt.sr_opt_oo_real),
                    crate::sr_backend::RealView::Real(&state.sr_opt.sr_opt_ho_real),
                    size,
                    1,
                )
            };
            crate::sr_backend::SrStages::assemble_s_g(
                &mut crate::sr_backend::COrderSr::default(),
                &crate::sr_backend::SrAssembleInput {
                    oo,
                    ho,
                    map: &mapping,
                    ld,
                    offset,
                    sta_del: data.modpara.dsr_opt_sta_del,
                    step_dt: data.modpara.dsr_opt_step_dt,
                },
                &mut s,
                &mut g,
            )
            .unwrap();
            let compare = |label: &str, actual: &[f64], expected: &[f64]| {
                // Matrix assembly and Cholesky use at most O(n squared) small
                // reductions for these fixed, regularized inputs.
                let bound = 32.0 * (n * n) as f64 * f64::EPSILON;
                crate::numerical_comparison::assert_values_close(
                    actual.iter().copied(),
                    expected.iter().copied(),
                    bound,
                    bound,
                    format!("{case} {label}"),
                );
            };
            compare("matrix", &s, &read(lines.next().unwrap()));
            compare("gradient", &g, &read(lines.next().unwrap()));
            let matrix = s.clone();
            let rhs = g.clone();
            cholesky_solve(&mut s, &mut g, n).unwrap();
            let mut expected_factor = read(lines.next().unwrap());
            let mut expected_solution = read(lines.next().unwrap());
            if let Some(replay) = julia_fixture::arm_directory(&fixtures)
                .map(|dir| {
                    dir.join(format!(
                        "sr_direct_fixed/{case}{suffix}/factor-solution.txt"
                    ))
                })
                .filter(|path| julia_fixture::exists(path))
            {
                let text = julia_fixture::read_text(replay).unwrap();
                let mut rows = text.lines().filter(|line| !line.starts_with('#'));
                expected_factor = read(rows.next().unwrap());
                expected_solution = read(rows.next().unwrap());
                assert!(rows.next().is_none());
            }
            compare("factor", &s, &expected_factor);
            if julia_fixture::kernel_class(&fixtures) == julia_fixture::KernelClass::Reference {
                compare("solution", &g, &expected_solution);
            } else {
                // Other kernels round the factorization and substitutions differently
                // (the factor above still agrees at the strict bound; the first
                // divergence is the solve, e.g. real/store0 solution[1] 1.8e-11 on
                // OpenBLAS Sandybridge against a 7.1e-13 bound).
                // Componentwise (Skeel) forward bound of a backward-stable solve:
                //   |dx_i| <= 4 n eps (|S^-1| (|S| |x| + |b|))_i
                // (first-order constant about 3). S^-1 columns come from the original
                // matrix, independent of the stored solution. Observed worst error /
                // bound over all 49 cases: 4e-3 Sandybridge, 8e-3 Nehalem, 4e-3
                // Prescott. The reference-kernel bound above is unchanged.
                let mut skeel = vec![0.0; n];
                for j in 0..n {
                    let mut work = matrix.clone();
                    let mut unit = vec![0.0; n];
                    unit[j] = 1.0;
                    cholesky_solve(&mut work, &mut unit, n).unwrap();
                    // t_j = (|S| |x| + |b|)_j; `unit` is column j of S^-1.
                    let t: f64 = (0..n)
                        .map(|k| matrix[j + n * k].abs() * expected_solution[k].abs())
                        .sum::<f64>()
                        + rhs[j].abs();
                    for (acc, v) in skeel.iter_mut().zip(&unit) {
                        *acc += v.abs() * t;
                    }
                }
                let mut worst_ratio = 0.0_f64;
                for (i, (actual, expected)) in g.iter().zip(&expected_solution).enumerate() {
                    let bound = 4.0 * n as f64 * f64::EPSILON * skeel[i];
                    let error = (actual - expected).abs();
                    worst_ratio = worst_ratio.max(error / bound.max(f64::MIN_POSITIVE));
                    assert!(
                        error <= bound,
                        "{case} solution[{i}]: forward error {error:e} exceeds the Skeel bound {bound:e}"
                    );
                }
                eprintln!("{case} direct solution: worst error / Skeel bound {worst_ratio:.3e}");
            }
            // A forward comparison alone can accept a bad solution to an
            // ill-conditioned system. Independently check backward error
            // using the original, unfactored covariance and gradient.
            let matrix_norm = (0..n)
                .map(|i| (0..n).map(|j| matrix[i + n * j].abs()).sum::<f64>())
                .fold(0.0, f64::max);
            let solution_norm = g.iter().map(|x| x.abs()).fold(0.0, f64::max);
            let rhs_norm = rhs.iter().map(|x| x.abs()).fold(0.0, f64::max);
            let residual = (0..n)
                .map(|i| ((0..n).map(|j| matrix[i + n * j] * g[j]).sum::<f64>() - rhs[i]).abs())
                .fold(0.0, f64::max);
            let denominator = matrix_norm * solution_norm + rhs_norm;
            assert!(
                residual <= 128.0 * n as f64 * f64::EPSILON * denominator,
                "{case}: Cholesky backward error {residual:e}, scale {denominator:e}"
            );
        }
    }

    #[test]
    fn rbm_indexed_updates_and_sparse_values_match_original_julia() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/rbm/production");
        let text = std::fs::read_to_string(root.join("updates.txt")).unwrap();
        let mut lines = text.lines().filter(|line| !line.starts_with('#'));
        let components =
            |values: &[Complex64]| values.iter().flat_map(|z| [z.re, z.im]).collect::<Vec<_>>();
        let expected = |line: &str| {
            line.split_whitespace()
                .map(|word| f64::from_bits(u64::from_str_radix(word, 16).unwrap()))
                .collect::<Vec<_>>()
        };
        // The archived Julia packed view discarded every unmapped slot.
        // Compare that mapped-only view here; the separate native parameter
        // fixtures and SR storage test assert complete canonical storage.
        let mapped_values = |data: &mut ExpertModeData| {
            let sizes = data.rbm_section_sizes();
            let mut offsets = [0; 9];
            for i in 1..9 {
                offsets[i] = offsets[i - 1] + sizes[i - 1];
            }
            let mut values = vec![Complex64::new(0.0, 0.0); data.count_rbm_parameters()];
            data.visit_rbm_terms_mut(|section, term| {
                values[offsets[section] + term.idx() as usize] = term.value();
            });
            components(&values)
        };
        while let Some(case) = lines.next() {
            let file = root.join(format!("namelist_{case}.def"));
            let mut data = crate::historical_orbital_model::historical_kernel_model(&file).unwrap();
            historical_overlay_stage::read_input_parameters(&mut data, &file).unwrap();
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
                values.extend(
                    d.orbital_terms
                        .iter()
                        .map(|t| d.slater_params[t.idx as usize]),
                );
                components(&values)
            };
            crate::numerical_comparison::assert_values_close(
                snapshot(&mut data),
                expected(lines.next().unwrap()),
                32.0 * f64::EPSILON,
                32.0 * f64::EPSILON,
                format!("{case} before"),
            );
            crate::numerical_comparison::assert_values_close(
                mapped_values(&mut data),
                expected(lines.next().unwrap()),
                32.0 * f64::EPSILON,
                32.0 * f64::EPSILON,
                format!("{case} packed before"),
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
            crate::numerical_comparison::assert_values_close(
                snapshot(&mut data),
                expected(lines.next().unwrap()),
                32.0 * f64::EPSILON,
                32.0 * f64::EPSILON,
                format!("{case} after"),
            );
            crate::numerical_comparison::assert_values_close(
                mapped_values(&mut data),
                expected(lines.next().unwrap()),
                32.0 * f64::EPSILON,
                32.0 * f64::EPSILON,
                format!("{case} packed after"),
            );
        }
    }

    #[test]
    fn rbm_parameter_updates_hit_all_shared_rows_and_leave_slater_at_final_offset() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/rbm/namelist_tied.def");
        let mut data = crate::historical_orbital_model::historical_kernel_model(path).unwrap();
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
            .slater_params
            .iter()
            .all(|&v| v == Complex64::new(0.0, 0.0)));
        update_parameter_value(&mut data, n_proj + n_rbm + 1, -0.25, 0.5, n_proj);
        assert_eq!(
            data.slater_params[data.orbital_terms[1].idx as usize],
            Complex64::new(-0.25, 0.5)
        );
    }

    fn two_parameter_problem(complex: bool) -> (ExpertModeData, VmcOptimizationState) {
        let mut data = ExpertModeData::new();
        data.modpara.n_orbital_idx = 2;
        data.slater_params = vec![Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0)];
        data.modpara.dsr_opt_sta_del = 0.0;
        data.optimization_flags = vec![1, 0, 1, 0];
        data.orbital_terms = (0..2)
            .map(|idx| OrbitalTerm {
                site1: 0,
                site2: 0,
                idx,
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
    fn sr_updates_declared_unmapped_slots_and_shared_mappings_only_once() {
        for complex in [false, true] {
            let (mut data, mut state) = two_parameter_problem(complex);
            data.modpara.dsr_opt_step_dt = 0.125;
            data.orbital_terms.truncate(1);
            data.orbital_terms.push(data.orbital_terms[0]);
            if complex {
                state.sr_opt.sr_opt_ho[4] = Complex64::new(0.5, 0.0);
            } else {
                state.sr_opt.sr_opt_ho_real[2] = 0.5;
            }
            let mappings = data.orbital_terms.clone();
            let result = if complex {
                stochastic_opt_complex(&mut data, &mut state)
            } else {
                stochastic_opt_real(&mut data, &mut state)
            };
            assert_eq!(result, 0);
            assert_eq!(
                data.slater_params,
                [Complex64::new(1.75, 0.0), Complex64::new(2.875, 0.0)]
            );
            assert_eq!(data.orbital_terms, mappings);
        }
    }

    #[test]
    fn real_nonfinite_update_preserves_all_parameters() {
        let (mut data, mut state) = two_parameter_problem(false);
        let before = data.slater_params.clone();
        assert_eq!(stochastic_opt_real(&mut data, &mut state), 1);
        assert_eq!(data.slater_params, before);
    }

    #[test]
    fn complex_nonfinite_update_preserves_all_parameters() {
        let (mut data, mut state) = two_parameter_problem(true);
        let before = data.slater_params.clone();
        assert_eq!(stochastic_opt_complex(&mut data, &mut state), 1);
        assert_eq!(data.slater_params, before);
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
            let before = data.slater_params.clone();
            let solve = if complex {
                stochastic_opt_complex
            } else {
                stochastic_opt_real
            };
            assert_eq!(solve(&mut data, &mut state), 1);
            assert_eq!(data.slater_params, before);
        }
    }

    #[test]
    fn incomplete_flags_keep_missing_components_fixed() {
        for complex in [false, true] {
            let (mut data, mut state) = two_parameter_problem(complex);
            data.optimization_flags = vec![1, 0];
            let before = data.slater_params[data.orbital_terms[1].idx as usize];
            let solve = if complex {
                stochastic_opt_complex
            } else {
                stochastic_opt_real
            };
            assert_eq!(solve(&mut data, &mut state), 0);
            assert_eq!(
                data.slater_params[data.orbital_terms[1].idx as usize],
                before
            );
        }
    }

    #[test]
    fn c_positive_factor_status_rejects_historical_julia_finite_bad_updates() {
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
            let before = data.slater_params.clone();
            let guard = observer::capture().unwrap();
            assert_eq!(solve(&mut data, &mut state), 1, "{header}");
            let records = guard.finish();
            assert_eq!(records.len(), 1);
            assert!(records[0].factor_info.unwrap() > 0);
            assert_eq!(records[0].solve_info, None);
            assert_eq!(records[0].increment, records[0].rhs);
            assert_eq!(data.slater_params, before);
            let actual: Vec<_> = data
                .slater_params
                .iter()
                .flat_map(|t| [t.re, t.im])
                .collect();
            let expected: Vec<_> = lines
                .next()
                .unwrap()
                .split_whitespace()
                .map(|s| f64::from_bits(u64::from_str_radix(s, 16).unwrap()))
                .collect();
            // Preserve and inspect the archived unsafe Julia behavior, rather
            // than adopting its finite substitution as a C-compatible oracle.
            if fields[1] == "indefinite" {
                assert_eq!(fields[2], "0");
                assert!(expected.iter().all(|value| value.is_finite()));
                assert_ne!(actual, expected, "historical bad update {header}");
            } else {
                assert_eq!(fields[2], "1");
                assert_eq!(actual, expected, "historical unchanged parameters {header}");
            }
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
        data.optimization_flags = vec![0, 0, 1, 0];
        // Explicit step and shift: the C defaults are 0.02/0.02, not 0.01/0.
        data.modpara.dsr_opt_step_dt = 0.01;
        data.modpara.dsr_opt_sta_del = 0.0;
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
        crate::numerical_comparison::assert_close(
            data.jastrow_terms[0].value.re,
            expected,
            16.0 * f64::EPSILON,
            16.0 * f64::EPSILON,
            "three identity-covariance updates",
        );
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

    #[test]
    fn non_positive_definite_factorization_preserves_nonzero_rhs() {
        // diag(1,-1): the second leading principal minor is not positive.
        // Independent DPOSV contract: INFO=2, with no substitution of b.
        let mut matrix = [1.0, 0.0, 0.0, -1.0];
        let mut rhs = [3.25_f64, -7.5_f64];
        let before = rhs.map(f64::to_bits);
        assert_eq!(cholesky_solve(&mut matrix, &mut rhs, 2), Err(()));
        assert_eq!(rhs.map(f64::to_bits), before);
    }

    #[test]
    fn complex_indefinite_direct_sr_rejects_finite_bad_update() {
        // Independent symmetric system [[1,2],[2,1]], with eigenvalues 3,-1.
        // Historical Julia could substitute through the incomplete factor and
        // accept a finite but invalid update; C DPOSV stops with INFO=2.
        let mut data = ExpertModeData::new();
        data.modpara.n_orbital_idx = 2;
        data.modpara.dsr_opt_red_cut = 0.0;
        data.modpara.dsr_opt_sta_del = 0.0;
        data.modpara.dsr_opt_step_dt = 0.5;
        data.slater_params = vec![Complex64::new(3.0, 0.25); 2];
        data.optimization_flags = vec![1, 0, 1, 0];
        let before = data.slater_params.clone();
        let mut state = VmcOptimizationState::zeros(1, 1, 0, 2, 1, 2, true, false);
        let lda = 2 * state.sr_opt.sr_opt_size;
        for (row, col, value) in [(2, 2, 1.0), (2, 4, 2.0), (4, 2, 2.0), (4, 4, 1.0)] {
            state.sr_opt.sr_opt_oo[col * lda + row] = Complex64::new(value, 0.0);
        }
        state.sr_opt.sr_opt_ho[2] = Complex64::new(1.0, 0.0);
        state.sr_opt.sr_opt_ho[4] = Complex64::new(2.0, 0.0);
        let guard = observer::capture().unwrap();
        assert_eq!(stochastic_opt_complex(&mut data, &mut state), 1);
        assert_eq!(data.slater_params, before);
        let records = guard.finish();
        assert_eq!(records.len(), 1);
        let record = &records[0];
        assert_eq!(record.matrix, [1.0, 2.0, 2.0, 1.0]);
        assert_eq!(record.rhs, [-1.0, -2.0]);
        assert_eq!(record.factor_info, Some(2));
        assert_eq!(record.solve_info, None);
        assert_eq!(record.status, Some(1));
        assert_eq!(
            record
                .increment
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            record.rhs.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn singular_direct_sr_stops_before_substitution_and_parameter_mutation() {
        // Independent singular system: zero covariance and zero gradient.
        // DPOSV's positive POTRF INFO must bypass POTRS, not produce NaNs.
        for complex in [false, true] {
            let mut data = ExpertModeData::new();
            data.modpara.n_orbital_idx = 2;
            data.modpara.dsr_opt_red_cut = 0.0;
            data.slater_params = vec![Complex64::new(3.0, 0.0); 2];
            data.optimization_flags = vec![1, 0, 1, 0];
            let before = data.slater_params.clone();
            let mut state = VmcOptimizationState::zeros(1, 1, 0, 2, 1, 2, complex, false);
            let guard = observer::capture().unwrap();
            let status = if complex {
                stochastic_opt_complex(&mut data, &mut state)
            } else {
                stochastic_opt_real(&mut data, &mut state)
            };
            let records = guard.finish();
            assert_eq!(status, 1);
            assert_eq!(data.slater_params, before);
            assert_eq!(records.len(), 1);
            let record = &records[0];
            assert_eq!(record.dimension, 2);
            assert_eq!(record.factor_info, Some(1));
            assert_eq!(record.solve_info, None);
            assert_eq!(record.status, Some(1));
            assert_eq!(record.increment, record.rhs);
            assert!(record.increment.iter().all(|value| value.is_finite()));
        }
    }
}

#[cfg(test)]
#[path = "sr_negative_stepdt_tests.rs"]
mod negative_stepdt_tests;

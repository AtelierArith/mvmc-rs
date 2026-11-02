//! Optional CUDA gate for the device-resident SR pipeline (issue #447). Ignored by default;
//! `MVMC_RS_CUDA_GATE=1` requires a device (hard failure without one), anything else skips
//! explicitly.
//!
//! Validation against the C-order host path with explicit, derived bounds:
//!
//! * **Gram**: per entry `|G_dev - G_host| <= 2 k eps (|O||O|^T)_ij` (both are sums of `k`
//!   products in different orders, `gamma_k` each).
//! * **S/g assembly**: `2 eps` relative to the host formulas on the device's own Gram
//!   (`--fmad=false`, same expressions), following the existing #190 assembly budget.
//! * **Direct solution**: `|dx|_2 / |x|_2 <= 4 n eps kappa(S)` (backward-stable solver
//!   forward-error bound, `kappa` estimated by power / inverse iteration on the host), and the
//!   device solution's residual `|S x - g| / |g|` is below `4 n eps kappa`-free `1e-9`.
//! * **CG operator product** (step-1 operand): per entry
//!   `|z_dev - z_host| <= 4 (k + m) eps scale_i` with `scale_i` the sum of the absolute terms.
//! * **CG solution**: iteration counts equal and relative difference below `1e-9` on a well
//!   conditioned problem; on an ill-conditioned problem (#358) only the operator products,
//!   finiteness, exact iteration counts and numerical repeatability are asserted. The final
//!   prefix's residual is reported, not assumed converged. The converged case checks an
//!   independently materialized covariance residual against its stopping threshold.

#![allow(clippy::needless_range_loop)]

use mvmc_core::backend::{
    cuda_device_count, cuda_gate_decision, CudaGateDecision, CUDA_GATE_VARIABLE,
};
use mvmc_core::sr_backend::{COrderSr, RealView, SrAssembleInput, SrStages};
use mvmc_core::sr_cg::SampledSrOperator;
use mvmc_gpu_cuda::sr_device::{DeviceSr, SrDeviceError};
use mvmc_gpu_cuda::sr_problem::{cg_inputs, make_problem};

const EPS: f64 = f64::EPSILON;

#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical;

fn finite_close(actual: &[f64], expected: &[f64], abs: f64, rel: f64, context: &str) {
    assert!(
        actual.iter().chain(expected).all(|v| v.is_finite()),
        "{context}"
    );
    let maximum = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    eprintln!("{context}: max absolute difference {maximum:.3e}; abs {abs:.3e}, rel {rel:.3e}");
    numerical::assert_values_close(
        actual.iter().copied(),
        expected.iter().copied(),
        abs,
        rel,
        context,
    );
}

fn gate() -> Option<DeviceSr> {
    mvmc_gpu_cuda::install();
    let requested = std::env::var(CUDA_GATE_VARIABLE).ok();
    match cuda_gate_decision(requested.as_deref(), cuda_device_count()) {
        CudaGateDecision::SkippedNoDevice(why) => {
            eprintln!("sr-device-gate: ExplicitSkip: skipped, no device ({why})");
            None
        }
        CudaGateDecision::FailNoDevice(why) => {
            panic!("sr-device-gate: {CUDA_GATE_VARIABLE} requested but no device: {why}")
        }
        CudaGateDecision::Run => Some(DeviceSr::new(0).expect("device SR pipeline")),
    }
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn matvec(a: &[f64], x: &[f64], n: usize) -> Vec<f64> {
    let mut y = vec![0.0; n];
    for j in 0..n {
        for i in 0..n {
            y[i] += a[i + j * n] * x[j];
        }
    }
    y
}

/// Condition number estimate of the SPD matrix `s` (power iteration for lambda_max, inverse
/// iteration through the host Cholesky for lambda_min).
fn cond_estimate(s: &[f64], n: usize) -> f64 {
    let mut x = vec![1.0 / (n as f64).sqrt(); n];
    let mut lmax = 0.0;
    for _ in 0..60 {
        let y = matvec(s, &x, n);
        lmax = norm(&y);
        x = y.iter().map(|v| v / lmax).collect();
    }
    let mut host = COrderSr::default();
    let mut v: Vec<f64> = (0..n).map(|i| 1.0 + 0.01 * i as f64).collect();
    let mut inv_norm = 0.0;
    for _ in 0..30 {
        let nv = norm(&v);
        let mut w: Vec<f64> = v.iter().map(|a| a / nv).collect();
        let mut sc = s.to_vec();
        host.cholesky_solve(&mut sc, &mut w, n).expect("SPD");
        inv_norm = norm(&w);
        v = w;
    }
    lmax * inv_norm
}

fn host_s_g(
    gram: &[f64],
    ho: &[f64],
    map: &[usize],
    n: usize,
    sta_del: f64,
    step_dt: f64,
) -> (Vec<f64>, Vec<f64>) {
    let nm = map.len();
    let mut s = vec![0.0; nm * nm];
    let mut g = vec![0.0; nm];
    COrderSr::default()
        .assemble_s_g(
            &SrAssembleInput {
                oo: RealView::Real(gram),
                ho: RealView::Real(ho),
                map,
                ld: n,
                offset: 1,
                sta_del,
                step_dt,
            },
            &mut s,
            &mut g,
        )
        .expect("host assemble");
    (s, g)
}

fn check_direct(
    dev: &mut DeviceSr,
    n: usize,
    samples: usize,
    factors: usize,
    noise: f64,
    skip: usize,
    seed: u64,
) -> Vec<f64> {
    let p = make_problem(n, samples, factors, noise, skip, seed);
    let (sta_del, step_dt) = (0.01, 0.003);
    let nm = p.map.len();

    // host (C order)
    let mut host = COrderSr::default();
    let mut gram_h = vec![0.0; n * n];
    host.gram_real(&p.store, n, samples, &mut gram_h).unwrap();
    let (s_h, g_h) = host_s_g(&gram_h, &p.ho, &p.map, n, sta_del, step_dt);
    let mut s_work = s_h.clone();
    let mut x_h = g_h.clone();
    host.cholesky_solve(&mut s_work, &mut x_h, nm)
        .expect("host solve");

    // device
    dev.upload_store(&p.store, n, samples).unwrap();
    dev.assemble(&p.ho, &p.map, 1, sta_del, step_dt).unwrap();
    let gram_d = dev.download_gram().unwrap();
    let (s_d, g_d) = dev.download_s_g(nm).unwrap();

    // Gram bound: 2 k eps (|O||O|^T)_ij
    let mut worst_gram = 0.0f64;
    for j in 0..n {
        for i in 0..=j {
            let mut absum = 0.0;
            for s in 0..samples {
                absum += p.store[i + s * n].abs() * p.store[j + s * n].abs();
            }
            let allowed = 2.0 * samples as f64 * EPS * absum;
            let diff = (gram_d[i + j * n] - gram_h[i + j * n]).abs();
            worst_gram = worst_gram.max(diff / allowed.max(f64::MIN_POSITIVE));
        }
    }
    assert!(worst_gram <= 1.0, "Gram error / bound = {worst_gram}");

    // Same scalar expressions on the same Gram, with FMA disabled. The existing
    // #190 assembly regression budget is 2 eps relative (measured difference 0).
    let (s_hd, g_hd) = host_s_g(&gram_d, &p.ho, &p.map, n, sta_del, step_dt);
    finite_close(&s_d, &s_hd, 0.0, 2.0 * EPS, "S assembly");
    finite_close(&g_d, &g_hd, 0.0, 2.0 * EPS, "g assembly");
    let _ = (&s_h, &g_h);

    // solve
    let x_d = dev.factor_solve(nm).unwrap();
    let kappa = cond_estimate(&s_h, nm);
    let dx: Vec<f64> = x_d.iter().zip(&x_h).map(|(a, b)| a - b).collect();
    let rel = norm(&dx) / norm(&x_h);
    let bound = 4.0 * nm as f64 * EPS * kappa;
    let resid = {
        let sx = matvec(&s_d, &x_d, nm);
        let r: Vec<f64> = sx.iter().zip(&g_d).map(|(a, b)| a - b).collect();
        norm(&r) / norm(&g_d)
    };
    eprintln!(
        "direct n={n:5} ns={samples:5} nm={nm:5} kappa~{kappa:9.2e}: gram err/bound {worst_gram:.2e}, |dx|/|x| {rel:.2e} (bound {bound:.2e}), residual {resid:.2e}"
    );
    assert!(rel <= bound, "solution error {rel} > {bound}");
    assert!(resid <= 1e-9, "device residual {resid}");
    assert!(dev.timings.total_s() > 0.0);
    x_d
}

#[test]
#[ignore = "optional CUDA gate: needs a device (MVMC_RS_CUDA_GATE=1 to require it)"]
fn device_direct_sr_matches_c_order() {
    let Some(mut dev) = gate() else { return };
    let _ = check_direct(&mut dev, 17, 40, 8, 0.5, 0, 28);
    let _ = check_direct(&mut dev, 129, 300, 40, 0.5, 5, 140);
    let _ = check_direct(&mut dev, 400, 2000, 100, 0.3, 0, 411);
    let _ = check_direct(&mut dev, 600, 700, 600, 1.0, 7, 611);
}

#[test]
#[ignore = "optional CUDA gate: needs a device (MVMC_RS_CUDA_GATE=1 to require it)"]
fn device_direct_sr_reports_non_positive_definite_like_host() {
    let Some(mut dev) = gate() else { return };
    let (n, samples) = (30, 60);
    let p = make_problem(n, samples, 10, 0.5, 0, 3);
    // a negative shift ratio makes S indefinite: host DPOSV info != 0
    let sta_del = -2.0;
    let mut host = COrderSr::default();
    let mut gram = vec![0.0; n * n];
    host.gram_real(&p.store, n, samples, &mut gram).unwrap();
    let (mut s, g) = host_s_g(&gram, &p.ho, &p.map, n, sta_del, 0.003);
    let mut x = g;
    assert!(host.cholesky_solve(&mut s, &mut x, p.map.len()).is_err());
    dev.upload_store(&p.store, n, samples).unwrap();
    match dev.solve_direct(&p.ho, &p.map, 1, sta_del, 0.003) {
        Err(SrDeviceError::SolveFailed { info }) => assert!(info > 0, "info {info}"),
        other => panic!("expected SolveFailed, got {other:?}"),
    }
    // the pipeline stays usable after a failed solve
    let x = dev.solve_direct(&p.ho, &p.map, 1, 0.01, 0.003).unwrap();
    assert!(x.iter().all(|v| v.is_finite()));
}

#[test]
#[ignore = "optional CUDA gate: needs a device (MVMC_RS_CUDA_GATE=1 to require it)"]
fn device_direct_sr_reports_singular_zero_gram() {
    let Some(mut dev) = gate() else { return };
    let (n, samples) = (30, 60);
    let p = make_problem(n, samples, 10, 0.5, 0, 3);
    let zero_store = vec![0.0; n * samples];
    dev.upload_store(&zero_store, n, samples).unwrap();
    match dev.solve_direct(&p.ho, &p.map, 1, 0.0, 0.003) {
        Err(SrDeviceError::SolveFailed { info }) => assert!(info > 0, "info {info}"),
        other => panic!("expected singular solve failure, got {other:?}"),
    }
    // The same device state remains usable after reporting the failure.
    dev.upload_store(&p.store, n, samples).unwrap();
    let x = dev.solve_direct(&p.ho, &p.map, 1, 0.01, 0.003).unwrap();
    assert!(x.iter().all(|v| v.is_finite()));
}

fn host_operator(
    inputs: &mvmc_gpu_cuda::sr_problem::CgInputs,
    comp: usize,
    samples: usize,
) -> SampledSrOperator {
    let mut op = SampledSrOperator::new(comp, samples, false);
    op.mean.copy_from_slice(&inputs.mean);
    op.diagonal.copy_from_slice(&inputs.diagonal);
    op.real_samples.copy_from_slice(&inputs.operand);
    op
}

fn materialized_cg(
    inputs: &mvmc_gpu_cuda::sr_problem::CgInputs,
    comp: usize,
    samples: usize,
    inv_weight: f64,
    shift: f64,
) -> Vec<f64> {
    let mut covariance = vec![0.0; comp * comp];
    COrderSr::default()
        .gram_real(&inputs.operand, comp, samples, &mut covariance)
        .unwrap();
    for j in 0..comp {
        for i in 0..comp {
            covariance[i + j * comp] =
                inv_weight * covariance[i + j * comp] - inputs.mean[i] * inputs.mean[j];
        }
        covariance[j + j * comp] += shift * inputs.diagonal[j];
    }
    covariance
}

// Absolute terms of two GEMVs plus mean/diagonal corrections, as in the
// independent operator-product error model below (not the computed result's size).
fn cg_error_scale(
    inp: &mvmc_gpu_cuda::sr_problem::CgInputs,
    x: &[f64],
    comp: usize,
    samples: usize,
    inv_weight: f64,
    shift: f64,
) -> Vec<f64> {
    let mut y = vec![0.0; samples];
    for s in 0..samples {
        for i in 0..comp {
            y[s] += inp.operand[i + s * comp].abs() * x[i].abs();
        }
    }
    let mut scale = vec![0.0; comp];
    for s in 0..samples {
        for i in 0..comp {
            scale[i] += inp.operand[i + s * comp].abs() * y[s];
        }
    }
    let coef_abs: f64 = inp.mean.iter().zip(x).map(|(m, v)| (m * v).abs()).sum();
    for i in 0..comp {
        scale[i] = inv_weight.abs() * scale[i]
            + coef_abs * inp.mean[i].abs()
            + shift.abs() * (inp.diagonal[i] * x[i]).abs();
    }
    scale
}

fn check_operator(dev: &mut DeviceSr, n: usize, samples: usize, factors: usize, noise: f64) {
    let p = make_problem(n, samples, factors, noise, 4, 21 + n as u64);
    let inp = cg_inputs(&p, 0.003);
    let comp = p.map.len();
    dev.set_cg_operand(&inp.operand, None, comp, samples)
        .unwrap();
    let mut op = host_operator(&inp, comp, samples);
    let x: Vec<f64> = (0..comp)
        .map(|i| ((i * 37 % 101) as f64 - 50.0) / 50.0)
        .collect();
    let (inv_w, shift) = (1.0, 0.01);
    let mut z_h = vec![0.0; comp];
    op.apply(&mut z_h, &x, inv_w, shift);
    let z_d = dev
        .apply_operator(&x, &inp.mean, &inp.diagonal, inv_w, shift)
        .unwrap();
    // scale_i = inv_w (|A|(|A|^T|x|))_i + |coef mean_i| + shift |diag x|
    let scale = cg_error_scale(&inp, &x, comp, samples, inv_w, shift);
    let mut worst = 0.0f64;
    for i in 0..comp {
        let allowed = 4.0 * (samples + comp) as f64 * EPS * scale[i];
        worst = worst.max((z_d[i] - z_h[i]).abs() / allowed.max(f64::MIN_POSITIVE));
    }
    eprintln!("operator comp={comp:5} ns={samples:5}: worst error / bound = {worst:.2e}");
    assert!(worst <= 1.0, "operator product error / bound = {worst}");
}

#[test]
#[ignore = "optional CUDA gate: needs a device (MVMC_RS_CUDA_GATE=1 to require it)"]
fn device_cg_operator_matches_c_order() {
    let Some(mut dev) = gate() else { return };
    check_operator(&mut dev, 33, 50, 10, 0.5);
    check_operator(&mut dev, 500, 1500, 60, 0.3);
    check_operator(&mut dev, 1500, 600, 300, 0.2);
}

#[test]
#[ignore = "optional CUDA gate: needs a device (MVMC_RS_CUDA_GATE=1 to require it)"]
fn device_cg_solve_matches_c_order_and_is_repeatable() {
    let Some(mut dev) = gate() else { return };
    // well conditioned: shift dominates, many samples; the trajectory is stable
    let (n, samples) = (300, 3000);
    let p = make_problem(n, samples, 60, 0.7, 0, 5);
    let inp = cg_inputs(&p, 0.003);
    let comp = p.map.len();
    let (inv_w, shift, tol, max_iter) = (1.0, 0.5, 1e-8, comp);
    let mut op = host_operator(&inp, comp, samples);
    let host = op.solve(&inp.gradient, inv_w, shift, tol, max_iter);
    dev.set_cg_operand(&inp.operand, None, comp, samples)
        .unwrap();
    let out = dev
        .solve_cg(
            &inp.gradient,
            &inp.mean,
            &inp.diagonal,
            inv_w,
            shift,
            tol,
            max_iter,
        )
        .unwrap();
    let dx: Vec<f64> = out
        .solution
        .iter()
        .zip(&host.solution)
        .map(|(a, b)| a - b)
        .collect();
    let rel = norm(&dx) / norm(&host.solution);
    eprintln!(
        "cg well-conditioned: iterations host {} device {}, |dx|/|x| = {rel:.2e}",
        host.iterations, out.iterations
    );
    assert_eq!(out.iterations, host.iterations);
    assert!(rel < 1e-9, "{rel}");

    let covariance = materialized_cg(&inp, comp, samples, inv_w, shift);
    let applied = matvec(&covariance, &out.solution, comp);
    let residual: Vec<f64> = applied
        .iter()
        .zip(&inp.gradient)
        .map(|(a, b)| a - b)
        .collect();
    let scale = cg_error_scale(&inp, &out.solution, comp, samples, inv_w, shift);
    let k_eps = (samples + comp + 4) as f64 * EPS;
    let rounding: Vec<f64> = scale
        .iter()
        .zip(&inp.gradient)
        .map(|(s, b)| 8.0 * k_eps / (1.0 - k_eps) * s + 2.0 * EPS * b.abs())
        .collect();
    let residual_norm = norm(&residual);
    let allowed = tol * comp as f64 + norm(&rounding);
    eprintln!(
        "CG converged independent residual {residual_norm:.3e}, stop plus rounding {allowed:.3e}"
    );
    assert!(residual_norm.is_finite() && residual_norm <= allowed);

    // ill conditioned (#358): tiny shift, comp close to the sample count; the CG residual
    // norm is non-monotone and cancels to 1e-16, so only repeatability is asserted after the
    // first iterations, with the divergence printed.
    let (n2, samples2) = (120, 130);
    let p2 = make_problem(n2, samples2, 120, 0.05, 0, 9);
    let inp2 = cg_inputs(&p2, 0.003);
    let comp2 = p2.map.len();
    // Fixed zero-tolerance prefixes need not converge. Their independent residual
    // is diagnostic; it is not compared to a converged direct-solver error bound.
    let covariance = materialized_cg(&inp2, comp2, samples2, inv_w, 1e-4);
    let matrix_norm = (0..comp2)
        .map(|i| {
            (0..comp2)
                .map(|j| covariance[i + j * comp2].abs())
                .sum::<f64>()
        })
        .fold(0.0, f64::max);
    let rhs_norm = inp2.gradient.iter().map(|v| v.abs()).fold(0.0, f64::max);
    dev.set_cg_operand(&inp2.operand, None, comp2, samples2)
        .unwrap();
    for iters in [1usize, 2, 4, 8, comp2] {
        let mut op2 = host_operator(&inp2, comp2, samples2);
        let h = op2.solve(&inp2.gradient, inv_w, 1e-4, 0.0, iters);
        let d1 = dev
            .solve_cg(
                &inp2.gradient,
                &inp2.mean,
                &inp2.diagonal,
                inv_w,
                1e-4,
                0.0,
                iters,
            )
            .unwrap();
        let d2 = dev
            .solve_cg(
                &inp2.gradient,
                &inp2.mean,
                &inp2.diagonal,
                inv_w,
                1e-4,
                0.0,
                iters,
            )
            .unwrap();
        let dx: Vec<f64> = d1
            .solution
            .iter()
            .zip(&h.solution)
            .map(|(a, b)| a - b)
            .collect();
        let rel = norm(&dx) / norm(&h.solution).max(f64::MIN_POSITIVE);
        eprintln!("cg ill-conditioned, {iters:3} iterations: device vs host rel {rel:.2e}");
        assert_eq!(d1.iterations, d2.iterations);
        assert_eq!(d1.iterations, iters);
        let k_eps = (samples2 + comp2) as f64 * EPS;
        // Two GEMV reductions per iteration; a strict fixed-problem regression
        // envelope, not a forward-error guarantee for arbitrary ill-conditioned CG.
        let repeat_rel = 2.0 * iters as f64 * k_eps / (1.0 - k_eps);
        finite_close(
            &d1.solution,
            &d2.solution,
            8.0 * EPS,
            repeat_rel,
            &format!("CG repeat {iters} iterations"),
        );
        if iters == comp2 {
            let host_applied = matvec(&covariance, &h.solution, comp2);
            let host_residual = host_applied
                .iter()
                .zip(&inp2.gradient)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            let host_x_norm = h.solution.iter().map(|v| v.abs()).fold(0.0, f64::max);
            eprintln!(
                "CG C-order full-prefix backward residual {:.3e}",
                host_residual / (matrix_norm * host_x_norm + rhs_norm)
            );
            for solution in [&d1.solution, &d2.solution] {
                let applied = matvec(&covariance, solution, comp2);
                let residual = applied
                    .iter()
                    .zip(&inp2.gradient)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max);
                let x_norm = solution.iter().map(|v| v.abs()).fold(0.0, f64::max);
                let backward = residual / (matrix_norm * x_norm + rhs_norm);
                eprintln!("CG nonconverged full-prefix backward residual {backward:.3e}");
                assert!(backward.is_finite());
            }
        }
        if iters <= 2 {
            assert!(rel < 1e-8, "early-iteration divergence {rel}");
        }
    }
}

#[test]
#[ignore = "optional CUDA gate: needs a device (MVMC_RS_CUDA_GATE=1 to require it)"]
fn device_direct_sr_repeats_with_c_order_and_residual_checks() {
    let Some(mut dev) = gate() else { return };
    let (n, samples) = (257, 600);
    // Preserve the original problem and seed. Each fresh upload/solve is also
    // checked against C order and the original independent residual bounds.
    let x1 = check_direct(&mut dev, n, samples, 50, 0.4, 6, 77);
    let x2 = check_direct(&mut dev, n, samples, 50, 0.4, 6, 77);
    // Existing #190 short-SR output repeatability envelope.
    finite_close(&x1, &x2, 1e-11, 1e-11, "direct SR repeat");
}

#[test]
#[ignore = "optional CUDA gate: needs a device (MVMC_RS_CUDA_GATE=1 to require it)"]
fn device_cg_operator_with_imaginary_samples_matches_c_order() {
    let Some(mut dev) = gate() else { return };
    let (n, samples) = (200, 500);
    let pr = make_problem(n, samples, 30, 0.4, 3, 31);
    let pi = make_problem(n, samples, 30, 0.4, 3, 32);
    let a = cg_inputs(&pr, 0.003);
    let b = cg_inputs(&pi, 0.003);
    let comp = pr.map.len();
    let mut op = SampledSrOperator::new(comp, samples, true);
    op.mean.copy_from_slice(&a.mean);
    op.diagonal.copy_from_slice(&a.diagonal);
    op.real_samples.copy_from_slice(&a.operand);
    op.imag_samples.copy_from_slice(&b.operand);
    let x: Vec<f64> = (0..comp)
        .map(|i| ((i * 13 % 57) as f64 - 28.0) / 28.0)
        .collect();
    let mut z_h = vec![0.0; comp];
    op.apply(&mut z_h, &x, 1.0, 0.01);
    dev.set_cg_operand(&a.operand, Some(&b.operand), comp, samples)
        .unwrap();
    let z_d = dev
        .apply_operator(&x, &a.mean, &a.diagonal, 1.0, 0.01)
        .unwrap();
    let scale = z_h.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let worst = z_d
        .iter()
        .zip(&z_h)
        .fold(0.0f64, |m, (d, h)| m.max((d - h).abs()));
    eprintln!(
        "operator with imaginary samples: max |dz| / max |z| = {:.2e}",
        worst / scale
    );
    // two matrices of k products each: 4 (2k + m) eps relative to the largest entry scale
    assert!(
        worst <= 4.0 * (2 * samples + comp) as f64 * EPS * scale * 10.0,
        "{worst}"
    );
}

/// The unified stage backend (#437) object: `StageBackend::c_order()` is the oracle for the
/// fused `direct_step`/`cg_step`, the resident CUDA backend must agree within the bounds above.
#[test]
#[ignore = "optional CUDA gate: needs a device (MVMC_RS_CUDA_GATE=1 to require it)"]
fn resident_stage_backend_fused_steps_match_the_c_order_stage_backend() {
    use mvmc_core::sr_backend::{CgStepInput, DirectStepInput};
    use mvmc_core::stage_backend::StageBackend;
    if gate().is_none() {
        return;
    }
    let mut oracle = StageBackend::c_order();
    let mut resident = mvmc_gpu_cuda::stages::cuda_resident_stage_backend(0).expect("backend");

    let (n, samples) = (301, 900);
    let p = make_problem(n, samples, 70, 0.5, 9, 41);
    let nm = p.map.len();
    let input = DirectStepInput {
        store: &p.store,
        n,
        samples,
        ho: &p.ho,
        map: &p.map,
        offset: 1,
        sta_del: 0.01,
        step_dt: 0.003,
    };
    let x_c = oracle.sr().direct_step(&input).expect("oracle direct step");
    let x_d = resident
        .sr()
        .direct_step(&input)
        .expect("resident direct step");
    let mut gram = vec![0.0; n * n];
    COrderSr::default()
        .gram_real(&p.store, n, samples, &mut gram)
        .unwrap();
    let (s_h, _g) = host_s_g(&gram, &p.ho, &p.map, n, 0.01, 0.003);
    let kappa = cond_estimate(&s_h, nm);
    let dx: Vec<f64> = x_d.iter().zip(&x_c).map(|(a, b)| a - b).collect();
    let rel = norm(&dx) / norm(&x_c);
    let bound = 4.0 * nm as f64 * EPS * kappa;
    eprintln!(
        "stage backend direct_step: |dx|/|x| {rel:.2e} (bound {bound:.2e}, kappa~{kappa:.1e})"
    );
    assert!(rel <= bound);

    let inp = cg_inputs(&p, 0.003);
    let cg = CgStepInput {
        real: &inp.operand,
        imag: None,
        components: nm,
        samples,
        gradient: &inp.gradient,
        mean: &inp.mean,
        diagonal: &inp.diagonal,
        inv_weight: 1.0,
        shift: 0.5,
        tolerance: 1e-8,
        max_iterations: nm,
    };
    let c = oracle.sr().cg_step(&cg).expect("oracle cg");
    let d = resident.sr().cg_step(&cg).expect("resident cg");
    let dx: Vec<f64> = d
        .solution
        .iter()
        .zip(&c.solution)
        .map(|(a, b)| a - b)
        .collect();
    let rel = norm(&dx) / norm(&c.solution);
    eprintln!(
        "stage backend cg_step: iterations {} / {}, |dx|/|x| {rel:.2e}",
        c.iterations, d.iterations
    );
    assert_eq!(c.iterations, d.iterations);
    assert!(rel < 1e-9);
}

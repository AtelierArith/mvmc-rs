//! Phase 4 — quantum-projection weight initialisation.
//!
//! Port target: `MVMCExpertModeParsers.jl/src/utils/qp_weight.jl` (Julia)
//!              `mVMC/src/mVMC/qp.c` (C reference)
//!
//! BIT-PARITY CRITICAL: reproduces the upstream Gauss–Legendre Newton–
//! Raphson loop, Legendre polynomial recurrence, and `QPFullWeight =
//! QPFixWeight` fan-out so the Pfaffian / sampler downstream sees the
//! same per-QP weights as Julia/C.

use num_complex::Complex64;

use crate::types::{ExpertModeData, QuantumProjectionWeights};

/// Newton–Raphson convergence threshold (`GAUSSLEG_EPS` in `qp.c`).
pub const GAUSSLEG_EPS: f64 = 5.0e-14;

/// Gauss–Legendre quadrature nodes + weights on `[x1, x2]` with `n`
/// points. Mirrors `GaussLeg` from `qp.c` (and `gauss_legendre` in the
/// upstream Julia port).
pub fn gauss_legendre(x1: f64, x2: f64, n: usize) -> (Vec<f64>, Vec<f64>) {
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    let m = (n + 1) / 2;
    let xm = 0.5 * (x2 + x1);
    let xl = 0.5 * (x2 - x1);
    let n_f = n as f64;
    for i in 0..m {
        // C: z = cos(pi * (i + 0.75) / (n + 0.5)). The Julia port uses
        // i (0-based) + 0.75 so the same expression carries over.
        let mut z = (std::f64::consts::PI * ((i as f64) + 0.75) / (n_f + 0.5)).cos();
        let mut pp;
        loop {
            let mut p1 = 1.0;
            let mut p2 = 0.0;
            for j in 1..=n {
                let p3 = p2;
                p2 = p1;
                let j_f = j as f64;
                p1 = ((2.0 * j_f - 1.0) * z * p2 - (j_f - 1.0) * p3) / j_f;
            }
            pp = n_f * (z * p1 - p2) / (z * z - 1.0);
            let z1 = z;
            z = z1 - p1 / pp;
            if (z - z1).abs() <= GAUSSLEG_EPS {
                break;
            }
        }
        x[i] = xm - xl * z;
        x[n - 1 - i] = xm + xl * z;
        w[i] = 2.0 * xl / ((1.0 - z * z) * pp * pp);
        w[n - 1 - i] = w[i];
    }
    (x, w)
}

/// `P_n(x)` Legendre polynomial (recurrence-evaluated). Mirrors
/// `legendre_poly` in upstream.
pub fn legendre_poly(x: f64, n: i64) -> f64 {
    if n <= 0 {
        return 1.0;
    } else if n == 1 {
        return x;
    }
    let mut p01 = 1.0;
    let mut p02 = x;
    let mut p03 = x;
    for i in 2..=n {
        let if_ = i as f64;
        p03 = ((2.0 * if_ - 1.0) * x * p02 - (if_ - 1.0) * p01) / if_;
        p01 = p02;
        p02 = p03;
    }
    p03
}

/// Populate `weights` from `modpara` + `para_qp_trans`. Mirrors
/// `init_qp_weight!(weights, modpara, para_qp_trans, opt_trans)` in
/// upstream.
pub fn init_qp_weight_inplace(
    weights: &mut QuantumProjectionWeights,
    nsp_gauss_leg: i64,
    nsp_stot: i64,
    nmp_trans_raw: i64,
    para_qp_trans: &[Complex64],
    opt_trans: &[Complex64],
) {
    let nmp_trans = nmp_trans_raw.unsigned_abs() as usize;
    if nsp_gauss_leg <= 0 || nmp_trans == 0 {
        return;
    }
    let n_leg = nsp_gauss_leg as usize;
    let nqp_fix = n_leg * nmp_trans;
    let nqp_opt_trans = opt_trans.len().max(1);
    let nqp_full = nqp_fix * nqp_opt_trans;

    weights.qp_full_weight = vec![Complex64::new(0.0, 0.0); nqp_full];
    weights.qp_fix_weight = vec![Complex64::new(0.0, 0.0); nqp_fix];
    weights.spgl_cos = vec![Complex64::new(0.0, 0.0); n_leg];
    weights.spgl_sin = vec![Complex64::new(0.0, 0.0); n_leg];
    weights.spgl_cos_sin = vec![Complex64::new(0.0, 0.0); n_leg];
    weights.spgl_cos_cos = vec![Complex64::new(0.0, 0.0); n_leg];
    weights.spgl_sin_sin = vec![Complex64::new(0.0, 0.0); n_leg];

    if n_leg == 1 {
        weights.spgl_cos[0] = Complex64::new(1.0, 0.0);
        weights.spgl_sin[0] = Complex64::new(0.0, 0.0);
        weights.spgl_cos_sin[0] = Complex64::new(0.0, 0.0);
        weights.spgl_cos_cos[0] = Complex64::new(1.0, 0.0);
        weights.spgl_sin_sin[0] = Complex64::new(0.0, 0.0);
        for j in 0..nmp_trans {
            if j < para_qp_trans.len() {
                weights.qp_fix_weight[j] = para_qp_trans[j];
            }
        }
    } else {
        let (beta, weight_gl) = gauss_legendre(0.0, std::f64::consts::PI, n_leg);
        for i in 0..n_leg {
            let beta_i = beta[i];
            let cos_h = (0.5 * beta_i).cos();
            let sin_h = (0.5 * beta_i).sin();
            weights.spgl_cos[i] = Complex64::new(cos_h, 0.0);
            weights.spgl_sin[i] = Complex64::new(sin_h, 0.0);
            weights.spgl_cos_sin[i] = Complex64::new(cos_h * sin_h, 0.0);
            weights.spgl_cos_cos[i] = Complex64::new(cos_h * cos_h, 0.0);
            weights.spgl_sin_sin[i] = Complex64::new(sin_h * sin_h, 0.0);

            let cos_beta = beta_i.cos();
            let w = 0.5 * beta_i.sin() * weight_gl[i] * legendre_poly(cos_beta, nsp_stot);
            for j in 0..nmp_trans {
                let idx = i + j * n_leg;
                if j < para_qp_trans.len() {
                    weights.qp_fix_weight[idx] = para_qp_trans[j] * w;
                }
            }
        }
    }

    update_qp_weight(weights, opt_trans);
}

/// Refresh `qp_full_weight` from `qp_fix_weight`. Mirrors
/// `update_qp_weight!`.
pub fn update_qp_weight(weights: &mut QuantumProjectionWeights, opt_trans: &[Complex64]) {
    let nqp_fix = weights.qp_fix_weight.len();
    if opt_trans.is_empty() {
        if weights.qp_full_weight.len() != nqp_fix {
            weights.qp_full_weight = vec![Complex64::new(0.0, 0.0); nqp_fix];
        }
        for j in 0..nqp_fix {
            weights.qp_full_weight[j] = weights.qp_fix_weight[j];
        }
    } else {
        let nqp_opt = opt_trans.len();
        let nqp_full = nqp_fix * nqp_opt;
        if weights.qp_full_weight.len() != nqp_full {
            weights.qp_full_weight = vec![Complex64::new(0.0, 0.0); nqp_full];
        }
        for i in 0..nqp_opt {
            let offset = i * nqp_fix;
            let tmp = opt_trans[i];
            for j in 0..nqp_fix {
                weights.qp_full_weight[offset + j] = tmp * weights.qp_fix_weight[j];
            }
        }
    }
}

/// `data.qp_weights = init_qp_weight!(data)` mirror.
pub fn init_qp_weight(data: &mut ExpertModeData) {
    let mut weights = data
        .qp_weights
        .take()
        .unwrap_or_else(QuantumProjectionWeights::new);
    init_qp_weight_inplace(
        &mut weights,
        data.modpara.nsp_gauss_leg,
        data.modpara.nsp_stot,
        data.modpara.nmp_trans,
        &data.para_qp_trans,
        &[],
    );
    data.qp_weights = Some(weights);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legendre_poly_first_few_match_known_values() {
        // P_0(0.3) = 1, P_1(0.3) = 0.3, P_2(0.3) = 0.5*(3*x^2 - 1) = -0.365
        assert!((legendre_poly(0.3, 0) - 1.0).abs() < 1e-15);
        assert!((legendre_poly(0.3, 1) - 0.3).abs() < 1e-15);
        assert!((legendre_poly(0.3, 2) + 0.365).abs() < 1e-15);
    }

    #[test]
    fn gauss_legendre_two_points_match_known_solution() {
        let (x, w) = gauss_legendre(-1.0, 1.0, 2);
        assert!((x[0] + (1.0_f64 / 3.0).sqrt()).abs() < 1e-12);
        assert!((x[1] - (1.0_f64 / 3.0).sqrt()).abs() < 1e-12);
        assert!((w[0] - 1.0).abs() < 1e-12);
        assert!((w[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn init_qp_weight_nspgaussleg_one_passes_through_para() {
        let mut weights = QuantumProjectionWeights::new();
        let para = vec![Complex64::new(2.0, 0.0), Complex64::new(-1.0, 0.5)];
        init_qp_weight_inplace(&mut weights, 1, 0, -2, &para, &[]);
        assert_eq!(weights.qp_fix_weight, para);
        assert_eq!(weights.qp_full_weight, para);
        assert_eq!(weights.spgl_cos, vec![Complex64::new(1.0, 0.0)]);
    }
}

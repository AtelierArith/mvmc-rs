//! Original Julia8bb test_qp_weight.jl M0282–288/M0293–298/M0316–319.
//! C qp.c60–81 uses GaussLeg and Stot Legendre weights; no oracle runtime.
//! Existing midpoint/Legendre/core tests and coefficient fixtures stay separate.
use mvmc_expert_parsers::utils::qp_weight::{gauss_legendre, init_qp_weight};
use mvmc_expert_parsers::ExpertModeData;
use num_complex::Complex64;

#[test]
fn original_four_node_pi_symmetry_and_eight_node_sine_integral() {
    let pi = std::f64::consts::PI;
    let (nodes, weights) = gauss_legendre(0.0, pi, 4);
    assert_eq!(nodes.len(), 4); // M0282
    assert_eq!(weights.len(), 4); // M0283
                                  // Original fixed n4 symmetry bound, not a new computed-output tolerance.
    const ORIGINAL_ABS: f64 = 1e-10;
    assert!((nodes[0] + nodes[3] - pi).abs() < ORIGINAL_ABS); // M0284
    assert!((nodes[1] + nodes[2] - pi).abs() < ORIGINAL_ABS); // M0285
    assert!((weights[0] - weights[3]).abs() < ORIGINAL_ABS); // M0286
    assert!((weights[1] - weights[2]).abs() < ORIGINAL_ABS); // M0287

    let (nodes, weights) = gauss_legendre(0.0, pi, 8);
    assert_eq!((nodes.len(), weights.len()), (8, 8));
    assert!(nodes.iter().all(|&node| node > 0.0 && node < pi));
    assert!(weights
        .iter()
        .all(|&weight| weight.is_finite() && weight > 0.0));
    let integral: f64 = nodes.iter().zip(&weights).map(|(&x, &w)| w * x.sin()).sum();
    // Independent calculus integral is 2, not another Rust quadrature result.
    // Gauss8 truncation <= pi^17*(8!)^4/(17*(16!)^3) < 4.81e-15
    // because every sixteenth derivative of sin has magnitude <=1. Original
    // 1e-10 also covers this low-order root/sum/libm roundoff; it is not a
    // solver, downstream sampling, or cross-language coefficient budget.
    assert!((integral - 2.0).abs() < ORIGINAL_ABS); // M0288
}

#[test]
fn original_four_node_stot_one_public_weights_and_trigonometric_products() {
    let mut data = ExpertModeData::new();
    data.modpara.nsp_gauss_leg = 4;
    data.modpara.nsp_stot = 1;
    data.modpara.nmp_trans = 1;
    data.para_qp_trans = vec![Complex64::new(1.0, 0.0)];
    init_qp_weight(&mut data);
    let weights = data
        .qp_weights
        .as_ref()
        .expect("M0293 initialized public weights");
    assert_eq!(weights.qp_full_weight.len(), 4); // M0294
    assert_eq!(weights.qp_fix_weight.len(), 4); // M0295
    assert_eq!(weights.spgl_cos.len(), 4); // M0296
    assert_eq!(weights.spgl_sin.len(), 4); // M0297
    assert!(weights.qp_fix_weight.iter().any(|w| w.norm() > 1e-10)); // M0298
    assert_eq!(weights.spgl_cos_sin.len(), 4);
    assert_eq!(weights.spgl_cos_cos.len(), 4);
    assert_eq!(weights.spgl_sin_sin.len(), 4);
    for i in 0..4 {
        let cos = weights.spgl_cos[i];
        let sin = weights.spgl_sin[i];
        // Unit-scale original half-angle identities, not independent fixtures
        // for every QPFixWeight coefficient. No tolerance is increased.
        assert!((cos.norm_sqr() + sin.norm_sqr() - 1.0).abs() < 1e-10); // M0316
        assert!((weights.spgl_cos_sin[i] - cos * sin).norm() < 1e-10); // M0317
        assert!((weights.spgl_cos_cos[i] - cos * cos).norm() < 1e-10); // M0318
        assert!((weights.spgl_sin_sin[i] - sin * sin).norm() < 1e-10); // M0319
    }
    // Stot1 has P1(cos(beta))=cos(beta): positive sin/quadrature coefficients
    // give first two positive and last two negative weights. This additional
    // analytic sign check detects silently using Stot0 without computed goldens.
    for (i, weight) in weights.qp_fix_weight.iter().enumerate() {
        assert!(weight.re.is_finite());
        assert_eq!(weight.im, 0.0);
        assert!(if i < 2 {
            weight.re > 0.0
        } else {
            weight.re < 0.0
        });
    }
    assert_eq!(data.modpara.nsp_stot, 1);
    assert_eq!(data.para_qp_trans, [Complex64::new(1.0, 0.0)]);
}

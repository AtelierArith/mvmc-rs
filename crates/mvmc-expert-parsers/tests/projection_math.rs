//! Numerical inputs to the Pfaffian and SR-CG kernels.
#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;
use mvmc_expert_parsers::utils::qp_weight::{gauss_legendre, init_qp_weight_inplace};
use mvmc_expert_parsers::QuantumProjectionWeights;
use num_complex::Complex64;

#[test]
fn projection_coefficients_match_julia_113_with_roundoff_bounds() {
    let mut rows = include_str!("../../../tests/fixtures/projection_math.txt")
        .lines()
        .filter(|s| !s.starts_with('#'));
    for n in [1, 2, 3, 4, 8, 16, 32, 64, 128] {
        let (beta, quadrature) = gauss_legendre(0.0, std::f64::consts::PI, n);
        let mut w = QuantumProjectionWeights::new();
        init_qp_weight_inplace(&mut w, n as i64, 0, 1, &[Complex64::new(1.0, 0.0)], &[]);
        for i in 0..n {
            let fields: Vec<_> = rows.next().unwrap().split_whitespace().collect();
            assert_eq!(fields[0].parse::<usize>().unwrap(), n);
            assert_eq!(fields[1].parse::<usize>().unwrap(), i);
            for (name, value, expected) in [
                ("beta", beta[i], fields[2]),
                ("quadrature", quadrature[i], fields[3]),
                ("cos_half", w.spgl_cos[i].re, fields[4]),
                ("sin_half", w.spgl_sin[i].re, fields[5]),
                ("cos_sin", w.spgl_cos_sin[i].re, fields[6]),
                ("cos_cos", w.spgl_cos_cos[i].re, fields[7]),
                ("sin_sin", w.spgl_sin_sin[i].re, fields[8]),
                ("qp_weight", w.qp_full_weight[i].re, fields[9]),
            ] {
                // Each root uses an n-term recurrence and Newton refinement.
                // Quadrature/QP weights additionally divide by (1-z*z): its
                // endpoint cancellation amplifies a root's rounding error.
                // Near-zero trigonometric coefficients need an absolute floor.
                let z = 2.0 * beta[i] / std::f64::consts::PI - 1.0;
                let weight_condition = if matches!(name, "quadrature" | "qp_weight") {
                    8.0 / (1.0 - z * z)
                } else {
                    0.0
                };
                numerical_comparison::assert_close(
                    value,
                    f64::from_bits(u64::from_str_radix(expected, 16).unwrap()),
                    16.0 * f64::EPSILON,
                    (32.0 * n as f64 + weight_condition) * f64::EPSILON,
                    format!("n={n} i={i} {name}"),
                );
            }
        }
    }
    assert!(rows.next().is_none());
}

#[test]
fn legendre_recurrence_matches_julia_with_roundoff_bounds() {
    for line in include_str!("../../../tests/fixtures/legendre_poly.txt")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let v: Vec<_> = line.split_whitespace().collect();
        let n = v[0].parse().unwrap();
        let x = f64::from_bits(u64::from_str_radix(v[1], 16).unwrap());
        let expected = u64::from_str_radix(v[2], 16).unwrap();
        let actual = mvmc_expert_parsers::utils::qp_weight::legendre_poly(x, n);
        // Recurrence roundoff grows with the number of polynomial steps.
        let bound = 16.0 * (n + 1) as f64 * f64::EPSILON;
        numerical_comparison::assert_close(
            actual,
            f64::from_bits(expected),
            bound,
            bound,
            format!("n={n} x={x}"),
        );
    }
}

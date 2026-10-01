//! Exact numerical inputs to the Pfaffian and SR-CG kernels.
use mvmc_expert_parsers::utils::qp_weight::{gauss_legendre, init_qp_weight_inplace};
use mvmc_expert_parsers::QuantumProjectionWeights;
use num_complex::Complex64;

#[test]
fn projection_coefficients_match_julia_113_numerical_bits() {
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
                assert_eq!(
                    value.to_bits(),
                    u64::from_str_radix(expected, 16).unwrap(),
                    "n={n} i={i} {name}"
                );
            }
        }
    }
    assert!(rows.next().is_none());
}

#[test]
fn legendre_recurrence_matches_julia_operation_order() {
    for line in include_str!("../../../tests/fixtures/legendre_poly.txt")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let v: Vec<_> = line.split_whitespace().collect();
        let n = v[0].parse().unwrap();
        let x = f64::from_bits(u64::from_str_radix(v[1], 16).unwrap());
        let expected = u64::from_str_radix(v[2], 16).unwrap();
        let actual = mvmc_expert_parsers::utils::qp_weight::legendre_poly(x, n);
        assert_eq!(actual.to_bits(), expected, "n={n} x={x}");
    }
}

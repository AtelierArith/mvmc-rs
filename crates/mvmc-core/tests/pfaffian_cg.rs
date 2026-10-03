//! The real Pfaffian/inverse inputs feeding the first source SR-CG step.
#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;
// The <=8 dimensional factorizations allow ~450 epsilon of local rounding;
// inverse entries allow the additional amplification by the fixed inputs.
use pfapack::{dsktf2, utu2inv_real, utu2pfa_real, PivotIndex1Based, SqMat};

// Verify backward error as well as fixture agreement, so the inverse budget
// cannot excuse an inaccurate result on the fixed, moderately conditioned inputs.
fn check_inverse_residual(
    original: &[num_complex::Complex64],
    inverse: &[num_complex::Complex64],
    n: usize,
    context: &str,
) {
    for row in 0..n {
        for column in 0..n {
            let products = (0..n).map(|k| original[k * n + row] * inverse[column * n + k]);
            let residual: num_complex::Complex64 = products.clone().sum();
            let scale: f64 = products.map(|z| z.norm()).sum();
            let budget = 32.0 * n as f64 * f64::EPSILON * (1.0 + scale);
            numerical_comparison::assert_values_close(
                [residual.re, residual.im],
                [if row == column { 1.0 } else { 0.0 }, 0.0],
                budget,
                0.0,
                format!("{context} inverse residual ({row},{column})"),
            );
        }
    }
}

#[test]
fn real_slater_pfaffians_and_inverses_match_julia_numerical_values() {
    let mut lines = include_str!("../../../tests/fixtures/pfaffian_cg/real.txt")
        .lines()
        .filter(|l| !l.starts_with('#'));
    let parse = |l: &str| -> Vec<f64> {
        l.split_whitespace()
            .map(|s| f64::from_bits(u64::from_str_radix(s, 16).unwrap()))
            .collect()
    };
    let mut cases = 0;
    while let Some(line) = lines.next() {
        let n: usize = line.parse().unwrap();
        let mut data = parse(lines.next().unwrap());
        let original = data.clone();
        let expected_piv: Vec<u32> = lines
            .next()
            .unwrap()
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        let ltl = parse(lines.next().unwrap());
        let pf = parse(lines.next().unwrap())[0];
        let inverse = parse(lines.next().unwrap());
        let mut piv = vec![PivotIndex1Based(0); n];
        let mut m = vec![0.0; n * n];
        let mut vt = vec![0.0; n - 1];
        {
            let mut a = SqMat::new(&mut data, n);
            dsktf2(&mut a, &mut piv).unwrap();
            assert_eq!(piv.iter().map(|p| p.0).collect::<Vec<_>>(), expected_piv);
            numerical_comparison::assert_close(
                utu2pfa_real(&a, &piv),
                pf,
                1e-13,
                1e-13,
                format!("QP {cases} Pfaffian"),
            );
        }
        for (i, (&a, &b)) in data.iter().zip(&ltl).enumerate() {
            numerical_comparison::assert_close(
                a,
                b,
                1e-13,
                1e-13,
                format!("QP {cases} LTL slot {i}"),
            );
        }
        utu2inv_real(
            &mut SqMat::new(&mut data, n),
            &piv,
            &mut vt,
            &mut SqMat::new(&mut m, n),
        );
        for (i, (&a, &b)) in data.iter().zip(&inverse).enumerate() {
            numerical_comparison::assert_close(
                a,
                b,
                512.0 * f64::EPSILON,
                512.0 * f64::EPSILON,
                format!("QP {cases} inverse slot {i}"),
            );
        }
        check_inverse_residual(
            &original
                .iter()
                .map(|&v| num_complex::Complex64::new(v, 0.0))
                .collect::<Vec<_>>(),
            &data
                .iter()
                .map(|&v| num_complex::Complex64::new(v, 0.0))
                .collect::<Vec<_>>(),
            n,
            "real Slater",
        );
        cases += 1;
    }
    assert_eq!(cases, 8);
}

#[test]
fn green_ratio_division_matches_julia_numerical_values() {
    use num_complex::Complex64 as C;
    for line in include_str!("../../../tests/fixtures/complex_division.txt").lines() {
        let v: Vec<f64> = line
            .split_whitespace()
            .map(|s| f64::from_bits(u64::from_str_radix(s, 16).unwrap()))
            .collect();
        let z = C::new(v[0], v[1]);
        let w = C::new(v[2], v[3]);
        let actual = mvmc_core::julia_complex::divide(z, w);
        numerical_comparison::assert_values_close(
            [actual.re, actual.im],
            [v[4], v[5]],
            0.0,
            16.0 * f64::EPSILON,
            "complex division",
        );
    }
}

#[test]
fn reciprocal_matches_julia_numerical_values() {
    use num_complex::Complex64 as C;
    for line in include_str!("../../../tests/fixtures/complex_reciprocal.txt").lines() {
        let v: Vec<f64> = line
            .split_whitespace()
            .map(|s| f64::from_bits(u64::from_str_radix(s, 16).unwrap()))
            .collect();
        let actual = mvmc_core::julia_complex::reciprocal(C::new(v[0], v[1]));
        numerical_comparison::assert_values_close(
            [actual.re, actual.im],
            [v[2], v[3]],
            0.0,
            16.0 * f64::EPSILON,
            "complex reciprocal",
        );
    }
}

#[test]
fn complex_slater_pfaffians_and_inverses_match_julia_turbo_values() {
    use num_complex::Complex64 as C;
    use pfapack::{utu2inv_complex, utu2pfa_complex, zsktf2_turbo};
    let mut lines = include_str!("../../../tests/fixtures/pfaffian_cg/complex.txt")
        .lines()
        .filter(|l| !l.starts_with('#'));
    let parse = |l: &str| -> Vec<C> {
        let doubles: Vec<f64> = l
            .split_whitespace()
            .map(|s| f64::from_bits(u64::from_str_radix(s, 16).unwrap()))
            .collect();
        doubles
            .as_chunks::<2>()
            .0
            .iter()
            .map(|v| C::new(v[0], v[1]))
            .collect()
    };
    let mut cases = 0;
    while let Some(line) = lines.next() {
        let n: usize = line.parse().unwrap();
        let mut data = parse(lines.next().unwrap());
        let original = data.clone();
        let piv_expected: Vec<u32> = lines
            .next()
            .unwrap()
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        let ltl = parse(lines.next().unwrap());
        let pf = parse(lines.next().unwrap())[0];
        let inverse = parse(lines.next().unwrap());
        let mut piv = vec![PivotIndex1Based(0); n];
        zsktf2_turbo(&mut SqMat::new(&mut data, n), &mut piv).unwrap();
        assert_eq!(piv.iter().map(|p| p.0).collect::<Vec<_>>(), piv_expected);
        let check = |actual: &[C], expected: &[C], kind: &str| {
            let bound = if kind == "inverse" {
                512.0 * f64::EPSILON
            } else {
                1e-13
            };
            numerical_comparison::assert_values_close(
                actual.iter().flat_map(|z| [z.re, z.im]),
                expected.iter().flat_map(|z| [z.re, z.im]),
                bound,
                bound,
                format!("QP {cases} {kind}"),
            );
        };
        check(&data, &ltl, "LTL");
        let actual = utu2pfa_complex(&SqMat::new(&mut data, n), &piv);
        check(&[actual], &[pf], "Pfaffian");
        let mut m = vec![C::new(0.0, 0.0); n * n];
        let mut vt = vec![C::new(0.0, 0.0); n - 1];
        utu2inv_complex(
            &mut SqMat::new(&mut data, n),
            &piv,
            &mut vt,
            &mut SqMat::new(&mut m, n),
        );
        check(&data, &inverse, "inverse");
        check_inverse_residual(&original, &data, n, "complex Slater");
        cases += 1;
    }
    assert_eq!(cases, 8);
}

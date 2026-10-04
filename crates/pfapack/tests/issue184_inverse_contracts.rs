//! Offline native-C/J fixtures; no toolbox/oracle runtime dependency.
use num_complex::Complex64;
use pfapack::{utu2inv_complex, utu2inv_real, zsktf2, zsktf2_c_compat, PivotIndex1Based, SqMat};

fn pairs(text: &str, count: usize) -> Vec<Complex64> {
    let x: Vec<f64> = text
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    assert_eq!(x.len(), 2 * count);
    x.as_chunks::<2>()
        .0
        .iter()
        .map(|x| Complex64::new(x[0], x[1]))
        .collect()
}

// The original Julia case is unseeded and conditional on INFO==0. This
// independently retained fixed input MUST factor successfully; no vacuous pass.
fn factorized_complex6_contract(c_order: bool) {
    let n = 6;
    let expected_pivots = [1, 1, 1, 1, 1, 6];
    let input = fixture(
        include_str!("../../../tests/fixtures/issue184_inverse/complex6_factorized.input.txt"),
        n * n,
        &expected_pivots,
        Some(("c", n)),
    );
    let factor =
        include_str!("../../../tests/fixtures/issue184_inverse/complex6_factorized.factor.txt");
    let (info, factor) = factor.split_once('\n').unwrap();
    assert_eq!(
        info.parse::<i32>().unwrap(),
        0,
        "native factorization must succeed"
    );
    let expected_factor = fixture(factor, n * n, &expected_pivots, None);
    let expected_c = fixture(
        include_str!("../../../tests/fixtures/issue184_inverse/complex6_factorized.c.txt"),
        2 * n * n + n - 1,
        &expected_pivots,
        None,
    );
    let expected_j = fixture(
        include_str!("../../../tests/fixtures/issue184_inverse/complex6_factorized.j.txt"),
        2 * n * n + n - 1,
        &expected_pivots,
        None,
    );
    let mut factor = input.clone();
    let mut piv = vec![PivotIndex1Based(0); n];
    let result = if c_order {
        zsktf2_c_compat(&mut SqMat::new(&mut factor, n), &mut piv)
    } else {
        zsktf2(&mut SqMat::new(&mut factor, n), &mut piv)
    };
    assert_eq!(result, Ok(()), "actual Rust factorization must succeed");
    assert_eq!(piv.iter().map(|p| p.0).collect::<Vec<_>>(), expected_pivots);
    assert!(piv.iter().enumerate().any(|(i, p)| p.0 != i as u32 + 1));
    // Separate factorization budget: five <=6-wide upper elimination stages;
    // complex multiply/add and division allowance8*5*6=240 rounded to256eps
    // abs+rel, unchanged from the inverse fixture bound. Native/J factor
    // difference4.45e-16, condition estimate40.63, and original-skew residual
    // jointly bound this retained input; not a universal matrix error bound.
    compare(&factor, &expected_factor);
    for poison in [0., 17.] {
        let mut m = vec![Complex64::new(poison, 0.); n * n];
        let mut vt = vec![Complex64::new(poison, 0.); n - 1];
        for _reuse in 0..2 {
            // Exercise inverse on ACTUAL Rust factor output, not a C-LTL injection.
            let mut a = factor.clone();
            let before = piv.clone();
            utu2inv_complex(
                &mut SqMat::new(&mut a, n),
                &piv,
                &mut vt,
                &mut SqMat::new(&mut m, n),
            );
            assert_eq!(piv, before);
            let actual = a.iter().chain(&m).chain(&vt).copied().collect::<Vec<_>>();
            compare(&actual, &expected_c);
            compare(&actual, &expected_j);
            residual(
                &a,
                include_str!(
                    "../../../tests/fixtures/issue184_inverse/complex6_factorized.operator.txt"
                ),
                n,
            );
        }
    }
}

#[test]
fn actual_c_compatible_factorization_feeds_full_inverse_and_original_skew_residual() {
    factorized_complex6_contract(true);
}

#[test]
fn actual_default_factorization_feeds_full_inverse_and_original_skew_residual() {
    // Distinct Julia-architecture arithmetic path, independently checked against
    // the same native C expected values, not selected instead of a failing path.
    factorized_complex6_contract(false);
}

fn fixture(
    text: &str,
    count: usize,
    pivots: &[u32],
    header: Option<(&str, usize)>,
) -> Vec<Complex64> {
    let mut tokens = text.split_whitespace();
    if let Some((kind, n)) = header {
        assert_eq!(tokens.next(), Some(kind));
        assert_eq!(tokens.next().unwrap().parse::<usize>().unwrap(), n);
    }
    let values = tokens
        .by_ref()
        .take(2 * count)
        .collect::<Vec<_>>()
        .join(" ");
    let values = pairs(&values, count);
    for &p in pivots {
        assert_eq!(tokens.next().unwrap().parse::<u32>().unwrap(), p);
    }
    assert!(tokens.next().is_none(), "extra fixture tokens");
    values
}

fn compare(actual: &[Complex64], expected: &[Complex64]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(a.re.is_finite() && a.im.is_finite());
        // Retained operators have measured infinity-condition estimates
        // <=40.63 and 256-bit C backward errors <4.20e-17 (see provenance).
        // Four short solves/products, <=6 terms, complex arithmetic allowance
        // 8*4*6=192 roundings rounded up to256: abs+rel budget, not bit parity.
        // A separate scaled residual guards against forward-error-only checks.
        let bound = 256.0 * f64::EPSILON * (1.0 + e.norm());
        assert!(
            (*a - *e).norm() <= bound,
            "entry {i}: actual={a}, expected={e}, bound={bound}"
        );
    }
}

fn residual(a: &[Complex64], operator: &str, n: usize) {
    let b = pairs(operator, n * n);
    let norm = |x: &[Complex64]| {
        (0..n)
            .map(|i| (0..n).map(|j| x[j * n + i].norm()).sum::<f64>())
            .fold(0., f64::max)
    };
    let scale = norm(&b) * norm(a) + 1.;
    let mut error = 0_f64;
    for i in 0..n {
        let mut row = 0.;
        for j in 0..n {
            let value: Complex64 = (0..n).map(|k| b[k * n + i] * a[j * n + k]).sum();
            row += (value - Complex64::new(f64::from(i == j), 0.)).norm();
        }
        error = error.max(row);
    }
    assert!(error.is_finite() && error / scale <= 256. * f64::EPSILON);
}

#[test]
fn real4_identity_preserves_full_inverse_workspace_contract() {
    let n = 4;
    let input = include_str!("../../../tests/fixtures/issue184_inverse/real4_identity.input.txt");
    let expected_pivots = [1, 2, 3, 4];
    let values = fixture(input, n * n, &expected_pivots, Some(("r", n)));
    for poison in [0., 17.] {
        let mut a: Vec<f64> = values.iter().map(|z| z.re).collect();
        let mut m = vec![poison; n * n];
        let mut vt = vec![poison; n - 1];
        let piv = vec![
            PivotIndex1Based(1),
            PivotIndex1Based(2),
            PivotIndex1Based(3),
            PivotIndex1Based(4),
        ];
        let before = piv.clone();
        for _repeat in 0..2 {
            a.iter_mut().zip(&values).for_each(|(a, v)| *a = v.re);
            utu2inv_real(
                &mut SqMat::new(&mut a, n),
                &piv,
                &mut vt,
                &mut SqMat::new(&mut m, n),
            );
            assert_eq!(piv, before);
            let actual: Vec<_> = a
                .iter()
                .chain(&m)
                .chain(&vt)
                .map(|&x| Complex64::new(x, 0.))
                .collect();
            compare(
                &actual,
                &fixture(
                    include_str!("../../../tests/fixtures/issue184_inverse/real4_identity.c.txt"),
                    2 * n * n + n - 1,
                    &expected_pivots,
                    None,
                ),
            );
            compare(
                &actual,
                &fixture(
                    include_str!("../../../tests/fixtures/issue184_inverse/real4_identity.j.txt"),
                    2 * n * n + n - 1,
                    &expected_pivots,
                    None,
                ),
            );
            residual(
                &actual[..n * n],
                include_str!(
                    "../../../tests/fixtures/issue184_inverse/real4_identity.operator.txt"
                ),
                n,
            );
        }
    }
}

#[test]
fn complex6_nontrivial_pivots_preserve_full_inverse_workspace_contract() {
    let n = 6;
    let input =
        include_str!("../../../tests/fixtures/issue184_inverse/complex6_pair_pivots.input.txt");
    let expected_pivots = [2, 1, 4, 3, 6, 5];
    let values = fixture(input, n * n, &expected_pivots, Some(("c", n)));
    for poison in [0., 17.] {
        let mut a = values.clone();
        let mut m = vec![Complex64::new(poison, 0.); n * n];
        let mut vt = vec![Complex64::new(poison, 0.); n - 1];
        let piv: Vec<_> = [2, 1, 4, 3, 6, 5]
            .into_iter()
            .map(PivotIndex1Based)
            .collect();
        let before = piv.clone();
        for _repeat in 0..2 {
            a.copy_from_slice(&values);
            utu2inv_complex(
                &mut SqMat::new(&mut a, n),
                &piv,
                &mut vt,
                &mut SqMat::new(&mut m, n),
            );
            assert_eq!(piv, before);
            let actual = a.iter().chain(&m).chain(&vt).copied().collect::<Vec<_>>();
            compare(
                &actual,
                &fixture(
                    include_str!(
                        "../../../tests/fixtures/issue184_inverse/complex6_pair_pivots.c.txt"
                    ),
                    2 * n * n + n - 1,
                    &expected_pivots,
                    None,
                ),
            );
            compare(
                &actual,
                &fixture(
                    include_str!(
                        "../../../tests/fixtures/issue184_inverse/complex6_pair_pivots.j.txt"
                    ),
                    2 * n * n + n - 1,
                    &expected_pivots,
                    None,
                ),
            );
            residual(
                &actual[..n * n],
                include_str!(
                    "../../../tests/fixtures/issue184_inverse/complex6_pair_pivots.operator.txt"
                ),
                n,
            );
        }
    }
}

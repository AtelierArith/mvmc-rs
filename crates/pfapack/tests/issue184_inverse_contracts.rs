//! Offline native-C/J fixtures; no toolbox/oracle runtime dependency.
use num_complex::Complex64;
use pfapack::{utu2inv_complex, utu2inv_real, PivotIndex1Based, SqMat};

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
        // These two retained operators have measured infinity-condition
        // estimates <=14.2 and 256-bit C backward errors <1.83e-17.
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

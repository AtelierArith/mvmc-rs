//! The real Pfaffian/inverse inputs feeding the first source SR-CG step.
use pfapack::{dsktf2, utu2inv_real, utu2pfa_real, PivotIndex1Based, SqMat};

#[test]
fn real_slater_pfaffians_and_inverses_match_julia_numerical_bits() {
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
            assert_eq!(
                utu2pfa_real(&a, &piv).to_bits(),
                pf.to_bits(),
                "QP {cases} Pfaffian"
            );
        }
        for (i, (&a, &b)) in data.iter().zip(&ltl).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "QP {cases} LTL slot {i}");
        }
        utu2inv_real(
            &mut SqMat::new(&mut data, n),
            &piv,
            &mut vt,
            &mut SqMat::new(&mut m, n),
        );
        for (i, (&a, &b)) in data.iter().zip(&inverse).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "QP {cases} inverse slot {i}: {a} != {b}"
            );
        }
        cases += 1;
    }
    assert_eq!(cases, 8);
}

#[test]
fn green_ratio_division_matches_julia_numerical_bits() {
    use num_complex::Complex64 as C;
    for line in include_str!("../../../tests/fixtures/complex_division.txt").lines() {
        let v: Vec<f64> = line
            .split_whitespace()
            .map(|s| f64::from_bits(u64::from_str_radix(s, 16).unwrap()))
            .collect();
        let z = C::new(v[0], v[1]);
        let w = C::new(v[2], v[3]);
        let actual = mvmc_core::julia_complex::divide(z, w);
        assert_eq!(
            (actual.re.to_bits(), actual.im.to_bits()),
            (v[4].to_bits(), v[5].to_bits())
        );
    }
}

#[test]
fn reciprocal_matches_julia_numerical_bits() {
    use num_complex::Complex64 as C;
    for line in include_str!("../../../tests/fixtures/complex_reciprocal.txt").lines() {
        let v: Vec<f64> = line
            .split_whitespace()
            .map(|s| f64::from_bits(u64::from_str_radix(s, 16).unwrap()))
            .collect();
        let actual = mvmc_core::julia_complex::reciprocal(C::new(v[0], v[1]));
        assert_eq!(
            (actual.re.to_bits(), actual.im.to_bits()),
            (v[2].to_bits(), v[3].to_bits())
        );
    }
}

#[test]
fn complex_slater_pfaffians_and_inverses_match_julia_turbo_bits() {
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
            .chunks_exact(2)
            .map(|v| C::new(v[0], v[1]))
            .collect()
    };
    let mut cases = 0;
    while let Some(line) = lines.next() {
        let n: usize = line.parse().unwrap();
        let mut data = parse(lines.next().unwrap());
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
            for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
                assert_eq!(
                    (a.re.to_bits(), a.im.to_bits()),
                    (b.re.to_bits(), b.im.to_bits()),
                    "QP {cases} {kind} slot {i}: {a} != {b}"
                );
            }
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
        cases += 1;
    }
    assert_eq!(cases, 8);
}

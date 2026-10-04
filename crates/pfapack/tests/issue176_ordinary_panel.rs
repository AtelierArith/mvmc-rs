// SOURCE only; independent algebra, not recorded Rust/C golden output.
use pfapack::{utu2inv_real, PivotIndex1Based, SqMat};
#[test]
fn ordinary_real_public_boundary_and_pivot_contracts() {
    for n in [64usize, 66, 128] {
        for permuted in [false, true] {
            // U=I+5e02+3e12,N^2=0,T block pairs7,2,... . Exact G=U^-T T^-1 U^-1.
            let mut factor = vec![0.; n * n];
            for i in (0..n).step_by(2) {
                factor[i + (i + 1) * n] = if i == 0 { 7. } else { 2. };
            }
            factor[3 * n] = 5.;
            factor[1 + 3 * n] = 3.;
            let mut piv: Vec<_> = (1..=n).map(|i| PivotIndex1Based(i as u32)).collect();
            // Independent known permutation swaps labels0/2 and1/3 once each.
            if permuted {
                piv[0] = PivotIndex1Based(3);
                piv[1] = PivotIndex1Based(4);
            }
            let mut vt = vec![0.; n - 1];
            let mut work = vec![91.; n * n];
            utu2inv_real(
                &mut SqMat::new(&mut factor, n),
                &piv,
                &mut vt,
                &mut SqMat::new(&mut work, n),
            );
            // Explicit C M initialization domains are not touched by upper-copy.
            for j in 0..n {
                assert_eq!(work[j + j * n], 1.);
                for i in j + 1..n {
                    assert_eq!(work[i + j * n], 0., "assigned M lower {n}/{i}/{j}");
                }
            }
            for i in 0..n - 1 {
                assert_eq!(work[i + (n - 1) * n], 0., "assigned final M column {n}/{i}");
            }
            for i in 0..n - 1 {
                let copied = if i % 2 == 1 {
                    0.
                } else if i == 0 {
                    -7.
                } else {
                    -2.
                };
                assert_eq!(vt[i], copied);
            }
            if n > 64 {
                // These exact assertions test explicit C zero/sign-copy assignments.
                // Not computed quotient equality or a whole C/Rust bitwise gate.
                for j in 0..n {
                    assert_eq!(factor[j + j * n], 0., "assigned diagonal {n}/{j}");
                    for i in j + 1..n {
                        assert_eq!(
                            factor[i + j * n],
                            -factor[j + i * n],
                            "assigned skew {n}/{i}/{j}"
                        );
                    }
                }
            }
            let mut g = vec![0.; n * n];
            for i in (0..n).step_by(2) {
                let v = if i == 0 { 1. / 7. } else { 0.5 };
                g[i + (i + 1) * n] = -v;
                g[i + 1 + i * n] = v;
            }
            g[2 * n] = 3. / 7.;
            g[1 + 2 * n] = -5. / 7.;
            g[2] = -3. / 7.;
            g[2 + n] = 5. / 7.;
            // Hand-declared permutation of index labels, no replay of production swaps.
            let mut map: Vec<_> = (0..n).collect();
            if permuted {
                map[..4].copy_from_slice(&[2, 3, 0, 1]);
            }
            let u = f64::EPSILON / 2.;
            let k = (2 * n + 4) as f64;
            let gamma = k * u / (1. - k * u);
            // Exact term envelope ||U^-T||inf*||T^-1||inf*||U^-1||inf=9*(1/2)*6=27.
            let budget = (2. * gamma / (1. - gamma)) * 27.;
            let prep_budget = gamma * 6. / (1. - gamma);
            for j in 0..n {
                for i in 0..n {
                    let expected = if i == j {
                        1.
                    } else if i == 0 && j == 2 {
                        -5.
                    } else if i == 1 && j == 2 {
                        -3.
                    } else {
                        0.
                    };
                    assert!(work[i + j * n].is_finite());
                    assert!(
                        (work[i + j * n] - expected).abs() <= prep_budget,
                        "independent TRTRI/M preparation {n}/{i}/{j}"
                    );
                }
            }
            for j in 0..n {
                for i in 0..n {
                    let x = factor[i + j * n];
                    let y = g[map[i] + map[j] * n];
                    assert!(x.is_finite());
                    assert!(
                        (x - y).abs() <= budget,
                        "analytic inverse {n}/{permuted}/{i}/{j}: {x} vs {y}"
                    );
                }
            }
            // Independent original A=U T U^T has only top4 active nontrivial block.
            // Build that exact dyadic algebra (no inverse kernel reuse), then permute.
            let mut t = vec![0.; n * n];
            for i in (0..n).step_by(2) {
                let v = if i == 0 { 7. } else { 2. };
                t[i + (i + 1) * n] = v;
                t[i + 1 + i * n] = -v;
            }
            let mut a = t.clone();
            for j in 0..n {
                a[j * n] += 5. * t[2 + j * n];
                a[1 + j * n] += 3. * t[2 + j * n];
            }
            let left = a.clone();
            for i in 0..n {
                a[i] += 5. * left[i + 2 * n];
                a[i + n] += 3. * left[i + 2 * n];
            }
            // All A construction values integral and <=32, exactly representable.
            // Residual allowance derives from forward envelope plus dot rounding,
            // not observed maxima. ||A||inf<=||U||inf||T||inf||U^T||inf=378.
            let dot_gamma = (2 * n) as f64 * u / (1. - (2 * n) as f64 * u);
            let residual_budget = 378. * budget + dot_gamma * 378. * (27. + budget);
            for i in 0..n {
                for j in 0..n {
                    let mut sum = 0.;
                    for k in 0..n {
                        sum += a[map[i] + map[k] * n] * factor[k + j * n];
                    }
                    let e = if i == j { 1. } else { 0. };
                    assert!(
                        (sum - e).abs() <= residual_budget,
                        "residual {n}/{permuted}/{i}/{j}"
                    );
                }
            }
        }
    }
}

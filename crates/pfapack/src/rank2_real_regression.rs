// Independent hand-dyadic operation and pivot controls. No runtime oracle.
use super::*;

#[test]
fn real_rank2_c_order_preserves_storage() {
    let p = 9_007_199_254_740_992.0;
    for kk0 in [2, 3, 4] {
        let lda = kk0 + 2;
        let k0 = kk0 + 1;
        let mut a = vec![3.0; lda * lda];
        for j in 0..kk0 {
            for i in 0..j {
                a[i + j * lda] = p;
            }
        }
        a[k0 * lda] = -p;
        a[kk0 * lda] = 1.0;
        for i in 1..kk0 {
            a[i + k0 * lda] = 1.0;
            a[i + kk0 * lda] = 1.0;
        }
        let before = a.clone();
        <f64 as UpperRank2Kernel>::update_upper_rank2(&mut a, lda, kk0, k0, 1.0);
        for index in 0..a.len() {
            let i = index % lda;
            let j = index / lda;
            let expected = if j < kk0 && i == j {
                0.0
            } else if j < kk0 && i < j {
                if i == 0 {
                    -1.0
                } else {
                    p - 1.0
                }
            } else {
                before[index]
            };
            if j < kk0 && i < j {
                // This bound is local to the proved dyadic tree, not gamma*p.
                // Products are exact; cancellation gives 0 then -1 exactly.
                // For i>0, p+1 ties to p, then p-1 is representable exactly.
                // The known one-unit rounding at p+1 is already in expected.
                // A positive unit-roundoff envelope cannot admit the grouped
                // result (error1); no arbitrary computed-bit equality gate.
                let absolute_bound = f64::EPSILON / 2.0;
                assert!(
                    (a[index] - expected).abs() <= absolute_bound,
                    "index={index} i={i} j={j} actual={} expected={expected} bound={absolute_bound}",
                    a[index]
                );
            } else {
                // Explicit assigned diagonal zero or unchanged copied storage.
                assert_eq!(a[index], expected, "index={index} i={i} j={j}");
            }
        }
    }
}

#[test]
fn real_factor_hand_dyadic_first_pivot_has_strict_margin() {
    // Upper last column magnitudes 1,2,4: maximum at row2, margin2.
    // This witnesses a fixed discrete search, not a generic portable tie gate.
    let n = 4;
    let mut a = vec![0.0; n * n];
    for (i, j, v) in [
        (0, 1, 3.0),
        (0, 2, 1.0),
        (1, 2, 2.0),
        (0, 3, 1.0),
        (1, 3, 2.0),
        (2, 3, 4.0),
    ] {
        a[i + j * n] = v;
        a[j + i * n] = -v;
    }
    let mut pivots = vec![PivotIndex1Based(0); n];
    dsktf2(&mut SqMat::new(&mut a, n), &mut pivots).expect("hand matrix Pf=12 is nonzero");
    assert_eq!(pivots[2], PivotIndex1Based(3));
}

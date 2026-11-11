#![cfg(feature = "faer-backend")]
use num_complex::Complex64;
use pfapack::faer_pfaffian::{ComplexPfaffianInverseWorkspace, RealPfaffianInverseWorkspace};
#[test]
fn permuted_pairs_have_small_backward_error_across_panels() {
    for n in [4, 8, 26, 64, 130] {
        for seed in 1u64..=3 {
            let mut state = seed;
            let mut draw = || {
                state = state.wrapping_mul(6364136223846793005u64).wrapping_add(1);
                (state >> 11) as f64 / ((1u64 << 53) as f64) - 0.5
            };
            let mut a = vec![Complex64::default(); n * n];
            for j in 0..n {
                for i in j + 1..n {
                    let value = Complex64::new(draw(), draw());
                    a[i + j * n] = value;
                    a[j + i * n] = -value;
                }
            }
            let real: Vec<_> = a.iter().map(|x| x.re).collect();
            let mut inv = vec![Complex64::default(); n * n];
            let mut inv_real = vec![0.0; n * n];
            ComplexPfaffianInverseWorkspace::new(n)
                .compute(&a, &mut inv)
                .unwrap();
            RealPfaffianInverseWorkspace::new(n)
                .compute(&real, &mut inv_real)
                .unwrap();
            let mut norm_a = 0.0f64;
            let mut norm_inv = 0.0f64;
            let mut norm_real_a = 0.0f64;
            let mut norm_real_inv = 0.0f64;
            let mut residual = 0.0f64;
            let mut residual_real = 0.0f64;
            for i in 0..n {
                norm_a = norm_a.max((0..n).map(|j| a[i + j * n].norm()).sum());
                norm_inv = norm_inv.max((0..n).map(|j| inv[i + j * n].norm()).sum());
                norm_real_a = norm_real_a.max((0..n).map(|j| real[i + j * n].abs()).sum());
                norm_real_inv = norm_real_inv.max((0..n).map(|j| inv_real[i + j * n].abs()).sum());
                let mut row = 0.0;
                let mut row_real = 0.0;
                for j in 0..n {
                    let mut r = -Complex64::new(f64::from(i == j), 0.0);
                    let mut rr = -f64::from(i == j);
                    for k in 0..n {
                        r = r + a[i + k * n] * inv[k + j * n];
                        rr += real[i + k * n] * inv_real[k + j * n];
                    }
                    row += r.norm();
                    row_real += rr.abs();
                }
                residual = residual.max(row);
                residual_real = residual_real.max(row_real);
            }
            // Normwise backward error, not forward accuracy on a possibly
            // ill-conditioned random matrix. The bound covers n-term dots
            // and both triangular factors without concealing conditioning.
            let bound = 64.0 * f64::EPSILON * n as f64;
            assert!(
                residual / (norm_a * norm_inv) <= bound,
                "complex n={n} seed={seed} residual={residual:e}"
            );
            assert!(
                residual_real / (norm_real_a * norm_real_inv) <= bound,
                "real n={n} seed={seed} residual={residual_real:e}"
            );
        }
    }
}

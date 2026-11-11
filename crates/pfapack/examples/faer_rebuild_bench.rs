//! Compare complete Pfaffian/inverse rebuilds with reusable scratch.
//! Run with OPENBLAS_NUM_THREADS=1 cargo run --release -p pfapack
//! --example faer_rebuild_bench --features 'faer-backend blas-backend'.

#[cfg(feature = "faer-backend")]
fn main() {
    use num_complex::Complex64;
    use pfapack::faer_pfaffian::{ComplexPfaffianInverseWorkspace, RealPfaffianInverseWorkspace};
    use pfapack::{PivotIndex1Based, SqMat};
    use std::hint::black_box;
    use std::time::{Duration, Instant};

    fn timing(mut run: impl FnMut()) -> f64 {
        run();
        let start = Instant::now();
        let mut reps = 0;
        while start.elapsed() < Duration::from_millis(250) {
            run();
            reps += 1;
        }
        start.elapsed().as_secs_f64() * 1e6 / reps as f64
    }

    println!("dtype,n,reference_us,faer_pf_lu_inverse_us,speedup");
    macro_rules! bench {
        ($t:ty, $ws:ty, $make:expr, $factor:path, $pfa:path, $inv:path, $label:literal) => {
            for n in [16, 32, 64, 128, 256, 512] {
                let zero = <$t>::default();
                let mut input = vec![zero; n * n];
                for j in 0..n {
                    for i in 0..j {
                        let re = if i % 2 == 0 && j == i + 1 {
                            8.0
                        } else {
                            ((i * 17 + j * 31) % 23) as f64 / 23.0 - 0.5
                        };
                        let value = ($make)(re, ((i * 11 + j * 7) % 19) as f64 / 38.0);
                        input[i + j * n] = value;
                        input[j + i * n] = -value;
                    }
                }
                let mut a = input.clone();
                let mut work = vec![zero; n * n];
                let mut vt = vec![zero; n - 1];
                let mut piv = vec![PivotIndex1Based(0); n];
                let reference = timing(|| {
                    a.copy_from_slice(&input);
                    let mut sm = SqMat::new(&mut a, n);
                    $factor(&mut sm, &mut piv).unwrap();
                    black_box($pfa(&sm, &piv));
                    $inv(&mut sm, &piv, &mut vt, &mut SqMat::new(&mut work, n));
                    black_box(&a);
                });
                let mut ws = <$ws>::new(n);
                let mut out = vec![zero; n * n];
                let candidate = timing(|| {
                    black_box(ws.compute(&input, &mut out).unwrap());
                    black_box(&out);
                });
                println!(
                    "{},{n},{reference:.3},{candidate:.3},{:.3}",
                    $label,
                    reference / candidate
                );
            }
        };
    }
    bench!(
        f64,
        RealPfaffianInverseWorkspace,
        |re, _im| re,
        pfapack::dsktf2,
        pfapack::utu2pfa_real,
        pfapack::utu2inv_real,
        "f64"
    );
    bench!(
        Complex64,
        ComplexPfaffianInverseWorkspace,
        Complex64::new,
        pfapack::zsktf2,
        pfapack::utu2pfa_complex,
        pfapack::utu2inv_complex,
        "c64"
    );
}

#[cfg(not(feature = "faer-backend"))]
fn main() {
    eprintln!("enable faer-backend to run this benchmark");
}

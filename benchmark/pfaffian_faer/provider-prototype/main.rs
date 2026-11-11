//! Compare complete Pfaffian/inverse rebuilds with reusable scratch.
//! Run with OPENBLAS_NUM_THREADS=1 cargo run --release -p pfapack
//! --example faer_rebuild_bench --features 'faer-backend blas-backend'.

#[cfg(feature = "faer-backend")]
fn main() {
    trait Norm {
        fn norm_value(self) -> f64;
    }
    impl Norm for f64 {
        fn norm_value(self) -> f64 {
            self.abs()
        }
    }
    impl Norm for num_complex::Complex64 {
        fn norm_value(self) -> f64 {
            self.norm()
        }
    }
    use faer::dyn_stack::{MemBuffer, MemStack};
    use faer::linalg::pfaffian::{pfaffian_inverse, pfaffian_inverse_scratch, PfaffianError};
    use faer::{MatMut, MatRef, Par};
    use num_complex::Complex64;
    macro_rules! workspace {
        ($name:ident,$t:ty) => {
            struct $name {
                n: usize,
                memory: MemBuffer,
            }
            impl $name {
                fn new(n: usize) -> Self {
                    Self {
                        n,
                        memory: MemBuffer::new(pfaffian_inverse_scratch::<$t>(n)),
                    }
                }
                fn compute(&mut self, a: &[$t], out: &mut [$t]) -> Result<$t, PfaffianError> {
                    pfaffian_inverse(
                        MatRef::from_column_major_slice(a, self.n, self.n),
                        MatMut::from_column_major_slice_mut(out, self.n, self.n),
                        Par::Seq,
                        MemStack::new(&mut self.memory),
                    )
                }
            }
        };
    }
    workspace!(RealPfaffianInverseWorkspace, f64);
    workspace!(ComplexPfaffianInverseWorkspace, Complex64);
    // Structural/numerical failures remain distinguishable; no catch-all
    // conversion of a numerical failure to a zero Pfaffian is allowed.
    assert_eq!(
        RealPfaffianInverseWorkspace::new(0).compute(&[], &mut []),
        Ok(1.0)
    );
    assert_eq!(
        RealPfaffianInverseWorkspace::new(3).compute(&[0.0; 9], &mut [0.0; 9]),
        Err(PfaffianError::OddDimension)
    );
    assert!(matches!(
        RealPfaffianInverseWorkspace::new(4).compute(&[0.0; 16], &mut [0.0; 16]),
        Err(PfaffianError::ZeroPivot { .. })
    ));
    assert_eq!(
        RealPfaffianInverseWorkspace::new(2).compute(&[0.0, f64::NAN, 1.0, 0.0], &mut [0.0; 4]),
        Err(PfaffianError::NumericalFailure)
    );
    assert_eq!(
        RealPfaffianInverseWorkspace::new(2).compute(&[0.0, -1e-320, 1e-320, 0.0], &mut [0.0; 4]),
        Err(PfaffianError::NumericalFailure)
    );
    let mut underflow = [0.0; 16];
    underflow[1] = -1e-200;
    underflow[4] = 1e-200;
    underflow[11] = -1e-200;
    underflow[14] = 1e-200;
    assert_eq!(
        RealPfaffianInverseWorkspace::new(4).compute(&underflow, &mut [0.0; 16]),
        Err(PfaffianError::NumericalFailure)
    );
    let mut empty = MemBuffer::new(faer::dyn_stack::StackReq::EMPTY);
    assert_eq!(
        pfaffian_inverse(
            MatRef::from_column_major_slice(&[0.0; 6], 2, 3),
            MatMut::from_column_major_slice_mut(&mut [0.0; 4], 2, 2),
            Par::Seq,
            MemStack::new(&mut empty)
        ),
        Err(PfaffianError::NonSquare)
    );
    // Strict-lower input authority also holds for the inverse, even when
    // the ignored upper triangle/diagonal contain NaNs. Padded windows prove
    // neither input nor output is assumed to have leading dimension n.
    let mut padded = faer::Mat::<f64>::from_fn(6, 6, |_, _| f64::NAN);
    padded[(2, 1)] = -2.0;
    padded[(3, 1)] = 0.0;
    padded[(4, 1)] = 0.0;
    padded[(3, 2)] = 0.0;
    padded[(4, 2)] = 0.0;
    padded[(4, 3)] = -5.0;
    let mut output = faer::Mat::<f64>::from_fn(6, 6, |_, _| 777.0);
    let mut scratch = MemBuffer::new(pfaffian_inverse_scratch::<f64>(4));
    let pf = pfaffian_inverse(
        padded.as_ref().subrows(1, 4).subcols(1, 4),
        output.as_mut().subrows_mut(1, 4).subcols_mut(1, 4),
        Par::Seq,
        MemStack::new(&mut scratch),
    )
    .unwrap();
    assert!((pf - 10.0).abs() <= 1e-14);
    assert!((output[(1, 2)] + 0.5).abs() <= 1e-15);
    assert!((output[(3, 4)] + 0.2).abs() <= 1e-15);
    for i in 0..6 {
        assert_eq!(output[(0, i)], 777.0);
        assert_eq!(output[(5, i)], 777.0);
        assert_eq!(output[(i, 0)], 777.0);
        assert_eq!(output[(i, 5)], 777.0);
    }
    macro_rules! verify {
        ($t:ty,$ws:ty,$make:expr) => {
            for n in [2, 6, 24, 26, 64, 130] {
                let mut state = 731u64 + n as u64;
                let mut draw = || {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    (state >> 11) as f64 / ((1u64 << 53) as f64) - 0.5
                };
                let mut a = vec![<$t>::default(); n * n];
                for j in 0..n {
                    for i in j + 1..n {
                        let v = ($make)(draw(), draw());
                        a[i + j * n] = v;
                        a[j + i * n] = -v;
                    }
                }
                let mut lower = vec![<$t>::default(); n * n];
                let mut diagonal = vec![<$t>::default(); n / 2];
                let mut perm = vec![0; n];
                let mut scratch = MemBuffer::new(faer::linalg::pfaffian::pfaffian_scratch::<$t>(n));
                faer::linalg::pfaffian::pfaffian_factor(
                    MatRef::from_column_major_slice(&a, n, n),
                    MatMut::from_column_major_slice_mut(&mut lower, n, n),
                    &mut perm,
                    &mut diagonal,
                    Par::Seq,
                    MemStack::new(&mut scratch),
                )
                .unwrap();
                for j in 0..n {
                    for i in 0..n {
                        let mut sum = <$t>::default();
                        for k in (0..n).step_by(2) {
                            sum += diagonal[k / 2]
                                * (lower[i + (k + 1) * n] * lower[j + k * n]
                                    - lower[i + k * n] * lower[j + (k + 1) * n]);
                        }
                        assert!(
                            (sum - a[perm[i] + perm[j] * n]).norm_value() < 1e-11,
                            "factor n={n} ({i},{j})"
                        );
                    }
                }
                let mut inv = vec![<$t>::default(); n * n];
                <$ws>::new(n).compute(&a, &mut inv).unwrap();
                let mut scale_a = 0.0f64;
                let mut scale_inv = 0.0f64;
                for i in 0..n {
                    scale_a = scale_a.max((0..n).map(|j| a[i + j * n].norm_value()).sum());
                    scale_inv = scale_inv.max((0..n).map(|j| inv[i + j * n].norm_value()).sum());
                }
                for j in 0..n {
                    for i in 0..n {
                        let mut sum = <$t>::default();
                        for k in 0..n {
                            sum += a[i + k * n] * inv[k + j * n];
                        }
                        let residual = (sum - ($make)(f64::from(i == j), 0.0)).norm_value();
                        assert!(
                            residual / (scale_a * scale_inv) < 1e-12,
                            "inverse n={n} ({i},{j}) {residual:e}"
                        );
                    }
                }
            }
        };
    }
    verify!(f64, RealPfaffianInverseWorkspace, |re, _im| re);
    verify!(Complex64, ComplexPfaffianInverseWorkspace, Complex64::new);
    #[cfg(feature = "provider-team")]
    {
        macro_rules! verify_team {
            ($t:ty, $make:expr) => {
                for n in [6, 130, 258] {
                    let mut a = vec![<$t>::default(); n * n];
                    for j in 0..n {
                        for i in j + 1..n {
                            let re = if j % 2 == 0 && i == j + 1 {
                                8.0
                            } else {
                                ((i * 17 + j * 31) % 23) as f64 / 23.0 - 0.5
                            };
                            let value = ($make)(re, ((i * 11 + j * 7) % 19) as f64 / 38.0);
                            a[i + j * n] = value;
                            a[j + i * n] = -value;
                        }
                    }
                    let mut output = vec![<$t>::default(); n * n];
                    let mut memory = MemBuffer::new(pfaffian_inverse_scratch::<$t>(n));
                    let mut run = |par: Par<'_>| {
                        let pf = pfaffian_inverse(
                            MatRef::from_column_major_slice(&a, n, n),
                            MatMut::from_column_major_slice_mut(&mut output, n, n),
                            par,
                            MemStack::new(&mut memory),
                        ).unwrap();
                        let mut norm_a = 0.0f64;
                        let mut norm_inv = 0.0f64;
                        for i in 0..n {
                            norm_a = norm_a.max((0..n).map(|j| a[i + j*n].norm_value()).sum());
                            norm_inv = norm_inv.max((0..n).map(|j| output[i + j*n].norm_value()).sum());
                        }
                        for j in 0..n {
                            for i in 0..n {
                                let mut sum = <$t>::default();
                                for k in 0..n {
                                    sum += a[i + k*n] * output[k + j*n];
                                }
                                let residual = (sum - ($make)(f64::from(i == j), 0.0)).norm_value();
                                assert!(residual <= 64.0 * f64::EPSILON * n as f64 * norm_a * norm_inv,
                                    "team inverse n={n} ({i},{j}) {residual:e}");
                            }
                        }
                        pf
                    };
                    let expected = run(Par::Seq);
                    for width in [2, 3, 8] {
                        let got = faer::team::with_team(width, |par| run(par));
                        // Stable bounded matrices; products have n/2 pivots.
                        assert!((got - expected).norm_value()
                            <= 64.0 * f64::EPSILON * n as f64 * expected.norm_value(),
                            "team Pfaffian n={n} width={width}");
                    }
                }
            };
        }
        verify_team!(f64, |re, _im| re);
        verify_team!(Complex64, Complex64::new);
        eprintln!("shared-factor team inverse/Pfaffian checks passed (2/3/8 participants)");
    }
    if std::env::var_os("MVMC_FAER_PROBE_VERIFY_ONLY").is_some() {
        return;
    }
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

    println!("dtype,n,reference_us,faer_shared_inverse_us,speedup");
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
                ws.compute(&input, &mut out).unwrap();
                for i in 0..n {
                    for j in 0..n {
                        let mut product = zero;
                        for k in 0..n {
                            product += input[i + k * n] * out[k + j * n];
                        }
                        let expected = ($make)(f64::from(i == j), 0.0);
                        assert!((product - expected).norm_value() < 1e-10, "n={n} ({i},{j})");
                    }
                }
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

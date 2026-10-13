use super::*;
use tenferro_gpu::cuda::{cuda_devices, CudaBackend};
use tenferro_tensor::BackendSessionHost;

#[test]
fn resident_geometry_fits_shared_memory_and_whole_warp_groups() {
    for n in (2..=crate::pfaffian::MAX_N).step_by(2) {
        for nqp in [1, 2, 3, 4, 8, 16, 32] {
            let g = IntervalGeometry {
                walkers: 1,
                n,
                ne: n / 2,
                nsite: n,
                nqp,
                ngutz: 0,
                nproj: 0,
            };
            if g.shared_bytes() <= 48 * 1024 {
                let (threads, groups, shared) = launch_geometry(g);
                assert!(shared <= 48 * 1024, "n={n} nqp={nqp}");
                assert_eq!(threads as usize % (32 * groups), 0);
                assert!(groups <= nqp);
            }
        }
    }
}

#[test]
#[ignore = "requires CUDA; queued inverse updates against independently refactored matrices"]
fn delayed_updates_flush_partial_queues_and_rebuild_boundaries() {
    use mvmc_gpu::{pfaffian_inverse_batched, testkit::random_planes, Backend};
    let mut backend = CudaBackend::new(cuda_devices().unwrap()[0].id()).unwrap();
    for (n, ns) in [(128, 80), (130, 80), (192, 128)] {
        let w = 2;
        let nq = 3; // a partial final QP group
        let g = IntervalGeometry {
            walkers: w,
            n,
            ne: n / 2,
            nsite: ns,
            nqp: nq,
            ngutz: 0,
            nproj: 0,
        };
        let slater = random_planes::<f64>(2 * ns, nq, 517 + n as u64);
        let ele: Vec<_> = (0..w * n).map(|i| (i % (n / 2)) as i32).collect();
        let gather = |cfg: &[i32]| {
            let mut matrices = vec![0.; w * nq * n * n];
            for walker in 0..w {
                for q in 0..nq {
                    for j in 0..n {
                        for i in 0..n {
                            let ri = cfg[walker * n + i] as usize + if i >= n / 2 { ns } else { 0 };
                            let rj = cfg[walker * n + j] as usize + if j >= n / 2 { ns } else { 0 };
                            matrices[(walker * nq + q) * n * n + i + j * n] =
                                slater[q * 4 * ns * ns + ri * 2 * ns + rj];
                        }
                    }
                }
            }
            matrices
        };
        let cpu = pfaffian_inverse_batched(&Backend::CpuPfapack, &gather(&ele), n, nq, w).unwrap();
        let inv: Vec<_> = cpu.inv.iter().map(|x| -x).collect();
        let gutz = vec![-1; ns];
        let jast = vec![-1; ns * ns];
        let host = IntervalHost {
            ele: &ele,
            inv: &inv,
            pf: &cpu.pf,
            slater: &slater,
            weights: &[1., 0.5, 0.25],
            gutz: &gutz,
            jast: &jast,
            params: &[],
        };
        let mut resident = backend
            .with_backend_session(|s| ResidentInterval::prepare(s, g, &host, 9, 79))
            .unwrap();
        let mut rng = sfmt19937::Sfmt19937Rng::new(79);
        let mut expected_cfg = ele.clone();
        let mut max_residual = 0f64;
        let mut max_relative = 0f64;
        let mut max_pf_error = 0f64;
        for (iteration, hops) in [0, 1, 2, 3, 4, 5, 9, 5].into_iter().enumerate() {
            let mut tape: Vec<_> = (0..w * hops * 5).map(|_| rng.gen_rand32()).collect();
            // Finite nonzero proposals always accept with log(u)=-Inf. This
            // deterministically covers every partial queue length and same-site
            // corrections while retaining independent random matrices/moves.
            for word in tape.iter_mut().skip(4).step_by(5) {
                *word = 0;
            }
            // Independent scalar replay of the proposal words, including the
            // partially occupied final site group and repeated electron slots.
            for hop in 0..hops {
                for walker in 0..w {
                    let words = &tape[5 * (walker + w * hop)..];
                    let spin = (words[1] >> 31) as usize;
                    let slot = words[0] as usize % (n / 2);
                    let cfg = &mut expected_cfg
                        [walker * n + spin * n / 2..walker * n + (spin + 1) * n / 2];
                    let empty: Vec<_> = (0..ns as i32).filter(|r| !cfg.contains(r)).collect();
                    cfg[slot] = empty[words[2] as usize % empty.len()];
                }
            }
            let borrowed = &mut resident;
            let (cfg, actual_inv, pf, accepted) = backend
                .with_backend_session(move |s| {
                    with_raw_exec(s, "delayed.residual", |exec| {
                        let r = &borrowed.sampler;
                        unsafe {
                            exec.upload(exec.ptr(&r.tape)?, &tape)?;
                            if iteration == 7 {
                                exec.upload(exec.ptr(&r.aw)?, &[ns as i32; 2])?;
                            }
                        }
                        borrowed.enqueue_hops(exec, hops)?;
                        let mut cfg = vec![0i32; w * n];
                        let mut inv = vec![0.; w * nq * n * n];
                        let mut pf = vec![0.; w * nq];
                        let mut accepted = vec![0i32; w];
                        let mut error = [0i32];
                        unsafe {
                            exec.download(&mut cfg, exec.ptr(&r.ele)?)?;
                            exec.download(&mut inv, exec.ptr(&r.inv)?)?;
                            exec.download(&mut pf, exec.ptr(&r.pf)?)?;
                            exec.download(&mut accepted, exec.ptr(&borrowed.total_accepts)?)?;
                            exec.download(&mut error, exec.ptr(&borrowed.error)?)?;
                        }
                        assert_eq!(error, [0]);
                        Ok((cfg, inv, pf, accepted))
                    })
                })
                .unwrap();
            assert_eq!(accepted, vec![hops as i32; w]);
            assert_eq!(cfg, expected_cfg);
            let planes = gather(&cfg);
            let cpu = pfaffian_inverse_batched(&Backend::CpuPfapack, &planes, n, nq, w).unwrap();
            for q in 0..w * nq {
                let a = &planes[q * n * n..(q + 1) * n * n];
                let expected = &cpu.inv[q * n * n..(q + 1) * n * n];
                let actual = &actual_inv[q * n * n..(q + 1) * n * n];
                let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
                let cond = norm(a) * norm(expected);
                // Accumulated rank-two updates use the same dimension/condition
                // bound as the factor gate, scaled by the 29 moves in this chain.
                let bound = 16. * 29. * n as f64 * f64::EPSILON * cond;
                let mut residual = 0f64;
                for j in 0..n {
                    for i in 0..n {
                        let x = (0..n)
                            .map(|k| -a[i + k * n] * actual[k + j * n])
                            .sum::<f64>();
                        residual = residual.max((x - f64::from(i == j)).abs());
                    }
                }
                let relative = actual
                    .iter()
                    .zip(expected)
                    .map(|(x, y)| (x + y).powi(2))
                    .sum::<f64>()
                    .sqrt()
                    / norm(expected);
                let pf_error = ((pf[q] - cpu.pf[q]) / cpu.pf[q]).abs();
                max_residual = max_residual.max(residual);
                max_relative = max_relative.max(relative);
                max_pf_error = max_pf_error.max(pf_error);
                assert!(residual<=bound && relative<=bound && pf_error<=bound,
                    "n={n} hop={hops} plane={q} residual={residual:e} inverse={relative:e} pf={pf_error:e} bound={bound:e}");
            }
        }
        eprintln!("delayed n={n} max inverse residual={max_residual:e} relative inverse={max_relative:e} relative pf={max_pf_error:e}");
    }
}

#[test]
#[ignore = "requires CUDA; independent CPU factors and inverse residuals"]
fn paired_rebuild_matches_cpu_and_preserves_failed_or_masked_walkers() {
    use mvmc_gpu::{pfaffian_inverse_batched, testkit::random_planes, Backend};
    let mut backend = CudaBackend::new(cuda_devices().unwrap()[0].id()).unwrap();
    // Non-power-of-two dimensions exercise partial panels; 512 reduces panel
    // capacity to fit the shared-memory budget. Cancellation invokes fallback.
    for (n, cancellation) in [
        (128, false),
        (130, false),
        (256, false),
        (512, false),
        (128, true),
    ] {
        let nq = 2;
        let g = IntervalGeometry {
            walkers: 2,
            n,
            ne: n / 2,
            nsite: n / 2,
            nqp: nq,
            ngutz: 0,
            nproj: 0,
        };
        let mut planes = random_planes::<f64>(n, nq, 771 + n as u64);
        if cancellation {
            let (first, second) = planes.split_at_mut(n * n);
            second.copy_from_slice(first);
        }
        let cpu = pfaffian_inverse_batched(&Backend::CpuPfapack, &planes, n, nq, 1).unwrap();
        // Slater is row-major; CPU factorization planes are column-major.
        let slater: Vec<_> = planes.iter().map(|x| -x).collect();
        let ele: Vec<_> = (0..g.walkers * n).map(|i| (i % g.ne) as i32).collect();
        let initial_inv = vec![0.; g.walkers * nq * n * n];
        let initial_pf = vec![1.; g.walkers * nq];
        let weights = [
            1.,
            if cancellation {
                -1. + 2f64.powi(-40)
            } else {
                0.125
            },
        ];
        let gutz = vec![-1; g.nsite];
        let jast = vec![-1; g.nsite * g.nsite];
        let host = IntervalHost {
            ele: &ele,
            inv: &initial_inv,
            pf: &initial_pf,
            slater: &slater,
            weights: &weights,
            gutz: &gutz,
            jast: &jast,
            params: &[],
        };
        let resident = backend
            .with_backend_session(|s| ResidentInterval::prepare(s, g, &host, 0, 79))
            .unwrap();
        backend.with_backend_session(move |s| {
            with_raw_exec(s, "paired.verify", |exec| {
            let module = exec.module(INTERVAL_MODULE, INTERVAL_KERNELS)?;
            let r = &resident.sampler;
            unsafe { exec.upload(exec.ptr(&r.rebuild)?, &[1i32, 0])?; }
            resident.enqueue_rebuild(exec, &module)?;
            let mut inv = initial_inv.clone();
            let mut pf = initial_pf.clone();
            let mut fallback = [0i32; 2];
            let mut error = [0i32];
            unsafe {
                exec.download(&mut inv, exec.ptr(&r.inv)?)?;
                exec.download(&mut pf, exec.ptr(&r.pf)?)?;
                exec.download(&mut fallback, exec.ptr(&resident.fallback)?)?;
                exec.download(&mut error, exec.ptr(&resident.error)?)?;
            }
            assert_eq!(error, [0]);
            assert_eq!(fallback, [i32::from(cancellation), 0]);
            assert!(inv[nq*n*n..].iter().all(|x| x.abs() < 1e-300));
            assert!(pf[nq..].iter().all(|x| (x - 1.).abs() < 1e-15));
            for q in 0..nq {
                let a = &planes[q*n*n..(q+1)*n*n];
                let expected = &cpu.inv[q*n*n..(q+1)*n*n];
                let actual = &inv[q*n*n..(q+1)*n*n];
                let norm = |v: &[f64]| v.iter().map(|x| x*x).sum::<f64>().sqrt();
                let cond = norm(a) * norm(expected);
                let bound = 16. * n as f64 * f64::EPSILON * cond;
                let diff = actual.iter().zip(expected).map(|(x,y)| (x+y).powi(2)).sum::<f64>().sqrt() / norm(expected);
                let pf_error = ((pf[q] - cpu.pf[q]) / cpu.pf[q]).abs();
                assert!(diff <= bound && pf_error <= bound, "n={n} q={q} inverse={diff:e} pf={pf_error:e} bound={bound:e}");
                let mut residual = 0f64;
                for j in 0..n { for i in 0..n {
                    let v = (0..n).map(|k| -a[i+k*n]*actual[k+j*n]).sum::<f64>();
                    residual = residual.max((v - f64::from(i==j)).abs());
                }}
                assert!(residual <= bound, "n={n} residual={residual:e} bound={bound:e}");
                eprintln!("paired n={n} q={q} cancellation={cancellation} inverse={diff:e} pf={pf_error:e} residual={residual:e} bound={bound:e}");
            }
            // Both the paired attempt and its conventional retry must fail on
            // a singular matrix, without publishing over the previous tables.
            unsafe { exec.zero_bytes(exec.ptr(&r.slater)?, slater.len()*8)?; }
            resident.enqueue_rebuild(exec, &module)?;
            let mut failed_inv = vec![0.; inv.len()];
            let mut failed_pf = vec![0.; pf.len()];
            unsafe {
                exec.download(&mut failed_inv, exec.ptr(&r.inv)?)?;
                exec.download(&mut failed_pf, exec.ptr(&r.pf)?)?;
                exec.download(&mut error, exec.ptr(&resident.error)?)?;
            }
            assert_eq!(error, [1]);
            assert!(failed_inv.iter().zip(&inv).all(|(a,b)| (a-b).abs() <= 1e-15));
            assert!(failed_pf.iter().zip(&pf).all(|(a,b)| (a-b).abs() <= 1e-15));
            Ok(())
            })
        }).unwrap();
    }
}

#[test]
#[ignore = "requires CUDA"]
fn failed_rebuild_is_latched_and_cannot_publish_results() {
    let devices = cuda_devices().unwrap();
    let mut backend = CudaBackend::new(devices[0].id()).unwrap();
    let g = IntervalGeometry {
        walkers: 2,
        n: 2,
        ne: 1,
        nsite: 2,
        nqp: 1,
        ngutz: 0,
        nproj: 0,
    };
    let mut slater = vec![0.0f64; 16];
    for i in 0..4 {
        for j in 0..i {
            slater[4 * i + j] = 1.;
            slater[4 * j + i] = -1.;
        }
    }
    let inv = [0., -1., 1., 0., 0., -1., 1., 0.];
    let h = IntervalHost {
        ele: &[0, 0, 1, 1],
        inv: &inv,
        pf: &[1., 1.],
        slater: &slater,
        weights: &[1.],
        gutz: &[-1; 2],
        jast: &[-1; 4],
        params: &[],
    };
    let mut resident = backend
        .with_backend_session(|s| ResidentInterval::prepare(s, g, &h, 7, 79))
        .unwrap();
    let borrowed = &mut resident;
    backend
        .with_backend_session(move |s| {
            with_raw_exec(s, "resident.failure.inject", |exec| {
                // Replace the Slater table by a singular one at the parameter-refresh
                // boundary, then request a rebuild. Failed factors must not publish zeros.
                let sampler = &borrowed.sampler;
                let module = exec.module(INTERVAL_MODULE, INTERVAL_KERNELS)?;
                unsafe {
                    exec.zero_bytes(exec.ptr(&sampler.slater)?, 16 * 8)?;
                    exec.upload(exec.ptr(&sampler.rebuild)?, &[1i32, 1])?;
                }
                borrowed.enqueue_rebuild(exec, &module)?;
                let mut actual_inv = [0.0f64; 8];
                let mut actual_pf = [0.0f64; 2];
                unsafe {
                    exec.download(&mut actual_inv, exec.ptr(&sampler.inv)?)?;
                    exec.download(&mut actual_pf, exec.ptr(&sampler.pf)?)?;
                }
                for (a, b) in actual_inv.iter().zip(inv) {
                    assert!((a - b).abs() < 1e-15);
                }
                for p in actual_pf {
                    assert!((p - 1.).abs() < 1e-15);
                }
                Ok(())
            })
        })
        .unwrap();
    assert!(backend
        .with_backend_session(|s| resident.check_status(s))
        .is_err());
    backend
        .with_backend_session(|s| resident.advance(s))
        .unwrap();
    assert!(backend
        .with_backend_session(|s| resident.snapshot(s))
        .is_err());
    assert!(backend
        .with_backend_session(|s| resident.accepted(s))
        .is_err());
}

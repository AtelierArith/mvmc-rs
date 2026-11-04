use super::*;
use tenferro_gpu::cuda::{cuda_devices, CudaEngine};

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
    let mut backend = CudaEngine::open(cuda_devices().unwrap()[0].id()).unwrap();
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
        let mut resident = {
            let s: &mut CudaEngine = &mut backend;
            ResidentInterval::prepare(s, g, &host, 9, 79)
        }
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
            let (cfg, actual_inv, pf, accepted) = {
                let s: &mut CudaEngine = &mut backend;
                {
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
                }
            }
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
    let mut backend = CudaEngine::open(cuda_devices().unwrap()[0].id()).unwrap();
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
        let resident = {
            let s: &mut CudaEngine = &mut backend;
            ResidentInterval::prepare(s, g, &host, 0, 79)
        }
        .unwrap();
        { let s: &mut CudaEngine = &mut backend; {
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
        } }.unwrap();
    }
}

#[test]
#[ignore = "requires CUDA"]
fn failed_rebuild_is_latched_and_cannot_publish_results() {
    let devices = cuda_devices().unwrap();
    let mut backend = CudaEngine::open(devices[0].id()).unwrap();
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
    let mut resident = {
        let s: &mut CudaEngine = &mut backend;
        ResidentInterval::prepare(s, g, &h, 7, 79)
    }
    .unwrap();
    let borrowed = &mut resident;
    {
        let s: &mut CudaEngine = &mut backend;
        {
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
        }
    }
    .unwrap();
    assert!({
        let s: &mut CudaEngine = &mut backend;
        resident.check_status(s)
    }
    .is_err());
    {
        let s: &mut CudaEngine = &mut backend;
        resident.advance(s)
    }
    .unwrap();
    assert!({
        let s: &mut CudaEngine = &mut backend;
        resident.snapshot(s)
    }
    .is_err());
    assert!({
        let s: &mut CudaEngine = &mut backend;
        resident.accepted(s)
    }
    .is_err());
}

#[test]
fn status_codes_match_cuda_header() {
    let header = include_str!("../../resident_status.cuh");
    let mut seen = 0;
    for line in header.lines() {
        let Some((name, rest)) = line.trim().split_once(" = ") else {
            continue;
        };
        let code: i32 = rest.split(',').next().unwrap().trim().parse().unwrap();
        let expected = match name {
            "RESIDENT_REBUILD" => ResidentFailure::Rebuild,
            "RESIDENT_ENERGY" => ResidentFailure::Energy,
            "RESIDENT_SR" => ResidentFailure::Sr,
            "RESIDENT_PARAMETERS" => ResidentFailure::Parameters,
            "RESIDENT_SCHEDULE" => ResidentFailure::Schedule,
            "RESIDENT_DERIVATIVES" => ResidentFailure::Derivatives,
            other => panic!("unmapped status {other}"),
        };
        assert_eq!(ResidentFailure::from_status(code), Some(expected), "{name}");
        assert_eq!(expected.code(), Some(code));
        seen += 1;
    }
    assert_eq!(seen, 6);
    assert_eq!(ResidentFailure::from_status(0), None);
    assert_eq!(
        ResidentFailure::from_status(99),
        Some(ResidentFailure::Unknown(99))
    );
    assert!(ResidentFailure::Sr.to_string().contains("device status 3"));
}

/// Column-major skew planes of the electron sites `cfg` (`[walker][n]`), as
/// gathered by `interval_resident_factor` from a row-major Slater table.
fn gather_planes(cfg: &[i32], slater: &[f64], n: usize, ns: usize, nq: usize) -> Vec<f64> {
    let (w, ne) = (cfg.len() / n, n / 2);
    let mut matrices = vec![0.; w * nq * n * n];
    for walker in 0..w {
        for q in 0..nq {
            for j in 0..n {
                for i in 0..n {
                    let ri = cfg[walker * n + i] as usize + if i >= ne { ns } else { 0 };
                    let rj = cfg[walker * n + j] as usize + if j >= ne { ns } else { 0 };
                    matrices[(walker * nq + q) * n * n + i + j * n] =
                        slater[q * 4 * ns * ns + ri * 2 * ns + rj];
                }
            }
        }
    }
    matrices
}

/// Independent CPU Pfaffians (PFAPACK); a zero pivot is an exact zero.
fn cpu_pfaffians(planes: &[f64], n: usize) -> Vec<f64> {
    use mvmc_gpu::{pfaffian_inverse_batched, Backend, PlaneStatus};
    let count = planes.len() / (n * n);
    let cpu = pfaffian_inverse_batched(&Backend::CpuPfapack, planes, n, count, 1).unwrap();
    cpu.pf
        .iter()
        .zip(&cpu.status)
        .map(|(&p, s)| if *s == PlaneStatus::Ok { p } else { 0. })
        .collect()
}

/// Max residual of `A * (-inv) - I` and the delayed-update bound for `moves`.
fn inverse_check(a: &[f64], inv: &[f64], reference: &[f64], n: usize, moves: f64) -> (f64, f64) {
    let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let bound = 16. * moves * n as f64 * f64::EPSILON * norm(a) * norm(reference);
    let mut residual = 0f64;
    for j in 0..n {
        for i in 0..n {
            let x = (0..n).map(|k| -a[i + k * n] * inv[k + j * n]).sum::<f64>();
            residual = residual.max((x - f64::from(i == j)).abs());
        }
    }
    (residual, bound)
}

/// Issue #573, Julia `cached_sampling.jl` `:zero_qp`. An accepted hop whose
/// total overlap is finite and nonzero while one candidate QP plane Pfaffian
/// is exactly zero refactors the walker in the hop kernel. The singular plane
/// becomes invalid (Pf 0) and the walker continues with full Pfaffians. A later
/// move off the singular site makes it valid again, back on the inverse path.
/// The all-singular case is an ordinary rejection. Every decision and table is
/// checked against an independent CPU replay with PFAPACK Pfaffians.
#[test]
#[ignore = "requires CUDA; per-plane recovery against independent CPU factors"]
fn accepted_hop_with_singular_candidate_plane_recovers_per_plane() {
    use mvmc_gpu::{pfaffian_inverse_batched, testkit::random_planes, Backend};
    let mut backend = CudaEngine::open(cuda_devices().unwrap()[0].id()).unwrap();
    // n=2: immediate updates; n=128: the four-update queue holds two pending
    // updates when the singular proposal arrives.
    for ((n, ns), all_planes) in [(2, 2), (128, 80)]
        .into_iter()
        .flat_map(|g| [(g, false), (g, true)])
    {
        let (w, nq, ne) = (2, 2, n / 2);
        let weights = [1., 0.5];
        let g = IntervalGeometry {
            walkers: w,
            n,
            ne,
            nsite: ns,
            nqp: nq,
            ngutz: 0,
            nproj: 0,
        };
        let mut slater = random_planes::<f64>(2 * ns, nq, 573 + n as u64);
        // Up-spin site `ne` is empty initially and is the first empty
        // destination. Zeroing its row/column makes every configuration that
        // occupies it exactly singular in the selected planes.
        for q in 0..if all_planes { nq } else { 1 } {
            for r in 0..2 * ns {
                slater[q * 4 * ns * ns + ne * 2 * ns + r] = 0.;
                slater[q * 4 * ns * ns + r * 2 * ns + ne] = 0.;
            }
        }
        let ele: Vec<_> = (0..w * n).map(|i| (i % ne) as i32).collect();
        let cpu = pfaffian_inverse_batched(
            &Backend::CpuPfapack,
            &gather_planes(&ele, &slater, n, ns, nq),
            n,
            nq,
            w,
        )
        .unwrap();
        assert!(cpu.pf.iter().all(|p| p.is_finite() && *p != 0.));
        let inv: Vec<_> = cpu.inv.iter().map(|x| -x).collect();
        let (gutz, jast) = (vec![-1; ns], vec![-1; ns * ns]);
        let host = IntervalHost {
            ele: &ele,
            inv: &inv,
            pf: &cpu.pf,
            slater: &slater,
            weights: &weights,
            gutz: &gutz,
            jast: &jast,
            params: &[],
        };
        let hops = 5;
        let mut resident = {
            let s: &mut CudaEngine = &mut backend;
            ResidentInterval::prepare(s, g, &host, hops, 79)
        }
        .unwrap();
        // Walker 0: two down hops, the up hop of slot s0 to site `ne`, a down
        // hop, then slot s0 moves off `ne`. Walker 1: five down hops. u=0
        // accepts every finite nonzero total.
        let mut rng = sfmt19937::Sfmt19937Rng::new(573);
        let mut tape: Vec<u32> = (0..w * hops * 5).map(|_| rng.gen_rand32()).collect();
        let slot = tape[5 * (2 * w)] % ne as u32;
        for hop in 0..hops {
            for walker in 0..w {
                let t = &mut tape[5 * (walker + w * hop)..5 * (walker + w * hop) + 5];
                t[4] = 0;
                if walker == 0 && (hop == 2 || hop == 4) {
                    t[0] = slot;
                    t[1] &= !(1 << 31);
                    if hop == 2 {
                        t[2] = 0;
                    }
                } else {
                    t[1] |= 1 << 31;
                }
            }
        }
        // Independent CPU replay: proposal words, then exact CPU totals.
        let mut expected = ele.clone();
        let mut expected_accepts = [0; 2];
        let mut checkpoints = vec![];
        for hop in 0..hops {
            for walker in 0..w {
                let words = &tape[5 * (walker + w * hop)..];
                let spin = (words[1] >> 31) as usize;
                let mut candidate = expected[walker * n..(walker + 1) * n].to_vec();
                let c = &mut candidate[spin * ne..(spin + 1) * ne];
                let empty: Vec<_> = (0..ns as i32).filter(|r| !c.contains(r)).collect();
                c[words[0] as usize % ne] = empty[words[2] as usize % empty.len()];
                let pf = cpu_pfaffians(&gather_planes(&candidate, &slater, n, ns, nq), n);
                let total = pf.iter().zip(weights).map(|(p, w)| p * w).sum::<f64>();
                if total.is_finite() && total != 0. {
                    expected[walker * n..(walker + 1) * n].copy_from_slice(&candidate);
                    expected_accepts[walker] += 1;
                }
            }
            checkpoints.push(expected.clone());
        }
        assert!(checkpoints[2][0..ne].contains(&(ne as i32)) || all_planes);
        assert!(!checkpoints[4][0..ne].contains(&(ne as i32)));
        let label = format!("n={n} all_planes={all_planes}");
        // Two calls: three proposals, then the remaining two from a new tape.
        for (call, range) in [(0usize, 0..3usize), (1, 3..5)] {
            let segment = &tape[5 * w * range.start..5 * w * range.end];
            let borrowed = &mut resident;
            let (cfg, actual_inv, pf, valid, logip, error) = {
                let s: &mut CudaEngine = &mut backend;
                with_raw_exec(s, "singular.plane", |exec| {
                    let r = &borrowed.sampler;
                    unsafe {
                        exec.upload(exec.ptr(&r.tape)?, segment)?;
                    }
                    borrowed.enqueue_hops(exec, range.len())?;
                    let mut cfg = vec![0i32; w * n];
                    let mut inv = vec![0f64; w * nq * n * n];
                    let mut pf = vec![0f64; w * nq];
                    let mut valid = vec![0i32; w * nq];
                    let mut logip = vec![0f64; w];
                    let mut error = [0i32];
                    unsafe {
                        exec.download(&mut cfg, exec.ptr(&r.ele)?)?;
                        exec.download(&mut inv, exec.ptr(&r.inv)?)?;
                        exec.download(&mut pf, exec.ptr(&r.pf)?)?;
                        exec.download(&mut valid, exec.ptr(&borrowed.valid)?)?;
                        exec.download(&mut logip, exec.ptr(&r.logip)?)?;
                        exec.download(&mut error, exec.ptr(&borrowed.error)?)?;
                    }
                    Ok((cfg, inv, pf, valid, logip, error[0]))
                })
            }
            .unwrap();
            assert_eq!(error, 0, "{label} call={call}");
            assert_eq!(cfg, checkpoints[range.end - 1], "{label} call={call}");
            let expected_valid: Vec<i32> = if call == 0 && !all_planes {
                vec![0, 1, 1, 1]
            } else {
                vec![1; 4]
            };
            assert_eq!(valid, expected_valid, "{label} call={call}");
            let planes = gather_planes(&cfg, &slater, n, ns, nq);
            let reference =
                pfaffian_inverse_batched(&Backend::CpuPfapack, &planes, n, nq, w).unwrap();
            let exact = cpu_pfaffians(&planes, n);
            for walker in 0..w {
                let ip = (0..nq)
                    .map(|q| weights[q] * exact[walker * nq + q])
                    .sum::<f64>();
                assert!(
                    (logip[walker] - ip.abs().ln()).abs() <= 1e-10,
                    "{label} call={call} walker={walker} logip"
                );
            }
            for plane in 0..w * nq {
                if valid[plane] == 0 {
                    assert_eq!(pf[plane], 0., "{label} invalid plane publishes Pf=0");
                    continue;
                }
                let span = plane * n * n..(plane + 1) * n * n;
                let (residual, bound) = inverse_check(
                    &planes[span.clone()],
                    &actual_inv[span.clone()],
                    &reference.inv[span],
                    n,
                    5.,
                );
                let pf_error = ((pf[plane] - exact[plane]) / exact[plane]).abs();
                assert!(
                    residual <= bound && pf_error <= bound,
                    "{label} call={call} plane={plane} residual={residual:e} pf={pf_error:e} bound={bound:e}"
                );
            }
        }
        // `accepted` reports the most recent call (proposals 3 and 4); an
        // accepted hop always moves to an empty site.
        let expected_last: Vec<i32> = (0..w)
            .map(|walker| {
                (3..5)
                    .filter(|&hop| {
                        checkpoints[hop][walker * n..(walker + 1) * n]
                            != checkpoints[hop - 1][walker * n..(walker + 1) * n]
                    })
                    .count() as i32
            })
            .collect();
        let accepted = {
            let s: &mut CudaEngine = &mut backend;
            resident.accepted(s)
        }
        .unwrap();
        assert_eq!(accepted, expected_last, "{label}");
        // Every proposal with a nonzero total accepts; only the all-singular
        // destination of walker 0 can be rejected.
        assert_eq!(expected_accepts[1], 5, "{label}");
        if !all_planes {
            assert_eq!(expected_accepts[0], 5, "{label}");
        }
    }
}

/// Julia `cached_sampling.jl` `:invalid` and `:threshold`: a walker whose
/// plane is flagged invalid although nonsingular takes the full-Pfaffian
/// proposal path and refactors on its first accept. Its trajectory matches
/// the inverse path; the flag is restored by that refactor. A preset accepted
/// window also crosses the scheduled rebuild boundary on the first accept.
#[test]
#[ignore = "requires CUDA; full-Pfaffian proposals against the inverse path"]
fn invalid_flag_full_pfaffian_path_matches_inverse_path() {
    use mvmc_gpu::{pfaffian_inverse_batched, testkit::random_planes, Backend};
    let mut backend = CudaEngine::open(cuda_devices().unwrap()[0].id()).unwrap();
    for (n, ns) in [(16, 12), (128, 80)] {
        for threshold in [false, true] {
            let (w, nq, ne, hops) = (3, 2, n / 2, 20);
            let g = IntervalGeometry {
                walkers: w,
                n,
                ne,
                nsite: ns,
                nqp: nq,
                ngutz: 0,
                nproj: 0,
            };
            let slater = random_planes::<f64>(2 * ns, nq, 88 + n as u64);
            let ele: Vec<_> = (0..w * n).map(|i| (i % ne) as i32).collect();
            let cpu = pfaffian_inverse_batched(
                &Backend::CpuPfapack,
                &gather_planes(&ele, &slater, n, ns, nq),
                n,
                nq,
                w,
            )
            .unwrap();
            let inv: Vec<_> = cpu.inv.iter().map(|x| -x).collect();
            let (gutz, jast) = (vec![-1; ns], vec![-1; ns * ns]);
            let host = IntervalHost {
                ele: &ele,
                inv: &inv,
                pf: &cpu.pf,
                slater: &slater,
                weights: &[1., 0.25],
                gutz: &gutz,
                jast: &jast,
                params: &[],
            };
            let mut rng = sfmt19937::Sfmt19937Rng::new(9281 + n as u32);
            let tape: Vec<u32> = (0..w * hops * 5).map(|_| rng.gen_rand32()).collect();
            let mut results = vec![];
            for flagged in [false, true] {
                let mut resident = {
                    let s: &mut CudaEngine = &mut backend;
                    ResidentInterval::prepare(s, g, &host, hops, 79)
                }
                .unwrap();
                let borrowed = &mut resident;
                let result = {
                    let s: &mut CudaEngine = &mut backend;
                    with_raw_exec(s, "invalid.flag", |exec| {
                        let r = &borrowed.sampler;
                        unsafe {
                            exec.upload(exec.ptr(&r.tape)?, &tape)?;
                            if flagged {
                                exec.upload(exec.ptr(&borrowed.valid)?, &[1i32, 0, 1, 1, 0, 1])?;
                            }
                            if threshold {
                                exec.upload(exec.ptr(&r.aw)?, &vec![ns as i32; w])?;
                            }
                        }
                        borrowed.enqueue_hops(exec, hops)?;
                        let mut cfg = vec![0i32; w * n];
                        let mut inv = vec![0f64; w * nq * n * n];
                        let mut pf = vec![0f64; w * nq];
                        let mut valid = vec![0i32; w * nq];
                        let mut logip = vec![0f64; w];
                        let mut accepted = vec![0i32; w];
                        let mut error = [0i32];
                        unsafe {
                            exec.download(&mut cfg, exec.ptr(&r.ele)?)?;
                            exec.download(&mut inv, exec.ptr(&r.inv)?)?;
                            exec.download(&mut pf, exec.ptr(&r.pf)?)?;
                            exec.download(&mut valid, exec.ptr(&borrowed.valid)?)?;
                            exec.download(&mut logip, exec.ptr(&r.logip)?)?;
                            exec.download(&mut accepted, exec.ptr(&borrowed.total_accepts)?)?;
                            exec.download(&mut error, exec.ptr(&borrowed.error)?)?;
                        }
                        assert_eq!(error, [0]);
                        Ok((cfg, inv, pf, valid, logip, accepted))
                    })
                }
                .unwrap();
                results.push(result);
            }
            let (reference, flagged) = (&results[0], &results[1]);
            let label = format!("n={n} threshold={threshold}");
            assert!(flagged.5[0] > 0 && flagged.5[1] > 0, "{label}: no accept");
            assert_eq!(flagged.0, reference.0, "{label} configurations");
            assert_eq!(flagged.5, reference.5, "{label} accepts");
            assert_eq!(
                flagged.3,
                vec![1; w * nq],
                "{label} refactor restores validity"
            );
            // Pfaffian ratios and the refactor round differently from rank-two
            // updates; both are checked against independent CPU factors.
            let planes = gather_planes(&reference.0, &slater, n, ns, nq);
            let exact = pfaffian_inverse_batched(&Backend::CpuPfapack, &planes, n, nq, w).unwrap();
            for result in [reference, flagged] {
                for plane in 0..w * nq {
                    let span = plane * n * n..(plane + 1) * n * n;
                    let (residual, bound) = inverse_check(
                        &planes[span.clone()],
                        &result.1[span.clone()],
                        &exact.inv[span],
                        n,
                        hops as f64,
                    );
                    let pf_error = ((result.2[plane] - exact.pf[plane]) / exact.pf[plane]).abs();
                    assert!(
                        residual <= bound && pf_error <= bound,
                        "{label} plane={plane} residual={residual:e} pf={pf_error:e} bound={bound:e}"
                    );
                }
                for walker in 0..w {
                    let ip = exact.pf[walker * nq] + 0.25 * exact.pf[walker * nq + 1];
                    assert!(
                        (result.4[walker] - ip.abs().ln()).abs() <= 1e-9,
                        "{label} logip"
                    );
                }
            }
        }
    }
}

/// Test-only entry for the dense fallback helpers. One block refactors `count`
/// skew matrices, given as the planes of a table with identity sites, with the
/// same routine as an in-kernel walker rebuild. Kept out of the production
/// module so it adds no compile or JIT time there.
const PROBE_MODULE: &str = concat!(
    include_str!("../../resident_dense.cuh"),
    r#"
extern "C" __global__ void resident_dense_probe(const double* table, double* inv, double* pf,
    int* valid, double* fac, double* work, int n, int count) {
  extern __shared__ char probe_mem[];
  int* rs = reinterpret_cast<int*>(probe_mem);
  double* vec = reinterpret_cast<double*>(rs + ((n + 1) & ~1));
  for (int i = threadIdx.x; i < n; i += blockDim.x) rs[i] = i;
  __syncthreads();
  resident_refactor_walker(inv, pf, valid, rs, table, n, count, n, fac, work, vec);
}
"#
);

/// Julia `paired_sampling_inverse.jl` "independently known Pfaffians and
/// inverses", for the in-kernel dense refactor: A = Q D Q^T with orthogonal Q
/// (Householder products, det Q = (-1)^reflections) and known Pf(A) =
/// det(Q) prod(d), including identity pairs, condition 1e6 and an exactly
/// singular plane with poisoned buffers.
#[test]
#[ignore = "requires CUDA; dense fallback factors against known Pfaffians"]
fn dense_refactor_matches_known_pfaffians_and_inverses() {
    let mut backend = CudaEngine::open(cuda_devices().unwrap()[0].id()).unwrap();
    let mut rng = sfmt19937::Sfmt19937Rng::new(548);
    let mut uniform = || rng.gen_rand32() as f64 / 4294967296.0 - 0.5;
    for n in [2usize, 6, 32, 64, 128, 130] {
        let count = 4;
        let mut table = vec![0f64; count * n * n];
        let mut expected_pf = vec![0f64; count];
        // Independent inverses: (Q D Q^T)^-1 = Q D' Q^T with d' = -1/d.
        let mut oracles = vec![vec![0f64; n * n]; 3];
        for b in 0..count {
            // Q = H_1 ... H_r, rows-major dense; b=0 is the identity.
            let reflections = if b == 0 { 0 } else { 3 };
            let mut q = vec![0f64; n * n];
            for i in 0..n {
                q[i * n + i] = 1.;
            }
            for _ in 0..reflections {
                let v: Vec<f64> = (0..n).map(|_| uniform()).collect();
                let vv = v.iter().map(|x| x * x).sum::<f64>();
                // Q <- Q (I - 2 v v^T / v^T v)
                for i in 0..n {
                    let dot = (0..n).map(|k| q[i * n + k] * v[k]).sum::<f64>();
                    for j in 0..n {
                        q[i * n + j] -= 2. * dot * v[j] / vv;
                    }
                }
            }
            let ds: Vec<f64> = (0..n / 2)
                .map(|k| {
                    if b == 2 && n > 2 {
                        10f64.powf(-3. + 6. * k as f64 / (n / 2 - 1) as f64)
                    } else {
                        1.
                    }
                })
                .collect();
            // A = Q D Q^T, then exact skew symmetrization.
            let mut a = vec![0f64; n * n];
            for i in 0..n {
                for j in 0..n {
                    let mut x = 0.;
                    for k in 0..n / 2 {
                        x += ds[k]
                            * (q[i * n + 2 * k] * q[j * n + 2 * k + 1]
                                - q[i * n + 2 * k + 1] * q[j * n + 2 * k]);
                    }
                    a[i * n + j] = x;
                }
            }
            if b < 3 {
                for i in 0..n {
                    for j in 0..n {
                        oracles[b][i * n + j] = (0..n / 2)
                            .map(|k| {
                                -(q[i * n + 2 * k] * q[j * n + 2 * k + 1]
                                    - q[i * n + 2 * k + 1] * q[j * n + 2 * k])
                                    / ds[k]
                            })
                            .sum::<f64>();
                    }
                }
            }
            for i in 0..n {
                for j in 0..n {
                    table[b * n * n + i * n + j] = if b == 3 {
                        0.
                    } else {
                        0.5 * (a[i * n + j] - a[j * n + i])
                    };
                }
            }
            expected_pf[b] = if b == 3 {
                0.
            } else {
                (if reflections % 2 == 1 { -1. } else { 1. }) * ds.iter().product::<f64>()
            };
        }
        let (inv, pf, valid) = {
            let s: &mut CudaEngine = &mut backend;
            with_raw_exec(s, "dense.probe", |exec| {
                let module = exec.module(PROBE_MODULE, &["resident_dense_probe"])?;
                let t = exec.alloc_bytes(table.len() * 8)?;
                let inv = exec.alloc_bytes(table.len() * 8)?;
                let pf = exec.alloc_bytes(count * 8)?;
                let valid = exec.alloc_bytes(count * 4)?;
                let fac = exec.alloc_bytes(n * n * 8)?;
                let work = exec.alloc_bytes(n * n * 8)?;
                let mut out = (
                    vec![0f64; table.len()],
                    vec![0f64; count],
                    vec![0i32; count],
                );
                unsafe {
                    exec.upload(exec.ptr(&t)?, &table)?;
                    exec.upload(exec.ptr(&inv)?, &vec![f64::NAN; table.len()])?;
                    exec.upload(exec.ptr(&fac)?, &vec![f64::NAN; n * n])?;
                    exec.upload(exec.ptr(&work)?, &vec![f64::NAN; n * n])?;
                    exec.launch(
                        module.function("resident_dense_probe")?,
                        (1, 1, 1),
                        128,
                        (((n + 1) & !1) * 4 + 2 * n * 8) as u32,
                        &[
                            Arg::P(exec.ptr(&t)?),
                            Arg::P(exec.ptr(&inv)?),
                            Arg::P(exec.ptr(&pf)?),
                            Arg::P(exec.ptr(&valid)?),
                            Arg::P(exec.ptr(&fac)?),
                            Arg::P(exec.ptr(&work)?),
                            Arg::I(n as i32),
                            Arg::I(count as i32),
                        ],
                    )?;
                    exec.download(&mut out.0, exec.ptr(&inv)?)?;
                    exec.download(&mut out.1, exec.ptr(&pf)?)?;
                    exec.download(&mut out.2, exec.ptr(&valid)?)?;
                }
                Ok(out)
            })
        }
        .unwrap();
        assert_eq!(valid, vec![1, 1, 1, 0], "n={n}");
        assert_eq!(pf[3], 0., "n={n}: singular plane");
        let infnorm = |m: &dyn Fn(usize, usize) -> f64| {
            (0..n)
                .map(|r| (0..n).map(|c| m(r, c).abs()).sum::<f64>())
                .fold(0f64, f64::max)
        };
        for b in 0..3 {
            // Column-major A(r,c) = table[r*n+c] (identity sites, skew table);
            // X = -inv is the published A^-1.
            let a = |r: usize, c: usize| table[b * n * n + r * n + c];
            let x = |r: usize, c: usize| -inv[b * n * n + r + c * n];
            let oracle = |r: usize, c: usize| oracles[b][r * n + c];
            let residual =
                infnorm(&|r, c| (0..n).map(|k| a(r, k) * x(k, c)).sum::<f64>() - f64::from(r == c))
                    / (infnorm(&a) * infnorm(&x));
            let condition = infnorm(&a) * infnorm(&x);
            let bound = 64. * n as f64 * f64::EPSILON;
            assert!(
                residual <= bound,
                "n={n} b={b} residual={residual:e} bound={bound:e}"
            );
            let forward = infnorm(&|r, c| x(r, c) - oracle(r, c)) / infnorm(&oracle);
            assert!(
                forward <= bound * condition,
                "n={n} b={b} forward={forward:e} bound={:e}",
                bound * condition
            );
            assert_eq!(
                infnorm(&|r, c| x(r, c) + x(c, r)),
                0.,
                "n={n} b={b}: exactly skew inverse"
            );
            let pf_error = ((pf[b] - expected_pf[b]) / expected_pf[b]).abs();
            assert!(
                pf_error <= bound * condition,
                "n={n} b={b} pf={} expected={} error={pf_error:e} bound={:e}",
                pf[b],
                expected_pf[b],
                bound * condition
            );
        }
        assert!(
            inv[3 * n * n..].iter().all(|x| x.is_nan()),
            "n={n}: singular plane unpublished"
        );
    }
}

/// A walker left with unfinished hops after the enqueued rounds (a broken
/// schedule invariant) is a typed schedule failure, never a short interval.
#[test]
#[ignore = "requires CUDA"]
fn unfinished_schedule_latches_typed_status() {
    let mut backend = CudaEngine::open(cuda_devices().unwrap()[0].id()).unwrap();
    let status = {
        let s: &mut CudaEngine = &mut backend;
        with_raw_exec(s, "schedule.status", |exec| {
            let module = exec.module(INTERVAL_MODULE, INTERVAL_KERNELS)?;
            let hop_off = exec.alloc_bytes(3 * 4)?;
            let error = exec.alloc_bytes(4)?;
            let mut out = [0i32; 2];
            unsafe {
                exec.zero_bytes(exec.ptr(&error)?, 4)?;
                for (slot, last) in [(0, 5i32), (1, 4)] {
                    exec.upload(exec.ptr(&hop_off)?, &[5i32, 5, last])?;
                    exec.launch(
                        module.function("interval_resident_complete")?,
                        (1, 1, 1),
                        128,
                        0,
                        &[
                            Arg::P(exec.ptr(&hop_off)?),
                            Arg::I(5),
                            Arg::P(exec.ptr(&error)?),
                            Arg::I(3),
                        ],
                    )?;
                    let mut word = [0i32];
                    exec.download(&mut word, exec.ptr(&error)?)?;
                    out[slot] = word[0];
                }
            }
            Ok(out)
        })
    }
    .unwrap();
    assert_eq!(
        status.map(ResidentFailure::from_status),
        [None, Some(ResidentFailure::Schedule)]
    );
}

/// Julia `batched_inverse.jl`: a rebuild with one exactly singular QP plane
/// publishes the walker with a per-plane mask, Pf = 0 for the singular plane
/// and the nonsingular plane's inverse. Slater planes are orthogonal skew
/// matrices (condition ~1, Householder Q), with holes; n=128 takes the paired
/// factor and its conventional fallback.
#[test]
#[ignore = "requires CUDA; mixed singular rebuild against independent CPU factors"]
fn mixed_singular_rebuild_publishes_valid_planes() {
    use mvmc_gpu::{pfaffian_inverse_batched, Backend};
    let mut backend = CudaEngine::open(cuda_devices().unwrap()[0].id()).unwrap();
    let mut rng = sfmt19937::Sfmt19937Rng::new(81);
    for (n, holes) in [(64, 0), (64, 3), (128, 0), (128, 3)] {
        let (w, nq, ne) = (3, 2, n / 2);
        let ns = ne + holes;
        let full = 2 * ns;
        let mut q = vec![0f64; full * full];
        for i in 0..full {
            q[i * full + i] = 1.;
        }
        for _ in 0..4 {
            let v: Vec<f64> = (0..full)
                .map(|_| rng.gen_rand32() as f64 / 4294967296.0 - 0.5)
                .collect();
            let vv = v.iter().map(|x| x * x).sum::<f64>();
            for i in 0..full {
                let dot = (0..full).map(|k| q[i * full + k] * v[k]).sum::<f64>();
                for j in 0..full {
                    q[i * full + j] -= 2. * dot * v[j] / vv;
                }
            }
        }
        // A = Q J Q^T, J the paired symplectic unit; exactly skew afterwards.
        let mut a = vec![0f64; full * full];
        for i in 0..full {
            for j in 0..full {
                a[i * full + j] = (0..full / 2)
                    .map(|k| {
                        q[i * full + 2 * k] * q[j * full + 2 * k + 1]
                            - q[i * full + 2 * k + 1] * q[j * full + 2 * k]
                    })
                    .sum::<f64>();
            }
        }
        let mut slater = vec![0f64; nq * full * full];
        for i in 0..full {
            for j in 0..full {
                slater[i * full + j] = 0.5 * (a[i * full + j] - a[j * full + i]);
            }
        }
        // Random distinct sites per spin and walker (partial Fisher-Yates).
        let mut ele = vec![0i32; w * n];
        for walker in 0..w {
            for spin in 0..2 {
                let mut sites: Vec<i32> = (0..ns as i32).collect();
                for k in 0..ne {
                    let j = k + rng.gen_rand32() as usize % (ns - k);
                    sites.swap(k, j);
                }
                ele[walker * n + spin * ne..walker * n + (spin + 1) * ne]
                    .copy_from_slice(&sites[..ne]);
            }
        }
        let g = IntervalGeometry {
            walkers: w,
            n,
            ne,
            nsite: ns,
            nqp: nq,
            ngutz: 0,
            nproj: 0,
        };
        let (gutz, jast) = (vec![-1; ns], vec![-1; ns * ns]);
        let initial_inv = vec![0.; w * nq * n * n];
        let initial_pf = vec![1.; w * nq];
        let host = IntervalHost {
            ele: &ele,
            inv: &initial_inv,
            pf: &initial_pf,
            slater: &slater,
            weights: &[1., 1.],
            gutz: &gutz,
            jast: &jast,
            params: &[],
        };
        let resident = {
            let s: &mut CudaEngine = &mut backend;
            ResidentInterval::prepare(s, g, &host, 0, 79)
        }
        .unwrap();
        let (inv, pf, valid, logip, error) = {
            let s: &mut CudaEngine = &mut backend;
            with_raw_exec(s, "mixed.rebuild", |exec| {
                let module = exec.module(INTERVAL_MODULE, INTERVAL_KERNELS)?;
                let r = &resident.sampler;
                unsafe {
                    exec.upload(exec.ptr(&r.rebuild)?, &vec![1i32; w])?;
                }
                resident.enqueue_rebuild(exec, &module)?;
                let mut out = (
                    vec![0f64; w * nq * n * n],
                    vec![0f64; w * nq],
                    vec![0i32; w * nq],
                    vec![0f64; w],
                    [0i32],
                );
                unsafe {
                    exec.download(&mut out.0, exec.ptr(&r.inv)?)?;
                    exec.download(&mut out.1, exec.ptr(&r.pf)?)?;
                    exec.download(&mut out.2, exec.ptr(&resident.valid)?)?;
                    exec.download(&mut out.3, exec.ptr(&r.logip)?)?;
                    exec.download(&mut out.4, exec.ptr(&resident.error)?)?;
                }
                Ok(out)
            })
        }
        .unwrap();
        let label = format!("n={n} holes={holes}");
        assert_eq!(error, [0], "{label}");
        assert_eq!(valid, [1, 0].repeat(w), "{label}");
        let planes = gather_planes(&ele, &slater, n, ns, nq);
        let cpu = pfaffian_inverse_batched(&Backend::CpuPfapack, &planes, n, nq, w).unwrap();
        let infnorm = |m: &dyn Fn(usize, usize) -> f64| {
            (0..n)
                .map(|r| (0..n).map(|c| m(r, c).abs()).sum::<f64>())
                .fold(0f64, f64::max)
        };
        let bound = 64. * n as f64 * f64::EPSILON;
        for walker in 0..w {
            assert_eq!(pf[walker * nq + 1], 0., "{label}: singular plane Pf");
            let plane = walker * nq;
            let x = |r: usize, c: usize| planes[plane * n * n + r + c * n];
            let y = |r: usize, c: usize| -inv[plane * n * n + r + c * n];
            let oracle = |r: usize, c: usize| cpu.inv[plane * n * n + r + c * n];
            let backward =
                infnorm(&|r, c| (0..n).map(|k| x(r, k) * y(k, c)).sum::<f64>() - f64::from(r == c))
                    / (infnorm(&x) * infnorm(&y));
            let condition = infnorm(&x) * infnorm(&oracle);
            let forward = infnorm(&|r, c| y(r, c) - oracle(r, c)) / infnorm(&oracle);
            let pf_error = ((pf[plane] - cpu.pf[plane]) / cpu.pf[plane]).abs();
            assert!(
                backward <= bound && forward <= bound * condition && pf_error <= bound * condition,
                "{label} walker={walker} backward={backward:e} forward={forward:e} pf={pf_error:e} cond={condition:e}"
            );
            assert!((logip[walker] - cpu.pf[plane].abs().ln()).abs() <= bound * condition);
        }
    }
}

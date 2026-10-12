//! Shared-tape parity of the interval sampler against the Julia GPU sampler (issue #547).
//!
//! Ignored by default (requires CUDA). The fixture `fixtures/interval_parity_j8.txt` is
//! generated on the reference environment by `fixtures/gen_interval_parity.jl`
//! (Julia 1.13.1, CUDA.jl 6.4.2, A100 sm_80): the Julia `gpu_sample!` ->
//! `cached_sampling_kernel!` path (one block per walker, `empty_destinations=true`,
//! `exchange_fraction=0`) on a projection-free synthetic state with a fixed device tape.
//! This test builds the same initial state, replays the same tape through the Rust
//! interval sampler, and compares the final configuration and occupancy. Because both
//! read the same tape, the RNG stream is out of scope and only the sampler semantics are
//! compared.

use mvmc_core::pfaffian::calc_m_all_real;
use mvmc_core::state::{InvMColMajor, SlaterElmFlat, ThreadedPfaPackWorkspace};
use mvmc_gpu_cuda::interval::{IntervalGeometry, IntervalHost, IntervalSampler};
use tenferro_gpu::cuda::{cuda_devices, gpu_available, CudaBackend};
use tenferro_tensor::BackendSessionHost;

/// Parsed fixture.
struct Fixture {
    nsite: usize,
    ne: usize,
    nqp: usize,
    npair: usize,
    steps: usize,
    slater: Vec<f64>,
    weights: Vec<f64>,
    ele: Vec<i32>,
    tape: Vec<u32>,
    final_ele: Vec<i32>,
    final_occupancy: Vec<i32>,
    accepted: i32,
}

fn load_fixture() -> Fixture {
    let text = include_str!("fixtures/interval_parity_j8.txt");
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();
    let mut i = 0usize;
    let head = |i: &mut usize| -> (String, usize) {
        let mut it = lines[*i].split_whitespace();
        let key = it.next().unwrap().to_string();
        let value: usize = it.next().unwrap().parse().unwrap();
        *i += 1;
        (key, value)
    };
    let array = |i: &mut usize, key: &str| -> Vec<f64> {
        let (k, count) = head(i);
        assert_eq!(k, key, "fixture key order");
        let out: Vec<f64> = (0..count).map(|k| lines[*i + k].parse().unwrap()).collect();
        *i += count;
        out
    };
    let (_, nsite) = head(&mut i);
    let (_, ne) = head(&mut i);
    let (_, nqp) = head(&mut i);
    let (_, npair) = head(&mut i);
    let (_, steps) = head(&mut i);
    let slater = array(&mut i, "slater");
    let weights = array(&mut i, "weights");
    let ele: Vec<i32> = array(&mut i, "ele").into_iter().map(|v| v as i32).collect();
    let tape: Vec<u32> = array(&mut i, "tape")
        .into_iter()
        .map(|v| v as u32)
        .collect();
    let final_ele: Vec<i32> = array(&mut i, "final_ele")
        .into_iter()
        .map(|v| v as i32)
        .collect();
    let final_occupancy: Vec<i32> = array(&mut i, "final_occupancy")
        .into_iter()
        .map(|v| v as i32)
        .collect();
    let (_, accepted) = head(&mut i);
    Fixture {
        nsite,
        ne,
        nqp,
        npair,
        steps,
        slater,
        weights,
        ele,
        tape,
        final_ele,
        final_occupancy,
        accepted: accepted as i32,
    }
}

fn occupancy_of(ele: &[i32], nsite: usize, ne: usize) -> Vec<i32> {
    let mut occ = vec![0i32; 2 * nsite];
    for spin in 0..2usize {
        for j in 0..ne {
            let site = ele[j + spin * ne] as usize;
            occ[site + spin * nsite] = 1;
        }
    }
    occ
}

#[test]
#[ignore = "requires CUDA"]
fn interval_shared_tape_matches_julia_gpu() {
    if !gpu_available() {
        eprintln!("skipped: no CUDA device");
        return;
    }
    let fx = load_fixture();
    let devices = cuda_devices().expect("device discovery");
    let mut backend = CudaBackend::new(devices[0].id()).expect("cuda backend");

    let n = fx.npair;
    let n2 = 2 * fx.nsite;

    // Initial state from the fixture, through the same host Pfaffian path the sampler uses.
    // Julia dumps `slater` in column-major (row fastest): index = row + n2*col + n2^2*q.
    let mut slater = SlaterElmFlat::<f64>::zeros(fx.nqp, fx.nsite);
    for q in 0..fx.nqp {
        for col in 0..n2 {
            for row in 0..n2 {
                slater.set(q, row, col, fx.slater[row + n2 * col + n2 * n2 * q]);
            }
        }
    }
    let ele_i64: Vec<i64> = fx.ele.iter().map(|&v| v as i64).collect();
    let mut inv = InvMColMajor::<f64>::zeros(fx.nqp, fx.ne);
    let mut pf = vec![0.0f64; fx.nqp];
    let pool = ThreadedPfaPackWorkspace::new(n, 1);
    calc_m_all_real(
        &ele_i64, &slater, &mut inv, &mut pf, 0, fx.nqp, fx.nsite, fx.ne, &pool,
    )
    .expect("host Pfaffian");
    let mut inv_all = Vec::with_capacity(fx.nqp * n * n);
    for q in 0..fx.nqp {
        inv_all.extend_from_slice(inv.qp_matrix_slice(q));
    }

    let geom = IntervalGeometry {
        walkers: 1,
        n,
        ne: fx.ne,
        nsite: fx.nsite,
        nqp: fx.nqp,
        ngutz: 0,
        nproj: 0,
    };
    let gutz = vec![-1i32; fx.nsite];
    let jast = vec![-1i32; fx.nsite * fx.nsite];
    let params = vec![0.0f64];
    let host = IntervalHost {
        ele: &fx.ele,
        inv: &inv_all,
        pf: &pf,
        slater: slater.as_slice(),
        weights: &fx.weights,
        gutz: &gutz,
        jast: &jast,
        params: &params,
    };

    let sampler = backend
        .with_backend_session(|s| IntervalSampler::prepare(s, geom, &host, fx.steps, 0.0))
        .expect("prepare");
    backend
        .with_backend_session(|s| sampler.upload_tape(s, &fx.tape))
        .expect("upload tape");

    // Run to completion, rebuilding a walker that asks (the Julia kernel rebuilds inline).
    let mut rounds = 0usize;
    let mut rebuilds = 0usize;
    let mut accepted = 0i32;
    loop {
        let results = backend
            .with_backend_session(|s| sampler.run(s))
            .expect("run");
        rounds += 1;
        let mut need = Vec::new();
        for (w, &(_, r, a)) in results.iter().enumerate() {
            accepted += a;
            if r != 0 {
                need.push(w);
            }
        }
        if need.is_empty() {
            break;
        }
        rebuilds += need.len();
        backend
            .with_backend_session(|s| sampler.rebuild(s, &need))
            .expect("rebuild");
    }

    let final_ele = backend
        .with_backend_session(|s| sampler.download_ele(s, 0))
        .expect("download ele");
    let occ = occupancy_of(&final_ele, fx.nsite, fx.ne);
    eprintln!(
        "julia  ele={:?} occ={:?} accepted={}",
        fx.final_ele, fx.final_occupancy, fx.accepted
    );
    eprintln!("rust   ele={final_ele:?} occ={occ:?} rounds={rounds} rebuilds={rebuilds}");
    assert_eq!(final_ele, fx.final_ele, "final configuration");
    assert_eq!(occ, fx.final_occupancy, "final occupancy");
    assert_eq!(accepted, fx.accepted, "accepted moves on the shared tape");
}

#[test]
#[ignore = "requires CUDA"]
fn interval_qp_tiles_and_batch_download_match_cpu_rebuild() {
    if !gpu_available() {
        eprintln!("skipped: no CUDA device");
        return;
    }
    let devices = cuda_devices().unwrap();
    let mut backend = CudaBackend::new(devices[0].id()).unwrap();
    // Small fallback, non-power-of-two QP tails, the L128 specialization, and n=256
    // where allocating 3*n*nqp doubles would exceed CUDA's default shared-memory limit.
    for (n, nqp) in [(8, 1), (64, 3), (128, 5), (256, 8)] {
        let (walkers, hops, ne, nsite) = (3, 4, n / 2, n + 2);
        let n2 = 2 * nsite;
        let geom = IntervalGeometry {
            walkers,
            n,
            ne,
            nsite,
            nqp,
            ngutz: 0,
            nproj: 0,
        };
        assert!(geom.shared_bytes() <= 48 * 1024);
        let mut slater = SlaterElmFlat::<f64>::zeros(nqp, nsite);
        let mut rng = sfmt19937::Sfmt19937Rng::new(79);
        for q in 0..nqp {
            for i in 0..n2 {
                for j in 0..i {
                    let v = rng.gen_rand32() as f64 / 4294967296.0 - 0.5;
                    slater.set(q, i, j, v);
                    slater.set(q, j, i, -v);
                }
            }
        }
        let pool = ThreadedPfaPackWorkspace::new(n, 1);
        let ele: Vec<i32> = (0..walkers)
            .flat_map(|w| (0..n).map(move |i| ((i + w) % nsite) as i32))
            .collect();
        let mut inv_all = Vec::new();
        let mut pf_all = Vec::new();
        for w in 0..walkers {
            let cfg: Vec<i64> = ele[w * n..(w + 1) * n].iter().map(|&v| v as i64).collect();
            let mut inv = InvMColMajor::<f64>::zeros(nqp, ne);
            let mut pf = vec![0.0; nqp];
            calc_m_all_real(&cfg, &slater, &mut inv, &mut pf, 0, nqp, nsite, ne, &pool).unwrap();
            for q in 0..nqp {
                inv_all.extend_from_slice(inv.qp_matrix_slice(q));
            }
            pf_all.extend(pf);
        }
        let weights = vec![1.0; nqp];
        let gutz = vec![-1; nsite];
        let jast = vec![-1; nsite * nsite];
        let host = IntervalHost {
            ele: &ele,
            inv: &inv_all,
            pf: &pf_all,
            slater: slater.as_slice(),
            weights: &weights,
            gutz: &gutz,
            jast: &jast,
            params: &[0.0],
        };
        let sampler = backend
            .with_backend_session(|s| IntervalSampler::prepare(s, geom, &host, hops, 0.0))
            .unwrap();
        let mut tape = vec![0; walkers * hops * 5];
        for h in 0..hops {
            for w in 0..walkers {
                let t = &mut tape[5 * (w + walkers * h)..5 * (w + walkers * h) + 5];
                t[0] = (h + w) as u32;
                t[1] = ((h % 2) as u32) << 31;
                t[2] = (h + 2 * w) as u32;
                // Acceptance uniform zero: these finite, nonsingular hops must all accept.
            }
        }
        backend
            .with_backend_session(|s| sampler.upload_tape(s, &tape))
            .unwrap();
        let counts = backend.with_backend_session(|s| sampler.run(s)).unwrap();
        assert_eq!(counts, vec![(hops as i32, 0, hops as i32); walkers]);
        let all = backend
            .with_backend_session(|s| sampler.download_all(s))
            .unwrap();
        for w in 0..walkers {
            let (cfg, pf, aw, logip) = backend
                .with_backend_session(|s| sampler.download_walker(s, w))
                .unwrap();
            assert_eq!(cfg, all.ele[w * n..(w + 1) * n]);
            assert_eq!(aw, all.aw[w]);
            assert_eq!(pf, all.pf[w * nqp..(w + 1) * nqp]); // exact transfer identity
            assert_eq!(logip, all.logip[w]);
            let cfg: Vec<i64> = cfg.iter().map(|&v| v as i64).collect();
            let mut inv = InvMColMajor::<f64>::zeros(nqp, ne);
            let mut expected = vec![0.0; nqp];
            calc_m_all_real(
                &cfg,
                &slater,
                &mut inv,
                &mut expected,
                0,
                nqp,
                nsite,
                ne,
                &pool,
            )
            .unwrap();
            for q in 0..nqp {
                // Four rank-two updates versus an independent pivoted factorization.
                // Allow accumulated FP64 update/factorization roundoff (no trajectory slack).
                let tol = 1e-10 + 1e-9 * expected[q].abs();
                assert!(
                    (pf[q] - expected[q]).abs() <= tol,
                    "n={n} q={q} w={w}: GPU {} CPU {}",
                    pf[q],
                    expected[q]
                );
            }
        }
    }
}

#[test]
#[ignore = "requires CUDA"]
fn interval_walker_upload_download_roundtrip() {
    if !gpu_available() {
        eprintln!("skipped: no CUDA device");
        return;
    }
    let devices = cuda_devices().expect("device discovery");
    let mut backend = CudaBackend::new(devices[0].id()).expect("cuda backend");

    let (nsite, ne, nqp) = (8usize, 4usize, 2usize);
    let n = 2 * ne;
    let n2 = 2 * nsite;
    let geom = IntervalGeometry {
        walkers: 1,
        n,
        ne,
        nsite,
        nqp,
        ngutz: 0,
        nproj: 0,
    };
    let ele0 = vec![0i32, 1, 2, 3, 4, 5, 6, 7];
    let inv0 = vec![0.25f64; nqp * n * n];
    let pf0 = vec![1.0f64, 0.5];
    let slater = vec![0.0f64; nqp * n2 * n2];
    let weights = vec![1.0f64, 1.0];
    let gutz = vec![-1i32; nsite];
    let jast = vec![-1i32; nsite * nsite];
    let params = vec![0.0f64];
    let host = IntervalHost {
        ele: &ele0,
        inv: &inv0,
        pf: &pf0,
        slater: &slater,
        weights: &weights,
        gutz: &gutz,
        jast: &jast,
        params: &params,
    };
    let sampler = backend
        .with_backend_session(|s| IntervalSampler::prepare(s, geom, &host, 4, 0.0))
        .expect("prepare");

    // Upload a distinct walker-0 state and read it back.
    let ele1 = vec![4i32, 6, 0, 2, 5, 7, 1, 3];
    let inv1 = vec![-0.125f64; nqp * n * n];
    let pf1 = vec![0.75f64, -0.25];
    backend
        .with_backend_session(|s| sampler.upload_walker(s, 0, &ele1, &inv1, &pf1, 3, -1.5))
        .expect("upload_walker");
    let (ele_back, pf_back, aw_back, logip_back) = backend
        .with_backend_session(|s| sampler.download_walker(s, 0))
        .expect("download_walker");
    assert_eq!(ele_back, ele1, "configuration round trip");
    assert_eq!(pf_back, pf1, "Pfaffian round trip");
    assert_eq!(aw_back, 3, "accepted-window round trip");
    assert_eq!(logip_back, -1.5, "log|IP| round trip");
    let ele_only = backend
        .with_backend_session(|s| sampler.download_ele(s, 0))
        .expect("download ele");
    assert_eq!(ele_only, ele1);
}

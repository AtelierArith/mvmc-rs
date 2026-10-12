//! Walker-sweep throughput of the raw-CUDA interval sampler at the L=128 geometry
//! (issue #547). Ignored by default; run on a GPU with `--ignored --nocapture`.
//!
//! One CUDA block per walker. A single block is latency-bound, so the throughput only grows
//! once enough walkers fill the device (the GPU-readiness design's claim). Each launch runs
//! `HOPS` hops per walker; kept below the `nsite`-accept rebuild window so no host rebuild is
//! interleaved, isolating device sampling throughput.

use std::time::Instant;

use mvmc_core::pfaffian::calc_m_all_real;
use mvmc_core::state::{InvMColMajor, SlaterElmFlat, ThreadedPfaPackWorkspace};
use mvmc_gpu_cuda::interval::{IntervalGeometry, IntervalHost, IntervalSampler};
use tenferro_gpu::cuda::{cuda_devices, gpu_available, CudaBackend};
use tenferro_tensor::BackendSessionHost;

const NSITE: usize = 128;
const NE: usize = 64;
const NQP: usize = 8;
const HOPS: usize = 256;

struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u32() as f64) / (u32::MAX as f64)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u32() as usize) % n
    }
}

fn log_ip(pf: &[f64], weights: &[f64]) -> f64 {
    let ip: f64 = weights.iter().zip(pf).map(|(w, p)| w * p).sum();
    ip.abs().ln()
}

fn random_config(rng: &mut Lcg) -> Vec<i32> {
    let mut ele = vec![0i32; 2 * NE];
    for spin in 0..2usize {
        let mut sites: Vec<usize> = (0..NSITE).collect();
        for i in 0..NE {
            let j = i + rng.below(NSITE - i);
            sites.swap(i, j);
            ele[i + spin * NE] = sites[i] as i32;
        }
    }
    ele
}

#[test]
#[ignore = "requires CUDA"]
fn interval_walker_sweep_l128() {
    if !gpu_available() {
        eprintln!("skipped: no CUDA device");
        return;
    }
    let devices = cuda_devices().expect("device discovery");
    let mut backend = CudaBackend::new(devices[0].id()).expect("cuda backend");

    let n = 2 * NE;
    let n2 = 2 * NSITE;
    let pool = ThreadedPfaPackWorkspace::new(n, 1);
    let mut rng = Lcg(0xdead_beef_1234_5678);

    let mut slater = SlaterElmFlat::<f64>::zeros(NQP, NSITE);
    for q in 0..NQP {
        for row in 0..n2 {
            for col in 0..n2 {
                slater.set(q, row, col, rng.next_f64() - 0.5);
            }
        }
    }
    let weights = vec![1.0f64; NQP];
    let gutz = vec![-1i32; NSITE];
    let jast = vec![-1i32; NSITE * NSITE];
    let params = vec![0.0f64; 1];

    eprintln!("== interval sampler walker sweep, nsite={NSITE} nqp={NQP} hops/walker={HOPS} ==");
    for walkers in [1usize, 16, 64, 128, 320] {
        // Per-walker valid state (same Slater table, distinct configurations).
        let mut ele_all = Vec::with_capacity(walkers * n);
        let mut inv_all = Vec::with_capacity(walkers * NQP * n * n);
        let mut pf_all = Vec::with_capacity(walkers * NQP);
        for _ in 0..walkers {
            let ele_i32 = random_config(&mut rng);
            let ele_i64: Vec<i64> = ele_i32.iter().map(|&v| v as i64).collect();
            let mut inv = InvMColMajor::<f64>::zeros(NQP, NE);
            let mut pf = vec![0.0f64; NQP];
            calc_m_all_real(
                &ele_i64, &slater, &mut inv, &mut pf, 0, NQP, NSITE, NE, &pool,
            )
            .expect("host Pfaffian");
            assert!(pf.iter().all(|v| v.is_finite() && *v != 0.0));
            ele_all.extend_from_slice(&ele_i32);
            for q in 0..NQP {
                inv_all.extend_from_slice(inv.qp_matrix_slice(q));
            }
            pf_all.extend_from_slice(&pf);
        }
        let geom = IntervalGeometry {
            walkers,
            n,
            ne: NE,
            nsite: NSITE,
            nqp: NQP,
            ngutz: 0,
            nproj: 1,
        };
        let host = IntervalHost {
            ele: &ele_all,
            inv: &inv_all,
            pf: &pf_all,
            slater: slater.as_slice(),
            weights: &weights,
            gutz: &gutz,
            jast: &jast,
            params: &params,
        };
        let tape: Vec<u32> = (0..walkers * HOPS * 5)
            .map(|_| rng.next_u32())
            .collect();
        let sampler = backend
            .with_backend_session(|s| IntervalSampler::prepare(s, geom, &host, HOPS, 0.0))
            .expect("prepare");
        backend
            .with_backend_session(|s| sampler.upload_tape(s, &tape))
            .expect("upload tape");
        // Warm up (NVRTC compile) with a short run, then measure a single fresh run: the
        // accepted-move window (`aw`) persists on the device, so a stale window would break
        // the next run immediately.
        let _ = backend
            .with_backend_session(|s| sampler.run(s))
            .expect("warmup");
        let _ = backend
            .with_backend_session(|s| sampler.download_pf(s, 0))
            .expect("warmup pf");
        let logip_all: Vec<f64> = (0..walkers)
            .map(|k| log_ip(&pf_all[k * NQP..(k + 1) * NQP], &weights))
            .collect();
        let aw0 = vec![0i32; walkers];
        backend
            .with_backend_session(|s| sampler.reset_offsets(s))
            .expect("reset offsets");
        backend
            .with_backend_session(|s| {
                sampler.upload_tables(s, &inv_all, &pf_all, &aw0, &logip_all)
            })
            .expect("reset tables");
        let t = Instant::now();
        let results = backend
            .with_backend_session(|s| sampler.run(s))
            .expect("run");
        let best = t.elapsed().as_secs_f64();
        let total_hops: usize = results.iter().map(|&(d, _, _)| d.max(0) as usize).sum();
        let rb_any = results.iter().any(|&(_, r, _)| r != 0);
        eprintln!(
            "W={walkers:4}  wall={:.4} s  hops_total={total_hops:7} rebuild_any={rb_any}  {:.1} ns/hop  {:.2} M hops/s",
            best,
            best / total_hops.max(1) as f64 * 1e9,
            total_hops as f64 / best / 1e6
        );
    }
}

const FULL_WALKERS: usize = 320;
const FULL_HOPS: usize = 2688;

/// Drain the sampler: run rounds, rebuilding every walker that asks, until all reach
/// `FULL_HOPS`. Returns `(hops advanced, rebuilds, rounds)`.
fn drain(backend: &mut CudaBackend, sampler: &IntervalSampler) -> (usize, usize, usize) {
    let mut hops = 0usize;
    let mut rebuilds = 0usize;
    let mut rounds = 0usize;
    loop {
        let results = backend
            .with_backend_session(|s| sampler.run(s))
            .expect("run");
        rounds += 1;
        let mut need = Vec::new();
        for (w, &(d, r, _a)) in results.iter().enumerate() {
            hops += d.max(0) as usize;
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
    (hops, rebuilds, rounds)
}

#[test]
#[ignore = "requires CUDA"]
fn interval_full_run_l128() {
    if !gpu_available() {
        eprintln!("skipped: no CUDA device");
        return;
    }
    let devices = cuda_devices().expect("device discovery");
    let mut backend = CudaBackend::new(devices[0].id()).expect("cuda backend");

    let n = 2 * NE;
    let n2 = 2 * NSITE;
    let pool = ThreadedPfaPackWorkspace::new(n, 1);
    let mut rng = Lcg(0xfeed_face_cafe_0001);

    let mut slater = SlaterElmFlat::<f64>::zeros(NQP, NSITE);
    for q in 0..NQP {
        for row in 0..n2 {
            for col in 0..n2 {
                slater.set(q, row, col, rng.next_f64() - 0.5);
            }
        }
    }
    let weights = vec![1.0f64; NQP];
    let gutz = vec![-1i32; NSITE];
    let jast = vec![-1i32; NSITE * NSITE];
    let params = vec![0.0f64; 1];

    let mut ele_all = Vec::with_capacity(FULL_WALKERS * n);
    let mut inv_all = Vec::with_capacity(FULL_WALKERS * NQP * n * n);
    let mut pf_all = Vec::with_capacity(FULL_WALKERS * NQP);
    for _ in 0..FULL_WALKERS {
        let ele_i32 = random_config(&mut rng);
        let ele_i64: Vec<i64> = ele_i32.iter().map(|&v| v as i64).collect();
        let mut inv = InvMColMajor::<f64>::zeros(NQP, NE);
        let mut pf = vec![0.0f64; NQP];
        calc_m_all_real(&ele_i64, &slater, &mut inv, &mut pf, 0, NQP, NSITE, NE, &pool)
            .expect("host Pfaffian");
        ele_all.extend_from_slice(&ele_i32);
        for q in 0..NQP {
            inv_all.extend_from_slice(inv.qp_matrix_slice(q));
        }
        pf_all.extend_from_slice(&pf);
    }
    let geom = IntervalGeometry {
        walkers: FULL_WALKERS,
        n,
        ne: NE,
        nsite: NSITE,
        nqp: NQP,
        ngutz: 0,
        nproj: 1,
    };
    let host = IntervalHost {
        ele: &ele_all,
        inv: &inv_all,
        pf: &pf_all,
        slater: slater.as_slice(),
        weights: &weights,
        gutz: &gutz,
        jast: &jast,
        params: &params,
    };
    let tape: Vec<u32> = (0..FULL_WALKERS * FULL_HOPS * 5)
        .map(|_| rng.next_u32())
        .collect();

    // Warm-up sampler: compiles the NVRTC module and runs the loop once (untimed).
    {
        let sampler = backend
            .with_backend_session(|s| IntervalSampler::prepare(s, geom, &host, FULL_HOPS, 0.0))
            .expect("prepare warmup");
        backend
            .with_backend_session(|s| sampler.upload_tape(s, &tape))
            .expect("upload warmup tape");
        let (h, rb, rounds) = drain(&mut backend, &sampler);
        eprintln!("warmup: hops={h} rebuilds={rb} rounds={rounds}");
    }

    // Timed sampler: fresh state, module already cached.
    let sampler = backend
        .with_backend_session(|s| IntervalSampler::prepare(s, geom, &host, FULL_HOPS, 0.0))
        .expect("prepare");
    backend
        .with_backend_session(|s| sampler.upload_tape(s, &tape))
        .expect("upload tape");
    let t = Instant::now();
    let (hops, rebuilds, rounds) = drain(&mut backend, &sampler);
    let secs = t.elapsed().as_secs_f64();
    eprintln!("== interval full-run L=128, W={FULL_WALKERS}, hops/walker={FULL_HOPS} ==");
    eprintln!(
        "wall={secs:.3} s  hops={hops}  rebuilds={rebuilds}  rounds={rounds}  {:.1} ns/hop  {:.2} M hops/s",
        secs / hops.max(1) as f64 * 1e9,
        hops as f64 / secs / 1e6
    );
    let pf0 = backend
        .with_backend_session(|s| sampler.download_pf(s, 0))
        .expect("download pf");
    assert!(pf0.iter().all(|v| v.is_finite()), "pf finite: {pf0:?}");
}

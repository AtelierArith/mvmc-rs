//! Device gate for the raw-CUDA interval sampler (issue #547).
//!
//! Runs `k_interval_hop` through `IntervalSampler`/`with_raw_exec` on a real CUDA device
//! with a small, valid Slater state built by the C-order host kernels. Ignored by default;
//! run with a GPU and `--ignored`.

use mvmc_core::pfaffian::calc_m_all_real;
use mvmc_core::state::{InvMColMajor, SlaterElmFlat, ThreadedPfaPackWorkspace};
use mvmc_gpu_cuda::interval::{IntervalGeometry, IntervalHost, IntervalSampler};
use tenferro_gpu::cuda::{cuda_devices, gpu_available, CudaBackend};
use tenferro_tensor::BackendSessionHost;

struct Lcg(u64);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f64) / ((1u64 << 31) as f64)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
}

#[test]
#[ignore = "requires CUDA"]
fn interval_kernel_runs_and_keeps_finite_state() {
    if !gpu_available() {
        eprintln!("skipped: no CUDA device");
        return;
    }
    let devices = cuda_devices().expect("device discovery");
    let mut backend = CudaBackend::new(devices[0].id()).expect("cuda backend");

    // Small, valid half-filled-ish state: 2 electrons/spin on 8 sites leaves 4 empty sites.
    let (nqp, nsite, ne) = (1usize, 8usize, 2usize);
    let n = 2 * ne;
    let n2 = 2 * nsite;
    let mut rng = Lcg(0x1234_5678_9abc_def0);
    let mut slater = SlaterElmFlat::<f64>::zeros(nqp, nsite);
    for row in 0..n2 {
        for col in 0..n2 {
            slater.set(0, row, col, rng.next_f64() - 0.5);
        }
    }
    let ele_idx: Vec<i64> = vec![0, 1, 2, 3];
    let mut inv = InvMColMajor::<f64>::zeros(nqp, ne);
    let mut pf = vec![0.0f64; nqp];
    let pool = ThreadedPfaPackWorkspace::new(n, 1);
    calc_m_all_real(
        &ele_idx, &slater, &mut inv, &mut pf, 0, nqp, nsite, ne, &pool,
    )
    .expect("host Pfaffian");
    assert!(
        pf[0].abs() > 0.0 && pf[0].is_finite(),
        "the test Slater table must have a nonzero finite Pfaffian"
    );

    let inv_flat: Vec<f64> = (0..nqp)
        .flat_map(|q| inv.qp_matrix_slice(q).to_vec())
        .collect();
    let geom = IntervalGeometry {
        walkers: 1,
        n,
        ne,
        nsite,
        nqp,
        ngutz: 0,
        nproj: 1,
    };
    let ele_i32: Vec<i32> = ele_idx.iter().map(|&v| v as i32).collect();
    let weights = vec![1.0f64; nqp];
    let gutz = vec![-1i32; nsite];
    let jast = vec![-1i32; nsite * nsite];
    let params = vec![0.0f64; 1];
    let host = IntervalHost {
        ele: &ele_i32,
        inv: &inv_flat,
        pf: &pf,
        slater: slater.as_slice(),
        weights: &weights,
        gutz: &gutz,
        jast: &jast,
        params: &params,
    };

    let total_hops = 64usize;
    let tape: Vec<u32> = (0..total_hops * 5).map(|_| rng.next_u32()).collect();

    let sampler = backend
        .with_backend_session(|session| IntervalSampler::prepare(session, geom, &host, total_hops, 0.0))
        .expect("prepare");

    backend
        .with_backend_session(|session| sampler.upload_tape(session, &tape))
        .expect("upload tape");

    // Advance until the window forces a rebuild; a rebuild is expected at nsite=8 accepts.
    let results = backend
        .with_backend_session(|session| sampler.run(session))
        .expect("run interval");
    let (done, rebuild, _accepts) = results[0];
    assert!(done > 0, "the interval must advance at least one hop");
    assert!(
        rebuild != 0 || done as usize == total_hops,
        "either the interval completed or a rebuild was requested (done={done}, rebuild={rebuild})"
    );
    eprintln!("interval: hops_done={done} rebuild={rebuild}");

    let pf_out = backend
        .with_backend_session(|session| sampler.download_pf(session, 0))
        .expect("download pf");
    let ele_out = backend
        .with_backend_session(|session| sampler.download_ele(session, 0))
        .expect("download ele");
    assert!(
        pf_out.iter().all(|v| v.is_finite()),
        "Pfaffians must stay finite: {pf_out:?}"
    );
    assert!(
        ele_out.iter().all(|&s| s >= 0 && (s as usize) < nsite),
        "configuration sites must stay in range: {ele_out:?}"
    );
    // With 4 empty sites and a nonzero acceptance weight, at least one hop should move an
    // electron somewhere in 64 proposals over a fresh tape (sanity, not a trajectory check).
    assert_ne!(ele_out, ele_i32, "expected at least one accepted hop");
}

use super::*;
use tenferro_gpu::cuda::{cuda_devices, CudaBackend};
use tenferro_tensor::BackendSessionHost;

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

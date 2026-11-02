//! Shared CUDA stage must not hold its mutex across walker collectives.
use mvmc_core::multichain::{ThreadComm, ThreadReducer};
use mvmc_core::sr_cg::SampledSrOperator;
use mvmc_core::stage_backend::{set_stage_backend_override, StageBackendKind};

#[test]
#[ignore = "optional CUDA gate: requires a device"]
fn shared_cuda_cg_product_and_solve_complete_for_two_walkers() {
    assert!(!tenferro_gpu::cuda::cuda_devices().unwrap().is_empty());
    mvmc_gpu_cuda::install();
    set_stage_backend_override(Some(StageBackendKind::Cuda(0)));
    let comm = ThreadComm::new(2);
    std::thread::scope(|scope| {
        for rank in 0..2 {
            let reducer = ThreadReducer::new(comm.clone(), rank, 0);
            scope.spawn(move || {
                let mut op = SampledSrOperator::new(2, 1, false);
                op.real_samples
                    .copy_from_slice(if rank == 0 { &[1.0, 0.0] } else { &[0.0, 1.0] });
                let mut x = if rank == 0 { [2.0, 3.0] } else { [0.0; 2] };
                let mut out = [0.0; 2];
                op.apply_with_reducer(&mut out, &mut x, 1.0, 0.0, &reducer)
                    .unwrap();
                assert!((out[0] - 2.0).abs() < 1e-11);
                assert!((out[1] - 3.0).abs() < 1e-11);
                let solution = op
                    .solve_with_reducer(&[2.0, 3.0], 1.0, 0.0, 1e-12, 4, &reducer)
                    .unwrap();
                assert!((solution.solution[0] - 2.0).abs() < 1e-11);
                assert!((solution.solution[1] - 3.0).abs() < 1e-11);
            });
        }
    });
    set_stage_backend_override(None);
}

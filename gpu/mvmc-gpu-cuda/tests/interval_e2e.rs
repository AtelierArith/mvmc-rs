//! End-to-end smoke of the device-resident interval group through the real optimizer
//! (issue #547). Ignored by default; requires CUDA.
//!
//! Runs `run_para_opt_multichain` on the Hubbard-chain L16 Opt input with the
//! `cuda-interval` engine (Gutz+Jastrow, hopping only, NExUpdatePath=0) and checks that the
//! whole stack runs and returns finite energies. The device trajectory is its own contract
//! (as the XORWOW lock-step sampler).
//!
//! Open issue: with `MVMC_INTERVAL_E2E_WALKERS > 1` the non-root walkers currently return a
//! NaN final energy even though the per-block sampling is finite (the group gate and the
//! shared-tape parity pass, so the divergence is in the measurement/SR hand-off, not the
//! kernel). The default is one walker, which exercises the full production path.

use std::path::Path;

use mvmc_core::multichain::{run_para_opt_multichain, ParaOptMultiChainConfig};
use mvmc_core::sampler_backend::{set_sampler_backend_override, SamplerBackendKind};
use mvmc_core::RunConfig;
use tenferro_gpu::cuda::gpu_available;

#[test]
#[ignore = "requires CUDA"]
fn interval_group_opt_hubbard_l16_e2e() {
    mvmc_gpu_cuda::install();
    if !gpu_available() {
        eprintln!("skipped: no CUDA device");
        return;
    }
    let namelist = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmark/hubbard_chain/inputs/hubbard_chain_L16/namelist.def");
    assert!(namelist.is_file(), "Hubbard L16 input is required: {namelist:?}");

    let mut run = RunConfig::new(4, "real");
    run.nsmp = Some(2);
    run.seed = Some(1);
    run.output_dir = Some(std::env::temp_dir().join("mvmc-interval-e2e"));

    set_sampler_backend_override(Some(SamplerBackendKind::CudaInterval(0)));
    let n_walkers: usize = std::env::var("MVMC_INTERVAL_E2E_WALKERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let walkers = run_para_opt_multichain(&ParaOptMultiChainConfig {
        namelist,
        run,
        group_base: 0,
        walkers: n_walkers,
    })
    .expect("interval Optimize");
    set_sampler_backend_override(None);

    assert_eq!(walkers.len(), n_walkers);
    let energies: Vec<f64> = walkers
        .iter()
        .map(|w| w.summary.final_energy_per_site)
        .collect();
    eprintln!("interval E2E final energy per site: {energies:?}");
    for e in &energies {
        assert!(e.is_finite(), "non-finite energy per site: {e}");
    }
}

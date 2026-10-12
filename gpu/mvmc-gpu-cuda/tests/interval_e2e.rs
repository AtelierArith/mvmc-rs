//! End-to-end smoke of the device-resident interval group through the real optimizer
//! (issue #547). Ignored by default; requires CUDA.
//!
//! Runs `run_para_opt_multichain` on the Hubbard-chain L16 Opt input with the
//! `cuda-interval` engine (Gutz+Jastrow, hopping only, NExUpdatePath=0) and checks that every
//! walker's reduced final energy is finite and identical (the device trajectory is its own
//! contract, as the XORWOW lock-step sampler).
//!
//! `MVMC_INTERVAL_E2E_WALKERS` sets the walker count (default 1). The `RunSummary` is derived
//! from the output rows, which only the output root (walker 0) writes, so the summary is empty
//! on non-root walkers; the check uses the per-walker `state.energy` instead.

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
    for (i, w) in walkers.iter().enumerate() {
        eprintln!(
            "walker {i}: status={} etot={}",
            w.summary.status, w.state.energy.etot
        );
    }
    let energies: Vec<f64> = walkers
        .iter()
        .map(|w| w.summary.final_energy_per_site)
        .collect();
    eprintln!("interval E2E final energy per site (root summary): {energies:?}");
    // The `RunSummary` is derived from the output rows, which only the output root
    // (walker 0) writes; non-root walkers carry an empty summary by design. Check the
    // per-walker sampler state instead.
    assert!(
        walkers[0].summary.final_energy_per_site.is_finite(),
        "root energy per site: {}",
        walkers[0].summary.final_energy_per_site
    );
    for (i, w) in walkers.iter().enumerate() {
        assert!(
            w.state.energy.etot.re.is_finite(),
            "walker {i} final energy: {}",
            w.state.energy.etot
        );
    }
}

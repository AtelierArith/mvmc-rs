//! L=128 whole-run timing of the interval group through `run_para_opt_multichain`
//! (issue #547). Ignored by default; requires CUDA.
//!
//! `MVMC_INTERVAL_E2E_WALKERS` (default 320) and `MVMC_INTERVAL_E2E_STEPS`/`_SMP` set the
//! walker count and the SR steps / samples per step. Prints the wall time and the final
//! reduced energy. This is the production multiwalker path with the device interval sampler;
//! the host measurement and SR reduction still run on the CPU.

use std::path::Path;
use std::time::Instant;

use mvmc_core::multichain::{run_para_opt_multichain, ParaOptMultiChainConfig};
use mvmc_core::sampler_backend::{set_sampler_backend_override, SamplerBackendKind};
use mvmc_core::RunConfig;
use tenferro_gpu::cuda::gpu_available;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[test]
#[ignore = "requires CUDA"]
fn interval_group_opt_hubbard_l128_timing() {
    mvmc_gpu_cuda::install();
    if !gpu_available() {
        eprintln!("skipped: no CUDA device");
        return;
    }
    let namelist = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmark/hubbard_chain/inputs/hubbard_chain_L128/namelist.def");
    assert!(namelist.is_file(), "Hubbard L128 input is required: {namelist:?}");

    let walkers = env_usize("MVMC_INTERVAL_E2E_WALKERS", 320);
    let steps = env_usize("MVMC_INTERVAL_E2E_STEPS", 2) as i64;
    let smp = env_usize("MVMC_INTERVAL_E2E_SMP", 2) as i64;

    let mut run = RunConfig::new(steps, "real");
    run.nsmp = Some(smp);
    run.seed = Some(1);
    run.output_dir = Some(std::env::temp_dir().join("mvmc-interval-e2e-l128"));

    set_sampler_backend_override(Some(SamplerBackendKind::CudaInterval(0)));
    let t = Instant::now();
    let result = run_para_opt_multichain(&ParaOptMultiChainConfig {
        namelist,
        run,
        group_base: 0,
        walkers,
    })
    .expect("interval Optimize L128");
    let secs = t.elapsed().as_secs_f64();
    set_sampler_backend_override(None);

    let etot = result[0].state.energy.etot.re;
    eprintln!(
        "interval E2E L128: walkers={walkers} steps={steps} smp={smp} wall={secs:.3} s \
         etot={etot}"
    );
    assert!(etot.is_finite(), "final energy: {etot}");
}

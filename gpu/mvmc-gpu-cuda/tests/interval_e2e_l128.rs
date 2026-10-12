//! L=128 whole-run timing of the interval group through `run_para_opt_multichain`
//! (issue #547). Ignored by default; requires CUDA.
//!
//! `MVMC_INTERVAL_E2E_WALKERS` (default 320) and `MVMC_INTERVAL_E2E_STEPS`/`_SMP` set the
//! walker count and the SR steps / final SR averaging window. Each walker retains
//! `NVMCSample` configurations from the input. `_NAMELIST` can select another workload,
//! and `_RECEIPT` writes all walkers' final configurations, RNGs and optimization history
//! after timing, for independent baseline/candidate validation. `_BACKEND=c-order`
//! selects the CPU reference with exactly the same input and walker topology. Prints the wall time and final
//! reduced energy. This is the production multiwalker path with the device interval sampler;
//! the host measurement and SR reduction still run on the CPU.

use std::io::Write;
use std::path::{Path, PathBuf};
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
    let namelist = std::env::var_os("MVMC_INTERVAL_E2E_NAMELIST")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../benchmark/hubbard_chain/inputs/hubbard_chain_L128/namelist.def")
        });
    assert!(
        namelist.is_file(),
        "Hubbard L128 input is required: {namelist:?}"
    );

    let walkers = env_usize("MVMC_INTERVAL_E2E_WALKERS", 320);
    let steps = env_usize("MVMC_INTERVAL_E2E_STEPS", 2) as i64;
    let smp = env_usize("MVMC_INTERVAL_E2E_SMP", 2) as i64;

    let mut run = RunConfig::new(steps, "real");
    run.nsmp = Some(smp);
    run.seed = Some(1);
    run.output_dir = std::env::var_os("MVMC_INTERVAL_E2E_OUTPUT").map(PathBuf::from);

    let backend = match std::env::var("MVMC_INTERVAL_E2E_BACKEND").as_deref() {
        Ok("c-order") => SamplerBackendKind::COrder,
        Ok("cuda-interval") | Err(_) => SamplerBackendKind::CudaInterval(0),
        Ok(value) => panic!("unsupported interval comparison backend: {value}"),
    };
    set_sampler_backend_override(Some(backend));
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
        "interval E2E L128: backend={backend:?} walkers={walkers} steps={steps} smp={smp} wall={secs:.3} s \
         etot={etot}"
    );
    assert!(etot.is_finite(), "final energy: {etot}");

    if let Some(path) = std::env::var_os("MVMC_INTERVAL_E2E_RECEIPT") {
        let mut file = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
        for walker in &result {
            let cfg = &walker.state.electron_config;
            let (rng, index) = walker.rng.state_snapshot();
            // One JSON record per walker. Integer/RNG fields are exact; computed floating
            // arrays are compared numerically by the benchmark's external validator.
            writeln!(file, "{{\"walker\":{},\"rng\":{:?},\"rng_index\":{},\"rng_words\":{},\"ele\":{:?},\"occ\":{:?},\"proj\":{:?},\"counter\":{:?},\"energy\":{:?},\"parameters\":{:?}}}",
                walker.walker, rng.as_slice(), index, walker.rng.words_consumed(),
                cfg.ele_idx, cfg.ele_num, cfg.ele_proj_cnt, cfg.counter,
                walker.state.opt_data.iter().map(|x| [x.energy.re, x.energy.im]).collect::<Vec<_>>(),
                walker.state.opt_data.iter().map(|x| x.parameters.iter().map(|p| [p.re,p.im]).collect::<Vec<_>>()).collect::<Vec<_>>()
            ).unwrap();
        }
    }
}

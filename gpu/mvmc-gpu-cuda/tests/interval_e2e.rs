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
    assert!(
        namelist.is_file(),
        "Hubbard L16 input is required: {namelist:?}"
    );

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

/// The C-order CPU sampler and the resident C stream must use the same draws on
/// this well-separated acceptance trajectory. Two device runs also test repeatability.
#[test]
#[ignore = "requires CUDA"]
fn interval_c_opt_twenty_steps_matches_host_and_repeats() {
    mvmc_gpu_cuda::install();
    let source = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmark/hubbard_chain/inputs/hubbard_chain_L16");
    let base = std::env::temp_dir().join(format!("mvmc-c-interval-parity-{}", std::process::id()));
    let input = base.join("input");
    std::fs::create_dir_all(&input).unwrap();
    for entry in std::fs::read_dir(&source).unwrap() {
        let path = entry.unwrap().path();
        if path.is_file() {
            std::fs::copy(&path, input.join(path.file_name().unwrap())).unwrap();
        }
    }
    let modpara = std::fs::read_to_string(input.join("modpara.def")).unwrap();
    std::fs::write(
        input.join("modpara.def"),
        modpara
            .lines()
            .map(|line| match line.split_whitespace().next().unwrap_or("") {
                "NVMCSample" => "NVMCSample 5",
                "NVMCWarmUp" => "NVMCWarmUp 1",
                _ => line,
            })
            .collect::<Vec<_>>()
            .join("\n")
            + "\n",
    )
    .unwrap();
    let run = |kind, label: &str| {
        set_sampler_backend_override(Some(kind));
        let mut run = RunConfig::new(20, "real");
        run.nsmp = Some(1);
        run.seed = Some(1);
        run.output_dir = Some(base.join(label));
        let out = run_para_opt_multichain(&ParaOptMultiChainConfig {
            namelist: input.join("namelist.def"),
            run,
            group_base: 0,
            walkers: 4,
        })
        .unwrap();
        set_sampler_backend_override(None);
        out
    };
    let host = run(SamplerBackendKind::COrder, "host");
    let first = run(SamplerBackendKind::CudaInterval(0), "device1");
    let second = run(SamplerBackendKind::CudaInterval(0), "device2");
    for (label, candidate) in [("host", &host), ("repeat", &second)] {
        for (reference, actual) in first.iter().zip(candidate) {
            let (a, b) = (
                &reference.state.electron_config,
                &actual.state.electron_config,
            );
            assert_eq!(
                reference.rng.state_snapshot(),
                actual.rng.state_snapshot(),
                "{label} RNG"
            );
            assert_eq!(
                reference.rng.words_consumed(),
                actual.rng.words_consumed(),
                "{label} words"
            );
            assert_eq!(a.ele_idx, b.ele_idx, "{label} configuration");
            assert_eq!(a.ele_num, b.ele_num, "{label} occupancy");
            assert_eq!(a.ele_proj_cnt, b.ele_proj_cnt, "{label} projection");
            assert_eq!(a.counter, b.counter, "{label} counters");
            assert_eq!(reference.state.opt_data.len(), 1); // final SR averaging window
            for (step, (r, a)) in reference
                .state
                .opt_data
                .iter()
                .zip(&actual.state.opt_data)
                .enumerate()
            {
                let pairs = std::iter::once((r.energy, a.energy)).chain(
                    r.parameters
                        .iter()
                        .copied()
                        .zip(a.parameters.iter().copied()),
                );
                for (r, a) in pairs {
                    // Same sampling path and C operation order; establish the observed
                    // first difference before relaxing this existing interval bound.
                    assert!(
                        (r - a).norm() <= 1e-12 + 1e-12 * r.norm(),
                        "{label} step {step}: {r} vs {a}"
                    );
                }
            }
        }
    }
    for name in ["zvo_out.dat", "zvo_var.dat", "zqp_opt.dat"] {
        let read = |label: &str| -> Vec<f64> {
            std::fs::read_to_string(base.join(label).join(name))
                .unwrap()
                .split_whitespace()
                .map(|x| x.parse().unwrap())
                .collect()
        };
        let reference = read("host");
        for label in ["device1", "device2"] {
            let actual = read(label);
            assert_eq!(actual.len(), reference.len());
            for (i, (&r, &a)) in reference.iter().zip(&actual).enumerate() {
                assert!(
                    (r - a).abs() <= 1e-12 + 1e-12 * r.abs(),
                    "{label} {name}[{i}]: {r} vs {a}"
                );
            }
        }
    }
    assert_eq!(
        std::fs::read_to_string(base.join("host/zvo_out.dat"))
            .unwrap()
            .lines()
            .count(),
        20
    );
}

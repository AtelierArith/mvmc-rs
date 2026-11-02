//! Developer-only multi-walker benchmark; see the accompanying #583 report.
use mvmc_core::multichain::{
    run_para_opt_multichain, run_phys_cal_multichain, MultiChainConfig, ParaOptMultiChainConfig,
};
use mvmc_core::run::{InitialDef, RunConfig};
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(
        args.len(),
        6,
        "usage: bench_walkers_583 opt|phys NAMELIST PARAMS WALKERS OUTPUT"
    );
    let namelist = PathBuf::from(&args[2]);
    let walkers: usize = args[4].parse().expect("walker count");
    assert!(walkers > 0);
    match args[1].as_str() {
        "opt" => {
            let mut run = RunConfig::new(20, "real");
            run.nsmp = Some(1);
            run.seed = Some(1);
            run.initial_def = InitialDef::None;
            run.output_dir = Some(PathBuf::from(&args[5]));
            let result = run_para_opt_multichain(&ParaOptMultiChainConfig {
                namelist,
                run,
                group_base: 0,
                walkers,
            })
            .expect("multi-walker Opt");
            for walker in result {
                println!(
                    "D|{}|{}|{:?}|{:?}",
                    walker.walker,
                    walker.rng.words_consumed(),
                    walker.rng.state_snapshot(),
                    walker.state.electron_config
                );
                println!(
                    "N|{}|{:.17e}",
                    walker.walker, walker.summary.final_energy_per_site
                );
            }
        }
        "phys" => {
            let cfg = MultiChainConfig {
                namelist,
                opt_para: Some(PathBuf::from(&args[3])),
                mode: "real".into(),
                seed: Some(1),
                group_base: 0,
                walkers,
                enable_opt_trans: true,
                samples: Some(20),
                threads: walkers,
            };
            let result = run_phys_cal_multichain(&cfg).expect("multi-walker PhysCal");
            for walker in result.walkers {
                let decisions: Vec<_> = walker
                    .decisions
                    .iter()
                    .map(|&(weight, draw)| (weight > draw, (draw * 4294967296.0) as u32))
                    .collect();
                println!(
                    "D|{}|{}|{:?}|{:?}|{:?}",
                    walker.walker,
                    walker.rng_words,
                    walker.result.final_rng.state_snapshot(),
                    walker.result.state.electron_config,
                    decisions
                );
                let energy = &walker.result.state.energy;
                println!(
                    "N|{}|{:.17e} {:.17e} {:.17e} {:.17e}",
                    walker.walker, energy.etot.re, energy.etot.im, energy.etot2.re, energy.etot2.im
                );
            }
        }
        _ => panic!("expected opt or phys"),
    }
    let (attempts, successes, busy, unavailable, participants) =
        mvmc_core::stage_backend::probe_cpu_snapshot();
    eprintln!("shared-team-probe attempts={attempts} successes={successes} busy={busy} unavailable={unavailable} participants={participants}");
}

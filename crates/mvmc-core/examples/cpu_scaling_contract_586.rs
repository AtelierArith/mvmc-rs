//! Developer-only full-state repeatability probe for the L128 CPU scaling report.
//! Run separately from performance measurements; no C/Julia runtime is needed.
use mvmc_core::run::{self, InitialDef, RunConfig};
use mvmc_core::sampling::driver::trace;
use mvmc_core::state::VmcOptimizationState;
use mvmc_core::{threading, SingleProcessReducer};
use num_complex::Complex64;
use sfmt19937::Sfmt19937Rng;
use std::path::PathBuf;

fn snapshot(state: &VmcOptimizationState, rng: &Sfmt19937Rng) -> serde_json::Value {
    let (words, index) = rng.state_snapshot();
    let electron = &state.electron_config;
    let mut numerical = Vec::new();
    let mut add = |value: Complex64| numerical.extend([value.re, value.im]);
    for value in [state.energy.wc, state.energy.etot, state.energy.etot2] {
        add(value);
    }
    for point in &state.opt_data {
        add(point.energy);
        add(point.energy_squared);
        for &value in &point.parameters {
            add(value);
        }
    }
    if let Some(physical) = &state.phys_quantities {
        for &value in physical
            .phys_cis_ajs
            .iter()
            .chain(&physical.phys_cis_ajs_ckt_alt)
            .chain(&physical.phys_cis_ajs_ckt_alt_dc)
        {
            add(value);
        }
    }
    numerical.extend(&state.slater_matrix.pf_m_real);
    numerical.extend(state.slater_matrix.inv_m_real.as_slice());
    numerical.extend(&state.sr_opt.sr_opt_oo_real);
    numerical.extend(&state.sr_opt.sr_opt_ho_real);
    assert!(numerical.iter().all(|value| value.is_finite()));
    serde_json::json!({
        "exact": {"rng_words": words.as_slice(), "rng_index": index,
            "rng_words_consumed": rng.words_consumed(),
            "ele_idx": electron.ele_idx, "ele_cfg": electron.ele_cfg,
            "ele_num": electron.ele_num, "ele_spn": electron.ele_spn,
            "ele_proj_cnt": electron.ele_proj_cnt, "counter": electron.counter},
        "numerical": numerical,
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(
        args.len(),
        5,
        "usage: cpu_scaling_contract_586 opt|phys NAMELIST PARAMS OUTPUT"
    );
    let output = PathBuf::from(&args[4]);
    std::fs::create_dir(&output).expect("fresh probe output directory");
    threading::install(|| {
        trace::start_decisions();
        let mut record = match args[1].as_str() {
            "opt" => {
                let mut config = RunConfig::new(20, "real");
                config.seed = Some(79);
                config.nsmp = Some(1);
                config.initial_def = InitialDef::None;
                config.output_dir = Some(output.join("output"));
                let (summary, state, rng) = run::run_para_opt_from_namelist_observed(
                    &args[2],
                    config,
                    &SingleProcessReducer,
                )
                .expect("L128 optimization probe");
                assert_eq!(summary.effective_nsteps, 20);
                snapshot(&state, &rng)
            }
            "phys" => {
                let preparation =
                    run::prepare_phys_cal_from_namelist(&args[2], &args[3], "real", Some(79))
                        .expect("L128 PhysCal preparation");
                let result = run::vmc_phys_cal_to_dir(preparation, output.join("output"))
                    .expect("L128 PhysCal probe");
                assert_eq!(result.iterations, 1);
                snapshot(&result.state, &result.final_rng)
            }
            _ => panic!("expected opt or phys"),
        };
        let decisions: Vec<_> = trace::finish_decisions()
            .into_iter()
            .map(|(weight, draw)| (weight > draw, (draw * 4294967296.0) as u32))
            .collect();
        record["exact"]["decisions"] = serde_json::to_value(decisions).unwrap();
        std::fs::write(
            output.join("contract.json"),
            serde_json::to_vec(&record).unwrap(),
        )
        .unwrap();
    });
}

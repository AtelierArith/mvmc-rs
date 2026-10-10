//! Optional developer capture of actual SR operands and RNG, without invoking an oracle.
use mvmc_core::run::{InitialDef, RunConfig};
use std::path::PathBuf;

fn main() {
    let mut args = std::env::args().skip(1);
    let namelist = PathBuf::from(args.next().expect("namelist.def"));
    let out = PathBuf::from(args.next().expect("NEW output directory"));
    let opt_trans = args.next().as_deref() == Some("-o");
    std::fs::create_dir(&out).expect("new capture directory");
    let mut config = RunConfig::new(1, "cmp");
    config.nsmp = Some(1);
    config.initial_def = InitialDef::None;
    config.enable_opt_trans = Some(opt_trans);
    config.output_dir = Some(out.join("output"));
    let mut capture = mvmc_core::sr::observer::capture_with_normalized().unwrap();
    let (_, state, rng) = mvmc_core::run::run_para_opt_from_namelist_observed(
        &namelist,
        config,
        &mvmc_core::SingleProcessReducer,
    )
    .expect("optimization capture");
    let normalized = capture.take_normalized();
    let solves = capture.finish();
    let pairs =
        |values: &[num_complex::Complex64]| values.iter().map(|v| [v.re, v.im]).collect::<Vec<_>>();
    let records = normalized
        .iter()
        .map(|n| {
            serde_json::json!({
                "step": n.step, "oo": pairs(&n.oo), "ho": pairs(&n.ho),
                "oo_real": n.oo_real, "ho_real": n.ho_real, "energy": pairs(&n.energy)
            })
        })
        .collect::<Vec<_>>();
    let systems = solves
        .iter()
        .map(|s| {
            serde_json::json!({
                "dimension": s.dimension, "matrix": s.matrix, "rhs": s.rhs,
                "increment": s.increment, "active_indices": s.active_indices,
                "flags": s.flags, "factor_info": s.factor_info, "solve_info": s.solve_info
            })
        })
        .collect::<Vec<_>>();
    let mut peek = rng.clone();
    let next: Vec<_> = (0..624).map(|_| peek.gen_rand32()).collect();
    let c = &state.electron_config;
    let record = serde_json::json!({
        "normalized": records, "systems": systems,
        "o_store": pairs(&state.sr_opt.sr_opt_o_store),
        "rng_words_consumed": rng.words_consumed(), "rng_next624": next,
        "ele_idx": c.ele_idx, "ele_cfg": c.ele_cfg, "ele_num": c.ele_num,
        "ele_spn": c.ele_spn, "ele_proj_cnt": c.ele_proj_cnt, "counter": c.counter
    });
    std::fs::write(
        out.join("diagnostics.json"),
        serde_json::to_vec_pretty(&record).unwrap(),
    )
    .unwrap();
}

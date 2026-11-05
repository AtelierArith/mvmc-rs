//! With `-o`, the OptTrans weights are optimized (#676).
//!
//! C `GetInfoOptTrans` (1.3.0 and 1.4.0) writes undoubled flags at `NProj + NSlater + i`,
//! which freezes the weights and flags unrelated components (a C defect,
//! tmisawa/Julia-mVMC#94). Rust optimizes the real weight components only,
//! as Julia and the resident GPU runners do.
use mvmc_core::run::{InitialDef, RunConfig};
use std::path::{Path, PathBuf};

fn prepared_input(tag: &str, steps: i64) -> PathBuf {
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/physcal_181/hubbard_chain_dh_opttrans/inputs");
    let directory =
        std::env::temp_dir().join(format!("mvmc-issue676-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir(&directory).unwrap();
    for entry in std::fs::read_dir(source).unwrap() {
        let path = entry.unwrap().path();
        if path.is_file() {
            std::fs::copy(&path, directory.join(path.file_name().unwrap())).unwrap();
        }
    }
    let modpara = directory.join("modpara.def");
    let text = std::fs::read_to_string(&modpara).unwrap();
    let text = text
        .lines()
        .map(|line| match line.split_whitespace().next() {
            Some("NVMCCalMode") => "NVMCCalMode 0".to_string(),
            Some("NSROptItrStep") => format!("NSROptItrStep {steps}"),
            Some("NSROptItrSmp") => "NSROptItrSmp 1".to_string(),
            _ => line.to_string(),
        })
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(modpara, text).unwrap();
    directory
}

fn values(path: &Path) -> Vec<f64> {
    std::fs::read_to_string(path)
        .unwrap()
        .split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect()
}

#[test]
fn enabled_opttrans_flags_only_real_weight_components() {
    let directory = prepared_input("flags", 1);
    let namelist = directory.join("namelist.def");
    let with =
        mvmc_expert_parsers::parse_expert_mode_files_with_opt_trans(&namelist, true).unwrap();
    let without =
        mvmc_expert_parsers::parse_expert_mode_files_with_opt_trans(&namelist, false).unwrap();
    std::fs::remove_dir_all(directory).unwrap();
    let count = with.count_opt_trans_parameters();
    assert_eq!(count, 3);
    let tail = 2 * (with.count_variational_parameters() - count);
    assert_eq!(without.optimization_flags.len(), tail);
    // C would also set components NProj + NSlater + i = 35..38 here.
    assert_eq!(
        &with.optimization_flags[..tail],
        without.optimization_flags.as_slice()
    );
    assert_eq!(&with.optimization_flags[tail..], &[1, 0, 1, 0, 1, 0]);
}

#[test]
fn enabled_opttrans_optimization_moves_the_real_weights() {
    let steps = 3;
    let directory = prepared_input("run", steps);
    let mut config = RunConfig::new(steps, "real");
    config.nsmp = Some(1);
    config.initial_def = InitialDef::None;
    config.enable_opt_trans = Some(true);
    config.output_dir = Some(directory.join("output"));
    let capture = mvmc_core::sr::observer::capture().unwrap();
    mvmc_core::run::run_para_opt_from_namelist_observed(
        directory.join("namelist.def"),
        config,
        &mvmc_core::SingleProcessReducer,
    )
    .unwrap();
    let systems = capture.finish();
    assert_eq!(systems.len(), steps as usize);
    let n_flags = systems[0].flags.len();
    let tail = n_flags - 6;
    for (step, system) in systems.iter().enumerate() {
        let active = &system.active_indices;
        assert_eq!(&active[active.len() - 3..], &[tail, tail + 2, tail + 4]);
        assert!(active
            .iter()
            .all(|&i| i < tail || (i - tail).is_multiple_of(2)));
        let increments = &system.increment[active.len() - 3..];
        assert!(
            increments.iter().all(|d| d.is_finite() && *d != 0.0),
            "step {step}: OptTrans increments {increments:?}"
        );
    }
    // With a one-step window `zqp_opt.dat` holds (real part, 0) pairs of the
    // energy, its square and every parameter; the last three parameters are
    // the OptTrans weights. Each step rescales them to unit maximum modulus,
    // so compare ratios with the `InOptTrans` input (0.25, -0.75, 0.5).
    let row = values(&directory.join("output/zqp_opt.dat"));
    std::fs::remove_dir_all(&directory).unwrap();
    let weights: Vec<f64> = row[row.len() - 6..].chunks(2).map(|v| v[0]).collect();
    let max = weights.iter().fold(0.0_f64, |m, w| m.max(w.abs()));
    assert!((max - 1.0).abs() < 1e-12, "normalized weights {weights:?}");
    for (k, initial) in [(0, 0.25 / -0.75), (2, 0.5 / -0.75)] {
        let ratio = weights[k] / weights[1];
        assert!(
            (ratio - initial).abs() > 1e-8,
            "weight ratio {k}/1 must move from {initial}: {ratio} ({weights:?})"
        );
    }
}

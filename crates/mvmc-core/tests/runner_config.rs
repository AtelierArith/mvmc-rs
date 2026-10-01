//! Julia run_para_opt_from_namelist configuration and summary contracts.
use mvmc_core::{run_para_opt_from_namelist, InitialDef, RunConfig};
use std::{fs, path::PathBuf};

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../extern/Julia-mVMC/examples/inputs/heisenberg_chain_real/namelist.def")
}

#[test]
fn invalid_options_fail_before_opening_inputs() {
    for (steps, mode, nsmp, reason) in [
        (0, "real", None, "nsteps must be positive"),
        (-1, "real", None, "nsteps must be positive"),
        (1, "wrong", None, "mode must be"),
        (1, "real", Some(0), "nsmp must be positive"),
        (1, "real", Some(-2), "nsmp must be positive"),
    ] {
        let mut config = RunConfig::new(steps, mode);
        config.nsmp = nsmp;
        assert!(run_para_opt_from_namelist("absent.def", config)
            .unwrap_err()
            .contains(reason));
    }
}

#[test]
fn final_window_must_fit_effective_steps() {
    let config = RunConfig::new(1, "real");
    let error = run_para_opt_from_namelist(fixture(), config).unwrap_err();
    assert!(
        error.contains("nsteps (1) must be >= nsmp (100)"),
        "{error}"
    );
    let mut config = RunConfig::new(1, "real");
    config.nsmp = Some(2);
    assert!(run_para_opt_from_namelist(fixture(), config)
        .unwrap_err()
        .contains("nsteps (1) must be >= nsmp (2)"));
}

#[test]
fn default_directory_and_summary_use_the_actual_last_window() {
    let mut config = RunConfig::new(2, "cmp"); // Label does not select actual mode.
    config.nsmp = Some(2);
    config.seed = Some(1);
    config.initial_def = InitialDef::None;
    let summary = run_para_opt_from_namelist(fixture(), config).unwrap();
    assert_eq!(summary.status, 0);
    assert_eq!(summary.effective_nsteps, 2);
    assert_eq!(summary.effective_nsmp, 2);
    assert!(summary.output_dir.is_absolute());
    assert_eq!(summary.zvo_first_n.len(), 2);
    let rows: Vec<Vec<f64>> = summary
        .zvo_first_n
        .iter()
        .map(|line| {
            line.split_whitespace()
                .map(|value| value.parse().unwrap())
                .collect()
        })
        .collect();
    assert_eq!(
        summary.ctest_values,
        [
            (rows[0][0] + rows[1][0]) / 2.0,
            (rows[0][1] + rows[1][1]) / 2.0
        ]
    );
    assert_eq!(summary.final_energy_per_site, rows[1][0] / 6.0);
    fs::remove_dir_all(summary.output_dir).unwrap();
}

#[test]
fn explicit_missing_initial_file_fails_instead_of_falling_back() {
    let mut config = RunConfig::new(1, "real");
    config.nsmp = Some(1);
    config.initial_def = InitialDef::Path(PathBuf::from("definitely-missing-initial.def"));
    let error = run_para_opt_from_namelist(fixture(), config).unwrap_err();
    assert!(error.contains("explicitly requested path"), "{error}");
}

fn copied_fixture(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("mvmc-runner-inputs-{}-{name}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    for entry in fs::read_dir(fixture().parent().unwrap()).unwrap() {
        let entry = entry.unwrap();
        if entry.path().is_file() {
            fs::copy(entry.path(), dir.join(entry.file_name())).unwrap();
        }
    }
    dir
}

#[test]
fn broken_auto_file_can_be_explicitly_skipped_and_custom_heads_are_read_back() {
    let dir = copied_fixture("auto");
    fs::write(dir.join("initial.def"), "broken").unwrap();
    let modpara = dir.join("modpara.def");
    let text = fs::read_to_string(&modpara)
        .unwrap()
        .replace("CDataFileHead  zvo", "CDataFileHead  custom");
    fs::write(modpara, text).unwrap();
    let mut config = RunConfig::new(1, "real");
    config.nsmp = Some(1);
    config.seed = Some(1);
    config.output_dir = Some(dir.join("outputs"));
    let error = run_para_opt_from_namelist(dir.join("namelist.def"), config.clone()).unwrap_err();
    assert!(error.contains("auto-detected initial.def"), "{error}");
    assert!(!dir.join("outputs").exists());
    config.initial_def = InitialDef::None;
    let summary = run_para_opt_from_namelist(dir.join("namelist.def"), config).unwrap();
    assert!(summary.output_dir.join("custom_out.dat").is_file());
    assert!(!summary.output_dir.join("zvo_out.dat").exists());
    assert_eq!(summary.zvo_first_n.len(), 1);
    assert_eq!(summary.ctest_values[0], summary.final_energy_per_site * 6.0);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn explicit_and_auto_valid_initial_files_produce_identical_trajectories() {
    let dir = copied_fixture("valid-initial");
    let data = mvmc_expert_parsers::parse_expert_mode_files(dir.join("namelist.def")).unwrap();
    let count = data.projection_layout().n_proj
        + mvmc_expert_parsers::utils::parameter_init::n_slater(&data);
    let mut record = "0 0 0 0 0 0".to_owned();
    for index in 0..count {
        record.push_str(&format!(" {} 0 0", (index as f64 + 1.0) / 100.0));
    }
    let initial = dir.join("initial.def");
    fs::write(&initial, record).unwrap();
    let mut config = RunConfig::new(2, "real");
    config.nsmp = Some(1);
    config.seed = Some(1);
    config.output_dir = Some(dir.join("auto"));
    let auto = run_para_opt_from_namelist(dir.join("namelist.def"), config.clone()).unwrap();
    config.output_dir = Some(dir.join("explicit"));
    config.initial_def = InitialDef::Path(initial);
    let explicit = run_para_opt_from_namelist(dir.join("namelist.def"), config).unwrap();
    assert_eq!(auto.zvo_first_n, explicit.zvo_first_n);
    assert_eq!(auto.ctest_values, explicit.ctest_values);
    assert_eq!(auto.final_energy_per_site, explicit.final_energy_per_site);
    let last: Vec<f64> = explicit.zvo_first_n[1]
        .split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect();
    assert_eq!(explicit.ctest_values, [last[0], last[1]]);
    fs::remove_dir_all(dir).unwrap();
}

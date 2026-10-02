//! Errors must reach the process status without creating misleading outputs.
use std::fs;
use std::path::PathBuf;
use std::process::Command;

struct TestDir(PathBuf);

impl TestDir {
    fn new(name: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("mvmc-contract-{name}-{}", std::process::id()));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn unsupported_projection_fails_before_creating_output_directory() {
    let dir = TestDir::new("spin-jastrow");
    let namelist = dir.0.join("namelist.def");
    fs::write(&namelist, "SpinJastrow missing.def\n").unwrap();
    let out_dir = dir.0.join("out");
    let output = Command::new(env!("CARGO_BIN_EXE_mvmc"))
        .arg(namelist)
        .args(["--nsteps", "1", "--out-dir"])
        .arg(&out_dir)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("SpinJastrow"));
    assert!(!out_dir.exists());
}

#[test]
fn missing_hamiltonian_input_cannot_run_a_different_model() {
    let dir = TestDir::new("missing-transfer");
    let namelist = dir.0.join("namelist.def");
    fs::write(&namelist, "Trans missing.def\n").unwrap();
    let out_dir = dir.0.join("out");
    let output = Command::new(env!("CARGO_BIN_EXE_mvmc"))
        .arg(namelist)
        .args(["--nsteps", "1", "--out-dir"])
        .arg(&out_dir)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("Trans file not found"));
    assert!(!out_dir.exists());
}

#[test]
fn nonfinite_sr_failure_stops_after_step_zero_with_nonzero_exit() {
    let dir = TestDir::new("failed-sr");
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../extern/Julia-mVMC/examples/inputs/heisenberg_chain_real");
    for entry in fs::read_dir(&fixture)
        .expect("checked-out Julia fixture")
        .flatten()
    {
        if entry.path().is_file() {
            fs::copy(entry.path(), dir.0.join(entry.file_name())).unwrap();
        }
    }
    let modpara = dir.0.join("modpara.def");
    let mut text = fs::read_to_string(&modpara).unwrap();
    text.push_str("\nDSROptStepDt NaN\n");
    fs::write(modpara, text).unwrap();
    let out_dir = dir.0.join("out");
    let output = Command::new(env!("CARGO_BIN_EXE_mvmc"))
        .arg(dir.0.join("namelist.def"))
        .args(["--nsteps", "2", "--nsmp", "2", "--seed", "1", "--out-dir"])
        .arg(&out_dir)
        .output()
        .unwrap();
    assert!(!output.status.success());
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(error.contains("direct SR failed at step 0"), "{error}");
    assert_eq!(
        fs::read_to_string(out_dir.join("zvo_out.dat"))
            .unwrap()
            .lines()
            .count(),
        1
    );
    assert!(!out_dir.join("zqp_opt.dat").exists());
}

#[test]
fn malformed_runner_options_are_not_silently_ignored() {
    for option in ["--nsteps", "--seed", "--nsmp"] {
        let output = Command::new(env!("CARGO_BIN_EXE_mvmc"))
            .args([option, "invalid"])
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(2));
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(
            error.contains(&format!("{option} requires an integer")),
            "{error}"
        );
    }
}

#[test]
fn timer_environment_controls_reports_without_changing_numerical_output() {
    let dir = TestDir::new("timers");
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../extern/Julia-mVMC/examples/inputs/heisenberg_chain_real/namelist.def");
    let run = |name: &str, enabled: &str, diagnostic: Option<&str>| {
        let out = dir.0.join(name);
        let mut command = Command::new(env!("CARGO_BIN_EXE_mvmc"));
        command
            .arg(&fixture)
            .args([
                "--nsteps",
                "1",
                "--nsmp",
                "1",
                "--seed",
                "1",
                "--initial-def",
                "none",
                "--out-dir",
            ])
            .arg(&out);
        for key in [
            "MVMC_C_TIMER",
            "MVMC_TIMER",
            "MVMC_CALHAM1_DIAG",
            "MVMC_SLATER_DIAG",
            "MVMC_MAINCAL_DIAG",
            "MVMC_WEIGHTAVG_DIAG",
        ] {
            command.env(key, "0");
        }
        command.env("MVMC_C_TIMER", enabled);
        if let Some(key) = diagnostic {
            command.env(key, "");
        }
        let result = command.output().unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        out
    };
    let baseline = run("disabled", "0", None);
    assert!(!baseline.join("zvo_CalcTimer.dat").exists());
    assert!(!baseline.join("zvo_CalcTimerDiag.dat").exists());
    let enabled = run("enabled", "false", None); // Literal "false" enables Julia's switch.
    assert!(enabled.join("zvo_CalcTimer.dat").is_file());
    assert!(!enabled.join("zvo_CalcTimerDiag.dat").exists());
    let legacy = run("legacy", "0", Some("MVMC_TIMER"));
    assert!(legacy.join("zvo_CalcTimer.dat").is_file());
    assert!(!legacy.join("zvo_CalcTimerDiag.dat").exists());
    assert_eq!(
        fs::read(legacy.join("zvo_out.dat")).unwrap(),
        fs::read(baseline.join("zvo_out.dat")).unwrap()
    );
    let golden = include_str!("../../../tests/fixtures/timers/julia_para_opt_zero.dat");
    let report = fs::read_to_string(enabled.join("zvo_CalcTimer.dat")).unwrap();
    assert_eq!(report.lines().count(), golden.lines().count());
    for (row, expected) in report.lines().zip(golden.lines()) {
        assert_eq!(&row[..row.len() - 12], &expected[..expected.len() - 12]);
        let value: f64 = row.split_whitespace().last().unwrap().parse().unwrap();
        assert!(value.is_finite() && value >= 0.0);
    }
    for (index, key) in [
        "MVMC_CALHAM1_DIAG",
        "MVMC_SLATER_DIAG",
        "MVMC_MAINCAL_DIAG",
        "MVMC_WEIGHTAVG_DIAG",
    ]
    .iter()
    .enumerate()
    {
        let diag = run(&format!("diag-{index}"), "0", Some(key));
        assert!(diag.join("zvo_CalcTimer.dat").is_file());
        assert!(diag.join("zvo_CalcTimerDiag.dat").is_file());
        assert_eq!(
            fs::read(diag.join("zvo_out.dat")).unwrap(),
            fs::read(baseline.join("zvo_out.dat")).unwrap()
        );
    }
    assert_eq!(
        fs::read(enabled.join("zvo_out.dat")).unwrap(),
        fs::read(baseline.join("zvo_out.dat")).unwrap()
    );
}

#[test]
fn rbm_namelists_reach_production_and_match_source_output() {
    let dir = TestDir::new("rbm");
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures");
    for case in [
        "rbm_real",
        "rbm_cmp",
        "rbm_general_cmp",
        "rbm_dh24_cmp",
        "rbm_fsz",
    ] {
        let out = dir.0.join(case);
        let output = Command::new(env!("CARGO_BIN_EXE_mvmc"))
            .arg(root.join(format!("rbm/run_{case}/namelist.def")))
            .args(["--nsteps", "1", "--nsmp", "1", "--seed", "1", "--out-dir"])
            .arg(&out)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{case}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        for name in ["zvo_out.dat", "zqp_opt.dat"] {
            assert_eq!(
                fs::read_to_string(out.join(name)).unwrap(),
                fs::read_to_string(
                    root.join(format!("sr_direct/{case}_store_runner/step-1-{name}"))
                )
                .unwrap(),
                "{case} {name}"
            );
        }
    }
}

#[test]
fn nonidentity_opttrans_namelists_reach_production_and_match_source_output() {
    let dir = TestDir::new("opttrans");
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures");
    for case in ["opt_real", "opt_cmp", "opt_fsz", "opt_dh24_rbm_cmp"] {
        let out = dir.0.join(case);
        let output = Command::new(env!("CARGO_BIN_EXE_mvmc"))
            .arg(root.join(format!("opttrans/run_{case}/namelist.def")))
            .args(["--nsteps", "1", "--nsmp", "1", "--seed", "1", "--out-dir"])
            .arg(&out)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{case}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        for name in ["zvo_out.dat", "zqp_opt.dat"] {
            assert_eq!(
                fs::read_to_string(out.join(name)).unwrap(),
                fs::read_to_string(
                    root.join(format!("sr_direct/{case}_store_runner/step-1-{name}"))
                )
                .unwrap(),
                "{case} {name}"
            );
        }
        assert!(!out.join("zqp_opttrans_opt.dat").exists());
    }
}

#[test]
fn opttrans_source_sr_failure_reaches_cli_status_and_preserves_output_boundary() {
    let dir = TestDir::new("opttrans-failed-sr");
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures");
    let input = root.join("opttrans/run_opt_real").canonicalize().unwrap();
    let mut namelist = String::new();
    for line in fs::read_to_string(input.join("namelist.def"))
        .unwrap()
        .lines()
    {
        let mut words = line.split_whitespace();
        let kind = words.next().unwrap();
        let path = input.join(words.next().unwrap());
        let path = if kind == "ModPara" {
            let mut text = fs::read_to_string(path).unwrap();
            text.push_str("\nNStore 0\n");
            let path = dir.0.join("modpara.def");
            fs::write(&path, text).unwrap();
            path
        } else {
            path
        };
        namelist.push_str(&format!("{kind} {}\n", path.display()));
    }
    fs::write(dir.0.join("namelist.def"), namelist).unwrap();
    let out = dir.0.join("out");
    let output = Command::new(env!("CARGO_BIN_EXE_mvmc"))
        .arg(dir.0.join("namelist.def"))
        .args(["--nsteps", "50", "--nsmp", "50", "--seed", "1", "--out-dir"])
        .arg(&out)
        .output()
        .unwrap();
    assert!(!output.status.success());
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(error.contains("direct SR failed at step 29"), "{error}");
    for name in ["zvo_out.dat", "zvo_var.dat"] {
        assert_eq!(
            fs::read_to_string(out.join(name)).unwrap(),
            fs::read_to_string(root.join(format!("sr_direct/opt_real_runner/step-50-{name}")))
                .unwrap(),
            "{name}"
        );
    }
    assert!(!out.join("zqp_opt.dat").exists());
}

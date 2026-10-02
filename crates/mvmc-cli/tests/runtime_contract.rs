//! Errors must reach the process status without creating misleading outputs.
#[path = "../../../tests/support/reference_slater.rs"]
mod reference_slater;
use reference_slater::declared_output;
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
        // Explicit legacy binary inputs preserve these Julia SR goldens.
        // Native integer flag 2 remains fixed for SR in production.
        let namelist = root.join(format!("c_orbital_inputs/namelist_{case}.def"));
        let data = mvmc_expert_parsers::parse_expert_mode_files(&namelist).unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_mvmc"))
            .arg(&namelist)
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
                declared_output(
                    &data,
                    name,
                    fs::read_to_string(
                        root.join(format!("sr_direct/{case}_store_runner/step-1-{name}"))
                    )
                    .unwrap()
                ),
                "{case} {name}"
            );
        }
    }
}

#[test]
fn positive_nonunit_flags_initialize_but_stay_fixed_through_cli_sr_steps() {
    let dir = TestDir::new("c-integer-flags");
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures");
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../extern/Julia-mVMC/examples/inputs/hubbard_chain_real");
    let mut namelist = String::new();
    for line in fs::read_to_string(source.join("namelist.def"))
        .unwrap()
        .lines()
    {
        let words: Vec<_> = line.split_whitespace().collect();
        if matches!(words[0], "Gutzwiller" | "Jastrow") {
            continue;
        }
        let input = if words[0] == "Orbital" {
            dir.0.join("orbital.def")
        } else if words[0] == "ModPara" {
            let text = fs::read_to_string(source.join(words[1])).unwrap();
            fs::write(
                dir.0.join("modpara.def"),
                text + "\nNVMCSample 50\nNStore 1\n",
            )
            .unwrap();
            dir.0.join("modpara.def")
        } else {
            source.join(words[1])
        };
        namelist.push_str(&format!("{} {}\n", words[0], input.display()));
    }
    fs::write(dir.0.join("namelist.def"), namelist).unwrap();
    let run = |flag: i64, steps: usize| {
        let orbital =
            fs::read_to_string(root.join("c_orbital_inputs/ap_hubbard_six_flag2.def")).unwrap();
        let mut lines: Vec<String> = orbital.lines().map(str::to_owned).collect();
        for (index, row) in lines.iter_mut().skip(5 + 36).enumerate() {
            *row = format!("{index} {flag}");
        }
        fs::write(dir.0.join("orbital.def"), lines.join("\n") + "\n").unwrap();
        let out = dir.0.join(format!("flag{flag}-steps{steps}"));
        let result = Command::new(env!("CARGO_BIN_EXE_mvmc"))
            .arg(dir.0.join("namelist.def"))
            .args([
                "--nsteps",
                &steps.to_string(),
                "--nsmp",
                "1",
                "--seed",
                "1",
                "--initial-def",
                "none",
                "--out-dir",
            ])
            .arg(&out)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        fs::read_to_string(out.join("zqp_opt.dat")).unwrap()
    };
    // Actual C InitParameter accepts >0; its SR filter requires exactly 1.
    // The complete AP flag-2 input is separately accepted by the native reader.
    let fixed = run(2, 1);
    assert!(fixed
        .split_whitespace()
        .map(|v| v.parse::<f64>().unwrap())
        .any(|v| v != 0.0));
    assert_eq!(fixed, run(2, 3));
    // Positive binary control proves this workload exercises an effective SR
    // update and would detect incorrectly treating flag 2 as a bool true.
    assert_ne!(run(1, 1), run(1, 3));
}

#[test]
fn nonidentity_opttrans_namelists_reach_production_and_match_source_output() {
    let dir = TestDir::new("opttrans");
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures");
    for case in ["opt_real", "opt_cmp", "opt_fsz", "opt_dh24_rbm_cmp"] {
        let out = dir.0.join(case);
        let namelist = if case == "opt_dh24_rbm_cmp" {
            root.join("c_orbital_inputs/namelist_opt_dh24_rbm_cmp.def")
        } else {
            root.join(format!("opttrans/run_{case}/namelist.def"))
        };
        let data = mvmc_expert_parsers::parse_expert_mode_files(&namelist).unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_mvmc"))
            .arg(&namelist)
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
                declared_output(
                    &data,
                    name,
                    fs::read_to_string(
                        root.join(format!("sr_direct/{case}_store_runner/step-1-{name}"))
                    )
                    .unwrap()
                ),
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
    let data = mvmc_expert_parsers::parse_expert_mode_files(dir.0.join("namelist.def")).unwrap();
    for name in ["zvo_out.dat", "zvo_var.dat"] {
        assert_eq!(
            fs::read_to_string(out.join(name)).unwrap(),
            declared_output(
                &data,
                name,
                fs::read_to_string(root.join(format!("sr_direct/opt_real_runner/step-50-{name}")))
                    .unwrap()
            ),
            "{name}"
        );
    }
    assert!(!out.join("zqp_opt.dat").exists());
}

#[test]
fn zero_translation_count_fails_before_creating_output_directory() {
    let dir = TestDir::new("c-zero-count");
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../extern/Julia-mVMC/examples/inputs/heisenberg_chain_real");
    for entry in fs::read_dir(&fixture).unwrap().flatten() {
        if entry.path().is_file() {
            fs::copy(entry.path(), dir.0.join(entry.file_name())).unwrap();
        }
    }
    let modpara = dir.0.join("modpara.def");
    let text = fs::read_to_string(&modpara)
        .unwrap()
        .replace("NMPTrans       -1", "NMPTrans       0");
    fs::write(modpara, text).unwrap();
    let out = dir.0.join("out");
    let output = Command::new(env!("CARGO_BIN_EXE_mvmc"))
        .arg(dir.0.join("namelist.def"))
        .args(["--nsteps", "1", "--nsmp", "1", "--out-dir"])
        .arg(&out)
        .output()
        .unwrap();
    assert!(!output.status.success());
    let err = String::from_utf8_lossy(&output.stderr);
    assert!(err.contains("NMPTrans"), "{err}");
    assert!(!out.exists());
}

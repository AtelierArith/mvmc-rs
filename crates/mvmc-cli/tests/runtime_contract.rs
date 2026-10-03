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

// Historical Julia runner goldens are checked by the core trajectory tests.
// These frontend checks use the complete C declarations (including fixed-zero
// padding) and C's explicit OptTrans activation/flags through both entry points.
fn check_cli_and_library_model(
    dir: &TestDir,
    case: &str,
    namelist: &std::path::Path,
    opt_trans: bool,
) -> String {
    let suffix = if opt_trans { "enabled" } else { "disabled" };
    let out = dir.0.join(format!("cli-{case}-{suffix}"));
    let data =
        mvmc_expert_parsers::parse_expert_mode_files_with_c_opt_trans(namelist, opt_trans).unwrap();
    assert!(
        data.input_errors.is_empty(),
        "{case}: {:?}",
        data.input_errors
    );
    let mode = if data.i_flg_orbital_general != 0 {
        "fsz"
    } else if mvmc_core::get_all_complex_flag(&data) {
        "cmp"
    } else {
        "real"
    };
    let mut command = Command::new(env!("CARGO_BIN_EXE_mvmc"));
    command
        .arg(namelist)
        .args(["--nsteps", "1", "--nsmp", "1", "--seed", "1", "--out-dir"])
        .arg(&out);
    if opt_trans {
        command.arg("--opt-trans");
    }
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "{case}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let summary = mvmc_core::run_para_opt_from_namelist(
        namelist,
        mvmc_core::RunConfig {
            nsmp: Some(1),
            seed: Some(1),
            enable_opt_trans: Some(opt_trans),
            output_dir: Some(dir.0.join(format!("library-{case}-{suffix}"))),
            ..mvmc_core::RunConfig::new(1, mode)
        },
    )
    .unwrap();
    assert_eq!(summary.status, 0);
    for name in [
        "zvo_out.dat",
        "zvo_var.dat",
        "zqp_opt.dat",
        "zqp_gutzwiller_opt.dat",
        "zqp_jastrow_opt.dat",
        "zqp_orbital_opt.dat",
        "zqp_chargeRBM_physlayer_opt.dat",
        "zqp_spinRBM_physlayer_opt.dat",
        "zqp_generalRBM_physlayer_opt.dat",
        "zqp_chargeRBM_hiddenlayer_opt.dat",
        "zqp_spinRBM_hiddenlayer_opt.dat",
        "zqp_generalRBM_hiddenlayer_opt.dat",
        "zqp_chargeRBM_physhidden_opt.dat",
        "zqp_spinRBM_physhidden_opt.dat",
        "zqp_generalRBM_physhidden_opt.dat",
    ] {
        assert_eq!(
            out.join(name).exists(),
            summary.output_dir.join(name).exists(),
            "{case} {name} presence"
        );
        if out.join(name).exists() {
            assert_eq!(
                fs::read(out.join(name)).unwrap(),
                fs::read(summary.output_dir.join(name)).unwrap(),
                "{case} {suffix} {name}"
            );
        }
    }
    fs::read_to_string(out.join("zvo_out.dat")).unwrap()
}

#[test]
fn rbm_namelists_match_library_with_complete_c_declarations() {
    let dir = TestDir::new("rbm");
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures");
    for case in [
        "rbm_real",
        "rbm_cmp",
        "rbm_general_cmp",
        "rbm_dh24_cmp",
        "rbm_fsz",
    ] {
        check_cli_and_library_model(
            &dir,
            case,
            &root.join(format!("c_orbital_inputs/namelist_{case}.def")),
            false,
        );
    }
}

#[test]
fn normal_interall_scientific_coefficients_and_c_counts_agree_in_cli_and_library() {
    let dir = TestDir::new("c-interall");
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures");
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../extern/Julia-mVMC/examples/inputs/hubbard_chain_real");
    let mut namelist = String::new();
    for line in fs::read_to_string(source.join("namelist.def"))
        .unwrap()
        .lines()
    {
        let fields: Vec<_> = line.split_whitespace().collect();
        if matches!(fields[0], "Trans" | "CoulombIntra") {
            continue;
        }
        let path = if fields[0] == "Orbital" {
            dir.0.join("orbital.def")
        } else {
            source.join(fields[1])
        };
        namelist.push_str(&format!("{} {}\n", fields[0], path.display()));
    }
    namelist.push_str("InterAll interall.def\n");
    fs::write(dir.0.join("namelist.def"), namelist).unwrap();
    let valid = "===\nNInterAll 1\nIgnored 99\n===\n===\n0 0 0 0 0 1 0 1 1e-3 -2e-3\n";
    fs::write(dir.0.join("interall.def"), valid).unwrap();
    let orbital =
        fs::read_to_string(root.join("c_orbital_inputs/ap_hubbard_six_flag2.def")).unwrap();
    for complex in [false, true] {
        let mut orbital_lines: Vec<_> = orbital.lines().map(str::to_owned).collect();
        orbital_lines[2] = format!("ComplexType {}", i64::from(complex));
        fs::write(dir.0.join("orbital.def"), orbital_lines.join("\n") + "\n").unwrap();
        let parsed =
            mvmc_expert_parsers::parse_expert_mode_files(dir.0.join("namelist.def")).unwrap();
        assert!(parsed.input_errors.is_empty(), "{:?}", parsed.input_errors);
        assert_eq!(parsed.inter_all_terms.len(), 1);
        assert_eq!(
            [
                parsed.inter_all_terms[0].value.re,
                parsed.inter_all_terms[0].value.im
            ],
            [1e-3, -2e-3]
        );
        assert_eq!(mvmc_core::get_all_complex_flag(&parsed), complex);
        check_cli_and_library_model(
            &dir,
            if complex {
                "interall-cmp"
            } else {
                "interall-real"
            },
            &dir.0.join("namelist.def"),
            false,
        );
    }
    fs::write(
        dir.0.join("interall.def"),
        valid.replace("NInterAll 1", "NInterAll 2"),
    )
    .unwrap();
    let output_dir = dir.0.join("invalid-count-output");
    let failed = Command::new(env!("CARGO_BIN_EXE_mvmc"))
        .arg(dir.0.join("namelist.def"))
        .args(["--nsteps", "1", "--out-dir"])
        .arg(&output_dir)
        .output()
        .unwrap();
    assert!(!failed.status.success());
    assert!(String::from_utf8_lossy(&failed.stderr).contains("InterAll"));
    assert!(!output_dir.exists());
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
fn nonidentity_opttrans_namelists_match_library_with_c_activation_and_flags() {
    let dir = TestDir::new("opttrans");
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures");
    for case in ["opt_real", "opt_cmp", "opt_fsz", "opt_dh24_rbm_cmp"] {
        let namelist = if case == "opt_dh24_rbm_cmp" {
            root.join("c_orbital_inputs/namelist_opt_dh24_rbm_cmp.def")
        } else {
            root.join(format!("opttrans/run_{case}/namelist.def"))
        };
        let disabled = check_cli_and_library_model(&dir, case, &namelist, false);
        let enabled = check_cli_and_library_model(&dir, case, &namelist, true);
        assert_ne!(
            disabled, enabled,
            "{case}: explicit OptTrans must activate the nonidentity sectors"
        );
    }
}

#[test]
fn opttrans_sr_failure_reaches_cli_status_and_preserves_failure_boundary() {
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
            text.push_str("\nNStore 0\nDSROptStepDt NaN\n");
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
        .args([
            "--opt-trans",
            "--nsteps",
            "50",
            "--nsmp",
            "50",
            "--seed",
            "1",
            "--out-dir",
        ])
        .arg(&out)
        .output()
        .unwrap();
    assert!(!output.status.success());
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(error.contains("direct SR failed at step 0"), "{error}");
    // A nonfinite step size rejects the first update. Sampling/output happen
    // before that update, including the active OptTrans sectors.
    let modpara = dir.0.join("modpara.def");
    let text = fs::read_to_string(&modpara).unwrap();
    fs::write(&modpara, text + "\nDSROptStepDt 0.003\n").unwrap();
    let baseline = dir.0.join("baseline");
    let successful = Command::new(env!("CARGO_BIN_EXE_mvmc"))
        .arg(dir.0.join("namelist.def"))
        .args([
            "--opt-trans",
            "--nsteps",
            "1",
            "--nsmp",
            "1",
            "--seed",
            "1",
            "--out-dir",
        ])
        .arg(&baseline)
        .output()
        .unwrap();
    assert!(
        successful.status.success(),
        "{}",
        String::from_utf8_lossy(&successful.stderr)
    );
    for name in ["zvo_out.dat", "zvo_var.dat"] {
        let actual = fs::read_to_string(out.join(name)).unwrap();
        assert_eq!(
            actual.lines().count(),
            1,
            "{name}: stop at the first failed update"
        );
        assert_eq!(
            actual,
            fs::read_to_string(baseline.join(name)).unwrap(),
            "{name}: SR failure must preserve the pre-update sample/output"
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

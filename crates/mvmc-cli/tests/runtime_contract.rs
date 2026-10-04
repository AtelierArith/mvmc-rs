//! Errors must reach the process status without creating misleading outputs.
#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;

// CLI and library use the same numerical kernels. A 1e-12 rounding budget
// covers accumulated sample/normalization output without relaxing file shape,
// indexed parameter coordinates or header formatting.
fn assert_numeric_output(actual: &str, expected: &str, indexed: bool, context: &str) {
    if indexed {
        let actual_lines: Vec<_> = actual.lines().collect();
        let expected_lines: Vec<_> = expected.lines().collect();
        assert!(actual_lines.len() >= 4 && expected_lines.len() >= 4);
        assert_eq!(
            &actual_lines[..4],
            &expected_lines[..4],
            "{context}: parameter header"
        );
        numerical_comparison::assert_numeric_text(
            &actual_lines[4..].join("\n"),
            &expected_lines[4..].join("\n"),
            1e-12,
            1e-12,
            &[0],
            context,
        );
    } else {
        numerical_comparison::assert_numeric_text(actual, expected, 1e-12, 1e-12, &[], context);
    }
}

fn assert_numeric_difference(first: &str, second: &str, context: &str) {
    let values = |text: &str| {
        text.split_whitespace()
            .map(|word| word.parse::<f64>().unwrap())
            .collect::<Vec<_>>()
    };
    let first = values(first);
    let second = values(second);
    assert_eq!(first.len(), second.len(), "{context}: vector length");
    assert!(
        first
            .iter()
            .zip(&second)
            .any(|(&a, &b)| !numerical_comparison::within(a, b, 1e-12, 1e-12)),
        "{context}: expected a numerical change beyond the rounding budget"
    );
}

use std::fs;
use std::path::PathBuf;
use std::process::Command;

#[cfg(feature = "mpi")]
fn mpi_cli_command() -> Command {
    let version = Command::new("mpirun").arg("--version").output().unwrap();
    assert!(version.status.success(), "cannot identify MPI launcher");
    let metadata = format!(
        "{}{}",
        String::from_utf8_lossy(&version.stdout),
        String::from_utf8_lossy(&version.stderr)
    );
    let mut command = Command::new("timeout");
    command.args(["--kill-after=5s", "45s", "mpirun"]);
    if metadata.contains("Open MPI") || metadata.contains("OpenRTE") {
        command.arg("--oversubscribe");
    } else {
        assert!(
            metadata.contains("HYDRA build details"),
            "unsupported MPI launcher metadata: {metadata}"
        );
        // Wait for each rank's own exit status after collective CLI errors,
        // rather than requesting Hydra's automatic peer cleanup.
        // The outer timeout still detects deadlocks and forces termination.
        command.arg("-disable-auto-cleanup");
    }
    command
}

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

fn copy_physcal_fixture(dir: &TestDir) -> (PathBuf, PathBuf) {
    copy_physcal_model_fixture(dir, "heisenberg_chain_real")
}

fn copy_physcal_model_fixture(dir: &TestDir, model: &str) -> (PathBuf, PathBuf) {
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!(
        "../../extern/Julia-mVMC/test/integration/reference/{model}/physcal_ref"
    ));
    let inputs = dir.0.join("inputs");
    fs::create_dir_all(&inputs).unwrap();
    for entry in fs::read_dir(source.join("inputs")).unwrap().flatten() {
        if entry.path().is_file() {
            fs::copy(entry.path(), inputs.join(entry.file_name())).unwrap();
        }
    }
    let fixed = dir.0.join("zqp_opt.dat");
    fs::copy(source.join("zqp_opt.dat"), &fixed).unwrap();
    (inputs.join("namelist.def"), fixed)
}

#[test]
fn physcal_cli_runs_fixed_parameters_and_writes_green_outputs() {
    let dir = TestDir::new("physcal-positive");
    let (namelist, fixed) = copy_physcal_fixture(&dir);
    let out_dir = dir.0.join("out");
    let output = Command::new(env!("CARGO_BIN_EXE_mvmc"))
        .args(["--physcal", fixed.to_str().unwrap()])
        .arg(namelist)
        .args(["--seed", "1", "--mode", "real", "--opt-trans", "--out-dir"])
        .arg(&out_dir)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("Completed 1 PhysCal samples"));
    for file in [
        "zvo_out_001.dat",
        "zvo_var_001.dat",
        "zvo_cisajs_001.dat",
        "zvo_cisajscktalt_001.dat",
        "zvo_cisajscktaltex_001.dat",
    ] {
        assert!(out_dir.join(file).is_file(), "missing {file}");
    }
    assert!(!out_dir.join("zvo_out.dat").exists());
    assert!(!out_dir.join("zvo_var.dat").exists());
}

#[test]
fn physcal_cli_rejects_missing_fixed_parameter_file_before_output() {
    let dir = TestDir::new("physcal-missing-fixed");
    let (namelist, _) = copy_physcal_fixture(&dir);
    let out_dir = dir.0.join("out");
    let output = Command::new(env!("CARGO_BIN_EXE_mvmc"))
        .arg(namelist)
        .args(["--physcal", dir.0.join("missing.dat").to_str().unwrap()])
        .args(["--out-dir", out_dir.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("fixed parameter file not found"));
    assert!(!out_dir.exists());
}

#[test]
fn physcal_cli_requires_group_communicator_before_output() {
    let dir = TestDir::new("physcal-grouped");
    let (namelist, fixed) = copy_physcal_fixture(&dir);
    let modpara = dir.0.join("inputs/modpara.def");
    let text = fs::read_to_string(&modpara)
        .unwrap()
        .replace("NSplitSize     1", "NSplitSize     2");
    fs::write(modpara, text).unwrap();
    let out_dir = dir.0.join("out");
    let output = Command::new(env!("CARGO_BIN_EXE_mvmc"))
        .arg(namelist)
        .args(["--physcal", fixed.to_str().unwrap(), "--out-dir"])
        .arg(&out_dir)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("requires an MPI group communicator"));
    assert!(!out_dir.exists());
}

#[test]
fn physcal_cli_rejects_grouped_lanczos_before_output() {
    let dir = TestDir::new("physcal-grouped-lanczos");
    let (namelist, fixed) = copy_physcal_fixture(&dir);
    let modpara = dir.0.join("inputs/modpara.def");
    let text = fs::read_to_string(&modpara)
        .unwrap()
        .replace("NSplitSize     1", "NSplitSize     2")
        .replace("NLanczosMode   0", "NLanczosMode   1");
    fs::write(modpara, text).unwrap();
    let out_dir = dir.0.join("out");
    let output = Command::new(env!("CARGO_BIN_EXE_mvmc"))
        .arg(namelist)
        .args(["--physcal", fixed.to_str().unwrap(), "--out-dir"])
        .arg(&out_dir)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("NSplitSize > 1 with NLanczosMode"));
    assert!(!out_dir.exists());
}

#[test]
fn physcal_cli_rejects_grouped_fsz_before_output() {
    let dir = TestDir::new("physcal-grouped-fsz");
    let (namelist, fixed) = copy_physcal_model_fixture(&dir, "heisenberg_chain_fsz");
    let modpara = dir.0.join("inputs/modpara.def");
    let text = fs::read_to_string(&modpara)
        .unwrap()
        .replace("NSplitSize     1", "NSplitSize     2");
    fs::write(modpara, text).unwrap();
    let out_dir = dir.0.join("out");
    let output = Command::new(env!("CARGO_BIN_EXE_mvmc"))
        .arg(namelist)
        .arg("--physcal")
        .arg(fixed)
        .arg("--out-dir")
        .arg(&out_dir)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("FSZ / general-orbital PhysCal"));
    assert!(!out_dir.exists());
}

#[cfg(feature = "mpi")]
#[test]
#[ignore = "requires a live MPI launcher; run explicitly in the MPI verification environment"]
fn grouped_normal_physcal_cli_succeeds_on_two_and_four_ranks() {
    let dir = TestDir::new("physcal-grouped-mpi-positive");
    let (namelist, fixed) = copy_physcal_fixture(&dir);
    let modpara = dir.0.join("inputs/modpara.def");
    let text = fs::read_to_string(&modpara)
        .unwrap()
        .replace("NSplitSize     1", "NSplitSize     2");
    fs::write(modpara, text).unwrap();
    for ranks in [2, 4] {
        let out_dir = dir.0.join(format!("out-{ranks}"));
        let output = mpi_cli_command()
            .arg("-n")
            .arg(ranks.to_string())
            .arg(env!("CARGO_BIN_EXE_mvmc"))
            .arg(&namelist)
            .arg("--physcal")
            .arg(&fixed)
            .args(["--seed", "1", "--mode", "real", "--out-dir"])
            .arg(&out_dir)
            .env("OMPI_ALLOW_RUN_AS_ROOT", "1")
            .env("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")
            .env("OPENBLAS_NUM_THREADS", "1")
            .env("OMP_NUM_THREADS", "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{ranks} ranks: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8_lossy(&output.stdout);
        for marker in [
            "model    :",
            "mode     :",
            "sample   :",
            "=== mvmc",
            "namelist :",
            "out-dir  :",
            "physcal  :",
            "seed     :",
            "=== Completed",
            "Output files written to:",
        ] {
            assert_eq!(
                stdout.matches(marker).count(),
                1,
                "{ranks} ranks duplicated {marker}: {stdout}"
            );
        }
        for file in [
            "zvo_out_001.dat",
            "zvo_var_001.dat",
            "zvo_cisajs_001.dat",
            "zvo_cisajscktalt_001.dat",
            "zvo_cisajscktaltex_001.dat",
        ] {
            assert!(
                out_dir.join(file).is_file(),
                "{ranks} ranks: missing {file}"
            );
        }
        assert!(!out_dir.join("zvo_out.dat").exists());
        assert!(!out_dir.join("zvo_var.dat").exists());
    }
}

#[test]
fn unsupported_projection_fails_before_creating_output_directory() {
    let dir = TestDir::new("spin-jastrow");
    let namelist = dir.0.join("namelist.def");
    // Do not let the unrelated zero-valued Rust ModPara default mask the
    // unsupported-section boundary. C's no-projection setting is explicitly 1.
    fs::write(dir.0.join("modpara.def"), "NMPTrans 1\n").unwrap();
    fs::write(&namelist, "ModPara modpara.def\nSpinJastrow missing.def\n").unwrap();
    let out_dir = dir.0.join("out");
    let output = Command::new(env!("CARGO_BIN_EXE_mvmc"))
        .arg(namelist)
        .args(["--nsteps", "1", "--out-dir"])
        .arg(&out_dir)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("SpinJastrow"),
        "actual stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!out_dir.exists());
}

#[cfg(feature = "mpi")]
#[test]
#[ignore = "requires a live MPI launcher; run explicitly in the MPI verification environment"]
fn asymmetric_physcal_cli_errors_are_collective_before_output() {
    let dir = TestDir::new("physcal-asymmetric-mpi");
    let (namelist, fixed) = copy_physcal_fixture(&dir);
    for fault in ["parse", "fixed", "setup"] {
        let output_dir = dir.0.join(format!("out-{fault}"));
        let missing = dir.0.join("missing");
        let blocked = dir.0.join("blocked");
        if fault == "setup" {
            fs::write(&blocked, "not a directory").unwrap();
        }
        let mut command = mpi_cli_command();
        for rank in 0..2 {
            if rank != 0 {
                command.arg(":");
            }
            command.args(["-n", "1"]).arg(env!("CARGO_BIN_EXE_mvmc"));
            command.arg(if fault == "parse" && rank == 1 {
                &missing
            } else {
                &namelist
            });
            command
                .arg("--physcal")
                .arg(if fault == "fixed" && rank == 1 {
                    &missing
                } else {
                    &fixed
                });
            command
                .args(["--seed", "1", "--out-dir"])
                .arg(if fault == "setup" && rank == 0 {
                    &blocked
                } else {
                    &output_dir
                });
        }
        let output = command
            .env("OMPI_ALLOW_RUN_AS_ROOT", "1")
            .env("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")
            .env("OPENBLAS_NUM_THREADS", "1")
            .env("OMP_NUM_THREADS", "1")
            .output()
            .unwrap();
        assert!(
            !output.status.success(),
            "{fault} unexpectedly succeeded: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert_ne!(
            output.status.code(),
            Some(124),
            "{fault} deadlocked: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_ne!(
            output.status.code(),
            Some(137),
            "{fault} required forced termination"
        );
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("another MPI rank"),
            "{fault}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            !output_dir.exists(),
            "{fault} mutated peer output directory"
        );
    }
}

#[cfg(feature = "mpi")]
#[test]
#[ignore = "requires a live MPI launcher; run explicitly in the MPI verification environment"]
fn mismatched_valid_cli_controls_fail_collectively_before_output() {
    let dir = TestDir::new("mismatched-cli-controls-mpi");
    let (namelist, fixed) = copy_physcal_fixture(&dir);
    let peer_dir = TestDir::new("mismatched-cli-controls-peer-mpi");
    let (peer_namelist, _) = copy_physcal_fixture(&peer_dir);
    let peer_modpara = peer_dir.0.join("inputs/modpara.def");
    let text = fs::read_to_string(&peer_modpara).unwrap();
    fs::write(
        &peer_modpara,
        text.replace("NDataQtySmp    1", "NDataQtySmp    2"),
    )
    .unwrap();
    let optimization_namelist = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../extern/Julia-mVMC/examples/inputs/heisenberg_chain_real/namelist.def");
    for fault in ["nsteps", "run-kind", "mode", "sample-count"] {
        let output_dir = dir.0.join(format!("out-{fault}"));
        let mut command = mpi_cli_command();
        for rank in 0..2 {
            if rank != 0 {
                command.arg(":");
            }
            command.args(["-n", "1"]).arg(env!("CARGO_BIN_EXE_mvmc"));
            command.arg(if fault == "nsteps" {
                &optimization_namelist
            } else if fault == "sample-count" && rank == 1 {
                &peer_namelist
            } else {
                &namelist
            });
            if fault != "nsteps" && !(fault == "run-kind" && rank == 1) {
                command.arg("--physcal").arg(&fixed);
            }
            command.args([
                "--nsteps",
                if fault == "nsteps" && rank == 1 {
                    "2"
                } else {
                    "1"
                },
            ]);
            command.args([
                "--mode",
                if fault == "mode" && rank == 1 {
                    "cmp"
                } else {
                    "real"
                },
            ]);
            command.args(["--seed", "1", "--out-dir"]).arg(&output_dir);
        }
        let output = command
            .env("OMPI_ALLOW_RUN_AS_ROOT", "1")
            .env("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")
            .env("OPENBLAS_NUM_THREADS", "1")
            .env("OMP_NUM_THREADS", "1")
            .output()
            .unwrap();
        assert!(
            !output.status.success(),
            "{fault} unexpectedly succeeded: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert_ne!(output.status.code(), Some(124), "{fault} deadlocked");
        assert_ne!(
            output.status.code(),
            Some(137),
            "{fault} required forced termination"
        );
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("CLI run controls differ"),
            "{fault}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            output.stdout.is_empty(),
            "{fault} printed before control agreement"
        );
        assert!(!output_dir.exists(), "{fault} mutated output directory");
    }
}

#[test]
fn missing_hamiltonian_input_cannot_run_a_different_model() {
    let dir = TestDir::new("missing-transfer");
    let namelist = dir.0.join("namelist.def");
    // Explicit accepted projection setting, leaving only the intended missing
    // Hamiltonian record as this negative fixture's diagnostic target.
    fs::write(dir.0.join("modpara.def"), "NMPTrans 1\n").unwrap();
    fs::write(&namelist, "ModPara modpara.def\nTrans missing.def\n").unwrap();
    let out_dir = dir.0.join("out");
    let output = Command::new(env!("CARGO_BIN_EXE_mvmc"))
        .arg(namelist)
        .args(["--nsteps", "1", "--out-dir"])
        .arg(&out_dir)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("Trans file not found"),
        "actual stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
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
    assert_numeric_output(
        &fs::read_to_string(legacy.join("zvo_out.dat")).unwrap(),
        &fs::read_to_string(baseline.join("zvo_out.dat")).unwrap(),
        false,
        "legacy observation output",
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
        assert_numeric_output(
            &fs::read_to_string(diag.join("zvo_out.dat")).unwrap(),
            &fs::read_to_string(baseline.join("zvo_out.dat")).unwrap(),
            false,
            "diag observation output",
        );
    }
    assert_numeric_output(
        &fs::read_to_string(enabled.join("zvo_out.dat")).unwrap(),
        &fs::read_to_string(baseline.join("zvo_out.dat")).unwrap(),
        false,
        "enabled observation output",
    );
}

// PhysCal (`NVMCCalMode=1`) emits its own C-format `OutputTimerPhysCal` report.
// The committed zero golden `tests/fixtures/timers/julia_phys_cal_zero.dat` was
// generated with `MVMCOptimizers.write_ctimer_phys_cal(CTimer(false), dir)` on
// Julia 1.13.1 from the julia-patch branch.
#[test]
fn physcal_timer_environment_controls_report_without_changing_output() {
    let dir = TestDir::new("physcal-timer");
    let (namelist, fixed) = copy_physcal_fixture(&dir);
    let run = |name: &str, enabled: &str, diagnostic: Option<&str>| {
        let out = dir.0.join(name);
        let mut command = Command::new(env!("CARGO_BIN_EXE_mvmc"));
        command
            .arg(&namelist)
            .args(["--physcal", fixed.to_str().unwrap()])
            .args(["--seed", "1", "--mode", "real", "--opt-trans", "--out-dir"])
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
    let enabled = run("enabled", "false", None); // Literal "false" enables C's switch.
    assert!(enabled.join("zvo_CalcTimer.dat").is_file());
    assert!(!enabled.join("zvo_CalcTimerDiag.dat").exists());

    // The timer is additive: every non-timer output must be byte-identical.
    let outputs = |path: &std::path::Path| {
        let mut names: Vec<String> = fs::read_dir(path)
            .unwrap()
            .flatten()
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .filter(|name| !name.starts_with("zvo_CalcTimer"))
            .collect();
        names.sort();
        names
    };
    let baseline_files = outputs(&baseline);
    let enabled_files = outputs(&enabled);
    assert_eq!(baseline_files, enabled_files);
    assert!(!baseline_files.is_empty());
    for name in &baseline_files {
        assert_eq!(
            fs::read(baseline.join(name)).unwrap(),
            fs::read(enabled.join(name)).unwrap(),
            "timer perturbed {name}"
        );
    }

    let golden = include_str!("../../../tests/fixtures/timers/julia_phys_cal_zero.dat");
    let report = fs::read_to_string(enabled.join("zvo_CalcTimer.dat")).unwrap();
    assert_eq!(report.lines().count(), golden.lines().count());
    for (row, expected) in report.lines().zip(golden.lines()) {
        assert_eq!(&row[..row.len() - 12], &expected[..expected.len() - 12]);
        let value: f64 = row.split_whitespace().last().unwrap().parse().unwrap();
        assert!(value.is_finite() && value >= 0.0);
    }

    // A diagnostic family enables the parent timer and writes the diag report.
    let diag = run("diag", "0", Some("MVMC_CALHAM1_DIAG"));
    assert!(diag.join("zvo_CalcTimer.dat").is_file());
    assert!(diag.join("zvo_CalcTimerDiag.dat").is_file());
    for name in &baseline_files {
        assert_eq!(
            fs::read(baseline.join(name)).unwrap(),
            fs::read(diag.join(name)).unwrap(),
            "diag timer perturbed {name}"
        );
    }
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
    } else if mvmc_core::get_all_complex_flag(&data).unwrap() {
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
            assert_numeric_output(
                &fs::read_to_string(out.join(name)).unwrap(),
                &fs::read_to_string(summary.output_dir.join(name)).unwrap(),
                name.starts_with("zqp_") && name != "zqp_opt.dat",
                &format!("{case} {suffix} {name}"),
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
        assert_eq!(mvmc_core::get_all_complex_flag(&parsed).unwrap(), complex);
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
        let text = fs::read_to_string(out.join("zqp_opt.dat")).unwrap();
        let fields: Vec<_> = text.split_whitespace().collect();
        // C's one-record window begins with measured Etot/Etot2 pairs;
        // these change with the sampled frame even when every Para is fixed.
        // Compare only the declared post-SR parameter pairs for this flag test.
        let parsed =
            mvmc_expert_parsers::parse_expert_mode_files(dir.0.join("namelist.def")).unwrap();
        assert_eq!(
            fields.len(),
            2 * (2 + parsed.count_variational_parameters())
        );
        fields[4..].join(" ") + "\n"
    };
    // Actual C InitParameter accepts >0; its SR filter requires exactly 1.
    // The complete AP flag-2 input is separately accepted by the native reader.
    let fixed = run(2, 1);
    assert!(fixed
        .split_whitespace()
        .map(|v| v.parse::<f64>().unwrap())
        .any(|v| v != 0.0));
    assert_numeric_output(&fixed, &run(2, 3), false, "nonunit flag remains fixed");
    // Positive binary control proves this workload exercises an effective SR
    // update and would detect incorrectly treating flag 2 as a bool true.
    assert_numeric_difference(&run(1, 1), &run(1, 3), "binary flag enables SR update");
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
        assert_numeric_difference(
            &disabled,
            &enabled,
            &format!("{case}: explicit OptTrans activates nonidentity sectors"),
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
        assert_numeric_output(
            &actual,
            &fs::read_to_string(baseline.join(name)).unwrap(),
            false,
            &format!("{name}: SR failure preserves pre-update sample/output"),
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

#[test]
fn paired_zero_translation_cli_rejects_and_signed_units_remain_valid() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    for (physcal, count) in [
        (false, 0),
        (true, 0),
        (false, 1),
        (false, -1),
        (true, 1),
        (true, -1),
    ] {
        // Exclusive ownership: never delete a pre-existing test directory.
        let dir = loop {
            let path = std::env::temp_dir().join(format!(
                "mvmc-projection-boundary-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => break TestDir(path),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => panic!("{e}"),
            }
        };
        let source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/physcal_181/heisenberg_chain_real");
        let inputs = dir.0.join("inputs");
        fs::create_dir(&inputs).unwrap();
        for entry in fs::read_dir(source.join("inputs")).unwrap() {
            let entry = entry.unwrap();
            assert!(entry.path().is_file());
            fs::copy(entry.path(), inputs.join(entry.file_name())).unwrap();
        }
        let fixed = dir.0.join("zqp_opt.dat");
        fs::copy(source.join("zqp_opt.dat"), &fixed).unwrap();
        let modpara = inputs.join("modpara.def");
        let original = fs::read_to_string(&modpara).unwrap();
        let mut replaced = 0;
        let mut edited = String::new();
        for line in original.lines() {
            let value = match line.split_whitespace().next().unwrap_or("") {
                "NMPTrans" => {
                    replaced += 1;
                    Some(count)
                }
                "NVMCCalMode" => Some(i32::from(physcal)),
                "NVMCWarmUp" | "NVMCInterval" | "NSROptItrStep" | "NSROptItrSmp" => Some(1),
                "NVMCSample" => Some(3),
                _ => None,
            };
            if let Some(value) = value {
                edited.push_str(&format!(
                    "{} {value}\n",
                    line.split_whitespace().next().unwrap()
                ));
            } else {
                edited.push_str(line);
                edited.push('\n');
            }
        }
        assert_eq!(replaced, 1);
        fs::write(&modpara, edited).unwrap();
        if count < 0 {
            // C GetInfoOrbitalAntiParallel requires the fourth sign column
            // for negative NMPTrans. The archived fixture omits it; make this
            // owned valid-input control explicit, not a parser-default test.
            // TransSym already has all four columns, with explicit +1 signs.
            let orbital = inputs.join("orbitalidx.def");
            let text = fs::read_to_string(&orbital).unwrap();
            let mut signed_rows = 0;
            let text = text
                .lines()
                .map(|line| {
                    let columns: Vec<_> = line.split_whitespace().collect();
                    if columns.len() == 3 && columns.iter().all(|word| word.parse::<i64>().is_ok())
                    {
                        signed_rows += 1;
                        format!("{line} 1\n")
                    } else {
                        format!("{line}\n")
                    }
                })
                .collect::<String>();
            assert_eq!(signed_rows, 36);
            fs::write(orbital, text).unwrap();
            let trans = fs::read_to_string(inputs.join("qptransidx.def")).unwrap();
            let rows: Vec<_> = trans
                .lines()
                .filter(|line| line.split_whitespace().count() == 4)
                .collect();
            assert_eq!(rows.len(), 12);
            assert!(rows
                .iter()
                .all(|line| line.split_whitespace().last() == Some("1")));
        }
        let namelist = inputs.join("namelist.def");
        if !physcal {
            let text = fs::read_to_string(&namelist).unwrap();
            fs::write(
                &namelist,
                text.lines()
                    .filter(|line| line.split_whitespace().next() != Some("TwoBodyGEx"))
                    .map(|line| format!("{line}\n"))
                    .collect::<String>(),
            )
            .unwrap();
        }
        let mut before: Vec<_> = fs::read_dir(&inputs)
            .unwrap()
            .map(|entry| {
                let path = entry.unwrap().path();
                let bytes = fs::read(&path).unwrap();
                (path, bytes)
            })
            .collect();
        before.push((fixed.clone(), fs::read(&fixed).unwrap()));
        let out = dir.0.join("must-not-exist");
        let mut command = Command::new(env!("CARGO_BIN_EXE_mvmc"));
        command
            .arg(&namelist)
            .args([
                "--seed",
                "11272",
                "--mode",
                "real",
                "--nsteps",
                "1",
                "--nsmp",
                "1",
                "--out-dir",
            ])
            .arg(&out);
        if physcal {
            command.arg("--physcal").arg(&fixed);
        }
        let result = command.output().unwrap();
        let stderr = String::from_utf8_lossy(&result.stderr);
        println!("ISSUE178_NMP_CLI physcal={physcal} count={count} status={} output_exists={} stdout={} stderr={stderr}", result.status, out.exists(), String::from_utf8_lossy(&result.stdout));
        for (path, bytes) in before {
            assert_eq!(fs::read(path).unwrap(), bytes);
        }
        if count == 0 {
            assert!(!result.status.success(), "{stderr}");
            assert!(stderr.contains("NMPTrans must be nonzero; use 1 for no translation projection (mVMC C contract)"), "{stderr}");
            assert!(!out.exists());
        } else {
            assert!(result.status.success(), "{stderr}");
            assert!(out
                .join(if physcal {
                    "zvo_out_001.dat"
                } else {
                    "zvo_out.dat"
                })
                .is_file());
        }
    }
}

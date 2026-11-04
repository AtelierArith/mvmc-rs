//! Issue #464: `MVMC_RS_SR_BACKEND` is validated once at startup, before any IO. Invalid
//! selectors and unavailable backends are usage errors (status 2) with a clear message, never a
//! panic (status 101); `tenferro` runs and `c-order` is unchanged. Issue #613: the tenferro
//! measurement Pfaffian runs and reproduces the C-order measurement.
#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;

use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures")
}

fn setup(tag: &str) -> (PathBuf, PathBuf) {
    let root = std::env::temp_dir().join(format!("issue464-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    let inputs = root.join("opt");
    fs::create_dir_all(&inputs).unwrap();
    let from = fixtures().join("issue347_varbin/opt_inputs");
    for entry in fs::read_dir(from).unwrap().flatten() {
        if entry.path().is_file() {
            fs::copy(entry.path(), inputs.join(entry.file_name())).unwrap();
        }
    }
    (root, inputs)
}

fn run(inputs: &Path, out: &Path, backend: Option<&str>) -> Output {
    run_env(inputs, out, backend, None)
}

fn run_env(inputs: &Path, out: &Path, backend: Option<&str>, measure: Option<&str>) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_mvmc"));
    cmd.arg(inputs.join("namelist.def"))
        .arg("--out-dir")
        .arg(out)
        .arg("--nsteps")
        .arg("2")
        .arg("--nsmp")
        .arg("1");
    match backend {
        Some(v) => cmd.env("MVMC_RS_SR_BACKEND", v),
        None => cmd.env_remove("MVMC_RS_SR_BACKEND"),
    };
    match measure {
        Some(v) => cmd.env("MVMC_RS_MEASURE_PF_BACKEND", v),
        None => cmd.env_remove("MVMC_RS_MEASURE_PF_BACKEND"),
    };
    cmd.output().unwrap()
}

fn assert_usage_error(output: &Output, out: &Path, needle: &str) {
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(2), "stderr: {stderr}");
    assert!(stderr.starts_with("error: "), "stderr: {stderr}");
    assert!(stderr.contains("MVMC_RS_SR_BACKEND"), "stderr: {stderr}");
    assert!(stderr.contains(needle), "stderr: {stderr}");
    assert!(!stderr.contains("panicked"), "stderr: {stderr}");
    assert!(
        !out.exists(),
        "no output directory may be created before the check"
    );
}

#[test]
fn invalid_selector_is_a_usage_error_before_io() {
    let (root, inputs) = setup("invalid");
    let out = root.join("out");
    let output = run(&inputs, &out, Some("bogus"));
    assert_usage_error(&output, &out, "not one of c-order, tenferro");
    let _ = fs::remove_dir_all(root);
}

#[test]
fn removed_cuda_selector_is_a_usage_error_before_io() {
    // The legacy CUDA stage backend was removed (#643): `cuda[:N]` is an invalid value like any
    // other, rejected by the preflight before any IO.
    for value in ["cuda", "cuda:0", "cuda:x"] {
        let (root, inputs) = setup(&format!("cuda-{}", value.replace(':', "-")));
        let out = root.join("out");
        let output = run(&inputs, &out, Some(value));
        assert_usage_error(&output, &out, "not one of c-order, tenferro");
        let _ = fs::remove_dir_all(root);
    }
}

#[test]
fn removed_cuda_measurement_consumer_is_a_usage_error_before_io() {
    let (root, inputs) = setup("consumer-cuda");
    let out = root.join("out");
    let output = Command::new(env!("CARGO_BIN_EXE_mvmc"))
        .arg(inputs.join("namelist.def"))
        .arg("--out-dir")
        .arg(&out)
        .args(["--nsteps", "2", "--nsmp", "1"])
        .env_remove("MVMC_RS_SR_BACKEND")
        .env_remove("MVMC_RS_MEASURE_PF_BACKEND")
        .env("MVMC_RS_MEASURE_BACKEND", "cuda")
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(2), "stderr: {stderr}");
    assert!(stderr.starts_with("error: "), "stderr: {stderr}");
    assert!(
        stderr.contains("MVMC_RS_MEASURE_BACKEND"),
        "stderr: {stderr}"
    );
    assert!(
        stderr.contains("not one of direct, c-order, tenferro"),
        "stderr: {stderr}"
    );
    assert!(!stderr.contains("panicked"), "stderr: {stderr}");
    assert!(!out.exists());
    let _ = fs::remove_dir_all(root);
}

#[test]
fn tenferro_runs_and_c_order_is_the_default() {
    let (root, inputs) = setup("tenferro");
    let out_c = root.join("out_c");
    let out_t = root.join("out_t");
    let out_d = root.join("out_d");
    let c = run(&inputs, &out_c, Some("c-order"));
    let d = run(&inputs, &out_d, None);
    let t = run(&inputs, &out_t, Some("tenferro"));
    for (name, o) in [("c-order", &c), ("default", &d), ("tenferro", &t)] {
        assert!(
            o.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&o.stderr)
        );
    }
    assert_eq!(
        fs::read(out_c.join("zqp_opt.dat")).unwrap(),
        fs::read(out_d.join("zqp_opt.dat")).unwrap(),
        "an unset selector is the C-order backend"
    );
    assert!(out_t.join("zqp_opt.dat").exists());
    let _ = fs::remove_dir_all(root);
}

#[test]
fn invalid_measurement_pfaffian_selector_is_a_usage_error_before_io() {
    let (root, inputs) = setup("measure");
    let out = root.join("out");
    let output = run_env(&inputs, &out, None, Some("bogus"));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(2), "stderr: {stderr}");
    assert!(stderr.starts_with("error: "), "stderr: {stderr}");
    assert!(
        stderr.contains("MVMC_RS_MEASURE_PF_BACKEND"),
        "stderr: {stderr}"
    );
    assert!(!stderr.contains("panicked"), "stderr: {stderr}");
    assert!(!out.exists());
    let _ = fs::remove_dir_all(&root);

    let (root, inputs) = setup("measure-cuda");
    let out = root.join("out");
    let output = run_env(&inputs, &out, None, Some("cuda"));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(2), "stderr: {stderr}");
    assert!(
        stderr.contains("MVMC_RS_MEASURE_PF_BACKEND"),
        "stderr: {stderr}"
    );
    assert!(
        stderr.contains("not one of calc-m-all, c-order, tenferro"),
        "stderr: {stderr}"
    );
    assert!(!out.exists());
    let _ = fs::remove_dir_all(root);
}

/// Run Opt (`--nsteps 2 --nsmp 1`) or, with `physcal`, PhysCal on the fixed parameters, with
/// the measurement Pfaffian selector `measure` (`None`: unset, the C-order `calc-m-all`).
fn run_measure(inputs: &Path, out: &Path, measure: Option<&str>, physcal: Option<&Path>) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_mvmc"));
    cmd.arg(inputs.join("namelist.def"))
        .arg("--out-dir")
        .arg(out)
        .env_remove("MVMC_RS_SR_BACKEND")
        .env_remove("MVMC_RS_MEASURE_BACKEND")
        .env("MVMC_RS_TENFERRO_THREADS", "1")
        .env("OPENBLAS_NUM_THREADS", "1");
    match physcal {
        Some(parameters) => cmd.arg("--physcal").arg(parameters),
        None => cmd.args(["--nsteps", "2", "--nsmp", "1"]),
    };
    match measure {
        Some(v) => cmd.env("MVMC_RS_MEASURE_PF_BACKEND", v),
        None => cmd.env_remove("MVMC_RS_MEASURE_PF_BACKEND"),
    };
    cmd.output().unwrap()
}

/// Issue #613: the tenferro measurement Pfaffian runs Opt and PhysCal and reproduces the
/// default C-order `calc-m-all` path. Its executor is the C-order PfaPack sequence, so no
/// roundoff difference is expected (none was observed); the bound only avoids a bitwise
/// assertion of computed values (#190) and is tighter than the native-C PhysCal bound
/// (`1e-13`, `1e-10`). The time files are compared exactly except for their wall-clock stamp:
/// they carry the acceptance counters, so equal rows mean the same sampling trajectory.
#[test]
fn tenferro_measurement_pfaffian_matches_c_order_in_opt_and_physcal() {
    let (root, opt_inputs) = setup("measure-tenferro");
    // PhysCal: a real Hubbard chain with Green functions on its fixed parameters.
    let phys = fixtures().join("native_c_physcal_181/hubbard_chain_real");
    let phys_inputs = phys.join("inputs");
    let phys_parameters = phys.join("zqp_opt.dat");
    for (mode, inputs, parameters) in [
        ("opt", &opt_inputs, None),
        ("phys", &phys_inputs, Some(phys_parameters.as_path())),
    ] {
        let outputs: Vec<PathBuf> = [None, Some("tenferro"), Some("tenferro-cpu")]
            .into_iter()
            .map(|measure| {
                let out = root.join(format!("out-{mode}-{}", measure.unwrap_or("default")));
                let output = run_measure(inputs, &out, measure, parameters);
                assert!(
                    output.status.success(),
                    "{mode} {measure:?}: {}",
                    String::from_utf8_lossy(&output.stderr)
                );
                out
            })
            .collect();
        let mut files: Vec<String> = fs::read_dir(&outputs[0])
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        files.sort();
        assert!(
            files.iter().any(|f| f.starts_with("zvo_out")),
            "{mode}: {files:?}"
        );
        for out in &outputs[1..] {
            let mut other: Vec<String> = fs::read_dir(out)
                .unwrap()
                .map(|e| e.unwrap().file_name().into_string().unwrap())
                .collect();
            other.sort();
            assert_eq!(other, files, "{mode}: file inventory");
            for file in &files {
                let reference = fs::read_to_string(outputs[0].join(file)).unwrap();
                let produced = fs::read_to_string(out.join(file)).unwrap();
                if file.starts_with("zvo_time") {
                    let rows = |text: &str| -> Vec<String> {
                        text.lines()
                            .map(|l| l.rsplit_once(": ").map_or(l, |(head, _)| head).to_owned())
                            .collect()
                    };
                    assert_eq!(rows(&produced), rows(&reference), "{mode}/{file}");
                    continue;
                }
                numerical_comparison::assert_numeric_text(
                    produced.trim_end(),
                    reference.trim_end(),
                    1e-14,
                    1e-13,
                    &[],
                    format!("{mode}/{file}"),
                );
            }
        }
    }
    let _ = fs::remove_dir_all(root);
}

/// Issue #613: the stage-backend measurement Pfaffian supports real, non-FSZ inputs; another
/// mode is a usage error after input parsing and before any output, not a panic.
#[test]
fn stage_measurement_pfaffian_rejects_complex_inputs_before_output() {
    let scenario = fixtures().join("native_c_physcal_181/heisenberg_chain_cmp");
    let root = std::env::temp_dir().join(format!("issue464-measure-cmp-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    for alias in ["tenferro", "c-order"] {
        let out = root.join(format!("out-{alias}"));
        let output = run_measure(
            &scenario.join("inputs"),
            &out,
            Some(alias),
            Some(&scenario.join("zqp_opt.dat")),
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert_eq!(output.status.code(), Some(2), "{alias}: {stderr}");
        assert!(
            stderr.contains("error: MVMC_RS_MEASURE_PF_BACKEND"),
            "{alias}: {stderr}"
        );
        assert!(stderr.contains("only real, non-FSZ"), "{alias}: {stderr}");
        assert!(stderr.contains("cmp calculation"), "{alias}: {stderr}");
        assert!(!stderr.contains("panicked"), "{alias}: {stderr}");
        assert!(!out.exists(), "{alias}: output created");
    }
    let _ = fs::remove_dir_all(root);
}

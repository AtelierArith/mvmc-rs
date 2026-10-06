//! Issue #464: `MVMC_RS_SR_BACKEND` is validated once at startup, before any IO. Invalid
//! selectors and unavailable backends are usage errors (status 2) with a clear message, never a
//! panic (status 101); `tenferro` runs and `c-order` is unchanged.
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
    assert_usage_error(&output, &out, "not one of c-order, tenferro, cuda[:N]");
    let _ = fs::remove_dir_all(root);
}

#[test]
fn malformed_cuda_ordinal_is_a_usage_error() {
    let (root, inputs) = setup("ordinal");
    let out = root.join("out");
    let output = run(&inputs, &out, Some("cuda:x"));
    assert_usage_error(&output, &out, "not one of");
    let _ = fs::remove_dir_all(root);
}

#[test]
fn cuda_without_a_registered_provider_is_a_usage_error_before_io() {
    // The stock binary has no CUDA provider (with or without the `gpu-cuda` feature).
    let (root, inputs) = setup("cuda");
    let out = root.join("out");
    let output = run(&inputs, &out, Some("cuda"));
    // Without the feature: "built without the `gpu-cuda` feature"; with it: "no CUDA provider registered".
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    assert!(
        stderr.contains("gpu-cuda") || stderr.contains("CUDA provider"),
        "stderr: {stderr}"
    );
    assert_usage_error(&output, &out, "unsupported");
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
    assert!(!out.exists());
    let _ = fs::remove_dir_all(root);
}

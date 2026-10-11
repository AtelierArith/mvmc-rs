//! Issue #519 milestone 1: stage B (local energy and Slater derivative) of the real, non-FSZ
//! measurement through the unified stage backend.
//!
//! (a) The routed C-order measurement stage is byte-identical to the pre-existing inline
//!     calls for a representative small real non-FSZ Hubbard input.
//! (b) A backend whose measurement stage cannot run the model (here the tenferro CPU backend
//!     on a Doublon-Holon model, which is outside its real `CalH1` fast path) is a hard error,
//!     never a silent CPU fallback. The tenferro stage itself is validated for plain real
//!     non-FSZ Hubbard inputs in `measurement_tenferro_518.rs`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use mvmc_core::measurement_batch::{set_measurement_consumer_override, MeasurementConsumer};
use mvmc_core::stage_backend::StageBackendKind;
use mvmc_core::RunConfig;

/// Both tests mutate process-wide overrides; serialize them inside one test binary.
static SERIAL: Mutex<()> = Mutex::new(());

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn rewrite_modpara(inputs: &Path, overrides: &[(&str, &str)]) {
    let modpara = std::fs::read_to_string(inputs.join("modpara.def")).unwrap();
    let mut out = String::new();
    let mut seen = Vec::new();
    for line in modpara.lines() {
        let key = line.split_whitespace().next().unwrap_or("");
        match overrides.iter().find(|(k, _)| k.eq_ignore_ascii_case(key)) {
            Some((k, v)) => {
                seen.push(*k);
                out.push_str(&format!("{k} {v}\n"));
            }
            None => {
                out.push_str(line);
                out.push('\n');
            }
        }
    }
    for (k, v) in overrides {
        if !seen.contains(k) {
            out.push_str(&format!("{k} {v}\n"));
        }
    }
    std::fs::write(inputs.join("modpara.def"), out).unwrap();
}

/// Short real non-FSZ optimization run; returns every non-timer output file's bytes.
fn run_optimization(fixture: &str, tag: &str) -> BTreeMap<String, Vec<u8>> {
    let source = repo_root().join("tests/fixtures/physcal_181").join(fixture);
    let root = std::env::temp_dir().join(format!(
        "mvmc-issue519-{}-{fixture}-{tag}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    let inputs = root.join("inputs");
    std::fs::create_dir_all(&inputs).unwrap();
    for entry in std::fs::read_dir(source.join("inputs")).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), inputs.join(entry.file_name())).unwrap();
    }
    rewrite_modpara(
        &inputs,
        &[
            ("NVMCCalMode", "0"),
            ("NSROptItrStep", "2"),
            ("NSROptItrSmp", "2"),
            ("NVMCSample", "20"),
            ("NVMCWarmUp", "5"),
            ("NSRCG", "0"),
            ("NStore", "1"),
            ("NDataIdxStart", "1"),
            ("NDataQtySmp", "1"),
            ("NLanczosMode", "0"),
        ],
    );
    let output = root.join("output");
    let mut config = RunConfig::new(2, "real");
    config.output_dir = Some(output.clone());
    config.seed = Some(1);
    config.enable_opt_trans = Some(false);
    mvmc_core::run_para_opt_from_namelist(inputs.join("namelist.def"), config)
        .unwrap_or_else(|e| panic!("{fixture}: {e}"));
    let mut files = BTreeMap::new();
    for entry in std::fs::read_dir(&output).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.contains("Timer") || name.starts_with("zvo_time") {
            continue;
        }
        files.insert(name.clone(), std::fs::read(entry.path()).unwrap());
    }
    assert!(!files.is_empty());
    let _ = std::fs::remove_dir_all(&root);
    files
}

#[test]
fn routed_c_order_measurement_is_byte_identical_to_the_inline_calls() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    set_measurement_consumer_override(Some(MeasurementConsumer::Direct));
    let direct = run_optimization("hubbard_chain_dh_real", "direct");
    set_measurement_consumer_override(Some(MeasurementConsumer::Stage(StageBackendKind::COrder)));
    let routed = run_optimization("hubbard_chain_dh_real", "routed");
    // The default (no override) is the C-order measurement stage too.
    set_measurement_consumer_override(None);
    let default = run_optimization("hubbard_chain_dh_real", "default");

    assert_eq!(
        direct.keys().collect::<Vec<_>>(),
        routed.keys().collect::<Vec<_>>()
    );
    for (name, bytes) in &direct {
        assert_eq!(bytes, &routed[name], "routed file {name} differs");
        assert_eq!(bytes, &default[name], "default file {name} differs");
    }
}

#[test]
fn unsupported_measurement_stage_is_a_hard_error() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    set_measurement_consumer_override(Some(MeasurementConsumer::Stage(
        StageBackendKind::TenferroCpu,
    )));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_optimization("hubbard_chain_dh_real", "unsupported")
    }));
    set_measurement_consumer_override(None);

    let payload = result.expect_err("expected the unsupported measurement stage to fail");
    let message = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
        .unwrap_or_default();
    assert!(
        message.contains("measurement stage failed"),
        "panic message did not name the measurement stage: {message}"
    );
}

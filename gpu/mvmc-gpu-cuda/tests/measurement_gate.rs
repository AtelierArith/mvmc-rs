//! Optional CUDA gate for the measurement stage A through the unified stage backend
//! (issues #422, #437). Ignored by default; `MVMC_RS_CUDA_GATE=1` requires a device.
//!
//! Fixed-parameter PhysCal on Hubbard fixtures: the sampler and its RNG are unchanged by the
//! measurement backend, so both runs measure the same saved configurations and only the
//! Pfaffian/inverse tables differ. Every numeric output token must agree with the default
//! `calc_m_all_real` run within the bound below; non-numeric tokens and the file set exactly.
//!
//! Bound: the tables differ by Pfaffian/inverse reordering error, O(n eps cond) per entry with
//! n = 6 electrons and well-conditioned fixtures (the harness bound for the same stage is
//! `abs 1e-12 / rel 1e-10`); measured quantities are sums of at most a few hundred table
//! entries, so `abs 1e-11 / rel 1e-9` (measured worst 1.8e-12 relative on an RTX 3060) leaves a
//! margin of ~500 yet fails on any layout or sign defect (O(1) errors).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use mvmc_core::backend::{
    cuda_device_count, cuda_gate_decision, CudaGateDecision, CUDA_GATE_VARIABLE,
};
use mvmc_core::measurement_batch::{
    set_measurement_backend_override, set_measurement_batch_size_override, MeasurementPfaffian,
};
use mvmc_core::stage_backend::StageBackendKind;

fn gate() -> bool {
    mvmc_gpu_cuda::install();
    let requested = std::env::var(CUDA_GATE_VARIABLE).ok();
    match cuda_gate_decision(requested.as_deref(), cuda_device_count()) {
        CudaGateDecision::SkippedNoDevice(why) => {
            eprintln!("measurement-gate: ExplicitSkip: skipped, no device ({why})");
            false
        }
        CudaGateDecision::FailNoDevice(why) => {
            panic!("measurement-gate: {CUDA_GATE_VARIABLE} requested but no device: {why}")
        }
        CudaGateDecision::Run => true,
    }
}

fn physcal(fixture: &str, tag: &str) -> BTreeMap<String, String> {
    let source = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/physcal_181")
        .join(fixture);
    let root: PathBuf = std::env::temp_dir().join(format!(
        "mvmc-measurement-gate-{}-{fixture}-{tag}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    let inputs = root.join("inputs");
    std::fs::create_dir_all(&inputs).unwrap();
    for entry in std::fs::read_dir(source.join("inputs")).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), inputs.join(entry.file_name())).unwrap();
    }
    std::fs::copy(source.join("zqp_opt.dat"), root.join("zqp_opt.dat")).unwrap();
    let preparation = mvmc_core::prepare_phys_cal_from_namelist(
        inputs.join("namelist.def"),
        root.join("zqp_opt.dat"),
        "real",
        Some(1),
    )
    .unwrap_or_else(|e| panic!("{fixture}: {e}"));
    let output = root.join("output");
    mvmc_core::vmc_phys_cal_to_dir(preparation, &output)
        .unwrap_or_else(|e| panic!("{fixture}: {e}"));
    let mut files = BTreeMap::new();
    for entry in std::fs::read_dir(&output).unwrap() {
        let name = entry.unwrap().file_name().to_string_lossy().into_owned();
        if name.contains("Timer") || name.starts_with("zvo_time") {
            continue;
        }
        files.insert(
            name.clone(),
            std::fs::read_to_string(output.join(&name)).unwrap(),
        );
    }
    let _ = std::fs::remove_dir_all(&root);
    files
}

fn close(a: f64, b: f64, abs: f64, rel: f64) -> bool {
    (a - b).abs() <= abs + rel * a.abs().max(b.abs())
}

#[test]
#[ignore = "optional CUDA gate: needs a device (MVMC_RS_CUDA_GATE=1 to require it)"]
fn cuda_measurement_stage_matches_calc_m_all() {
    if !gate() {
        return;
    }
    let (abs, rel) = (1e-11, 1e-9);
    for fixture in ["hubbard_chain_real", "hubbard_all_terms_lanczos2"] {
        set_measurement_batch_size_override(Some(7));
        set_measurement_backend_override(None);
        let reference = physcal(fixture, "ref");
        set_measurement_backend_override(Some(MeasurementPfaffian::Stage(StageBackendKind::Cuda(
            0,
        ))));
        let device = physcal(fixture, "cuda");
        set_measurement_backend_override(None);
        assert_eq!(
            reference.keys().collect::<Vec<_>>(),
            device.keys().collect::<Vec<_>>(),
            "{fixture}: file set"
        );
        let mut worst = 0.0_f64;
        let mut compared = 0usize;
        for (name, text) in &reference {
            let (a, b): (Vec<&str>, Vec<&str>) = (
                text.split_whitespace().collect(),
                device[name].split_whitespace().collect(),
            );
            assert_eq!(a.len(), b.len(), "{fixture} {name}: token count");
            for (x, y) in a.iter().zip(&b) {
                match (x.parse::<f64>(), y.parse::<f64>()) {
                    (Ok(x), Ok(y)) => {
                        let scale = x.abs().max(y.abs()).max(1e-300);
                        worst = worst.max((x - y).abs() / scale);
                        compared += 1;
                        assert!(close(x, y, abs, rel), "{fixture} {name}: {x} vs {y}");
                    }
                    _ => assert_eq!(x, y, "{fixture} {name}: non-numeric token"),
                }
            }
        }
        assert!(compared > 0);
        println!("{fixture}: {compared} numeric tokens, worst relative difference {worst:.2e}");
    }
    set_measurement_batch_size_override(None);
}

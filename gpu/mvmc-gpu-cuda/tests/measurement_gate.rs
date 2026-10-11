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
    set_measurement_backend_override, set_measurement_batch_size_override,
    set_measurement_consumer_override, MeasurementConsumer, MeasurementPfaffian,
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

/// Issue #518 milestone 1: the measurement stage B (local energy through the transfer
/// ratio table; Slater derivative through the per-sector QP reductions) through the
/// tenferro CUDA placement, against the C-order consumer.
///
/// Both runs sample the same saved configurations (the measurement backend does not touch
/// the sampler or its RNG); only the measurement arithmetic differs. Bound: the stage
/// re-associates sums of at most `n_qp * n_size^2` separately rounded products per output,
/// so per-sample deviations are O(1e-15) for these well-conditioned L=6 fixtures; after
/// two SR steps the propagated differences stay under `abs 1e-11 / rel 1e-11` (measured
/// worst 1.37e-12 absolute and 7.5e-13 relative on the A100, about 7-13x inside the
/// bound), which still fails on any layout or sign defect (O(1) errors).
#[test]
#[ignore = "optional CUDA gate: needs a device (MVMC_RS_CUDA_GATE=1 to require it)"]
fn cuda_measurement_stage_b_matches_c_order() {
    if !gate() {
        return;
    }
    let (abs, rel) = (1e-11, 1e-11);
    set_measurement_batch_size_override(Some(7));

    // PhysCal: the local energy of every saved sample.
    set_measurement_consumer_override(None);
    let reference = physcal("hubbard_chain_real", "stageB-ref");
    set_measurement_consumer_override(Some(MeasurementConsumer::Stage(StageBackendKind::Cuda(0))));
    let device = physcal("hubbard_chain_real", "stageB-cuda");
    set_measurement_consumer_override(None);
    compare_stage_b_files("physcal:hubbard_chain_real", &reference, &device, abs, rel);

    // Optimization: the local energy and the Slater derivative, then two SR steps.
    let reference = optimization(None, "stageB-opt-ref");
    let device = optimization(
        Some(MeasurementConsumer::Stage(StageBackendKind::Cuda(0))),
        "stageB-opt-cuda",
    );
    compare_stage_b_files("opt:hubbard_chain_real", &reference, &device, abs, rel);

    set_measurement_consumer_override(None);
    set_measurement_batch_size_override(None);
}

fn compare_stage_b_files(
    label: &str,
    reference: &BTreeMap<String, String>,
    device: &BTreeMap<String, String>,
    abs: f64,
    rel: f64,
) {
    assert_eq!(
        reference.keys().collect::<Vec<_>>(),
        device.keys().collect::<Vec<_>>(),
        "{label}: file set"
    );
    let mut worst = (0.0_f64, 0.0_f64);
    let mut compared = 0usize;
    for (name, text) in reference {
        let (a, b): (Vec<&str>, Vec<&str>) = (
            text.split_whitespace().collect(),
            device[name].split_whitespace().collect(),
        );
        assert_eq!(a.len(), b.len(), "{label} {name}: token count");
        for (x, y) in a.iter().zip(&b) {
            match (x.parse::<f64>(), y.parse::<f64>()) {
                (Ok(x), Ok(y)) => {
                    let scale = x.abs().max(y.abs()).max(1e-300);
                    worst.0 = worst.0.max((x - y).abs());
                    if scale > 1e-6 {
                        worst.1 = worst.1.max((x - y).abs() / scale);
                    }
                    compared += 1;
                    assert!(close(x, y, abs, rel), "{label} {name}: {x} vs {y}");
                }
                _ => assert_eq!(x, y, "{label} {name}: non-numeric token"),
            }
        }
    }
    assert!(compared > 0);
    println!(
        "{label}: {compared} numeric tokens, worst absolute {:.2e}, worst relative (|v| > 1e-6) {:.2e}",
        worst.0, worst.1
    );
}

fn optimization(consumer: Option<MeasurementConsumer>, tag: &str) -> BTreeMap<String, String> {
    let source = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/physcal_181/hubbard_chain_real");
    let root: PathBuf = std::env::temp_dir().join(format!(
        "mvmc-measurement-gate-{}-stageB-{tag}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    let inputs = root.join("inputs");
    std::fs::create_dir_all(&inputs).unwrap();
    for entry in std::fs::read_dir(source.join("inputs")).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), inputs.join(entry.file_name())).unwrap();
    }
    let modpara = std::fs::read_to_string(inputs.join("modpara.def")).unwrap();
    let overrides = [
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
    ];
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
        if !seen.contains(&k) {
            out.push_str(&format!("{k} {v}\n"));
        }
    }
    std::fs::write(inputs.join("modpara.def"), out).unwrap();

    set_measurement_consumer_override(consumer);
    let output = root.join("output");
    let mut config = mvmc_core::RunConfig::new(2, "real");
    config.output_dir = Some(output.clone());
    config.seed = Some(1);
    config.enable_opt_trans = Some(false);
    mvmc_core::run_para_opt_from_namelist(inputs.join("namelist.def"), config)
        .unwrap_or_else(|e| panic!("{tag}: {e}"));
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

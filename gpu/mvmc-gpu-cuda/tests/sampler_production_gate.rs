//! Optional CUDA gate for the production sampler routing (issue #517). Ignored by default;
//! `MVMC_RS_CUDA_GATE=1` requires a device (hard failure without one), anything else skips
//! explicitly.
//!
//! `MVMC_RS_SAMPLER_BACKEND=cuda` (selected by override here) routes the production optimization
//! and PhysCal sampling through the device-resident lock-step sampler. With no numerical
//! acceptance divergence the run must equal C order exactly: same SFMT state and draw count, same
//! saved configurations and counters, same final energy. Any flip would appear as a difference
//! here, not as noise (the sampler gate of `sampler_gate.rs` is the teacher-forced locator).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use mvmc_core::backend::{
    cuda_device_count, cuda_gate_decision, CudaGateDecision, CUDA_GATE_VARIABLE,
};
use mvmc_core::sampler_backend::{set_sampler_backend_override, SamplerBackendKind};
use mvmc_core::{PhysCalResult, RunConfig, SingleProcessReducer};

static SERIAL: Mutex<()> = Mutex::new(());

fn gate() -> bool {
    mvmc_gpu_cuda::install();
    let requested = std::env::var(CUDA_GATE_VARIABLE).ok();
    match cuda_gate_decision(requested.as_deref(), cuda_device_count()) {
        CudaGateDecision::SkippedNoDevice(why) => {
            eprintln!("sampler-production-gate: ExplicitSkip: skipped, no device ({why})");
            false
        }
        CudaGateDecision::FailNoDevice(why) => {
            panic!("sampler-production-gate: {CUDA_GATE_VARIABLE} requested but no device: {why}")
        }
        CudaGateDecision::Run => true,
    }
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read_output_files(output: &Path) -> BTreeMap<String, String> {
    let mut files = BTreeMap::new();
    for entry in std::fs::read_dir(output).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.contains("Timer") || name.starts_with("zvo_time") {
            continue;
        }
        files.insert(name, std::fs::read_to_string(entry.path()).unwrap());
    }
    files
}

/// Copy the Hubbard L=16 inputs to `root/inputs` and apply the modpara overrides.
fn stage_hubbard_inputs(root: &Path, overrides: &[(&str, &str)]) {
    let source = repo_root().join("benchmark/hubbard_chain/inputs/hubbard_chain_L16");
    let inputs = root.join("inputs");
    let _ = std::fs::remove_dir_all(root);
    std::fs::create_dir_all(&inputs).unwrap();
    for entry in std::fs::read_dir(&source).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name();
        if name == "StdFace.def" {
            continue;
        }
        std::fs::copy(entry.path(), inputs.join(name)).unwrap();
    }
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

struct OptOutcome {
    files: BTreeMap<String, String>,
    rng: ([u32; 624], usize),
    consumed: u128,
    ele_idx: Vec<i64>,
    counter: Vec<i64>,
}

fn run_opt(kind: SamplerBackendKind, root: &Path) -> OptOutcome {
    set_sampler_backend_override(Some(kind));
    let output = root.join("output");
    let _ = std::fs::remove_dir_all(&output);
    let mut config = RunConfig::new(3, "real");
    config.nsmp = Some(3);
    config.output_dir = Some(output.clone());
    config.seed = Some(1);
    let result = mvmc_core::run_para_opt_from_namelist_observed(
        root.join("inputs/namelist.def"),
        config,
        &SingleProcessReducer,
    );
    set_sampler_backend_override(None);
    let (_, state, rng) = result.unwrap_or_else(|e| panic!("{kind:?}: {e}"));
    OptOutcome {
        files: read_output_files(&output),
        rng: rng.state_snapshot(),
        consumed: rng.words_consumed(),
        ele_idx: state.electron_config.ele_idx.clone(),
        counter: state.electron_config.counter.to_vec(),
    }
}

#[test]
#[ignore = "optional CUDA gate: needs a device (MVMC_RS_CUDA_GATE=1 to require it)"]
fn production_optimization_cuda_equals_c_order() {
    if !gate() {
        return;
    }
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let root = std::env::temp_dir().join(format!("mvmc-issue517-gpu-opt-{}", std::process::id()));
    stage_hubbard_inputs(
        &root,
        &[
            ("NVMCCalMode", "0"),
            ("NSROptItrStep", "3"),
            ("NSROptItrSmp", "3"),
            ("NVMCSample", "20"),
            ("NVMCWarmUp", "10"),
            ("NDataQtySmp", "1"),
            ("NLanczosMode", "0"),
            ("NStore", "1"),
        ],
    );
    let reference = run_opt(SamplerBackendKind::COrder, &root);
    let routed = run_opt(SamplerBackendKind::Cuda(0), &root);
    let _ = std::fs::remove_dir_all(&root);

    assert_eq!(routed.rng, reference.rng, "SFMT state");
    assert_eq!(routed.consumed, reference.consumed, "draw count");
    assert_eq!(routed.ele_idx, reference.ele_idx, "saved configurations");
    assert_eq!(routed.counter, reference.counter, "counters");
    assert_eq!(
        reference.files.keys().collect::<Vec<_>>(),
        routed.files.keys().collect::<Vec<_>>()
    );
    for (name, text) in &reference.files {
        assert_eq!(text, &routed.files[name], "output {name}");
    }
}

fn run_physcal(kind: SamplerBackendKind, root: &Path) -> (PhysCalResult, BTreeMap<String, String>) {
    set_sampler_backend_override(Some(kind));
    let mut preparation =
        mvmc_core::prepare_phys_cal_without_parameter_file_with_reducer_and_opt_trans(
            root.join("inputs/namelist.def"),
            "real",
            Some(1),
            &SingleProcessReducer,
            false,
        )
        .unwrap();
    preparation.data.modpara.n_data_qty_smp = 2;
    let output = root.join(format!("output-{kind:?}"));
    let _ = std::fs::remove_dir_all(&output);
    let result =
        mvmc_core::vmc_phys_cal_with_reducer(preparation, Some(&output), &SingleProcessReducer);
    set_sampler_backend_override(None);
    let result = result.unwrap_or_else(|e| panic!("{kind:?}: {e}"));
    (result, read_output_files(&output))
}

#[test]
#[ignore = "optional CUDA gate: needs a device (MVMC_RS_CUDA_GATE=1 to require it)"]
fn production_physcal_cuda_equals_c_order() {
    if !gate() {
        return;
    }
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let root =
        std::env::temp_dir().join(format!("mvmc-issue517-gpu-physcal-{}", std::process::id()));
    stage_hubbard_inputs(
        &root,
        &[
            ("NVMCCalMode", "1"),
            ("NDataQtySmp", "2"),
            ("NLanczosMode", "0"),
            ("NVMCSample", "20"),
            ("NVMCWarmUp", "10"),
        ],
    );
    let (reference, reference_files) = run_physcal(SamplerBackendKind::COrder, &root);
    let (routed, routed_files) = run_physcal(SamplerBackendKind::Cuda(0), &root);
    let _ = std::fs::remove_dir_all(&root);

    assert_eq!(
        routed.final_rng.state_snapshot(),
        reference.final_rng.state_snapshot(),
        "SFMT state"
    );
    assert_eq!(
        routed.final_rng.words_consumed(),
        reference.final_rng.words_consumed(),
        "draw count"
    );
    assert_eq!(
        routed.state.electron_config.ele_idx,
        reference.state.electron_config.ele_idx
    );
    assert_eq!(
        routed.state.electron_config.counter,
        reference.state.electron_config.counter
    );
    assert_eq!(routed.state.energy.etot, reference.state.energy.etot);
    assert_eq!(routed_files, reference_files, "output files");
}

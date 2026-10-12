//! Issue #517 milestone 1: the production sampler stage seam.
//!
//! A host model of the device-resident lock-step sampler ([`HostService`] behind the device
//! protocol of issue #434) is registered as the non-C-order kind and selected by override. The
//! production optimization and PhysCal runners must then equal the C-order run exactly: the same
//! SFMT state and draw count, the same saved configurations and counters, and byte-identical
//! output files. This runs in normal CI without a device; the CUDA service's own equivalent gate
//! is `gpu/mvmc-gpu-cuda/tests/sampler_gate.rs`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use mvmc_core::device_sampler::{run_lockstep_real_one, HostService, LockstepOptions};
use mvmc_core::reducer::{Reducer, SingleProcessReducer};
use mvmc_core::sampler_backend::{
    check_serial_reducer, check_supported_mode, open_sampler_backend, parse_sampler_backend,
    register_sampler_backend, set_sampler_backend_override, SamplerBackendKind, SamplerStages,
    SamplerTiming, SAMPLER_BACKEND_VARIABLE,
};
use mvmc_core::sampling::driver::SampleStats;
use mvmc_core::stage_backend::StageError;
use mvmc_core::state::VmcOptimizationState;
use mvmc_core::{PhysCalResult, RunConfig};
use mvmc_expert_parsers::ExpertModeData;
use num_complex::Complex64;
use sfmt19937::Sfmt19937Rng;

static SERIAL: Mutex<()> = Mutex::new(());
static ROUTED_SAMPLES: AtomicUsize = AtomicUsize::new(0);
static RUN_STARTS: AtomicUsize = AtomicUsize::new(0);

/// Host model of the CUDA sampler stage: the validated lock-step protocol on [`HostService`].
struct HostSampler {
    service: HostService,
}

impl SamplerStages for HostSampler {
    fn begin_run(&mut self) {
        RUN_STARTS.fetch_add(1, Ordering::SeqCst);
    }

    fn label(&self) -> String {
        "host-model".to_string()
    }

    fn provider(&self) -> String {
        "HostService behind the device protocol".to_string()
    }

    fn sample_real(
        &mut self,
        data: &ExpertModeData,
        state: &mut VmcOptimizationState,
        rng: &mut Sfmt19937Rng,
        _timing: &mut dyn SamplerTiming,
        reducer: &dyn Reducer,
    ) -> Result<SampleStats, StageError> {
        ROUTED_SAMPLES.fetch_add(1, Ordering::SeqCst);
        let (run, _stats) = run_lockstep_real_one(
            data,
            state,
            rng,
            reducer,
            &mut self.service,
            LockstepOptions::default(),
        )
        .map_err(StageError::Failed)?;
        run.stats.map_err(StageError::Failed)
    }
}

fn install() {
    // Opened by `shared` on first use; the kind needs no provider because it is pre-registered.
    let _ = register_sampler_backend(
        SamplerBackendKind::Cuda(0),
        Box::new(HostSampler {
            service: HostService::new(),
        }),
    );
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

/// Copy the inputs of `fixture` to `root/inputs` and apply the `overrides` to modpara.def.
fn stage_inputs(fixture: &str, root: &Path, overrides: &[(&str, &str)]) -> PathBuf {
    let source = repo_root().join("tests/fixtures/physcal_181").join(fixture);
    stage_inputs_from(&source, root, overrides)
}

/// Copy `source` (a directory with the input files, or its `inputs/` subtree) to
/// `root/inputs` and apply the `overrides` to modpara.def.
fn stage_inputs_from(source: &Path, root: &Path, overrides: &[(&str, &str)]) -> PathBuf {
    let source = if source.join("inputs").is_dir() {
        source.join("inputs")
    } else {
        source.to_path_buf()
    };
    let inputs = root.join("inputs");
    let _ = std::fs::remove_dir_all(root);
    std::fs::create_dir_all(&inputs).unwrap();
    for entry in std::fs::read_dir(&source).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), inputs.join(entry.file_name())).unwrap();
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
    inputs.join("namelist.def")
}

/// Compare output file maps with a short failure message (never dump whole files).
fn assert_output_files_equal(
    label: &str,
    reference: &BTreeMap<String, String>,
    routed: &BTreeMap<String, String>,
) {
    assert_eq!(
        reference.keys().collect::<Vec<_>>(),
        routed.keys().collect::<Vec<_>>(),
        "{label}: output file set"
    );
    for (name, text) in reference {
        assert_eq!(text, &routed[name], "{label}: output {name}");
    }
}

struct OptOutcome {
    files: BTreeMap<String, String>,
    rng: ([u32; 624], usize),
    consumed: u128,
    ele_idx: Vec<i64>,
    ele_cfg: Vec<i64>,
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
        ele_cfg: state.electron_config.ele_cfg.clone(),
        counter: state.electron_config.counter.to_vec(),
    }
}

#[test]
fn parse_accepts_the_documented_selectors() {
    assert_eq!(
        parse_sampler_backend("").unwrap(),
        SamplerBackendKind::COrder
    );
    assert_eq!(
        parse_sampler_backend("c-order").unwrap(),
        SamplerBackendKind::COrder
    );
    assert_eq!(
        parse_sampler_backend(" CUDA ").unwrap(),
        SamplerBackendKind::Cuda(0)
    );
    assert_eq!(
        parse_sampler_backend("cuda:3").unwrap(),
        SamplerBackendKind::Cuda(3)
    );
    assert!(parse_sampler_backend("rocm").is_err());
    assert!(parse_sampler_backend("cuda:x").is_err());
}

#[test]
fn complex_and_fsz_are_hard_unsupported_for_a_device_sampler() {
    let cuda = SamplerBackendKind::Cuda(0);
    assert!(check_supported_mode(cuda, false, false).is_ok());
    for (all_complex, use_fsz) in [(true, false), (false, true), (true, true)] {
        let error = check_supported_mode(cuda, all_complex, use_fsz).unwrap_err();
        assert!(error.contains(SAMPLER_BACKEND_VARIABLE), "{error}");
    }
    assert!(check_supported_mode(SamplerBackendKind::COrder, true, true).is_ok());
}

struct TwoRankReducer;

impl Reducer for TwoRankReducer {
    fn allreduce_sum_f64(&self, _buf: &mut [f64]) {}
    fn allreduce_sum_c64(&self, _buf: &mut [Complex64]) {}
    fn allreduce_sum_i64(&self, _buf: &mut [i64]) {}
    fn broadcast_i64(&self, _root: usize, _buf: &mut [i64]) -> Result<(), String> {
        Ok(())
    }
    fn world_size(&self) -> usize {
        2
    }
    fn rank(&self) -> usize {
        0
    }
}

#[test]
fn multi_process_reducers_are_hard_unsupported() {
    assert!(check_serial_reducer(&SingleProcessReducer).is_ok());
    let error = check_serial_reducer(&TwoRankReducer).unwrap_err();
    assert!(matches!(error, StageError::Unsupported(_)), "{error:?}");
}

#[test]
fn cuda_kind_without_a_provider_is_unsupported_not_a_fallback() {
    let error = open_sampler_backend(SamplerBackendKind::Cuda(0))
        .err()
        .expect("no CUDA provider in this build");
    assert!(matches!(error, StageError::Unsupported(_)), "{error:?}");
}

/// The production optimization loop: the host-model device sampler must reproduce the C-order
/// run exactly (RNG, saved chain, counters, output files).
#[test]
fn production_optimization_host_model_equals_c_order() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    install();
    let root = std::env::temp_dir().join(format!("mvmc-issue517-opt-{}", std::process::id()));
    stage_inputs(
        "hubbard_chain_dh_real",
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
    let before = ROUTED_SAMPLES.load(Ordering::SeqCst);
    let starts_before = RUN_STARTS.load(Ordering::SeqCst);
    let routed = run_opt(SamplerBackendKind::Cuda(0), &root);
    assert_eq!(RUN_STARTS.load(Ordering::SeqCst), starts_before + 1);
    assert!(
        ROUTED_SAMPLES.load(Ordering::SeqCst) > before,
        "the host-model sampler was not used"
    );
    let _ = std::fs::remove_dir_all(&root);

    assert_eq!(routed.rng, reference.rng, "SFMT state");
    assert_eq!(routed.consumed, reference.consumed, "draw count");
    assert_eq!(routed.ele_idx, reference.ele_idx, "saved configurations");
    assert_eq!(routed.ele_cfg, reference.ele_cfg, "saved electron configs");
    assert_eq!(routed.counter, reference.counter, "counters");
    assert_output_files_equal("opt", &reference.files, &routed.files);
}

/// The production PhysCal loop: same exactness through a random-initialized wavefunction. The
/// Hubbard L=64 chain is chosen because the host inverse factorizations reach the size where
/// OpenBLAS threading changes their last bits: a sampler backend whose setup left a different
/// BLAS thread count would break the byte identity (the pin added to the PhysCal entry point
/// by issue #517).
#[test]
fn production_physcal_host_model_equals_c_order() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    install();
    let benchmark = repo_root().join("benchmark/hubbard_chain/inputs/hubbard_chain_L64");
    assert!(benchmark.is_dir(), "Hubbard L=64 benchmark input required");
    let root = std::env::temp_dir().join(format!("mvmc-issue517-physcal-{}", std::process::id()));
    let namelist = stage_inputs_from(
        &benchmark,
        &root,
        &[
            ("NVMCCalMode", "1"),
            ("NDataQtySmp", "1"),
            ("NVMCSample", "20"),
            ("NVMCWarmUp", "10"),
            ("NLanczosMode", "0"),
        ],
    );

    let run = |kind: SamplerBackendKind| -> (PhysCalResult, BTreeMap<String, String>) {
        set_sampler_backend_override(Some(kind));
        let mut preparation =
            mvmc_core::prepare_phys_cal_without_parameter_file_with_reducer_and_opt_trans(
                &namelist,
                "real",
                Some(1),
                &SingleProcessReducer,
                false,
            )
            .unwrap();
        preparation.data.modpara.n_data_qty_smp = 1;
        let output = root.join(format!("output-{kind:?}"));
        let _ = std::fs::remove_dir_all(&output);
        let result =
            mvmc_core::vmc_phys_cal_with_reducer(preparation, Some(&output), &SingleProcessReducer);
        set_sampler_backend_override(None);
        let result = result.unwrap_or_else(|e| panic!("{kind:?}: {e}"));
        (result, read_output_files(&output))
    };

    let (reference, reference_files) = run(SamplerBackendKind::COrder);
    let before = ROUTED_SAMPLES.load(Ordering::SeqCst);
    let (routed, routed_files) = run(SamplerBackendKind::Cuda(0));
    assert!(
        ROUTED_SAMPLES.load(Ordering::SeqCst) > before,
        "the host-model sampler was not used"
    );
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
        routed.state.electron_config.ele_cfg,
        reference.state.electron_config.ele_cfg
    );
    assert_eq!(
        routed.state.electron_config.ele_num,
        reference.state.electron_config.ele_num
    );
    assert_eq!(
        routed.state.electron_config.ele_proj_cnt,
        reference.state.electron_config.ele_proj_cnt
    );
    assert_eq!(
        routed.state.electron_config.counter,
        reference.state.electron_config.counter
    );
    assert_eq!(
        routed.state.energy.etot, reference.state.energy.etot,
        "final energy"
    );
    assert_output_files_equal("physcal", &reference_files, &routed_files);
}

/// A multi-process reducer selected onto the CUDA sampler must fail in validation, before any
/// sampling: the device chain is the serial production chain.
#[test]
fn multi_process_reducer_rejects_the_cuda_sampler_before_sampling() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    install();
    let root = std::env::temp_dir().join(format!("mvmc-issue517-mpi-{}", std::process::id()));
    let namelist = stage_inputs(
        "hubbard_chain_dh_real",
        &root,
        &[
            ("NVMCCalMode", "0"),
            ("NSROptItrStep", "1"),
            ("NSROptItrSmp", "1"),
            ("NVMCSample", "5"),
        ],
    );
    set_sampler_backend_override(Some(SamplerBackendKind::Cuda(0)));
    let mut config = RunConfig::new(1, "real");
    config.nsmp = Some(1);
    let result =
        mvmc_core::run_para_opt_from_namelist_with_reducer(namelist, config, &TwoRankReducer);
    set_sampler_backend_override(None);
    let _ = std::fs::remove_dir_all(&root);
    let error = result.unwrap_err();
    assert!(
        error.contains(SAMPLER_BACKEND_VARIABLE) || error.contains("single-process"),
        "{error}"
    );
}

/// Complex and FSZ inputs selected onto the CUDA sampler must fail before sampling; there is no
/// silent C-order fallback.
#[test]
fn complex_and_fsz_optimization_reject_the_cuda_sampler() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    install();
    for (fixture, mode) in [
        ("heisenberg_chain_cmp", "cmp"),
        ("heisenberg_chain_fsz", "fsz"),
    ] {
        let root =
            std::env::temp_dir().join(format!("mvmc-issue517-{fixture}-{}", std::process::id()));
        let namelist = stage_inputs(
            fixture,
            &root,
            &[
                ("NVMCCalMode", "0"),
                ("NSROptItrStep", "1"),
                ("NSROptItrSmp", "1"),
                ("NVMCSample", "5"),
            ],
        );
        set_sampler_backend_override(Some(SamplerBackendKind::Cuda(0)));
        let mut config = RunConfig::new(1, mode);
        config.nsmp = Some(1);
        config.output_dir = None;
        let result = mvmc_core::run_para_opt_from_namelist(namelist, config);
        set_sampler_backend_override(None);
        let _ = std::fs::remove_dir_all(&root);
        let error = result.unwrap_err();
        assert!(
            error.contains(SAMPLER_BACKEND_VARIABLE),
            "{fixture}: {error}"
        );
    }
}

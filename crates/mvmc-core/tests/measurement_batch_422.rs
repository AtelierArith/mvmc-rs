//! Issue #422: the sample-batched measurement pipeline is a pure reordering.
//!
//! Every output file (`zvo_*`, `zqp_*`, SRinfo, Green/Lanczos files; everything except
//! wall-clock content) must be byte-identical for batch sizes 1 (the former serial order),
//! 3, 7 and "all samples", for real, complex, FSZ, Doublon-Holon, RBM, OptTrans and PhysCal
//! with Green and Lanczos. Batch size 1 is itself checked against `main` fixtures by the
//! existing native-C and Julia fixture tests; this file checks batch-size invariance.
//!
//! The batch size is a process-wide override, so the sweep runs sequentially in one test.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use mvmc_core::measurement_batch::set_measurement_batch_size_override;
use mvmc_core::RunConfig;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn copy_inputs(fixture: &str, tag: &str) -> (PathBuf, PathBuf, PathBuf) {
    let source = repo_root().join("tests/fixtures/physcal_181").join(fixture);
    let root = std::env::temp_dir().join(format!(
        "mvmc-issue422-{}-{fixture}-{tag}",
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
    (root, inputs, source)
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

/// All non-wall-clock output files with their bytes.
fn snapshot(output: &Path) -> BTreeMap<String, Vec<u8>> {
    let mut files = BTreeMap::new();
    for entry in std::fs::read_dir(output).unwrap() {
        let name = entry.unwrap().file_name().to_string_lossy().into_owned();
        if name.contains("Timer") || name.starts_with("zvo_time") {
            continue;
        }
        files.insert(name.clone(), std::fs::read(output.join(&name)).unwrap());
    }
    assert!(!files.is_empty());
    files
}

fn optimization(
    fixture: &str,
    mode: &str,
    opt_trans: bool,
    batch: usize,
) -> BTreeMap<String, Vec<u8>> {
    set_measurement_batch_size_override(Some(batch));
    let (root, inputs, _) = copy_inputs(fixture, &format!("opt{batch}"));
    rewrite_modpara(
        &inputs,
        &[
            ("NVMCCalMode", "0"),
            ("NSROptItrStep", "3"),
            ("NSROptItrSmp", "3"),
            ("NVMCSample", "40"),
            ("NVMCWarmUp", "10"),
            ("NSRCG", "0"),
            ("NStore", "1"),
            ("NDataIdxStart", "1"),
            ("NDataQtySmp", "1"),
            ("NLanczosMode", "0"),
        ],
    );
    let output = root.join("output");
    let mut config = RunConfig::new(3, mode);
    config.output_dir = Some(output.clone());
    config.seed = Some(1);
    config.enable_opt_trans = Some(opt_trans);
    mvmc_core::run_para_opt_from_namelist(inputs.join("namelist.def"), config)
        .unwrap_or_else(|e| panic!("{fixture} batch={batch}: {e}"));
    let files = snapshot(&output);
    let _ = std::fs::remove_dir_all(&root);
    files
}

fn physcal(fixture: &str, mode: &str, batch: usize) -> BTreeMap<String, Vec<u8>> {
    set_measurement_batch_size_override(Some(batch));
    let (root, inputs, _) = copy_inputs(fixture, &format!("phys{batch}"));
    let preparation = mvmc_core::prepare_phys_cal_from_namelist(
        inputs.join("namelist.def"),
        root.join("zqp_opt.dat"),
        mode,
        Some(1),
    )
    .unwrap_or_else(|e| panic!("{fixture} batch={batch}: {e}"));
    let output = root.join("output");
    mvmc_core::vmc_phys_cal_to_dir(preparation, &output)
        .unwrap_or_else(|e| panic!("{fixture} batch={batch}: {e}"));
    let files = snapshot(&output);
    let _ = std::fs::remove_dir_all(&root);
    files
}

fn assert_batch_invariant(label: &str, run: impl Fn(usize) -> BTreeMap<String, Vec<u8>>) {
    // 1 is the former serial order; 100_000 is clamped to "all samples in one batch".
    let reference = run(1);
    for batch in [3usize, 7, 100_000] {
        let other = run(batch);
        assert_eq!(
            reference.keys().collect::<Vec<_>>(),
            other.keys().collect::<Vec<_>>(),
            "{label}: file set differs for batch {batch}"
        );
        for (name, bytes) in &reference {
            assert!(
                bytes == &other[name],
                "{label}: {name} differs between batch 1 and batch {batch}"
            );
        }
    }
}

#[test]
fn batched_measurement_is_byte_identical_for_all_modes() {
    let cases: [(&str, &str, bool); 5] = [
        ("hubbard_chain_real", "real", false),
        ("hubbard_chain_dh_real", "real", false),
        ("heisenberg_chain_cmp", "cmp", false),
        ("heisenberg_chain_fsz", "fsz", false),
        ("hubbard_chain_dh_rbm_opttrans", "cmp", true),
    ];
    for (fixture, mode, opt_trans) in cases {
        assert_batch_invariant(&format!("opt:{fixture}"), |b| {
            optimization(fixture, mode, opt_trans, b)
        });
    }
    // PhysCal: Green functions (all), Lanczos (mode 2), complex and FSZ.
    for (fixture, mode) in [
        ("hubbard_all_terms_lanczos1", "real"),
        ("hubbard_all_terms_lanczos2", "real"),
        ("hubbard_chain_real", "real"),
        ("heisenberg_chain_cmp", "cmp"),
        ("heisenberg_chain_fsz", "fsz"),
    ] {
        assert_batch_invariant(&format!("physcal:{fixture}"), |b| physcal(fixture, mode, b));
    }
    set_measurement_batch_size_override(None);
}

// ---- stage A through the unified stage backend (issues #422, #437) --------------------------

use mvmc_core::measurement_batch::{set_measurement_backend_override, MeasurementPfaffian};
use mvmc_core::stage_backend::StageBackendKind;

fn assert_same_bytes(
    label: &str,
    reference: &BTreeMap<String, Vec<u8>>,
    other: &BTreeMap<String, Vec<u8>>,
) {
    assert_eq!(
        reference.keys().collect::<Vec<_>>(),
        other.keys().collect::<Vec<_>>(),
        "{label}: file set differs"
    );
    for (name, bytes) in reference {
        assert!(bytes == &other[name], "{label}: {name} differs");
    }
}

/// The C-order `PfaffianStages` routed through one batched call per chunk is the same PfaPack
/// sequence as `calc_m_all_real` with the same plane assembly and the `invM = -X^-1` flip, so
/// every output file is byte-identical to the default path for every batch size (real,
/// non-FSZ modes: optimization with Doublon-Holon/Gutzwiller/Jastrow and PhysCal with Green
/// and Lanczos).
#[test]
fn stage_backend_c_order_measurement_is_byte_identical_to_calc_m_all() {
    for batch in [1usize, 3, 100_000] {
        for (fixture, opt_trans) in [
            ("hubbard_chain_real", false),
            ("hubbard_chain_dh_real", false),
        ] {
            set_measurement_backend_override(None);
            let reference = optimization(fixture, "real", opt_trans, batch);
            set_measurement_backend_override(Some(MeasurementPfaffian::Stage(
                StageBackendKind::COrder,
            )));
            let staged = optimization(fixture, "real", opt_trans, batch);
            assert_same_bytes(&format!("opt:{fixture} batch={batch}"), &reference, &staged);
        }
        for fixture in ["hubbard_all_terms_lanczos2", "hubbard_chain_real"] {
            set_measurement_backend_override(None);
            let reference = physcal(fixture, "real", batch);
            set_measurement_backend_override(Some(MeasurementPfaffian::Stage(
                StageBackendKind::COrder,
            )));
            let staged = physcal(fixture, "real", batch);
            assert_same_bytes(
                &format!("physcal:{fixture} batch={batch}"),
                &reference,
                &staged,
            );
        }
    }
    set_measurement_backend_override(None);
    set_measurement_batch_size_override(None);
}

/// A backend without the stage (tenferro 0.7.1 has no Pfaffian) is a hard error, never a
/// fallback to the C-order kernels.
#[test]
#[should_panic(expected = "measurement Pfaffian stage failed")]
fn stage_backend_without_pfaffian_is_a_hard_error() {
    set_measurement_backend_override(Some(MeasurementPfaffian::Stage(
        StageBackendKind::TenferroCpu,
    )));
    let _ = physcal("hubbard_chain_real", "real", 3);
}

/// Complex mode has no stage-backend Pfaffian (the stage trait is real `f64`): hard error.
#[test]
#[should_panic(expected = "supports only real, non-FSZ mode")]
fn stage_backend_in_complex_mode_is_a_hard_error() {
    set_measurement_backend_override(Some(MeasurementPfaffian::Stage(StageBackendKind::COrder)));
    let _ = physcal("heisenberg_chain_cmp", "cmp", 3);
}

//! Independent C block/flag literals for manual A005; no runtime/oracle invocation.

use mvmc_expert_parsers::{
    refresh_rbm_optimization_flags as refresh, ChargeRBMHiddenLayerTerm, ChargeRBMPhysHiddenTerm,
    ChargeRBMPhysLayerTerm, ExpertModeData, GeneralRBMHiddenLayerTerm, GeneralRBMPhysHiddenTerm,
    GeneralRBMPhysLayerTerm, ModParaParameters, RbmDefinitionKey as Key,
    RbmOptimizationSource as Source, SpinRBMHiddenLayerTerm, SpinRBMPhysHiddenTerm,
    SpinRBMPhysLayerTerm,
};
use num_complex::Complex64;
use std::{
    fs, io,
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};

const KEYS: [Key; 9] = [
    Key::ChargePhysLayer,
    Key::SpinPhysLayer,
    Key::GeneralPhysLayer,
    Key::ChargeHiddenLayer,
    Key::SpinHiddenLayer,
    Key::GeneralHiddenLayer,
    Key::ChargePhysHidden,
    Key::SpinPhysHidden,
    Key::GeneralPhysHidden,
];
// Complete geometry for Nsite=2 and each hidden dimension=2. Coefficient1
// has no coordinate assignment: declared reserved slots still receive flags.
const MAPPINGS: [&str; 9] = [
    "0 0\n1 0\n",
    "0 0\n1 0\n",
    "0 0 0\n1 0 0\n0 1 0\n1 1 0\n",
    "0 0\n1 0\n",
    "0 0\n1 0\n",
    "0 0\n1 0\n",
    "0 0 0\n0 1 0\n1 0 0\n1 1 0\n",
    "0 0 0\n0 1 0\n1 0 0\n1 1 0\n",
    "0 0 0 0\n0 0 1 0\n1 0 0 0\n1 0 1 0\n\
     0 1 0 0\n0 1 1 0\n1 1 0 0\n1 1 1 0\n",
];
static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Files(PathBuf);
impl Files {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "mvmc-a005-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn write(&self, section: usize, header: i32) -> PathBuf {
        let path = self.0.join(format!("{section}.def"));
        fs::write(
            &path,
            format!(
                "===\nNParameter 2\nComplex {header}\n===\n===\n{}99 2\n-4 -3\n",
                MAPPINGS[section]
            ),
        )
        .unwrap();
        path
    }
}
impl Drop for Files {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn data() -> ExpertModeData {
    ExpertModeData {
        modpara: ModParaParameters {
            nsite: 2,
            nneuron_charge: 2,
            nneuron_spin: 2,
            nneuron_general: 2,
            n_orbital_idx: 2,
            ..Default::default()
        },
        n_gutzwiller_idx: 1,
        n_jastrow_idx: 2,
        rbm_section_widths: [2; 9],
        rbm_params: vec![Complex64::new(-0.25, 0.5); 18],
        slater_params: vec![Complex64::new(2.0, -3.0); 2],
        charge_rbm_phys_layer_terms: vec![ChargeRBMPhysLayerTerm {
            site: 0,
            idx: 0,
            value: Complex64::new(-0.25, 0.5),
            is_complex: true,
        }],
        spin_rbm_phys_layer_terms: vec![SpinRBMPhysLayerTerm {
            site: 0,
            idx: 0,
            value: Complex64::new(-0.25, 0.5),
            is_complex: false,
        }],
        general_rbm_phys_layer_terms: vec![GeneralRBMPhysLayerTerm {
            site: 0,
            spin: 1,
            idx: 0,
            value: Complex64::new(-0.25, 0.5),
            is_complex: false,
        }],
        charge_rbm_hidden_layer_terms: vec![ChargeRBMHiddenLayerTerm {
            site: 0,
            idx: 0,
            value: Complex64::new(-0.25, 0.5),
            is_complex: true,
        }],
        spin_rbm_hidden_layer_terms: vec![SpinRBMHiddenLayerTerm {
            site: 0,
            idx: 0,
            value: Complex64::new(-0.25, 0.5),
            is_complex: false,
        }],
        general_rbm_hidden_layer_terms: vec![GeneralRBMHiddenLayerTerm {
            site: 0,
            idx: 0,
            value: Complex64::new(-0.25, 0.5),
            is_complex: false,
        }],
        charge_rbm_phys_hidden_terms: vec![ChargeRBMPhysHiddenTerm {
            site1: 0,
            site2: 1,
            idx: 0,
            value: Complex64::new(-0.25, 0.5),
            is_complex: true,
        }],
        spin_rbm_phys_hidden_terms: vec![SpinRBMPhysHiddenTerm {
            site1: 0,
            site2: 1,
            idx: 0,
            value: Complex64::new(-0.25, 0.5),
            is_complex: false,
        }],
        general_rbm_phys_hidden_terms: vec![GeneralRBMPhysHiddenTerm {
            site1: 0,
            spin: 1,
            site2: 1,
            idx: 0,
            value: Complex64::new(-0.25, 0.5),
            is_complex: false,
        }],
        optimization_flags: vec![71; 46], // prefix6 + RBM36 + Slater4
        input_errors: vec!["unrelated retained diagnostic".into()],
        ..Default::default()
    }
}
fn source(key: Key, path: &Path) -> Source<'_> {
    Source { key, path }
}
fn only_flags(before: &ExpertModeData, after: &ExpertModeData) {
    let mut expected = before.clone();
    expected
        .optimization_flags
        .clone_from(&after.optimization_flags);
    assert_eq!(format!("{after:?}"), format!("{expected:?}"));
    assert_eq!(
        after.optimization_flags.len(),
        before.optimization_flags.len()
    );
}
fn reject(
    actual: &mut ExpertModeData,
    widths: &[usize; 9],
    sources: &[Source<'_>],
    kind: io::ErrorKind,
) {
    let before = format!("{actual:?}");
    assert_eq!(refresh(actual, widths, sources).unwrap_err().kind(), kind);
    assert_eq!(format!("{actual:?}"), before);
}

#[test]
fn all_nine_blocks_have_literal_offsets_signed_local_headers_and_reserved_flags() {
    let files = Files::new();
    let paths: Vec<_> = [1, 0, -1, 2, -2, 0, 1, 0, -1]
        .into_iter()
        .enumerate()
        .map(|(section, header)| files.write(section, header))
        .collect();
    let sources: Vec<_> = KEYS
        .into_iter()
        .zip(&paths)
        .map(|(key, path)| source(key, path))
        .collect();
    let mut actual = data();
    let before = actual.clone();
    refresh(&mut actual, &[2; 9], &sources).unwrap();
    assert_eq!(
        &actual.optimization_flags[6..42],
        &[
            2, 2, -3, -3, 2, 0, -3, 0, 2, 0, -3, 0, 2, 2, -3, -3, 2, 0, -3, 0, 2, 0, -3, 0, 2, 2,
            -3, -3, 2, 0, -3, 0, 2, 0, -3, 0,
        ]
    );
    assert_eq!(&actual.optimization_flags[..6], &[71; 6]);
    assert_eq!(&actual.optimization_flags[42..], &[71; 4]);
    only_flags(&before, &actual);
}

#[test]
fn selected_last_block_uses_all_unselected_widths_without_opening_their_files() {
    let files = Files::new();
    let path = files.write(8, 1);
    let mut actual = data();
    let before = actual.clone();
    refresh(
        &mut actual,
        &[2; 9],
        &[source(Key::GeneralPhysHidden, &path)],
    )
    .unwrap();
    assert_eq!(&actual.optimization_flags[..38], &[71; 38]);
    assert_eq!(&actual.optimization_flags[38..42], &[2, 2, -3, -3]);
    assert_eq!(&actual.optimization_flags[42..], &[71; 4]);
    only_flags(&before, &actual);
}

#[test]
fn unbound_explicit_widths_do_not_install_metadata_or_activate_zero_sections() {
    let files = Files::new();
    let path = files.write(0, 1);
    let mut actual = data();
    actual.rbm_section_widths = [0; 9];
    let before = actual.clone();
    refresh(&mut actual, &[2; 9], &[source(Key::ChargePhysLayer, &path)]).unwrap();
    assert_eq!(&actual.optimization_flags[6..10], &[2, 2, -3, -3]);
    only_flags(&before, &actual);
    actual.charge_rbm_phys_layer_terms.clear();
    let before = format!("{actual:?}");
    let mut inactive = [2; 9];
    inactive[0] = 0;
    refresh(
        &mut actual,
        &inactive,
        &[source(Key::ChargePhysLayer, &path)],
    )
    .unwrap();
    assert_eq!(format!("{actual:?}"), before); // parsed but original inactive skip
}

#[test]
fn empty_noop_and_duplicate_selection_fail_before_io_without_mutation() {
    let mut actual = data();
    actual.n_gutzwiller_idx = -1;
    let before = format!("{actual:?}");
    refresh(&mut actual, &[usize::MAX; 9], &[]).unwrap();
    assert_eq!(format!("{actual:?}"), before);
    let path = Path::new("a005-not-opened.def");
    reject(
        &mut actual,
        &[2; 9],
        &[
            source(Key::ChargePhysLayer, path),
            source(Key::ChargePhysLayer, path),
        ],
        io::ErrorKind::InvalidInput,
    );
}

#[test]
fn later_io_or_strict_geometry_failure_rolls_back_earlier_selected_flags() {
    let files = Files::new();
    let valid = files.write(0, 1);
    let missing = files.0.join("missing.def");
    let malformed = files.0.join("incomplete.def");
    fs::write(
        &malformed,
        "===\nNParameter 2\nComplex 1\n===\n===\n0 0\n0 1\n",
    )
    .unwrap();
    let mut actual = data();
    for (path, kind) in [
        (&missing, io::ErrorKind::NotFound),
        (&malformed, io::ErrorKind::InvalidData),
    ] {
        reject(
            &mut actual,
            &[2; 9],
            &[
                source(Key::ChargePhysLayer, &valid),
                source(Key::SpinPhysLayer, path),
            ],
            kind,
        );
    }
}

#[test]
fn loaded_count_mapping_capacity_and_checked_arithmetic_errors_preserve_every_field() {
    let files = Files::new();
    let path = files.write(8, 1);
    let sources = [source(Key::GeneralPhysHidden, &path)];
    let mut actual = data();
    let mut wrong = [2; 9];
    wrong[1] = 3;
    reject(&mut actual, &wrong, &sources, io::ErrorKind::InvalidData);
    actual.charge_rbm_phys_layer_terms[0].idx = 2;
    reject(&mut actual, &[2; 9], &sources, io::ErrorKind::InvalidData);
    actual.charge_rbm_phys_layer_terms[0].idx = 0;
    actual.optimization_flags.truncate(41);
    reject(&mut actual, &[2; 9], &sources, io::ErrorKind::InvalidData);
    actual.optimization_flags = vec![71; 46];
    actual.n_gutzwiller_idx = i64::MAX;
    actual.n_jastrow_idx = i64::MAX;
    reject(&mut actual, &[2; 9], &sources, io::ErrorKind::InvalidData);
    actual.n_gutzwiller_idx = 1;
    actual.n_jastrow_idx = 2;
    actual.rbm_section_widths[8] = 0;
    let mut width = [2; 9];
    width[8] = 3;
    reject(&mut actual, &width, &sources, io::ErrorKind::InvalidData); // file still count2
}

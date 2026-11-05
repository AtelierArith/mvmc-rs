//! Exact Julia parser contract; Hamiltonian support has a separate runtime gate.
#[path = "../../../tests/support/historical_orbital_model.rs"]
mod historical_orbital_model;
#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;
use std::path::{Path, PathBuf};

use historical_orbital_model::historical_interall_model::parse_interall_def;
use historical_orbital_model::historical_kernel_model as parse_expert_mode_files;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/interall")
}

#[test]
fn orchestration_replaces_repeated_sections_and_records_missing_files() {
    let data = parse_expert_mode_files(root().join("namelist.def")).unwrap();
    assert!(data.input_errors.is_empty());
    assert_eq!(
        data.inter_all_terms,
        parse_interall_def(root().join("parser_cases.def")).unwrap()
    );
    let replaced = parse_expert_mode_files(root().join("namelist_replace.def")).unwrap();
    assert!(replaced.input_errors.is_empty());
    assert_eq!(
        replaced.inter_all_terms,
        parse_interall_def(root().join("replace.def")).unwrap()
    );
    let missing = parse_expert_mode_files(root().join("namelist_missing.def")).unwrap();
    assert!(missing.inter_all_terms.is_empty());
    assert_eq!(missing.input_errors.len(), 1);
    assert!(missing.input_errors[0].contains("InterAll file not found"));
}

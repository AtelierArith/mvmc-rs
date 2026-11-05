//! Canonical DH4 definition and combined DH2/DH4 layout boundaries.
#[path = "../../../tests/support/historical_orbital_model.rs"]
mod historical_orbital_model;
#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;
use std::path::{Path, PathBuf};

use historical_orbital_model::historical_kernel_model as parse_expert_mode_files;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/dh4")
}

#[test]
fn dh4_reserves_ten_dense_parameters_per_index_after_dh2_before_orbital_flags() {
    for name in ["orbital_first", "dh_first", "alias"] {
        let data = parse_expert_mode_files(root().join(format!("namelist_{name}.def"))).unwrap();
        assert_eq!(data.projection_layout().n_dh2, 2, "{name}");
        assert_eq!(data.projection_layout().n_dh4, 2, "{name}");
        assert_eq!(data.projection_layout().dh4_offset, 17, "{name}");
        assert_eq!(data.projection_layout().n_proj, 37, "{name}");
        assert_eq!(data.optimization_flags.len(), 82, "{name}");
    }
}

#[test]
fn required_dh4_missing_and_invalid_definitions_fail_during_parsing() {
    for name in ["invalid", "missing", "invalid_alias", "missing_alias"] {
        assert!(
            parse_expert_mode_files(root().join(format!("namelist_{name}.def"))).is_err(),
            "{name}"
        );
    }
}

#[test]
fn c_reads_modpara_before_a_dh4_definition_listed_first() {
    let data = parse_expert_mode_files(root().join("namelist_before_modpara.def")).unwrap();
    assert_eq!(data.modpara.nsite, 3);
    assert_eq!(data.projection_layout().n_dh4, 1);
    assert_eq!(data.doublon_holon_4site_indices.len(), 1);
    assert_eq!(data.doublon_holon_4site_params.len(), 10);
}

#[test]
fn combined_validation_reports_dh4_shapes_and_all_four_neighbors_in_source_order() {
    let mut d = mvmc_expert_parsers::ExpertModeData::new();
    d.modpara.nsite = 3;
    d.doublon_holon_4site_indices = vec![
        mvmc_expert_parsers::DoublonHolon4SiteIndex {
            neighbors: vec![[0; 4]; 2],
        },
        mvmc_expert_parsers::DoublonHolon4SiteIndex {
            neighbors: vec![[-1, 3, 0, 0], [2, i64::MAX, 0, 0], [0; 4]],
        },
        mvmc_expert_parsers::DoublonHolon4SiteIndex {
            neighbors: vec![[0; 4]; 3],
        },
    ];
    let result = mvmc_expert_parsers::validate_expert_mode_data(&d);
    let errors: Vec<_> = result
        .errors
        .iter()
        .filter(|e| e.starts_with("DH4"))
        .cloned()
        .collect();
    assert_eq!(
        errors,
        vec![
            "DH4 index 0: neighbors must be 3 x 4".to_string(),
            "DH4 index 1 site 0 neighbor 0=-1 out of range [0, 2]".to_string(),
            "DH4 index 1 site 0 neighbor 1=3 out of range [0, 2]".to_string(),
            format!(
                "DH4 index 1 site 1 neighbor 1={} out of range [0, 2]",
                i64::MAX
            ),
        ]
    );
    assert!(!result.is_valid);
}

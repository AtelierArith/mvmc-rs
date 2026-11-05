//! Canonical strict DH2 table and final projection-layout contracts.
#[path = "../../../tests/support/historical_orbital_model.rs"]
mod historical_orbital_model;
#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;
use std::path::{Path, PathBuf};

use historical_orbital_model::historical_kernel_model as parse_expert_mode_files;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/dh2")
}

#[test]
fn dh2_reserves_six_dense_parameters_per_index_before_orbital_flags() {
    for name in ["orbital_first", "dh_first", "alias"] {
        let data = parse_expert_mode_files(root().join(format!("namelist_{name}.def"))).unwrap();
        assert_eq!(data.projection_layout().n_proj, 17, "{name}");
        assert_eq!(data.optimization_flags.len(), 42, "{name}");
    }
}

#[test]
fn required_dh2_missing_invalid_and_missing_modpara_definitions_fail_during_parsing() {
    for name in ["invalid", "missing", "before_modpara"] {
        assert!(
            parse_expert_mode_files(root().join(format!("namelist_{name}.def"))).is_err(),
            "{name}"
        );
    }
}

#[test]
fn c_reads_modpara_before_a_dh2_definition_listed_first() {
    let dir = std::env::temp_dir().join(format!("mvmc-dh2-order-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("namelist.def"),
        format!(
            "DH2 {}\nModPara {}\n",
            root().join("complex.def").display(),
            root().join("modpara.def").display()
        ),
    )
    .unwrap();
    let data = parse_expert_mode_files(dir.join("namelist.def")).unwrap();
    assert_eq!(data.modpara.nsite, 3);
    assert_eq!(data.projection_layout().n_dh2, 1);
    assert_eq!(data.doublon_holon_2site_indices.len(), 1);
    assert_eq!(data.doublon_holon_2site_params.len(), 6);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn combined_validation_reports_dh_table_shapes_and_neighbors_in_source_order() {
    let mut d = mvmc_expert_parsers::ExpertModeData::new();
    d.modpara.nsite = 3;
    d.doublon_holon_2site_indices = vec![
        mvmc_expert_parsers::DoublonHolon2SiteIndex {
            neighbors: vec![[0, 0]; 2],
        },
        mvmc_expert_parsers::DoublonHolon2SiteIndex {
            neighbors: vec![[-1, 3], [2, i64::MAX], [0, 0]],
        },
        mvmc_expert_parsers::DoublonHolon2SiteIndex {
            neighbors: vec![[0, 0]; 3],
        },
    ];
    let result = mvmc_expert_parsers::validate_expert_mode_data(&d);
    let dh_errors: Vec<_> = result
        .errors
        .iter()
        .filter(|e| e.starts_with("DH2"))
        .cloned()
        .collect();
    assert_eq!(
        dh_errors,
        vec![
            "DH2 index 0: neighbors must be 3 x 2".to_string(),
            "DH2 index 1 site 0 neighbor 0=-1 out of range [0, 2]".to_string(),
            "DH2 index 1 site 0 neighbor 1=3 out of range [0, 2]".to_string(),
            format!(
                "DH2 index 1 site 1 neighbor 1={} out of range [0, 2]",
                i64::MAX
            ),
        ]
    );
    assert!(!result.is_valid);
}

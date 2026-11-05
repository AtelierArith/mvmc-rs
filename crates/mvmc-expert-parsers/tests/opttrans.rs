mod common;
#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;
use common::historical_kernel_model as parse_expert_mode_files;
use mvmc_expert_parsers::{parse_expert_mode_files_with_opt_trans, ExpertModeData};
use num_complex::Complex64;
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/opttrans")
}

#[test]
fn opttrans_sector_count_and_real_only_component_flags_follow_julia() {
    let data = parse_expert_mode_files(root().join("namelist_valid.def")).unwrap();
    assert_eq!(data.n_qp_opt_trans, 2);
    assert_eq!(data.optimization_flags, vec![1, 0, 1, 0]);
}

#[test]
fn c_disabled_opttrans_ignores_definition_and_keeps_default_layout() {
    let data =
        parse_expert_mode_files_with_opt_trans(root().join("namelist_valid.def"), false).unwrap();
    assert_eq!(data.n_qp_opt_trans, 1);
    assert!(data.para_qp_opt_trans.is_empty());
    assert!(data.opt_trans.is_empty());
    assert!(data.qp_opt_trans.is_empty());
    assert!(data.qp_opt_trans_sgn.is_empty());
    assert!(data.optimization_flags.is_empty());
}

// C `GetInfoOptTrans` (1.3.0 and 1.4.0) (`readdef.c:2799-2816`) writes undoubled flags at
// `NProj + NSlater + i`, freezing the weights and activating unrelated
// projection/Slater components. That is a C defect (#676,
// tmisawa/Julia-mVMC#94); Rust activates only the real weight components.
#[test]
fn enabled_opttrans_activates_only_real_weight_components() {
    let data =
        parse_expert_mode_files_with_opt_trans(root().join("namelist_valid.def"), true).unwrap();
    assert_eq!(data.optimization_flags, vec![1, 0, 1, 0]);
}

#[test]
fn opttrans_flags_follow_projection_rbm_and_slater_blocks() {
    let mut data = ExpertModeData::new();
    data.n_gutzwiller_idx = 3;
    data.modpara.n_orbital_idx = 4;
    data.opt_trans = vec![Complex64::new(0.5, 0.0), Complex64::new(0.75, 0.0)];
    data.optimization_flags = vec![0; 14];
    mvmc_expert_parsers::set_opt_trans_opt_flags(&mut data);
    // Projection and Slater components keep their flags; C's stray writes at
    // components 7 and 8 are not reproduced.
    assert_eq!(&data.optimization_flags[..14], &[0; 14]);
    assert_eq!(&data.optimization_flags[14..], &[1, 0, 1, 0]);
}

mod common;
#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;
use common::historical_kernel_model as parse_expert_mode_files;
use std::path::PathBuf;

#[test]
fn all_nine_rbm_sections_keep_raw_flags_at_historical_mapped_widths() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/rbm");
    for name in [
        "ChargeRBM_PhysLayer",
        "SpinRBM_PhysLayer",
        "GeneralRBM_PhysLayer",
        "ChargeRBM_HiddenLayer",
        "SpinRBM_HiddenLayer",
        "GeneralRBM_HiddenLayer",
        "ChargeRBM_PhysHidden",
        "SpinRBM_PhysHidden",
        "GeneralRBM_PhysHidden",
    ] {
        let data = parse_expert_mode_files(root.join(format!("namelist_{name}.def"))).unwrap();
        let complex = name.starts_with("General");
        assert_eq!(
            data.optimization_flags,
            vec![1, i64::from(complex), 2, 2 * i64::from(complex), 0, 0],
            "{name}"
        );
    }
}

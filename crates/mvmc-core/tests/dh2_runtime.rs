//! DH2 production support and canonical complex-mode selection.
#[path = "../../mvmc-expert-parsers/tests/support/historical_component_sequence.rs"]
mod historical_component_sequence;
#[path = "../../../tests/support/historical_orbital_model.rs"]
mod historical_orbital_model;
#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;
use historical_orbital_model::historical_kernel_model as parse_expert_mode_files;
use std::path::Path;

#[test]
fn supported_dh2_sections_pass_runtime_validation_with_or_without_namelist() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/dh2");
    for name in [
        "orbital_first",
        "dh_first",
        "alias",
        "real",
        "empty",
        "replacement",
    ] {
        // A repeated archived keyword is component-sequence metadata, not C
        // loader acceptance. The helper checks duplicate rejection, separately
        // reads the sections and builds a valid single-keyword final model.
        let mut data = historical_component_sequence::model(
            &root.join(format!("namelist_{name}.def")),
            "DH2",
            |path| parse_expert_mode_files(path),
        );
        // Normal orbitals need 2Sz = 0 (C readdef.c:691-696).
        data.modpara.two_sz = 0;
        // Use the same final single-keyword metadata as the valid loader stage,
        // not the archived component sequence preserved by the parser helper.
        let final_definition = data.namelist.iter().rposition(|(kind, _)| kind == "DH2");
        // Archived zero-count DH is a no-DH programmatic layout, not a valid
        // required C definition. The shared helper separately asserts rejection.
        let active = data.projection_layout().n_dh2 != 0;
        data.namelist = data
            .namelist
            .iter()
            .enumerate()
            .filter(|(index, (kind, _))| {
                kind != "DH2" || (active && Some(*index) == final_definition)
            })
            .map(|(_, entry)| entry.clone())
            .collect();
        mvmc_core::validation::validate_para_opt(&data).unwrap();
        data.namelist.clear();
        mvmc_core::validation::validate_para_opt(&data).unwrap();
    }
}

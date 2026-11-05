//! DH4 production support and canonical runtime-mode selection.
#[path = "../../mvmc-expert-parsers/tests/support/historical_component_sequence.rs"]
mod historical_component_sequence;
#[path = "../../../tests/support/historical_orbital_model.rs"]
mod historical_orbital_model;
#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;
use historical_orbital_model::historical_kernel_model as parse_expert_mode_files;
use std::path::Path;

#[test]
fn parsed_and_programmatic_dh4_sections_pass_runtime_validation() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/dh4");
    for name in [
        "orbital_first",
        "dh_first",
        "alias",
        "real",
        "empty",
        "replacement",
        "reverse_replacement",
        "combined_real",
        "only_dh4_complex",
        "empty_last",
        "empty_real",
        "ap_parallel",
        "general",
    ] {
        // Runtime validity of the final historical model is separate from
        // rejection of its archived repeated filename keywords.
        let mut data = historical_component_sequence::model(
            &root.join(format!("namelist_{name}.def")),
            "DH4",
            |path| parse_expert_mode_files(path),
        );
        let final_definition = data.namelist.iter().rposition(|(kind, _)| kind == "DH4");
        // Archived zero-count DH is a no-DH programmatic layout, not a valid
        // required C definition. The shared helper separately asserts rejection.
        let active = data.projection_layout().n_dh4 != 0;
        data.namelist = data
            .namelist
            .iter()
            .enumerate()
            .filter(|(index, (kind, _))| {
                kind != "DH4" || (active && Some(*index) == final_definition)
            })
            .map(|(_, entry)| entry.clone())
            .collect();
        mvmc_core::validation::validate_para_opt(&data).unwrap();
        data.namelist.clear();
        mvmc_core::validation::validate_para_opt(&data).unwrap();
    }
}

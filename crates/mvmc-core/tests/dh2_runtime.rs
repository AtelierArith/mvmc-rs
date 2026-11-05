//! DH2 production support and canonical complex-mode selection.
#[path = "../../mvmc-expert-parsers/tests/support/historical_component_sequence.rs"]
mod historical_component_sequence;
#[path = "../../../tests/support/historical_orbital_model.rs"]
mod historical_orbital_model;
#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;
use historical_orbital_model::historical_kernel_model as parse_expert_mode_files;
use mvmc_core::ExpertModeData;
use mvmc_expert_parsers::utils::parameter_init::all_complex_flag;
use num_complex::Complex64;
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

#[test]
fn dh2_runtime_mode_uses_declarations_and_ignores_loaded_imaginary_values() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/dh2");
    let fixture = std::fs::read_to_string(root.join("mode.txt")).unwrap();
    for line in fixture.lines().filter(|l| !l.starts_with('#')) {
        let row: Vec<i64> = line
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        let mut data = ExpertModeData::new();
        data.doublon_holon_2site_complex = row[0] != 0;
        data.doublon_holon_2site_params = vec![Complex64::new(0.125, row[1] as f64 / 4.0)];
        if row[2] != -1 {
            data.complex_flags = vec![row[2]];
        }
        assert_eq!(
            mvmc_core::run::get_all_complex_flag(&data).unwrap(),
            // C uses definition flags; the archived Julia result in row[3]
            // also inferred the mode from loaded imaginary coefficients.
            if row[2] == -1 {
                row[0] != 0
            } else {
                row[2] != 0
            },
            "{line}"
        );
        // Initializing Slater parameters uses declarations, never loaded values
        // or the optional runtime override; changing this would alter draw count.
        assert_eq!(all_complex_flag(&data).unwrap(), row[0] != 0, "{line}");
    }
}

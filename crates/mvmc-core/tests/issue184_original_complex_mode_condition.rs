//! Original Julia8bb unsupported_inputs.jl:262–265, M1032.
//! Typed helper/query scope only: declaration-only data is not a valid C model.
//! C readdef.c641–644 includes DH2/DH4 declaration flags in AllComplexFlag.
use mvmc_core::get_all_complex_flag;
use mvmc_expert_parsers::ExpertModeData;

#[test]
fn original_declaration_only_dh2_marks_public_optimizer_mode_complex() {
    let mut data = ExpertModeData::new();
    assert!(data.complex_flags.is_empty());
    assert!(data.doublon_holon_2site_indices.is_empty());
    assert!(data.doublon_holon_2site_params.is_empty());
    assert!(!get_all_complex_flag(&data));
    // Exact original input: no coefficients, index table or imaginary values.
    data.doublon_holon_2site_complex = true;
    assert!(get_all_complex_flag(&data)); // Original M1032 expected literal true.
    assert!(data.doublon_holon_2site_complex);
    assert!(!data.doublon_holon_4site_complex);
    assert!(data.complex_flags.is_empty());
    assert!(data.doublon_holon_2site_indices.is_empty());
    assert!(data.doublon_holon_2site_params.is_empty());
    // Helper query is fresh/read-only, not cached after the original assertion.
    data.doublon_holon_2site_complex = false;
    assert!(!get_all_complex_flag(&data));
}

#[test]
fn sibling_declaration_only_dh4_participates_without_materializing_coefficients() {
    // Additional source-bound sibling, not another original M1032 assertion.
    let mut data = ExpertModeData::new();
    data.doublon_holon_4site_complex = true;
    assert!(!data.doublon_holon_2site_complex);
    assert!(data.complex_flags.is_empty());
    assert!(data.doublon_holon_4site_indices.is_empty());
    assert!(data.doublon_holon_4site_params.is_empty());
    assert!(get_all_complex_flag(&data));
    assert!(data.doublon_holon_4site_complex);
    assert!(data.doublon_holon_4site_indices.is_empty());
    assert!(data.doublon_holon_4site_params.is_empty());
    assert!(data.complex_flags.is_empty());
}

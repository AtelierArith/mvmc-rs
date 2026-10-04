//! A006 manual architecture, separately observed native admission; no file loader.

use mvmc_expert_parsers::{
    judge_orbital_mode, ExpertModeData, NativeOrbitalModeStatus as Native,
    OrbitalModeWarning as Warning,
};
use num_complex::Complex64;

fn data(general: i64, ap: i64, parallel: i64) -> ExpertModeData {
    ExpertModeData {
        i_flg_orbital_general: general,
        i_flg_orbital_anti_parallel: ap,
        i_flg_orbital_parallel: parallel,
        n_orbital_anti_parallel: 3,
        optimization_flags: vec![2, -1, 0, 7],
        slater_params: vec![Complex64::new(-0.25, 0.5)],
        rbm_params: vec![Complex64::new(2.0, -3.0)],
        ..Default::default()
    }
}

#[test]
fn all_eight_binary_cases_separate_original_manual_update_from_c_admission() {
    for (g, ap, p, expected_g, native, warning) in [
        (0, 0, 0, 0, Native::MissingAntiParallel, None),
        (0, 0, 1, 0, Native::MissingAntiParallel, None),
        (0, 1, 0, 0, Native::Accepted, None),
        (0, 1, 1, 1, Native::Accepted, None),
        (1, 0, 0, 1, Native::Accepted, None),
        (
            1,
            0,
            1,
            1,
            Native::MultipleDefinitions,
            Some(Warning::ConflictingDefinitions),
        ),
        (
            1,
            1,
            0,
            1,
            Native::MultipleDefinitions,
            Some(Warning::ConflictingDefinitions),
        ),
        (
            1,
            1,
            1,
            1,
            Native::MultipleDefinitions,
            Some(Warning::ConflictingDefinitions),
        ),
    ] {
        let mut actual = data(g, ap, p);
        let mut expected = actual.clone();
        expected.i_flg_orbital_general = expected_g;
        let report = judge_orbital_mode(&mut actual);
        assert_eq!(report.general_mode, expected_g);
        assert_eq!(report.native_status, native);
        assert_eq!(report.warning, warning);
        assert_eq!(format!("{actual:?}"), format!("{expected:?}"));
    }
}

#[test]
fn repeated_inferred_general_call_retains_original_conflict_observation() {
    let mut actual = data(0, 1, 1);
    let first = judge_orbital_mode(&mut actual);
    assert_eq!(first.native_status, Native::Accepted);
    assert_eq!(first.warning, None);
    let before = format!("{actual:?}");
    let second = judge_orbital_mode(&mut actual);
    assert_eq!(second.native_status, Native::MultipleDefinitions);
    assert_eq!(second.warning, Some(Warning::ConflictingDefinitions));
    assert_eq!(format!("{actual:?}"), before);
}

#[test]
fn nonbinary_annotations_are_not_new_mandatory_errors_or_metadata_repairs() {
    for (g, ap, p, expected_g, native) in [
        (2, 1, 1, 1, Native::Accepted),
        (-1, 0, 0, -1, Native::MissingAntiParallel),
        (1, 2, 0, 1, Native::MultipleDefinitions),
        (0, 2, 1, 0, Native::MissingAntiParallel),
    ] {
        let mut actual = data(g, ap, p);
        actual
            .native_complex_headers
            .insert("OrbitalGeneral".into(), -2);
        actual
            .native_complex_declarations
            .insert("Orbitals".into(), true);
        let mut expected = actual.clone();
        expected.i_flg_orbital_general = expected_g;
        let report = judge_orbital_mode(&mut actual);
        assert_eq!(report.native_status, native);
        assert_eq!(report.warning, None);
        assert_eq!(format!("{actual:?}"), format!("{expected:?}"));
    }
}

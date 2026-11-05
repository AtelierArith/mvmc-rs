mod historical_overlay_stage {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/support/historical_overlay_stage.rs"
    ));
}
#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;
use historical_orbital_model::historical_kernel_model as parse_expert_mode_files;
use historical_overlay_stage::read_input_parameters;
use mvmc_core::initial_params::{read_initial_def, read_opt_para_file};
use num_complex::Complex64;
use std::path::{Path, PathBuf};
fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/rbm")
}

#[test]
fn full_initial_and_strict_optimized_records_scatter_rbm_before_slater() {
    let mut data = parse_expert_mode_files(root().join("namelist_all.def")).unwrap();
    assert!(read_initial_def(&mut data, root().join("production/initial.def")).unwrap());
    assert_eq!(
        read_opt_para_file(&mut data, root().join("production/initial.def")).unwrap(),
        36
    );
    let mut offsets = [5; 9];
    for i in 1..9 {
        offsets[i] = offsets[i - 1] + 3;
    }
    data.visit_rbm_terms_mut(|s, t| {
        let idx = offsets[s] + t.idx() as usize;
        assert_eq!(
            t.value(),
            Complex64::new((idx + 1) as f64 / 8.0, -((idx + 1) as f64) / 16.0)
        );
    });
    for t in data.orbital_terms {
        let idx = 5 + 27 + t.idx as usize;
        assert_eq!(
            data.slater_params[t.idx as usize],
            Complex64::new((idx + 1) as f64 / 8.0, -((idx + 1) as f64) / 16.0)
        );
    }
}

#[test]
fn nine_optional_overlays_share_indices_follow_last_values_and_ignore_unknown_names() {
    let namelist = root().join("production/namelist_all.def");
    let mut data = parse_expert_mode_files(&namelist).unwrap();
    read_input_parameters(&mut data, &namelist).unwrap();
    let mut visits = 0;
    data.visit_rbm_terms_mut(|_, t| {
        visits += 1;
        let expected = match t.idx() {
            0 => Complex64::new(0.125, -0.0625),
            2 => Complex64::new(0.25, -0.125),
            _ => unreachable!(),
        };
        assert_eq!(t.value(), expected);
    });
    assert_eq!(visits, 18);
}

#[path = "../../../tests/support/historical_orbital_model.rs"]
mod historical_orbital_model;

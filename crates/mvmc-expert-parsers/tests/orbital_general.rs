use std::path::Path;

use mvmc_expert_parsers::parse_expert_mode_files;

#[test]
fn parsed_general_and_ap_parallel_cache_the_same_spin_site_matrices() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/orbital_general");
    let general =
        parse_expert_mode_files(root.join("../c_orbital_inputs/namelist_general.def")).unwrap();
    assert!(
        general.input_errors.is_empty(),
        "{:?}",
        general.input_errors
    );
    let parallel =
        parse_expert_mode_files(root.join("../c_orbital_inputs/namelist_ap_parallel.def")).unwrap();
    assert!(
        parallel.input_errors.is_empty(),
        "{:?}",
        parallel.input_errors
    );
    let indices = general
        .orbital_idx_matrix
        .as_ref()
        .expect("General caches at parse time");
    assert_eq!(indices.len(), 6);
    assert!(indices.iter().all(|row| row.len() == 6));
    assert_eq!(general.orbital_idx_matrix, parallel.orbital_idx_matrix);
    assert_eq!(general.orbital_sgn_matrix, parallel.orbital_sgn_matrix);
    assert_eq!(indices[0][1], 9, "up-up");
    assert_eq!(indices[3][4], 10, "down-down");
    assert_eq!(indices[0][3], 0, "up-down");
    assert_eq!(indices[4][2], 7, "down-up mirrored from up-down");
}

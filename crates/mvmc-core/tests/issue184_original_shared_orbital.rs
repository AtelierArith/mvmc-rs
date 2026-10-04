//! Original M0585 shared coefficient observation, not independent shadow repair.
use std::collections::BTreeMap;
use std::path::Path;

use mvmc_core::{pack_parameters, unpack_parameters};
use num_complex::Complex64;

#[test]
fn original_m0585_unpack_assigns_shared_orbital_coefficients() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/original_heisenberg_parser_184/namelist.def");
    let mut data = mvmc_expert_parsers::parse_expert_mode_files(path).unwrap();
    assert!(data.input_errors.is_empty(), "{:?}", data.input_errors);
    let original_mapping = data.orbital_terms.clone();
    let mut occurrences = BTreeMap::new();
    for term in &original_mapping {
        assert!(term.idx >= 0);
        *occurrences.entry(term.idx as usize).or_insert(0usize) += 1;
    }
    assert!(occurrences.values().any(|&count| count > 1));
    let offset = data.projection_layout().n_proj + data.count_rbm_parameters();
    let before = pack_parameters(&data).unwrap();
    // Original Julia pack .+ ComplexF64(0.1,-0.1), not Rust-generated goldens.
    let expected: Vec<_> = before
        .iter()
        .map(|value| *value + Complex64::new(0.1, -0.1))
        .collect();
    unpack_parameters(&mut data, &expected).unwrap();
    assert_eq!(data.orbital_terms, original_mapping);
    assert_eq!(pack_parameters(&data).unwrap(), expected);
    for (idx, &count) in &occurrences {
        let assigned = before[offset + idx] + Complex64::new(0.1, -0.1);
        assert_eq!(data.slater_params[*idx], assigned);
        let observations: Vec<_> = data
            .orbital_terms
            .iter()
            .filter(|term| term.idx as usize == *idx)
            .map(|term| data.slater_params[term.idx as usize])
            .collect();
        assert_eq!(observations.len(), count);
        assert!(observations.iter().all(|&value| value == assigned));
    }
    // Include every declared dense slot, not merely slots with mapping rows.
    // Any unreferenced slot keeps its independently assigned packed value;
    // observations of a shared idx must not overwrite unrelated slots.
    for (idx, actual) in data.slater_params.iter().enumerate() {
        assert_eq!(*actual, before[offset + idx] + Complex64::new(0.1, -0.1));
        if !occurrences.contains_key(&idx) {
            assert_eq!(*actual, expected[offset + idx]);
        }
    }
}

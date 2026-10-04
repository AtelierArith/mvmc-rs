//! Julia test_read_input_parameters.jl cases for currently supported factors.
use mvmc_expert_parsers::utils::read_input_parameters::{
    parse_input_parameter_file, read_input_parameters,
};
use mvmc_expert_parsers::{ExpertModeData, OrbitalTerm};
use num_complex::Complex64;
use std::fs;

fn directory(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("mvmc-overlay-{}-{name}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    dir
}
fn definition(count: usize, rows: &str) -> String {
    format!("===\nNOrbitalIdx {count}\nComplexType 1\n===\n===\n{rows}")
}
fn data() -> ExpertModeData {
    let mut d = ExpertModeData::new();
    d.modpara.n_orbital_idx = 5;
    d.slater_params = vec![Complex64::new(7.0, 0.0); 5];
    d.n_orbital_anti_parallel = 3;
    d.i_flg_orbital_anti_parallel = 1;
    d.i_flg_orbital_parallel = 1;
    for idx in [0, 1, 3, 4, 3] {
        d.orbital_terms.push(OrbitalTerm {
            site1: 0,
            site2: 1,
            idx,
            is_complex: true,
            sign: 1,
        });
    }
    d
}
#[test]
fn permissive_parser_preserves_julia_fallback_and_last_duplicate() {
    let dir = directory("permissive");
    let path = dir.join("in.def");
    fs::write(
        &path,
        definition(
            3,
            "0 1 2\n-1 9 9\nbad 3 4\n1 broken 5\n0 6 7 # overwrite\n2 8\n",
        ),
    )
    .unwrap();
    let params = parse_input_parameter_file(&path).unwrap();
    assert_eq!(params.len(), 2);
    assert_eq!(params[&0], Complex64::new(6.0, 7.0));
    assert_eq!(params[&1], Complex64::new(0.0, 5.0));
    assert!(parse_input_parameter_file(dir.join("missing.def"))
        .unwrap()
        .is_empty());
    fs::remove_dir_all(dir).unwrap();
}
#[test]
fn overlays_follow_c_keyword_order_and_exact_declared_parallel_offset() {
    let dir = directory("order");
    fs::write(
        dir.join("namelist.def"),
        "InOrbitalGeneral g.def\nInOrbitalParallel p.def\nInOrbital missing.def\n",
    )
    .unwrap();
    fs::write(dir.join("p.def"), definition(2, "1 20 0.2\n0 10 0.1\n")).unwrap();
    fs::write(dir.join("g.def"), definition(1, "3 30 0.3\n")).unwrap();
    let mut d = data();
    read_input_parameters(&mut d, dir.join("namelist.def")).unwrap();
    let vals: Vec<_> = d
        .orbital_terms
        .iter()
        .map(|t| d.slater_params[t.idx as usize])
        .collect();
    assert_eq!(
        vals,
        [
            Complex64::new(7.0, 0.0),
            Complex64::new(7.0, 0.0),
            Complex64::new(30.0, 0.3),
            Complex64::new(20.0, 0.2),
            Complex64::new(30.0, 0.3)
        ]
    );
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn duplicate_c_overlay_keywords_are_rejected_before_mutation() {
    let dir = directory("duplicate-keyword");
    fs::write(
        dir.join("namelist.def"),
        "InOrbitalGeneral first.def\nInOrbitalGeneral second.def\n",
    )
    .unwrap();
    fs::write(dir.join("first.def"), definition(1, "0 1 0\n")).unwrap();
    fs::write(dir.join("second.def"), definition(1, "0 2 0\n")).unwrap();
    let mut data = data();
    let error = read_input_parameters(&mut data, dir.join("namelist.def")).unwrap_err();
    assert!(
        error.contains("duplicate keyword InOrbitalGeneral"),
        "{error}"
    );
    assert!(data
        .slater_params
        .iter()
        .all(|&value| value == Complex64::new(7.0, 0.0)));
    fs::remove_dir_all(dir).unwrap();
}
#[test]
fn malformed_parallel_overlay_is_atomic() {
    let dir = directory("strict");
    fs::write(dir.join("namelist.def"), "InOrbitalParallel p.def\n").unwrap();
    for (count, rows) in [
        (2, "0 1 0\n0 2 0\n"),
        (3, "0 1 0\n1 2 0\n"),
        (2, "0 1 0\n"),
        (2, "0 1 0\n2 2 0\n"),
        (2, "0 NaN 0\n1 2 0\n"),
        (2, "0 1 0 extra\n1 2 0\n"),
    ] {
        fs::write(dir.join("p.def"), definition(count, rows)).unwrap();
        let mut d = data();
        assert!(read_input_parameters(&mut d, dir.join("namelist.def")).is_err());
        assert!(d
            .slater_params
            .iter()
            .all(|&v| v == Complex64::new(7.0, 0.0)));
    }
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn orbital_overlay_preserves_unmapped_internal_and_trailing_declared_slots() {
    let dir = directory("unmapped");
    fs::write(dir.join("namelist.def"), "InOrbitalGeneral full.def\n").unwrap();
    fs::write(
        dir.join("full.def"),
        definition(5, "0 1 0\n1 2 0\n2 8 -1\n3 4 0\n4 5 0\n"),
    )
    .unwrap();
    let mut data = data();
    data.orbital_terms.truncate(2);
    let mappings = data.orbital_terms.clone();
    read_input_parameters(&mut data, dir.join("namelist.def")).unwrap();
    assert_eq!(
        data.slater_params,
        [
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(8.0, -1.0),
            Complex64::new(4.0, 0.0),
            Complex64::new(5.0, 0.0),
        ]
    );
    assert_eq!(data.orbital_terms, mappings);
    fs::remove_dir_all(dir).unwrap();
}

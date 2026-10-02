//! Canonical DH4 definition and combined DH2/DH4 layout boundaries.
use std::path::{Path, PathBuf};

use mvmc_expert_parsers::parse_expert_mode_files;
use mvmc_expert_parsers::parsers::doublon_holon::{
    parse_doublon_holon_4site_content, parse_doublon_holon_4site_def,
};
use mvmc_expert_parsers::utils::parameter_init::{all_complex_flag, init_parameter};
use num_complex::Complex64;
use sfmt19937::Sfmt19937Rng;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/dh4")
}

#[test]
fn dh4_reserves_ten_dense_parameters_per_index_after_dh2_before_orbital_flags() {
    for name in ["orbital_first", "dh_first", "alias"] {
        let data = parse_expert_mode_files(root().join(format!("namelist_{name}.def"))).unwrap();
        assert_eq!(data.projection_layout().n_dh2, 2, "{name}");
        assert_eq!(data.projection_layout().n_dh4, 2, "{name}");
        assert_eq!(data.projection_layout().dh4_offset, 17, "{name}");
        assert_eq!(data.projection_layout().n_proj, 37, "{name}");
        assert_eq!(data.optimization_flags.len(), 82, "{name}");
    }
}

#[test]
fn required_dh4_missing_and_invalid_definitions_fail_during_parsing() {
    for name in ["invalid", "missing", "invalid_alias", "missing_alias"] {
        assert!(
            parse_expert_mode_files(root().join(format!("namelist_{name}.def"))).is_err(),
            "{name}"
        );
    }
}

#[test]
fn c_reads_modpara_before_a_dh4_definition_listed_first() {
    let data = parse_expert_mode_files(root().join("namelist_before_modpara.def")).unwrap();
    assert_eq!(data.modpara.nsite, 3);
    assert_eq!(data.projection_layout().n_dh4, 1);
    assert_eq!(data.doublon_holon_4site_indices.len(), 1);
    assert_eq!(data.doublon_holon_4site_params.len(), 10);
}

fn integers(line: &str) -> Vec<i64> {
    line.split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect()
}

fn check_bits(values: impl IntoIterator<Item = Complex64>, line: &str, label: &str) {
    let actual: Vec<_> = values
        .into_iter()
        .flat_map(|v| [v.re.to_bits(), v.im.to_bits()])
        .collect();
    let expected: Vec<_> = line
        .split_whitespace()
        .map(|s| u64::from_str_radix(s, 16).unwrap())
        .collect();
    assert_eq!(actual, expected, "{label}");
}

#[test]
fn strict_tables_neighbors_flags_errors_and_last_line_match_original_julia() {
    let fixture = std::fs::read_to_string(root().join("parser.txt")).unwrap();
    let mut lines = fixture.lines().filter(|line| !line.starts_with('#'));
    while let Some(header) = lines.next() {
        let parts: Vec<_> = header.split_whitespace().collect();
        let content = std::fs::read_to_string(root().join(format!("{}.def", parts[0]))).unwrap();
        let result = parse_doublon_holon_4site_content(&content, parts[1].parse().unwrap());
        assert_eq!(result.is_success(), parts[2] == "1", "{header}");
        assert_eq!(
            result.line_number,
            parts[3].parse::<usize>().unwrap(),
            "{header}"
        );
        assert_eq!(result.error_message, lines.next().unwrap(), "{header}");
        if let Some(definition) = result.data {
            let shape = integers(lines.next().unwrap());
            assert_eq!(definition.indices.len(), shape[0] as usize);
            assert_eq!(definition.is_complex, shape[1] != 0);
            for table in definition.indices {
                let flat: Vec<_> = table.neighbors.into_iter().flatten().collect();
                assert_eq!(flat, integers(lines.next().unwrap()), "{header}");
            }
            assert_eq!(
                definition.opt_flags,
                integers(lines.next().unwrap())
                    .into_iter()
                    .map(|i| i != 0)
                    .collect::<Vec<_>>(),
                "{header}"
            );
        }
    }
    assert!(parse_doublon_holon_4site_def(root().join("absent.def"), 3).is_err());
}

#[test]
fn final_layout_component_flags_projection_packing_initial_values_and_rng_match_julia() {
    let fixture = std::fs::read_to_string(root().join("initial.txt")).unwrap();
    let mut lines = fixture.lines().filter(|line| !line.starts_with('#'));
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
        assert_eq!(lines.next().unwrap(), name);
        let mut data =
            parse_expert_mode_files(root().join(format!("namelist_{name}.def"))).unwrap();
        let layout = data.projection_layout();
        assert_eq!(
            [
                layout.n_gutzwiller,
                layout.n_jastrow,
                layout.n_spinjastrow,
                layout.n_dh2,
                layout.n_dh4,
                layout.gutzwiller_offset,
                layout.jastrow_offset,
                layout.spinjastrow_offset,
                layout.dh2_offset,
                layout.dh4_offset,
                layout.n_proj
            ]
            .map(|i| i as i64)
            .as_slice(),
            integers(lines.next().unwrap()),
            "{name} layout"
        );
        assert_eq!(
            data.optimization_flags,
            integers(lines.next().unwrap())
                .into_iter()
                .map(|i| i != 0)
                .collect::<Vec<_>>(),
            "{name} flags"
        );
        let mode = integers(lines.next().unwrap());
        assert_eq!(data.doublon_holon_2site_complex, mode[0] != 0);
        assert_eq!(data.doublon_holon_4site_complex, mode[1] != 0);
        assert_eq!(all_complex_flag(&data), mode[2] != 0);
        assert_eq!(data.i_flg_orbital_general, mode[3]);
        assert_eq!(data.n_orbital_anti_parallel, mode[4]);
        assert_eq!(data.modpara.n_orbital_idx, mode[5]);
        assert_eq!(data.doublon_holon_4site_params.len(), 10 * layout.n_dh4);
        let definition = data.doublon_holon_4site_indices.clone();
        let dh2_definition = data.doublon_holon_2site_indices.clone();
        let flags = data.optimization_flags.clone();
        for (i, value) in data.doublon_holon_2site_params.iter_mut().enumerate() {
            *value = Complex64::new((i + 1) as f64 / 8.0, -(i as f64 + 1.0) / 16.0);
        }
        for (i, value) in data.doublon_holon_4site_params.iter_mut().enumerate() {
            *value = Complex64::new((i + 1) as f64 / 32.0, -(i as f64 + 1.0) / 64.0);
        }
        check_bits(data.projection_parameters(), lines.next().unwrap(), name);
        let mut rng = Sfmt19937Rng::new(11272);
        init_parameter(&mut data, &mut rng);
        assert!(data
            .doublon_holon_4site_params
            .iter()
            .all(|v| *v == Complex64::new(0.0, 0.0)));
        assert_eq!(data.doublon_holon_4site_indices, definition);
        assert_eq!(data.doublon_holon_2site_indices, dh2_definition);
        assert!(data
            .doublon_holon_2site_params
            .iter()
            .all(|v| *v == Complex64::new(0.0, 0.0)));
        assert_eq!(data.optimization_flags, flags);
        let values = data
            .projection_parameters()
            .into_iter()
            .chain(data.orbital_terms.iter().map(|t| t.value));
        check_bits(values, lines.next().unwrap(), name);
        let words = integers(lines.next().unwrap());
        assert_eq!(words.len(), 624);
        assert_eq!(
            (0..624)
                .map(|_| i64::from(rng.gen_rand32()))
                .collect::<Vec<_>>(),
            words,
            "{name} RNG"
        );
    }
    assert!(lines.next().is_none());
}

#[test]
fn combined_validation_reports_dh4_shapes_and_all_four_neighbors_in_source_order() {
    let mut d = mvmc_expert_parsers::ExpertModeData::new();
    d.modpara.nsite = 3;
    d.doublon_holon_4site_indices = vec![
        mvmc_expert_parsers::DoublonHolon4SiteIndex {
            neighbors: vec![[0; 4]; 2],
        },
        mvmc_expert_parsers::DoublonHolon4SiteIndex {
            neighbors: vec![[-1, 3, 0, 0], [2, i64::MAX, 0, 0], [0; 4]],
        },
        mvmc_expert_parsers::DoublonHolon4SiteIndex {
            neighbors: vec![[0; 4]; 3],
        },
    ];
    let result = mvmc_expert_parsers::validate_expert_mode_data(&d);
    let errors: Vec<_> = result
        .errors
        .iter()
        .filter(|e| e.starts_with("DH4"))
        .cloned()
        .collect();
    assert_eq!(
        errors,
        vec![
            "DH4 index 0: neighbors must be 3 x 4".to_string(),
            "DH4 index 1 site 0 neighbor 0=-1 out of range [0, 2]".to_string(),
            "DH4 index 1 site 0 neighbor 1=3 out of range [0, 2]".to_string(),
            format!(
                "DH4 index 1 site 1 neighbor 1={} out of range [0, 2]",
                i64::MAX
            ),
        ]
    );
    assert!(!result.is_valid);
}

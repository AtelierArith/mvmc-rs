//! Canonical strict DH2 table and final projection-layout contracts.
#[path = "support/historical_component_sequence.rs"]
mod historical_component_sequence;
#[path = "../../../tests/support/historical_optimization_flags.rs"]
mod historical_optimization_flags;
#[path = "../../../tests/support/historical_orbital_model.rs"]
mod historical_orbital_model;
#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;
use std::path::{Path, PathBuf};

use historical_orbital_model::historical_kernel_model as parse_expert_mode_files;
use mvmc_expert_parsers::parsers::doublon_holon::{
    parse_doublon_holon_2site_content, parse_doublon_holon_2site_def,
};
use mvmc_expert_parsers::utils::parameter_init::{all_complex_flag, init_parameter};
use num_complex::Complex64;
use sfmt19937::Sfmt19937Rng;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/dh2")
}

#[test]
fn dh2_reserves_six_dense_parameters_per_index_before_orbital_flags() {
    for name in ["orbital_first", "dh_first", "alias"] {
        let data = parse_expert_mode_files(root().join(format!("namelist_{name}.def"))).unwrap();
        assert_eq!(data.projection_layout().n_proj, 17, "{name}");
        assert_eq!(data.optimization_flags.len(), 42, "{name}");
    }
}

#[test]
fn required_dh2_missing_invalid_and_missing_modpara_definitions_fail_during_parsing() {
    for name in ["invalid", "missing", "before_modpara"] {
        assert!(
            parse_expert_mode_files(root().join(format!("namelist_{name}.def"))).is_err(),
            "{name}"
        );
    }
}

#[test]
fn c_reads_modpara_before_a_dh2_definition_listed_first() {
    let dir = std::env::temp_dir().join(format!("mvmc-dh2-order-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("namelist.def"),
        format!(
            "DH2 {}\nModPara {}\n",
            root().join("complex.def").display(),
            root().join("modpara.def").display()
        ),
    )
    .unwrap();
    let data = parse_expert_mode_files(dir.join("namelist.def")).unwrap();
    assert_eq!(data.modpara.nsite, 3);
    assert_eq!(data.projection_layout().n_dh2, 1);
    assert_eq!(data.doublon_holon_2site_indices.len(), 1);
    assert_eq!(data.doublon_holon_2site_params.len(), 6);
    std::fs::remove_dir_all(dir).unwrap();
}

fn integers(line: &str) -> Vec<i64> {
    line.split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect()
}

// Initial coefficients use only a few scale/divide/trig operations;
// 1e-14 covers their rounding while SFMT words and layout stay exact.
fn check_computed_values(values: impl IntoIterator<Item = Complex64>, expected: &str, label: &str) {
    numerical_comparison::assert_values_close(
        values.into_iter().flat_map(|v| [v.re, v.im]),
        numerical_comparison::hex_values(expected),
        1e-14,
        1e-14,
        label,
    );
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
fn historical_julia_diagnostics_except_c_integer_flag_extensions() {
    let fixture = std::fs::read_to_string(root().join("parser.txt")).unwrap();
    let mut lines = fixture.lines().filter(|line| !line.starts_with('#'));
    while let Some(header) = lines.next() {
        let parts: Vec<_> = header.split_whitespace().collect();
        let content = std::fs::read_to_string(root().join(format!("{}.def", parts[0]))).unwrap();
        let result = parse_doublon_holon_2site_content(&content, parts[1].parse().unwrap());
        // Julia rejected these integer flags; the native C DH readers accept
        // them. Keep the historical diagnostic record unchanged and verify
        // the new values separately against c_dh_flags.txt.
        if matches!(parts[0], "negative_opt" | "opt_bounds" | "nonbinary_opt") {
            assert!(result.is_success(), "{header}");
            assert_eq!(
                result.data.as_ref().unwrap().opt_flags[0],
                if parts[0] == "negative_opt" { -1 } else { 2 }
            );
            lines.next().unwrap(); // archived Julia error text
            continue;
        }
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
                integers(lines.next().unwrap()),
                "{header}"
            );
        }
    }
    assert!(parse_doublon_holon_2site_def(root().join("absent.def"), 3).is_err());
}

#[test]
fn layout_and_mapped_values_match_julia_while_declared_slot_rng_matches_c() {
    let fixture = std::fs::read_to_string(root().join("initial.txt")).unwrap();
    let mut lines = fixture.lines().filter(|line| !line.starts_with('#'));
    for name in [
        "orbital_first",
        "dh_first",
        "alias",
        "real",
        "empty",
        "replacement",
    ] {
        assert_eq!(lines.next().unwrap(), name);
        let mut data = historical_component_sequence::model(
            &root().join(format!("namelist_{name}.def")),
            "DH2",
            |path| parse_expert_mode_files(path),
        );
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
        let expected_flags = historical_optimization_flags::c_orbital_representation(
            &data,
            integers(lines.next().unwrap()),
        );
        assert_eq!(data.optimization_flags, expected_flags, "{name} flags");
        let mode = integers(lines.next().unwrap());
        assert_eq!(data.doublon_holon_2site_complex, mode[0] != 0);
        assert_eq!(all_complex_flag(&data).unwrap(), mode[1] != 0);
        assert_eq!(data.doublon_holon_2site_params.len(), 6 * layout.n_dh2);
        let definition = data.doublon_holon_2site_indices.clone();
        let flags = data.optimization_flags.clone();
        for (i, value) in data.doublon_holon_2site_params.iter_mut().enumerate() {
            *value = Complex64::new((i + 1) as f64 / 8.0, -(i as f64 + 1.0) / 16.0);
        }
        check_bits(data.projection_parameters(), lines.next().unwrap(), name);
        let mut rng = Sfmt19937Rng::new(11272);
        init_parameter(&mut data, &mut rng).unwrap();
        assert!(data
            .doublon_holon_2site_params
            .iter()
            .all(|v| *v == Complex64::new(0.0, 0.0)));
        assert_eq!(data.doublon_holon_2site_indices, definition);
        assert_eq!(data.optimization_flags, flags);
        let values = data.projection_parameters().into_iter().chain(
            data.orbital_terms
                .iter()
                .map(|t| data.slater_params[t.idx as usize]),
        );
        check_computed_values(values, lines.next().unwrap(), name);
        let historical_words = integers(lines.next().unwrap());
        assert_eq!(historical_words.len(), 624);
        // Historical sparse Julia records omit declared slot 3. Compare RNG
        // with C at the full declared width, including the complete AP/P cases.
        let c_rows: Vec<_> =
            include_str!("../../../tests/fixtures/orbital_general/c_declared_flags.txt")
                .lines()
                .filter(|line| !line.starts_with('#'))
                .collect();
        let c_record = c_rows
            .as_chunks::<4>()
            .0
            .iter()
            .find(|record| {
                record[0]
                    == format!(
                        "{} {} 11272",
                        data.modpara.n_orbital_idx,
                        i64::from(all_complex_flag(&data).unwrap())
                    )
            })
            .unwrap();
        let words = integers(c_record[3]);
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
fn combined_validation_reports_dh_table_shapes_and_neighbors_in_source_order() {
    let mut d = mvmc_expert_parsers::ExpertModeData::new();
    d.modpara.nsite = 3;
    d.doublon_holon_2site_indices = vec![
        mvmc_expert_parsers::DoublonHolon2SiteIndex {
            neighbors: vec![[0, 0]; 2],
        },
        mvmc_expert_parsers::DoublonHolon2SiteIndex {
            neighbors: vec![[-1, 3], [2, i64::MAX], [0, 0]],
        },
        mvmc_expert_parsers::DoublonHolon2SiteIndex {
            neighbors: vec![[0, 0]; 3],
        },
    ];
    let result = mvmc_expert_parsers::validate_expert_mode_data(&d);
    let dh_errors: Vec<_> = result
        .errors
        .iter()
        .filter(|e| e.starts_with("DH2"))
        .cloned()
        .collect();
    assert_eq!(
        dh_errors,
        vec![
            "DH2 index 0: neighbors must be 3 x 2".to_string(),
            "DH2 index 1 site 0 neighbor 0=-1 out of range [0, 2]".to_string(),
            "DH2 index 1 site 0 neighbor 1=3 out of range [0, 2]".to_string(),
            format!(
                "DH2 index 1 site 1 neighbor 1={} out of range [0, 2]",
                i64::MAX
            ),
        ]
    );
    assert!(!result.is_valid);
}

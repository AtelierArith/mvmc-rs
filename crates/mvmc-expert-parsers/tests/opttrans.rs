mod common;
#[path = "../../../tests/support/historical_optimization_flags.rs"]
mod historical_optimization_flags;
#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;
use common::historical_kernel_model as parse_expert_mode_files;
use mvmc_expert_parsers::parsers::opttrans::{parse_opttrans_content, parse_opttrans_def};
use mvmc_expert_parsers::utils::parameter_init::init_parameter;
use mvmc_expert_parsers::utils::read_input_parameters::read_input_parameters;
use mvmc_expert_parsers::{
    parse_expert_mode_files_with_c_opt_trans, parse_expert_mode_files_with_opt_trans,
    ExpertModeData,
};
use num_complex::Complex64;
use sfmt19937::Sfmt19937Rng;
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/opttrans")
}

#[test]
fn opttrans_sector_count_and_real_only_component_flags_follow_julia() {
    let data = parse_expert_mode_files(root().join("namelist_valid.def")).unwrap();
    assert_eq!(data.n_qp_opt_trans, 2);
    assert_eq!(data.optimization_flags, vec![1, 0, 1, 0]);
}

#[test]
fn c_disabled_opttrans_ignores_definition_and_keeps_default_layout() {
    let data =
        parse_expert_mode_files_with_opt_trans(root().join("namelist_valid.def"), false).unwrap();
    assert_eq!(data.n_qp_opt_trans, 1);
    assert!(data.para_qp_opt_trans.is_empty());
    assert!(data.opt_trans.is_empty());
    assert!(data.qp_opt_trans.is_empty());
    assert!(data.qp_opt_trans_sgn.is_empty());
    assert!(data.optimization_flags.is_empty());
}

#[test]
fn c_enabled_opttrans_writes_consecutive_native_flags() {
    let data =
        parse_expert_mode_files_with_c_opt_trans(root().join("namelist_valid.def"), true).unwrap();
    assert_eq!(data.optimization_flags, vec![1, 1, 0, 0]);
}

#[test]
fn c_opttrans_flag_base_is_declared_projection_plus_slater_width() {
    let mut data = ExpertModeData::new();
    data.n_gutzwiller_idx = 3;
    data.opt_trans = vec![Complex64::new(0.5, 0.0), Complex64::new(0.75, 0.0)];
    mvmc_expert_parsers::set_opt_trans_c_opt_flags(&mut data);
    assert_eq!(&data.optimization_flags[..7], &[0, 0, 0, 1, 1, 0, 0]);
}

// Initial coefficients use only a few scale/divide/trig operations;
// 1e-14 covers their rounding while SFMT words and layout stay exact.
fn computed_values(values: impl IntoIterator<Item = Complex64>, expected: &str, label: &str) {
    numerical_comparison::assert_values_close(
        values.into_iter().flat_map(|v| [v.re, v.im]),
        numerical_comparison::hex_values(expected),
        1e-14,
        1e-14,
        label,
    );
}

fn bits(values: impl IntoIterator<Item = Complex64>, expected: &str, label: &str) {
    let actual: Vec<_> = values
        .into_iter()
        .flat_map(|v| [v.re.to_bits(), v.im.to_bits()])
        .collect();
    let expected: Vec<_> = expected
        .split_whitespace()
        .map(|s| u64::from_str_radix(s, 16).unwrap())
        .collect();
    assert_eq!(actual, expected, "{label}");
}

fn integers<T: std::str::FromStr>(line: &str) -> Vec<T>
where
    T::Err: std::fmt::Debug,
{
    line.split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect()
}

fn state<'a>(data: &ExpertModeData, lines: &mut impl Iterator<Item = &'a str>, label: &str) {
    assert_eq!(
        data.n_qp_opt_trans,
        lines.next().unwrap().parse::<i64>().unwrap(),
        "{label}"
    );
    // Initialization copies these parsed coefficients; it does not compute them.
    bits(
        data.para_qp_opt_trans.iter().copied(),
        lines.next().unwrap(),
        label,
    );
    bits(data.opt_trans.iter().copied(), lines.next().unwrap(), label);
    assert_eq!(
        data.qp_opt_trans
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>(),
        integers::<i64>(lines.next().unwrap()),
        "{label}"
    );
    assert_eq!(
        data.qp_opt_trans_sgn
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>(),
        integers::<i64>(lines.next().unwrap()),
        "{label}"
    );
}

#[test]
fn strict_definitions_errors_and_atomic_replacements_match_canonical_julia() {
    let fixture = std::fs::read_to_string(root().join("parser.txt")).unwrap();
    let mut lines = fixture.lines().filter(|line| !line.starts_with('#'));
    let mut cases = 0;
    while let Some(header) = lines.next() {
        let fields: Vec<_> = header.split_whitespace().collect();
        let mut data = ExpertModeData::new();
        data.modpara.nsite = 2;
        data.modpara.nmp_trans = -1;
        parse_opttrans_def(&mut data, root().join("valid.def")).unwrap();
        data.opt_trans.fill(Complex64::new(7.0, -9.0));
        data.modpara.nsite = fields[1].parse().unwrap();
        data.modpara.nmp_trans = fields[2].parse().unwrap();
        let path = root().join(fields[0]);
        let result = parse_opttrans_def(&mut data, &path);
        assert_eq!(result.is_ok(), fields[3] == "1", "{header}");
        let err = result
            .err()
            .map(|e| e.to_string().replace(path.to_str().unwrap(), fields[0]))
            .unwrap_or_default();
        assert_eq!(err, lines.next().unwrap(), "{header}");
        state(&data, &mut lines, header);
        let content = std::fs::read_to_string(path).unwrap();
        let parsed = parse_opttrans_content(
            &content,
            data.modpara.nsite,
            data.modpara.nmp_trans,
            fields[0],
        );
        assert_eq!(parsed.is_ok(), fields[3] == "1", "{header}");
        if let Ok(parsed) = parsed {
            assert_eq!(parsed.site_maps, data.qp_opt_trans, "{header}");
            assert_eq!(parsed.site_signs, data.qp_opt_trans_sgn, "{header}");
        }
        cases += 1;
    }
    assert_eq!(cases, 264);
}

#[test]
fn component_layout_initial_values_and_next_rng_state_match_common_julia_cases_and_c_order() {
    let fixture = std::fs::read_to_string(root().join("initial.txt")).unwrap();
    let records: Vec<_> = fixture
        .lines()
        .filter(|line| !line.starts_with('#'))
        .collect();
    assert_eq!(records.len() % 10, 0);
    let mut cases = 0;
    for record in records.as_chunks::<10>().0.iter() {
        let header = record[0];
        let fields: Vec<_> = header.split_whitespace().collect();
        // C reads ModPara before OptTrans regardless of namelist order.
        // Keep Julia's historical skipped-definition records unchanged;
        // compare these reordered inputs to the equivalent valid layout.
        let expected = if fields[0] == "before_modpara" {
            records
                .as_chunks::<10>()
                .0
                .iter()
                .find(|row| row[0] == format!("layout {}", fields[1]))
                .unwrap()
        } else {
            record
        };
        let mut lines = expected[1..].iter().copied();
        let mut data =
            parse_expert_mode_files(root().join(format!("namelist_{}.def", fields[0]))).unwrap();
        match fields[1] {
            "complex" => data.modpara.complex_flag = 1,
            "inactive" => data.optimization_flags.fill(0),
            "empty_para" => data.para_qp_opt_trans.clear(),
            "parsed" => {}
            _ => panic!("unknown mode: {header}"),
        }
        data.opt_trans.fill(Complex64::new(7.0, -9.0));
        let flags = historical_optimization_flags::c_orbital_representation(
            &data,
            integers::<i64>(lines.next().unwrap()),
        );
        // Julia stored these raw positive flags as bool. This old workload
        // checks initialization (>0); raw integer assembly has native C tests.
        assert_eq!(
            data.optimization_flags
                .iter()
                .map(|&v| i64::from(v > 0))
                .collect::<Vec<_>>(),
            flags,
            "{header}"
        );
        assert_eq!(
            data.count_opt_trans_parameters(),
            lines.next().unwrap().parse::<usize>().unwrap(),
            "{header}"
        );
        let mut rng = Sfmt19937Rng::new(11272);
        init_parameter(&mut data, &mut rng);
        state(&data, &mut lines, header);
        let mut values = data.projection_parameters();
        data.visit_rbm_terms_mut(|_, term| values.push(term.value()));
        values.extend(
            data.orbital_terms
                .iter()
                .map(|t| data.slater_params[t.idx as usize]),
        );
        computed_values(values, lines.next().unwrap(), header);
        let historical_words = integers::<u32>(lines.next().unwrap());
        let expected_words = if data.modpara.n_orbital_idx == 4 {
            common::declared_slater_rng(&data)
        } else {
            historical_words
        };
        assert_eq!(expected_words.len(), 624);
        let next: Vec<_> = (0..624).map(|_| rng.gen_rand32()).collect();
        assert_eq!(next, expected_words, "{header}");
        cases += 1;
    }
    assert_eq!(cases, 44);
}

#[test]
fn strict_complex_overlays_commit_atomically_and_preserve_initial_weights() {
    let fixture = std::fs::read_to_string(root().join("overlays.txt")).unwrap();
    let mut lines = fixture.lines().filter(|line| !line.starts_with('#'));
    while let Some(header) = lines.next() {
        let fields: Vec<_> = header.split_whitespace().collect();
        let mut data = parse_expert_mode_files(root().join("namelist_valid.def")).unwrap();
        if fields[0] == "inactive" {
            data.opt_trans.clear();
        }
        let path = root().join(if matches!(fields[0], "valid" | "inactive") {
            "overlay.def".to_owned()
        } else {
            format!("overlay_{}.def", fields[0])
        });
        let result = read_input_parameters(
            &mut data,
            root().join(format!("namelist_overlay_{}.def", fields[0])),
        );
        assert_eq!(result.is_ok(), fields[1] == "1", "{header}");
        let err = result
            .err()
            .map(|e| {
                e.replace(
                    path.to_str().unwrap(),
                    path.file_name().unwrap().to_str().unwrap(),
                )
            })
            .unwrap_or_default();
        assert_eq!(err, lines.next().unwrap(), "{header}");
        bits(data.opt_trans, lines.next().unwrap(), header);
        bits(data.para_qp_opt_trans, lines.next().unwrap(), header);
    }
}

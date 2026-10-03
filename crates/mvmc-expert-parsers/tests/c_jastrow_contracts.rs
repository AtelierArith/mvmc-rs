//! Actual-C directional Jastrow results, consumed without a C/toolbox dependency.
use mvmc_expert_parsers::parsers::jastrow::parse_jastrow_content;
use mvmc_expert_parsers::utils::opt_flag::set_projection_opt_flags;
use mvmc_expert_parsers::{ExpertModeData, GutzwillerTerm};
use num_complex::Complex64;

const CONTRACTS: &str = include_str!("../../../tests/fixtures/jastrow/c_reader_contracts.txt");

fn records() -> Vec<&'static str> {
    CONTRACTS
        .lines()
        .filter(|line| !line.starts_with('#'))
        .collect()
}

#[test]
fn directional_jastrow_mapping_matches_actual_c_reader() {
    let rows = records();
    assert_eq!(rows.len(), 144 * 4);
    let mut checked = 0;
    for record in rows.as_chunks::<4>().0.iter() {
        let header: Vec<_> = record[0].split_whitespace().collect();
        if header[6] != "0" {
            continue;
        }
        let nsite = header[1].parse().unwrap();
        let section = parse_jastrow_content(&record[1].replace('|', "\n"), nsite).unwrap();
        let (matrix, width) = (section.idx_matrix, section.n_jastrow_idx);
        assert_eq!(
            section.terms.len(),
            width as usize,
            "{} declared coefficient storage",
            header[0]
        );
        let expected: Vec<i64> = record[2]
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        assert_eq!(
            matrix.into_iter().flatten().collect::<Vec<_>>(),
            expected,
            "{} direction/index assignments",
            header[0]
        );
        assert_eq!(
            width,
            header[3].parse::<i64>().unwrap(),
            "{} declared width",
            header[0]
        );
        checked += 1;
    }
    assert_eq!(checked, 69);
}

#[test]
fn jastrow_flags_preserve_read_order_signed_values_and_nonzero_offset() {
    let rows = records();
    let mut checked = 0;
    for record in rows.as_chunks::<4>().0.iter() {
        let header: Vec<_> = record[0].split_whitespace().collect();
        if header[6] != "0" {
            continue;
        }
        let section =
            parse_jastrow_content(&record[1].replace('|', "\n"), header[1].parse().unwrap())
                .unwrap();
        let mut data = ExpertModeData::new();
        data.n_gutzwiller_idx = 2;
        data.gutzwiller_terms = (0..2)
            .map(|site| GutzwillerTerm {
                site,
                value: Complex64::new(0.0, 0.0),
                is_complex: false,
            })
            .collect();
        data.n_jastrow_idx = section.n_jastrow_idx;
        data.jastrow_terms = section.terms;
        set_projection_opt_flags(
            &mut data,
            &[(0, 3), (1, -2)].into(),
            &section.opt_flags,
            false,
            section.is_complex,
        );
        let expected: Vec<i64> = record[3]
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        assert_eq!(
            data.optimization_flags, expected,
            "{} raw flags and prefix",
            header[0]
        );
        checked += 1;
    }
    assert_eq!(checked, 69);
}

#[test]
fn namelist_jastrow_geometry_and_declared_flag_count_match_c_acceptance() {
    use mvmc_expert_parsers::parse_expert_mode_files;
    use std::fs;
    let rows = records();
    let dir = std::env::temp_dir().join(format!("mvmc-c-jastrow-contracts-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    fs::write(
        dir.join("namelist.def"),
        "Jastrow j.def\nGutzwiller g.def\nModPara modpara.def\n",
    )
    .unwrap();
    let mut failures = Vec::new();
    for record in rows.as_chunks::<4>().0.iter() {
        let header: Vec<_> = record[0].split_whitespace().collect();
        let nsite: usize = header[1].parse().unwrap();
        let expected = header[6] == "0";
        fs::write(
            dir.join("modpara.def"),
            format!("Nsite {nsite}\nNElec 1\nNMPTrans -1\n"),
        )
        .unwrap();
        let maps: String = (0..nsite).map(|site| format!("{site} 0\n")).collect();
        fs::write(
            dir.join("g.def"),
            format!("===\nWidth 2\nComplex 0\n===\n===\n{maps}99 3\n99 -2\n"),
        )
        .unwrap();
        fs::write(dir.join("j.def"), record[1].replace('|', "\n")).unwrap();
        let data = parse_expert_mode_files(dir.join("namelist.def")).unwrap();
        if data.input_errors.is_empty() != expected {
            failures.push(format!(
                "{}: Rust={}, C={expected}",
                header[0],
                data.input_errors.is_empty()
            ));
        } else if expected {
            assert_eq!(
                data.n_jastrow_idx,
                header[3].parse::<i64>().unwrap(),
                "{}",
                header[0]
            );
            assert_eq!(
                data.jastrow_terms.len(),
                data.n_jastrow_idx as usize,
                "{}",
                header[0]
            );
            let expected_matrix: Vec<i64> = record[2]
                .split_whitespace()
                .map(|v| v.parse().unwrap())
                .collect();
            assert_eq!(
                data.jastrow_idx.into_iter().flatten().collect::<Vec<_>>(),
                expected_matrix,
                "{}",
                header[0]
            );
            let expected_flags: Vec<i64> = record[3]
                .split_whitespace()
                .map(|v| v.parse().unwrap())
                .collect();
            assert_eq!(data.optimization_flags, expected_flags, "{}", header[0]);
        }
    }
    fs::remove_dir_all(dir).unwrap();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn malformed_jastrow_scans_and_unsafe_indices_have_bounded_rust_errors() {
    // C unchecked integer scans and invalid parameter dimensions are excluded
    // from native execution; these assertions are Rust safety guarantees.
    for maps in [
        "0 1 -1 1 0 0",
        "0 1 1 1 0 0",
        "0 1 x 1 0 0",
        "0 1 0 1 0 2147483648",
    ] {
        let text = format!("===\nWidth 1\nComplex 0\n===\n===\n{maps}\n0 1\n");
        assert!(parse_jastrow_content(&text, 2).is_err(), "{maps}");
    }
    for flags in ["x 1", "0 x", "0", "0 2147483648"] {
        let text = format!("===\nWidth 1\nComplex 0\n===\n===\n0 1 0 1 0 0\n{flags}\n");
        assert!(parse_jastrow_content(&text, 2).is_err(), "{flags}");
    }
    for width in ["-1", "2147483648", "illegal"] {
        let text = format!("===\nWidth {width}\nComplex 0\n===\n===\n0 1 0 1 0 0\n0 1\n");
        assert!(parse_jastrow_content(&text, 2).is_err(), "{width}");
    }
    for nsite in [0, 1, -1, i64::MAX] {
        let text = "===\nWidth 1\nComplex 0\n===\n===\n0 1 0 1 0 0\n0 1\n";
        assert!(parse_jastrow_content(text, nsite).is_err(), "{nsite}");
    }
    assert!(parse_jastrow_content("0 1 0\n1 0 0\n0 1\n", 2).is_err());
}

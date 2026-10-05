//! Actual-C InterAll acceptance, scan carry and coefficient-bit expectations.
use mvmc_expert_parsers::parse_expert_mode_files;
use mvmc_expert_parsers::parsers::interall::parse_interall_content;
use std::fs;

#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;

#[test]
fn interall_headers_counts_sites_spins_and_scan_values_match_native_c() {
    let fixture = include_str!("../../../tests/fixtures/interall/c_reader.txt");
    let rows: Vec<_> = fixture
        .lines()
        .filter(|row| !row.starts_with('#'))
        .collect();
    let directory =
        std::env::temp_dir().join(format!("mvmc-c-interall-reader-{}", std::process::id()));
    fs::create_dir_all(&directory).unwrap();
    let mut failures = Vec::new();
    let mut accepted = 0;
    for record in rows.as_chunks::<3>().0 {
        let header: Vec<_> = record[0].split_whitespace().collect();
        let name = header[0];
        let expected = header[3] == "1";
        accepted += usize::from(expected);
        let payload: Vec<_> = record[1]
            .as_bytes()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect();
        fs::write(directory.join("interall.def"), payload).unwrap();
        fs::write(
            directory.join("modpara.def"),
            format!(
                "--------------------\nModel_Parameters 0\n--------------------\nVMC_Cal_Parameters\n--------------------\nCDataFileHead zvo\nCParaFileHead zqp\n--------------------\nNsite {}\nNe 1\nNMPTrans 1\n{}",
                header[1],
                // C modpara rejects an explicit `2Sz -1`; FSZ (-1) is the default.
                if header[2] == "-1" { String::new() } else { format!("2Sz {}\n", header[2]) }
            ),
        )
        .unwrap();
        fs::write(
            directory.join("namelist.def"),
            "ModPara modpara.def\nInterAll interall.def\n",
        )
        .unwrap();
        let data = parse_expert_mode_files(directory.join("namelist.def")).unwrap();
        let actual = data.input_errors.is_empty();
        if actual != expected {
            failures.push(format!(
                "{name}: Rust acceptance={actual}, C={expected}; {:?}",
                data.input_errors
            ));
            continue;
        }
        if !expected {
            continue;
        }
        let expected_terms: Vec<_> = record[2].split('|').filter(|row| !row.is_empty()).collect();
        if data.inter_all_terms.len() != expected_terms.len() {
            failures.push(format!(
                "{name}: Rust terms={}, C={}",
                data.inter_all_terms.len(),
                expected_terms.len()
            ));
            continue;
        }
        for (index, (term, row)) in data.inter_all_terms.iter().zip(expected_terms).enumerate() {
            let fields: Vec<_> = row.split_whitespace().collect();
            let indices = [
                term.site0, term.spin0, term.site1, term.spin1, term.site2, term.spin2, term.site3,
                term.spin3,
            ];
            let expected_indices: Vec<_> = fields[..8]
                .iter()
                .map(|value| value.parse::<i64>().unwrap())
                .collect();
            let expected_bits = [
                u64::from_str_radix(fields[8], 16).unwrap(),
                u64::from_str_radix(fields[9], 16).unwrap(),
            ];
            if indices.as_slice() != expected_indices
                || ![term.value.re.to_bits(), term.value.im.to_bits()]
                    .into_iter()
                    .zip(expected_bits)
                    .all(|(actual, expected)| {
                        numerical_comparison::arithmetic_bits_match(actual, expected)
                    })
            {
                failures.push(format!(
                    "{name} row {index}: Rust={indices:?} {:016x} {:016x}, C={row}",
                    term.value.re.to_bits(),
                    term.value.im.to_bits()
                ));
            }
            if term.is_complex != (f64::from_bits(expected_bits[1]).abs() > 1e-14) {
                failures.push(format!(
                    "{name} row {index}: coefficient classification differs"
                ));
            }
        }
    }
    fs::remove_dir_all(directory).unwrap();
    assert_eq!(rows.len(), 463 * 3);
    assert_eq!(accepted, 411);
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures
            .iter()
            .take(20)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn unsafe_interall_indices_and_dimensions_have_bounded_rust_errors() {
    // Native C does not check these integer/spin ranges. Do not execute its
    // undefined scans or unsafe downstream accesses to invent a parity result.
    for body in [
        "2147483648 0 0 0 0 1 0 1 1 0",
        "0 2147483648 0 0 0 1 0 1 1 0",
        "0 2 0 2 0 1 0 1 1 0",
        "0 -1 0 -1 0 1 0 1 1 0",
    ] {
        let text = format!("===\nNInterAll 1\n===\n===\n===\n{body}\n");
        assert!(parse_interall_content(&text, 4, -1).is_err(), "{body}");
    }
    let valid = "===\nNInterAll 1\n===\n===\n===\n0 0 0 0 0 1 0 1 1 0\n";
    for site_count in [-1, 0, i64::MAX] {
        assert!(parse_interall_content(valid, site_count, -1).is_err());
    }
    for count in ["-1", "2147483648"] {
        assert!(parse_interall_content(
            &valid.replace("NInterAll 1", &format!("NInterAll {count}")),
            4,
            -1
        )
        .is_err());
    }
}

#[test]
fn generic_loader_rejects_duplicate_keyword_and_reports_single_entry_errors() {
    let directory =
        std::env::temp_dir().join(format!("mvmc-interall-atomic-{}", std::process::id()));
    fs::create_dir_all(&directory).unwrap();
    let valid = "===\nNInterAll 1\n===\n===\n===\n0 0 0 0 0 1 0 1 1e-3 -2e-3\n";
    fs::write(directory.join("good.def"), valid).unwrap();
    fs::write(
        directory.join("bad.def"),
        valid.replace("NInterAll 1", "NInterAll 2"),
    )
    .unwrap();
    fs::write(
        directory.join("modpara.def"),
        "--------------------\nModel_Parameters 0\n--------------------\nVMC_Cal_Parameters\n--------------------\nCDataFileHead zvo\nCParaFileHead zqp\n--------------------\nNsite 4\nNe 1\nNMPTrans 1\n2Sz 0\n",
    )
    .unwrap();
    fs::write(
        directory.join("namelist.def"),
        "InterAll good.def\nInterAll bad.def\nModPara modpara.def\n",
    )
    .unwrap();
    let error = parse_expert_mode_files(directory.join("namelist.def")).unwrap_err();
    let mvmc_expert_parsers::ParseError::InvalidInput { message } = error else {
        panic!("duplicate keyword must be InvalidInput: {error:?}");
    };
    assert!(message.contains("duplicate keyword InterAll"));
    fs::write(
        directory.join("namelist.def"),
        "InterAll good.def\nModPara modpara.def\n",
    )
    .unwrap();
    let data = parse_expert_mode_files(directory.join("namelist.def")).unwrap();
    assert_eq!(
        data.inter_all_terms,
        parse_interall_content(valid, 4, 0).unwrap()
    );
    assert!(data.input_errors.is_empty());
    fs::write(
        directory.join("namelist.def"),
        "InterAll bad.def\nModPara modpara.def\n",
    )
    .unwrap();
    let invalid = parse_expert_mode_files(directory.join("namelist.def")).unwrap();
    assert!(invalid.inter_all_terms.is_empty());
    assert_eq!(invalid.input_errors.len(), 1);
    assert!(invalid.input_errors[0].contains("InterAll"));
    assert!(invalid.input_errors[0].contains("row count"));
    fs::remove_dir_all(directory).unwrap();
}

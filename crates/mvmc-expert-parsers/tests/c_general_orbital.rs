//! Actual C General reader expectations, independent of the C toolbox.
use mvmc_expert_parsers::parse_expert_mode_files;
use mvmc_expert_parsers::parsers::orbital::{parse_orbital_content, OrbitalKind};
use std::fs;

#[test]
fn complete_six_column_general_rows_flags_and_matrices_match_c() {
    let records: Vec<_> =
        include_str!("../../../tests/fixtures/orbital_general/c_general_reader.txt")
            .lines()
            .filter(|line| !line.starts_with('#'))
            .collect();
    assert_eq!(records.len(), 5 * 184);
    let dir = std::env::temp_dir().join(format!("mvmc-c-general-reader-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let mut failures = Vec::new();
    let mut accepted = 0;
    for record in records.as_chunks::<5>().0.iter() {
        let header: Vec<_> = record[0].split_whitespace().collect();
        let nsite: usize = header[1].parse().unwrap();
        let expected = header[3] == "1" && header[6] == "0";
        accepted += usize::from(expected);
        fs::write(dir.join("general.def"), record[1].replace('|', "\n")).unwrap();
        fs::write(
            dir.join("modpara.def"),
            format!(
                "--------------------\nModel_Parameters 0\n--------------------\nVMC_Cal_Parameters\n--------------------\nCDataFileHead zvo\nCParaFileHead zqp\n--------------------\nNsite {nsite}\nNe 2\nNSPGaussLeg 1\nNMPTrans {}2\n",
                if header[2] == "1" { "-" } else { "" }
            ),
        )
        .unwrap();
        fs::write(
            dir.join("namelist.def"),
            "ModPara modpara.def\nOrbitalGeneral general.def\n",
        )
        .unwrap();
        let result = parse_expert_mode_files(dir.join("namelist.def"));
        let actual = result
            .as_ref()
            .is_ok_and(|data| data.input_errors.is_empty());
        if actual != expected {
            failures.push(format!("{}: Rust={actual}, C={expected}", header[0]));
            continue;
        }
        if !expected {
            continue;
        }
        let data = result.unwrap();
        let integers = |text: &str| {
            text.split_whitespace()
                .map(|field| field.parse::<i64>().unwrap())
                .collect::<Vec<_>>()
        };
        let indices: Vec<_> = data
            .orbital_idx_matrix
            .as_ref()
            .unwrap()
            .iter()
            .flatten()
            .copied()
            .collect();
        let signs: Vec<_> = data
            .orbital_sgn_matrix
            .as_ref()
            .unwrap()
            .iter()
            .flatten()
            .copied()
            .collect();
        let flags: Vec<_> = data.optimization_flags.iter().step_by(2).copied().collect();
        if indices != integers(record[2])
            || signs != integers(record[3])
            || flags != integers(record[4])
        {
            failures.push(format!(
                "{}: C index/sign/row-order flags differ",
                header[0]
            ));
        }
        assert_eq!(
            data.modpara.n_orbital_idx,
            header[4].parse::<i64>().unwrap(),
            "{} declared width",
            header[0]
        );
        assert_eq!(data.i_flg_orbital_general, 1);
        assert_eq!(data.orbital_terms.len(), 2 * nsite * nsite - nsite);
    }
    fs::remove_dir_all(dir).unwrap();
    assert_eq!(accepted, 104);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn malformed_general_fields_and_unsafe_indices_return_bounded_errors() {
    // Native C does unchecked sscanf/spin/index accesses on these inputs.
    // Assert safe Rust diagnostics, not a fabricated C rejection result.
    let valid = include_str!("../../../tests/fixtures/c_orbital_inputs/general_three.def");
    for replacement in [
        "0 3 0 1",
        "0 0 1",
        "0 2 1 0 0 1",
        "0 -1 1 0 0 1",
        "0 0 1 0 15 1",
        "0 0 1 0 -1 1",
        "0 0 1 0 x 1",
    ] {
        let mut lines: Vec<_> = valid.lines().collect();
        lines[5] = replacement;
        assert!(
            parse_orbital_content(&lines.join("\n"), 3, OrbitalKind::General).is_err(),
            "{replacement}"
        );
    }
    assert!(parse_orbital_content(valid, 0, OrbitalKind::General).is_err());
    assert!(parse_orbital_content(valid, i64::MAX, OrbitalKind::General).is_err());
}

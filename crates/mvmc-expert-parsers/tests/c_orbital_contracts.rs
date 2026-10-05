//! Checked-in actual-C AP/P reader results; no toolbox/compiler dependency.
use mvmc_expert_parsers::parse_expert_mode_files;
use std::fs;

#[test]
fn orbital_headers_mapping_rows_and_flag_fields_match_c_acceptance() {
    let rows: Vec<_> =
        include_str!("../../../tests/fixtures/orbital_general/c_reader_contracts.txt")
            .lines()
            .filter(|line| !line.starts_with('#'))
            .collect();
    assert_eq!(rows.len(), 92 * 2);
    let dir = std::env::temp_dir().join(format!("mvmc-c-orbital-contracts-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let mut failures = Vec::new();
    let mut accepted = 0;
    for record in rows.as_chunks::<2>().0.iter() {
        let fields: Vec<_> = record[0].split_whitespace().collect();
        let nsite: usize = fields[2].parse().unwrap();
        let expected = fields[3] == "1" && fields[6] == "0";
        accepted += usize::from(expected);
        fs::write(
            dir.join("modpara.def"),
            format!("--------------------\nModel_Parameters 0\n--------------------\nVMC_Cal_Parameters\n--------------------\nCDataFileHead zvo\nCParaFileHead zqp\n--------------------\nNsite {nsite}\nNe 1\nNMPTrans -1\n"),
        )
        .unwrap();
        fs::write(dir.join("tested.def"), record[1].replace('|', "\n")).unwrap();
        let mut namelist = "ModPara modpara.def\n".to_owned();
        if fields[1] == "P" {
            // The C P reader is supplied an AP prefix of seven slots. A valid
            // complete AP definition also supplies that prefix to Rust.
            let mappings: String = (0..nsite)
                .flat_map(|i| (0..nsite).map(move |j| format!("{i} {j} 0 1\n")))
                .collect();
            let flags: String = (0..7).map(|i| format!("{i} 1\n")).collect();
            fs::write(
                dir.join("ap.def"),
                format!("===\nNOrbitalIdx 7\nComplexType 0\n===\n===\n{mappings}{flags}"),
            )
            .unwrap();
            namelist.push_str("Orbital ap.def\nOrbitalParallel tested.def\n");
        } else {
            namelist.push_str("OrbitalAntiParallel tested.def\n");
        }
        fs::write(dir.join("namelist.def"), namelist).unwrap();
        let result = parse_expert_mode_files(dir.join("namelist.def"));
        let actual = result
            .as_ref()
            .is_ok_and(|data| data.input_errors.is_empty());
        if actual != expected {
            failures.push(format!("{}: Rust={actual}, C={expected}", fields[0]));
        } else if expected {
            let data = result.unwrap();
            let width: i64 = fields[4].parse().unwrap();
            let expected_width = if fields[1] == "P" {
                7 + 2 * width
            } else {
                width
            };
            if data.modpara.n_orbital_idx != expected_width {
                failures.push(format!(
                    "{}: Rust width={}, C width={expected_width}",
                    fields[0], data.modpara.n_orbital_idx
                ));
            }
            let start = if fields[1] == "P" { 14 } else { 0 };
            let real_flags: Vec<_> = data.optimization_flags[start..]
                .iter()
                .step_by(2)
                .copied()
                .collect();
            let expected_flags: Vec<i64> = fields[7..]
                .iter()
                .map(|field| field.parse().unwrap())
                .collect();
            assert_eq!(
                real_flags, expected_flags,
                "{} row-order real flags",
                fields[0]
            );
        }
    }
    fs::remove_dir_all(dir).unwrap();
    assert_eq!(accepted, 47);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

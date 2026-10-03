//! Checked-in actual-C results; no toolbox or C runtime dependency.
use mvmc_expert_parsers::parsers::gutzwiller::parse_gutzwiller_content;
use mvmc_expert_parsers::utils::opt_flag::set_projection_opt_flags;
use mvmc_expert_parsers::ExpertModeData;

const CONTRACTS: &str = include_str!("../../../tests/fixtures/gutzwiller/c_reader_contracts.txt");

fn integers(text: &str) -> Vec<i64> {
    text.split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect()
}

#[test]
fn accepted_c_gutzwiller_rows_preserve_declared_slots_maps_and_raw_component_flags() {
    let rows: Vec<_> = CONTRACTS
        .lines()
        .filter(|line| !line.starts_with('#'))
        .collect();
    assert_eq!(rows.len(), 144 * 4);
    let mut accepted = 0;
    for record in rows.as_chunks::<4>().0.iter() {
        let header: Vec<_> = record[0].split_whitespace().collect();
        if header[6] != "0" {
            continue;
        }
        let name = header[0];
        let nsite: usize = header[1].parse().unwrap();
        let width: i64 = header[3].parse().unwrap();
        let section = parse_gutzwiller_content(&record[1].replace('|', "\n"), nsite as i64)
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        assert_eq!(section.n_gutzwiller_idx, width, "{name}");
        assert_eq!(section.terms.len(), width as usize, "{name}");
        assert_eq!(
            (0..nsite)
                .map(|i| *section.site_idx_map.get(&(i as i64)).unwrap_or(&0))
                .collect::<Vec<_>>(),
            integers(record[2]),
            "{name}"
        );
        let mut data = ExpertModeData::new();
        data.n_gutzwiller_idx = width;
        data.gutzwiller_terms = section.terms;
        set_projection_opt_flags(
            &mut data,
            &section.opt_flags,
            &Default::default(),
            section.is_complex,
            false,
        );
        assert_eq!(data.optimization_flags, integers(record[3]), "{name}");
        accepted += 1;
    }
    assert_eq!(accepted, 72);
}

#[test]
fn complete_gutzwiller_contract_is_enforced_by_namelist_loading() {
    use mvmc_expert_parsers::parse_expert_mode_files;
    use std::fs;
    let rows: Vec<_> = CONTRACTS
        .lines()
        .filter(|line| !line.starts_with('#'))
        .collect();
    let dir = std::env::temp_dir().join(format!("mvmc-c-gutz-contracts-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    fs::write(
        dir.join("namelist.def"),
        "Gutzwiller g.def\nModPara modpara.def\n",
    )
    .unwrap();
    let mut rejected = 0;
    for record in rows.as_chunks::<4>().0.iter() {
        let header: Vec<_> = record[0].split_whitespace().collect();
        let expected = header[6] == "0";
        fs::write(
            dir.join("modpara.def"),
            format!("Nsite {}\nNElec 1\nNMPTrans -1\n", header[1]),
        )
        .unwrap();
        fs::write(dir.join("g.def"), record[1].replace('|', "\n")).unwrap();
        let data = parse_expert_mode_files(dir.join("namelist.def")).unwrap();
        assert_eq!(
            data.input_errors.is_empty(),
            expected,
            "{}: {:?}",
            header[0],
            data.input_errors
        );
        if expected {
            assert_eq!(data.gutzwiller_idx, integers(record[2]), "{}", header[0]);
            assert_eq!(
                data.optimization_flags,
                integers(record[3]),
                "{}",
                header[0]
            );
        } else {
            assert!(data.gutzwiller_terms.is_empty());
            assert!(data.gutzwiller_idx.is_empty());
            assert_eq!(data.n_gutzwiller_idx, 0);
            rejected += 1;
        }
    }
    fs::remove_dir_all(dir).unwrap();
    assert_eq!(rejected, 72);
}

#[test]
fn unsafe_gutzwiller_indices_and_malformed_scans_have_bounded_diagnostics() {
    // These are Rust safety diagnostics, not C unchecked-scan parity claims.
    for body in [
        "-1 0 0 1",
        "2 0 0 1",
        "0 -1 0 1",
        "0 1 0 1",
        "0 0 x 1",
        "0 0 0 x",
        "0 0 0",
        "0 0 0 2147483648",
    ] {
        let text = format!("===\nWidth 1\nComplex 0\n===\n===\n{body}\n");
        assert!(parse_gutzwiller_content(&text, 1).is_err(), "{body}");
    }
    for header in ["0", "-1", "2147483648", "illegal"] {
        let text = format!("===\nWidth {header}\nComplex 0\n===\n===\n0 0 0 1\n");
        assert!(parse_gutzwiller_content(&text, 1).is_err(), "{header}");
    }
    assert!(parse_gutzwiller_content("0 0\n0 1\n", 1).is_err());
    assert!(parse_gutzwiller_content("===\nWidth 1\nComplex 0\n===\n===\n0 0 0 1\n", 0).is_err());
}

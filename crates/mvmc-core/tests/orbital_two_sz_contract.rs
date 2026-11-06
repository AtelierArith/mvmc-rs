//! C `ReadDefFileNInt` orbital/`2Sz` admission (readdef.c:689-701, #719).
//!
//! Normal orbitals need `2Sz = 0`; `2Sz != 0`, including the `-1` default of a
//! ModPara without `2Sz`, needs general orbitals, and then `2Sz` must be even
//! or `-1`. Both CPU entry points (Opt and PhysCal) apply the C contract.

use std::fs;
use std::path::{Path, PathBuf};

use mvmc_core::validation::{validate_orbital_two_sz, validate_para_opt, validate_phys_cal};
use mvmc_expert_parsers::{parse_expert_mode_files, ExpertModeData};

mod support;

struct TempInput(PathBuf);

impl Drop for TempInput {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Copy a reference input and replace its `2Sz` line (`None` removes it).
fn input_with_two_sz(model: &str, two_sz: Option<i64>) -> (TempInput, ExpertModeData) {
    let root = support::julia_mvmc_root()
        .unwrap_or_else(|| support::missing_fixture("orbital-two-sz", "Julia-mVMC checkout"));
    let source = root
        .join("test/integration/reference")
        .join(model)
        .join("inputs");
    let id = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "mvmc-rs-two-sz-{model}-{}-{id}",
        std::process::id()
    ));
    fs::create_dir(&dir).unwrap();
    let guard = TempInput(dir.clone());
    for entry in fs::read_dir(&source).unwrap() {
        let path = entry.unwrap().path();
        if path.is_file() {
            fs::copy(&path, dir.join(path.file_name().unwrap())).unwrap();
        }
    }
    let modpara = fs::read_to_string(dir.join("modpara.def")).unwrap();
    let mut lines: Vec<String> = modpara
        .lines()
        .filter(|line| line.split_whitespace().next() != Some("2Sz"))
        .map(str::to_owned)
        .collect();
    if let Some(value) = two_sz {
        let at = lines
            .iter()
            .position(|line| line.split_whitespace().next() == Some("Nsite"))
            .expect("Nsite line");
        lines.insert(at + 1, format!("2Sz {value}"));
    }
    fs::write(dir.join("modpara.def"), lines.join("\n") + "\n").unwrap();
    let data = parse_expert_mode_files(Path::new(&dir).join("namelist.def")).unwrap();
    assert_eq!(data.modpara.two_sz, two_sz.unwrap_or(-1));
    (guard, data)
}

fn assert_both_entry_points_reject(data: &ExpertModeData, needle: &str) {
    for (entry, result) in [
        ("Opt", validate_para_opt(data)),
        ("PhysCal", validate_phys_cal(data)),
    ] {
        let error = result.expect_err(entry);
        assert!(error.contains(needle), "{entry}: {error}");
        assert!(error.contains("readdef.c"), "{entry}: {error}");
    }
}

#[test]
fn normal_orbitals_with_nonzero_two_sz_are_rejected() {
    let (_guard, data) = input_with_two_sz("hubbard_chain_real", Some(2));
    assert_eq!(data.i_flg_orbital_general, 0);
    assert_both_entry_points_reject(&data, "requires OrbitalParallel or OrbitalGeneral");
}

#[test]
fn normal_orbitals_with_omitted_two_sz_are_rejected() {
    let (_guard, data) = input_with_two_sz("hubbard_chain_real", None);
    assert_both_entry_points_reject(&data, "the default when ModPara omits 2Sz");
}

#[test]
fn normal_orbitals_with_zero_two_sz_are_accepted() {
    let (_guard, data) = input_with_two_sz("hubbard_chain_real", Some(0));
    validate_para_opt(&data).unwrap();
    validate_phys_cal(&data).unwrap();
}

#[test]
fn general_orbitals_with_odd_two_sz_are_rejected() {
    let (_guard, data) = input_with_two_sz("hubbard_chain_fsz", Some(1));
    assert_eq!(data.i_flg_orbital_general, 1);
    assert_both_entry_points_reject(&data, "must be an even number");
}

#[test]
fn general_orbitals_with_even_or_omitted_two_sz_are_accepted() {
    for two_sz in [None, Some(0), Some(2)] {
        let (_guard, data) = input_with_two_sz("hubbard_chain_fsz", two_sz);
        validate_para_opt(&data).unwrap_or_else(|e| panic!("{two_sz:?}: {e}"));
        validate_phys_cal(&data).unwrap_or_else(|e| panic!("{two_sz:?}: {e}"));
    }
}

#[test]
fn programmatic_two_sz_contract_follows_c_conditions() {
    let mut data = ExpertModeData::new();
    for (general, two_sz, ok) in [
        (0, 0, true),
        (0, -1, false),
        (0, 2, false),
        (0, -2, false),
        (1, 0, true),
        (1, -1, true),
        (1, 2, true),
        (1, -2, true),
        (1, 3, false),
        (1, -3, false),
    ] {
        data.i_flg_orbital_general = general;
        data.modpara.two_sz = two_sz;
        assert_eq!(
            validate_orbital_two_sz(&data).is_ok(),
            ok,
            "general={general} 2Sz={two_sz}"
        );
    }
}

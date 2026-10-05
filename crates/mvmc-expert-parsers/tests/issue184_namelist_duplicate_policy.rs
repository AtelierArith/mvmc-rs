//! C keyword-slot preflight at all public loaders, not metadata-utility deduplication.
use mvmc_expert_parsers::{
    parse_expert_mode_files, parse_expert_mode_files_with_c_opt_trans,
    parse_expert_mode_files_with_opt_trans, ExpertModeData, ParseError,
};
use std::{
    fs, io,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Input(PathBuf);
impl Input {
    fn new(namelist: &str) -> Self {
        let path = loop {
            let path = std::env::temp_dir().join(format!(
                "issue184-duplicate-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => break path,
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("exclusive input: {error}"),
            }
        };
        fs::write(path.join("namelist.def"), namelist).unwrap();
        fs::write(path.join("first.def"), "--------------------\nModel_Parameters 0\n--------------------\nVMC_Cal_Parameters\n--------------------\nCDataFileHead zvo\nCParaFileHead zqp\n--------------------\nNsite 2\nNe 1\n").unwrap();
        fs::write(path.join("second.def"), "--------------------\nModel_Parameters 0\n--------------------\nVMC_Cal_Parameters\n--------------------\nCDataFileHead zvo\nCParaFileHead zqp\n--------------------\nNsite 4\nNe 2\n").unwrap();
        Self(path)
    }
    fn parse(&self, entry: usize) -> Result<ExpertModeData, ParseError> {
        let path = self.0.join("namelist.def");
        match entry {
            0 => parse_expert_mode_files(path),
            1 => parse_expert_mode_files_with_opt_trans(path, false),
            2 => parse_expert_mode_files_with_c_opt_trans(path, false),
            _ => unreachable!(),
        }
    }
}
impl Drop for Input {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

fn assert_duplicate(error: ParseError, canonical: &str) {
    let ParseError::InvalidInput { message } = error else {
        panic!("duplicate preflight must return InvalidInput, got {error:?}");
    };
    assert!(
        message.contains(&format!("duplicate keyword {canonical}")),
        "{message}"
    );
}

#[test]
fn every_public_loader_rejects_same_and_mixed_case_c_keyword_slots() {
    for keyword in ["ModPara", "modpara", "mOdPaRa"] {
        let input = Input::new(&format!("ModPara first.def\n{keyword} second.def\n"));
        for entry in 0..3 {
            assert_duplicate(input.parse(entry).unwrap_err(), "ModPara");
            assert_eq!(
                fs::read_to_string(input.0.join("first.def")).unwrap(),
                "--------------------\nModel_Parameters 0\n--------------------\nVMC_Cal_Parameters\n--------------------\nCDataFileHead zvo\nCParaFileHead zqp\n--------------------\nNsite 2\nNe 1\n"
            );
            assert_eq!(
                fs::read_to_string(input.0.join("second.def")).unwrap(),
                "--------------------\nModel_Parameters 0\n--------------------\nVMC_Cal_Parameters\n--------------------\nCDataFileHead zvo\nCParaFileHead zqp\n--------------------\nNsite 4\nNe 2\n"
            );
        }
    }
}

#[test]
fn duplicate_preflight_precedes_missing_children_and_inactive_opttrans() {
    for (namelist, canonical) in [
        ("ModPara absent-a.def\nMODPARA absent-b.def\n", "ModPara"),
        ("OptTrans absent-a.def\nopttrans absent-b.def\n", "OptTrans"),
        (
            "InOptTrans absent-a.def\ninopttrans absent-b.def\n",
            "InOptTrans",
        ),
    ] {
        let input = Input::new(namelist);
        for entry in 0..3 {
            assert_duplicate(input.parse(entry).unwrap_err(), canonical);
        }
    }
}

#[test]
fn existing_aliases_share_slots_without_merging_distinct_c_orbital_keywords() {
    for (canonical, alias) in [
        ("DH2", "DoublonHolon2Site"),
        ("DH4", "DoublonHolon4Site"),
        ("TransSym", "QPTrans"),
    ] {
        let input = Input::new(&format!("{canonical} absent-a.def\n{alias} absent-b.def\n"));
        for entry in 0..3 {
            assert_duplicate(input.parse(entry).unwrap_err(), canonical);
        }
    }
    // Orbital and OrbitalAntiParallel are distinct native slots, not aliases.
    let input =
        Input::new("ModPara first.def\nOrbital absent-a.def\nOrbitalAntiParallel absent-b.def\n");
    for entry in 0..3 {
        let data = input.parse(entry).unwrap();
        assert_eq!(data.modpara.nsite, 2);
        assert_eq!(data.input_errors.len(), 2); // Missing children, not C-valid input.
    }
}

#[test]
fn case_insensitive_overlay_dispatch_uses_the_same_canonical_identity() {
    let input = Input::new("modpara first.def\ngutzwiller g.def\ningutzwiller in.def\n");
    fs::write(
        input.0.join("g.def"),
        "===\nNGutzwillerIdx 1\nComplexType 1\n===\n===\n0 0\n1 0\n0 1\n",
    )
    .unwrap();
    fs::write(
        input.0.join("in.def"),
        "===\nNGutzwillerIdx 1\nComplexType 1\n===\n===\n0 0.5 0.3\n",
    )
    .unwrap();
    for entry in 0..3 {
        let mut data = input.parse(entry).unwrap();
        assert!(data.input_errors.is_empty());
        mvmc_expert_parsers::utils::read_input_parameters::read_input_parameters(
            &mut data,
            input.0.join("namelist.def"),
        )
        .unwrap();
        assert_eq!(data.gutzwiller_terms[0].value.re, 0.5);
        assert_eq!(data.gutzwiller_terms[0].value.im, 0.3);
    }
}

#[test]
fn overlay_reader_rejects_mixed_case_duplicates_before_changing_parameters() {
    let input = Input::new("InGutzwiller absent-a.def\ningutzwiller absent-b.def\n");
    let mut data = ExpertModeData::new();
    let before = data.projection_parameters();
    let error = mvmc_expert_parsers::utils::read_input_parameters::read_input_parameters(
        &mut data,
        input.0.join("namelist.def"),
    )
    .unwrap_err();
    assert!(error.contains("duplicate keyword InGutzwiller"));
    assert_eq!(data.projection_parameters(), before);
}

#[test]
fn distinct_keywords_parse_case_insensitively_and_preserve_raw_metadata() {
    let input = Input::new("mOdPaRa first.def\nInGutzwiller absent.def\n");
    for entry in 0..3 {
        let data = input.parse(entry).unwrap();
        assert!(data.input_errors.is_empty());
        assert_eq!(data.modpara.nsite, 2);
        assert_eq!(data.namelist[0], ("mOdPaRa".into(), "first.def".into()));
    }
}

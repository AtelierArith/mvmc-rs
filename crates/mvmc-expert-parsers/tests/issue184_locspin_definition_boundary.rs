//! #267 native complete-buffer parsing, not sampling/model acceptance.
use mvmc_expert_parsers::{
    parse_expert_mode_files,
    parsers::locspin::{parse_locspin_content, parse_locspin_definition},
    ParseError,
};
use std::{
    fs, io,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
struct Input(PathBuf);
impl Input {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "mvmc-locspin-{}-{nonce}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn definition(&self, header: &str, body: &str) -> PathBuf {
        let path = self.0.join("locspin.def");
        fs::write(
            &path,
            format!("===\n{header}\n===\nsite scalar\n===\n{body}"),
        )
        .unwrap();
        path
    }
    fn values(&self, header: &str, body: &str, nsite: i64) -> (i64, Vec<(i64, i64)>) {
        let d = parse_locspin_definition(self.definition(header, body), nsite).unwrap();
        (
            d.nlocal_spin,
            d.terms
                .into_iter()
                .map(|t| (t.site, t.spin_value))
                .collect(),
        )
    }
}
impl Drop for Input {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn locspin_complete_permuted_table_is_site_indexed_not_input_order() {
    assert_eq!(
        Input::new().values("NlocalSpin 1", "2 0\n0 1\n1 0\n", 3),
        (1, vec![(0, 1), (1, 0), (2, 0)])
    );
}
#[test]
fn locspin_header_token_is_informational_and_not_body_or_one_count() {
    assert_eq!(
        Input::new().values("Arbitrary +7tail", "0 0\n1 0\n", 2),
        (7, vec![(0, 0), (1, 0)])
    );
    // Parser-only signed header; existing runtime policy remains separate.
    assert_eq!(Input::new().values("NlocalSpin -1", "0 1\n", 1).0, -1);
}
#[test]
fn locspin_complete_negative_and_nonbinary_scalars_are_not_normalized() {
    assert_eq!(
        Input::new().values("NlocalSpin 0", "0 -1\n1 2\n2 17\n", 3),
        (0, vec![(0, -1), (1, 2), (2, 17)])
    );
}
#[test]
fn locspin_scalar_omission_preserves_initialized_then_previous_integer() {
    assert_eq!(
        Input::new().values("NlocalSpin 0", "0\n1 -2\n2\n", 3),
        (0, vec![(0, 0), (1, -2), (2, -2)])
    );
}
#[test]
fn locspin_decimal_prefix_and_defined_stale_scalar_site_are_preserved() {
    let input = Input::new();
    assert_eq!(
        input.values("NlocalSpin 0", "0 2suffix\n1 bad\n2+3\n", 3),
        (0, vec![(0, 2), (1, 2), (2, 3)])
    );
    // First failed conversion preserves initialized site0/value0; the next
    // explicit site1 completes the whole native buffer, so no hole exists.
    assert_eq!(
        input.values("NlocalSpin 0", "\n1 2\n", 2),
        (0, vec![(0, 0), (1, 2)])
    );
}
#[test]
fn locspin_exact_nsite_body_and_complete_coverage_are_safety_boundaries() {
    let input = Input::new();
    for body in ["0 1\n", "0 1\n1 0\n2 0\n", "0 1\n0 0\n"] {
        assert_eq!(
            parse_locspin_definition(input.definition("NlocalSpin 0", body), 2)
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidData
        );
    }
}
#[test]
fn locspin_bounds_overflow_missing_header_reject_before_partial_publication() {
    let input = Input::new();
    for body in ["-1 0\n1 0\n", "2 0\n1 0\n", "0 2147483648\n1 0\n"] {
        assert_eq!(
            parse_locspin_definition(input.definition("NlocalSpin 0", body), 2)
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidData
        );
    }
    assert!(parse_locspin_definition(input.definition("NlocalSpin bad", "0 0\n1 0\n"), 2).is_err());
}
#[test]
fn public_locspin_loader_rejects_advertised_missing_or_incomplete_definition() {
    let input = Input::new();
    fs::write(input.0.join("modpara.def"), concat!("--------------------\nModel_Parameters 0\n--------------------\nVMC_Cal_Parameters\n--------------------\nCDataFileHead zvo\nCParaFileHead zqp\n--------------------\n", "NSite 2\nNe 1\n")).unwrap();
    let path = input.0.join("namelist.def");
    fs::write(&path, "ModPara modpara.def\nLocSpin locspin.def\n").unwrap();
    assert!(matches!(
        parse_expert_mode_files(&path),
        Err(ParseError::InvalidInput { .. })
    ));
    input.definition("Header 0", "1 2\n0 -1\n");
    let data = parse_expert_mode_files(&path).unwrap();
    assert_eq!(data.modpara.nlocspin, 0);
    assert_eq!(
        data.locspin_terms
            .iter()
            .map(|t| (t.site, t.spin_value))
            .collect::<Vec<_>>(),
        [(0, -1), (1, 2)]
    );
    input.definition("Header 0", "0 0\n");
    assert!(matches!(
        parse_expert_mode_files(&path),
        Err(ParseError::InvalidInput { .. })
    ));
}
#[test]
fn locspin_permissive_content_helper_remains_separate() {
    let content = "===\nNlocalSpin 0\n===\nsite scalar\n===\n0 -1\n1 2\ninvalid\n";
    assert_eq!(
        parse_locspin_content(content)
            .iter()
            .map(|t| (t.site, t.spin_value))
            .collect::<Vec<_>>(),
        [(1, 2)]
    );
}

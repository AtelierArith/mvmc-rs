//! F039/F041 Transfer boundary; C readdef.c1973-2001, SHA256
//! 6c53cb832f93d6cbfd7cea955fbb693738af5536b913d36af32b98eed38c32d9.
//! Independent literals only: no toolbox/reference program is read or invoked.

use std::fs;
use std::io;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use mvmc_expert_parsers::parsers::trans::{parse_trans_content, parse_trans_definition};
use mvmc_expert_parsers::{parse_expert_mode_files, ParseError};
use num_complex::Complex64;

struct Fixture(PathBuf);

impl Fixture {
    fn new() -> Self {
        static SEQUENCE: AtomicU64 = AtomicU64::new(0);
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "mvmc-transfer-boundary-{}-{nonce}-{}",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&directory).unwrap();
        Self(directory)
    }

    fn write(&self, count: usize, body: &str) -> PathBuf {
        let path = self.0.join("trans.def");
        fs::write(
            &path,
            format!("===\nNTransfer {count}\n===\nindices\n===\n{body}"),
        )
        .unwrap();
        path
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn transfer_definition_preserves_original_complex_and_spin_coordinates() {
    let fixture = Fixture::new();
    let path = fixture.write(2, "0 0 1 1 1.25 -0.5\n1 1 0 0 -0.75 0.125\n");
    let terms = parse_trans_definition(path, 2).unwrap();
    assert_eq!(terms.len(), 2);
    assert_eq!(terms[0].value, Complex64::new(1.25, -0.5));
    assert_eq!(terms[1].value, Complex64::new(-0.75, 0.125));
    assert_eq!((terms[0].site1, terms[0].site2), (0, 1));
    assert_eq!((terms[0].spin1.as_code(), terms[0].spin2.as_code()), (0, 1));
    assert_eq!((terms[1].spin1.as_code(), terms[1].spin2.as_code()), (1, 0));
}

#[test]
fn transfer_definition_reads_c_hexadecimal_coefficients_not_zero_fallbacks() {
    let fixture = Fixture::new();
    let path = fixture.write(1, "0 0 1 1 0x1.4p0 -0x1p-1\n");
    let terms = parse_trans_definition(path, 2).unwrap();
    assert_eq!(terms[0].value, Complex64::new(1.25, -0.5));
}

#[test]
fn transfer_zero_count_ignores_body_and_positive_count_is_exact() {
    let fixture = Fixture::new();
    assert!(
        parse_trans_definition(fixture.write(0, "not a record\n"), 2)
            .unwrap()
            .is_empty()
    );
    for (count, body) in [(2, "0 0 1 0 1 0\n"), (1, "0 0 1 0 1 0\n1 0 0 0 1 0\n")] {
        assert_eq!(
            parse_trans_definition(fixture.write(count, body), 2)
                .unwrap_err()
                .kind(),
            io::ErrorKind::InvalidData
        );
    }
}

#[test]
fn transfer_consumed_invalid_row_cannot_publish_partial_payload() {
    let fixture = Fixture::new();
    for invalid in [
        "-1 0 1 0 1 0",
        "0 0 2 0 1 0",
        "0 2 1 0 1 0",
        "0 0 1 0 rubbish 0",
        "0 0 1 0",
    ] {
        let path = fixture.write(2, &format!("0 0 1 0 1 0\n{invalid}\n"));
        assert_eq!(
            parse_trans_definition(path, 2).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
    }
}

#[test]
fn native_five_field_rows_retain_initialized_or_previous_imaginary_value() {
    // Native dImValue is initialized once at1977, outside the fgets loop;
    // sscanf1979-1981 leaves it unchanged after five successful conversions.
    let fixture = Fixture::new();
    let path = fixture.write(
        4,
        "0 0 1 0 1.25\n1 0 0 0 -0.75 -0.5\n0 1 1 1 0.125\n1 1 0 1 0x1p-1\n",
    );
    let terms = parse_trans_definition(path, 2).unwrap();
    assert_eq!(terms.len(), 4);
    for (term, expected) in terms.iter().zip([
        Complex64::new(1.25, 0.0),
        Complex64::new(-0.75, -0.5),
        Complex64::new(0.125, -0.5),
        Complex64::new(0.5, -0.5),
    ]) {
        assert_eq!(term.value, expected);
    }
}

#[test]
fn public_loader_routes_transfer_definition_errors_instead_of_silent_success() {
    let fixture = Fixture::new();
    fs::write(fixture.0.join("modpara.def"), concat!("--------------------\nModel_Parameters 0\n--------------------\nVMC_Cal_Parameters\n--------------------\nCDataFileHead zvo\nCParaFileHead zqp\n--------------------\n", "NSite 2\nNe 1\n")).unwrap();
    let namelist = fixture.0.join("namelist.def");
    fs::write(&namelist, "ModPara modpara.def\nTrans trans.def\n").unwrap();
    fixture.write(1, "0 0 1 0 0x1.4p0 -0x1p-1\n");
    let data = parse_expert_mode_files(&namelist).unwrap();
    assert_eq!(data.transfer_terms[0].value, Complex64::new(1.25, -0.5));
    fixture.write(2, "0 0 1 0 1 0\n");
    assert!(matches!(
        parse_expert_mode_files(&namelist),
        Err(ParseError::InvalidInput { .. })
    ));
}

#[test]
fn public_loader_missing_transfer_is_required_not_an_incomplete_success() {
    let fixture = Fixture::new();
    fs::write(fixture.0.join("modpara.def"), concat!("--------------------\nModel_Parameters 0\n--------------------\nVMC_Cal_Parameters\n--------------------\nCDataFileHead zvo\nCParaFileHead zqp\n--------------------\n", "NSite 2\nNe 1\n")).unwrap();
    let namelist = fixture.0.join("namelist.def");
    fs::write(&namelist, "ModPara modpara.def\nTrans absent.def\n").unwrap();
    match parse_expert_mode_files(namelist) {
        Err(ParseError::InvalidInput { message }) => {
            assert!(message.contains("Required Trans file not found"));
        }
        result => panic!("missing Transfer must reject assembly: {result:?}"),
    }
}

#[test]
fn transfer_zero_count_short_header_is_defined_body_ignore_not_stale_data() {
    // C ReadBuffInt118-124 reads the count from line2. The later five fgets
    // calls874 ignore EOF; GetTransferInfo1978 returns before using the buffer
    // when count0. Thus no five-line completeness check is native authority
    // for this zero-count case; no uninitialized or stale field is observed.
    let fixture = Fixture::new();
    let path = fixture.0.join("trans.def");
    fs::write(&path, "===\nNTransfer 0\n").unwrap();
    assert!(parse_trans_definition(&path, 2).unwrap().is_empty());
    fs::write(&path, "===\nNTransfer 1\n").unwrap();
    assert_eq!(
        parse_trans_definition(path, 2).unwrap_err().kind(),
        io::ErrorKind::InvalidData
    );
}

#[test]
fn headerless_julia_helper_remains_distinct_from_c_definition_validation() {
    let terms = parse_trans_content("0 0 1 1 1.25 -0.5\ninvalid\n");
    assert_eq!(terms.len(), 1);
    assert_eq!(terms[0].value, Complex64::new(1.25, -0.5));
}

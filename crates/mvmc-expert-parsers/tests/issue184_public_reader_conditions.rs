//! Direct public utility conditions, distinct from Expert loader/runtime validation.
//! No numerical oracle, reference runtime, or regenerated fixture expectation.
use mvmc_expert_parsers::utils::file::{parse_namelist_content, read_def_file};
use std::{
    fs, io,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

static NEXT_DIRECTORY: AtomicUsize = AtomicUsize::new(0);

struct InputDirectory(PathBuf);

impl InputDirectory {
    fn new() -> Self {
        loop {
            let index = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "issue184-public-reader-{}-{index}",
                std::process::id()
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("exclusive input directory: {error}"),
            }
        }
    }
}

impl Drop for InputDirectory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove only exclusively created test input");
    }
}

#[test]
fn direct_public_reader_missing_file_and_directory_are_errors() {
    let input = InputDirectory::new();
    assert_eq!(
        read_def_file(input.0.join("missing.def"))
            .unwrap_err()
            .kind(),
        io::ErrorKind::NotFound
    );
    // Directory-read error kinds differ across platforms; neither may return content.
    assert!(read_def_file(&input.0).is_err());
    assert!(input.0.is_dir());
    assert!(!input.0.join("missing.def").exists());
}

#[test]
fn direct_public_reader_rejects_invalid_utf8_without_changing_bytes() {
    let input = InputDirectory::new();
    let path = input.0.join("invalid.def");
    let bytes = [b'N', b'S', 0xff, b'\n'];
    fs::write(&path, bytes).unwrap();
    assert_eq!(
        read_def_file(&path).unwrap_err().kind(),
        io::ErrorKind::InvalidData
    );
    assert_eq!(fs::read(path).unwrap(), bytes);
}

#[test]
fn direct_public_reader_preserves_empty_and_complete_utf8_content() {
    let input = InputDirectory::new();
    let path = input.0.join("content.def");
    fs::write(&path, []).unwrap();
    assert_eq!(read_def_file(&path).unwrap(), "");
    let content = "# exact bytes\r\nModPara modpara.def\n# UTF-8 comment: 日本語\n";
    fs::write(&path, content).unwrap();
    assert_eq!(read_def_file(&path).unwrap(), content);
    assert_eq!(fs::read(path).unwrap(), content.as_bytes());
}

#[test]
fn public_namelist_ignores_fewer_tokens_and_extra_ascii_fields() {
    let content = "\n# comment\n// comment\nModPara\n\tTrans\ttrans.def\textra\tfields # ignored\nModPara modpara.def // tail\n";
    assert_eq!(
        parse_namelist_content(content),
        [
            ("Trans".to_owned(), "trans.def".to_owned()),
            ("ModPara".to_owned(), "modpara.def".to_owned())
        ]
    );
    assert!(parse_namelist_content("\nModPara\n # comment\n").is_empty());
}

#[test]
fn public_namelist_preserves_duplicate_records_and_original_order() {
    // This utility returns metadata pairs, not a validated C filename table.
    // Duplicate handling/rejection in a full input loader is a separate contract.
    assert_eq!(
        parse_namelist_content("ModPara first.def\nTrans trans.def\nModPara second.def\n"),
        [
            ("ModPara".to_owned(), "first.def".to_owned()),
            ("Trans".to_owned(), "trans.def".to_owned()),
            ("ModPara".to_owned(), "second.def".to_owned())
        ]
    );
}

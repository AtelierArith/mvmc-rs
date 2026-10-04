//! Original Julia M0416–M0420 architecture; not C model input acceptance.
use mvmc_expert_parsers::utils::file::{get_file_info, validate_file_exists};
use std::{
    fs, io,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
    time::UNIX_EPOCH,
};

static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Inputs(PathBuf);
impl Inputs {
    fn new() -> Self {
        loop {
            let path = std::env::temp_dir().join(format!(
                "issue255-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("exclusive input directory: {error}"),
            }
        }
    }
}
impl Drop for Inputs {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove exclusive test inputs");
    }
}

#[test]
fn original_m0416_m0420_file_metadata_and_existence() {
    let input = Inputs::new();
    let path = input.0.join("test.def");
    fs::write(&path, b"NSite = 4\nNElec = 2\n").unwrap();
    let info = get_file_info(&path).unwrap();
    assert_eq!(info.filename, "test.def");
    assert_eq!(info.filepath, path);
    assert!(info.exists);
    assert_eq!(info.size_bytes, 20); // Independent original literal byte length.
    assert!(info.size_bytes > 0);
    assert!(validate_file_exists(&path));
    let missing = input.0.join("nonexistent.def");
    assert!(!validate_file_exists(&missing));
    let absent = get_file_info(&missing).unwrap();
    assert_eq!(absent.filename, "nonexistent.def");
    assert_eq!(absent.filepath, missing);
    assert!(!absent.exists);
    assert_eq!((absent.size_bytes, absent.last_modified), (0, UNIX_EPOCH));
    assert_eq!(
        info.last_modified,
        fs::metadata(&path).unwrap().modified().unwrap()
    );
}

#[test]
fn empty_file_is_regular_but_directory_is_not() {
    let input = Inputs::new();
    let path = input.0.join("empty.def");
    fs::write(&path, []).unwrap();
    let empty = get_file_info(&path).unwrap();
    assert!(empty.exists);
    assert_eq!(empty.size_bytes, 0);
    assert!(validate_file_exists(&path));
    assert_eq!(
        empty.last_modified,
        fs::metadata(&path).unwrap().modified().unwrap()
    );
    let directory = get_file_info(&input.0).unwrap();
    assert!(!directory.exists);
    assert_eq!(
        (directory.size_bytes, directory.last_modified),
        (0, UNIX_EPOCH)
    );
    assert!(!validate_file_exists(&input.0));
}

#[test]
fn metadata_errors_propagate_while_predicate_is_false() {
    let path = PathBuf::from("invalid\0path");
    assert_eq!(
        get_file_info(&path).unwrap_err().kind(),
        io::ErrorKind::InvalidInput
    );
    assert!(!validate_file_exists(&path));
}

#[test]
fn controlled_modification_time_is_reported_without_content_reads() {
    let input = Inputs::new();
    let path = input.0.join("modified.def");
    fs::write(&path, [0xff]).unwrap();
    let expected = UNIX_EPOCH + std::time::Duration::from_secs(1_700_000_000);
    let file = fs::OpenOptions::new().write(true).open(&path).unwrap();
    file.set_times(fs::FileTimes::new().set_modified(expected))
        .unwrap();
    let info = get_file_info(&path).unwrap();
    assert_eq!(info.last_modified, expected);
    assert_eq!(
        info.last_modified,
        fs::metadata(&path).unwrap().modified().unwrap()
    );
    assert_eq!(info.size_bytes, 1);
    assert!(info.exists);
}

#[cfg(unix)]
#[test]
fn regular_symlink_follows_target_broken_symlink_is_absent() {
    let input = Inputs::new();
    let target = input.0.join("target.def");
    fs::write(&target, b"abc").unwrap();
    let link = input.0.join("link.def");
    std::os::unix::fs::symlink(&target, &link).unwrap();
    let info = get_file_info(&link).unwrap();
    assert_eq!(info.filename, "link.def");
    assert_eq!(info.filepath, link);
    assert!(info.exists);
    assert_eq!(info.size_bytes, 3);
    assert_eq!(
        info.last_modified,
        fs::metadata(&target).unwrap().modified().unwrap()
    );
    assert!(validate_file_exists(&link));
    fs::remove_file(&target).unwrap();
    let broken = get_file_info(&link).unwrap();
    assert!(!broken.exists);
    assert_eq!((broken.size_bytes, broken.last_modified), (0, UNIX_EPOCH));
    assert!(!validate_file_exists(&link));
}

#[cfg(unix)]
#[test]
fn metadata_identity_preserves_non_utf8_filename() {
    use std::os::unix::ffi::OsStringExt;
    let input = Inputs::new();
    let name = std::ffi::OsString::from_vec(vec![b'x', 0xff]);
    let path = input.0.join(&name);
    fs::write(&path, [0xff]).unwrap();
    let info = get_file_info(&path).unwrap();
    assert_eq!(info.filename, name);
    assert_eq!(info.filepath, path);
    assert!(info.exists);
    assert_eq!(info.size_bytes, 1);
    assert!(validate_file_exists(&path));
}

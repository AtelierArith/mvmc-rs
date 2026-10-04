//! Direct definition boundaries, not context-bound Nsite or numerical parity.
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use mvmc_expert_parsers::parsers::green::{parse_green_one_def, parse_green_two_def};

static NEXT: AtomicUsize = AtomicUsize::new(0);

fn with_file<T>(content: &str, check: impl FnOnce(&Path) -> T) -> T {
    let dir = std::env::temp_dir().join(format!(
        "issue308-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir(&dir).unwrap();
    let path = dir.join("green.def");
    std::fs::write(&path, content).unwrap();
    let result = check(&path);
    std::fs::remove_file(&path).unwrap();
    std::fs::remove_dir(&dir).unwrap();
    result
}

fn lengths(path: &Path) -> [io::Result<usize>; 2] {
    [
        parse_green_one_def(path).map(|v| v.len()),
        parse_green_two_def(path).map(|v| v.len()),
    ]
}

#[test]
fn direct_green_readers_preserve_declared_duplicate_order() {
    with_file(
        "header\nGreen 3\nheader\nheader\nheader\n1 0 0 1 1 1 0 0\n0 1 1 0 0 0 1 1\n1 0 0 1 1 1 0 0\n",
        |path| {
            let one = parse_green_one_def(path).unwrap();
            let two = parse_green_two_def(path).unwrap();
            let one_rows: Vec<_> = one.iter().map(|t| (t.site1, t.spin1.as_code(), t.site2, t.spin2.as_code())).collect();
            assert_eq!(one_rows, [(1, 0, 0, 1), (0, 1, 1, 0), (1, 0, 0, 1)]);
            let two_rows: Vec<_> = two.iter().map(|t| (t.site1, t.spin1.as_code(), t.site2, t.spin2.as_code(), t.site3, t.spin3.as_code(), t.site4, t.spin4.as_code())).collect();
            assert_eq!(two_rows, [(1, 0, 0, 1, 1, 1, 0, 0), (0, 1, 1, 0, 0, 0, 1, 1), (1, 0, 0, 1, 1, 1, 0, 0)]);
        },
    );
}

#[test]
fn direct_green_zero_count_ignores_body() {
    with_file("header\nGreen 0\ninvalid body", |path| {
        for result in lengths(path) {
            assert_eq!(result.unwrap(), 0);
        }
    });
}

#[test]
fn direct_green_invalid_definitions_fail_instead_of_dropping_rows() {
    for content in [
        "",
        "header\nGreen -1",
        "header\nGreen overflow",
        "header\nGreen 1",
        "h\nGreen 1\nh\nh\nh\n",
        "h\nGreen 2\nh\nh\nh\n0 0 1 1 0 0 1 1\n",
        "h\nGreen 1\nh\nh\nh\n0 0 1 1 0 0 1 1\n0 0 1 1 0 0 1 1\n",
        "h\nGreen 1\nh\nh\nh\n0 0 invalid 1 0 0 1 1\n",
        "h\nGreen 1\nh\nh\nh\n0 2 1 1 0 0 1 1\n",
        "h\nGreen 1\nh\nh\nh\n-1 0 1 1 0 0 1 1\n",
    ] {
        with_file(content, |path| {
            for result in lengths(path) {
                assert_eq!(result.unwrap_err().kind(), io::ErrorKind::InvalidData);
            }
        });
    }
    with_file("h\nGreen 1\nh\nh\nh\n0 0 1 1\n", |path| {
        assert_eq!(parse_green_one_def(path).unwrap().len(), 1);
        assert_eq!(
            parse_green_two_def(path).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
    });
}

#[test]
fn direct_green_missing_file_preserves_io_kind() {
    with_file("", |path| {
        let missing = path.with_file_name("absent.def");
        for result in lengths(&missing) {
            assert_eq!(result.unwrap_err().kind(), io::ErrorKind::NotFound);
        }
    });
}

//! C defects of the 3D lattices that the Rust port deliberately does not reproduce (#356).
//!
//! `tests/fixtures/stdface/<case>/expected/` holds the corrected behaviour (the C StdFace with
//! `c_toolbox/stdface/3d_defects.patch`, checked byte for byte by `stdface_c_fixtures.rs`);
//! `c_historical/` holds the unmodified C output. These tests (1) pin that the Rust output differs
//! from the historical C output in every defect case, and (2) check the corrected behaviour
//! against properties derived independently of any C build.
//! The defect list is in `tests/fixtures/stdface/README_3d_defects.md`.

use std::fs;
use std::path::{Path, PathBuf};

use mvmc_stdface::{stdface_main_bytes, StdFaceError};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/stdface")
}

fn historical_cases() -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(root())
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().join("c_historical").is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

fn run_rust(case: &str) -> (i32, String, PathBuf) {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let work = std::env::temp_dir().join(format!(
        "mvmc-stdface-defects-{}-{id}-{case}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&work);
    fs::create_dir_all(&work).unwrap();
    let input = fs::read(root().join(case).join("StdFace.def")).unwrap();
    match stdface_main_bytes("StdFace.def", Some(&input), &work) {
        Ok(report) => (0, report.log, work),
        Err(failure) => match failure.error {
            StdFaceError::Exit(code) => (code & 0xff, failure.log, work),
            StdFaceError::Io(message) => panic!("{case}: {message}"),
        },
    }
}

fn pairs_of(path: &Path) -> Vec<(i64, i64)> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|l| !l.starts_with('=') && !l.starts_with("NExchange"))
        .filter_map(|l| {
            let f: Vec<_> = l.split_whitespace().collect();
            (f.len() == 3).then(|| (f[0].parse().unwrap(), f[1].parse().unwrap()))
        })
        .collect()
}

#[test]
fn rust_differs_from_the_historical_c_output_in_every_defect_case() {
    let cases = historical_cases();
    assert!(cases.len() >= 25, "{} defect cases", cases.len());
    for name in [
        "pyrochlore_kondo_defect",
        "fcc_kondo_fields_defect",
        "pyrochlore_hubbard_gc_fields_gamma_y",
        "ortho_spin_V2_accepted_by_c",
    ] {
        assert!(cases.iter().any(|c| c == name), "{name}");
    }
    for case in cases {
        let (status, log, work) = run_rust(&case);
        let historical = root().join(&case).join("c_historical");
        let c_status: i32 = fs::read_to_string(historical.join("exit_status"))
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        let c_log = fs::read(historical.join("stdout.txt")).unwrap();
        let mut same = (c_status & 0xff) == status && c_log == log.as_bytes();
        for entry in fs::read_dir(&historical).unwrap().filter_map(|e| e.ok()) {
            let name = entry.file_name().to_string_lossy().into_owned();
            if matches!(name.as_str(), "stdout.txt" | "exit_status" | "stderr.txt") {
                continue;
            }
            same &= fs::read(work.join(&name)).ok() == Some(fs::read(entry.path()).unwrap());
        }
        assert!(!same, "{case}: Rust reproduces the historical C defect");
        let _ = fs::remove_dir_all(work);
    }
}

#[test]
fn pyrochlore_kondo_couples_each_conduction_site_to_its_own_local_spin() {
    // 2 x 2 x 1 cells of 4 sites: sites 0..16 are local spins, 16..32 conduction sites.
    let (status, _, work) = run_rust("pyrochlore_kondo_defect");
    assert_eq!(status, 0);
    let pairs = pairs_of(&work.join("exchange.def"));
    assert_eq!(pairs.len(), 16);
    for (conduction, local) in pairs {
        assert_eq!(
            conduction,
            local + 16,
            "conduction {conduction} / local {local}"
        );
    }
    // Unmodified C: every local spin of a cell couples to conduction site 3 of that cell.
    let historical = pairs_of(&root().join("pyrochlore_kondo_defect/c_historical/exchange.def"));
    assert!(historical.iter().filter(|&&(c, _)| c == 19).count() == 4);
    let _ = fs::remove_dir_all(work);
}

fn trans_count(path: &Path) -> usize {
    let text = fs::read_to_string(path).unwrap();
    let line = text.lines().find(|l| l.starts_with("NTransfer")).unwrap();
    line.split_whitespace().nth(1).unwrap().parse().unwrap()
}

#[test]
fn fcc_kondo_local_spins_feel_the_field() {
    // 8 cells, 16 sites, Gamma, Gamma_y and h nonzero: each of the 8 local spins gets
    // -h*Sz for 2 states (2 terms) plus the two transverse terms (2) that C omitted.
    let (status, _, work) = run_rust("fcc_kondo_fields_defect");
    assert_eq!(status, 0);
    let fixed = trans_count(&work.join("trans.def"));
    let historical = trans_count(&root().join("fcc_kondo_fields_defect/c_historical/trans.def"));
    assert_eq!(fixed - historical, 8 * 4);
    let _ = fs::remove_dir_all(work);
}

#[test]
fn hubbard_local_gamma_y_terms_fit_the_allocation_where_c_overflows() {
    // Pyrochlore with mu = 0, h, Gamma, Gamma_y nonzero: six local transfer terms per site
    // (C allocates for four). C exits with SIGSEGV (-11); the corrected output is complete.
    let (status, _, work) = run_rust("pyrochlore_hubbard_gc_fields_gamma_y");
    assert_eq!(status, 0);
    // 16 sites; the six raw on-site terms per site (two diagonal, Gamma and Gamma_y each two
    // transverse) are merged by index when written: four entries per site.
    let text = fs::read_to_string(work.join("trans.def")).unwrap();
    let on_site = text
        .lines()
        .filter_map(|l| {
            let f: Vec<_> = l.split_whitespace().collect();
            (f.len() == 6 && f.iter().take(4).all(|t| t.parse::<i64>().is_ok()))
                .then(|| f[0] == f[2])
        })
        .filter(|&same_site| same_site)
        .count();
    assert_eq!(on_site, 16 * 4);
    let c_status = fs::read_to_string(
        root().join("pyrochlore_hubbard_gc_fields_gamma_y/c_historical/exit_status"),
    )
    .unwrap();
    assert_eq!(c_status.trim(), "-11");
    let _ = fs::remove_dir_all(work);
}

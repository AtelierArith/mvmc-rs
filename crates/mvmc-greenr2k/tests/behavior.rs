//! Behaviour that is not a plain parity check: error paths, the FullDiag branch
//! (which gfortran cannot run) and the command-line contract.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/greenr2k")
}

fn copy_dir(from: &Path, to: &Path) {
    fs::create_dir_all(to).unwrap();
    for entry in fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_dir(&entry.path(), &target);
        } else {
            fs::copy(entry.path(), target).unwrap();
        }
    }
}

fn work_dir(case: &str, tag: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("greenr2k-{tag}-{case}"));
    let _ = fs::remove_dir_all(&dir);
    copy_dir(&fixtures().join(case).join("inputs"), &dir);
    dir
}

#[test]
fn missing_two_body_index_lists_it_and_stops() {
    let dir = work_dir("chain6_missing_index", "behavior");
    let mut stdout = Vec::new();
    let err = mvmc_greenr2k::run("namelist.def", "geometry.dat", &dir, &mut stdout).unwrap_err();
    assert!(matches!(err, mvmc_greenr2k::Error::MissingIndices));
    let expected =
        fs::read_to_string(fixtures().join("chain6_missing_index/expected/stdout.txt")).unwrap();
    // Signed zeros of the reciprocal vectors are not a contract (LAPACK prints -0.0).
    let unsigned = |text: &str| text.replace("-0.0000000000", " 0.0000000000");
    assert_eq!(
        unsigned(&String::from_utf8(stdout).unwrap()),
        unsigned(&expected)
    );
    assert!(!dir.join("kpath.gp").exists());
}

/// Fortran `STOP "msg"` exits with status 0 (also after the missing-index error,
/// see `chain6_missing_index/expected/exit_code.txt`); the Rust tool reports the
/// failure with a non-zero status instead.
#[test]
fn missing_index_exit_status_is_nonzero_unlike_fortran() {
    let fortran_status =
        fs::read_to_string(fixtures().join("chain6_missing_index/expected/exit_code.txt")).unwrap();
    assert_eq!(fortran_status.trim(), "0");
    let dir = work_dir("chain6_missing_index", "cli");
    let out = Command::new(env!("CARGO_BIN_EXE_greenr2k"))
        .args(["namelist.def", "geometry.dat"])
        .current_dir(&dir)
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(
        String::from_utf8(out.stderr).unwrap(),
        "STOP Missing indices for the Green function.\n"
    );
}

#[test]
fn command_line_runs_in_the_current_directory() {
    let dir = work_dir("chain6_mvmc", "cli");
    let out = Command::new(env!("CARGO_BIN_EXE_greenr2k"))
        .args(["namelist.def", "geometry.dat"])
        .current_dir(&dir)
        .output()
        .unwrap();
    assert!(out.status.success());
    let expected = fs::read_to_string(fixtures().join("chain6_mvmc/expected/stdout.txt")).unwrap();
    assert_eq!(
        String::from_utf8(out.stdout).unwrap().lines().count(),
        expected.lines().count()
    );
    assert!(dir.join("output/zvo_corr.dat").exists());
    assert!(dir.join("kpath.gp").exists());
}

#[test]
fn usage_error_without_arguments() {
    let out = Command::new(env!("CARGO_BIN_EXE_greenr2k"))
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8(out.stderr).unwrap().starts_with("Usage:"));
}

/// CalcType 2 (FullDiag) cannot run under gfortran (`Constant string in input
/// format`), so there is no Fortran reference. The Rust tool skips the 26
/// characters of the format's constant string as other compilers do and must then
/// behave exactly like LOBCG with the same number of states.
#[test]
fn fulldiag_matches_lobcg_with_the_same_state_count() {
    let lobcg = work_dir("chain6_lobcg", "lobcg");
    let mut sink = Vec::new();
    mvmc_greenr2k::run("namelist.def", "geometry.dat", &lobcg, &mut sink).unwrap();

    let full = work_dir("chain6_lobcg", "fulldiag");
    fs::write(full.join("calcmod.def"), "CalcType 2\n").unwrap();
    fs::write(
        full.join("output/CHECK_Memory.dat"),
        format!("  MAX DIMENSION idim_max=1{:16}\n", 2),
    )
    .unwrap();
    let mut sink = Vec::new();
    mvmc_greenr2k::run("namelist.def", "geometry.dat", &full, &mut sink).unwrap();
    assert!(String::from_utf8(sink)
        .unwrap()
        .contains("Method : Full Diagonalization"));
    for name in [
        "output/zvo_corr_eigen0.dat",
        "output/zvo_corr_eigen1.dat",
        "output/zvo_corr_eigen0.dat.frmsf",
        "output/zvo_corr_eigen1.dat.frmsf",
        "kpath.gp",
    ] {
        assert_eq!(
            fs::read_to_string(full.join(name)).unwrap(),
            fs::read_to_string(lobcg.join(name)).unwrap(),
            "{name}"
        );
    }
}

/// gfortran reads the unquoted item `./modpara.def` as `.` because `/` ends
/// list-directed input; the Rust tool keeps path separators in file names.
#[test]
fn file_names_with_slashes_are_read_whole() {
    let dir = work_dir("chain6_mvmc", "slash");
    let namelist = fs::read_to_string(dir.join("namelist.def"))
        .unwrap()
        .replace("modpara.def", "./modpara.def");
    fs::write(dir.join("namelist.def"), namelist).unwrap();
    let mut stdout = Vec::new();
    mvmc_greenr2k::run("namelist.def", "geometry.dat", &dir, &mut stdout).unwrap();
    assert!(String::from_utf8(stdout)
        .unwrap()
        .contains("ModPara file : ./modpara.def"));
}

/// A keyword whose value is on the next record is still found (list-directed
/// input continues on the following record), and blank records are skipped.
#[test]
fn values_may_continue_on_the_next_record() {
    let dir = work_dir("chain6_mvmc", "continue");
    let namelist = fs::read_to_string(dir.join("namelist.def"))
        .unwrap()
        .replace("OneBodyG  greenone.def", "\n\nOneBodyG\n   greenone.def");
    fs::write(dir.join("namelist.def"), namelist).unwrap();
    let mut stdout = Vec::new();
    mvmc_greenr2k::run("namelist.def", "geometry.dat", &dir, &mut stdout).unwrap();
    assert!(String::from_utf8(stdout)
        .unwrap()
        .contains("OneBodyG file : greenone.def"));
}

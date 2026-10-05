//! Wannier90 behaviour where the C program reads uninitialised memory (#357); the Rust port
//! defines it. These inputs have no deterministic C output, so there is no C fixture.
use std::fs;
use std::path::{Path, PathBuf};

use mvmc_stdface::{stdface_main_bytes_in, StdFaceError};

fn fixture(case: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/stdface")
        .join(case)
}

fn out_dir(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("mvmc-w90-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).unwrap();
    path
}

#[test]
fn spin_model_without_onsite_u_reports_an_error_instead_of_reading_garbage() {
    // `wannier_square_hubbard_no_ur` ships hopping data but no `zvo_ur.dat`: the spin
    // super-exchange needs the on-site U of every orbital (C: uninitialised `Uspin`).
    let out = out_dir("nou");
    let input = b"model = \"Spin\"\nlattice = \"wannier90\"\nW = 2\nL = 2\n2Sz = 0\n";
    let result = stdface_main_bytes_in(
        "StdFace.def",
        Some(input),
        &fixture("wannier_square_hubbard_no_ur"),
        &out,
    );
    let failure = result.expect_err("must fail");
    assert_eq!(failure.error, StdFaceError::Exit(-1));
    assert!(failure
        .stderr
        .contains("the on-site Coulomb U of a Wannier orbital is not found"));
    let _ = fs::remove_dir_all(out);
}

#[test]
fn missing_hopping_ur_and_jr_files_are_skipped_without_terms() {
    // C frees uninitialised pointers when a file is skipped (crash); the port keeps the message
    // and produces no terms.
    let out = out_dir("skip");
    let dir = out_dir("skip-data");
    fs::copy(
        fixture("wannier_square_hubbard").join("zvo_geom.dat"),
        dir.join("zvo_geom.dat"),
    )
    .unwrap();
    let input = b"model = \"Hubbard\"\nlattice = \"wannier90\"\nW = 2\nL = 2\nncond = 4\n2Sz = 0\n";
    let report = stdface_main_bytes_in("StdFace.def", Some(input), &dir, &out).unwrap();
    for name in ["zvo_hr.dat", "zvo_ur.dat", "zvo_jr.dat"] {
        assert!(
            report
                .log
                .contains(&format!("Skip to read the file {name}.")),
            "{name}"
        );
    }
    assert!(out.join("namelist.def").is_file());
    let _ = fs::remove_dir_all(out);
    let _ = fs::remove_dir_all(dir);
}

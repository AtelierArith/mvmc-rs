//! #357: `mvmc -s` on a Wannier90 lattice reads `zvo_*.dat` from the working directory (as C),
//! generates the Expert files (including `initial.def` and `wan2site.dat`) and runs them.
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

struct TestDir(PathBuf);
impl TestDir {
    fn new(name: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("mvmc-issue357-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}
impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn fixture(case: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/stdface")
        .join(case)
}

fn stage(case: &str, dir: &TestDir, extra: &str) {
    for entry in fs::read_dir(fixture(case)).unwrap().flatten() {
        let name = entry.file_name();
        if entry.path().is_file() && name != "StdFace.def" {
            fs::copy(entry.path(), dir.0.join(name)).unwrap();
        }
    }
    let mut input = fs::read_to_string(fixture(case).join("StdFace.def")).unwrap();
    input.push_str(extra);
    fs::write(dir.0.join("stan.in"), input).unwrap();
}

#[test]
fn wannier90_square_with_double_counting_runs_from_standard_input() {
    let dir = TestDir::new("run");
    stage(
        "wannier_square_dc_full",
        &dir,
        "NSROptItrStep = 2\nNSROptItrSmp = 1\nNVMCSample = 10\nRndSeed = 1\n",
    );
    let output = Command::new(env!("CARGO_BIN_EXE_mvmc"))
        .args(["-s", "stan.in"])
        .current_dir(&dir.0)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(stdout.contains("initial.def is written."), "{stdout}");
    assert!(stdout.contains("Completed 2 SR steps"), "{stdout}");
    for name in [
        "namelist.def",
        "initial.def",
        "wan2site.dat",
        "lattice.xsf",
        "geometry.dat",
    ] {
        assert!(dir.0.join(name).is_file(), "{name}");
    }
}

#[test]
fn missing_geometry_file_exits_255_with_the_c_stderr_message() {
    let dir = TestDir::new("missing");
    stage("wannier_square_hubbard", &dir, "");
    fs::remove_file(dir.0.join("zvo_geom.dat")).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_mvmc"))
        .args(["--dry-run", "stan.in"])
        .current_dir(&dir.0)
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(255));
    assert!(String::from_utf8_lossy(&output.stderr)
        .contains("Error: Fail to open the file zvo_geom.dat."));
}

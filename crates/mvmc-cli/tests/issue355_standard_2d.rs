//! #355: `mvmc -s` on 2D lattices generates the C Expert files and runs the optimizer.
use std::fs;
use std::path::PathBuf;
use std::process::Command;

struct TestDir(PathBuf);
impl TestDir {
    fn new(name: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("mvmc-issue355-{}-{name}", std::process::id()));
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

fn run_standard(name: &str, input: &str) {
    let dir = TestDir::new(name);
    fs::write(dir.0.join("stan.in"), input).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_mvmc"))
        .args(["-s", "stan.in"])
        .current_dir(&dir.0)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "{name}: {stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(stdout.contains("Completed 2 SR steps"), "{name}: {stdout}");
    assert!(dir.0.join("namelist.def").is_file());
    assert!(dir.0.join("output").is_dir());
}

const RUN: &str = "NSROptItrStep = 2\nNSROptItrSmp = 1\nNVMCSample = 10\nRndSeed = 1\n";

#[test]
fn square_hubbard_runs_from_standard_input() {
    run_standard(
        "square",
        &format!(
            "model = \"Hubbard\"\nlattice = \"square\"\nW = 2\nL = 2\nU = 4.0\nt = 1.0\nncond = 4\n2Sz = 0\n{RUN}"
        ),
    );
}

#[test]
fn honeycomb_spin_runs_from_standard_input() {
    run_standard(
        "honeycomb",
        &format!(
            "model = \"Spin\"\nlattice = \"honeycomb\"\nW = 2\nL = 2\nJ = 1.0\n2Sz = 0\n{RUN}"
        ),
    );
}

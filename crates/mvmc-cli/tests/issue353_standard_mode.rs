//! #353: Standard mode in the CLI. `-s stan.in` runs StdFace (generate the Expert files into
//! the output directory) and then the generated `namelist.def`, as C `vmcmain -s`;
//! `--dry-run` generates only, as `vmcdry.out`. C exit status 255 (`StdFace_exit(-1)`) and the
//! C `stdout` text are reproduced.
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

struct TestDir(PathBuf);
impl TestDir {
    fn new(name: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("mvmc-issue353-{}-{name}", std::process::id()));
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

fn run(args: &[&str], cwd: &Path) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_mvmc"))
        .args(args)
        .current_dir(cwd)
        .output()
        .unwrap()
}

#[test]
fn dry_run_generates_the_c_files_and_prints_the_c_log() {
    let dir = TestDir::new("dry");
    let input = fixture("HubbardChain").join("StdFace.def");
    let out = dir.0.join("gen");
    let output = run(
        &[
            "--dry-run",
            input.to_str().unwrap(),
            "--out-dir",
            out.to_str().unwrap(),
        ],
        &dir.0,
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let expected = fixture("HubbardChain").join("expected");
    for name in [
        "namelist.def",
        "modpara.def",
        "trans.def",
        "orbitalidx.def",
        "qptransidx.def",
    ] {
        assert_eq!(
            fs::read(out.join(name)).unwrap(),
            fs::read(expected.join(name)).unwrap(),
            "{name}"
        );
    }
    // The C log names the input exactly as given on the command line.
    let log = String::from_utf8(output.stdout).unwrap();
    let want = fs::read_to_string(expected.join("stdout.txt"))
        .unwrap()
        .replace("StdFace.def", input.to_str().unwrap());
    assert_eq!(log, want);
    // Generate only: nothing is run.
    assert!(!out.join("output").exists());
}

#[test]
fn standard_mode_generates_then_runs_the_generated_namelist() {
    let dir = TestDir::new("standard");
    fs::write(
        dir.0.join("stan.in"),
        "L = 6\nLsub = 2\nmodel = \"Hubbard\"\nlattice = \"chain\"\nU = 4.0\nt = 1.0\nncond = 6\n\
         NSROptItrStep = 3\nNSROptItrSmp = 2\nNVMCSample = 20\n2Sz = 0\nRndSeed = 1\n",
    )
    .unwrap();
    let output = run(&["-s", "stan.in"], &dir.0);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    // StdFace output precedes the run banner and the files land in the working directory.
    let generated = stdout.find("Input files are generated.").unwrap();
    let completed = stdout.find("Completed 3 SR steps").unwrap();
    assert!(generated < completed);
    for name in [
        "namelist.def",
        "modpara.def",
        "trans.def",
        "geometry.dat",
        "lattice.gp",
    ] {
        assert!(dir.0.join(name).is_file(), "{name}");
    }
    assert!(dir.0.join("output").is_dir());
}

#[test]
fn standard_mode_failure_prints_the_c_message_and_exits_255() {
    let dir = TestDir::new("failure");
    fs::write(
        dir.0.join("stan.in"),
        "model = \"Hubbard\"\nlattice = \"chain\"\nFooBar = 1\n",
    )
    .unwrap();
    let output = run(&["-s", "stan.in"], &dir.0);
    assert_eq!(output.status.code(), Some(255));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("ERROR ! Unsupported Keyword in Standard mode!"),
        "{stdout}"
    );
    assert!(!dir.0.join("namelist.def").exists());
    assert!(!dir.0.join("output").exists());
}

#[test]
fn standard_mode_missing_input_reports_the_c_error() {
    let dir = TestDir::new("missing");
    let output = run(&["--dry-run", "nonexistent.in"], &dir.0);
    assert_eq!(output.status.code(), Some(255));
    assert!(String::from_utf8_lossy(&output.stdout)
        .contains("ERROR !  Cannot open input file nonexistent.in !"));
}

//! #367: a negative `DSROptStepDt` selects C's "Diagonalization Mode"
//! (readdef.c:959-965): rank 0 prints `remark: Diagonalization Mode`, the sign
//! is dropped, and the SR run is identical to the positive step. The only other
//! C effect is the `sEigen*` SRinfo header spelling (initfile.c:47-59).
use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

struct TestDir(PathBuf);
impl TestDir {
    fn new(name: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("mvmc-issue367-{}-{name}", std::process::id()));
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

fn run(name: &str, extra_modpara: &str) -> (Output, PathBuf, TestDir) {
    let dir = TestDir::new(name);
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../extern/Julia-mVMC/examples/inputs/heisenberg_chain_real");
    for entry in fs::read_dir(&fixture).unwrap().flatten() {
        if entry.path().is_file() {
            fs::copy(entry.path(), dir.0.join(entry.file_name())).unwrap();
        }
    }
    let modpara = dir.0.join("modpara.def");
    let mut text = fs::read_to_string(&modpara).unwrap();
    text.push_str(extra_modpara);
    fs::write(modpara, text).unwrap();
    let out_dir = dir.0.join("out");
    let output = Command::new(env!("CARGO_BIN_EXE_mvmc"))
        .arg(dir.0.join("namelist.def"))
        .args(["--nsteps", "2", "--nsmp", "2", "--seed", "1", "--out-dir"])
        .arg(&out_dir)
        .output()
        .unwrap();
    (output, out_dir, dir)
}

#[test]
fn negative_step_dt_runs_like_the_positive_step_and_reports_diagonalization_mode() {
    let (negative, negative_out, _a) = run("negative", "\nDSROptStepDt -0.01\n");
    let (positive, positive_out, _b) = run("positive", "\nDSROptStepDt 0.01\n");
    let negative_stderr = String::from_utf8_lossy(&negative.stderr);
    assert!(negative.status.success(), "{negative_stderr}");
    assert!(positive.status.success());
    assert!(
        negative_stderr.contains("remark: Diagonalization Mode"),
        "{negative_stderr}"
    );
    assert!(!String::from_utf8_lossy(&positive.stderr).contains("Diagonalization Mode"));
    // Same control path, same operands: deterministic runs must reproduce
    // exactly (only the sign handling differs, and it is removed).
    for file in ["zvo_out.dat", "zqp_opt.dat"] {
        assert_eq!(
            fs::read_to_string(negative_out.join(file)).unwrap(),
            fs::read_to_string(positive_out.join(file)).unwrap(),
            "{file}"
        );
    }
}

#[test]
fn srinfo_header_spelling_follows_sr_flag() {
    // Direct SR: the header spells diagonalization mode; SR-CG adds the C 1.4.0
    // " info" title (initfile.c:47-58). NSROptCGMaxIter is large and
    // NSRCGAbortOnFail off so the short CG run never aborts on its limit.
    for (name, dt, cg, expected) in [
        (
            "direct_negative",
            "-0.01",
            "0",
            "#Npara Msize optCut diagCut sEigenMax  sEigenMin    absRmax       imax",
        ),
        (
            "cg_positive",
            "0.01",
            "1",
            "#Npara Msize optCut diagCut sDiagMax  sDiagMin    absRmax       imax info",
        ),
    ] {
        let (output, out_dir, _dir) = run(
            name,
            &format!("\nDSROptStepDt {dt}\nNSRCG {cg}\nNSRCGAbortOnFail 0\n"),
        );
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let info = fs::read_to_string(out_dir.join("zvo_SRinfo.dat")).unwrap();
        assert_eq!(info.lines().next().unwrap(), expected, "{name}");
    }
}

#[test]
fn sr_cg_with_negative_step_dt_is_rejected_before_output() {
    // C 1.4.0 ReadDefFileNInt aborts SR-CG in diagonalization mode
    // (readdef.c:967-975); NSRCG 2 is DiagScale CG and is rejected too.
    for (name, cg) in [("cg_negative", "1"), ("diag_scale_negative", "2")] {
        let (output, out_dir, _dir) = run(name, &format!("\nDSROptStepDt -0.01\nNSRCG {cg}\n"));
        assert!(!output.status.success(), "{name}");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains(
                "error: SR-CG (NSRCG=1) cannot be combined with diagonalization mode \
                 (DSROptStepDt < 0)"
            ),
            "{name}: {stderr}"
        );
        assert!(!out_dir.exists(), "{name}: no output may be created");
    }
}

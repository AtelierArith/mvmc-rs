//! #344: an unknown modpara.def keyword is a C read error ("keyword ... is
//! incorrect"); the CLI must exit nonzero, name the keyword and write nothing.
use std::fs;
use std::path::PathBuf;
use std::process::Command;

const HEADER: &str = "--------------------\nModel_Parameters 0\n--------------------\nVMC_Cal_Parameters\n--------------------\nCDataFileHead zvo\nCParaFileHead zqp\n--------------------\n";

struct TestDir(PathBuf);
impl TestDir {
    fn new(name: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("mvmc-issue344-{}-{name}", std::process::id()));
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

fn run_with_modpara(name: &str, body: &str) -> (std::process::Output, PathBuf, TestDir) {
    let dir = TestDir::new(name);
    fs::write(dir.0.join("modpara.def"), format!("{HEADER}{body}")).unwrap();
    fs::write(dir.0.join("namelist.def"), "ModPara modpara.def\n").unwrap();
    let out_dir = dir.0.join("out");
    let output = Command::new(env!("CARGO_BIN_EXE_mvmc"))
        .arg(dir.0.join("namelist.def"))
        .arg("--out-dir")
        .arg(&out_dir)
        .output()
        .unwrap();
    (output, out_dir, dir)
}

#[test]
fn unknown_modpara_keyword_exits_nonzero_naming_the_keyword() {
    for keyword in [
        "BogusKeyword",
        "useDiagScale",
        "NElec",
        "NFileFlushInterval",
    ] {
        let (output, out_dir, _dir) = run_with_modpara(keyword, &format!("Nsite 2\n{keyword} 1\n"));
        assert!(!output.status.success(), "{keyword}: must fail");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains(keyword), "{keyword}: stderr was {stderr}");
        assert!(
            stderr.contains("incorrect"),
            "{keyword}: stderr was {stderr}"
        );
        assert!(!out_dir.exists(), "{keyword}: no output may be created");
    }
}

#[test]
fn explicit_two_sz_minus_one_is_rejected_like_c() {
    let (output, out_dir, _dir) = run_with_modpara("sz", "2Sz -1\n");
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("2Sz"));
    assert!(!out_dir.exists());
}

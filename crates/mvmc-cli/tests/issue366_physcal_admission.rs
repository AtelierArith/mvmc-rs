//! Issue #366: C readdef.c rejects unsupported namelist sections (BF/BFRange
//! "Back Flow is not supported") and unknown keywords while reading the
//! namelist, independent of NVMCCalMode. PhysCal must apply the same
//! admission checks as optimization, before any output is created.
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::atomic::{AtomicUsize, Ordering},
};

struct Work(PathBuf);
impl Work {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "issue366-admission-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Work {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn model_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/physcal_181/two-samples/heisenberg_chain_real")
}

/// Copy the PhysCal fixture and append extra namelist lines plus their file.
fn inputs(work: &Work, extra_namelist: &str) -> PathBuf {
    let dir = work.0.join("inputs");
    fs::create_dir_all(&dir).unwrap();
    for entry in fs::read_dir(model_root().join("inputs")).unwrap().flatten() {
        if entry.path().is_file() {
            fs::copy(entry.path(), dir.join(entry.file_name())).unwrap();
        }
    }
    let namelist = dir.join("namelist.def");
    let mut text = fs::read_to_string(&namelist).unwrap();
    text.push_str(extra_namelist);
    fs::write(&namelist, text).unwrap();
    fs::write(dir.join("extra.def"), "====\nNExtra 0\n====\n").unwrap();
    namelist
}

fn run_physcal(namelist: &Path, out: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_mvmc"))
        .arg(namelist)
        .arg("--physcal")
        .arg(model_root().join("zqp_opt.dat"))
        .args(["--seed", "1", "--mode", "real", "--out-dir"])
        .arg(out)
        .env("OMP_NUM_THREADS", "1")
        .env("OPENBLAS_NUM_THREADS", "1")
        .output()
        .unwrap()
}

#[test]
fn physcal_rejects_unsupported_sections_before_output() {
    for kind in ["BF", "BFRange", "SpinJastrow", "BogusSection", "InBogus"] {
        let work = Work::new();
        let namelist = inputs(&work, &format!("{kind} extra.def\n"));
        let out = work.0.join("out");
        let output = run_physcal(&namelist, &out);
        assert!(!output.status.success(), "{kind}: PhysCal must fail");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains(kind), "{kind}: stderr was {stderr}");
        assert!(!out.exists(), "{kind}: output directory was created");
    }
}

#[test]
fn physcal_still_accepts_the_supported_fixture() {
    let work = Work::new();
    let namelist = inputs(&work, "");
    let out = work.0.join("out");
    let output = run_physcal(&namelist, &out);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

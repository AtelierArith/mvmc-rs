//! CLI rejection boundary for bounded malformed records, not a claim that C's
//! unchecked fscanf defines rejection semantics for malformed input.
use std::{
    fs,
    path::PathBuf,
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
};

struct OwnedDir(PathBuf);
impl OwnedDir {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        loop {
            let path = std::env::temp_dir().join(format!(
                "issue184-cli-fixed-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => panic!("{e}"),
            }
        }
    }
}
impl Drop for OwnedDir {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn actual_cli_missing_and_malformed_fixed_records_never_create_requested_output_directory() {
    let dir = OwnedDir::new();
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "../../extern/Julia-mVMC/test/integration/reference/heisenberg_chain_real/physcal_ref",
    );
    let inputs = dir.0.join("inputs");
    fs::create_dir(&inputs).unwrap();
    for entry in fs::read_dir(source.join("inputs")).unwrap() {
        let entry = entry.unwrap();
        assert!(entry.file_type().unwrap().is_file());
        fs::copy(entry.path(), inputs.join(entry.file_name())).unwrap();
    }
    let valid = fs::read_to_string(source.join("zqp_opt.dat")).unwrap();
    let tokens: Vec<_> = valid.split_whitespace().collect();
    assert_eq!(tokens.len(), 48); // Six diagnostics + fourteen declared triples.
    for (case, text, message) in [
        ("missing", None, "fixed parameter file not found"),
        ("short", Some(tokens[..47].join(" ")), "too short"),
        (
            "mid-token",
            Some({
                let mut t = tokens.clone();
                t[27] = "malformed";
                t.join(" ")
            }),
            "non-numeric token 'malformed' at field 28",
        ),
        ("trailing", Some(format!("{valid} 1")), "trailing floats"),
    ] {
        let fixed = dir.0.join(format!("{case}.dat"));
        if let Some(text) = text {
            fs::write(&fixed, text).unwrap();
        }
        let out = dir.0.join(format!("never-created-{case}"));
        assert!(!out.exists());
        let result = Command::new(env!("CARGO_BIN_EXE_mvmc"))
            .arg(inputs.join("namelist.def"))
            .arg("--physcal")
            .arg(&fixed)
            .args(["--seed", "11272", "--out-dir"])
            .arg(&out)
            .current_dir(&dir.0)
            .env_remove("OMPI_COMM_WORLD_SIZE")
            .env_remove("PMI_SIZE")
            .output()
            .unwrap();
        assert!(!result.status.success(), "{case}: unexpectedly ran PhysCal");
        let stderr = String::from_utf8_lossy(&result.stderr);
        assert!(
            stderr.contains(message),
            "{case}: wrong rejection: {stderr}"
        );
        assert!(!out.exists(), "{case}: created requested output directory");
        assert!(!String::from_utf8_lossy(&result.stdout).contains("Completed"));
        assert_eq!(
            fs::read_to_string(source.join("zqp_opt.dat")).unwrap(),
            valid
        );
    }
}

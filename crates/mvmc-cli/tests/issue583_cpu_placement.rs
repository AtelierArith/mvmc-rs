//! Placement errors must be reported before reading input or creating output.
use std::process::Command;

#[test]
fn placement_requires_valid_request_and_multi_rank_linux_mpi() {
    for (index, value) in ["0", "bad", "2"].into_iter().enumerate() {
        let root =
            std::env::temp_dir().join(format!("issue583-placement-{}-{index}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let out = root.join("output");
        let output = Command::new(env!("CARGO_BIN_EXE_mvmc"))
            .arg(root.join("missing.def"))
            .args(["--out-dir"])
            .arg(&out)
            .env("MVMC_RS_MPI_CPUS_PER_RANK", value)
            .env("MVMC_RS_TENFERRO_WORKER_BUDGET", "0")
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert_eq!(output.status.code(), Some(2), "{stderr}");
        let expected = if value == "2" {
            if cfg!(all(feature = "mpi", target_os = "linux")) {
                "requires a multi-rank MPI launch"
            } else {
                "requires Linux and the MPI feature"
            }
        } else {
            "must be a positive integer"
        };
        assert!(stderr.contains(expected), "{stderr}");
        assert!(!stderr.contains("panicked"), "{stderr}");
        assert!(!out.exists());
        std::fs::remove_dir_all(root).unwrap();
    }
}

//! C-facing indexed TransSym is distinct from Julia generic QPTrans terms.
use mvmc_expert_parsers::{parse_expert_mode_files, parse_expert_mode_files_with_c_opt_trans};
use num_complex::Complex64;
use std::{
    fs, io,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Input(PathBuf);
impl Input {
    fn new(keyword: &str, payload: &str) -> Self {
        let path = loop {
            let path = std::env::temp_dir().join(format!(
                "issue184-qp-payload-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => break path,
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(e) => panic!("exclusive input: {e}"),
            }
        };
        fs::write(path.join("modpara.def"), "Nsite 2\nNMPTrans 1\n").unwrap();
        fs::write(
            path.join("namelist.def"),
            format!("ModPara modpara.def\n{keyword} projection.def\n"),
        )
        .unwrap();
        fs::write(path.join("projection.def"), payload).unwrap();
        Self(path)
    }
}
impl Drop for Input {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn c_zero_count_is_not_rejected_by_generic_intent_inference() {
    for keyword in ["QPTrans", "TransSym"] {
        for payload in ["===\nAnything 0\n", "0 0.25 0.5\n1 0.75 -0.5\n"] {
            let input = Input::new(keyword, payload);
            for c_mode in [false, true] {
                let data = if c_mode {
                    parse_expert_mode_files_with_c_opt_trans(input.0.join("namelist.def"), false)
                } else {
                    parse_expert_mode_files(input.0.join("namelist.def"))
                }
                .unwrap();
                // ReadBuffInt skips line one, ignores the label, and sscanf %d
                // reads the integer prefix 0 from 0.75. GetInfoTransSym count0
                // reads no body. These bytes cannot establish generic intent.
                assert!(data.input_errors.is_empty());
                assert_eq!(data.n_qp_trans, 0);
                assert!(data.qp_trans_entries.is_empty());
                assert!(data.para_qp_trans.is_empty());
            }
        }
    }
}

#[test]
fn indexed_alias_and_canonical_transsym_preserve_complex_weights_and_maps() {
    for keyword in ["QPTrans", "TransSym"] {
        for label in ["NQPTrans", "ArbitraryLabel"] {
            let input = Input::new(
                keyword,
                &format!("===\n{label} 1\n===\n===\n===\n0 1 0.5\n0 0 1 1\n0 1 0 1\n"),
            );
            for c_mode in [false, true] {
                let data = if c_mode {
                    parse_expert_mode_files_with_c_opt_trans(input.0.join("namelist.def"), false)
                } else {
                    parse_expert_mode_files(input.0.join("namelist.def"))
                }
                .unwrap();
                assert!(data.input_errors.is_empty());
                assert_eq!(data.n_qp_trans, 1);
                assert_eq!(data.para_qp_trans, [Complex64::new(1.0, 0.5)]);
                assert_eq!(data.qp_trans_entries[0].site_map, [1, 0]);
                assert_eq!(data.qp_trans_entries[0].site_sign, [1, 1]);
            }
        }
    }
}

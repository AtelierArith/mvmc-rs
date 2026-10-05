//! S070/S071: public parse -> declared AP width -> indexed P overlay.
//! Julia literals and the explicit complete C-input adaptation are documented
//! in issue-184-original-ap-offset.md; this is not a numerical runner test.
use mvmc_expert_parsers::parse_expert_mode_files;
use mvmc_expert_parsers::utils::read_input_parameters::read_input_parameters;
use num_complex::Complex64 as C;
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Bundle(PathBuf);
impl Bundle {
    fn new(ap_count: usize, ap_rows: &str, overlay: &str, parallel_first: bool) -> Self {
        let path = loop {
            let path = std::env::temp_dir().join(format!(
                "issue184-ap-offset-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => break path,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("exclusive AP-offset fixture: {error}"),
            }
        };
        fs::write(path.join("mod.def"), "--------------------\nModel_Parameters 0\n--------------------\nVMC_Cal_Parameters\n--------------------\nCDataFileHead zvo\nCParaFileHead zqp\n--------------------\nNSite 2\nNe 1\nNMPTrans 1\n").unwrap();
        fs::write(path.join("ap.def"), definition(ap_count, ap_rows)).unwrap();
        // C consumes only the upper-triangle physical pair; the two spin
        // sectors are expanded by the public parser, not supplied as rows.
        fs::write(path.join("p.def"), definition(1, "0 1 0 1\n0 1\n")).unwrap();
        fs::write(path.join("in.def"), definition(2, overlay)).unwrap();
        let orbital = if parallel_first {
            "OrbitalParallel p.def\nOrbitalAntiParallel ap.def\n"
        } else {
            "OrbitalAntiParallel ap.def\nOrbitalParallel p.def\n"
        };
        fs::write(
            path.join("namelist.def"),
            format!("ModPara mod.def\n{orbital}InOrbitalParallel in.def\n"),
        )
        .unwrap();
        Self(path)
    }
}
impl Drop for Bundle {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove exclusive AP-offset fixture");
    }
}
fn definition(count: usize, rows: &str) -> String {
    format!("=============================================\nNOrbitalIdx {count}\nComplexType 0\n=============================================\n=============================================\n{rows}")
}

#[test]
fn original_s070_public_loader_preserves_ap_width_before_parallel_expansion() {
    let bundle = Bundle::new(
        2,
        "0 0 0 1\n0 1 1 1\n1 0 1 1\n1 1 0 1\n0 1\n1 1\n",
        "0 11 0\n1 22 0\n",
        false,
    );
    let data = parse_expert_mode_files(bundle.0.join("namelist.def")).unwrap();
    assert!(data.input_errors.is_empty(), "{:?}", data.input_errors);
    assert_eq!(data.n_orbital_anti_parallel, 2);
    assert_eq!(data.i_flg_orbital_anti_parallel, 1);
    assert_eq!(data.i_flg_orbital_parallel, 1);
    assert_eq!(data.modpara.n_orbital_idx, 4);
    assert!(data.orbital_terms.iter().any(|term| term.idx == 2));
    assert!(data.orbital_terms.iter().any(|term| term.idx == 3));
}

#[test]
fn original_s071_public_loader_reserves_unmapped_ap_slot_and_scatter_uses_header_offset() {
    for parallel_first in [false, true] {
        let bundle = Bundle::new(
            3,
            "0 0 0 1\n0 1 0 1\n1 0 1 1\n1 1 1 1\n0 1\n1 1\n2 1\n",
            // Original independent values11/22, reversed records additionally
            // distinguish indexed scatter from record-order copying.
            "1 22 0\n0 11 0\n",
            parallel_first,
        );
        let mut data = parse_expert_mode_files(bundle.0.join("namelist.def")).unwrap();
        assert!(data.input_errors.is_empty(), "{:?}", data.input_errors);
        assert_eq!(data.n_orbital_anti_parallel, 3);
        assert_eq!(data.modpara.n_orbital_idx, 5);
        assert!(data.orbital_terms.iter().any(|term| term.idx == 3));
        assert!(data.orbital_terms.iter().any(|term| term.idx == 4));
        assert!(data.orbital_terms.iter().all(|term| term.idx != 2));
        // Exact dyadic sentinels isolate reserved AP storage from P writes;
        // no Rust-generated expected outputs or parameter initialization.
        data.slater_params = vec![
            C::new(0.25, 0.0),
            C::new(-0.5, 0.0),
            C::new(0.125, -0.25),
            C::new(0.0, 0.0),
            C::new(0.0, 0.0),
        ];
        let before = data.clone();
        read_input_parameters(&mut data, bundle.0.join("namelist.def")).unwrap();
        assert_eq!(
            data.slater_params,
            [
                C::new(0.25, 0.0),
                C::new(-0.5, 0.0),
                C::new(0.125, -0.25),
                C::new(11.0, 0.0),
                C::new(22.0, 0.0)
            ]
        );
        assert_eq!(data.n_orbital_anti_parallel, before.n_orbital_anti_parallel);
        assert_eq!(data.optimization_flags, before.optimization_flags);
        assert_eq!(data.projection_parameters(), before.projection_parameters());
        assert_eq!(data.rbm_parameters(), before.rbm_parameters());
        assert_eq!(data.orbital_idx_matrix, before.orbital_idx_matrix);
        assert_eq!(data.orbital_sgn_matrix, before.orbital_sgn_matrix);
    }
}

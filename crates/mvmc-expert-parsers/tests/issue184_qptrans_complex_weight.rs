//! Independent C TransSym reader contract; retained public-parser RED is external.
//! Complete C TransSym contract, independent literal dyadic coefficient.
//! Actual independent C reader acquired 1+0.5i for both AP modes; expectations
//! below are literal input coefficients. No toolbox access from normal tests.
//! C authority: mVMC1.3.0 readdef.c2232–2265, source SHA256
//! 6c53cb832f93d6cbfd7cea955fbb693738af5536b913d36af32b98eed38c32d9.
//! Standalone GCC13.3.0 -O0 -ffp-contract=off observed STATUS0 and
//! WEIGHT 0x1p+0 0x1p-1 under AP0/AP1; no BLAS/RNG/MPI execution.
use mvmc_expert_parsers::{parse_expert_mode_files, ExpertModeData};
use num_complex::Complex64;
use std::{
    fs, io,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

static NEXT_BUNDLE: AtomicUsize = AtomicUsize::new(0);
struct Bundle(PathBuf);
impl Bundle {
    fn new(nmp: i64, imaginary: &str) -> Self {
        let path = loop {
            let id = NEXT_BUNDLE.fetch_add(1, Ordering::Relaxed);
            let candidate = std::env::temp_dir().join(format!(
                "issue184-qptrans-complex-{}-{id}",
                std::process::id()
            ));
            match fs::create_dir(&candidate) {
                Ok(()) => break candidate,
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(e) => panic!("exclusive input directory: {e}"),
            }
        };
        let bundle = Self(path);
        fs::write(
            bundle.0.join("modpara.def"),
            format!("--------------------\nModel_Parameters 0\n--------------------\nVMC_Cal_Parameters\n--------------------\nCDataFileHead zvo\nCParaFileHead zqp\n--------------------\nNsite 4\nNe 1\nNMPTrans {nmp}\n"),
        )
        .unwrap();
        fs::write(
            bundle.0.join("namelist.def"),
            "ModPara modpara.def\nTransSym qptransidx.def\n",
        )
        .unwrap();
        fs::write(
            bundle.0.join("qptransidx.def"),
            format!(
                "===\nNQPTrans 1\n===\nTrIdx_TrWeight_and_TrIdx_i_xi\n===\n\
             0 1.00000 {imaginary}\n\
             0 0 1 -1\n0 1 2 1\n0 2 3 -1\n0 3 0 1\n"
            ),
        )
        .unwrap();
        bundle
    }
    fn parse(&self) -> ExpertModeData {
        let data = parse_expert_mode_files(self.0.join("namelist.def")).unwrap();
        assert!(data.input_errors.is_empty(), "{:?}", data.input_errors);
        data
    }
}
impl Drop for Bundle {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove exclusively created test input");
    }
}

fn assert_maps_and_signs(data: &ExpertModeData, nmp: i64) {
    assert_eq!(data.n_qp_trans, 1);
    assert_eq!(data.qp_trans_entries.len(), 1);
    let entry = &data.qp_trans_entries[0];
    assert_eq!(entry.site_map, [1, 2, 3, 0]);
    // Public raw Rust representation is intentionally NOT C-normalized SIGN.
    assert_eq!(entry.site_sign, [-1, 1, -1, 1]);
    let effective: Vec<_> = (0..4)
        .map(|site| entry.boundary_sign(site, nmp < 0))
        .collect();
    let expected = if nmp < 0 {
        [-1, 1, -1, 1]
    } else {
        [1, 1, 1, 1]
    };
    assert_eq!(effective, expected);
}

#[test]
fn zero_imaginary_periodic_and_ap_sign_controls() {
    for nmp in [1, -1] {
        let bundle = Bundle::new(nmp, "0.00000");
        let data = bundle.parse();
        assert_maps_and_signs(&data, nmp);
        assert_eq!(data.qp_trans_entries[0].weight, Complex64::new(1.0, 0.0));
        assert_eq!(data.para_qp_trans, [Complex64::new(1.0, 0.0)]);
    }
}

fn assert_complex_weight(nmp: i64) {
    let bundle = Bundle::new(nmp, "0.50000");
    let data = bundle.parse();
    assert_maps_and_signs(&data, nmp);
    // Literal supported input, not computed FP tolerance or Rust-made oracle.
    // Expected first RED boundary is this actual public parsed coefficient.
    assert_eq!(
        data.qp_trans_entries[0].weight,
        Complex64::new(1.0, 0.5),
        "public TransSym parser discarded the input imaginary coefficient"
    );
    assert_eq!(data.para_qp_trans, [Complex64::new(1.0, 0.5)]);
}
#[test]
fn complex_weight_survives_public_periodic_parser() {
    assert_complex_weight(1);
}
#[test]
fn complex_weight_survives_public_ap_parser() {
    assert_complex_weight(-1);
}

#[test]
fn coefficient_columns_survive_content_definition_and_full_entry_paths() {
    use mvmc_expert_parsers::parsers::qptrans::{parse_qptrans_content, parse_qptrans_def};
    for (columns, expected) in [
        ("1.00000", Complex64::new(1.0, 0.0)),
        ("1.00000 0.00000", Complex64::new(1.0, 0.0)),
        ("-1.25000 -0.50000", Complex64::new(-1.25, -0.5)),
        ("1.00000 -0.50000", Complex64::new(1.0, -0.5)),
    ] {
        let bundle = Bundle::new(1, "0.00000");
        let path = bundle.0.join("qptransidx.def");
        let content = fs::read_to_string(&path)
            .unwrap()
            .replace("0 1.00000 0.00000", &format!("0 {columns}"));
        fs::write(&path, &content).unwrap();
        let section = parse_qptrans_content(&content, 4);
        assert_eq!(section.n_qp_trans, 1);
        assert_eq!(section.entries.len(), 1);
        assert_eq!(section.entries[0].weight, expected);
        let definition = parse_qptrans_def(&path, 4).unwrap();
        assert_eq!(definition.entries[0].weight, expected);
        let data = bundle.parse();
        assert_maps_and_signs(&data, 1);
        assert_eq!(data.qp_trans_entries[0].weight, expected);
        assert_eq!(data.para_qp_trans, [expected]);
    }
}

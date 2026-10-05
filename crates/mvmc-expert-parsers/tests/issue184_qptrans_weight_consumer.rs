//! Public parser-to-weight copy contract, NSPGaussLeg=1 without OptTrans.
//! Literal 1+0.5i was independently accepted by C GetInfoTransSym2232–2265
//! under AP0/AP1; this is not a new native acquisition or sampling proof.
//! C qp.c InitQPWeight48–59 copies ParaQPTrans into QPFixWeight;
//! UpdateQPWeight140–143 copies it into QPFullWeight without OptTrans.
//! No arithmetic tolerance or C/Julia/toolbox runtime is used here.

use mvmc_expert_parsers::{
    parse_expert_mode_files, utils::qp_weight::init_qp_weight, ExpertModeData,
};
use num_complex::Complex64;
use std::{
    fs, io,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

static NEXT_INPUT: AtomicUsize = AtomicUsize::new(0);

struct Input(PathBuf);

impl Input {
    fn new(nmp: i64) -> Self {
        let path = loop {
            let id = NEXT_INPUT.fetch_add(1, Ordering::Relaxed);
            let candidate = std::env::temp_dir().join(format!(
                "issue184-qptrans-consumer-{}-{id}",
                std::process::id()
            ));
            match fs::create_dir(&candidate) {
                Ok(()) => break candidate,
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("exclusive input directory: {error}"),
            }
        };
        let input = Self(path);
        fs::write(
            input.0.join("modpara.def"),
            format!("--------------------\nModel_Parameters 0\n--------------------\nVMC_Cal_Parameters\n--------------------\nCDataFileHead zvo\nCParaFileHead zqp\n--------------------\nNsite 4\nNe 1\nNMPTrans {nmp}\nNSPGaussLeg 1\n"),
        )
        .unwrap();
        fs::write(
            input.0.join("namelist.def"),
            "ModPara modpara.def\nTransSym qptransidx.def\n",
        )
        .unwrap();
        fs::write(
            input.0.join("qptransidx.def"),
            "===\nNQPTrans 1\n===\nTrIdx_TrWeight_and_TrIdx_i_xi\n===\n\
             0 1.00000 0.50000\n\
             0 0 1 -1\n0 1 2 1\n0 2 3 -1\n0 3 0 1\n",
        )
        .unwrap();
        input
    }

    fn parse(&self) -> ExpertModeData {
        let data = parse_expert_mode_files(self.0.join("namelist.def")).unwrap();
        assert!(data.input_errors.is_empty(), "{:?}", data.input_errors);
        data
    }
}

impl Drop for Input {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove exclusively owned input");
    }
}

fn assert_public_copy(nmp: i64) {
    let input = Input::new(nmp);
    let mut data = input.parse();
    let coefficient = Complex64::new(1.0, 0.5);
    assert_eq!(data.modpara.nmp_trans, nmp);
    assert_eq!(data.modpara.nsp_gauss_leg, 1);
    assert_eq!(data.n_qp_trans, 1);
    assert_eq!(data.qp_trans_entries.len(), 1);
    assert_eq!(data.qp_trans_entries[0].weight, coefficient);
    assert_eq!(data.para_qp_trans, [coefficient]);
    assert!(data.opt_trans.is_empty());
    let parsed_entries = data.qp_trans_entries.clone();
    let parsed_coefficients = data.para_qp_trans.clone();

    init_qp_weight(&mut data);

    assert_eq!(data.modpara.nmp_trans, nmp);
    assert_eq!(data.qp_trans_entries, parsed_entries);
    assert_eq!(data.para_qp_trans, parsed_coefficients);
    let weights = data
        .qp_weights
        .as_ref()
        .expect("public initialized weights");
    assert_eq!(weights.qp_fix_weight, [coefficient]);
    assert_eq!(weights.qp_full_weight, [coefficient]);
    assert_eq!(weights.spgl_cos, [Complex64::new(1.0, 0.0)]);
    assert_eq!(weights.spgl_sin, [Complex64::new(0.0, 0.0)]);
}

#[test]
fn periodic_complex_transsym_reaches_public_qp_weights() {
    assert_public_copy(1);
}

#[test]
fn antiperiodic_complex_transsym_reaches_public_qp_weights() {
    assert_public_copy(-1);
}

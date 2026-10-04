//! Original Julia8bb test_qp_weight.jl:119–164, M0305–315.
//! C qp.c InitQPWeight48–59 copies the single quadrature sector; UpdateQPWeight
//! 127–145 copies without OptTrans or multiplies OptTrans[i]*QPFixWeight[j]
//! at offset i*NQPFix+j. Expectations are original exact dyadic literals.
//! Rust slice presence selects the branch; C uses FlagOptTrans and allocated
//! counts. This helper test does not establish that a definition enables the
//! production flag, nor certify C reallocation or any sampling trajectory.
use mvmc_expert_parsers::utils::qp_weight::{init_qp_weight, update_qp_weight};
use mvmc_expert_parsers::{ExpertModeData, QuantumProjectionWeights};
use num_complex::Complex64;

fn real(values: &[f64]) -> Vec<Complex64> {
    values
        .iter()
        .map(|&value| Complex64::new(value, 0.0))
        .collect()
}

#[test]
fn original_data_opttrans_initialization_preserves_fixed_and_orders_full_weights() {
    let mut data = ExpertModeData::new();
    data.modpara.nsp_gauss_leg = 1;
    data.modpara.nsp_stot = 0;
    data.modpara.nmp_trans = 2;
    data.para_qp_trans = real(&[1.0, 2.0]);
    data.opt_trans = real(&[0.5, 1.5]);
    init_qp_weight(&mut data);
    let weights = data.qp_weights.as_ref().unwrap();
    // M0305/M0306: original expected arrays, not computed by this test.
    assert_eq!(weights.qp_fix_weight, real(&[1.0, 2.0]));
    assert_eq!(weights.qp_full_weight, real(&[0.5, 1.0, 1.5, 3.0]));
    assert_eq!(data.para_qp_trans, real(&[1.0, 2.0]));
    assert_eq!(data.opt_trans, real(&[0.5, 1.5]));
}

#[test]
fn original_update_switches_from_copy_to_two_opttrans_blocks_without_changing_fixed() {
    let mut weights = QuantumProjectionWeights::new();
    weights.qp_fix_weight = real(&[1.0, 2.0, 3.0]);
    update_qp_weight(&mut weights, &[]);
    assert_eq!(weights.qp_full_weight.len(), 3); // M0307
    assert_eq!(weights.qp_full_weight, weights.qp_fix_weight); // M0308

    let opt_trans = real(&[0.5, 1.5]);
    update_qp_weight(&mut weights, &opt_trans);
    assert_eq!(weights.qp_full_weight.len(), 6); // M0309
    assert_eq!(weights.qp_full_weight[0], Complex64::new(0.5, 0.0)); // M0310
    assert_eq!(weights.qp_full_weight[1], Complex64::new(1.0, 0.0)); // M0311
    assert_eq!(weights.qp_full_weight[2], Complex64::new(1.5, 0.0)); // M0312
    assert_eq!(weights.qp_full_weight[3], Complex64::new(1.5, 0.0)); // M0313
    assert_eq!(weights.qp_full_weight[4], Complex64::new(3.0, 0.0)); // M0314
    assert_eq!(weights.qp_full_weight[5], Complex64::new(4.5, 0.0)); // M0315
    assert_eq!(weights.qp_fix_weight, real(&[1.0, 2.0, 3.0]));
    assert_eq!(opt_trans, real(&[0.5, 1.5]));
}

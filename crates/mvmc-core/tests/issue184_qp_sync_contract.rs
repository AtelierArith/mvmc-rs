//! Original Julia S128/S129 local synchronization contracts.
//!
//! C parameter.c:134–178 normalizes OptTrans, not ParaQPTrans.
//! These are the original synthetic constructors, not C runnable models.
use mvmc_core::{
    reducer::SingleProcessReducer,
    sync::{sync_modified_parameter, sync_modified_parameter_local},
};
use mvmc_expert_parsers::ExpertModeData;
use num_complex::Complex64;

fn bits(values: &[Complex64]) -> Vec<[u64; 2]> {
    values
        .iter()
        .map(|z| [z.re.to_bits(), z.im.to_bits()])
        .collect()
}

fn original_data() -> ExpertModeData {
    let mut data = ExpertModeData::new();
    data.n_qp_opt_trans = 1;
    data.para_qp_trans = vec![Complex64::new(1.0, 0.0), Complex64::new(-2.0, 2.0)];
    data
}

fn sync_and_check_fields(data: &mut ExpertModeData, broadcast: bool) {
    let original_para = bits(&data.para_qp_trans);
    let original_flags = data.optimization_flags.clone();
    let original_definition = bits(&data.para_qp_opt_trans);
    if broadcast {
        sync_modified_parameter(data, &SingleProcessReducer);
    } else {
        sync_modified_parameter_local(data, true);
    }
    assert_eq!(bits(&data.para_qp_trans), original_para);
    assert_eq!(bits(&data.para_qp_opt_trans), original_definition);
    assert_eq!(data.optimization_flags, original_flags);
    // Sync takes no RNG argument and uses no global RNG. This test makes no
    // RNG/sampling claim; deterministic trajectory evidence belongs to runner gates.
}

#[test]
fn original_s128_para_qp_trans_is_not_normalized() {
    for broadcast in [false, true] {
        let mut data = original_data();
        assert!(data.opt_trans.is_empty());
        sync_and_check_fields(&mut data, broadcast);
        assert_eq!(data.n_qp_opt_trans, 1);
        assert!(data.opt_trans.is_empty());
    }
}

#[test]
fn original_s129_opttrans_scales_by_five_without_changing_para_qp_trans() {
    for broadcast in [false, true] {
        let mut data = original_data();
        data.opt_trans = vec![Complex64::new(3.0, 4.0), Complex64::new(1.0, 0.0)];
        sync_and_check_fields(&mut data, broadcast);
        // Independent analytic max(|3+4i|,|1|)=5. Preserve original Julia atol1e-14.
        let expected = [Complex64::new(0.6, 0.8), Complex64::new(0.2, 0.0)];
        assert_eq!(data.opt_trans.len(), expected.len());
        for (actual, expected) in data.opt_trans.iter().zip(expected) {
            assert!((*actual - expected).norm() <= 1e-14);
        }
    }
}

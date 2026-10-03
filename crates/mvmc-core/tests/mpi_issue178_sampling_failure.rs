//! Phase 1: real kernel/sampler failure, before any MPI implementation.
//! Finite overflow operands are deliberate failure-boundary input, not a claim
//! that huge parameters are a supported C physical workload. No NaN injection,
//! FaultReducer, oracle runtime, or production observation hook is used.

use mvmc_core::{
    pfaffian::calc_m_all_fsz_real, sampling::vmc_make_sample_fsz_real,
    state::ThreadedPfaPackWorkspace, CalcMAllError, ExpertModeData, VmcOptimizationState,
};
use num_complex::Complex64;
use sfmt19937::Sfmt19937Rng;

fn data() -> ExpertModeData {
    let mut data = ExpertModeData::new();
    data.modpara.nsite = 2;
    data.modpara.nelec = 2;
    data.modpara.two_sz = 0;
    data.modpara.nvmc_warmup = 1;
    data.modpara.nvmc_sample = 3;
    data.modpara.nvmc_interval = 1;
    data.i_flg_orbital_general = 1;
    data.n_gutzwiller_idx = 1;
    data.gutzwiller_idx = vec![0; 2];
    data.gutzwiller_terms = (0..2)
        .map(|site| mvmc_expert_parsers::GutzwillerTerm {
            site,
            value: Complex64::new(0.0, 0.0),
            is_complex: false,
        })
        .collect();
    data
}

fn state() -> VmcOptimizationState {
    let mut state = VmcOptimizationState::zeros(2, 2, 1, 1, 1, 3, false, true);
    let scale = 2.0_f64.powi(600);
    assert!(scale.is_finite());
    for i in 0..4 {
        for j in i + 1..4 {
            state
                .slater_matrix
                .slater_elm
                .set(0, i, j, Complex64::new(scale, 0.0));
            state
                .slater_matrix
                .slater_elm
                .set(0, j, i, Complex64::new(-scale, 0.0));
        }
    }
    assert!(state
        .slater_matrix
        .slater_elm
        .as_slice()
        .iter()
        .all(|z| z.re.is_finite() && z.im.is_finite()));
    // Saved planes/counters are separate from initializer-owned tmp planes.
    state.electron_config.ele_idx.fill(7);
    state.electron_config.ele_cfg.fill(8);
    state.electron_config.ele_num.fill(9);
    state.electron_config.ele_proj_cnt.fill(10);
    state.electron_config.ele_spn.fill(11);
    state.electron_config.counter.fill(12);
    state.electron_config.counter[9] = 0; // Force genuine initialization.
    state
}

#[test]
fn finite_overflow_pfaffian_and_real_sampler_return_actual_error() {
    // Every principal 4x4 skew matrix here has Pfaffian +/- scale^2.
    // 2^1200 exceeds f64's exponent range, though every operand is finite.
    let mut kernel = state();
    let pool = ThreadedPfaPackWorkspace::new(4, 1);
    let error = calc_m_all_fsz_real(
        &[0, 1, 0, 1],
        &[0, 0, 1, 1],
        &mut kernel.slater_matrix,
        0,
        1,
        2,
        2,
        &pool,
    )
    .unwrap_err();
    assert_eq!(error, CalcMAllError::NonFinitePfaffian { qp: 0 });

    let data = data();
    let data_before = format!("{data:?}");
    let mut repeated_checkpoint = None;
    for trial in 0..2 {
        let mut state = state();
        let saved = state.electron_config.clone();
        let slater_bits = |state: &VmcOptimizationState| {
            state
                .slater_matrix
                .slater_elm
                .as_slice()
                .iter()
                .map(|z| (z.re.to_bits(), z.im.to_bits()))
                .collect::<Vec<_>>()
        };
        let slater_before = slater_bits(&state);
        let inverse_before = state.slater_matrix.inv_m.as_slice().to_vec();
        let pf_before = state.slater_matrix.pf_m.clone();
        let inverse_real_before = state.slater_matrix.inv_m_real.as_slice().to_vec();
        let pf_real_before = state.slater_matrix.pf_m_real.clone();
        let scratch_before = state.electron_config.tmp_ele_idx.clone();
        let mut rng = Sfmt19937Rng::new(1);
        rng.gen_rand32(); // Nonzero caller cursor/count, not a reseed repair.
        let before = rng.state_snapshot();
        let before_count = rng.words_consumed();
        let error = vmc_make_sample_fsz_real(&data, &mut state, &mut rng).unwrap_err();
        assert_eq!(error, CalcMAllError::NonFinitePfaffian { qp: 0 });
        assert_eq!(state.electron_config.ele_idx, saved.ele_idx);
        assert_eq!(state.electron_config.ele_cfg, saved.ele_cfg);
        assert_eq!(state.electron_config.ele_num, saved.ele_num);
        assert_eq!(state.electron_config.ele_proj_cnt, saved.ele_proj_cnt);
        assert_eq!(state.electron_config.ele_spn, saved.ele_spn);
        assert_eq!(state.electron_config.counter, saved.counter);
        assert_eq!(slater_bits(&state), slater_before);
        assert_eq!(format!("{data:?}"), data_before);
        // The 101 actual initializer attempts consume placement draws and may
        // publish tmp configuration and the real Slater copy, never saved
        // samples/counters. Failed factorization is staged privately and does
        // not publish inverse/Pfaffian results. The entire state/RNG is not unchanged.
        assert_ne!(state.electron_config.tmp_ele_idx, scratch_before);
        assert_eq!(state.electron_config.tmp_ele_num, vec![1; 4]);
        assert_eq!(state.electron_config.tmp_ele_spn, vec![0, 0, 1, 1]);
        assert_eq!(state.electron_config.tmp_ele_proj_cnt, vec![2]);
        assert_eq!(
            state.slater_matrix.inv_m.as_slice(),
            inverse_before,
            "failed factorization must not publish complex results"
        );
        assert_eq!(state.slater_matrix.pf_m, pf_before);
        assert_eq!(
            state.slater_matrix.inv_m_real.as_slice(),
            inverse_real_before
        );
        assert_eq!(state.slater_matrix.pf_m_real, pf_real_before);
        assert_eq!(
            state.slater_matrix.slater_elm_real.as_slice(),
            state
                .slater_matrix
                .slater_elm
                .as_slice()
                .iter()
                .map(|z| z.re)
                .collect::<Vec<_>>()
        );
        assert!(state
            .electron_config
            .tmp_ele_idx
            .iter()
            .all(|&site| (0..2).contains(&site)));
        assert!(rng.words_consumed() >= before_count + 101 * 4);
        let after = rng.state_snapshot();
        let after_count = rng.words_consumed();
        assert_ne!(after, before);
        let mut next624 = [0; 624];
        rng.dump_rand32(&mut next624);
        assert_eq!(
            rng.state_snapshot(),
            after,
            "peek must not advance raw state/cursor"
        );
        assert_eq!(rng.words_consumed(), after_count);
        let checkpoint = (after, after_count, next624);
        if let Some(previous) = &repeated_checkpoint {
            assert_eq!(
                &checkpoint, previous,
                "same input/seed failure is repeatable"
            );
        }
        repeated_checkpoint = Some(checkpoint);
        println!("ISSUE178_SERIAL trial={trial} error={error} before_count={before_count} after_count={after_count} before_cursor={} after_cursor={} scratch={:?}",
            before.1, after.1, state.electron_config.tmp_ele_idx);
        if std::env::var_os("MVMC_ISSUE178_SERIAL_DIAGNOSTIC").is_some() {
            println!(
                "ISSUE178_SERIAL before_raw={:?} after_raw={:?} next624={next624:?}",
                before.0, after.0
            );
        }
    }
}

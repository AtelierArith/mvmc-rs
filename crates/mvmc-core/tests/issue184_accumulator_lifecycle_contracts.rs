//! Public lifecycle adaptations of Julia test_unit/test_unit_threading.jl:
//! SR lines 119–160, PhysCal 162–222, aggregate 265–356.
//! Source SHA256 aa35b0d2eba231b97654fc12f586468513698da61924b21982e43bd3fd0b62b4.
//! No local-merge, cached-identity, or Julia-only store-clear equivalence claimed.
use mvmc_core::observables::{accumulate_two_body_gex_sample, clear_phys_quantity};
use mvmc_core::state::{PhysicalQuantities, SROptData, VmcOptimizationState};
use num_complex::Complex64;

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

#[test]
fn original_sr_operands_keep_sample_layout_and_reset_only_transient_buffers() {
    let mut state = VmcOptimizationState::zeros(2, 1, 1, 1, 1, 3, false, false);
    state.sr_opt = SROptData::zeros(2, 3, false);
    let sr = &mut state.sr_opt;
    assert_eq!(
        [
            sr.sr_opt_oo.len(),
            sr.sr_opt_ho.len(),
            sr.sr_opt_o.len(),
            sr.sr_opt_o_store.len()
        ],
        [24, 4, 4, 12]
    );
    assert_eq!(
        [
            sr.sr_opt_oo_real.len(),
            sr.sr_opt_ho_real.len(),
            sr.sr_opt_o_real.len(),
            sr.sr_opt_o_store_real.len()
        ],
        [8, 2, 2, 6]
    );
    sr.sr_opt_oo[..3].copy_from_slice(&[c(1., 1.), c(2., 0.), c(-1., 2.)]);
    sr.sr_opt_ho[..2].copy_from_slice(&[c(2., 0.), c(1., -1.)]);
    sr.sr_opt_o.fill(c(9., -2.));
    sr.sr_opt_oo_real[..3].copy_from_slice(&[1., 2., 3.]);
    sr.sr_opt_ho_real.copy_from_slice(&[7., 8.]);
    sr.sr_opt_o_real.fill(9.);
    sr.sr_opt_o_store_slice_mut(0)[..2].copy_from_slice(&[c(3., 0.), c(4., 0.)]);
    sr.sr_opt_o_store_slice_mut(1)[..2].copy_from_slice(&[c(5., 0.), c(6., 0.)]);
    sr.sr_opt_o_store_real_slice_mut(0)
        .copy_from_slice(&[11., 12.]);
    sr.sr_opt_o_store_real_slice_mut(1)
        .copy_from_slice(&[13., 14.]);
    let complex_store = vec![
        c(3., 0.),
        c(4., 0.),
        c(0., 0.),
        c(0., 0.),
        c(5., 0.),
        c(6., 0.),
        c(0., 0.),
        c(0., 0.),
        c(0., 0.),
        c(0., 0.),
        c(0., 0.),
        c(0., 0.),
    ];
    assert_eq!(sr.sr_opt_o_store, complex_store);
    assert_eq!(sr.sr_opt_o_store_real, [11., 12., 13., 14., 0., 0.]);
    clear_phys_quantity(&mut state);
    for values in [
        &state.sr_opt.sr_opt_oo,
        &state.sr_opt.sr_opt_ho,
        &state.sr_opt.sr_opt_o,
    ] {
        assert!(values.iter().all(|&value| value == c(0., 0.)));
    }
    for values in [
        &state.sr_opt.sr_opt_oo_real,
        &state.sr_opt.sr_opt_ho_real,
        &state.sr_opt.sr_opt_o_real,
    ] {
        assert!(values.iter().all(|&value| value == 0.));
    }
    // Rust's public reset preserves stored samples; Julia local store-clear is separate.
    assert_eq!(state.sr_opt.sr_opt_o_store, complex_store);
    assert_eq!(
        state.sr_opt.sr_opt_o_store_real,
        [11., 12., 13., 14., 0., 0.]
    );
}

#[test]
fn original_factored_green_uses_sample_values_and_conjugates_second_factor() {
    let sample_values = [c(4., 2.), c(-1., 3.)];
    let mut result = [c(0., 0.)];
    accumulate_two_body_gex_sample(&mut result, &sample_values, &[(0, 1)], c(0.5, 0.));
    // Independent literal: .5*(4+2i)*(-1-3i) = 1-7i.
    // At most eight scalar multiply/add operations, operands <=4, result <=7:
    // 16 epsilon absolute plus 8 epsilon relative is a conservative rounding budget.
    for (actual, expected) in [(result[0].re, 1.), (result[0].im, -7.)] {
        assert!(actual.is_finite());
        assert!(
            (actual - expected).abs() <= 16. * f64::EPSILON + 8. * f64::EPSILON * expected.abs()
        );
    }
    assert_eq!(sample_values, [c(4., 2.), c(-1., 3.)]);
}

#[test]
fn original_aggregate_dimensions_reset_phys_and_preserve_workspace_in_both_modes() {
    for complex in [false, true] {
        let mut state = VmcOptimizationState::zeros(2, 1, 1, 2, 1, 2, complex, false);
        state.phys_quantities = Some(PhysicalQuantities::zeros(1, 1, 1));
        assert_eq!(state.sr_opt.sr_opt_size, 3);
        assert_eq!(state.sr_opt.sr_opt_oo.len(), 48);
        assert_eq!(state.sr_opt.sr_opt_o.len(), 6);
        assert_eq!(state.workspace.proj_cnt_new.len(), 1);
        assert_eq!(state.workspace.pf_m_new_real.len(), 1);
        state.workspace.proj_cnt_new[0] = 9;
        state.workspace.pf_m_new_real[0] = 17.;
        state.energy.wc = c(2., 0.);
        state.energy.etot = c(7., 0.);
        state.energy.etot2 = c(12.5, 0.);
        state.energy.sztot = c(5.5, 0.);
        state.energy.sztot2 = c(18.5, 0.);
        state.sr_opt.sr_opt_oo.fill(c(11., 0.));
        state.sr_opt.sr_opt_ho.fill(c(12., 0.));
        state.sr_opt.sr_opt_o.fill(c(21., 1.));
        state.sr_opt.sr_opt_oo_real.fill(31.);
        state.sr_opt.sr_opt_ho_real.fill(32.);
        state.sr_opt.sr_opt_o_real.fill(34.);
        state.sr_opt.sr_opt_o_store_real.fill(33.);
        let phys = state.phys_quantities.as_mut().unwrap();
        assert_eq!(
            [
                phys.phys_lanczos_qqqq.len(),
                phys.phys_lanczos_qcisajsq.len(),
                phys.phys_lanczos_qcisajscktaltq.len(),
                phys.phys_lanczos_qcisajscktaltq_dc.len()
            ],
            [16, 4, 4, 4]
        );
        for values in [
            &mut phys.phys_lanczos_qqqq,
            &mut phys.phys_lanczos_qcisajsq,
            &mut phys.phys_lanczos_qcisajscktaltq,
            &mut phys.phys_lanczos_qcisajscktaltq_dc,
            &mut phys.local_cis_ajs,
            &mut phys.phys_cis_ajs,
            &mut phys.phys_cis_ajs_ckt_alt,
            &mut phys.local_cis_ajs_ckt_alt_dc,
            &mut phys.phys_cis_ajs_ckt_alt_dc,
        ] {
            values.fill(c(13., 1.));
        }
        clear_phys_quantity(&mut state);
        assert_eq!(
            [
                state.energy.wc,
                state.energy.etot,
                state.energy.etot2,
                state.energy.sztot,
                state.energy.sztot2
            ],
            [c(0., 0.); 5]
        );
        for (values, length) in [
            (&state.sr_opt.sr_opt_oo, 48),
            (&state.sr_opt.sr_opt_ho, 6),
            (&state.sr_opt.sr_opt_o, 6),
        ] {
            assert_eq!(values.len(), length);
            assert!(values.iter().all(|&x| x == c(0., 0.)));
        }
        assert_eq!(
            state.sr_opt.sr_opt_oo_real,
            if complex { vec![] } else { vec![0.; 15] }
        );
        assert_eq!(
            state.sr_opt.sr_opt_ho_real,
            if complex { vec![] } else { vec![0.; 3] }
        );
        assert_eq!(
            state.sr_opt.sr_opt_o_real,
            if complex { vec![] } else { vec![0.; 3] }
        );
        assert_eq!(
            state.sr_opt.sr_opt_o_store_real,
            if complex { vec![] } else { vec![33.; 6] }
        );
        let phys = state
            .phys_quantities
            .as_ref()
            .expect("reset preserves PhysCal mode");
        // Independent literal lengths, not a second call to the constructor under test.
        for (values, length) in [
            (&phys.phys_lanczos_qqqq, 16),
            (&phys.phys_lanczos_qcisajsq, 4),
            (&phys.phys_lanczos_qcisajscktaltq, 4),
            (&phys.phys_lanczos_qcisajscktaltq_dc, 4),
            (&phys.local_cis_ajs, 1),
            (&phys.phys_cis_ajs, 1),
            (&phys.phys_cis_ajs_ckt_alt, 1),
            (&phys.local_cis_ajs_ckt_alt_dc, 1),
            (&phys.phys_cis_ajs_ckt_alt_dc, 1),
        ] {
            assert_eq!(values.len(), length);
            assert!(values.iter().all(|&x| x == c(0., 0.)));
        }
        assert_eq!(state.workspace.proj_cnt_new, [9]);
        assert_eq!(state.workspace.pf_m_new_real, [17.]);
    }
}

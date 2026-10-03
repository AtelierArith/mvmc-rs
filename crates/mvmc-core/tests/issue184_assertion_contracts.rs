//! Focused non-InterAll assertion contracts from pinned Julia 8bb1b9e8.
//! C parameter.c:134-152,202-252,255-313 is authoritative for DH flags/order.
//! SHA-256: 46ad04622f4475337028cee633bd76ce318a6d5058d03f202b55204500399fb0.
//! Expectations below are analytical constants, not Rust-generated fixtures.
use mvmc_core::parallel::partition_range;
use mvmc_core::sync::sync_modified_parameter_local;
use mvmc_expert_parsers::{
    DoublonHolon2SiteIndex, DoublonHolon4SiteIndex, ExpertModeData, GutzwillerTerm, JastrowTerm,
};
use num_complex::Complex64;

fn dh_model(four: bool) -> ExpertModeData {
    let mut data = ExpertModeData::new();
    data.modpara.nsite = 2;
    data.gutzwiller_terms.push(GutzwillerTerm {
        site: 0,
        value: Complex64::new(if four { 20.0 } else { 10.0 }, 0.0),
        is_complex: true,
    });
    if four {
        data.doublon_holon_4site_indices
            .push(DoublonHolon4SiteIndex {
                neighbors: vec![[1, 0, 1, 0], [0, 1, 0, 1]],
            });
        data.doublon_holon_4site_params = (1..=10)
            .map(|i| Complex64::new(f64::from(i), 0.0))
            .collect();
    } else {
        data.doublon_holon_2site_indices
            .push(DoublonHolon2SiteIndex {
                neighbors: vec![[1, 0], [0, 1]],
            });
        data.doublon_holon_2site_params =
            (1..=6).map(|i| Complex64::new(f64::from(i), 0.0)).collect();
    }
    data.optimization_flags = vec![1; 2 * data.projection_layout().n_proj];
    data
}

#[test]
fn fixed_gutzwiller_real_flag_prevents_dh2_shift() {
    let mut data = dh_model(false);
    data.optimization_flags[0] = 0;
    let original = data.projection_parameters();
    let flags = data.optimization_flags.clone();
    sync_modified_parameter_local(&mut data, true);
    assert_eq!(data.projection_parameters(), original);
    assert_eq!(data.optimization_flags, flags);
}

#[test]
fn absent_explicit_flags_do_not_enable_dh2_shift() {
    // Julia missing-flag policy is a synthetic API boundary, not a C empty
    // OptFlag allocation: C's valid model has an allocated flag per component.
    let mut data = dh_model(false);
    data.optimization_flags.clear();
    let original = data.projection_parameters();
    sync_modified_parameter_local(&mut data, true);
    assert_eq!(data.projection_parameters(), original);
    assert!(data.optimization_flags.is_empty());
}

#[test]
fn fixed_dh_real_flag_prevents_each_dh_family_shift() {
    for four in [false, true] {
        let mut data = dh_model(four);
        let layout = data.projection_layout();
        let offset = if four {
            layout.dh4_offset
        } else {
            layout.dh2_offset
        };
        data.optimization_flags[2 * offset] = 0;
        let original = data.projection_parameters();
        let flags = data.optimization_flags.clone();
        sync_modified_parameter_local(&mut data, true);
        assert_eq!(data.projection_parameters(), original, "DH4={four}");
        assert_eq!(data.optimization_flags, flags);
    }
}

#[test]
fn every_dh_real_component_can_independently_disable_its_family_shift() {
    // C SetFlagShift visits OptFlag[2*i] for all six/ten declared DH slots.
    for (four, width) in [(false, 6), (true, 10)] {
        for slot in 0..width {
            let mut data = dh_model(four);
            let layout = data.projection_layout();
            let offset = if four {
                layout.dh4_offset
            } else {
                layout.dh2_offset
            };
            data.optimization_flags[2 * (offset + slot)] = 0;
            let original = data.projection_parameters();
            let flags = data.optimization_flags.clone();
            sync_modified_parameter_local(&mut data, true);
            assert_eq!(
                data.projection_parameters(),
                original,
                "DH4={four}, slot={slot}"
            );
            assert_eq!(data.optimization_flags, flags);
        }
    }
}

#[test]
fn fixed_imaginary_flags_do_not_disable_c_real_dh_shift_selection() {
    for four in [false, true] {
        let mut data = dh_model(four);
        for component in data.optimization_flags.iter_mut().skip(1).step_by(2) {
            *component = 0;
        }
        let flags = data.optimization_flags.clone();
        sync_modified_parameter_local(&mut data, true);
        assert_eq!(data.optimization_flags, flags);
        if four {
            // Means of alternating five-bin groups are 5 and 6.
            assert_eq!(data.gutzwiller_terms[0].value, Complex64::new(31.0, 0.0));
            assert_eq!(
                data.doublon_holon_4site_params,
                [-4.0, -4.0, -2.0, -2.0, 0.0, 0.0, 2.0, 2.0, 4.0, 4.0]
                    .map(|x| Complex64::new(x, 0.0))
            );
        } else {
            assert_eq!(data.gutzwiller_terms[0].value, Complex64::new(17.0, 0.0));
            assert_eq!(
                data.doublon_holon_2site_params,
                [-2.0, -2.0, 0.0, 0.0, 2.0, 2.0].map(|x| Complex64::new(x, 0.0))
            );
        }
    }
}

#[test]
fn c_dh_shift_precedes_gutzwiller_jastrow_shift() {
    let mut data = dh_model(false);
    data.jastrow_terms.push(JastrowTerm {
        site1: 0,
        site2: 1,
        value: Complex64::new(20.0, 0.0),
        is_complex: true,
    });
    data.optimization_flags = vec![1; 2 * data.projection_layout().n_proj];
    let flags = data.optimization_flags.clone();
    sync_modified_parameter_local(&mut data, true);
    // DH means (1+3+5)/3=3 and (2+4+6)/3=4 compensate Gutz 10 ->17.
    // The subsequent GJ mean (17+20)/2=18.5 yields -1.5 and +1.5.
    assert_eq!(data.gutzwiller_terms[0].value, Complex64::new(-1.5, 0.0));
    assert_eq!(data.jastrow_terms[0].value, Complex64::new(1.5, 0.0));
    assert_eq!(
        data.doublon_holon_2site_params,
        [-2.0, -2.0, 0.0, 0.0, 2.0, 2.0].map(|x| Complex64::new(x, 0.0))
    );
    assert_eq!(data.optimization_flags, flags);
}

#[test]
fn physcal_invalid_mode_is_rejected_before_missing_input_access() {
    let error = mvmc_core::prepare_phys_cal_from_namelist(
        "nonexistent_namelist.def",
        "nonexistent_fixed.dat",
        "bogus",
        Some(1),
    )
    .unwrap_err();
    assert_eq!(error, "mode must be :real, :cmp, or :fsz; got :bogus");
}

#[test]
fn original_julia_splitloop_eight_cases_match_c_for_every_rank() {
    for (length, size, expected) in [
        (10, 2, vec![0..5, 5..10]),
        (10, 3, vec![0..3, 3..6, 6..10]),
        (3, 2, vec![0..1, 1..3]),
        (5, 4, vec![0..1, 1..2, 2..3, 3..5]),
        (4, 4, vec![0..1, 1..2, 2..3, 3..4]),
        (2, 4, vec![0..1, 1..2, 2..2, 2..2]),
        (1, 4, vec![0..1, 1..1, 1..1, 1..1]),
        (0, 2, vec![0..0, 0..0]),
    ] {
        for (rank, expected) in expected.into_iter().enumerate() {
            assert_eq!(
                partition_range(length, size, rank),
                expected,
                "length={length} size={size} rank={rank}"
            );
        }
    }
    // Julia split_range's 1-based inclusive representation is intentionally
    // replaced by Rust's 0-based half-open Range, including empty ranges.
    assert_eq!(partition_range(10, 3, 0), 0..3);
    assert_eq!(partition_range(10, 3, 2), 6..10);
    assert!(partition_range(2, 4, 3).is_empty());
}

#[test]
fn projection_count_helper_replacement_preserves_declared_sparse_and_dh_widths() {
    // Julia expert_types.jl:904 is projection_layout(data).n_proj; C reserves
    // six/ten coefficients per DH2/DH4 index, irrespective of mapped values.
    for (gutz, jastrow, dh2, dh4, expected) in [
        (0, 0, 0, 0, 0),
        (3, 4, 0, 0, 7),
        (1, 0, 1, 0, 7),
        (0, 0, 0, 1, 10),
        (3, 4, 1, 1, 23),
    ] {
        let mut data = ExpertModeData::new();
        data.n_gutzwiller_idx = gutz;
        data.n_jastrow_idx = jastrow;
        // Declared nonzero blocks deliberately have no mapped coefficients.
        data.doublon_holon_2site_indices = (0..dh2)
            .map(|_| DoublonHolon2SiteIndex { neighbors: vec![] })
            .collect();
        data.doublon_holon_4site_indices = (0..dh4)
            .map(|_| DoublonHolon4SiteIndex { neighbors: vec![] })
            .collect();
        assert_eq!(data.projection_layout().n_proj, expected);
        assert_eq!(data.projection_parameters().len(), expected);
        assert!(data.optimization_flags.is_empty());
    }
    // Synthetic layout helper contract, not acceptance of incomplete runtime input.
    let data = dh_model(false);
    assert_eq!(data.projection_layout().n_proj, 7);
    assert_eq!(data.gutzwiller_terms[0].value, Complex64::new(10.0, 0.0));
}

#[test]
fn opttrans_count_helper_returns_stored_length_without_mutation() {
    // Julia read_input_parameters.jl:161 and Rust both return stored length;
    // this does not claim that every synthetic payload is runnable PhysCal.
    for length in [0, 1, 2, 7] {
        let mut data = ExpertModeData::new();
        data.opt_trans = vec![Complex64::new(0.5, -0.25); length];
        let original = data.opt_trans.clone();
        assert_eq!(data.count_opt_trans_parameters(), length);
        assert_eq!(data.opt_trans, original);
        assert!(data.optimization_flags.is_empty());
    }
}

#[test]
fn combined_dh_projection_packing_and_component_flags_match_original_source_contract() {
    // Julia test_doublon_holon_parser.jl:134-172; synthetic layout/flag
    // assembly, not a runnable model (the two DH tables have different rows).
    // Six/ten declared coefficients and real-mode imaginary zero flags follow C.
    let mut data = dh_model(false);
    data.jastrow_terms.push(JastrowTerm {
        site1: 0,
        site2: 1,
        value: Complex64::new(20.0, 0.0),
        is_complex: false,
    });
    data.doublon_holon_4site_indices
        .push(DoublonHolon4SiteIndex {
            neighbors: vec![[1, 0, 1, 0], [0, 1, 0, 1]],
        });
    data.doublon_holon_4site_params = (11..=20)
        .map(|value| Complex64::new(f64::from(value), 0.0))
        .collect();
    data.doublon_holon_2site_complex = true;
    data.doublon_holon_4site_complex = false;
    data.doublon_holon_2site_opt_flags = vec![1, 0, 1, 0, 1, 0];
    data.doublon_holon_4site_opt_flags = vec![0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
    let layout = data.projection_layout();
    assert_eq!(
        (
            layout.n_gutzwiller,
            layout.n_jastrow,
            layout.n_dh2,
            layout.n_dh4
        ),
        (1, 1, 1, 1)
    );
    assert_eq!(
        (layout.dh2_offset, layout.dh4_offset, layout.n_proj),
        (2, 8, 18)
    );
    let expected: Vec<_> = [
        10, 20, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    ]
    .into_iter()
    .map(|value| Complex64::new(f64::from(value), 0.0))
    .collect();
    assert_eq!(data.projection_parameters(), expected);
    data.optimization_flags = vec![1; 36];
    mvmc_expert_parsers::utils::opt_flag::set_dh_opt_flags(&mut data);
    assert_eq!(&data.optimization_flags[..4], &[1, 1, 1, 1]);
    assert_eq!(
        &data.optimization_flags[4..16],
        &[1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
    );
    assert_eq!(
        &data.optimization_flags[16..36],
        &[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0]
    );
    assert_eq!(data.projection_parameters(), expected);
}

#[test]
fn fsz_lanczos_validation_rejects_both_original_julia_mode_expansions() {
    // S212 is a synthetic validation-only case, not numerical C-FSZ support.
    for mode in [1, 2] {
        let mut data = ExpertModeData::new();
        data.modpara.lanczos_mode = mode;
        data.modpara.nmp_trans = 1;
        data.i_flg_orbital_general = 1;
        let error = mvmc_core::validation::validate_phys_cal(&data).unwrap_err();
        assert!(error.contains("FSZ/general"), "mode={mode}: {error}");
        assert_eq!(data.modpara.lanczos_mode, mode);
        assert_eq!(data.i_flg_orbital_general, 1);
    }
}

#[test]
fn global_lanczos_validator_expands_original_valid_and_invalid_modes() {
    // Julia unsupported_inputs.jl:273-288. Global validation is not a
    // promise that every Hamiltonian or runner accepts these modes.
    for mode in [0, 1, 2] {
        let mut data = ExpertModeData::new();
        data.modpara.lanczos_mode = mode;
        assert_eq!(
            mvmc_core::validation::validate_supported_modpara(&data.modpara),
            Ok(())
        );
        assert_eq!(data.modpara.lanczos_mode, mode);
    }
    for mode in [-1, 3] {
        let mut data = ExpertModeData::new();
        data.modpara.lanczos_mode = mode;
        let error = mvmc_core::validation::validate_supported_modpara(&data.modpara).unwrap_err();
        assert_eq!(
            error,
            format!("NLanczosMode must be 0, 1, or 2; got {mode}")
        );
        assert_eq!(data.modpara.lanczos_mode, mode);
    }
}

#[test]
fn original_cg_option_loop_rejects_each_nonzero_submode_independently() {
    // Synthetic Julia validation contract: no solver run or C numerical
    // equivalence is implied by these unsupported-option diagnostics.
    for diagonal in [true, false] {
        let mut data = ExpertModeData::new();
        let label = if diagonal {
            data.modpara.use_diag_scale = 1;
            "useDiagScale != 0"
        } else {
            data.modpara.rescale_smat = 1;
            "RescaleSmat != 0"
        };
        let error = mvmc_core::validation::validate_supported_modpara(&data.modpara).unwrap_err();
        assert!(error.contains(label));
        assert!(error.contains("not supported"));
        assert_eq!(data.modpara.use_diag_scale, i64::from(diagonal));
        assert_eq!(data.modpara.rescale_smat, i64::from(!diagonal));
    }
}

#[test]
fn spin_flip_transfer_lanczos_rejection_has_mode_independent_wording() {
    // S213's Transfer branch only. Its separate InterAll branch is excluded.
    // Julia says "spin-flip Transfer"; Rust says "spin-changing". This test
    // preserves the existing diagnostic difference, rather than silently
    // claiming identical exception text or changing production validation.
    for mode in [1, 2] {
        let mut data = ExpertModeData::new();
        data.modpara.lanczos_mode = mode;
        data.modpara.nmp_trans = 1;
        data.transfer_terms.push(mvmc_expert_parsers::TransferTerm {
            site1: 0,
            spin1: mvmc_expert_parsers::Spin::Up,
            site2: 1,
            spin2: mvmc_expert_parsers::Spin::Down,
            value: Complex64::new(1.0, 0.0),
        });
        let error = mvmc_core::validation::validate_phys_cal(&data).unwrap_err();
        assert!(error.contains("spin-changing"), "mode={mode}: {error}");
        assert!(!error.contains("R1"), "mode={mode}: {error}");
        assert_eq!(data.transfer_terms.len(), 1);
        assert_eq!(data.transfer_terms[0].value, Complex64::new(1.0, 0.0));
    }
}
#[test]
fn green_two_ex_valid_c_row_preserves_first_pair_and_reverses_second_pair() {
    // C readdef.c:GetInfoTwoBodyGEx reads x0..x7 and stores one-body indices
    // with (x0,x1,x2,x3), then (x6,x7,x4,x5). Julia's literal valid row
    // is reproduced; malformed-input rejection is deliberately not asserted.
    let content = "=============================================\nNCisAjsCktAlt 1\n=============================================\n======== Factored two-body Green ============\n=============================================\n0 0 1 0 2 1 3 1\n";
    let terms = mvmc_expert_parsers::parsers::green::parse_green_two_ex_content(content).unwrap();
    use mvmc_expert_parsers::{GreenTwoExTerm, Spin};
    assert_eq!(
        terms,
        vec![GreenTwoExTerm {
            site1: 0,
            spin1: Spin::Up,
            site2: 1,
            spin2: Spin::Up,
            site3: 3,
            spin3: Spin::Down,
            site4: 2,
            spin4: Spin::Down,
        }]
    );
}
#[test]
fn original_qp_legendre_midpoint_and_opttrans_major_products() {
    use mvmc_expert_parsers::utils::qp_weight::{gauss_legendre, legendre_poly, update_qp_weight};
    let (nodes, weights) = gauss_legendre(0.0, 1.0, 1);
    assert_eq!(nodes.len(), 1);
    assert_eq!(weights.len(), 1);
    assert!((nodes[0] - 0.5).abs() < 1e-10);
    assert!((weights[0] - 1.0).abs() < 1e-10);
    // Analytic P0..P3 at 1/2, not values generated by the Rust recurrence.
    for (degree, expected) in [(0, 1.0), (1, 0.5), (2, -0.125), (3, -0.4375)] {
        assert!((legendre_poly(0.5, degree) - expected).abs() < 1e-10);
    }
    let mut weights = mvmc_expert_parsers::QuantumProjectionWeights::new();
    weights.qp_fix_weight = [1.0, 2.0, 3.0].map(|x| Complex64::new(x, 0.0)).to_vec();
    update_qp_weight(&mut weights, &[]);
    assert_eq!(weights.qp_full_weight, weights.qp_fix_weight);
    update_qp_weight(
        &mut weights,
        &[Complex64::new(0.5, 0.0), Complex64::new(1.5, 0.0)],
    );
    assert_eq!(
        weights.qp_full_weight,
        [0.5, 1.0, 1.5, 1.5, 3.0, 4.5].map(|x| Complex64::new(x, 0.0))
    );
}
#[test]
fn original_type_sr_buffers_use_size_three_and_seven_samples() {
    for complex in [true, false] {
        let sr = mvmc_core::SROptData::zeros(3, 7, complex);
        assert_eq!(sr.sr_opt_size, 3);
        assert_eq!(sr.sr_opt_oo.len(), 48);
        assert_eq!(sr.sr_opt_ho.len(), 6);
        assert_eq!(sr.sr_opt_o.len(), 6);
        assert_eq!(sr.sr_opt_o_store.len(), 42);
        let expected = if complex {
            [0, 0, 0, 0]
        } else {
            [15, 3, 3, 21]
        };
        assert_eq!(
            [
                sr.sr_opt_oo_real.len(),
                sr.sr_opt_ho_real.len(),
                sr.sr_opt_o_real.len(),
                sr.sr_opt_o_store_real.len(),
            ],
            expected
        );
    }
}

#[test]
fn original_type_electron_buffers_distinguish_normal_and_fsz() {
    for fsz in [false, true] {
        let ec = mvmc_core::ElectronConfiguration::zeros(5, 4, 3, 11, fsz);
        assert_eq!(
            [
                ec.ele_idx.len(),
                ec.ele_cfg.len(),
                ec.ele_num.len(),
                ec.ele_proj_cnt.len()
            ],
            [30, 40, 40, 55]
        );
        assert_eq!(
            [
                ec.tmp_ele_idx.len(),
                ec.tmp_ele_cfg.len(),
                ec.tmp_ele_num.len(),
                ec.tmp_ele_proj_cnt.len()
            ],
            [6, 8, 8, 11]
        );
        assert_eq!(
            [
                ec.burn_ele_cfg.len(),
                ec.burn_ele_num.len(),
                ec.burn_ele_proj_cnt.len()
            ],
            [8, 8, 11]
        );
        assert_eq!(ec.counter.len(), 10);
        assert_eq!(
            [
                ec.ele_spn.len(),
                ec.tmp_ele_spn.len(),
                ec.burn_ele_spn.len(),
                ec.burn_ele_idx.len()
            ],
            if fsz { [30, 6, 6, 39] } else { [0, 0, 0, 33] }
        );
    }
}
#[test]
fn original_sampling_hop_and_revert_preserve_exact_electron_buffers() {
    let mut idx = [0, 2, 1, 3];
    let mut cfg = [0, -1, 1, -1, -1, 0, -1, 1];
    let mut num = [1, 0, 1, 0, 0, 1, 0, 1];
    let original = (idx, cfg, num);
    mvmc_core::sampling::update_ele_config(0, 0, 1, 0, &mut idx, &mut cfg, &mut num, 4, 2);
    assert_eq!(idx, [1, 2, 1, 3]);
    assert_eq!(cfg, [-1, 0, 1, -1, -1, 0, -1, 1]);
    assert_eq!(num, [0, 1, 1, 0, 0, 1, 0, 1]);
    mvmc_core::sampling::revert_ele_config(0, 0, 1, 0, &mut idx, &mut cfg, &mut num, 4, 2);
    assert_eq!((idx, cfg, num), original);
}
#[test]
fn original_sampling_dh2_literal_counts_and_real_parameter_log_contract() {
    let mut data = mvmc_expert_parsers::ExpertModeData::new();
    data.modpara.nsite = 4;
    data.doublon_holon_2site_indices
        .push(mvmc_expert_parsers::DoublonHolon2SiteIndex {
            neighbors: vec![[1, 2], [0, 2], [0, 1], [0, 1]],
        });
    data.doublon_holon_2site_params = (1..=6)
        .map(|i| Complex64::new(i as f64, 10.0 * i as f64))
        .collect();
    let mut counts = [99; 6];
    // Site0 doublon / site1 holon / sites2,3 singly occupied.
    mvmc_core::sampling::make_proj_cnt(&mut counts, &[1, 0, 1, 0, 1, 0, 0, 1], &data);
    assert_eq!(counts, [0, 0, 1, 1, 0, 0]);
    // C projection.c uses creal(Proj): literal exponent3+4=7,
    // new exponent2+8+6=16, difference9. Imaginary parts are irrelevant.
    assert_eq!(mvmc_core::sampling::log_proj_val(&counts, &data), 7.0);
    assert_eq!(
        mvmc_core::sampling::log_proj_ratio(&[0, 1, 0, 2, 0, 1], &counts, &data),
        9.0
    );
}

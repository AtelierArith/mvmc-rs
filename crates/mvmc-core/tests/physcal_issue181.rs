//! Issue #181: serial PhysCal and Lanczos contracts.
//!
//! Numerical references are native C 1.4.0 runs (`native_c_physcal_181`,
//! `native_c_lanczos_576`) and standalone C kernel records; the inputs come from
//! `tests/fixtures/physcal_181`. The Julia CPU trajectory comparisons were
//! removed in #695.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

#[test]
fn original_io134_physical_quantities_shapes_are_explicit() {
    // S137/M0614-M0622: original constructor counts, not a model-sized proxy.
    let physical = mvmc_core::state::PhysicalQuantities::zeros(2, 1, 3);
    assert_eq!(physical.phys_cis_ajs.len(), 2);
    assert_eq!(physical.phys_cis_ajs_ckt_alt.len(), 1);
    assert_eq!(physical.phys_cis_ajs_ckt_alt_dc.len(), 3);
    assert_eq!(physical.phys_lanczos_qqqq.len(), 16);
    assert_eq!(physical.phys_lanczos_qcisajsq.len(), 8);
    assert_eq!(physical.phys_lanczos_qcisajscktaltq.len(), 4);
    assert_eq!(physical.phys_lanczos_qcisajscktaltq_dc.len(), 12);
    // Julia stores canonical index vectors on PhysicalQuantities; Rust keeps
    // them on parsed ExpertModeData. Do not invent duplicate state storage.
}

#[test]
fn original_io134_canonical_pairs_keep_c_file_order_and_append_constituents() {
    use mvmc_expert_parsers::{ExpertModeData, GreenOneTerm, GreenTwoExTerm, Spin};
    let one = |first, second, spin| GreenOneTerm {
        site1: first,
        site2: second,
        spin1: spin,
        spin2: spin,
    };
    let a = one(0, 1, Spin::Up);
    let b = one(1, 0, Spin::Down);
    let c = one(2, 3, Spin::Down);
    let mut data = ExpertModeData::new();
    data.green_one_terms = vec![a, b];
    data.canonicalize_green_two_ex();
    assert_eq!(data.green_one_terms, [a, b]);
    assert!(data.green_two_ex_indices.is_empty());
    // No GEx: preserve duplicates and input order, unlike unconditional dedup.
    data.green_one_terms = vec![a, a, b];
    data.canonicalize_green_two_ex();
    assert_eq!(data.green_one_terms, [a, a, b]);
    // GEx: C deduplicates initial rows, then appends missing constituents in
    // term order. Pair indices are zero-based in Rust, one-based in Julia.
    data.green_one_terms = vec![a, a];
    data.green_two_ex_terms = vec![GreenTwoExTerm {
        site1: 0,
        site2: 1,
        site3: 2,
        site4: 3,
        spin1: Spin::Up,
        spin2: Spin::Up,
        spin3: Spin::Down,
        spin4: Spin::Down,
    }];
    data.canonicalize_green_two_ex();
    assert_eq!(data.green_one_terms, [a, c]);
    assert_eq!(data.green_two_ex_indices, [(0, 1)]);
    let swapped = GreenTwoExTerm {
        site1: 2,
        site2: 3,
        site3: 0,
        site4: 1,
        spin1: Spin::Down,
        spin2: Spin::Down,
        spin3: Spin::Up,
        spin4: Spin::Up,
    };
    data.green_two_ex_terms.push(swapped);
    data.canonicalize_green_two_ex();
    assert_eq!(data.green_one_terms, [a, c]);
    assert_eq!(data.green_two_ex_indices, [(0, 1), (1, 0)]);
    let physical = mvmc_core::state::PhysicalQuantities::zeros(
        data.green_one_terms.len(),
        data.green_two_ex_indices.len(),
        0,
    );
    assert_eq!(physical.phys_cis_ajs.len(), 2);
    assert_eq!(physical.local_cis_ajs.len(), 2);
    assert_eq!(physical.phys_cis_ajs_ckt_alt.len(), 2);
    assert!(physical.phys_cis_ajs_ckt_alt_dc.is_empty());
    data.green_one_terms = vec![a];
    data.green_two_ex_terms = vec![GreenTwoExTerm {
        site1: 0,
        site2: 1,
        site3: 0,
        site4: 1,
        spin1: Spin::Up,
        spin2: Spin::Up,
        spin3: Spin::Up,
        spin4: Spin::Up,
    }];
    data.canonicalize_green_two_ex();
    assert_eq!(data.green_one_terms, [a]);
    assert_eq!(data.green_two_ex_indices, [(0, 0)]);
}

#[test]
fn original_io134_normal_writer_keeps_literal_canonical_row_and_complex_pair() {
    use mvmc_expert_parsers::{ExpertModeData, GreenOneTerm, GreenTwoExTerm, Spin};
    use num_complex::Complex64;
    let mut data = ExpertModeData::new();
    data.modpara.nsite = 4;
    data.modpara.n_data_idx_start = 1;
    data.green_one_terms = vec![GreenOneTerm {
        site1: 0,
        site2: 1,
        spin1: Spin::Up,
        spin2: Spin::Up,
    }];
    data.green_two_ex_indices = vec![(0, 0)];
    // C file presence follows declared NTwoBodyGEx, unlike Julia's standalone
    // output helper which can supply only the PhysicalQuantities pair list.
    data.green_two_ex_terms = vec![GreenTwoExTerm {
        site1: 0,
        site2: 1,
        site3: 0,
        site4: 1,
        spin1: Spin::Up,
        spin2: Spin::Up,
        spin3: Spin::Up,
        spin4: Spin::Up,
    }];
    let mut state = mvmc_core::VmcOptimizationState::zeros(4, 2, 0, 0, 1, 4, true, false);
    let mut physical = mvmc_core::state::PhysicalQuantities::zeros(1, 1, 0);
    physical.phys_cis_ajs[0] = Complex64::new(1.25, 0.0);
    physical.phys_cis_ajs_ckt_alt[0] = Complex64::new(2.5, -1.0);
    state.phys_quantities = Some(physical);
    let out = output_dir("original-io134-normal-writer", "literal");
    mvmc_core::io::output_phys_data(&data, &state, 0, Some(&out), false).unwrap();
    assert_eq!(
        read_values(&out.join("zvo_cisajs_001.dat")),
        [0.0, 0.0, 1.0, 0.0, 1.25, 0.0]
    );
    let factored = fs::read_to_string(out.join("zvo_cisajscktaltex_001.dat")).unwrap();
    assert_eq!(
        factored
            .lines()
            .filter(|line| !line.trim().is_empty())
            .count(),
        1
    );
    assert_eq!(
        read_values(&out.join("zvo_cisajscktaltex_001.dat")),
        [2.5, -1.0]
    );
    fs::remove_dir_all(out).unwrap();
}

#[test]
fn original_io134_qqqq_cells_follow_c_flatten_and_conjugation() {
    use mvmc_core::lanczos::{accumulate_lanczos_qqqq, LanczosEnergyError};
    use num_complex::Complex64;
    // S143/M0642-M0647, independently evaluated fixed scalar products.
    // C vmccal.c calculateQQQQ uses rq,ri as the conjugated left operand.
    let mut complex = [Complex64::new(0.0, 0.0); 16];
    accumulate_lanczos_qqqq(
        &mut complex,
        0.5,
        Complex64::new(2.0, 3.0),
        Complex64::new(5.0, 7.0),
        true,
    )
    .unwrap();
    assert_eq!(complex[0], Complex64::new(0.5, 0.0));
    assert_eq!(complex[2], Complex64::new(1.0, -1.5));
    assert_eq!(complex[15], Complex64::new(37.0, 0.0));
    let mut real = [Complex64::new(0.0, 0.0); 16];
    accumulate_lanczos_qqqq(
        &mut real,
        0.5,
        Complex64::new(2.0, 0.0),
        Complex64::new(5.0, 0.0),
        false,
    )
    .unwrap();
    assert_eq!(real[2], Complex64::new(1.0, 0.0));
    assert_eq!(real[15], Complex64::new(12.5, 0.0));
    let mut short = [Complex64::new(0.0, 0.0); 15];
    assert_eq!(
        accumulate_lanczos_qqqq(
            &mut short,
            0.5,
            Complex64::new(2.0, 0.0),
            Complex64::new(5.0, 0.0),
            false
        ),
        Err(LanczosEnergyError::QqqqTooShort),
    );
    assert_eq!(short, [Complex64::new(0.0, 0.0); 15]);
}

#[test]
fn original_io134_factored_accumulation_conjugates_and_adds() {
    use num_complex::Complex64;
    let mut sums = [Complex64::new(0.0, 0.0)];
    let locals = [Complex64::new(2.0, 1.0), Complex64::new(3.0, -4.0)];
    // S140: independent integer arithmetic gives (2+11i)/2 per sample.
    for expected in [Complex64::new(1.0, 5.5), Complex64::new(2.0, 11.0)] {
        mvmc_core::observables::accumulate_two_body_gex_sample(
            &mut sums,
            &locals,
            &[(0, 1)],
            Complex64::new(0.5, 0.0),
        );
        assert_eq!(sums[0], expected);
        assert_eq!(
            locals,
            [Complex64::new(2.0, 1.0), Complex64::new(3.0, -4.0)]
        );
    }
}

#[test]
fn original_io134_occupied_green_dispatch_and_fsz_state_publication() {
    use mvmc_expert_parsers::{ExpertModeData, GreenOneTerm, GreenTwoTerm, Spin};
    use num_complex::Complex64;
    let mut data = ExpertModeData::new();
    data.modpara.nsite = 2;
    data.modpara.nelec = 1;
    let idx = [0, 1];
    let cfg = [0, -1, -1, 1];
    let num = [1, 0, 0, 1];
    let mut state = mvmc_core::VmcOptimizationState::zeros(2, 1, 0, 0, 1, 1, true, false);
    // S141's two number operators have analytical occupancy one. This is
    // actual public kernel dispatch, not the private normal measurement loop.
    for (site, spin) in [(0, 0), (1, 1)] {
        assert_eq!(
            mvmc_core::observables::green_func1(
                site,
                site,
                spin,
                spin,
                Complex64::new(1.0, 0.0),
                &data,
                &state,
                &idx,
                &cfg,
                &num,
                &[]
            ),
            Complex64::new(1.0, 0.0)
        );
    }
    data.i_flg_orbital_general = 1;
    data.green_one_terms = vec![
        GreenOneTerm {
            site1: 0,
            site2: 0,
            spin1: Spin::Up,
            spin2: Spin::Up,
        },
        GreenOneTerm {
            site1: 0,
            site2: 1,
            spin1: Spin::Up,
            spin2: Spin::Down,
        },
    ];
    data.green_two_terms = vec![
        GreenTwoTerm {
            site1: 0,
            site2: 0,
            site3: 0,
            site4: 0,
            spin1: Spin::Up,
            spin2: Spin::Up,
            spin3: Spin::Up,
            spin4: Spin::Up,
        },
        GreenTwoTerm {
            site1: 0,
            site2: 1,
            site3: 0,
            site4: 1,
            spin1: Spin::Up,
            spin2: Spin::Down,
            spin3: Spin::Up,
            spin4: Spin::Down,
        },
    ];
    data.green_two_ex_indices = vec![(0, 0)];
    state.phys_quantities = Some(mvmc_core::state::PhysicalQuantities::zeros(2, 1, 2));
    // S142: C-supported occupancy shortcuts: diagonal=1, occupied target=0.
    // Rust API publishes into state; Julia's acc overload instead leaves state
    // untouched. Do not claim M0640-M0641 detached routing from this test.
    mvmc_core::observables::calculate_green_func_fsz(
        &data,
        &mut state,
        0.5,
        Complex64::new(1.0, 0.0),
        &idx,
        &cfg,
        &num,
        &[],
        &[0, 1],
    );
    let physical = state.phys_quantities.as_ref().unwrap();
    let local = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
    let weighted = [Complex64::new(0.5, 0.0), Complex64::new(0.0, 0.0)];
    assert_eq!(physical.local_cis_ajs, local);
    assert_eq!(physical.local_cis_ajs_ckt_alt_dc, local);
    assert_eq!(physical.phys_cis_ajs, weighted);
    assert_eq!(physical.phys_cis_ajs_ckt_alt_dc, weighted);
    assert_eq!(physical.phys_cis_ajs_ckt_alt, [Complex64::new(0.5, 0.0)]);
}

#[test]
fn original_io134_fixed_loader_family_offsets_and_duplicates() {
    use mvmc_expert_parsers::{
        ExpertModeData, GeneralRBMHiddenLayerTerm, GeneralRBMPhysHiddenTerm,
        GeneralRBMPhysLayerTerm,
    };
    use num_complex::Complex64;
    // S158-S160: literal acquisition records, independently specified in the
    // original Julia test. C ReadInitParameter: Proj, RBM, Slater, OptTrans.
    for (family, extra, count) in [
        ("dh", "1.10 -0.10 9.9 1.20 -0.20 9.9 1.30 -0.30 9.9 1.40 -0.40 9.9 1.50 -0.50 9.9 1.60 -0.60 9.9", 11),
        ("opttrans", "", 6),
        ("rbm", "0.61 -0.01 9.9 0.62 -0.02 9.9 0.71 -0.11 9.9 0.81 -0.21 9.9 0.82 -0.22 9.9", 10),
    ] {
        let mut data = ExpertModeData::new();
        data.n_gutzwiller_idx = 2;
        data.n_jastrow_idx = 1;
        data.modpara.n_orbital_idx = 2;
        data.slater_params = vec![Complex64::new(99.0, 99.0); 2];
        if family == "dh" {
            data.doublon_holon_2site_indices.push(mvmc_expert_parsers::DoublonHolon2SiteIndex { neighbors: vec![[1, 0], [0, 1]] });
            data.doublon_holon_2site_params = vec![Complex64::new(0.0, 0.0); 6];
        } else if family == "opttrans" {
            data.para_qp_opt_trans = vec![Complex64::new(1.0, 0.0)];
            data.opt_trans = vec![Complex64::new(1.0, 0.0)];
        } else {
            // Explicit C declared widths, independently of duplicate mappings.
            data.rbm_section_widths = [0, 0, 2, 0, 0, 1, 0, 0, 2];
            data.general_rbm_phys_layer_terms = [0, 0, 1].into_iter().enumerate().map(|(site, idx)| GeneralRBMPhysLayerTerm {
                site: site as i64, spin: 0, idx, value: Complex64::new(0.0, 0.0), is_complex: true,
            }).collect();
            data.general_rbm_hidden_layer_terms.push(GeneralRBMHiddenLayerTerm { site: 0, idx: 0, value: Complex64::new(0.0, 0.0), is_complex: true });
            data.general_rbm_phys_hidden_terms = [0, 1].map(|idx| GeneralRBMPhysHiddenTerm { site1: idx, site2: 0, spin: 0, idx, value: Complex64::new(0.0, 0.0), is_complex: true }).to_vec();
        }
        let tail = if family == "opttrans" { "0.70 -0.80 9.9" } else { "" };
        let record = format!("1 2 3 4 5 6 0.10 0 9.9 0.20 0 9.9 0.30 0 9.9 {extra} 0.40 -0.10 9.9 0.50 -0.20 9.9 {tail}");
        let dir = output_dir("original-io134-loader", family);
        let path = dir.join("zqp_opt.dat");
        fs::write(&path, record).unwrap();
        assert_eq!(mvmc_core::read_opt_para_file(&mut data, &path).unwrap(), count);
        assert_eq!(data.slater_params, [Complex64::new(0.4, -0.1), Complex64::new(0.5, -0.2)]);
        if family == "dh" {
            assert_eq!(data.doublon_holon_2site_params, [
                Complex64::new(1.1,-0.1), Complex64::new(1.2,-0.2), Complex64::new(1.3,-0.3),
                Complex64::new(1.4,-0.4), Complex64::new(1.5,-0.5), Complex64::new(1.6,-0.6),
            ]);
        } else if family == "opttrans" {
            assert_eq!(data.opt_trans, [Complex64::new(0.7, -0.8)]);
        } else {
            assert_eq!(data.general_rbm_phys_layer_terms.iter().map(|t| t.value).collect::<Vec<_>>(), [Complex64::new(0.61,-0.01), Complex64::new(0.61,-0.01), Complex64::new(0.62,-0.02)]);
            assert_eq!(data.general_rbm_hidden_layer_terms[0].value, Complex64::new(0.71,-0.11));
            assert_eq!(data.general_rbm_phys_hidden_terms.iter().map(|t| t.value).collect::<Vec<_>>(), [Complex64::new(0.81,-0.21), Complex64::new(0.82,-0.22)]);
        }
        fs::remove_dir_all(dir).unwrap();
    }
}

#[test]
fn original_io134_loader_c_eof_nonfinite_and_atomic_tail_contracts() {
    use mvmc_expert_parsers::ExpertModeData;
    use num_complex::Complex64;
    let mut data = ExpertModeData::new();
    data.modpara.n_orbital_idx = 1;
    data.slater_params = vec![Complex64::new(7.0, 8.0)];
    let dir = output_dir("original-io134-loader", "edge-records");
    let path = dir.join("zqp_opt.dat");
    // C's complete-record EOF loop retains the final record. Historical Julia
    // rejects extra records; that stricter policy is not C parity.
    fs::write(&path, "1 2 3 4 5 6 0.4 -0.1 9.9 1 2 3 4 5 6 0.5 -0.2 9.9").unwrap();
    assert_eq!(mvmc_core::read_opt_para_file(&mut data, &path).unwrap(), 1);
    assert_eq!(data.slater_params, [Complex64::new(0.5, -0.2)]);
    // Malformed tails are bounded Rust diagnostics: C unchecked fscanf does
    // not supply an atomicity/rejection oracle for these invalid records.
    for record in [
        "1 2 3 4 5 6 0.4 -0.1 9.9 1 2 3 4 5 6 0.5 bad 9.9",
        "1 2 3 4 5 6 0.4 -0.1 9.9 0.123",
        "1 2 3 4 5 6 0.4 -0.1 9.9 1 2 3",
    ] {
        fs::write(&path, record).unwrap();
        let before = format!("{data:?}");
        assert!(mvmc_core::read_opt_para_file(&mut data, &path).is_err());
        assert_eq!(format!("{data:?}"), before);
    }
    // S157 expands NaN/Inf/-Inf separately. C accepts their fscanf spelling;
    // Julia's nonfinite rejection is an intentional stricter loader contract.
    for token in ["NaN", "Inf", "-Inf"] {
        fs::write(&path, format!("1 2 3 4 5 6 {token} 0 9.9")).unwrap();
        assert_eq!(mvmc_core::read_opt_para_file(&mut data, &path).unwrap(), 1);
        assert!(!data.slater_params[0].re.is_finite());
    }
    let before = format!("{data:?}");
    fs::write(&path, "").unwrap();
    assert_eq!(mvmc_core::read_opt_para_file(&mut data, &path).unwrap(), 0);
    assert_eq!(format!("{data:?}"), before);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn original_io134_public_runner_normal_routing_and_qcaq_are_repeatable() {
    use mvmc_expert_parsers::{GreenOneTerm, Spin};
    use num_complex::Complex64;
    let fixture = physcal_root().join("two-samples/hubbard_chain_real");
    let mut outputs = Vec::new();
    for repeat in 0..2 {
        let mut preparation = mvmc_core::prepare_phys_cal_from_namelist(
            fixture.join("inputs/namelist.def"),
            fixture.join("zqp_opt.dat"),
            "real",
            Some(1),
        )
        .unwrap();
        preparation.data.modpara.n_data_qty_smp = 1;
        preparation.data.green_one_terms = (0..preparation.data.modpara.nsite)
            .flat_map(|site| {
                [Spin::Up, Spin::Down].map(move |spin| GreenOneTerm {
                    site1: site,
                    site2: site,
                    spin1: spin,
                    spin2: spin,
                })
            })
            .collect();
        preparation.data.green_two_terms.clear();
        preparation.data.green_two_ex_terms.clear();
        preparation.data.green_two_ex_indices.clear();
        let particles = 2 * preparation.data.modpara.nelec;
        let out = output_dir("original-io134-public-normal", &format!("repeat{repeat}"));
        let result = mvmc_core::vmc_phys_cal_to_dir(preparation, &out).unwrap();
        assert_eq!(result.iterations, 1);
        let physical = result.state.phys_quantities.as_ref().unwrap();
        let n = result.data.green_one_terms.len();
        assert_eq!(physical.local_cis_ajs.len(), n);
        assert_eq!(physical.phys_lanczos_qcisajsq.len(), 4 * n);
        // Number operators are analytic occupations. This observes the actual
        // private normal measurement loop via its public runner, not a helper
        // that reconstructs weighted accumulators in the test.
        let nsample = result.data.modpara.nvmc_sample as usize;
        let width = 2 * result.data.modpara.nsite as usize;
        let saved = &result.state.electron_config.ele_num;
        assert_eq!(saved.len(), nsample * width);
        for (index, term) in result.data.green_one_terms.iter().enumerate() {
            let orbital = term.site1 as usize
                + if term.spin1 == Spin::Down {
                    width / 2
                } else {
                    0
                };
            assert_eq!(
                physical.local_cis_ajs[index],
                Complex64::new(saved[(nsample - 1) * width + orbital] as f64, 0.0)
            );
        }
        let occupancy: Complex64 = physical.phys_cis_ajs.iter().copied().sum();
        assert!((occupancy.re - particles as f64).abs() <= 256.0 * f64::EPSILON);
        assert_eq!(occupancy.im, 0.0);
        // N commutes with H and equals the fixed particle count. Native C 1.4.0
        // moments of the same inputs and seed (`hubbard_lanczos1`; the Lanczos
        // mode does not change sampling) supply E/H^2; no Rust-generated golden
        // values. This is representative six-site coverage, not S144's two-site
        // helper with arbitrary H1=2,H2=5 and explicit half-weight.
        let moments =
            read_values(&fixture.join(
                "../../../native_c_physcal_181/hubbard_lanczos1/expected/zvo_ls_qqqq_007.dat",
            ));
        assert_eq!(moments.len(), 16);
        for (block, moment) in [(0, 1.0), (1, moments[2]), (2, moments[2]), (3, moments[3])] {
            let actual: Complex64 = physical.phys_lanczos_qcisajsq[block * n..(block + 1) * n]
                .iter()
                .copied()
                .sum();
            let expected = particles as f64 * moment;
            assert!(
                (actual.re - expected).abs() <= 1e-10 + 1e-10 * expected.abs(),
                "block {block}: {actual} vs {expected}"
            );
            assert_eq!(actual.im, 0.0);
        }
        outputs.push(snapshot(&out));
        fs::remove_dir_all(out).unwrap();
    }
    // Same input, seed=1, real serial configuration and backend, exact emitted
    // bytes. This is within-implementation repeatability, not cross-language
    // bitwise computed-float or trajectory acceptance.
    assert_eq!(outputs[0], outputs[1]);
}

#[test]
fn third_hubbard_lanczos_consumed19_matches_independent_c_all_five_outputs() {
    // Native C 1.4.0 run of the physcal_181/third-hubbard inputs (#695).
    let fixture = physcal_root().join("../native_c_physcal_181/third_hubbard_lanczos2");
    let preparation = mvmc_core::prepare_phys_cal_from_namelist(
        fixture.join("inputs/namelist.def"),
        fixture.join("zqp_opt.dat"),
        "real",
        Some(1),
    )
    .unwrap();
    assert_eq!(preparation.n_para_consumed, 19);
    assert_eq!(preparation.data.modpara.lanczos_mode, 2);
    assert_eq!(preparation.data.modpara.n_data_qty_smp, 1);
    let before = fixed_values(&preparation.data);
    let out = output_dir("third-hubbard-native-reference", "real-seed1");
    let result = mvmc_core::vmc_phys_cal_to_dir(preparation, &out).unwrap();
    assert_eq!(result.iterations, 1);
    assert_eq!(fixed_values(&result.data), before);
    // Original lanczos_equivalent.jl's THIRD model includes factored Green.
    // These are native C 1.4.0 values, not the synthetic IO134 moments or a
    // historical Julia solver golden. Strict rows/indices remain.
    for name in [
        "zvo_ls_out_001.dat",
        "zvo_ls_qqqq_001.dat",
        "zvo_ls_cisajs_001.dat",
        "zvo_ls_cisajscktalt_001.dat",
        "zvo_ls_cisajscktaltex_001.dat",
    ] {
        let actual = out.join(name);
        let expected = fixture.join("expected").join(name);
        assert!(actual.is_file(), "{name}");
        assert!(expected.is_file(), "{name}");
        assert_reference(&actual, &expected);
    }
    fs::remove_dir_all(out).unwrap();
}

#[test]
fn original_io134_factored_supported_cases_reach_public_validation() {
    let fixture = physcal_root().join("third-hubbard");
    let preparation = mvmc_core::prepare_phys_cal_from_namelist(
        fixture.join("inputs/namelist.def"),
        fixture.join("zqp_opt.dat"),
        "real",
        Some(1),
    )
    .unwrap();
    let mut data = preparation.data;
    // S154's standalone Julia hook ignores unrelated settings. A full valid
    // parsed control ensures Rust's PUBLIC validator reaches the intended
    // combinations, not a masking invalid sample-count or Lanczos guard.
    data.modpara.lanczos_mode = 0;
    assert!(!data.green_two_ex_terms.is_empty());
    assert_eq!(data.i_flg_orbital_general, 0);
    assert!(mvmc_core::validation::validate_phys_cal(&data).is_ok());
    data.i_flg_orbital_general = 1;
    assert!(mvmc_core::validation::validate_phys_cal(&data).is_ok());
    data.green_two_ex_terms.clear();
    data.green_two_ex_indices.clear();
    assert!(mvmc_core::validation::validate_phys_cal(&data).is_ok());
}

#[test]
fn original_io134_four_output_indices_use_c_signed_start_plus_sample() {
    use mvmc_expert_parsers::{ExpertModeData, GreenOneTerm, Spin};
    let mut data = ExpertModeData::new();
    data.modpara.nsite = 2;
    data.green_one_terms = vec![GreenOneTerm {
        site1: 0,
        site2: 1,
        spin1: Spin::Up,
        spin2: Spin::Up,
    }];
    let mut state = mvmc_core::VmcOptimizationState::zeros(2, 1, 0, 0, 1, 1, true, false);
    state.phys_quantities = Some(mvmc_core::state::PhysicalQuantities::zeros(1, 0, 0));
    for (start, sample, expected) in [(1, 0, 1), (1, 3, 4), (7, 0, 7), (7, 2, 9)] {
        data.modpara.n_data_idx_start = start;
        let out = output_dir("original-io134-index", &format!("{start}-{sample}"));
        mvmc_core::io::output_phys_data(&data, &state, sample, Some(&out), false).unwrap();
        assert!(out.join(format!("zvo_cisajs_{expected:03}.dat")).is_file());
        assert!(out.join(format!("zvo_out_{expected:03}.dat")).is_file());
        assert!(out.join(format!("zvo_var_{expected:03}.dat")).is_file());
        // C indexed filenames, NOT Julia's shared zvo_out.dat/var.dat append.
        assert!(!out.join("zvo_out.dat").exists());
        assert!(!out.join("zvo_var.dat").exists());
        fs::remove_dir_all(out).unwrap();
    }
}

#[test]
fn original_io134_synthetic_lanczos_writes_energy_and_all_green_kinds() {
    use mvmc_expert_parsers::{ExpertModeData, GreenOneTerm, GreenTwoExTerm, GreenTwoTerm, Spin};
    use num_complex::Complex64;
    // S146/S147: original fixed moments and QPhysQ, not a Hubbard replay.
    // Independently substitute alpha=-3/4 into C CalculateEneByAlpha and
    // CalculatePhysVal: norm=5/8, E=-7/2; one=-4/5, direct=-2/5,
    // factored=9/10. These exact rational expectations do not use Rust output.
    for (mode, index) in [(1, 7), (2, 4)] {
        let mut data = ExpertModeData::new();
        data.modpara.nsite = 2;
        data.modpara.lanczos_mode = mode;
        data.modpara.n_data_idx_start = index;
        data.green_one_terms.push(GreenOneTerm {
            site1: 0,
            spin1: Spin::Up,
            site2: 1,
            spin2: Spin::Up,
        });
        data.green_two_terms.push(GreenTwoTerm {
            site1: 0,
            spin1: Spin::Up,
            site2: 1,
            spin2: Spin::Up,
            site3: 1,
            spin3: Spin::Down,
            site4: 0,
            spin4: Spin::Down,
        });
        data.green_two_ex_terms.push(GreenTwoExTerm {
            site1: 0,
            spin1: Spin::Up,
            site2: 1,
            spin2: Spin::Up,
            site3: 0,
            spin3: Spin::Up,
            site4: 1,
            spin4: Spin::Up,
        });
        // Rust's parsed-data equivalent of Julia pq.cis_ajs_ckt_alt_idx=[(1,1)].
        data.green_two_ex_indices = vec![(0, 0)];
        let mut state = mvmc_core::VmcOptimizationState::zeros(2, 1, 0, 0, 1, 1, true, false);
        let mut physical = mvmc_core::state::PhysicalQuantities::zeros(1, 1, 1);
        for (slot, value) in [(2, 1.0), (3, 2.0), (10, 3.0), (11, 1.0), (15, 5.0)] {
            physical.phys_lanczos_qqqq[slot] = Complex64::new(value, 0.0);
        }
        physical.phys_lanczos_qcisajsq = [1.0, 2.0, 3.0, 4.0]
            .map(|x| Complex64::new(x, 0.0))
            .to_vec();
        physical.phys_lanczos_qcisajscktaltq = [9.0, 10.0, 11.0, 13.0]
            .map(|x| Complex64::new(x, 0.0))
            .to_vec();
        physical.phys_lanczos_qcisajscktaltq_dc = [5.0, 6.0, 7.0, 8.0]
            .map(|x| Complex64::new(x, 0.0))
            .to_vec();
        let moments = physical.phys_lanczos_qqqq.clone();
        state.phys_quantities = Some(physical);
        let out = output_dir("original-io134-ls", &format!("mode{mode}"));
        mvmc_core::io::output_phys_data(&data, &state, 0, Some(&out), false).unwrap();
        let energy = read_values(&out.join(format!("zvo_ls_out_{index:03}.dat")));
        assert_eq!(energy.len(), 3);
        assert_eq!(energy[0], -3.5);
        assert_eq!(energy[2], -0.75);
        assert_eq!(
            read_values(&out.join(format!("zvo_ls_qqqq_{index:03}.dat"))),
            moments.iter().map(|z| z.re).collect::<Vec<_>>()
        );
        if mode == 2 {
            for (kind, indices, expected) in [
                ("cisajs", vec![0.0, 0.0, 1.0, 0.0], -0.8),
                (
                    "cisajscktalt",
                    vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                    -0.4,
                ),
                ("cisajscktaltex", vec![], 0.9),
            ] {
                let rows =
                    fs::read_to_string(out.join(format!("zvo_ls_{kind}_{index:03}.dat"))).unwrap();
                assert_eq!(
                    rows.lines().filter(|line| !line.trim().is_empty()).count(),
                    1
                );
                let values = rows
                    .split_whitespace()
                    .map(|word| word.parse::<f64>().unwrap())
                    .collect::<Vec<_>>();
                assert_eq!(values.len(), indices.len() + 2);
                assert_eq!(&values[..indices.len()], indices.as_slice());
                assert!((values[indices.len()] - expected).abs() <= 2.0 * f64::EPSILON);
                assert_eq!(values[indices.len() + 1], 0.0);
            }
        }
        fs::remove_dir_all(out).unwrap();
    }
}

fn output_dir(test: &str, name: &str) -> PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    for _ in 0..10_000 {
        let index = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "mvmc-issue181-{test}-{name}-{}-{index}",
            std::process::id(),
        ));
        match fs::create_dir(&path) {
            Ok(()) => return path,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => panic!("create exclusive PhysCal test directory: {error}"),
        }
    }
    panic!("cannot allocate exclusive PhysCal test directory");
}

fn read_values(path: &Path) -> Vec<f64> {
    fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()))
        .split_whitespace()
        .map(|value| {
            value
                .parse::<f64>()
                .unwrap_or_else(|error| panic!("{}: {value}: {error}", path.display()))
        })
        .collect()
}

fn assert_reference(actual: &Path, expected: &Path) {
    // Independent contracts: Julia reference tools/green_compare.jl and
    // lanczos_equivalent.jl. Julia's scalar isapprox uses max(atol, rtol*scale),
    // not their sum. Lanczos references specify absolute errors only.
    let name = expected.file_name().unwrap().to_str().unwrap();
    let (indices, width, atol, rtol) = if name.starts_with("zvo_ls_out_") {
        (0, Some(3), 1.0e-8, 0.0)
    } else if name.starts_with("zvo_ls_qqqq_") {
        (0, Some(16), 1.0e-10, 0.0)
    } else if name.starts_with("zvo_ls_cisajscktaltex_") {
        (0, None, 1.0e-8, 0.0)
    } else if name.starts_with("zvo_ls_cisajscktalt_") {
        (8, Some(10), 1.0e-8, 0.0)
    } else if name.starts_with("zvo_ls_cisajs_") {
        (4, Some(6), 1.0e-8, 0.0)
    } else if name.starts_with("zvo_cisajscktaltex_") {
        (0, None, 1.0e-12, 1.0e-9)
    } else if name.starts_with("zvo_cisajscktalt_") {
        (8, Some(10), 1.0e-12, 1.0e-9)
    } else if name.starts_with("zvo_cisajs_") {
        (4, Some(6), 1.0e-12, 1.0e-10)
    } else if name.starts_with("zvo_out_") {
        (0, Some(6), 1.0e-12, 1.0e-10)
    } else if name.starts_with("zvo_var_") {
        (0, None, 1.0e-12, 1.0e-10)
    } else {
        panic!("no independent output contract for {name}");
    };
    let actual_text = fs::read_to_string(actual).unwrap();
    let expected_text = fs::read_to_string(expected).unwrap();
    let rows = |text: &str| {
        text.lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                line.split_whitespace()
                    .map(str::to_owned)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };
    let actual_rows = rows(&actual_text);
    let expected_rows = rows(&expected_text);
    assert_eq!(actual_rows.len(), expected_rows.len(), "{name} row count");
    if indices == 0 {
        assert_eq!(expected_rows.len(), 1, "{name} value-only output row count");
    }
    if name.contains("cisajscktaltex_") {
        // C outputData/PhysCalLanczos writes one ordered row containing one
        // real/imaginary pair per TwoBodyGEx term. A row-per-term Rust writer
        // is an output-contract defect, even if flattening gives equal values.
        let inputs = expected.parent().unwrap().parent().unwrap().join("inputs");
        if inputs.join("namelist.def").is_file() {
            let namelist = fs::read_to_string(inputs.join("namelist.def")).unwrap();
            let definition = namelist
                .lines()
                .find_map(|line| {
                    let mut fields = line.split_whitespace();
                    (fields.next() == Some("TwoBodyGEx")).then(|| fields.next().unwrap())
                })
                .expect("nonempty factored reference requires TwoBodyGEx input");
            let definition = fs::read_to_string(inputs.join(definition)).unwrap();
            let count: usize = definition
                .lines()
                .nth(1)
                .unwrap()
                .split_whitespace()
                .nth(1)
                .unwrap()
                .parse()
                .unwrap();
            assert_eq!(
                expected_rows[0].len(),
                2 * count,
                "{name} independent TwoBodyGEx width"
            );
        }
    }
    for (row, (a, e)) in actual_rows.iter().zip(&expected_rows).enumerate() {
        let width = width.unwrap_or(e.len());
        assert_eq!(e.len(), width, "{name} reference row {row} columns");
        assert_eq!(a.len(), width, "{name} actual row {row} columns");
        if name.contains("cisajscktaltex_") {
            assert_eq!(width % 2, 0, "{name} real/imaginary pairs");
        }
        for column in 0..width {
            if column < indices {
                // Parse as integers as well as checking the reference tokens;
                // fractional or approximately equal indices must never pass.
                assert_eq!(
                    a[column].parse::<i64>().unwrap(),
                    e[column].parse::<i64>().unwrap(),
                    "{name} row {row} index column {column}"
                );
                assert_eq!(
                    a[column], e[column],
                    "{name} row {row} index column {column}"
                );
            } else {
                let a = a[column].parse::<f64>().unwrap();
                let e = e[column].parse::<f64>().unwrap();
                let scale = a.abs().max(e.abs());
                let tolerance = f64::max(atol, rtol * scale);
                // Retain the old gate's upper bound too: adopting an
                // independent contract must not loosen any existing check.
                let tolerance = tolerance.min(1.0e-10 + 1.0e-8 * scale);
                // #576: against the C 1.4.0 reference (both use the rbOld
                // exchange update) the largest Lanczos output difference on
                // Linux x86_64 is 7.8e-11 absolute (1.1e-13 relative, value
                // 7.17e2) and the moments differ by at most 9.1e-13; see the
                // fixture README. The bound max(1e-10, 1e-12 |value|) keeps a
                // margin of about 9 in the relative term for other BLAS kernels.
                let tolerance = if expected
                    .components()
                    .any(|component| component.as_os_str() == "native_c_lanczos_576")
                    && name.starts_with("zvo_ls_")
                {
                    tolerance.min(f64::max(1.0e-10, 1.0e-12 * scale))
                } else {
                    tolerance
                };
                assert!(a.is_finite() && e.is_finite() && (a - e).abs() <= tolerance,
                    "{name} row {row} column {column}: actual={a:.17e}, expected={e:.17e}, tolerance={tolerance:.3e}");
            }
        }
    }
}

fn snapshot(dir: &Path) -> BTreeSet<(String, String)> {
    fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap())
        // The `_time_` file ends each row with a wall-clock ctime string.
        .filter(|entry| !entry.file_name().to_string_lossy().contains("_time_"))
        .map(|entry| {
            (
                entry.file_name().to_string_lossy().into_owned(),
                fs::read_to_string(entry.path()).unwrap(),
            )
        })
        .collect()
}

fn fixed_values(data: &mvmc_core::ExpertModeData) -> Vec<num_complex::Complex64> {
    data.projection_parameters()
        .into_iter()
        .chain(data.rbm_parameters())
        .chain(data.slater_params.iter().copied())
        .chain(data.opt_trans.iter().copied())
        .collect()
}

fn assert_fixed_values(data: &mvmc_core::ExpertModeData, expected: &[num_complex::Complex64]) {
    let actual = fixed_values(data);
    assert_eq!(actual.len(), expected.len(), "fixed parameter width");
    for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
        for (a, e) in [(a.re, e.re), (a.im, e.im)] {
            // Independent C cabs vs Rust hypot normalization can differ by
            // rounding; this bound is tighter than every output contract.
            assert!(
                (a - e).abs() <= 1.0e-14 * e.abs().max(1.0),
                "fixed parameter {index}: actual={a:.17e}, expected={e:.17e}"
            );
        }
    }
}

fn physcal_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/physcal_181")
}

fn reference_integers(path: &Path) -> Vec<i64> {
    fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("{}: {error}", path.display()))
        .split_whitespace()
        .map(|value| value.parse().unwrap())
        .collect()
}

fn assert_discrete(label: &str, actual: &[i64], expected: &[i64]) {
    assert_eq!(actual.len(), expected.len(), "{label} width");
    if let Some(index) = actual.iter().zip(expected).position(|(a, e)| a != e) {
        panic!(
            "{label}: first discrete divergence at element {index}: actual={}, reference={}",
            actual[index], expected[index]
        );
    }
}

#[test]
fn factored_output_requires_one_ordered_row_of_pairs() {
    let dir = output_dir("factored-contract", "schema");
    fs::create_dir_all(&dir).unwrap();
    let actual = dir.join("actual.dat");
    for name in [
        "zvo_cisajscktaltex_001.dat",
        "zvo_ls_cisajscktaltex_001.dat",
    ] {
        let expected = dir.join(name);
        fs::write(&expected, "0.25 0 0.5 0 0.75 0\n").unwrap();
        fs::write(&actual, "0.25 0 0.5 0 0.75 0\n").unwrap();
        assert_reference(&actual, &expected);
        for bad in [
            "0.25 0\n0.5 0\n0.75 0\n",
            "0.5 0 0.25 0 0.75 0\n",
            "0.25 0 0.5 0 0.75\n",
            "0.25 0 0.5 0 0.75 0 1 0\n",
        ] {
            fs::write(&actual, bad).unwrap();
            assert!(
                std::panic::catch_unwind(|| assert_reference(&actual, &expected)).is_err(),
                "{name} accepted invalid ordered row: {bad}"
            );
        }
    }
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn green_reference_schema_rejects_identical_malformed_factored_files() {
    // Original GreenCompare M1329: byte equality must not bypass the C
    // single-ordered-row contract. This asserts the Rust gate's schema,
    // not Julia's GreenCompareResult exact/fallback/detail fields.
    let dir = output_dir("green-identical-malformed", "original-m1329");
    let actual = dir.join("actual.dat");
    let expected = dir.join("zvo_cisajscktaltex_001.dat");
    let malformed = "2.5 -1.0\n0.3 0.1\n";
    fs::write(&actual, malformed).unwrap();
    fs::write(&expected, "2.5 -1.0 0.3 0.1 \n").unwrap();
    // M1328: malformed candidate versus a valid single-line reference.
    assert!(std::panic::catch_unwind(|| assert_reference(&actual, &expected)).is_err());
    fs::write(&expected, malformed).unwrap();
    assert_eq!(fs::read(&actual).unwrap(), fs::read(&expected).unwrap());
    assert!(std::panic::catch_unwind(|| assert_reference(&actual, &expected)).is_err());
    // Literal original positive control: the same ordered values in one row.
    fs::write(&actual, "2.5 -1.0 0.3 0.1 \n").unwrap();
    fs::write(&expected, "2.5 -1.0 0.3 0.1 \n").unwrap();
    assert_reference(&actual, &expected);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn green_reference_gate_requires_candidate_and_reference_files() {
    // Original GreenCompare M1333/M1334 use ONE_A and a missing nope.dat
    // on each side. Rust's assertion-based test API fails on missing files;
    // it does not return Julia's structured "missing" diagnostic result.
    let dir = output_dir("green-missing-files", "original-m1333-m1334");
    let actual = dir.join("actual.dat");
    let expected = dir.join("zvo_cisajs_001.dat");
    let one_a = "0 0 1 0 1.0 0.0 \n1 1 0 1 2.0 0.0 \n\n";
    fs::write(&actual, one_a).unwrap();
    assert!(!expected.exists());
    assert!(std::panic::catch_unwind(|| assert_reference(&actual, &expected)).is_err());
    fs::write(&expected, one_a).unwrap();
    assert_reference(&actual, &expected);
    // M1330-M1332: different fixed formatting, identical numeric payload.
    // Rust deliberately exposes no raw-exact/fallback result flags.
    fs::write(&actual, "0 0 1 0 1.0 0.0 \n\n").unwrap();
    fs::write(&expected, "0 0 1 0 1.0 0.0\n").unwrap();
    assert_ne!(fs::read(&actual).unwrap(), fs::read(&expected).unwrap());
    assert_reference(&actual, &expected);
    let missing = dir.join("nope.dat");
    assert!(!missing.exists());
    assert!(std::panic::catch_unwind(|| assert_reference(&missing, &expected)).is_err());
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn no_factored_terms_preserve_one_body_order_and_duplicate_rows() {
    use mvmc_expert_parsers::{ExpertModeData, GreenOneTerm, Spin};
    let mut data = ExpertModeData::new();
    data.modpara.nsite = 2;
    data.modpara.nelec = 1;
    data.modpara.nmp_trans = 1; // Valid projection input for the mode2 duplicate guard.
    data.modpara.n_data_idx_start = 7;
    data.green_one_terms = vec![
        GreenOneTerm {
            site1: 1,
            spin1: Spin::Down,
            site2: 0,
            spin2: Spin::Down,
        },
        GreenOneTerm {
            site1: 0,
            spin1: Spin::Up,
            site2: 1,
            spin2: Spin::Up,
        },
        GreenOneTerm {
            site1: 0,
            spin1: Spin::Up,
            site2: 1,
            spin2: Spin::Up,
        },
    ];
    let mut state = mvmc_core::VmcOptimizationState::zeros(2, 1, 0, 0, 1, 1, true, false);
    let mut phys = mvmc_core::state::PhysicalQuantities::zeros(3, 0, 0);
    phys.phys_cis_ajs = [1.0, 2.0, 3.0]
        .map(|x| num_complex::Complex64::new(x, 0.0))
        .to_vec();
    state.phys_quantities = Some(phys);
    let out = output_dir("one-body-duplicates", "synthetic");
    fs::create_dir_all(&out).unwrap();
    // C GetInfoOneBodyG normal branch retains file order and duplicate rows;
    // outputData prints each entry, rather than deduplicating at the writer.
    mvmc_core::io::output_phys_data(&data, &state, 0, Some(&out), false).unwrap();
    let rows = fs::read_to_string(out.join("zvo_cisajs_007.dat")).unwrap();
    let rows = rows
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.split_whitespace().collect::<Vec<_>>())
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), 3);
    for (row, (indices, value)) in rows.iter().zip([
        (["1", "1", "0", "1"], 1.0),
        (["0", "0", "1", "0"], 2.0),
        (["0", "0", "1", "0"], 3.0),
    ]) {
        assert_eq!(row.len(), 6);
        assert_eq!(&row[..4], &indices);
        assert_eq!(row[4].parse::<f64>().unwrap(), value);
        assert_eq!(row[5].parse::<f64>().unwrap(), 0.0);
    }
    data.modpara.lanczos_mode = 2;
    let error = mvmc_core::validation::validate_phys_cal(&data).unwrap_err();
    assert!(error.contains("duplicate OneBodyG"), "{error}");
    let reference = physcal_root().join("duplicate-reader/result.txt");
    assert_eq!(
        fs::read_to_string(reference).unwrap(),
        "declared_records=2\ncount_after_mode2_dedup=1\nnormal_reader_status=1\nwith_gex_canonical_count=1\nindirect_reader_status=0\n"
    );
    // C readdef.c:645–653 deduplicates in mode2, then :1046 selects the
    // normal reader without GEx. :2299 rejects two rows vs reduced count1.
    // Rust's safe early diagnostic does not emulate the upstream extra write.
    let _ = fs::remove_dir_all(out);
}

#[test]
fn singular_lanczos_alpha_writes_nan_without_aborting() {
    let mut data = mvmc_expert_parsers::ExpertModeData::new();
    data.modpara.nsite = 2;
    data.modpara.lanczos_mode = 1;
    data.modpara.n_data_idx_start = 7;
    let mut state = mvmc_core::VmcOptimizationState::zeros(2, 1, 0, 0, 1, 1, true, false);
    state.phys_quantities = Some(mvmc_core::state::PhysicalQuantities::zeros(0, 0, 0));
    let out = output_dir("singular-alpha", "synthetic");
    fs::create_dir_all(&out).unwrap();
    // All-zero moments give 0/0 in C CalculateEne. This additionally guards
    // Julia/Rust's defined no-abort fallback; no generic nonfinite tolerance.
    mvmc_core::io::output_phys_data(&data, &state, 0, Some(&out), false).unwrap();
    let values = read_values(&out.join("zvo_ls_out_007.dat"));
    assert_eq!(values.len(), 3);
    assert!(values.iter().all(|x| x.is_nan()));
    let moments = read_values(&out.join("zvo_ls_qqqq_007.dat"));
    assert_eq!(moments.len(), 16);
    assert!(moments.iter().all(|x| *x == 0.0));
    let _ = fs::remove_dir_all(out);
}

#[test]
fn duplicate_one_body_with_gex_matches_native_c_canonical_layout_fixture() {
    let fixture = physcal_root().join("duplicate-reader");
    let expected = reference_integers(&fixture.join("canonical-layout.txt"));
    let data = mvmc_expert_parsers::parse_expert_mode_files(fixture.join("namelist.def")).unwrap();
    assert!(data.input_errors.is_empty(), "{:?}", data.input_errors);
    assert_eq!(
        expected.len(),
        2 + 4 * expected[0] as usize + 2 * expected[1] as usize
    );
    assert_eq!(
        data.green_one_terms.len(),
        expected[0] as usize,
        "C canonical OneBodyG count"
    );
    assert_eq!(
        data.green_two_ex_indices.len(),
        expected[1] as usize,
        "C factored pair count"
    );
    let mut actual = vec![
        data.green_one_terms.len() as i64,
        data.green_two_ex_indices.len() as i64,
    ];
    for term in &data.green_one_terms {
        actual.extend([
            term.site1,
            i64::from(term.spin1.as_code()),
            term.site2,
            i64::from(term.spin2.as_code()),
        ]);
    }
    for &(first, second) in &data.green_two_ex_indices {
        actual.extend([first as i64, second as i64]);
    }
    assert_discrete("C duplicate-with-GEx canonical layout", &actual, &expected);
    let fixed = fixture
        .parent()
        .unwrap()
        .join("heisenberg_chain_real/zqp_opt.dat");
    for mode in [0, 2] {
        let mut preparation = mvmc_core::prepare_phys_cal_from_namelist(
            fixture.join("namelist.def"),
            &fixed,
            "real",
            Some(1),
        )
        .unwrap();
        preparation.data.modpara.lanczos_mode = mode;
        let out = output_dir("duplicate-gex-c-layout", &format!("mode-{mode}"));
        let _result = mvmc_core::vmc_phys_cal_to_dir(preparation, &out).unwrap();
        for suffix in ["cisajs", "ls_cisajs"] {
            if suffix.starts_with("ls_") && mode == 0 {
                continue;
            }
            let text = fs::read_to_string(out.join(format!("zvo_{suffix}_001.dat"))).unwrap();
            if suffix == "ls_cisajs" && text.is_empty() {
                // This exact-eigenstate fixed state has a negative-discriminant Lanczos
                // step, for which C leaves every zvo_ls_* file empty; the ls row layout
                // is covered by the perturbed native-C scenarios.
                continue;
            }
            let rows = text
                .lines()
                .filter(|row| !row.trim().is_empty())
                .collect::<Vec<_>>();
            assert_eq!(
                rows.len(),
                expected[0] as usize,
                "mode {mode} {suffix} C row count"
            );
            let columns = rows[0].split_whitespace().collect::<Vec<_>>();
            assert_eq!(columns.len(), 6);
            for column in 0..4 {
                assert_eq!(
                    columns[column].parse::<i64>().unwrap(),
                    expected[2 + column],
                    "mode {mode} {suffix} discrete column {column}"
                );
            }
            assert!(text.ends_with("\n\n"));
        }
        fs::remove_dir_all(out).unwrap();
    }
}

#[test]
fn energy_only_physcal_zero_green_file_sets_follow_native_c_all_lanczos_modes() {
    let root = physcal_root().to_owned();
    let fixture = root.join("energy-only");
    for mode in 0..=2 {
        let mut preparation = mvmc_core::prepare_phys_cal_from_namelist(
            fixture.join("namelist.def"),
            // Perturbed (non-eigenstate) fixed values: an exact eigenstate has a
            // singular Lanczos alpha, for which C writes empty files instead.
            root.join("../native_c_physcal_181/heisenberg_real_lanczos2/zqp_opt.dat"),
            "real",
            Some(1),
        )
        .unwrap();
        assert!(preparation.data.green_one_terms.is_empty());
        assert!(preparation.data.green_two_terms.is_empty());
        assert!(preparation.data.green_two_ex_terms.is_empty());
        preparation.data.modpara.n_data_idx_start = 7;
        preparation.data.modpara.lanczos_mode = mode;
        let out = output_dir("zero-green", &format!("mode-{mode}"));
        let result = mvmc_core::vmc_phys_cal_to_dir(preparation, &out).unwrap();
        assert_eq!(result.iterations, 1);
        let mut actual = fs::read_dir(&out)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect::<BTreeSet<_>>();
        // C InitFile creates one `_time_` file in PhysCal mode as well.
        assert!(actual.remove("zvo_time_007.dat"));
        let expected = fs::read_to_string(fixture.join(format!("mode-{mode}-files.txt")))
            .unwrap()
            .lines()
            .map(str::to_owned)
            .collect::<BTreeSet<_>>();
        assert_eq!(
            actual, expected,
            "zero-Green Lanczos mode {mode} C file set"
        );
        if mode == 2 {
            for suffix in ["ls_cisajs", "ls_cisajscktalt", "ls_cisajscktaltex"] {
                assert_eq!(
                    fs::read(out.join(format!("zvo_{suffix}_007.dat"))).unwrap(),
                    b"\n",
                    "mode2 zero-entry {suffix} bytes"
                );
            }
        }
        for suffix in ["out", "var"] {
            let bytes = fs::read(out.join(format!("zvo_{suffix}_007.dat"))).unwrap();
            assert!(bytes.ends_with(b"\n"));
            assert!(!bytes.ends_with(b"\n\n"));
        }
        fs::remove_dir_all(out).unwrap();
    }
}

#[test]
fn lanczos_interall_matches_native_c_modes_one_and_two() {
    let root =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/native_c_lanczos_576");
    for (name, mode, orbital_mode) in [
        ("interall_real1", 1, "real"),
        ("interall_real2", 2, "real"),
        ("interall_complex1", 1, "cmp"),
        ("interall_complex2", 2, "cmp"),
        ("interall_nodal1", 1, "real"),
        ("interall_nodal2", 2, "real"),
    ] {
        let fixture = root.join(name);
        let preparation = mvmc_core::prepare_phys_cal_from_namelist(
            fixture.join("inputs/namelist.def"),
            fixture.join("zqp_opt.dat"),
            orbital_mode,
            Some(1),
        )
        .unwrap();
        assert_eq!(preparation.data.modpara.lanczos_mode, mode);
        assert!(!preparation.data.inter_all_terms.is_empty());
        let out = output_dir("interall-lanczos", name);
        mvmc_core::vmc_phys_cal_to_dir(preparation, &out).unwrap();
        for entry in fs::read_dir(fixture.join("expected")).unwrap() {
            let path = entry.unwrap().path();
            assert_reference(&out.join(path.file_name().unwrap()), &path);
        }
        let _ = fs::remove_dir_all(out);
    }
}

#[test]
fn fixed_records_overlays_and_normalization_follow_native_c_stages() {
    use mvmc_expert_parsers::utils::{
        parameter_init::init_parameter, read_input_parameters::read_input_parameters,
    };
    let root = physcal_root().to_owned();
    for name in [
        "hubbard_chain_dh_overlays",
        "hubbard_chain_dh_opttrans",
        "hubbard_chain_dh_rbm_opttrans",
        "unnormalized-reserved-slater",
    ] {
        let fixture = root.join(name);
        let native = fixture.join("native-c-stages");
        let namelist = fixture.join("inputs/namelist.def");
        let mut data =
            mvmc_expert_parsers::parse_expert_mode_files_with_opt_trans(&namelist, true).unwrap();
        let mut rng = sfmt19937::Sfmt19937Rng::new(1);
        init_parameter(&mut data, &mut rng).unwrap();
        for stage in ["initialized", "fixed", "overlaid", "synchronized"] {
            match stage {
                "fixed" => {
                    assert_eq!(
                        mvmc_core::read_opt_para_file(&mut data, fixture.join("zqp_opt.dat"))
                            .unwrap(),
                        data.count_variational_parameters()
                    );
                }
                "overlaid" => read_input_parameters(&mut data, &namelist).unwrap(),
                "synchronized" => mvmc_core::sync::sync_modified_parameter_local(&mut data, false),
                _ => {}
            }
            let numbers = read_values(&native.join(format!("{stage}-parameters.txt")));
            let (pairs, remainder) = numbers.as_chunks::<2>();
            assert!(remainder.is_empty(), "complete native C parameter records");
            let expected = pairs
                .iter()
                .map(|pair| num_complex::Complex64::new(pair[0], pair[1]))
                .collect::<Vec<_>>();
            if stage == "fixed" || stage == "overlaid" {
                assert_eq!(
                    fixed_values(&data),
                    expected,
                    "{name} {stage} stored coefficients"
                );
            } else {
                assert_fixed_values(&data, &expected);
            }
            let expected_rng = reference_integers(&native.join(format!("{stage}-next624.txt")));
            assert_eq!(expected_rng.len(), 624);
            let mut peek = rng.clone();
            let actual_rng = (0..624)
                .map(|_| i64::from(peek.gen_rand32()))
                .collect::<Vec<_>>();
            assert_discrete(
                &format!("native C {name} {stage} RNG"),
                &actual_rng,
                &expected_rng,
            );
            let count = reference_integers(&native.join(format!("{stage}-draw-count.txt")));
            let mut counted = sfmt19937::Sfmt19937Rng::new(1);
            for _ in 0..count[0] {
                counted.gen_rand32();
            }
            let words = (0..624)
                .map(|_| i64::from(counted.gen_rand32()))
                .collect::<Vec<_>>();
            assert_discrete(
                &format!("native C {name} {stage} draw count"),
                &words,
                &expected_rng,
            );
        }
        if name == "unnormalized-reserved-slater" {
            assert_eq!(
                data.slater_params,
                [(-1.0, 0.0), (0.5, 0.0), (0.25, 0.0), (4.0, 0.0)]
                    .map(|(re, im)| num_complex::Complex64::new(re, im))
            );
            assert!(data.orbital_terms.iter().all(|term| term.idx < 2));
            // A wrong sync-before-overlay phase would leave -4,2,1,16.
            let prepared = mvmc_core::prepare_phys_cal_from_namelist(
                &namelist,
                fixture.join("zqp_opt.dat"),
                "real",
                Some(1),
            )
            .unwrap();
            assert_eq!(prepared.data.slater_params, data.slater_params);
            assert_eq!(prepared.n_para_consumed, 6);
            let out = output_dir("reserved-storage", name);
            let mut state = mvmc_core::VmcOptimizationState::zeros(6, 3, 2, 6, 1, 1, false, false);
            state.energy.etot = num_complex::Complex64::new(1.0, 0.0);
            state.energy.etot2 = num_complex::Complex64::new(2.0, 0.0);
            state.phys_quantities = Some(mvmc_core::state::PhysicalQuantities::zeros(0, 0, 0));
            mvmc_core::io::output_phys_data(&prepared.data, &state, 0, Some(&out), false).unwrap();
            let record = read_values(&out.join("zvo_var_001.dat"));
            assert_eq!(record.len(), 6 + 3 * 6);
            assert_eq!(
                &record[12..],
                &[-1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.25, 0.0, 0.0, 4.0, 0.0, 0.0]
            );
            let _ = fs::remove_dir_all(out);
        }
    }
}

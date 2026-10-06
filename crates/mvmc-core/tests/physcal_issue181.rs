//! Issue #181: serial PhysCal and non-InterAll Lanczos verification.
//!
//! These are opt-in because they replay the reference sampling scenarios.  Run
//! with `MVMC_RS_PHYSCAL_181=1 cargo nextest run -p mvmc-core --test
//! physcal_issue181`.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

#[allow(dead_code)]
#[path = "../../../tests/support/julia_fixture.rs"]
mod julia_fixture;
mod support;
use support::{julia_mvmc_root, report_gate, require_gate, GateStatus};

#[derive(Clone, Copy)]
struct Model {
    name: &'static str,
    mode: &'static str,
}

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

const PHYSCAL_MODELS: &[Model] = &[
    Model {
        name: "heisenberg_chain_real",
        mode: "real",
    },
    Model {
        name: "heisenberg_chain_cmp",
        mode: "cmp",
    },
    Model {
        name: "heisenberg_chain_fsz",
        mode: "fsz",
    },
    Model {
        name: "hubbard_chain_real",
        mode: "real",
    },
    Model {
        name: "hubbard_chain_dh_real",
        mode: "real",
    },
    Model {
        name: "kondo_chain_real",
        mode: "real",
    },
];

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
    let fixture = trajectory_fixture(PHYSCAL_MODELS[0])
        .parent()
        .unwrap()
        .join("two-samples/hubbard_chain_real");
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
        // N commutes with H and equals the fixed particle count. Independently
        // acquired moments supply E/H^2; no Rust-generated golden values.
        // This is representative six-site coverage, not S144's two-site
        // helper with arbitrary H1=2,H2=5 and explicit half-weight.
        let moments = read_values(&fixture.join("expected/zvo_ls_qqqq_007.dat"));
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
    let fixture = trajectory_fixture(PHYSCAL_MODELS[0])
        .parent()
        .unwrap()
        .join("third-hubbard");
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
    // These are independently acquired C values, not the synthetic IO134
    // moments or a historical Julia solver golden. Strict rows/indices remain.
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
    let fixture = trajectory_fixture(PHYSCAL_MODELS[0])
        .parent()
        .unwrap()
        .join("third-hubbard");
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

fn fixture(root: &Path, model: Model) -> PathBuf {
    root.join("test/integration/reference")
        .join(model.name)
        .join("physcal_ref")
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

fn prepare(root: &Path, model: Model, seed: i64) -> (mvmc_core::PhysCalPreparation, PathBuf) {
    let fixture = fixture(root, model);
    let namelist = fixture.join("inputs/namelist.def");
    let opt = fixture.join("zqp_opt.dat");
    assert!(namelist.is_file(), "{}", namelist.display());
    assert!(opt.is_file(), "{}", opt.display());
    let preparation =
        mvmc_core::prepare_phys_cal_from_namelist(&namelist, &opt, model.mode, Some(seed))
            .unwrap_or_else(|error| panic!("{}: {error}", model.name));
    (preparation, fixture)
}

fn fixed_values(data: &mvmc_core::ExpertModeData) -> Vec<num_complex::Complex64> {
    data.projection_parameters()
        .into_iter()
        .chain(data.rbm_parameters())
        .chain(data.slater_params.iter().copied())
        .chain(data.opt_trans.iter().copied())
        .collect()
}

fn reference_fixed_values(fixture: &Path) -> (Vec<num_complex::Complex64>, usize) {
    // These fixtures have no In*.def overlays. In C, initial.def is not
    // implicitly read beside namelist.def: the explicit zqp argument supplies
    // ReadInitParameter's input. In particular FSZ's neighboring initial.def
    // must not replace these values.
    let inputs = fixture.join("inputs");
    let namelist = fs::read_to_string(inputs.join("namelist.def")).unwrap();
    let mut projection = 0;
    let mut slater = 0;
    for line in namelist
        .lines()
        .filter(|line| !line.trim().starts_with('#'))
    {
        let fields = line.split_whitespace().collect::<Vec<_>>();
        if fields.is_empty() {
            continue;
        }
        assert!(
            !fields[0].starts_with("In"),
            "uncovered C overlay contract: {line}"
        );
        let multiplier = match fields[0] {
            "Gutzwiller" | "Jastrow" => 1,
            "DH2" => 6,
            "DH4" => 10,
            "OrbitalParallel" => 2, // C: separate up-up and down-down slots.
            "Orbital" | "OrbitalAntiParallel" | "OrbitalGeneral" => 1,
            kind if kind.contains("RBM") || kind == "OptTrans" => {
                panic!("independent fixed-value layout not covered for {kind}")
            }
            _ => continue,
        };
        let definition = fs::read_to_string(inputs.join(fields[1])).unwrap();
        let count = definition
            .lines()
            .nth(1)
            .unwrap()
            .split_whitespace()
            .nth(1)
            .unwrap()
            .parse::<usize>()
            .unwrap();
        if fields[0].starts_with("Orbital") {
            slater += multiplier * count;
        } else {
            projection += multiplier * count;
        }
    }
    let count = projection + slater;
    let values = read_values(&fixture.join("zqp_opt.dat"));
    let record_width = 6 + 3 * count;
    assert!(!values.is_empty());
    assert_eq!(
        values.len() % record_width,
        0,
        "complete C parameter records"
    );
    // C reads until EOF, so the final complete record wins.
    let record = &values[values.len() - record_width..];
    let (triples, remainder) = record[6..].as_chunks::<3>();
    assert!(remainder.is_empty(), "complete C parameter triples");
    let mut expected = triples
        .iter()
        .map(|triple| num_complex::Complex64::new(triple[0], triple[1]))
        .collect::<Vec<_>>();
    // C SyncModifiedParameter rescales the Slater block to D_AmpMax=4.
    // Correlation shifts are disabled in this PhysCal preparation contract.
    let maximum = expected[projection..]
        .iter()
        .map(|v| v.re.hypot(v.im))
        .fold(0.0_f64, f64::max);
    assert!(maximum > 0.0);
    for value in &mut expected[projection..] {
        *value *= 4.0 / maximum;
    }
    (expected, count)
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

fn trajectory_fixture(model: Model) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/physcal_181")
        .join(model.name)
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

fn assert_saved_trajectory(model: Model, state: &mvmc_core::VmcOptimizationState) {
    let stage = trajectory_fixture(model).join("sample-0");
    assert_saved_trajectory_at(&stage, model.name, state);
}

fn assert_saved_trajectory_at(stage: &Path, label: &str, state: &mvmc_core::VmcOptimizationState) {
    let config = &state.electron_config;
    for (name, values) in [
        ("ele_idx", config.ele_idx.as_slice()),
        ("ele_cfg", config.ele_cfg.as_slice()),
        ("ele_num", config.ele_num.as_slice()),
        ("ele_proj_cnt", config.ele_proj_cnt.as_slice()),
        ("ele_spn", config.ele_spn.as_slice()),
        ("counter", config.counter.as_slice()),
    ] {
        let mut actual = values.to_vec();
        let mut expected = reference_integers(&stage.join(format!("{name}.txt")));
        if name == "counter" {
            // Julia has no exchange counters (slots 2, 3); C and Rust count them and
            // `mvmc-cli/tests/issue181_native_c_physcal.rs` compares them with native C.
            (actual[2], actual[3], expected[2], expected[3]) = (0, 0, 0, 0);
        }
        assert_discrete(&format!("{label} {name}"), &actual, &expected);
    }
}

fn assert_rng_checkpoint(model: Model, stage: &str, rng: &sfmt19937::Sfmt19937Rng) {
    let fixture = trajectory_fixture(model).join(stage);
    assert_rng_at(&fixture, &format!("{} {stage}", model.name), rng);
}

fn assert_rng_at(fixture: &Path, label: &str, rng: &sfmt19937::Sfmt19937Rng) {
    let expected = reference_integers(&fixture.join("next624.txt"));
    assert_eq!(expected.len(), 624, "{label} complete RNG block");
    let count = reference_integers(&fixture.join("draw-count.txt"));
    assert_eq!(count.len(), 1);
    assert!(count[0] >= 0);
    assert_eq!(
        rng.words_consumed(),
        count[0] as u128,
        "{label} actual primitive draw count"
    );
    // Validate the independent reference's documented position from seed 1,
    // then check the actual sampler position without advancing its RNG.
    let mut position = sfmt19937::Sfmt19937Rng::new(1);
    for _ in 0..count[0] {
        position.gen_rand32();
    }
    let next = |rng: &sfmt19937::Sfmt19937Rng| {
        let mut peek = rng.clone();
        (0..624)
            .map(|_| i64::from(peek.gen_rand32()))
            .collect::<Vec<_>>()
    };
    assert_discrete(
        &format!("{label} documented draw count"),
        &next(&position),
        &expected,
    );
    assert_discrete(&format!("{label} next624 RNG"), &next(rng), &expected);
}

fn check_independent_sampling_trajectory(model: Model) {
    use mvmc_expert_parsers::utils::{parameter_init::init_parameter, qp_weight::init_qp_weight};
    let fixture = trajectory_fixture(model);
    let mut preparation = mvmc_core::prepare_phys_cal_from_namelist(
        fixture.join("inputs/namelist.def"),
        fixture.join("zqp_opt.dat"),
        model.mode,
        Some(1),
    )
    .unwrap();
    assert_rng_checkpoint(model, "seeded", &preparation.rng);
    let consumed = reference_integers(&fixture.join("consumed-count.txt"));
    assert_eq!(consumed, vec![preparation.n_para_consumed as i64]);
    let expected = read_values(&fixture.join("fixed-parameters.txt"));
    assert_eq!(expected.len() % 2, 0);
    let (pairs, remainder) = expected.as_chunks::<2>();
    assert!(remainder.is_empty(), "complete complex fixture records");
    let expected = pairs
        .iter()
        .map(|pair| num_complex::Complex64::new(pair[0], pair[1]))
        .collect::<Vec<_>>();
    assert_fixed_values(&preparation.data, &expected);
    // C's single InitParameter advances RNG; the fixed zqp/overlay values are
    // restored before sampling. No runtime reference program is called here.
    init_parameter(&mut preparation.data.clone(), &mut preparation.rng).unwrap();
    assert_rng_checkpoint(model, "initialized", &preparation.rng);
    let data = &mut preparation.data;
    data.modpara.nmp_trans = data.modpara.nmp_trans.abs().max(1);
    data.modpara.vmc_calc_mode = 1;
    init_qp_weight(data);
    let complex = mvmc_core::get_all_complex_flag(data).unwrap();
    let fsz = data.i_flg_orbital_general != 0;
    let n_qp = data.qp_weights.as_ref().unwrap().qp_full_weight.len();
    let mut state = mvmc_core::VmcOptimizationState::zeros(
        data.modpara.nsite as usize,
        data.modpara.nelec as usize,
        data.projection_layout().n_proj,
        data.count_variational_parameters(),
        n_qp,
        data.modpara.nvmc_sample as usize,
        complex,
        fsz,
    );
    state.validate_declared_mode(data).unwrap();
    if fsz {
        mvmc_core::slater_update::update_slater_elm_fsz(data, &mut state);
    } else {
        mvmc_core::slater_update::update_slater_elm(data, &mut state);
    }
    assert_eq!(
        data.modpara.n_data_qty_smp, 1,
        "reference checkpoint coverage"
    );
    let stats = match (fsz, complex) {
        (false, false) => mvmc_core::vmc_make_sample_real(data, &mut state, &mut preparation.rng)
            .expect("normal real reference preparation"),
        (false, true) => mvmc_core::vmc_make_sample(data, &mut state, &mut preparation.rng)
            .expect("normal complex reference preparation"),
        (true, true) => {
            mvmc_core::sampling::driver::vmc_make_sample_fsz(data, &mut state, &mut preparation.rng)
        }
        (true, false) => {
            mvmc_core::sampling::vmc_make_sample_fsz_real(data, &mut state, &mut preparation.rng)
                .unwrap()
        }
    };
    assert_eq!(stats.saved, data.modpara.nvmc_sample as usize);
    assert_saved_trajectory(model, &state);
    assert_rng_checkpoint(model, "sample-0", &preparation.rng);
}

macro_rules! trajectory_test {
    ($name:ident, $index:expr) => {
        #[test]
        fn $name() {
            check_independent_sampling_trajectory(PHYSCAL_MODELS[$index]);
        }
    };
}

trajectory_test!(independent_trajectory_heisenberg_real, 0);
trajectory_test!(independent_trajectory_heisenberg_cmp, 1);
trajectory_test!(independent_trajectory_heisenberg_fsz, 2);
trajectory_test!(independent_trajectory_hubbard_real, 3);
trajectory_test!(independent_trajectory_hubbard_dh_real, 4);
trajectory_test!(independent_trajectory_kondo_real, 5);

#[test]
fn independent_trajectory_hubbard_dh_overlays() {
    check_independent_sampling_trajectory(Model {
        name: "hubbard_chain_dh_overlays",
        mode: "real",
    });
}

#[test]
fn independent_trajectory_hubbard_dh_opttrans() {
    check_independent_sampling_trajectory(Model {
        name: "hubbard_chain_dh_opttrans",
        mode: "real",
    });
}

#[test]
fn independent_trajectory_hubbard_dh_rbm_opttrans() {
    check_independent_sampling_trajectory(Model {
        name: "hubbard_chain_dh_rbm_opttrans",
        mode: "cmp",
    });
}

#[test]
fn independent_c_definition_component_flags() {
    for model in PHYSCAL_MODELS.iter().copied().chain([
        Model {
            name: "hubbard_chain_dh_overlays",
            mode: "real",
        },
        Model {
            name: "hubbard_chain_dh_opttrans",
            mode: "real",
        },
        Model {
            name: "hubbard_chain_dh_rbm_opttrans",
            mode: "cmp",
        },
    ]) {
        let fixture = trajectory_fixture(model);
        let preparation = mvmc_core::prepare_phys_cal_from_namelist(
            fixture.join("inputs/namelist.def"),
            fixture.join("zqp_opt.dat"),
            model.mode,
            Some(1),
        )
        .unwrap();
        let expected = reference_integers(&fixture.join("optimization-flags.txt"));
        let written = reference_integers(&fixture.join("optimization-flags-written.txt"));
        assert_eq!(expected.len(), preparation.data.optimization_flags.len());
        assert_eq!(written.len(), expected.len());
        // GetInfoOpt leaves real-section imaginary cells untouched; native
        // GetInfoOptTrans writes only a contiguous, non-tail slice. Do not
        // invent allocation contents for cells the C reader never writes.
        let actual = preparation
            .data
            .optimization_flags
            .iter()
            .zip(&written)
            .filter_map(|(flag, written)| (*written == 1).then_some(*flag))
            .collect::<Vec<_>>();
        let expected = expected
            .iter()
            .zip(&written)
            .filter_map(|(flag, written)| (*written == 1).then_some(*flag))
            .collect::<Vec<_>>();
        assert_discrete(
            &format!("{} C component flags", model.name),
            &actual,
            &expected,
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
    let reference = trajectory_fixture(PHYSCAL_MODELS[0])
        .parent()
        .unwrap()
        .join("duplicate-reader/result.txt");
    assert_eq!(
        fs::read_to_string(reference).unwrap(),
        "declared_records=2\ncount_after_mode2_dedup=1\nnormal_reader_status=1\nwith_gex_canonical_count=1\nindirect_reader_status=0\n"
    );
    // C readdef.c:577–585 deduplicates in mode2, then :1046 selects the
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
fn weighted_energy_and_green_averages_match_independent_kernel_fixture() {
    let dir = trajectory_fixture(PHYSCAL_MODELS[0])
        .parent()
        .unwrap()
        .join("weighted-average");
    let complex_values = |path: &Path| {
        let values = read_values(path);
        assert_eq!(values.len(), 24);
        let (pairs, remainder) = values.as_chunks::<2>();
        assert!(remainder.is_empty(), "complete complex fixture records");
        pairs
            .iter()
            .map(|pair| num_complex::Complex64::new(pair[0], pair[1]))
            .collect::<Vec<_>>()
    };
    let input = complex_values(&dir.join("accumulated.txt"));
    let expected = complex_values(&dir.join("averaged.txt"));
    let mut state = mvmc_core::VmcOptimizationState::zeros(2, 1, 0, 0, 1, 1, true, false);
    state.energy.wc = input[0];
    state.energy.etot = input[1];
    state.energy.etot2 = input[2];
    state.energy.sztot = input[3];
    state.energy.sztot2 = input[4];
    let mut phys = mvmc_core::state::PhysicalQuantities::zeros(3, 2, 2);
    phys.phys_cis_ajs.copy_from_slice(&input[5..8]);
    phys.phys_cis_ajs_ckt_alt.copy_from_slice(&input[8..10]);
    phys.phys_cis_ajs_ckt_alt_dc.copy_from_slice(&input[10..12]);
    state.phys_quantities = Some(phys);
    mvmc_core::average::weight_average_we(&mut state);
    mvmc_core::observables::weight_average_green_func_fsz(&mut state);
    assert_eq!(
        state.energy.wc, input[0],
        "normalization must retain weight"
    );
    let mut actual = vec![
        state.energy.wc,
        state.energy.etot,
        state.energy.etot2,
        state.energy.sztot,
        state.energy.sztot2,
    ];
    let phys = state.phys_quantities.as_ref().unwrap();
    actual.extend_from_slice(&phys.phys_cis_ajs);
    actual.extend_from_slice(&phys.phys_cis_ajs_ckt_alt);
    actual.extend_from_slice(&phys.phys_cis_ajs_ckt_alt_dc);
    for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
        // Small well-conditioned reciprocal/multiply fixture (|Wc| > 2).
        // Retain the existing one-body portable numeric contract; no RNG or
        // configuration comparison is relaxed by this kernel-only test.
        for (a, e) in [(a.re, e.re), (a.im, e.im)] {
            assert!(a.is_finite() && e.is_finite());
            assert!(
                (a - e).abs() <= 1e-12_f64.max(1e-10 * a.abs().max(e.abs())),
                "weighted component {index}: actual={a} reference={e}"
            );
        }
    }
    let native = read_values(
        &dir.parent()
            .unwrap()
            .join("native-c-weighted-green/synthetic-green.txt"),
    );
    let (pairs, remainder) = native.as_chunks::<2>();
    assert!(remainder.is_empty());
    assert_eq!(pairs.len(), actual.len() - 5);
    for (index, (a, e)) in actual[5..].iter().zip(pairs).enumerate() {
        for (a, e) in [(a.re, e[0]), (a.im, e[1])] {
            assert!(
                (a - e).abs() <= 1e-12_f64.max(1e-10 * a.abs().max(e.abs())),
                "native C synthetic Green {index}: actual={a} reference={e}"
            );
        }
    }
}

#[test]
fn duplicate_one_body_with_gex_matches_native_c_canonical_layout_fixture() {
    let fixture = trajectory_fixture(PHYSCAL_MODELS[0])
        .parent()
        .unwrap()
        .join("duplicate-reader");
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
        let result = mvmc_core::vmc_phys_cal_to_dir(preparation, &out).unwrap();
        assert_saved_trajectory(PHYSCAL_MODELS[0], &result.state);
        assert_rng_checkpoint(PHYSCAL_MODELS[0], "sample-0", &result.final_rng);
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

// Optional failed-test stderr artifact. This is a measurement-only diagnostic
// replay on private clones of the actual saved frame, NOT production observation
// of individual additions, an independent oracle, or a second sampler.
fn fsz_measurement_diagnostic(
    data: &mvmc_expert_parsers::ExpertModeData,
    actual: &mvmc_core::VmcOptimizationState,
    rng: &sfmt19937::Sfmt19937Rng,
    frame: usize,
) {
    use mvmc_core::observables::{calculate_ip_complex, calculate_local_energy_fsz};
    use num_complex::Complex64;
    let mut peek = rng.clone();
    let next = (0..624).map(|_| peek.gen_rand32()).collect::<Vec<_>>();
    eprintln!(
        "FSZ181 frame={frame} scope=private-measurement-replay cfg={:?} count={} next624={next:?}",
        actual.electron_config,
        rng.words_consumed()
    );
    if !mvmc_core::get_all_complex_flag(data).unwrap() || data.has_rbm_terms() {
        eprintln!(
            "FSZ181 unsupported replay mode: only native complex/no-RBM; no production changes"
        );
        return;
    }
    let ns = data.modpara.nsite as usize;
    let ne = data.modpara.nelec as usize;
    let nq = actual.slater_matrix.pf_m.len();
    let mut state = mvmc_core::VmcOptimizationState::zeros(
        ns,
        ne,
        data.projection_layout().n_proj,
        data.count_variational_parameters(),
        nq,
        data.modpara.nvmc_sample as usize,
        true,
        true,
    );
    // Only the actual Slater input plane is needed; inverse/Pfaffian and
    // observable scratch are independently allocated and rebuilt below.
    state.slater_matrix.slater_elm = actual.slater_matrix.slater_elm.clone();
    let pool = mvmc_core::state::ThreadedPfaPackWorkspace::new(2 * ne, 1);
    let mut empty = data.clone();
    empty.coulomb_intra_terms.clear();
    empty.coulomb_inter_terms.clear();
    empty.hund_terms.clear();
    empty.transfer_terms.clear();
    empty.pair_hop_terms.clear();
    empty.exchange_terms.clear();
    empty.inter_all_terms.clear();
    let mut sum = Complex64::new(0.0, 0.0);
    let mut sum_abs_im = 0.0;
    for walker in 0..data.modpara.nvmc_sample as usize {
        let cfg = &actual.electron_config;
        let idx = cfg.ele_idx_slice(walker);
        let spins = cfg.ele_spn_slice(walker);
        if idx.iter().all(|&x| x == 0) || idx.iter().all(|&x| x < 0) {
            eprintln!("FSZ181 frame={frame} walker={walker} skipped=empty-config");
            continue;
        }
        let result = mvmc_core::pfaffian::calc_m_all_fsz_complex(
            idx,
            spins,
            &state.slater_matrix.slater_elm,
            &mut state.slater_matrix.inv_m,
            &mut state.slater_matrix.pf_m,
            0,
            nq,
            ns,
            ne,
            &pool,
        );
        if let Err(error) = result {
            eprintln!("FSZ181 frame={frame} walker={walker} refresh-error={error:?}");
            continue;
        }
        let ip = calculate_ip_complex(&state.slater_matrix.pf_m, 0, nq, data);
        eprintln!(
            "FSZ181 frame={frame} walker={walker} ip={ip:?} pf={:?} qp_weights={:?}",
            state.slater_matrix.pf_m, data.qp_weights
        );
        for qp in 0..nq {
            eprintln!(
                "FSZ181 frame={frame} walker={walker} qp={qp} inverse={:?}",
                state.slater_matrix.inv_m.qp_matrix_slice(qp)
            );
        }
        if ip.norm() < 1e-100 {
            eprintln!("FSZ181 frame={frame} walker={walker} skipped=tiny-ip");
            continue;
        }
        let mut evaluate = |input: &mvmc_expert_parsers::ExpertModeData| {
            calculate_local_energy_fsz(
                ip,
                input,
                &mut state,
                idx,
                cfg.ele_cfg_slice(walker),
                cfg.ele_num_slice(walker),
                cfg.ele_proj_cnt_slice(walker),
                spins,
            )
        };
        let full = evaluate(data);
        let mut reconstructed = Complex64::new(0.0, 0.0);
        macro_rules! contributions {
            ($field:ident) => {
                for (term_index, term) in data.$field.iter().enumerate() {
                    let mut isolated = empty.clone();
                    isolated.$field.push(term.clone());
                    let contribution = evaluate(&isolated);
                    sum_abs_im += contribution.im.abs();
                    reconstructed += contribution;
                    eprintln!("FSZ181 frame={frame} walker={walker} family={} term={term_index} operands={term:?} isolated_contribution={contribution:?} reconstructed={reconstructed:?}", stringify!($field));
                }
            };
        }
        contributions!(coulomb_intra_terms);
        contributions!(coulomb_inter_terms);
        contributions!(hund_terms);
        contributions!(transfer_terms);
        contributions!(pair_hop_terms);
        contributions!(exchange_terms);
        contributions!(inter_all_terms);
        sum += full;
        eprintln!("FSZ181 frame={frame} walker={walker} full_local={full:?} isolated_sum={reconstructed:?} sum={sum:?} sum_abs_im={sum_abs_im:.17e}");
    }
    eprintln!("FSZ181 frame={frame} replay_sum={sum:?} sum_abs_im={sum_abs_im:.17e}; not an oracle or tolerance justification");
}

#[test]
fn two_sample_runners_match_independent_saved_states_rng_and_ordered_outputs() {
    struct RawGreenFrame {
        sample: usize,
        use_fsz: bool,
        weight: num_complex::Complex64,
        arrays: [Vec<num_complex::Complex64>; 3],
    }
    struct RawGreenCapture {
        frames: std::cell::RefCell<Vec<RawGreenFrame>>,
        completed: std::cell::RefCell<Vec<usize>>,
        fixture: PathBuf,
        label: &'static str,
    }
    impl mvmc_core::run::PhysCalGreenObserver for RawGreenCapture {
        fn sample_completed(
            &self,
            data: &mvmc_expert_parsers::ExpertModeData,
            sample: usize,
            state: &mvmc_core::VmcOptimizationState,
            rng: &sfmt19937::Sfmt19937Rng,
        ) {
            // Actual production sampler boundary; all discrete assertions
            // precede measurement and therefore any numerical failure.
            let stage = self.fixture.join(format!("sample-{sample}"));
            let label = format!("{} sample {sample}", self.label);
            assert_saved_trajectory_at(&stage, &label, state);
            assert_rng_at(&stage, &label, rng);
            if self.label == "heisenberg_chain_fsz"
                && std::env::var("MVMC_PHYSCAL181_FSZ_DIAGNOSTIC").as_deref() == Ok("1")
            {
                fsz_measurement_diagnostic(data, state, rng, sample);
            }
            self.completed.borrow_mut().push(sample);
        }
        fn accumulated(&self, view: mvmc_core::run::PhysCalGreenView<'_>) {
            self.frames.borrow_mut().push(RawGreenFrame {
                sample: view.sample,
                use_fsz: view.use_fsz,
                weight: view.weight,
                arrays: [
                    view.one_body.to_vec(),
                    view.factored_two_body.to_vec(),
                    view.direct_two_body.to_vec(),
                ],
            });
        }
    }
    #[derive(Default)]
    struct RecordingReducer(std::cell::RefCell<Vec<Vec<num_complex::Complex64>>>);
    impl mvmc_core::Reducer for RecordingReducer {
        fn allreduce_sum_f64(&self, _: &mut [f64]) {}
        fn allreduce_sum_i64(&self, _: &mut [i64]) {}
        fn allreduce_sum_c64(&self, values: &mut [num_complex::Complex64]) {
            self.0.borrow_mut().push(values.to_vec());
        }
    }
    // Platform-specific independent two-sample references override the archived
    // Linux PhysCal lineage when present, matching the CG runner overlays.
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures");
    let root = julia_fixture::arm_directory(&fixtures)
        .map(|dir| dir.join("physcal_181/two-samples"))
        .filter(|dir| dir.is_dir())
        .unwrap_or_else(|| {
            trajectory_fixture(PHYSCAL_MODELS[0])
                .parent()
                .unwrap()
                .join("two-samples")
        });
    for model in PHYSCAL_MODELS.iter().copied().chain([
        Model {
            name: "hubbard_chain_dh_overlays",
            mode: "real",
        },
        Model {
            name: "hubbard_chain_dh_opttrans",
            mode: "real",
        },
        Model {
            name: "hubbard_chain_dh_rbm_opttrans",
            mode: "cmp",
        },
    ]) {
        let fixture = root.join(model.name);
        let preparation = mvmc_core::prepare_phys_cal_from_namelist(
            fixture.join("inputs/namelist.def"),
            fixture.join("zqp_opt.dat"),
            model.mode,
            Some(1),
        )
        .unwrap();
        assert_eq!(preparation.data.modpara.n_data_qty_smp, 2);
        assert_eq!(preparation.data.modpara.n_data_idx_start, 7);
        let before = fixed_values(&preparation.data);
        let flags = preparation.data.optimization_flags.clone();
        let lanczos_mode = preparation.data.modpara.lanczos_mode;
        let use_fsz = preparation.data.i_flg_orbital_general != 0;
        let complex_sr_calls = if mvmc_core::get_all_complex_flag(&preparation.data).unwrap() {
            2
        } else {
            0
        };
        let out = output_dir("two-independent-frames", model.name);
        let reducer = RecordingReducer::default();
        let raw_green = std::rc::Rc::new(RawGreenCapture {
            frames: std::cell::RefCell::new(Vec::new()),
            completed: std::cell::RefCell::new(Vec::new()),
            fixture: fixture.clone(),
            label: model.name,
        });
        let observer = mvmc_core::run::install_physcal_green_observer(raw_green.clone()).unwrap();
        let result =
            mvmc_core::vmc_phys_cal_with_reducer(preparation, Some(&out), &reducer).unwrap();
        drop(observer);
        assert_eq!(result.iterations, 2);
        assert_eq!(*raw_green.completed.borrow(), [0, 1]);
        // Check discrete/fixed contracts before any numerical comparison, so a
        // floating-point failure cannot hide a configuration or RNG mismatch.
        assert_eq!(
            fixed_values(&result.data),
            before,
            "{} fixed coefficients",
            model.name
        );
        assert_eq!(
            result.data.optimization_flags, flags,
            "{} flags",
            model.name
        );
        assert_saved_trajectory_at(&fixture.join("sample-1"), model.name, &result.state);
        assert_rng_at(&fixture.join("sample-1"), model.name, &result.final_rng);
        // Observe the actual serial runner before normalization, not a replay
        // sampler or a fixture-loaded averaging kernel. These independent
        // Julia accumulators use the C-ordered accumulation contract; the
        // native C kernel comparison below independently checks their means.
        let raw_frames = raw_green.frames.borrow();
        assert_eq!(raw_frames.len(), 2, "{} actual raw frames", model.name);
        for (sample, frame) in raw_frames.iter().enumerate() {
            assert_eq!(frame.sample, sample);
            assert_eq!(frame.use_fsz, use_fsz);
            let stage = fixture.join(format!("accumulated-{sample}"));
            let energy = read_values(&stage.join("energy.txt"));
            assert_eq!(energy.len(), 10, "complete independent energy record");
            assert_eq!(
                frame.weight,
                num_complex::Complex64::new(energy[0], energy[1]),
                "{} sample {sample} raw weight",
                model.name
            );
            for (actual, file) in frame
                .arrays
                .iter()
                .zip(["one.txt", "factored.txt", "direct.txt"])
            {
                let numbers = read_values(&stage.join(file));
                // The accumulated two-body "direct" array of the FSZ record sums
                // cancelling terms: the one-ulp GEMV/ztrmm differences of a kernel
                // without the reference lineage's FMA arithmetic (inverse entries first,
                // docs/NUMERICAL_COMPARISONS.md "BLAS provider matrix", #455) reach
                // 1.9e-9 relative / 6e-11 absolute there (Sandybridge, Nehalem), while
                // the one-body and factored arrays stay within the strict bound.
                let (abs_bound, rel_bound) = if file == "direct.txt"
                    && julia_fixture::kernel_class(&fixtures)
                        == julia_fixture::KernelClass::Unverified
                {
                    (1e-9_f64, 1e-7_f64)
                } else {
                    (1e-12_f64, 1e-10_f64)
                };
                let (pairs, remainder) = numbers.as_chunks::<2>();
                assert!(remainder.is_empty(), "complete raw complex records");
                assert_eq!(
                    actual.len(),
                    pairs.len(),
                    "{} sample {sample} raw {file} shape",
                    model.name
                );
                for (column, (actual, expected)) in actual.iter().zip(pairs).enumerate() {
                    for (a, e) in [(actual.re, expected[0]), (actual.im, expected[1])] {
                        assert!(a.is_finite() && e.is_finite());
                        assert!(
                            (a - e).abs() <= abs_bound.max(rel_bound * a.abs().max(e.abs())),
                            "{} sample {sample} raw {file} column {column}: {a} != {e}",
                            model.name
                        );
                    }
                }
            }
        }
        let calls = reducer.0.borrow();
        // reduce_accumulators: energy, active SR OO/HO (complex only for
        // AllComplexFlag; otherwise f64), three Green arrays, four LS arrays.
        // Energy remains raw; Green arrays have already been
        // normalized inside accumulate_observables (C PhysCal boundary).
        let calls_per_frame = 8 + complex_sr_calls;
        assert_eq!(
            calls.len(),
            2 * calls_per_frame,
            "{} two reduction frames",
            model.name
        );
        for sample in 0..2 {
            for (slot, stage, file) in [
                (0, "accumulated", "energy.txt"),
                (1 + complex_sr_calls, "averaged", "one.txt"),
                (2 + complex_sr_calls, "averaged", "factored.txt"),
                (3 + complex_sr_calls, "averaged", "direct.txt"),
            ] {
                let numbers = read_values(&fixture.join(format!("{stage}-{sample}/{file}")));
                let (pairs, remainder) = numbers.as_chunks::<2>();
                assert!(remainder.is_empty(), "complete complex stage records");
                let expected = pairs
                    .iter()
                    .map(|v| num_complex::Complex64::new(v[0], v[1]))
                    .collect::<Vec<_>>();
                let actual = &calls[calls_per_frame * sample + slot];
                assert_eq!(
                    actual.len(),
                    expected.len(),
                    "{} sample {sample} reduction {file} shape",
                    model.name
                );
                // The FSZ accumulated energy inherits the same cancellation as the raw
                // "direct" array above: 7e-11 absolute / 1.8e-10 relative on reference
                // BLAS (the only other-provider deviation; Sandybridge, Nehalem and
                // Prescott stay within the strict bound for the energy record).
                let (abs_bound, rel_bound) = if file == "energy.txt"
                    && model.name == "heisenberg_chain_fsz"
                    && julia_fixture::kernel_class(&fixtures)
                        == julia_fixture::KernelClass::Unverified
                {
                    (1e-9_f64, 1e-7_f64)
                } else {
                    (1e-12_f64, 1e-10_f64)
                };
                for (column, (a, e)) in actual.iter().zip(expected).enumerate() {
                    for (a, e) in [(a.re, e.re), (a.im, e.im)] {
                        assert!(a.is_finite() && e.is_finite());
                        assert!(
                            (a - e).abs() <= abs_bound.max(rel_bound * a.abs().max(e.abs())),
                            "{} sample {sample} {stage} {file} column {column}: {a} != {e}",
                            model.name
                        );
                    }
                }
                if file != "energy.txt" {
                    let native = read_values(
                        &root
                            .parent()
                            .unwrap()
                            .join("native-c-weighted-green")
                            .join(model.name)
                            .join(format!("frame-{sample}"))
                            .join(file),
                    );
                    let (pairs, remainder) = native.as_chunks::<2>();
                    assert!(remainder.is_empty());
                    let expected = pairs
                        .iter()
                        .map(|v| num_complex::Complex64::new(v[0], v[1]))
                        .collect::<Vec<_>>();
                    assert_eq!(actual.len(), expected.len(), "native C Green shape");
                    for (column, (a, e)) in actual.iter().zip(expected).enumerate() {
                        for (a, e) in [(a.re, e.re), (a.im, e.im)] {
                            assert!(a.is_finite() && e.is_finite());
                            assert!(
                                (a - e).abs() <= 1e-12_f64.max(1e-10 * a.abs().max(e.abs())),
                                "{} sample {sample} native C {file} column {column}: {a} != {e}",
                                model.name
                            );
                        }
                    }
                }
            }
        }
        let expected_dir = fixture.join("expected");
        let names = |dir: &Path| {
            fs::read_dir(dir)
                .unwrap()
                .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
                .collect::<BTreeSet<_>>()
        };
        let expected_names = names(&expected_dir);
        // Every enabled LS output now has an independent numerical fixture
        // for BOTH frames; file-presence checks alone cannot satisfy parity.
        for index in 7..=8 {
            if lanczos_mode > 0 {
                for suffix in ["ls_out", "ls_qqqq"] {
                    assert!(expected_names.contains(&format!("zvo_{suffix}_{index:03}.dat")));
                }
            }
            if lanczos_mode > 1 {
                for suffix in ["ls_cisajs", "ls_cisajscktalt", "ls_cisajscktaltex"] {
                    assert!(expected_names.contains(&format!("zvo_{suffix}_{index:03}.dat")));
                }
            }
        }
        // C InitFile also creates one `_time_` file (NDataIdxStart=7); it is not
        // part of the numerical reference set (its rows end in a ctime string).
        let mut actual_names = names(&out);
        assert!(actual_names.iter().any(|name| name == "zvo_time_007.dat"));
        actual_names.retain(|name| !name.contains("_time_"));
        assert_eq!(
            actual_names, expected_names,
            "{} two-frame exact file set",
            model.name
        );
        for name in expected_names {
            assert_reference(&out.join(&name), &expected_dir.join(&name));
        }
        // Weight is not printed in out.dat; prove the actual second frame
        // retained its independently observed accumulator normalization weight.
        let energy = read_values(&fixture.join("averaged-1/energy.txt"));
        assert_eq!(energy.len(), 10);
        assert_eq!(
            result.state.energy.wc,
            num_complex::Complex64::new(energy[0], energy[1])
        );
        fs::remove_dir_all(out).unwrap();
    }
}

#[test]
fn energy_only_physcal_zero_green_file_sets_follow_native_c_all_lanczos_modes() {
    let root = trajectory_fixture(PHYSCAL_MODELS[0])
        .parent()
        .unwrap()
        .to_owned();
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
fn all_six_non_interall_terms_match_independent_lanczos_modes() {
    for (name, mode) in [
        ("hubbard_all_terms_lanczos1", 1),
        ("hubbard_all_terms_lanczos2", 2),
    ] {
        let model = Model { name, mode: "real" };
        let fixture = trajectory_fixture(model);
        let preparation = mvmc_core::prepare_phys_cal_from_namelist(
            fixture.join("inputs/namelist.def"),
            fixture.join("zqp_opt.dat"),
            "real",
            Some(1),
        )
        .unwrap();
        let data = &preparation.data;
        assert_eq!(data.modpara.lanczos_mode, mode);
        assert!(data.inter_all_terms.is_empty());
        assert!(data
            .transfer_terms
            .iter()
            .any(|term| term.value.norm() > 0.0));
        assert!(data
            .coulomb_intra_terms
            .iter()
            .any(|term| term.value != 0.0));
        let metadata = fs::read_to_string(
            fixture
                .parent()
                .unwrap()
                .join("all-terms-reader/ordered-pairs.txt"),
        )
        .unwrap();
        let actual = data
            .coulomb_inter_terms
            .iter()
            .map(|t| ("CoulombInter", t.site1, t.site2, t.value))
            .chain(
                data.hund_terms
                    .iter()
                    .map(|t| ("Hund", t.site1, t.site2, t.value)),
            )
            .chain(
                data.exchange_terms
                    .iter()
                    .map(|t| ("Exchange", t.site1, t.site2, t.value)),
            )
            .chain(
                data.pair_hop_terms
                    .iter()
                    .map(|t| ("PairHop", t.site1, t.site2, t.value)),
            )
            .collect::<Vec<_>>();
        let expected = metadata
            .lines()
            .map(|line| {
                let fields = line.split_whitespace().collect::<Vec<_>>();
                assert_eq!(fields.len(), 4);
                (
                    fields[0],
                    fields[1].parse::<i64>().unwrap(),
                    fields[2].parse::<i64>().unwrap(),
                    fields[3].parse::<f64>().unwrap(),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            actual, expected,
            "actual C ordered Hamiltonian reader records"
        );
        let fixed = reference_fixed_values(&fixture);
        assert_eq!(preparation.n_para_consumed, fixed.1);
        assert_eq!(
            reference_integers(&fixture.join("consumed-count.txt")),
            vec![fixed.1 as i64]
        );
        assert_fixed_values(data, &fixed.0);
        let flags = reference_integers(&fixture.join("optimization-flags.txt"));
        let written = reference_integers(&fixture.join("optimization-flags-written.txt"));
        assert_eq!(flags.len(), data.optimization_flags.len());
        assert_eq!(written.len(), flags.len());
        for (index, (&flag, &mask)) in flags.iter().zip(&written).enumerate() {
            if mask == 1 {
                assert_eq!(
                    data.optimization_flags[index], flag,
                    "C written flag {index}"
                );
            }
        }
        assert_rng_checkpoint(model, "seeded", &preparation.rng);
        let mut initialized_rng = preparation.rng.clone();
        mvmc_expert_parsers::utils::parameter_init::init_parameter(
            &mut data.clone(),
            &mut initialized_rng,
        )
        .unwrap();
        assert_rng_checkpoint(model, "initialized", &initialized_rng);
        let out = output_dir("all-terms-lanczos", name);
        let result = mvmc_core::vmc_phys_cal_to_dir(preparation, &out).unwrap();
        assert_fixed_values(&result.data, &fixed.0);
        assert_saved_trajectory(model, &result.state);
        assert_rng_checkpoint(model, "sample-0", &result.final_rng);
        for entry in fs::read_dir(fixture.join("expected")).unwrap() {
            let path = entry.unwrap().path();
            let file = path.file_name().unwrap();
            assert_reference(&out.join(file), &path);
        }
        let expected_ls = fs::read_dir(fixture.join("expected"))
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .filter(|n| n.to_string_lossy().starts_with("zvo_ls_"))
            .collect::<BTreeSet<_>>();
        let actual_ls = fs::read_dir(&out)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .filter(|n| n.to_string_lossy().starts_with("zvo_ls_"))
            .collect::<BTreeSet<_>>();
        assert_eq!(actual_ls, expected_ls);
        assert_eq!(expected_ls.len(), if mode == 1 { 2 } else { 5 });
        let _ = fs::remove_dir_all(out);
    }
}

#[test]
fn both_measurement_frames_normalize_independent_accumulators_in_order() {
    let root = trajectory_fixture(PHYSCAL_MODELS[0])
        .parent()
        .unwrap()
        .join("two-samples");
    let read_complex = |path: &Path| {
        let numbers = read_values(path);
        assert_eq!(numbers.len() % 2, 0);
        let (pairs, remainder) = numbers.as_chunks::<2>();
        assert!(remainder.is_empty(), "complete complex fixture records");
        pairs
            .iter()
            .map(|v| num_complex::Complex64::new(v[0], v[1]))
            .collect::<Vec<_>>()
    };
    for name in PHYSCAL_MODELS.iter().map(|model| model.name).chain([
        "hubbard_chain_dh_overlays",
        "hubbard_chain_dh_opttrans",
        "hubbard_chain_dh_rbm_opttrans",
    ]) {
        for sample in 0..2 {
            let accumulated = root.join(name).join(format!("accumulated-{sample}"));
            let averaged = root.join(name).join(format!("averaged-{sample}"));
            let input = read_complex(&accumulated.join("energy.txt"));
            assert_eq!(input.len(), 5);
            let one = read_complex(&accumulated.join("one.txt"));
            let factored = read_complex(&accumulated.join("factored.txt"));
            let direct = read_complex(&accumulated.join("direct.txt"));
            let mut state = mvmc_core::VmcOptimizationState::zeros(2, 1, 0, 0, 1, 1, true, false);
            state.energy.wc = input[0];
            state.energy.etot = input[1];
            state.energy.etot2 = input[2];
            state.energy.sztot = input[3];
            state.energy.sztot2 = input[4];
            let mut phys = mvmc_core::state::PhysicalQuantities::zeros(
                one.len(),
                factored.len(),
                direct.len(),
            );
            phys.phys_cis_ajs = one;
            phys.phys_cis_ajs_ckt_alt = factored;
            phys.phys_cis_ajs_ckt_alt_dc = direct;
            state.phys_quantities = Some(phys);
            mvmc_core::average::weight_average_we(&mut state);
            mvmc_core::observables::weight_average_green_func_fsz(&mut state);
            assert_eq!(state.energy.wc, input[0]);
            let phys = state.phys_quantities.as_ref().unwrap();
            for (file, actual) in [
                (
                    "energy.txt",
                    vec![
                        state.energy.wc,
                        state.energy.etot,
                        state.energy.etot2,
                        state.energy.sztot,
                        state.energy.sztot2,
                    ],
                ),
                ("one.txt", phys.phys_cis_ajs.clone()),
                ("factored.txt", phys.phys_cis_ajs_ckt_alt.clone()),
                ("direct.txt", phys.phys_cis_ajs_ckt_alt_dc.clone()),
            ] {
                let expected = read_complex(&averaged.join(file));
                assert_eq!(
                    actual.len(),
                    expected.len(),
                    "{name} sample {sample} {file} shape"
                );
                for (column, (a, e)) in actual.iter().zip(expected).enumerate() {
                    for (a, e) in [(a.re, e.re), (a.im, e.im)] {
                        assert!(a.is_finite() && e.is_finite());
                        assert!(
                            (a - e).abs() <= 1e-12_f64.max(1e-10 * a.abs().max(e.abs())),
                            "{name} sample {sample} {file} column {column}: {a} != {e}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn fixed_records_overlays_and_normalization_follow_native_c_stages() {
    use mvmc_expert_parsers::utils::{
        parameter_init::init_parameter, read_input_parameters::read_input_parameters,
    };
    let root = trajectory_fixture(PHYSCAL_MODELS[0])
        .parent()
        .unwrap()
        .to_owned();
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
            mvmc_expert_parsers::parse_expert_mode_files_with_c_opt_trans(&namelist, true).unwrap();
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

#[test]
#[ignore = "optional PhysCal gate; set MVMC_RS_PHYSCAL_181=1 and explicitly run ignored tests"]
fn six_supported_physcal_models_preserve_fixed_values_flags_and_outputs() {
    require_gate("physcal-issue181", "MVMC_RS_PHYSCAL_181");
    let root = julia_mvmc_root().unwrap_or_else(|| {
        support::missing_fixture("physcal-issue181", "Julia-mVMC checkout not found")
    });

    for model in PHYSCAL_MODELS {
        let (preparation, fixture) = prepare(&root, *model, 1);
        let before = preparation.data.clone();
        let (expected_values, expected_count) = reference_fixed_values(&fixture);
        assert_fixed_values(&before, &expected_values);
        assert_eq!(before.count_variational_parameters(), expected_count);
        assert_eq!(preparation.n_para_consumed, expected_count);
        assert!(before
            .optimization_flags
            .iter()
            .all(|flag| *flag == 0 || *flag == 1));
        assert!(
            before.inter_all_terms.is_empty(),
            "{} is InterAll",
            model.name
        );
        if model.mode == "fsz" {
            assert_ne!(before.i_flg_orbital_general, 0, "{}", model.name);
            assert_ne!(before.i_flg_orbital_parallel, 0, "{}", model.name);
            assert!(
                fixture.join("inputs/initial.def").is_file(),
                "FSZ overlay fixture missing"
            );
        }

        let out = output_dir("matrix", model.name);
        let result = mvmc_core::vmc_phys_cal_to_dir(preparation, &out).unwrap();
        assert_saved_trajectory(*model, &result.state);
        assert_rng_checkpoint(*model, "sample-0", &result.final_rng);
        assert_fixed_values(&result.data, &expected_values);
        assert_eq!(
            fixed_values(&result.data),
            fixed_values(&before),
            "{} fixed-value preservation",
            model.name
        );
        assert_eq!(
            result.data.optimization_flags, before.optimization_flags,
            "{} optimization flags",
            model.name
        );
        assert_eq!(result.iterations, 1, "{} iterations", model.name);

        let expected_dir = fixture.join("expected");
        let expected_names = fs::read_dir(&expected_dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .chain(["zvo_out_001.dat".to_owned(), "zvo_var_001.dat".to_owned()])
            .collect::<BTreeSet<_>>();
        let actual_names = fs::read_dir(&out)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect::<BTreeSet<_>>();
        assert_eq!(
            actual_names, expected_names,
            "{} output file set",
            model.name
        );
        for name in expected_names {
            if name.starts_with("zvo_out_") || name.starts_with("zvo_var_") {
                // Independent two-sample reference's first frame has the same
                // fixed inputs/draw stage; only the file index changes 7→1.
                let expected = trajectory_fixture(*model)
                    .parent()
                    .unwrap()
                    .join("two-samples")
                    .join(model.name)
                    .join("expected")
                    .join(name.replace("_001.", "_007."));
                assert_reference(&out.join(&name), &expected);
            } else {
                assert_reference(&out.join(&name), &expected_dir.join(name));
            }
        }
        let _ = fs::remove_dir_all(out);
        report_gate("physcal-issue181", GateStatus::Pass, model.name);
    }
}

#[test]
#[ignore = "optional PhysCal gate; set MVMC_RS_PHYSCAL_181=1 and explicitly run ignored tests"]
fn serial_physcal_indexes_multiple_samples_and_reruns_deterministically() {
    require_gate("physcal-issue181-contract", "MVMC_RS_PHYSCAL_181");
    let root = julia_mvmc_root().unwrap_or_else(|| {
        support::missing_fixture("physcal-issue181-contract", "Julia-mVMC checkout not found")
    });
    let model = PHYSCAL_MODELS[0];
    let out = output_dir("rerun", model.name);

    let (mut preparation, _) = prepare(&root, model, 1);
    preparation.data.modpara.n_data_idx_start = 7;
    preparation.data.modpara.n_data_qty_smp = 2;
    let first = mvmc_core::vmc_phys_cal_to_dir(preparation, &out).unwrap();
    assert_eq!(first.iterations, 2);
    let independent = trajectory_fixture(model)
        .parent()
        .unwrap()
        .join("two-samples")
        .join(model.name);
    assert_saved_trajectory_at(&independent.join("sample-1"), "second sample", &first.state);
    assert_rng_at(
        &independent.join("sample-1"),
        "actual runner final RNG",
        &first.final_rng,
    );
    for entry in fs::read_dir(independent.join("expected")).unwrap() {
        let entry = entry.unwrap();
        assert_reference(&out.join(entry.file_name()), &entry.path());
    }
    let first_snapshot = snapshot(&out);
    assert!(first_snapshot
        .iter()
        .any(|(name, _)| name.contains("_007.")));
    assert!(first_snapshot
        .iter()
        .any(|(name, _)| name.contains("_008.")));

    let (mut rerun, _) = prepare(&root, model, 1);
    rerun.data.modpara.n_data_idx_start = 7;
    rerun.data.modpara.n_data_qty_smp = 2;
    let second = mvmc_core::vmc_phys_cal_to_dir(rerun, &out).unwrap();
    assert_eq!(second.state.electron_config, first.state.electron_config);
    assert_eq!(second.state.energy, first.state.energy);
    assert_eq!(snapshot(&out), first_snapshot);
    let _ = fs::remove_dir_all(out);
}

#[test]
#[ignore = "optional PhysCal gate; set MVMC_RS_PHYSCAL_181=1 and explicitly run ignored tests"]
fn non_interall_lanczos_modes_one_and_two_write_the_defined_files() {
    require_gate("physcal-issue181-lanczos", "MVMC_RS_PHYSCAL_181");
    let root = julia_mvmc_root().unwrap_or_else(|| {
        support::missing_fixture("physcal-issue181-lanczos", "Julia-mVMC checkout not found")
    });

    for model_name in [
        "hubbard_chain_lanczos",
        "spin_chain_lanczos",
        "hubbard_chain_real",
    ] {
        let model = Model {
            name: model_name,
            mode: "real",
        };
        let (_, fixture) = prepare(&root, model, 1);
        for lanczos_mode in [1, 2] {
            let (mut preparation, _) = prepare(&root, model, 1);
            preparation.data.modpara.lanczos_mode = lanczos_mode;
            let (expected_values, expected_count) = reference_fixed_values(&fixture);
            assert_eq!(preparation.n_para_consumed, expected_count);
            assert_fixed_values(&preparation.data, &expected_values);
            assert!(preparation.data.inter_all_terms.is_empty());
            if model_name.starts_with("hubbard") {
                assert!(
                    !preparation.data.transfer_terms.is_empty(),
                    "Lanczos hopping branch"
                );
                assert!(
                    !preparation.data.coulomb_intra_terms.is_empty(),
                    "Lanczos diagonal interaction branch"
                );
            } else {
                assert!(
                    !preparation.data.exchange_terms.is_empty(),
                    "Lanczos exchange branch"
                );
                assert!(
                    !preparation.data.coulomb_inter_terms.is_empty()
                        && !preparation.data.hund_terms.is_empty(),
                    "Lanczos spin diagonal branch"
                );
            }
            let out = output_dir(&format!("lanczos-{lanczos_mode}"), model.name);
            let result = mvmc_core::vmc_phys_cal_to_dir(preparation, &out).unwrap();
            assert_fixed_values(&result.data, &expected_values);

            let expected_ls_names = fs::read_dir(fixture.join("expected"))
                .unwrap()
                .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
                .filter(|name| {
                    name.starts_with("zvo_ls_")
                        && (lanczos_mode == 2
                            || name.starts_with("zvo_ls_out_")
                            || name.starts_with("zvo_ls_qqqq_"))
                })
                .chain((lanczos_mode == 2).then_some("zvo_ls_cisajscktaltex_001.dat".to_owned()))
                .collect::<BTreeSet<_>>();
            let actual_ls_names = fs::read_dir(&out)
                .unwrap()
                .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
                .filter(|name| name.starts_with("zvo_ls_"))
                .collect::<BTreeSet<_>>();
            assert_eq!(
                actual_ls_names, expected_ls_names,
                "{model_name} Lanczos mode {lanczos_mode} file set"
            );

            for name in ["zvo_ls_out_001.dat", "zvo_ls_qqqq_001.dat"] {
                assert_reference(&out.join(name), &fixture.join("expected").join(name));
            }
            for name in [
                "zvo_ls_cisajs_001.dat",
                "zvo_ls_cisajscktalt_001.dat",
                "zvo_ls_cisajscktaltex_001.dat",
            ] {
                let expected_path = fixture.join("expected").join(name);
                let should_exist = lanczos_mode == 2;
                assert_eq!(
                    out.join(name).is_file(),
                    should_exist,
                    "{model_name} {name}"
                );
                if should_exist {
                    if expected_path.is_file() {
                        assert_reference(&out.join(name), &expected_path);
                    } else {
                        let namelist =
                            fs::read_to_string(fixture.join("inputs/namelist.def")).unwrap();
                        assert!(
                            !namelist
                                .lines()
                                .any(|line| line.split_whitespace().next() == Some("TwoBodyGEx")),
                            "missing nonempty independent {name} reference for {model_name}"
                        );
                        assert!(
                            fs::read_to_string(out.join(name))
                                .unwrap()
                                .trim()
                                .is_empty(),
                            "{model_name} {name} should be empty when the fixture has no terms"
                        );
                    }
                }
            }
            let _ = fs::remove_dir_all(out);
        }
    }
}

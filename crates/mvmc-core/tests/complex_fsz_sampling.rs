//! Exact original-source intermediate state for both complex FSZ proposal families.

use mvmc_core::{sampling::driver::vmc_make_sample_fsz, ExpertModeData, VmcOptimizationState};
use mvmc_expert_parsers::{GutzwillerTerm, JastrowTerm, LocSpinTerm};
use num_complex::Complex64;
use sfmt19937::Sfmt19937Rng;

fn integers(line: &str) -> Vec<i64> {
    line.split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect()
}

fn complex_bits(line: &str) -> Vec<Complex64> {
    let v: Vec<_> = line
        .split_whitespace()
        .map(|v| f64::from_bits(u64::from_str_radix(v, 16).unwrap()))
        .collect();
    v.as_chunks::<2>()
        .0
        .iter()
        .map(|z| Complex64::new(z[0], z[1]))
        .collect()
}

fn check_matrix_state(label: &str, state: &VmcOptimizationState, pf: &str, inv: &str) {
    let pf = complex_bits(pf);
    let inv = complex_bits(inv);
    let actual_inv: Vec<_> = (0..2)
        .flat_map(|qp| {
            state.slater_matrix.inv_m.as_slice()[qp * 17..qp * 17 + 16]
                .iter()
                .copied()
        })
        .collect();
    for (name, actual, expected) in [
        ("pf", &state.slater_matrix.pf_m, &pf),
        ("inverse", &actual_inv, &inv),
    ] {
        assert_eq!(actual.len(), expected.len());
        let actual: Vec<_> = actual
            .iter()
            .flat_map(|z| [z.re.to_bits(), z.im.to_bits()])
            .collect();
        let expected: Vec<_> = expected
            .iter()
            .flat_map(|z| [z.re.to_bits(), z.im.to_bits()])
            .collect();
        assert_eq!(actual, expected, "{label} {name}");
    }
}

#[test]
fn complex_fsz_proposals_burn_reuse_inverse_pfaffian_counters_and_rng_match_julia() {
    let mut lines = if cfg!(all(
        target_os = "linux",
        target_env = "gnu",
        target_arch = "x86_64"
    )) {
        include_str!("../../../tests/fixtures/linux_gnu_julia/complex_fsz/sampling.txt")
    } else {
        include_str!("../../../tests/fixtures/complex_fsz/sampling.txt")
    }
    .lines()
    .filter(|l| !l.starts_with('#'));
    for (name, local, two_sz, path) in [
        ("conduction", false, -1, 0),
        ("fixed_sz", false, 0, 0),
        ("local", true, -1, 2),
        ("cancelled", false, -1, 0),
        ("failure", false, -1, 0),
    ] {
        for &calls in if name == "failure" {
            &[1][..]
        } else {
            &[1, 2, 3, 10][..]
        } {
            assert_eq!(lines.next().unwrap(), format!("{name} {calls}"));
            let mut data = ExpertModeData::new();
            data.modpara.nsite = 4;
            data.modpara.nelec = 2;
            data.modpara.nmp_trans = 2;
            data.modpara.nvmc_warmup = 0;
            data.modpara.nvmc_sample = 1;
            data.modpara.nvmc_interval = 1;
            data.modpara.two_sz = two_sz;
            data.modpara.nex_update_path = path;
            data.i_flg_orbital_general = 1;
            data.complex_flags = vec![1];
            data.locspin_terms = (0..4)
                .map(|site| LocSpinTerm {
                    site,
                    spin_value: i64::from(local),
                })
                .collect();
            data.n_qp_trans = 2;
            data.para_qp_trans = vec![Complex64::new(1.0, 0.0), Complex64::new(-0.375, 0.0)];
            if name == "cancelled" {
                data.para_qp_trans[1] = Complex64::new(-1.0, 0.0);
            }
            data.n_gutzwiller_idx = 1;
            data.gutzwiller_idx = vec![0; 4];
            data.gutzwiller_terms = vec![GutzwillerTerm {
                site: 0,
                value: Complex64::new(0.125, 0.0),
                is_complex: false,
            }];
            data.n_jastrow_idx = 1;
            data.jastrow_idx = (0..4)
                .map(|i| (0..4).map(|j| if i == j { -1 } else { 0 }).collect())
                .collect();
            data.jastrow_terms = vec![JastrowTerm {
                site1: 0,
                site2: 1,
                value: Complex64::new(-0.2, 0.0),
                is_complex: false,
            }];
            mvmc_expert_parsers::utils::qp_weight::init_qp_weight(&mut data);
            let mut state = VmcOptimizationState::zeros(4, 2, 2, 0, 2, 1, true, true);
            for qp in 0..2 {
                for i in 0..8 {
                    for j in i + 1..8 {
                        let plane = if name == "cancelled" { 0 } else { qp };
                        // Match the source's integer subtraction before division.
                        let z = if name == "failure" {
                            Complex64::new(f64::NAN, f64::NAN)
                        } else {
                            Complex64::new(
                                (((17 * i + 13 * j + 7 * plane) % 31) as i64 - 15) as f64 / 7.0
                                    + 0.125,
                                (((11 * i + 3 * j + plane) % 19) as i64 - 9) as f64 / 13.0,
                            )
                        };
                        state.slater_matrix.slater_elm.set(qp, i, j, z);
                        state.slater_matrix.slater_elm.set(qp, j, i, -z);
                    }
                }
            }
            if calls == 1 {
                let mut probe = VmcOptimizationState::zeros(4, 2, 2, 0, 2, 1, true, true);
                probe.slater_matrix.slater_elm = state.slater_matrix.slater_elm.clone();
                let mut probe_rng = Sfmt19937Rng::new(11272);
                let result = mvmc_core::sampling::initial::make_initial_sample_fsz(
                    &data,
                    &mut probe,
                    &mut probe_rng,
                    0,
                    2,
                    &mvmc_core::state::ThreadedPfaPackWorkspace::new(4, 1),
                );
                assert_eq!(result.is_err(), name == "failure");
                let c = &probe.electron_config;
                for values in [
                    &c.tmp_ele_idx,
                    &c.tmp_ele_cfg,
                    &c.tmp_ele_num,
                    &c.tmp_ele_proj_cnt,
                    &c.tmp_ele_spn,
                ] {
                    assert_eq!(values, &integers(lines.next().unwrap()), "{name} initial");
                }
                let initial_pf = lines.next().unwrap();
                let initial_inverse = lines.next().unwrap();
                let words: Vec<u32> = lines
                    .next()
                    .unwrap()
                    .split_whitespace()
                    .map(|v| v.parse().unwrap())
                    .collect();
                assert_eq!(
                    (0..624).map(|_| probe_rng.gen_rand32()).collect::<Vec<_>>(),
                    words,
                    "{name} initial RNG"
                );
                check_matrix_state(
                    &format!("{name} initial"),
                    &probe,
                    initial_pf,
                    initial_inverse,
                );
            }
            let mut rng = Sfmt19937Rng::new(11272);
            for _ in 0..calls {
                assert_eq!(
                    vmc_make_sample_fsz(&data, &mut state, &mut rng).saved,
                    usize::from(name != "failure")
                );
            }
            let c = &state.electron_config;
            for values in [
                &c.ele_idx,
                &c.ele_cfg,
                &c.ele_num,
                &c.ele_proj_cnt,
                &c.ele_spn,
                &c.tmp_ele_idx,
                &c.tmp_ele_cfg,
                &c.tmp_ele_num,
                &c.tmp_ele_proj_cnt,
                &c.tmp_ele_spn,
                &c.burn_ele_idx,
            ] {
                assert_eq!(
                    values,
                    &integers(lines.next().unwrap()),
                    "{name} calls={calls}"
                );
            }
            assert_eq!(
                c.counter.as_slice(),
                integers(lines.next().unwrap()),
                "{name} calls={calls} counters"
            );
            let expected_pf = lines.next().unwrap();
            let expected_inverse = lines.next().unwrap();
            let words: Vec<u32> = lines
                .next()
                .unwrap()
                .split_whitespace()
                .map(|v| v.parse().unwrap())
                .collect();
            assert_eq!(words.len(), 624);
            assert_eq!(
                (0..624).map(|_| rng.gen_rand32()).collect::<Vec<_>>(),
                words,
                "{name} calls={calls} RNG"
            );
            check_matrix_state(
                &format!("{name} calls={calls}"),
                &state,
                expected_pf,
                expected_inverse,
            );
        }
    }
    assert!(lines.next().is_none());
}

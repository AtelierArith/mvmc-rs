//! General operator ratios against original Julia kernels and analytic Fock tests.
use mvmc_core::observables::{
    calculate_local_energy, calculate_local_energy_fsz, green_func1_fsz, green_func2,
    green_func2_fsz,
};
use mvmc_core::{ExpertModeData, VmcOptimizationState};
use mvmc_expert_parsers::utils::qp_weight::init_qp_weight;
use mvmc_expert_parsers::{
    CoulombInterTerm, CoulombIntraTerm, ExchangeTerm, GutzwillerTerm, HundTerm, JastrowTerm,
};
use num_complex::Complex64;

fn integers(line: &str) -> Vec<i64> {
    line.split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect()
}

fn complex_bits(line: &str) -> Vec<Complex64> {
    let bits: Vec<_> = line
        .split_whitespace()
        .map(|v| f64::from_bits(u64::from_str_radix(v, 16).unwrap()))
        .collect();
    bits.chunks_exact(2)
        .map(|v| Complex64::new(v[0], v[1]))
        .collect()
}

fn green_data(complex: bool) -> ExpertModeData {
    let mut data = ExpertModeData::new();
    data.modpara.nsite = 4;
    data.modpara.nelec = 2;
    data.modpara.nmp_trans = 2;
    data.complex_flags = vec![i64::from(complex)];
    data.n_qp_trans = 2;
    data.para_qp_trans = vec![Complex64::new(1.0, 0.0), Complex64::new(-0.375, 0.0)];
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
    init_qp_weight(&mut data);
    data
}

#[test]
fn exhaustive_four_site_two_body_ratios_match_original_julia() {
    let input = include_str!("../../../tests/fixtures/interall/green_normal.txt");
    let mut lines = input.lines().filter(|l| !l.starts_with('#'));
    for _ in 0..4 {
        let complex = lines.next().unwrap() == "1";
        let idx = integers(lines.next().unwrap());
        let cfg = integers(lines.next().unwrap());
        let num = integers(lines.next().unwrap());
        let cnt = integers(lines.next().unwrap());
        let mut data = green_data(complex);
        let mut state = VmcOptimizationState::zeros(4, 2, 2, 0, 2, 1, complex, false);
        let slater = complex_bits(lines.next().unwrap());
        state
            .slater_matrix
            .slater_elm
            .as_mut_slice()
            .copy_from_slice(&slater);
        let pf = complex_bits(lines.next().unwrap());
        state.slater_matrix.pf_m.copy_from_slice(&pf);
        let inverse = complex_bits(lines.next().unwrap());
        for qp in 0..2 {
            state.slater_matrix.inv_m.as_mut_slice()[qp * 17..qp * 17 + 16]
                .copy_from_slice(&inverse[qp * 16..qp * 16 + 16]);
        }
        if !complex {
            for (a, b) in state
                .slater_matrix
                .slater_elm_real
                .as_mut_slice()
                .iter_mut()
                .zip(&slater)
            {
                *a = b.re;
            }
            for (a, b) in state.slater_matrix.pf_m_real.iter_mut().zip(&pf) {
                *a = b.re;
            }
            for qp in 0..2 {
                for i in 0..16 {
                    state.slater_matrix.inv_m_real.as_mut_slice()[qp * 17 + i] =
                        inverse[qp * 16 + i].re;
                }
            }
        }
        let ip = complex_bits(lines.next().unwrap())[0];
        let expected_energy = complex_bits(lines.next().unwrap())[0];
        let before = state.slater_matrix.inv_m.as_slice().to_vec();
        let before_real = state.slater_matrix.inv_m_real.as_slice().to_vec();
        let mut greens = Vec::with_capacity(1024);
        for _ in 0..1024 {
            let row: Vec<_> = lines.next().unwrap().split_whitespace().collect();
            let ops: Vec<usize> = row[..6].iter().map(|v| v.parse().unwrap()).collect();
            let expected = complex_bits(&row[6..].join(" "))[0];
            let actual = green_func2(
                ops[0],
                ops[1],
                ops[2],
                ops[3],
                ops[4] as u8,
                ops[5] as u8,
                ip,
                &data,
                &mut state,
                &idx,
                &cfg,
                &num,
                &cnt,
            );
            greens.push(actual);
            // Exact source arithmetic, including signs of zero.
            for (a, e) in [(actual.re, expected.re), (actual.im, expected.im)] {
                assert_eq!(a.to_bits(),e.to_bits(),"complex={complex}, idx={idx:?}, ops={ops:?}, actual={actual}, expected={expected}");
            }
        }
        assert_eq!(state.slater_matrix.inv_m.as_slice(), before);
        assert_eq!(state.slater_matrix.inv_m_real.as_slice(), before_real);
        assert_eq!(state.slater_matrix.pf_m, pf);
        let g = |i: usize, j: usize, k: usize, l: usize, s: usize, t: usize| {
            greens[(s * 2 + t) * 256 + ((i * 4 + j) * 4 + k) * 4 + l]
        };
        let mut equivalent = 0.7 * g(0, 0, 0, 0, 0, 1);
        for s in 0..2 {
            for t in 0..2 {
                equivalent -= 0.3 * g(0, 0, 3, 3, s, t);
            }
        }
        equivalent -= 0.125 * (g(0, 0, 3, 3, 0, 0) + g(0, 0, 3, 3, 1, 1));
        equivalent += 0.2 * (g(0, 1, 1, 0, 0, 1) + g(0, 1, 1, 0, 1, 0));
        equivalent += 0.15 * (g(0, 0, 0, 0, 0, 1) + g(0, 0, 0, 0, 1, 0));
        assert!((equivalent - expected_energy).norm() < 2e-14);
        data.coulomb_intra_terms = vec![CoulombIntraTerm {
            site: 0,
            value: 0.7,
        }];
        data.coulomb_inter_terms = vec![CoulombInterTerm {
            site1: 0,
            site2: 3,
            value: -0.3,
        }];
        data.hund_terms = vec![HundTerm {
            site1: 0,
            site2: 3,
            value: 0.125,
        }];
        data.exchange_terms = vec![
            ExchangeTerm {
                site1: 0,
                site2: 1,
                value: 0.2,
            },
            ExchangeTerm {
                site1: 0,
                site2: 0,
                value: 0.15,
            },
        ];
        let energy = calculate_local_energy(ip, &data, &mut state, &idx, &cfg, &num, &cnt);
        assert_eq!(energy.re.to_bits(), expected_energy.re.to_bits());
        assert_eq!(energy.im.to_bits(), expected_energy.im.to_bits());
    }
    assert!(lines.next().is_none());
}

#[test]
fn exhaustive_fsz_one_and_two_body_spin_changes_match_original_julia_bits() {
    let input = include_str!("../../../tests/fixtures/interall/green_fsz.txt");
    let mut lines = input.lines().filter(|l| !l.starts_with('#'));
    for case in 0..6 {
        let complex = lines.next().unwrap() == "1";
        let idx = integers(lines.next().unwrap());
        let spins = integers(lines.next().unwrap());
        let cfg = integers(lines.next().unwrap());
        let num = integers(lines.next().unwrap());
        let cnt = integers(lines.next().unwrap());
        let mut data = green_data(complex);
        data.i_flg_orbital_general = 1;
        let mut state = VmcOptimizationState::zeros(4, 2, 2, 0, 2, 1, complex, true);
        let slater = complex_bits(lines.next().unwrap());
        state
            .slater_matrix
            .slater_elm
            .as_mut_slice()
            .copy_from_slice(&slater);
        let pf = complex_bits(lines.next().unwrap());
        state.slater_matrix.pf_m.copy_from_slice(&pf);
        let inv = complex_bits(lines.next().unwrap());
        for qp in 0..2 {
            state.slater_matrix.inv_m.as_mut_slice()[qp * 17..qp * 17 + 16]
                .copy_from_slice(&inv[qp * 16..qp * 16 + 16]);
        }
        let ip = complex_bits(lines.next().unwrap())[0];
        let before = state.slater_matrix.inv_m.as_slice().to_vec();
        let expected_one = complex_bits(lines.next().unwrap());
        let expected_two = complex_bits(lines.next().unwrap());
        assert_eq!(expected_one.len(), 64);
        assert_eq!(expected_two.len(), 4096);
        let mut index = 0;
        for s in 0..2 {
            for t in 0..2 {
                for ri in 0..4 {
                    for rj in 0..4 {
                        let actual = green_func1_fsz(
                            ri, rj, s, t, ip, &data, &mut state, &idx, &cfg, &num, &cnt, &spins,
                        );
                        let expected = expected_one[index];
                        index += 1;
                        assert_eq!(
                            [actual.re.to_bits(), actual.im.to_bits()],
                            [expected.re.to_bits(), expected.im.to_bits()],
                            "case={case}, one={:?}",
                            [ri, rj, s as usize, t as usize]
                        );
                    }
                }
            }
        }
        let mut index = 0;
        for s in 0..2 {
            for t in 0..2 {
                for u in 0..2 {
                    for v in 0..2 {
                        for ri in 0..4 {
                            for rj in 0..4 {
                                for rk in 0..4 {
                                    for rl in 0..4 {
                                        let actual = green_func2_fsz(
                                            ri, rj, rk, rl, s, t, u, v, ip, &data, &mut state,
                                            &idx, &cfg, &num, &cnt, &spins,
                                        );
                                        let expected = expected_two[index];
                                        index += 1;
                                        assert_eq!([actual.re.to_bits(),actual.im.to_bits()],[expected.re.to_bits(),expected.im.to_bits()],"case={case}, two={:?}, actual={actual}, expected={expected}", [ri,rj,rk,rl,s as usize,t as usize,u as usize,v as usize]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(state.slater_matrix.inv_m.as_slice(), before);
        assert_eq!(state.slater_matrix.pf_m, pf);
        assert_eq!(state.slater_matrix.slater_elm.as_slice(), slater);
        let expected = complex_bits(lines.next().unwrap());
        data.coulomb_intra_terms = vec![CoulombIntraTerm {
            site: 0,
            value: 0.7,
        }];
        data.coulomb_inter_terms = vec![CoulombInterTerm {
            site1: 0,
            site2: 3,
            value: -0.3,
        }];
        data.hund_terms = vec![HundTerm {
            site1: 0,
            site2: 3,
            value: 0.125,
        }];
        data.exchange_terms = vec![
            ExchangeTerm {
                site1: 0,
                site2: 1,
                value: 0.2,
            },
            ExchangeTerm {
                site1: 0,
                site2: 0,
                value: 0.15,
            },
        ];
        let energy =
            calculate_local_energy_fsz(ip, &data, &mut state, &idx, &cfg, &num, &cnt, &spins);
        assert_eq!(
            [energy.re.to_bits(), energy.im.to_bits()],
            [expected[0].re.to_bits(), expected[0].im.to_bits()]
        );
        data.inter_all_terms = mvmc_expert_parsers::parsers::interall::parse_interall_content(
            "0 0 0 0 3 1 3 1 -0.5 0.125\n0 0 0 1 3 1 3 0 -0.375 0.1875\n0 0 0 1 2 0 2 1 0.125 -0.25\n1 1 2 0 2 0 0 1 -0.25 -0.375\n1 1 0 0 2 0 3 1 0.5 0.125\n0 0 0 1 3 1 3 0 -0.375 0.1875\n-1 0 0 1 3 1 3 0 0.25 0.125\n"
        );
        let energy =
            calculate_local_energy_fsz(ip, &data, &mut state, &idx, &cfg, &num, &cnt, &spins);
        assert_eq!(
            [energy.re.to_bits(), energy.im.to_bits()],
            [expected[1].re.to_bits(), expected[1].im.to_bits()],
            "case={case} InterAll energy"
        );
    }
    assert!(lines.next().is_none());
}

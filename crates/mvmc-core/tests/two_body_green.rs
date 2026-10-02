//! General operator ratios against original Julia kernels and analytic Fock tests.
use mvmc_core::observables::{
    calculate_local_energy, calculate_local_energy_fsz, green_func1_fsz, green_func2,
    green_func2_fsz,
};
use mvmc_core::{ExpertModeData, VmcOptimizationState};
use mvmc_expert_parsers::utils::qp_weight::init_qp_weight;
use mvmc_expert_parsers::{
    CoulombInterTerm, CoulombIntraTerm, DoublonHolon2SiteIndex, ExchangeTerm, GutzwillerTerm,
    HundTerm, InterAllTerm, JastrowTerm, PairHopTerm,
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

fn add_dh2_green_model(data: &mut ExpertModeData) {
    data.doublon_holon_2site_indices = vec![
        DoublonHolon2SiteIndex {
            neighbors: vec![[1, 2], [0, 2], [0, 1], [0, 1]],
        },
        DoublonHolon2SiteIndex {
            neighbors: vec![[0, 0], [0, 0], [3, 3], [2, 2]],
        },
    ];
    data.doublon_holon_2site_params = (1..=12)
        .map(|i| Complex64::new(i as f64 / 128.0, -(i as f64) / 256.0))
        .collect();
}

fn check_pairhop_energy(
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    ip: Complex64,
    configuration: [&[i64]; 4],
    spins: Option<&[i64]>,
    expected: &[Complex64],
) {
    let [idx, cfg, num, cnt] = configuration;
    let local_energy = |data: &ExpertModeData, state: &mut VmcOptimizationState| {
        if let Some(spins) = spins {
            calculate_local_energy_fsz(ip, data, state, idx, cfg, num, cnt, spins)
        } else {
            calculate_local_energy(ip, data, state, idx, cfg, num, cnt)
        }
    };
    let input = include_str!("../../../tests/fixtures/pairhop/hamiltonian.def");
    let section = mvmc_expert_parsers::parsers::pairhop::parse_pairhop_content(input);
    assert!(section.is_success());
    let mut combined = data.clone();
    combined.pair_hop_terms = section.terms;
    combined.pair_hop_terms.push(PairHopTerm {
        site1: -1,
        site2: 0,
        value: 0.125,
    });
    let mut pure = combined.clone();
    pure.coulomb_intra_terms.clear();
    pure.coulomb_inter_terms.clear();
    pure.hund_terms.clear();
    pure.exchange_terms.clear();
    for (label, input, expected) in [
        ("pure", &pure, expected[0]),
        ("combined", &combined, expected[1]),
    ] {
        let actual = local_energy(input, state);
        assert_eq!(
            [actual.re.to_bits(), actual.im.to_bits()],
            [expected.re.to_bits(), expected.im.to_bits()],
            "PairHop {label}, spins={spins:?}, idx={idx:?}"
        );
    }
    let mut equivalent = data.clone();
    equivalent.inter_all_terms = combined
        .pair_hop_terms
        .iter()
        .filter(|t| (0..4).contains(&t.site1) && (0..4).contains(&t.site2))
        .map(|t| InterAllTerm {
            site0: t.site1,
            spin0: 0,
            site1: t.site2,
            spin1: 0,
            site2: t.site1,
            spin2: 1,
            site3: t.site2,
            spin3: 1,
            value: Complex64::new(t.value, 0.0),
            is_complex: false,
        })
        .collect();
    let actual = if spins.is_some() {
        local_energy(&equivalent, state)
    } else {
        let mut energy = local_energy(data, state);
        for t in &equivalent.inter_all_terms {
            energy += t.value
                * green_func2(
                    t.site0 as usize,
                    t.site1 as usize,
                    t.site2 as usize,
                    t.site3 as usize,
                    0,
                    1,
                    ip,
                    data,
                    state,
                    idx,
                    cfg,
                    num,
                    cnt,
                );
        }
        energy
    };
    assert_eq!(
        [actual.re.to_bits(), actual.im.to_bits()],
        [expected[2].re.to_bits(), expected[2].im.to_bits()]
    );
    assert!((actual - expected[1]).norm() < 2e-14);
}

#[test]
fn exhaustive_four_site_two_body_ratios_match_original_julia() {
    check_normal_green("");
}
#[test]
fn exhaustive_dh2_normal_two_body_ratios_match_original_julia() {
    check_normal_green("dh2");
}
fn check_normal_green(factor: &str) {
    let input = green_fixture(factor, "green_normal.txt");
    let mut lines = input.lines().filter(|l| !l.starts_with('#'));
    let mut pairhop = include_str!("../../../tests/fixtures/pairhop/energy_normal.txt")
        .lines()
        .filter(|l| !l.starts_with('#'));
    for _ in 0..4 {
        let complex = lines.next().unwrap() == "1";
        let idx = integers(lines.next().unwrap());
        let cfg = integers(lines.next().unwrap());
        let num = integers(lines.next().unwrap());
        let cnt = integers(lines.next().unwrap());
        let mut data = green_data(complex);
        if matches!(factor, "dh2" | "dh24") {
            add_dh2_green_model(&mut data);
        }
        if matches!(factor, "dh4" | "dh24") {
            add_dh4_green_model(&mut data);
        }
        if factor == "rbm" {
            add_rbm_green_model(&mut data);
        }
        let mut state = VmcOptimizationState::zeros(
            4,
            2,
            data.projection_layout().n_proj,
            0,
            2,
            1,
            complex,
            false,
        );
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
        if factor.is_empty() {
            check_pairhop_energy(
                &data,
                &mut state,
                ip,
                [&idx, &cfg, &num, &cnt],
                None,
                &complex_bits(pairhop.next().unwrap()),
            );
        }
        assert_eq!(state.slater_matrix.inv_m.as_slice(), before);
        assert_eq!(state.slater_matrix.inv_m_real.as_slice(), before_real);
    }
    assert!(lines.next().is_none());
    if factor.is_empty() {
        assert!(pairhop.next().is_none());
    }
}

#[test]
fn exhaustive_fsz_one_and_two_body_spin_changes_match_original_julia_bits() {
    check_fsz_green("");
}
#[test]
fn exhaustive_dh2_fsz_one_and_two_body_spin_changes_match_original_julia_bits() {
    check_fsz_green("dh2");
}
fn check_fsz_green(factor: &str) {
    let input = green_fixture(factor, "green_fsz.txt");
    let mut lines = input.lines().filter(|l| !l.starts_with('#'));
    let mut pairhop = include_str!("../../../tests/fixtures/pairhop/energy_fsz.txt")
        .lines()
        .filter(|l| !l.starts_with('#'));
    for case in 0..6 {
        let complex = lines.next().unwrap() == "1";
        let idx = integers(lines.next().unwrap());
        let spins = integers(lines.next().unwrap());
        let cfg = integers(lines.next().unwrap());
        let num = integers(lines.next().unwrap());
        let cnt = integers(lines.next().unwrap());
        let mut data = green_data(complex);
        if matches!(factor, "dh2" | "dh24") {
            add_dh2_green_model(&mut data);
        }
        if matches!(factor, "dh4" | "dh24") {
            add_dh4_green_model(&mut data);
        }
        if factor == "rbm" {
            add_rbm_green_model(&mut data);
        }
        data.i_flg_orbital_general = 1;
        let mut state = VmcOptimizationState::zeros(
            4,
            2,
            data.projection_layout().n_proj,
            0,
            2,
            1,
            complex,
            true,
        );
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
        if factor.is_empty() {
            check_pairhop_energy(
                &data,
                &mut state,
                ip,
                [&idx, &cfg, &num, &cnt],
                Some(&spins),
                &complex_bits(pairhop.next().unwrap()),
            );
        }
        assert_eq!(state.slater_matrix.inv_m.as_slice(), before);
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
    if factor.is_empty() {
        assert!(pairhop.next().is_none());
    }
}

fn green_fixture(factor: &str, name: &str) -> String {
    let directory = match factor {
        "" => "interall",
        "dh2" => "dh2",
        "dh4" => "dh4/kernels",
        "dh24" => "dh4/combined",
        "rbm" => "rbm/production",
        _ => panic!("unknown factor {factor}"),
    };
    std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures")
            .join(directory)
            .join(name),
    )
    .unwrap()
}
fn add_dh4_green_model(data: &mut ExpertModeData) {
    data.doublon_holon_4site_indices = vec![
        mvmc_expert_parsers::DoublonHolon4SiteIndex {
            neighbors: vec![[1, 2, 3, 0], [0, 2, 3, 1], [0, 1, 3, 2], [0, 1, 2, 3]],
        },
        mvmc_expert_parsers::DoublonHolon4SiteIndex {
            neighbors: vec![[1, 1, 1, 1], [1, 1, 3, 3], [3, 3, 0, 0], [2, 2, 2, 2]],
        },
    ];
    data.doublon_holon_4site_params = (1..=20)
        .map(|i| Complex64::new(i as f64 / 128.0, -(i as f64) / 256.0))
        .collect();
}
#[test]
fn exhaustive_dh4_normal_two_body_ratios_match_original_julia() {
    check_normal_green("dh4");
}
#[test]
fn exhaustive_dh24_normal_two_body_ratios_match_original_julia() {
    check_normal_green("dh24");
}
#[test]
fn exhaustive_dh4_fsz_one_and_two_body_spin_changes_match_original_julia_bits() {
    check_fsz_green("dh4");
}
#[test]
fn exhaustive_dh24_fsz_one_and_two_body_spin_changes_match_original_julia_bits() {
    check_fsz_green("dh24");
}

fn add_rbm_green_model(data: &mut ExpertModeData) {
    let file = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/rbm/production/namelist_all.def");
    let mut parsed = mvmc_expert_parsers::parse_expert_mode_files(&file).unwrap();
    mvmc_expert_parsers::utils::read_input_parameters::read_input_parameters(&mut parsed, &file)
        .unwrap();
    data.rbm_section_widths = parsed.rbm_section_widths;
    data.rbm_params = parsed.rbm_params.clone();
    data.charge_rbm_phys_layer_terms = parsed.charge_rbm_phys_layer_terms;
    data.spin_rbm_phys_layer_terms = parsed.spin_rbm_phys_layer_terms;
    data.general_rbm_phys_layer_terms = parsed.general_rbm_phys_layer_terms;
    data.charge_rbm_hidden_layer_terms = parsed.charge_rbm_hidden_layer_terms;
    data.spin_rbm_hidden_layer_terms = parsed.spin_rbm_hidden_layer_terms;
    data.general_rbm_hidden_layer_terms = parsed.general_rbm_hidden_layer_terms;
    data.charge_rbm_phys_hidden_terms = parsed.charge_rbm_phys_hidden_terms;
    data.spin_rbm_phys_hidden_terms = parsed.spin_rbm_phys_hidden_terms;
    data.general_rbm_phys_hidden_terms = parsed.general_rbm_phys_hidden_terms;
    data.modpara.nneuron_charge = 2;
    data.modpara.nneuron_spin = 3;
    data.modpara.nneuron_general = 4;
}
#[test]
fn exhaustive_rbm_normal_two_body_ratios_match_original_julia_bits() {
    check_normal_green("rbm");
}
#[test]
fn exhaustive_rbm_fsz_one_and_two_body_spin_changes_match_original_julia_bits() {
    check_fsz_green("rbm");
}

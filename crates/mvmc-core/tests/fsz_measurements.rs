//! Defined serial C FSZ measurements, including real-mode complex shadows.
use mvmc_core::observables::{calculate_green_func_fsz, weight_average_green_func_fsz};
use mvmc_core::state::PhysicalQuantities;
use mvmc_core::{ExpertModeData, VmcOptimizationState};
use mvmc_expert_parsers::utils::qp_weight::init_qp_weight;
use mvmc_expert_parsers::{GreenOneTerm, GreenTwoTerm, GutzwillerTerm, JastrowTerm, Spin};
use num_complex::Complex64;

fn integers(line: &str) -> Vec<i64> {
    line.split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect()
}

fn scalars(line: &str) -> Vec<f64> {
    line.split_whitespace()
        .map(|v| f64::from_bits(u64::from_str_radix(v, 16).unwrap()))
        .collect()
}

fn complexes(line: &str) -> Vec<Complex64> {
    scalars(line)
        .as_chunks::<2>()
        .0
        .iter()
        .map(|v| Complex64::new(v[0], v[1]))
        .collect()
}

fn bits(values: &[Complex64]) -> Vec<[u64; 2]> {
    values
        .iter()
        .map(|v| [v.re.to_bits(), v.im.to_bits()])
        .collect()
}

#[test]
fn fsz_measurements_match_native_c_locals_and_ordered_weighted_accumulators() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    let fixture = include_str!("../../../tests/fixtures/interall/c_fsz_measurements_linux_gnu.txt");
    #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
    let fixture = include_str!("../../../tests/fixtures/interall/c_fsz_measurements.txt");
    let mut lines = fixture.lines().filter(|line| !line.starts_with('#'));
    let mut differences = Vec::new();
    for case in 0..12 {
        let complex = lines.next().unwrap() == "1";
        let idx = integers(lines.next().unwrap());
        let spins = integers(lines.next().unwrap());
        let cfg = integers(lines.next().unwrap());
        let num = integers(lines.next().unwrap());
        let cnt = integers(lines.next().unwrap());
        let parameters = complexes(lines.next().unwrap());
        let mut data = ExpertModeData::new();
        data.modpara.nsite = 4;
        data.modpara.nelec = 2;
        data.modpara.nmp_trans = 2;
        data.i_flg_orbital_general = 1;
        data.complex_flags = vec![i64::from(complex)];
        data.n_qp_trans = 2;
        data.para_qp_trans = vec![Complex64::new(1.0, 0.0), Complex64::new(-0.375, 0.0)];
        data.n_gutzwiller_idx = 1;
        data.gutzwiller_idx = vec![0; 4];
        data.gutzwiller_terms = vec![GutzwillerTerm {
            site: 0,
            value: parameters[0],
            is_complex: false,
        }];
        data.n_jastrow_idx = 1;
        data.jastrow_idx = (0..4)
            .map(|i| (0..4).map(|j| if i == j { -1 } else { 0 }).collect())
            .collect();
        data.jastrow_terms = vec![JastrowTerm {
            site1: 0,
            site2: 1,
            value: parameters[1],
            is_complex: false,
        }];
        init_qp_weight(&mut data);
        for s in [Spin::Up, Spin::Down] {
            for t in [Spin::Up, Spin::Down] {
                for i in 0..4 {
                    for j in 0..4 {
                        data.green_one_terms.push(GreenOneTerm {
                            site1: i,
                            spin1: s,
                            site2: j,
                            spin2: t,
                        });
                    }
                }
            }
        }
        for s in [Spin::Up, Spin::Down] {
            for t in [Spin::Up, Spin::Down] {
                for u in [Spin::Up, Spin::Down] {
                    for v in [Spin::Up, Spin::Down] {
                        for i in 0..4 {
                            for j in 0..4 {
                                for k in 0..4 {
                                    for l in 0..4 {
                                        data.green_two_terms.push(GreenTwoTerm {
                                            site1: i,
                                            spin1: s,
                                            site2: j,
                                            spin2: t,
                                            site3: k,
                                            spin3: u,
                                            site4: l,
                                            spin4: v,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        for original in [73, 3072] {
            data.green_two_terms.push(data.green_two_terms[original]);
        }
        data.green_two_ex_indices = vec![
            (0, 0),
            (19, 35),
            (35, 19),
            (18, 63),
            (63, 0),
            (19, 19),
            (27, 35),
            (0, 0),
        ];
        let mut state = VmcOptimizationState::zeros(4, 2, 2, 0, 2, 1, complex, true);
        state.phys_quantities = Some(PhysicalQuantities::zeros(64, 8, 4098));
        state
            .slater_matrix
            .slater_elm
            .as_mut_slice()
            .copy_from_slice(&complexes(lines.next().unwrap()));
        state
            .slater_matrix
            .pf_m
            .copy_from_slice(&complexes(lines.next().unwrap()));
        let inv = complexes(lines.next().unwrap());
        for qp in 0..2 {
            state.slater_matrix.inv_m.as_mut_slice()[qp * 17..qp * 17 + 16]
                .copy_from_slice(&inv[qp * 16..qp * 16 + 16]);
        }
        let ip = complexes(lines.next().unwrap())[0];
        let weights = scalars(lines.next().unwrap());
        assert_eq!(weights.len(), 3);
        let expected_one = complexes(lines.next().unwrap());
        let expected_direct = complexes(lines.next().unwrap());
        assert_eq!(expected_one.len(), 64);
        assert_eq!(expected_direct.len(), 4098);
        // C's calgrn_fsz consumes the complex shadow in both modes. Poison
        // independent scalar storage to detect accidental real-kernel dispatch.
        state
            .slater_matrix
            .slater_elm_real
            .as_mut_slice()
            .fill(13.0);
        state.slater_matrix.inv_m_real.as_mut_slice().fill(-7.0);
        state.slater_matrix.pf_m_real.fill(19.0);
        let before_matrix = state.slater_matrix.clone();
        let before_configuration = state.electron_config.clone();
        assert_eq!(mvmc_core::get_all_complex_flag(&data), complex);
        for (sample, weight) in weights.into_iter().enumerate() {
            state.energy.wc += Complex64::new(weight, 0.0);
            let expected_weighted_one = complexes(lines.next().unwrap());
            let expected_weighted_direct = complexes(lines.next().unwrap());
            let expected_factored = complexes(lines.next().unwrap());
            calculate_green_func_fsz(
                &data, &mut state, weight, ip, &idx, &cfg, &num, &cnt, &spins,
            );
            assert_eq!(state.slater_matrix, before_matrix);
            assert_eq!(state.electron_config, before_configuration);
            let phys = state.phys_quantities.as_ref().unwrap();
            for (name, actual, expected) in [
                ("local one", &phys.local_cis_ajs, &expected_one),
                (
                    "local direct",
                    &phys.local_cis_ajs_ckt_alt_dc,
                    &expected_direct,
                ),
                ("weighted one", &phys.phys_cis_ajs, &expected_weighted_one),
                (
                    "weighted direct",
                    &phys.phys_cis_ajs_ckt_alt_dc,
                    &expected_weighted_direct,
                ),
                ("factored", &phys.phys_cis_ajs_ckt_alt, &expected_factored),
            ] {
                assert_eq!(actual.len(), expected.len(), "{case} {sample} {name}");
                for (operator, (actual, expected)) in
                    bits(actual).into_iter().zip(bits(expected)).enumerate()
                {
                    if actual != expected {
                        differences.push(format!(
                            "case={case} sample={sample} {name}[{operator}]: Rust={actual:x?}, C={expected:x?}"
                        ));
                    }
                }
            }
        }
        weight_average_green_func_fsz(&mut state);
        let phys = state.phys_quantities.as_ref().unwrap();
        for (name, actual) in [
            ("normalized one", &phys.phys_cis_ajs),
            ("normalized direct", &phys.phys_cis_ajs_ckt_alt_dc),
            ("normalized factored", &phys.phys_cis_ajs_ckt_alt),
        ] {
            let expected = complexes(lines.next().unwrap());
            assert_eq!(actual.len(), expected.len(), "{case} {name}");
            for (operator, (actual, expected)) in
                bits(actual).into_iter().zip(bits(&expected)).enumerate()
            {
                if actual != expected {
                    differences.push(format!(
                        "case={case} {name}[{operator}]: Rust={actual:x?}, C={expected:x?}"
                    ));
                }
            }
        }
        assert_eq!(state.slater_matrix, before_matrix);
        assert_eq!(state.electron_config, before_configuration);
    }
    assert!(lines.next().is_none());
    assert!(
        differences.is_empty(),
        "{} C differences, first: {:?}",
        differences.len(),
        differences.first()
    );
}

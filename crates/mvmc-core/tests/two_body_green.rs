//! General operator ratios against original Julia kernels and analytic Fock tests.
mod historical_overlay_stage {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/support/historical_overlay_stage.rs"
    ));
}
#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;
use mvmc_core::observables::{
    calculate_local_energy, calculate_local_energy_fsz, green_func1_fsz_complex,
    green_func1_fsz_real, green_func2_complex, green_func2_fsz_complex, green_func2_fsz_real,
    green_func2_real,
};
use mvmc_core::{ExpertModeData, VmcOptimizationState};
use mvmc_expert_parsers::utils::qp_weight::init_qp_weight;
use mvmc_expert_parsers::{
    CoulombInterTerm, CoulombIntraTerm, ExchangeTerm, GutzwillerTerm, HundTerm, InterAllTerm,
    JastrowTerm, PairHopTerm, Spin, TransferTerm,
};
use num_complex::Complex64;

#[path = "../../../tests/support/historical_orbital_model.rs"]
mod historical_orbital_model;

fn integers(line: &str) -> Vec<i64> {
    line.split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect()
}

// Fixed four-electron kernels: short determinants, projection exponentials and
// QP quotients. This budget is for each component, including cancellation.
const GREEN_ROUNDOFF: f64 = 1e-13;
// Complete Hamiltonians also sum up to 4098 weighted operators sequentially.
const ENERGY_ROUNDOFF: f64 = 1e-12;
fn complex_within(actual: Complex64, expected: Complex64, bound: f64) -> bool {
    numerical_comparison::within(actual.re, expected.re, bound, bound)
        && numerical_comparison::within(actual.im, expected.im, bound, bound)
}
fn complex_bits(line: &str) -> Vec<Complex64> {
    let bits: Vec<_> = line
        .split_whitespace()
        .map(|v| f64::from_bits(u64::from_str_radix(v, 16).unwrap()))
        .collect();
    bits.as_chunks::<2>()
        .0
        .iter()
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
fn normal_complex_interall_matches_native_c_green_kernels_and_ordered_sums() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    let fixture = include_str!("../../../tests/fixtures/interall/c_complex_green_linux_gnu.txt");
    #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
    let fixture = include_str!("../../../tests/fixtures/interall/c_complex_green.txt");
    let mut lines = fixture.lines().filter(|line| !line.starts_with('#'));
    let mut failures = Vec::new();
    for case in 0..4 {
        assert_eq!(lines.next().unwrap(), "1");
        let idx = integers(lines.next().unwrap());
        let cfg = integers(lines.next().unwrap());
        let num = integers(lines.next().unwrap());
        let cnt = integers(lines.next().unwrap());
        let projection = complex_bits(lines.next().unwrap());
        let mut data = green_data(true);
        data.gutzwiller_terms[0].value = projection[0];
        data.jastrow_terms[0].value = projection[1];
        let mut state = VmcOptimizationState::zeros(4, 2, 2, 0, 2, 1, true, false);
        let slater = complex_bits(lines.next().unwrap());
        let pf = complex_bits(lines.next().unwrap());
        let inverse = complex_bits(lines.next().unwrap());
        state
            .slater_matrix
            .slater_elm
            .as_mut_slice()
            .copy_from_slice(&slater);
        state.slater_matrix.pf_m.copy_from_slice(&pf);
        for qp in 0..2 {
            state.slater_matrix.inv_m.as_mut_slice()[qp * 17..qp * 17 + 16]
                .copy_from_slice(&inverse[qp * 16..qp * 16 + 16]);
        }
        let ip = complex_bits(lines.next().unwrap())[0];
        let before_inverse = state.slater_matrix.inv_m.as_slice().to_vec();
        for operator in 0..1024 {
            let row: Vec<_> = lines.next().unwrap().split_whitespace().collect();
            let ops: Vec<usize> = row[..6].iter().map(|v| v.parse().unwrap()).collect();
            let value = complex_bits(&row[6..8].join(" "))[0];
            let expected = complex_bits(&row[8..10].join(" "))[0];
            let actual = green_func2_complex(
                ops[0],
                ops[1],
                ops[2],
                ops[3],
                ops[4] as u8,
                ops[5] as u8,
                ip,
                &data,
                &state,
                &idx,
                &cfg,
                &num,
                &cnt,
            );
            if !complex_within(actual, expected, GREEN_ROUNDOFF) {
                failures.push(format!(
                    "case {case} operator {operator} {ops:?}: Rust={:016x} {:016x}, C={:016x} {:016x}",
                    actual.re.to_bits(), actual.im.to_bits(), expected.re.to_bits(), expected.im.to_bits(),
                ));
            }
            data.inter_all_terms.push(InterAllTerm {
                site0: ops[0] as i64,
                spin0: ops[4] as i64,
                site1: ops[1] as i64,
                spin1: ops[4] as i64,
                site2: ops[2] as i64,
                spin2: ops[5] as i64,
                site3: ops[3] as i64,
                spin3: ops[5] as i64,
                value,
                is_complex: value.im != 0.0,
            });
        }
        let expected_energy = complex_bits(lines.next().unwrap())[0];
        let energy = calculate_local_energy(ip, &data, &mut state, &idx, &cfg, &num, &cnt);
        if !complex_within(energy, expected_energy, ENERGY_ROUNDOFF) {
            failures.push(format!(
                "case {case} ordered energy: Rust={energy:?}, C={expected_energy:?}"
            ));
        }
        let expected_pairhop = complex_bits(lines.next().unwrap())[0];
        data.inter_all_terms.clear();
        data.pair_hop_terms = [
            (0, 3, 0.7),
            (3, 0, -0.2),
            (0, 0, -0.125),
            (0, 1, 0.375),
            (1, 2, -0.35),
            (0, 3, 0.0625),
        ]
        .into_iter()
        .map(|(site1, site2, value)| PairHopTerm {
            site1,
            site2,
            value,
        })
        .collect();
        let pairhop = calculate_local_energy(ip, &data, &mut state, &idx, &cfg, &num, &cnt);
        if !complex_within(pairhop, expected_pairhop, ENERGY_ROUNDOFF) {
            failures.push(format!(
                "case {case} PairHop: Rust={pairhop:?}, C={expected_pairhop:?}"
            ));
        }
        assert_eq!(state.slater_matrix.inv_m.as_slice(), before_inverse);
        assert_eq!(state.slater_matrix.pf_m, pf);
    }
    assert!(lines.next().is_none());
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures
            .iter()
            .take(20)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn normal_real_interall_matches_native_c_green_kernels_and_ordered_sums() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    let fixture = include_str!("../../../tests/fixtures/interall/c_real_green_linux_gnu.txt");
    #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
    let fixture = include_str!("../../../tests/fixtures/interall/c_real_green.txt");
    let mut lines = fixture.lines().filter(|line| !line.starts_with('#'));
    for case in 0..4 {
        assert_eq!(lines.next().unwrap(), "0");
        let idx = integers(lines.next().unwrap());
        let cfg = integers(lines.next().unwrap());
        let num = integers(lines.next().unwrap());
        let cnt = integers(lines.next().unwrap());
        let projection = complex_bits(lines.next().unwrap())[0];
        let mut data = green_data(false);
        data.gutzwiller_terms[0].value.re = projection.re;
        data.jastrow_terms[0].value.re = projection.im;
        let mut state = VmcOptimizationState::zeros(4, 2, 2, 0, 2, 1, false, false);
        let slater = complex_bits(lines.next().unwrap());
        let pf = complex_bits(lines.next().unwrap());
        let inverse = complex_bits(lines.next().unwrap());
        for (dst, src) in state
            .slater_matrix
            .slater_elm_real
            .as_mut_slice()
            .iter_mut()
            .zip(slater)
        {
            *dst = src.re;
        }
        for (dst, src) in state.slater_matrix.pf_m_real.iter_mut().zip(&pf) {
            *dst = src.re;
        }
        state.slater_matrix.pf_m.copy_from_slice(&pf);
        for qp in 0..2 {
            for i in 0..16 {
                state.slater_matrix.inv_m_real.as_mut_slice()[qp * 17 + i] =
                    inverse[qp * 16 + i].re;
            }
        }
        let ip = complex_bits(lines.next().unwrap())[0];
        let before_inverse = state.slater_matrix.inv_m_real.as_slice().to_vec();
        for operator in 0..1024 {
            let row: Vec<_> = lines.next().unwrap().split_whitespace().collect();
            let ops: Vec<usize> = row[..6].iter().map(|v| v.parse().unwrap()).collect();
            let value = complex_bits(&row[6..8].join(" "))[0];
            let expected = u64::from_str_radix(row[8], 16).unwrap();
            let actual = green_func2_real(
                ops[0],
                ops[1],
                ops[2],
                ops[3],
                ops[4] as u8,
                ops[5] as u8,
                ip.re,
                &data,
                &state,
                &idx,
                &cfg,
                &num,
                &cnt,
            );
            numerical_comparison::assert_close(
                actual,
                f64::from_bits(expected),
                GREEN_ROUNDOFF,
                GREEN_ROUNDOFF,
                format!("C case {case}, operator {operator}: {ops:?}"),
            );
            data.inter_all_terms.push(InterAllTerm {
                site0: ops[0] as i64,
                spin0: ops[4] as i64,
                site1: ops[1] as i64,
                spin1: ops[4] as i64,
                site2: ops[2] as i64,
                spin2: ops[5] as i64,
                site3: ops[3] as i64,
                spin3: ops[5] as i64,
                value,
                is_complex: value.im != 0.0,
            });
        }
        let expected_energy = u64::from_str_radix(lines.next().unwrap(), 16).unwrap();
        let energy = calculate_local_energy(ip, &data, &mut state, &idx, &cfg, &num, &cnt);
        numerical_comparison::assert_close(
            energy.re,
            f64::from_bits(expected_energy),
            ENERGY_ROUNDOFF,
            ENERGY_ROUNDOFF,
            format!("C ordered energy case {case}"),
        );
        numerical_comparison::assert_close(
            energy.im,
            0.0,
            4.0 * f64::EPSILON,
            0.0,
            "real energy imaginary",
        );
        assert_eq!(state.slater_matrix.inv_m_real.as_slice(), before_inverse);
        assert_eq!(state.slater_matrix.pf_m, pf);
    }
    assert!(lines.next().is_none());
}

#[test]
fn native_fsz_green_rejects_rbm_before_density_reductions() {
    let mut data = green_data(true);
    add_rbm_green_model(&mut data);
    data.i_flg_orbital_general = 1;
    let state = VmcOptimizationState::zeros(4, 2, 2, 0, 2, 1, true, true);
    let idx = [0, 2, 1, 3];
    let cfg = [0, -1, 1, -1, -1, 2, -1, 3];
    let num = [1, 0, 1, 0, 0, 1, 0, 1];
    let cnt = [0, 0];
    let spins = [0, 0, 1, 1];
    let ip = Complex64::new(1.0, 0.0);
    // Even a density shortcut must not silently discard a requested RBM.
    for (real, two_body) in [(false, false), (false, true), (true, false), (true, true)] {
        let error = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if real {
                let value = if two_body {
                    green_func2_fsz_real(
                        0, 0, 0, 0, 0, 0, 0, 0, ip.re, &data, &state, &idx, &cfg, &num, &cnt,
                        &spins,
                    )
                } else {
                    green_func1_fsz_real(
                        0, 0, 0, 0, ip.re, &data, &state, &idx, &cfg, &num, &cnt, &spins,
                    )
                };
                Complex64::new(value, 0.0)
            } else if two_body {
                green_func2_fsz_complex(
                    0, 0, 0, 0, 0, 0, 0, 0, ip, &data, &state, &idx, &cfg, &num, &cnt, &spins,
                )
            } else {
                green_func1_fsz_complex(
                    0, 0, 0, 0, ip, &data, &state, &idx, &cfg, &num, &cnt, &spins,
                )
            }
        }))
        .unwrap_err();
        assert_eq!(
            error.downcast_ref::<&str>(),
            Some(&"native C FSZ Green kernel does not support RBM")
        );
    }
}

#[test]
fn exhaustive_fsz_green_kernels_match_native_c_values() {
    check_native_fsz_green::<false>();
}

#[test]
fn exhaustive_real_fsz_green_kernels_match_native_c_values() {
    check_native_fsz_green::<true>();
}

fn check_native_fsz_green<const REAL: bool>() {
    let fixture = if REAL {
        #[cfg(all(target_os = "linux", target_env = "gnu"))]
        let fixture =
            include_str!("../../../tests/fixtures/interall/c_fsz_real_green_linux_gnu.txt");
        #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
        let fixture = include_str!("../../../tests/fixtures/interall/c_fsz_real_green.txt");
        fixture
    } else {
        #[cfg(all(target_os = "linux", target_env = "gnu"))]
        let fixture = include_str!("../../../tests/fixtures/interall/c_fsz_green_linux_gnu.txt");
        #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
        let fixture = include_str!("../../../tests/fixtures/interall/c_fsz_green.txt");
        fixture
    };
    let mut lines = fixture.lines().filter(|line| !line.starts_with('#'));
    let mut mismatches = 0;
    let mut first = None;
    for case in 0..if REAL { 6 } else { 12 } {
        let complex = lines.next().unwrap() == "1";
        let idx = integers(lines.next().unwrap());
        let spins = integers(lines.next().unwrap());
        let cfg = integers(lines.next().unwrap());
        let num = integers(lines.next().unwrap());
        let cnt = integers(lines.next().unwrap());
        let parameters = complex_bits(lines.next().unwrap());
        let mut data = green_data(complex);
        data.gutzwiller_terms[0].value = parameters[0];
        data.jastrow_terms[0].value = parameters[1];
        data.i_flg_orbital_general = 1;
        let mut state = VmcOptimizationState::zeros(4, 2, 2, 0, 2, 1, complex, true);
        let slater = complex_bits(lines.next().unwrap());
        state
            .slater_matrix
            .slater_elm
            .as_mut_slice()
            .copy_from_slice(&slater);
        if REAL {
            for (target, source) in state
                .slater_matrix
                .slater_elm_real
                .as_mut_slice()
                .iter_mut()
                .zip(&slater)
            {
                *target = source.re;
            }
        }
        let pf = complex_bits(lines.next().unwrap());
        state.slater_matrix.pf_m.copy_from_slice(&pf);
        if REAL {
            for (target, source) in state.slater_matrix.pf_m_real.iter_mut().zip(&pf) {
                *target = source.re;
            }
        }
        let inv = complex_bits(lines.next().unwrap());
        for qp in 0..2 {
            state.slater_matrix.inv_m.as_mut_slice()[qp * 17..qp * 17 + 16]
                .copy_from_slice(&inv[qp * 16..qp * 16 + 16]);
            if REAL {
                for (target, source) in state.slater_matrix.inv_m_real.as_mut_slice()
                    [qp * 17..qp * 17 + 16]
                    .iter_mut()
                    .zip(&inv[qp * 16..qp * 16 + 16])
                {
                    *target = source.re;
                }
            }
        }
        let ip = complex_bits(lines.next().unwrap())[0];
        if REAL {
            state
                .slater_matrix
                .slater_elm
                .as_mut_slice()
                .fill(Complex64::new(7.0, -2.0));
            state.slater_matrix.pf_m.fill(Complex64::new(-11.0, 3.0));
            state
                .slater_matrix
                .inv_m
                .as_mut_slice()
                .fill(Complex64::new(13.0, -5.0));
        }
        let before = state.slater_matrix.inv_m.as_slice().to_vec();
        let before_pf = state.slater_matrix.pf_m.clone();
        let before_slater = state.slater_matrix.slater_elm.as_slice().to_vec();
        let before_real = state.slater_matrix.inv_m_real.as_slice().to_vec();
        let before_pf_real = state.slater_matrix.pf_m_real.clone();
        let before_slater_real = state.slater_matrix.slater_elm_real.as_slice().to_vec();
        let expected_one = complex_bits(lines.next().unwrap());
        let expected_two = complex_bits(lines.next().unwrap());
        assert_eq!(expected_one.len(), 64);
        assert_eq!(expected_two.len(), 4096);
        let mut check = |actual: Complex64, expected: Complex64, operator: Vec<usize>| {
            // Scalar C returns only a real component. Signed zeros are
            // numerically equivalent in both fixture families.
            let actual = if REAL {
                Complex64::new(actual.re, 0.0)
            } else {
                actual
            };
            let bound = if operator.is_empty() {
                ENERGY_ROUNDOFF
            } else {
                GREEN_ROUNDOFF
            };
            if !complex_within(actual, expected, bound) {
                mismatches += 1;
                first.get_or_insert_with(|| {
                    format!(
                    "case={case}, operator={operator:?}, actual={actual:.17e}, C={expected:.17e}"
                )
                });
            }
        };
        let mut index = 0;
        for s in 0..2 {
            for t in 0..2 {
                for ri in 0..4 {
                    for rj in 0..4 {
                        check(
                            if REAL {
                                Complex64::new(
                                    green_func1_fsz_real(
                                        ri, rj, s, t, ip.re, &data, &state, &idx, &cfg, &num, &cnt,
                                        &spins,
                                    ),
                                    0.0,
                                )
                            } else {
                                green_func1_fsz_complex(
                                    ri, rj, s, t, ip, &data, &state, &idx, &cfg, &num, &cnt, &spins,
                                )
                            },
                            expected_one[index],
                            vec![ri, rj, s as usize, t as usize],
                        );
                        index += 1;
                    }
                }
            }
        }
        let mut index = 0;
        let mut interall_energy = Complex64::new(0.0, 0.0);
        let mut duplicate_contributions = [Complex64::new(0.0, 0.0); 2];
        for s in 0..2 {
            for t in 0..2 {
                for u in 0..2 {
                    for v in 0..2 {
                        for ri in 0..4 {
                            for rj in 0..4 {
                                for rk in 0..4 {
                                    for rl in 0..4 {
                                        let actual = if REAL {
                                            Complex64::new(
                                                green_func2_fsz_real(
                                                    ri, rj, rk, rl, s, t, u, v, ip.re, &data,
                                                    &state, &idx, &cfg, &num, &cnt, &spins,
                                                ),
                                                0.0,
                                            )
                                        } else {
                                            green_func2_fsz_complex(
                                                ri, rj, rk, rl, s, t, u, v, ip, &data, &state,
                                                &idx, &cfg, &num, &cnt, &spins,
                                            )
                                        };
                                        check(
                                            actual,
                                            expected_two[index],
                                            vec![
                                                ri, rj, rk, rl, s as usize, t as usize, u as usize,
                                                v as usize,
                                            ],
                                        );
                                        let coefficient = Complex64::new(
                                            (index % 7) as f64 / 16.0 - 3.0 / 16.0,
                                            (index % 11) as f64 / 32.0 - 5.0 / 32.0,
                                        );
                                        let contribution = if REAL {
                                            Complex64::new(coefficient.re * actual.re, 0.0)
                                        } else {
                                            coefficient * actual
                                        };
                                        interall_energy += contribution;
                                        if index == 73 {
                                            duplicate_contributions[0] = contribution;
                                        }
                                        if index == 3072 {
                                            duplicate_contributions[1] = contribution;
                                        }
                                        index += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        for contribution in duplicate_contributions {
            interall_energy += contribution;
        }
        let expected_energy = complex_bits(lines.next().unwrap())[0];
        check(interall_energy, expected_energy, vec![4098]);
        assert_eq!(state.slater_matrix.inv_m.as_slice(), before);
        assert_eq!(state.slater_matrix.pf_m, before_pf);
        assert_eq!(state.slater_matrix.slater_elm.as_slice(), before_slater);
        assert_eq!(state.slater_matrix.inv_m_real.as_slice(), before_real);
        assert_eq!(state.slater_matrix.pf_m_real, before_pf_real);
        assert_eq!(
            state.slater_matrix.slater_elm_real.as_slice(),
            before_slater_real
        );
    }
    assert!(lines.next().is_none());
    assert_eq!(mismatches, 0, "first native C mismatch: {first:?}");
}

#[test]
fn complete_fsz_hamiltonians_match_native_c_values() {
    check_native_fsz_hamiltonian::<false>();
}

#[test]
fn complete_real_fsz_hamiltonians_match_native_c_values() {
    check_native_fsz_hamiltonian::<true>();
}

fn check_native_fsz_hamiltonian<const REAL: bool>() {
    let fixture = if REAL {
        #[cfg(all(target_os = "linux", target_env = "gnu"))]
        let fixture =
            include_str!("../../../tests/fixtures/interall/c_fsz_real_hamiltonian_linux_gnu.txt");
        #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
        let fixture = include_str!("../../../tests/fixtures/interall/c_fsz_real_hamiltonian.txt");
        fixture
    } else {
        #[cfg(all(target_os = "linux", target_env = "gnu"))]
        let fixture =
            include_str!("../../../tests/fixtures/interall/c_fsz_hamiltonian_linux_gnu.txt");
        #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
        let fixture = include_str!("../../../tests/fixtures/interall/c_fsz_hamiltonian.txt");
        fixture
    };
    let mut lines = fixture.lines().filter(|line| !line.starts_with('#'));
    let mut failures = Vec::new();
    for case in 0..if REAL { 6 } else { 12 } {
        let _input_complex = lines.next().unwrap() == "1";
        let idx = integers(lines.next().unwrap());
        let spins = integers(lines.next().unwrap());
        let cfg = integers(lines.next().unwrap());
        let num = integers(lines.next().unwrap());
        let cnt = integers(lines.next().unwrap());
        let parameters = complex_bits(lines.next().unwrap());
        let mut data = green_data(!REAL);
        data.i_flg_orbital_general = 1;
        data.gutzwiller_terms[0].value = parameters[0];
        data.jastrow_terms[0].value = parameters[1];
        let mut state = VmcOptimizationState::zeros(4, 2, 2, 0, 2, 1, !REAL, true);
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
        if REAL {
            for (target, source) in state
                .slater_matrix
                .slater_elm_real
                .as_mut_slice()
                .iter_mut()
                .zip(&slater)
            {
                *target = source.re;
            }
            for (target, source) in state.slater_matrix.pf_m_real.iter_mut().zip(&pf) {
                *target = source.re;
            }
            for qp in 0..2 {
                for (target, source) in state.slater_matrix.inv_m_real.as_mut_slice()
                    [qp * 17..qp * 17 + 16]
                    .iter_mut()
                    .zip(&inv[qp * 16..qp * 16 + 16])
                {
                    *target = source.re;
                }
            }
        }
        let ip = complex_bits(lines.next().unwrap())[0];
        let energies = complex_bits(lines.next().unwrap());
        assert_eq!(energies.len(), 6);
        // Header mode determines the storage family, independently of the
        // imaginary Hamiltonian couplings. Poison the unused family so this
        // gate also detects accidental reads of the other matrix buffers.
        if REAL {
            state
                .slater_matrix
                .slater_elm
                .as_mut_slice()
                .fill(Complex64::new(13.0, -7.0));
            state
                .slater_matrix
                .inv_m
                .as_mut_slice()
                .fill(Complex64::new(-11.0, 19.0));
            state.slater_matrix.pf_m.fill(Complex64::new(17.0, -5.0));
        } else {
            state
                .slater_matrix
                .slater_elm_real
                .as_mut_slice()
                .fill(13.0);
            state.slater_matrix.inv_m_real.as_mut_slice().fill(-11.0);
            state.slater_matrix.pf_m_real.fill(17.0);
        }
        let before = state.slater_matrix.clone();
        for (group, expected) in energies.into_iter().enumerate() {
            data.coulomb_intra_terms.clear();
            data.coulomb_inter_terms.clear();
            data.hund_terms.clear();
            data.transfer_terms.clear();
            data.pair_hop_terms.clear();
            data.exchange_terms.clear();
            data.inter_all_terms.clear();
            for i in 0..4 {
                if group == 0 || group == 5 {
                    data.coulomb_intra_terms.push(CoulombIntraTerm {
                        site: i,
                        value: (i - 1) as f64 / 8.0,
                    });
                }
                for j in 0..4 {
                    let k = i * 4 + j;
                    if group == 0 || group == 5 {
                        data.coulomb_inter_terms.push(CoulombInterTerm {
                            site1: i,
                            site2: j,
                            value: (k % 5 - 2) as f64 / 16.0,
                        });
                        data.hund_terms.push(HundTerm {
                            site1: i,
                            site2: j,
                            value: (k % 7 - 3) as f64 / 32.0,
                        });
                    }
                    if group == 2 || group == 5 {
                        data.pair_hop_terms.push(PairHopTerm {
                            site1: i,
                            site2: j,
                            value: (k % 11 - 5) as f64 / 16.0,
                        });
                    }
                    if group == 3 || group == 5 {
                        data.exchange_terms.push(ExchangeTerm {
                            site1: i,
                            site2: j,
                            value: (k % 13 - 6) as f64 / 32.0,
                        });
                    }
                }
            }
            if group == 1 || group == 5 {
                for s in [Spin::Up, Spin::Down] {
                    for t in [Spin::Up, Spin::Down] {
                        for i in 0..4 {
                            for j in 0..4 {
                                let k = data.transfer_terms.len() as i64;
                                data.transfer_terms.push(TransferTerm {
                                    site1: i,
                                    spin1: s,
                                    site2: j,
                                    spin2: t,
                                    value: Complex64::new(
                                        (k % 7 - 3) as f64 / 16.0,
                                        (k % 11 - 5) as f64 / 32.0,
                                    ),
                                });
                            }
                        }
                    }
                }
                data.transfer_terms.push(data.transfer_terms[19]);
            }
            if !data.pair_hop_terms.is_empty() {
                for k in [0, 7] {
                    data.pair_hop_terms.push(data.pair_hop_terms[k]);
                }
            }
            if !data.exchange_terms.is_empty() {
                for k in [0, 11] {
                    data.exchange_terms.push(data.exchange_terms[k]);
                }
            }
            if group == 4 || group == 5 {
                for s in 0..2 {
                    for t in 0..2 {
                        for u in 0..2 {
                            for v in 0..2 {
                                for i in 0..4 {
                                    for j in 0..4 {
                                        for k in 0..4 {
                                            for l in 0..4 {
                                                let n = data.inter_all_terms.len() as i64;
                                                data.inter_all_terms.push(InterAllTerm {
                                                    site0: i,
                                                    spin0: s,
                                                    site1: j,
                                                    spin1: t,
                                                    site2: k,
                                                    spin2: u,
                                                    site3: l,
                                                    spin3: v,
                                                    is_complex: n % 11 != 5,
                                                    value: Complex64::new(
                                                        (n % 7 - 3) as f64 / 16.0,
                                                        (n % 11 - 5) as f64 / 32.0,
                                                    ),
                                                });
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                for k in [73, 3072] {
                    data.inter_all_terms.push(data.inter_all_terms[k]);
                }
            }
            assert_eq!(mvmc_core::get_all_complex_flag(&data).unwrap(), !REAL);
            let actual =
                calculate_local_energy_fsz(ip, &data, &mut state, &idx, &cfg, &num, &cnt, &spins);
            if !complex_within(actual, expected, ENERGY_ROUNDOFF) {
                failures.push(format!(
                    "case={case} group={group}: actual={actual:?} C={expected:?}"
                ));
            }
            assert_eq!(state.slater_matrix, before);
        }
    }
    assert!(lines.next().is_none());
    assert!(
        failures.is_empty(),
        "{} native Hamiltonian differences: {:?}",
        failures.len(),
        failures.first()
    );
}

fn add_rbm_green_model(data: &mut ExpertModeData) {
    let file = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/rbm/production/namelist_all.def");
    // The archived RBM control files intentionally contain sparse Julia
    // tables.  Use the historical test model, which restores those tables
    // after validating the complete C-compatible controls.
    let mut parsed = crate::historical_orbital_model::historical_kernel_model(&file).unwrap();
    historical_overlay_stage::read_input_parameters(&mut parsed, &file).unwrap();
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

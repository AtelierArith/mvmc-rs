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
    calculate_local_energy, calculate_local_energy_fsz, green_func1_fsz, green_func1_fsz_complex,
    green_func1_fsz_real, green_func2, green_func2_complex, green_func2_fsz,
    green_func2_fsz_complex, green_func2_fsz_real, green_func2_real,
};
use mvmc_core::{ExpertModeData, VmcOptimizationState};
use mvmc_expert_parsers::utils::qp_weight::init_qp_weight;
use mvmc_expert_parsers::{
    CoulombInterTerm, CoulombIntraTerm, DoublonHolon2SiteIndex, ExchangeTerm, GutzwillerTerm,
    HundTerm, InterAllTerm, JastrowTerm, PairHopTerm, Spin, TransferTerm,
};
use num_complex::Complex64;

use historical_orbital_model::historical_interall_model;
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
fn check_complex(
    actual: Complex64,
    expected: Complex64,
    bound: f64,
    context: impl std::fmt::Display,
) {
    numerical_comparison::assert_values_close(
        [actual.re, actual.im],
        [expected.re, expected.im],
        bound,
        bound,
        context,
    );
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

// Archived Julia energies use complex historical Green ratios even when the
// model header is real. Keep these independent Julia kernel expectations as
// such; production C Hamiltonians have separate native oracle tests below.
fn historical_fsz_energy(
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    ip: Complex64,
    configuration: [&[i64]; 4],
    spins: &[i64],
) -> Complex64 {
    let [idx, cfg, num, cnt] = configuration;
    assert!(data.transfer_terms.is_empty());
    let n = data.modpara.nsite as usize;
    let valid = |i: i64| (0..n as i64).contains(&i);
    let mut energy = Complex64::new(0.0, 0.0);
    for t in &data.coulomb_intra_terms {
        if valid(t.site) {
            let i = t.site as usize;
            energy += Complex64::new(t.value * (num[i] * num[i + n]) as f64, 0.0);
        }
    }
    for t in &data.coulomb_inter_terms {
        if valid(t.site1) && valid(t.site2) {
            let (i, j) = (t.site1 as usize, t.site2 as usize);
            energy += Complex64::new(
                t.value * (num[i] + num[i + n]) as f64 * (num[j] + num[j + n]) as f64,
                0.0,
            );
        }
    }
    for t in &data.hund_terms {
        if valid(t.site1) && valid(t.site2) {
            let (i, j) = (t.site1 as usize, t.site2 as usize);
            energy -= Complex64::new(
                t.value * (num[i] * num[j] + num[i + n] * num[j + n]) as f64,
                0.0,
            );
        }
    }
    let mut green = |sites: [usize; 4], codes: [u8; 4]| {
        let [i, j, k, l] = sites;
        let [s, t, u, v] = codes;
        green_func2_fsz(
            i, j, k, l, s, t, u, v, ip, data, state, idx, cfg, num, cnt, spins,
        )
    };
    for t in &data.pair_hop_terms {
        if valid(t.site1) && valid(t.site2) {
            let (i, j) = (t.site1 as usize, t.site2 as usize);
            energy += t.value * green([i, j, i, j], [0, 0, 1, 1]);
        }
    }
    for t in &data.exchange_terms {
        if valid(t.site1) && valid(t.site2) {
            let (i, j) = (t.site1 as usize, t.site2 as usize);
            energy +=
                t.value * (green([i, j, j, i], [0, 0, 1, 1]) + green([i, j, j, i], [1, 1, 0, 0]));
        }
    }
    for t in &data.inter_all_terms {
        let sites = [t.site0, t.site1, t.site2, t.site3];
        if sites.iter().all(|&site| valid(site)) {
            energy += t.value
                * green(
                    sites.map(|i| i as usize),
                    [t.spin0, t.spin1, t.spin2, t.spin3].map(|s| s as u8),
                );
        }
    }
    energy
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
            historical_fsz_energy(data, state, ip, [idx, cfg, num, cnt], spins)
        } else if mvmc_core::run::get_all_complex_flag(data).unwrap() {
            // Keep the archived Julia PairHop energy bits as a historical
            // kernel check. Production complex PairHop now follows C's
            // different quotient, independently checked below against C.
            let mut baseline = data.clone();
            baseline.pair_hop_terms.clear();
            baseline.exchange_terms.clear();
            let mut energy = calculate_local_energy(ip, &baseline, state, idx, cfg, num, cnt);
            for term in &data.pair_hop_terms {
                if !(0..4).contains(&term.site1) || !(0..4).contains(&term.site2) {
                    continue;
                }
                energy += term.value
                    * green_func2(
                        term.site1 as usize,
                        term.site2 as usize,
                        term.site1 as usize,
                        term.site2 as usize,
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
            // Julia adds Exchange after PairHop. Preserve the archived
            // addition order as well as its Green quotient bits.
            for term in &data.exchange_terms {
                let (ri, rj) = (term.site1 as usize, term.site2 as usize);
                let first = green_func2(ri, rj, rj, ri, 0, 1, ip, data, state, idx, cfg, num, cnt);
                let second = green_func2(ri, rj, rj, ri, 1, 0, ip, data, state, idx, cfg, num, cnt);
                energy += term.value * (first + second);
            }
            energy
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
        check_complex(
            actual,
            expected,
            ENERGY_ROUNDOFF,
            format!("PairHop {label}, spins={spins:?}, idx={idx:?}"),
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
        // Historical Julia normal quotients differ from the native C Green
        // kernels. Native fixtures check the production InterAll path.
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
    check_complex(
        actual,
        expected[2],
        ENERGY_ROUNDOFF,
        "complex reference value",
    );
    assert!((actual - expected[1]).norm() < 2e-14);
}

#[test]
fn exhaustive_four_site_two_body_ratios_match_original_julia() {
    check_normal_green("");
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
                &mut state,
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
                &mut state,
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
            // Preserve the source operation order, allowing local floating-point roundoff.
            for (a, e) in [(actual.re, expected.re), (actual.im, expected.im)] {
                numerical_comparison::assert_close(
                    a,
                    e,
                    GREEN_ROUNDOFF,
                    GREEN_ROUNDOFF,
                    format!("complex={complex}, idx={idx:?}, ops={ops:?}"),
                );
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
        numerical_comparison::assert_close(
            energy.re,
            expected_energy.re,
            ENERGY_ROUNDOFF,
            ENERGY_ROUNDOFF,
            "energy real",
        );
        numerical_comparison::assert_close(
            energy.im,
            expected_energy.im,
            ENERGY_ROUNDOFF,
            ENERGY_ROUNDOFF,
            "energy imaginary",
        );
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
fn native_fsz_green_rejects_rbm_before_density_reductions() {
    let mut data = green_data(true);
    add_rbm_green_model(&mut data);
    data.i_flg_orbital_general = 1;
    let mut state = VmcOptimizationState::zeros(4, 2, 2, 0, 2, 1, true, true);
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
                        0, 0, 0, 0, 0, 0, 0, 0, ip.re, &data, &mut state, &idx, &cfg, &num, &cnt,
                        &spins,
                    )
                } else {
                    green_func1_fsz_real(
                        0, 0, 0, 0, ip.re, &data, &mut state, &idx, &cfg, &num, &cnt, &spins,
                    )
                };
                Complex64::new(value, 0.0)
            } else if two_body {
                green_func2_fsz_complex(
                    0, 0, 0, 0, 0, 0, 0, 0, ip, &data, &mut state, &idx, &cfg, &num, &cnt, &spins,
                )
            } else {
                green_func1_fsz_complex(
                    0, 0, 0, 0, ip, &data, &mut state, &idx, &cfg, &num, &cnt, &spins,
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
                                        ri, rj, s, t, ip.re, &data, &mut state, &idx, &cfg, &num,
                                        &cnt, &spins,
                                    ),
                                    0.0,
                                )
                            } else {
                                green_func1_fsz_complex(
                                    ri, rj, s, t, ip, &data, &mut state, &idx, &cfg, &num, &cnt,
                                    &spins,
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
                                                    &mut state, &idx, &cfg, &num, &cnt, &spins,
                                                ),
                                                0.0,
                                            )
                                        } else {
                                            green_func2_fsz_complex(
                                                ri, rj, rk, rl, s, t, u, v, ip, &data, &mut state,
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

#[test]
fn exhaustive_fsz_one_and_two_body_spin_changes_match_original_julia_values() {
    check_fsz_green("");
}
#[test]
fn exhaustive_dh2_fsz_one_and_two_body_spin_changes_match_original_julia_values() {
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
                        check_complex(
                            actual,
                            expected,
                            GREEN_ROUNDOFF,
                            format!("case={case}, one={:?}", [ri, rj, s as usize, t as usize]),
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
                                        check_complex(actual, expected, GREEN_ROUNDOFF, format!("case={case}, two={:?}, actual={actual}, expected={expected}", [ri,rj,rk,rl,s as usize,t as usize,u as usize,v as usize]));
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
        let energy = historical_fsz_energy(&data, &mut state, ip, [&idx, &cfg, &num, &cnt], &spins);
        check_complex(
            energy,
            expected[0],
            ENERGY_ROUNDOFF,
            "complex reference value",
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
        data.inter_all_terms = historical_interall_model::parse_interall_content(
            "0 0 0 0 3 1 3 1 -0.5 0.125\n0 0 0 1 3 1 3 0 -0.375 0.1875\n0 0 0 1 2 0 2 1 0.125 -0.25\n1 1 2 0 2 0 0 1 -0.25 -0.375\n1 1 0 0 2 0 3 1 0.5 0.125\n0 0 0 1 3 1 3 0 -0.375 0.1875\n-1 0 0 1 3 1 3 0 0.25 0.125\n"
        );
        let energy = historical_fsz_energy(&data, &mut state, ip, [&idx, &cfg, &num, &cnt], &spins);
        check_complex(
            energy,
            expected[1],
            ENERGY_ROUNDOFF,
            format!("case={case} InterAll energy"),
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
fn exhaustive_dh4_fsz_one_and_two_body_spin_changes_match_original_julia_values() {
    check_fsz_green("dh4");
}
#[test]
fn exhaustive_dh24_fsz_one_and_two_body_spin_changes_match_original_julia_values() {
    check_fsz_green("dh24");
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
#[test]
fn exhaustive_rbm_normal_two_body_ratios_match_original_julia_values() {
    check_normal_green("rbm");
}
#[test]
fn exhaustive_rbm_fsz_one_and_two_body_spin_changes_match_original_julia_values() {
    check_fsz_green("rbm");
}

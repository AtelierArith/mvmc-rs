//! Source-level real FSZ sampler parity; see the fixture README for the
//! reference's unused save-log argument adaptation.

#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;
use mvmc_core::sampling::vmc_make_sample_fsz_real;
use mvmc_core::{ExpertModeData, VmcOptimizationState};
use mvmc_expert_parsers::{GutzwillerTerm, JastrowTerm, LocSpinTerm, QuantumProjectionWeights};
use num_complex::Complex64;
use sfmt19937::Sfmt19937Rng;

fn integers(line: &str) -> Vec<i64> {
    line.split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect()
}
fn bits(line: &str) -> Vec<u64> {
    line.split_whitespace()
        .map(|v| u64::from_str_radix(v, 16).unwrap())
        .collect()
}
fn data(case: &str) -> ExpertModeData {
    let mut d = ExpertModeData::default();
    d.modpara.nsite = 4;
    d.modpara.nelec = 2;
    d.modpara.two_sz = if case == "fixed" { 0 } else { -1 };
    d.modpara.nex_update_path = match case {
        "localspin" => 2,
        "mixed" => 3,
        _ => 1,
    };
    d.modpara.nvmc_warmup = 2;
    d.modpara.nvmc_sample = 3;
    d.modpara.nvmc_interval = 2;
    d.i_flg_orbital_general = 1;
    d.qp_weights = Some(QuantumProjectionWeights {
        qp_full_weight: vec![Complex64::new(0.625, 0.0), Complex64::new(0.375, 0.0)],
        ..Default::default()
    });
    d.n_gutzwiller_idx = 1;
    d.n_jastrow_idx = 1;
    d.gutzwiller_idx = vec![0; 4];
    d.jastrow_idx = vec![vec![0; 4]; 4];
    d.gutzwiller_terms = (0..4)
        .map(|site| GutzwillerTerm {
            site,
            value: Complex64::new(0.13, 0.0),
            is_complex: false,
        })
        .collect();
    d.jastrow_terms = (0..4)
        .flat_map(|site1| {
            (site1 + 1..4).map(move |site2| JastrowTerm {
                site1,
                site2,
                value: Complex64::new(-0.17, 0.0),
                is_complex: false,
            })
        })
        .collect();
    d.locspin_terms = match case {
        "localspin" => (0..4)
            .map(|site| LocSpinTerm {
                site,
                spin_value: 1,
            })
            .collect(),
        "mixed" => vec![LocSpinTerm {
            site: 0,
            spin_value: 1,
        }],
        _ => vec![],
    };
    d
}
fn state() -> VmcOptimizationState {
    let mut s = VmcOptimizationState::zeros(4, 2, 2, 0, 2, 3, false, true);
    for qp in 0..2 {
        for i in 0..8 {
            for j in i + 1..8 {
                let z = Complex64::new(
                    (((17 * i + 13 * j + 7 * qp) % 31) as i64 - 15) as f64 / 7.0 + 0.125,
                    0.0,
                );
                s.slater_matrix.slater_elm.set(qp, i, j, z);
                s.slater_matrix.slater_elm.set(qp, j, i, -z);
            }
        }
    }
    s
}

#[test]
fn real_fsz_sampler_matches_julia_configurations_counters_inverse_and_full_rng_blocks() {
    let fixture = if cfg!(all(
        target_os = "linux",
        target_env = "gnu",
        target_arch = "x86_64"
    )) {
        include_str!("../../../tests/fixtures/linux_gnu_julia/real_fsz/sampling.txt")
    } else {
        include_str!("../../../tests/fixtures/real_fsz/sampling.txt")
    };
    let mut lines = fixture.lines().filter(|line| !line.starts_with('#'));
    while let Some(header) = lines.next() {
        let mut header_fields = header.split_whitespace();
        let case = header_fields.next().unwrap();
        let steps: usize = header_fields.next().unwrap().parse().unwrap();
        let counters = integers(lines.next().unwrap());
        let configs: Vec<_> = (0..5).map(|_| integers(lines.next().unwrap())).collect();
        let burn = integers(lines.next().unwrap());
        let pf = bits(lines.next().unwrap());
        let inverse = bits(lines.next().unwrap());
        let rng_block = integers(lines.next().unwrap());
        let d = data(case);
        let mut s = state();
        let mut rng = Sfmt19937Rng::new(1);
        for step in 1..=steps {
            let stats = vmc_make_sample_fsz_real(&d, &mut s, &mut rng).unwrap();
            assert_eq!(stats.saved, 3, "{header}");
            if step == steps {
                assert_eq!(
                    stats.accepted as i64,
                    counters[1] + counters[3] + counters[5]
                );
            }
        }
        let c = &s.electron_config;
        assert_eq!(c.counter.as_slice(), counters, "{header} counters");
        for (label, actual, expected) in ["idx", "cfg", "num", "proj", "spins"]
            .into_iter()
            .zip([
                &c.ele_idx,
                &c.ele_cfg,
                &c.ele_num,
                &c.ele_proj_cnt,
                &c.ele_spn,
            ])
            .zip(configs)
            .map(|((label, actual), expected)| (label, actual, expected))
        {
            assert_eq!(actual, &expected, "{header} {label}");
        }
        let actual_burn = c.burn_ele_idx[..26].to_vec();
        assert_eq!(actual_burn, burn, "{header} burn");
        // RNG and the complete saved/burn state must pass before comparing
        // archived macOS Julia native-library results numerically.
        assert_eq!(
            (0..624)
                .map(|_| rng.gen_rand32() as i64)
                .collect::<Vec<_>>(),
            rng_block,
            "{header} RNG"
        );
        let actual_pf = s.slater_matrix.pf_m_real.to_vec();
        let actual_inverse: Vec<_> = (0..2)
            .flat_map(|qp| {
                s.slater_matrix
                    .inv_m_real
                    .qp_matrix_slice(qp)
                    .iter()
                    .copied()
            })
            .collect();
        numerical_comparison::assert_values_close(
            actual_pf,
            pf.into_iter().map(f64::from_bits),
            512.0 * f64::EPSILON,
            512.0 * f64::EPSILON,
            format!("{header} pf"),
        );
        numerical_comparison::assert_values_close(
            actual_inverse,
            inverse.into_iter().map(f64::from_bits),
            512.0 * f64::EPSILON,
            512.0 * f64::EPSILON,
            format!("{header} inverse"),
        );
    }
}

#[test]
fn failed_real_fsz_initialization_preserves_saved_samples_and_counters() {
    let d = data("conduction");
    let mut s = state();
    s.slater_matrix
        .slater_elm
        .as_mut_slice()
        .fill(Complex64::new(f64::NAN, 0.0));
    s.electron_config.ele_idx.fill(7);
    s.electron_config.ele_cfg.fill(7);
    s.electron_config.ele_num.fill(7);
    s.electron_config.ele_proj_cnt.fill(7);
    s.electron_config.ele_spn.fill(7);
    let saved = s.electron_config.clone();
    let error = vmc_make_sample_fsz_real(&d, &mut s, &mut Sfmt19937Rng::new(1)).unwrap_err();
    assert_eq!(error, mvmc_core::CalcMAllError::NonFinitePfaffian { qp: 0 });
    assert_eq!(s.electron_config.ele_idx, saved.ele_idx);
    assert_eq!(s.electron_config.ele_cfg, saved.ele_cfg);
    assert_eq!(s.electron_config.ele_num, saved.ele_num);
    assert_eq!(s.electron_config.ele_proj_cnt, saved.ele_proj_cnt);
    assert_eq!(s.electron_config.ele_spn, saved.ele_spn);
    assert_eq!(s.electron_config.counter, saved.counter);
}

#[test]
fn real_fsz_two_electron_proposal_and_update_match_julia_values() {
    use mvmc_core::sampling::{
        calculate_new_pf_m_two_fsz_real_flat, update_m_all_two_fsz_real_flat,
    };
    use mvmc_core::{InvMColMajor, SlaterElmFlat};
    let fixture = include_str!("../../../tests/fixtures/real_fsz/moves.txt");
    let mut lines = fixture.lines().filter(|line| !line.starts_with('#'));
    while let Some(header) = lines.next() {
        let dims = integers(header);
        let ns = dims[0] as usize;
        let ne = dims[1] as usize;
        let size = 2 * ne;
        let stride = size * size + 1;
        let mut slater = SlaterElmFlat::zeros(2, ns);
        let slater_values: Vec<_> = bits(lines.next().unwrap())
            .into_iter()
            .map(f64::from_bits)
            .collect();
        slater.as_mut_slice().copy_from_slice(&slater_values);
        let mut pf: Vec<_> = bits(lines.next().unwrap())
            .into_iter()
            .map(f64::from_bits)
            .collect();
        let inv_values: Vec<_> = bits(lines.next().unwrap())
            .into_iter()
            .map(f64::from_bits)
            .collect();
        let proposed = bits(lines.next().unwrap());
        let updated_pf = bits(lines.next().unwrap());
        let updated_inv = bits(lines.next().unwrap());
        let mut inverse = InvMColMajor::zeros(2, ne);
        for qp in 0..2 {
            inverse
                .qp_matrix_slice_mut(qp)
                .copy_from_slice(&inv_values[qp * size * size..(qp + 1) * size * size]);
        }
        let mut idx: Vec<_> = (0..size as i64).collect();
        let spins: Vec<_> = (0..size as i64).map(|i| i % 2).collect();
        idx.swap(0, 1);
        let mut new_pf = vec![0.0; 2];
        calculate_new_pf_m_two_fsz_real_flat(
            0,
            0,
            1,
            1,
            &mut new_pf,
            &idx,
            &spins,
            &slater,
            inverse.as_slice(),
            stride,
            &pf,
            0,
            2,
            ns,
            ne,
        );
        numerical_comparison::assert_values_close(
            new_pf.iter().copied(),
            proposed.into_iter().map(f64::from_bits),
            1e-13,
            1e-13,
            format!("{header} proposed"),
        );
        update_m_all_two_fsz_real_flat(
            0,
            0,
            1,
            1,
            0,
            1,
            &idx,
            &spins,
            &slater,
            inverse.as_mut_slice(),
            stride,
            &mut pf,
            0,
            2,
            ns,
            ne,
        );
        numerical_comparison::assert_values_close(
            pf.iter().copied(),
            updated_pf.into_iter().map(f64::from_bits),
            512.0 * f64::EPSILON,
            512.0 * f64::EPSILON,
            format!("{header} accepted pf"),
        );
        numerical_comparison::assert_values_close(
            (0..2).flat_map(|qp| inverse.qp_matrix_slice(qp).iter().copied()),
            updated_inv.into_iter().map(f64::from_bits),
            512.0 * f64::EPSILON,
            512.0 * f64::EPSILON,
            format!("{header} accepted inverse"),
        );
    }
}

#[test]
fn real_fsz_proposals_use_julia_scalar_arithmetic_on_shared_bilinear_inputs() {
    use mvmc_core::sampling::calculate_new_pf_m_two_fsz_real_flat;
    use mvmc_core::{InvMColMajor, SlaterElmFlat};
    let inputs = include_str!("../../../tests/fixtures/pfaffian_cg/two_hop_bilinear.txt");
    let expected = include_str!("../../../tests/fixtures/real_fsz/proposals.txt");
    let mut lines = inputs
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'));
    for row in expected.lines().filter(|l| !l.starts_with('#')) {
        let mut fields = row.split_whitespace();
        let n: usize = fields.next().unwrap().parse().unwrap();
        let expected = u64::from_str_radix(fields.next().unwrap(), 16).unwrap();
        assert_eq!(lines.next().unwrap().parse::<usize>().unwrap(), n);
        let values: Vec<_> = bits(lines.next().unwrap())
            .into_iter()
            .map(f64::from_bits)
            .collect();
        let mut inverse = InvMColMajor::zeros(1, n / 2);
        inverse
            .qp_matrix_slice_mut(0)
            .copy_from_slice(&values[1..1 + n * n]);
        let mut slater = SlaterElmFlat::zeros(1, n);
        slater.as_mut_slice()[..n].copy_from_slice(&values[1 + n * n..1 + n * n + n]);
        slater.as_mut_slice()[2 * n..3 * n].copy_from_slice(&values[1 + n * n + n..]);
        let idx: Vec<_> = (0..n as i64).collect();
        let mut proposed = [0.0];
        calculate_new_pf_m_two_fsz_real_flat(
            0,
            0,
            1,
            0,
            &mut proposed,
            &idx,
            &vec![0; n],
            &slater,
            inverse.as_slice(),
            n * n + 1,
            &[1.0],
            0,
            1,
            n,
            n / 2,
        );
        // Bilinear proposal has O(n^2) products/additions, including cancellation.
        let rounding = 8.0 * (n * n) as f64 * f64::EPSILON;
        numerical_comparison::assert_close(
            proposed[0],
            f64::from_bits(expected),
            rounding,
            rounding,
            format!("size {n} proposal"),
        );
    }
    assert!(lines.next().is_none());
}

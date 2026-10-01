//! Julia 1.13.1 real FSZ setup fixtures, including full post-retry RNG blocks.
use mvmc_core::pfaffian::{calc_m_all_fsz_real, CalcMAllError};
use mvmc_core::sampling::initial::make_initial_sample_fsz_real;
use mvmc_core::{ExpertModeData, SlaterMatrixData, ThreadedPfaPackWorkspace, VmcOptimizationState};
use num_complex::Complex64;
use sfmt19937::Sfmt19937Rng;

fn floats(line: &str) -> Vec<f64> {
    line.split_whitespace()
        .map(|v| f64::from_bits(u64::from_str_radix(v, 16).unwrap()))
        .collect()
}

fn complex(line: &str) -> Vec<Complex64> {
    floats(line)
        .chunks_exact(2)
        .map(|z| Complex64::new(z[0], z[1]))
        .collect()
}

fn integers(line: &str) -> Vec<i64> {
    line.split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect()
}

fn sentinels(s: &mut SlaterMatrixData) {
    s.pf_m.fill(Complex64::new(37.0, 11.0));
    s.inv_m.as_mut_slice().fill(Complex64::new(37.0, 11.0));
    s.pf_m_real.fill(23.0);
    s.inv_m_real.as_mut_slice().fill(23.0);
}

fn assert_bits(actual: impl IntoIterator<Item = f64>, expected: &[f64]) {
    let actual: Vec<_> = actual.into_iter().map(f64::to_bits).collect();
    let expected: Vec<_> = expected.iter().map(|v| v.to_bits()).collect();
    assert_eq!(actual, expected);
}

#[test]
fn real_fsz_calculation_and_initial_retries_match_julia() {
    let fixture = include_str!("../../../tests/fixtures/real_fsz/setup.txt");
    let mut lines = fixture.lines().filter(|line| !line.starts_with('#'));
    for _ in 0..16 {
        let header = integers(lines.next().unwrap());
        let ns = header[0] as usize;
        let ne = header[1] as usize;
        let idx = integers(lines.next().unwrap());
        let spins = integers(lines.next().unwrap());
        let slater = complex(lines.next().unwrap());
        let pf = floats(lines.next().unwrap());
        let inverse = floats(lines.next().unwrap());
        let real_pf = floats(lines.next().unwrap());
        let real_inverse = floats(lines.next().unwrap());
        let mut s = SlaterMatrixData::zeros(4, ns, ne, false);
        sentinels(&mut s);
        s.slater_elm.as_mut_slice().copy_from_slice(&slater);
        let pool = ThreadedPfaPackWorkspace::new(2 * ne, 1);
        calc_m_all_fsz_real(&idx, &spins, &mut s, 1, 3, ns, ne, &pool).unwrap();
        assert_bits(s.pf_m.iter().flat_map(|z| [z.re, z.im]), &pf);
        assert_bits(s.pf_m_real.iter().copied(), &real_pf);
        assert_bits(
            (0..4).flat_map(|qp| {
                s.inv_m
                    .qp_matrix_slice(qp)
                    .iter()
                    .flat_map(|z| [z.re, z.im])
            }),
            &inverse,
        );
        assert_bits(
            (0..4).flat_map(|qp| s.inv_m_real.qp_matrix_slice(qp).iter().copied()),
            &real_inverse,
        );
        // Rust has a scratch pad after each QP matrix; copying must skip it.
        let stride = (2 * ne).pow(2) + 1;
        for qp in 0..4 {
            assert_eq!(
                s.inv_m.as_slice()[qp * stride + stride - 1],
                Complex64::new(37.0, 11.0)
            );
            assert_eq!(s.inv_m_real.as_slice()[qp * stride + stride - 1], 23.0);
        }
        let before = s.clone();
        calc_m_all_fsz_real(&idx, &spins, &mut s, 2, 2, ns, ne, &pool).unwrap();
        assert_eq!(s, before);
    }
    for kind in ["retry", "failure", "zero", "localspin", "magnetized"] {
        let header = lines.next().unwrap();
        assert_eq!(header, format!("{kind} {}", i32::from(kind == "failure")));
        let expected: Vec<_> = (0..5).map(|_| integers(lines.next().unwrap())).collect();
        let rng_expected = integers(lines.next().unwrap());
        let mut data = ExpertModeData::default();
        data.modpara.nsite = 3;
        data.modpara.nelec = 1;
        data.modpara.two_sz = -1;
        let mut state = VmcOptimizationState::zeros(3, 1, 0, 0, 1, 1, false, true);
        let s = &mut state.slater_matrix;
        sentinels(s);
        s.slater_elm.as_mut_slice().fill(Complex64::new(
            if kind == "zero" { 0.0 } else { f64::NAN },
            0.0,
        ));
        if kind == "retry" {
            s.slater_elm.set(0, 0, 4, Complex64::new(1.25, 0.0));
            s.slater_elm.set(0, 4, 0, Complex64::new(-1.25, 0.0));
            for i in 0..6 {
                s.slater_elm.set(0, i, i, Complex64::new(0.0, 0.0));
            }
        }
        if matches!(kind, "localspin" | "magnetized") {
            for i in 0..6 {
                s.slater_elm.set(0, i, i, Complex64::new(0.0, 0.0));
                for j in i + 1..6 {
                    let z = Complex64::new((i + 2 * j + 1) as f64 / 7.0, 0.0);
                    s.slater_elm.set(0, i, j, z);
                    s.slater_elm.set(0, j, i, -z);
                }
            }
            if kind == "localspin" {
                data.locspin_terms = vec![mvmc_expert_parsers::LocSpinTerm {
                    site: 0,
                    spin_value: 1,
                }];
            } else {
                data.modpara.two_sz = 2;
            }
        }
        let mut rng = Sfmt19937Rng::new(1);
        let pool = ThreadedPfaPackWorkspace::new(2, 1);
        if kind == "retry" {
            // Prove the seed's first proposal is rejected by this Slater
            // table, so the fixture actually covers successful retries.
            let mut first = VmcOptimizationState::zeros(3, 1, 0, 0, 1, 1, false, true);
            let c = &mut first.electron_config;
            mvmc_core::sampling::make_initial_sample_fsz(
                &mut c.tmp_ele_idx,
                &mut c.tmp_ele_cfg,
                &mut c.tmp_ele_num,
                &mut c.tmp_ele_proj_cnt,
                &mut c.tmp_ele_spn,
                &data,
                &[0; 3],
                &mut Sfmt19937Rng::new(1),
            )
            .unwrap();
            assert_eq!(c.tmp_ele_idx, [0, 2]);
            assert_ne!(c.tmp_ele_idx, expected[0]);
            assert!(calc_m_all_fsz_real(
                &c.tmp_ele_idx,
                &c.tmp_ele_spn,
                &mut state.slater_matrix,
                0,
                1,
                3,
                1,
                &pool,
            )
            .is_err());
        }
        let result = make_initial_sample_fsz_real(&data, &mut state, &mut rng, 0, 1, &pool);
        if kind == "failure" {
            assert_eq!(result, Err(CalcMAllError::NonFinitePfaffian { qp: 0 }));
            assert_eq!(state.slater_matrix.pf_m_real, [23.0]);
            assert!(state
                .slater_matrix
                .inv_m_real
                .as_slice()
                .iter()
                .all(|&v| v == 23.0));
            assert_eq!(state.slater_matrix.pf_m, [Complex64::new(37.0, 11.0)]);
            assert!(state
                .slater_matrix
                .inv_m
                .as_slice()
                .iter()
                .all(|&v| v == Complex64::new(37.0, 11.0)));
        } else {
            result.unwrap();
            if kind == "zero" {
                assert_eq!(state.slater_matrix.pf_m_real, [0.0]);
                assert!(state
                    .slater_matrix
                    .inv_m_real
                    .qp_matrix_slice(0)
                    .iter()
                    .all(|v| v.is_nan()));
            }
        }
        let c = &state.electron_config;
        for (actual, expected) in [
            &c.tmp_ele_idx,
            &c.tmp_ele_cfg,
            &c.tmp_ele_num,
            &c.tmp_ele_proj_cnt,
            &c.tmp_ele_spn,
        ]
        .into_iter()
        .zip(expected)
        {
            assert_eq!(actual, &expected, "{kind}");
        }
        assert_eq!(
            (0..624)
                .map(|_| rng.gen_rand32() as i64)
                .collect::<Vec<_>>(),
            rng_expected,
            "{kind} RNG"
        );
    }
    assert!(lines.next().is_none());
}

#[test]
fn real_fsz_later_qp_failure_publishes_no_partial_results() {
    let mut s = SlaterMatrixData::zeros(4, 3, 1, false);
    sentinels(&mut s);
    for qp in 0..4 {
        s.slater_elm.set(qp, 0, 5, Complex64::new(1.25, 0.0));
        s.slater_elm.set(qp, 5, 0, Complex64::new(-1.25, 0.0));
    }
    s.slater_elm
        .qp_slice_mut(2)
        .fill(Complex64::new(f64::NAN, 0.0));
    let before = s.clone();
    let pool = ThreadedPfaPackWorkspace::new(2, 1);
    assert_eq!(
        calc_m_all_fsz_real(&[0, 2], &[0, 1], &mut s, 1, 3, 3, 1, &pool),
        Err(CalcMAllError::NonFinitePfaffian { qp: 2 }),
    );
    assert_eq!(s.pf_m, before.pf_m);
    assert_eq!(s.inv_m, before.inv_m);
    assert_eq!(s.pf_m_real, before.pf_m_real);
    assert_eq!(s.inv_m_real, before.inv_m_real);
}

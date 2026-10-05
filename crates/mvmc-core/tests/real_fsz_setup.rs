//! Julia 1.13.1 real FSZ setup fixtures, including full post-retry RNG blocks.

#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;
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
        .as_chunks::<2>()
        .0
        .iter()
        .map(|z| Complex64::new(z[0], z[1]))
        .collect()
}

fn integers(line: &str) -> Vec<i64> {
    line.split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect()
}

const NATIVE_C: &str = include_str!("../../../tests/fixtures/real_fsz/c_setup.txt");

fn hex_values(fields: &[&str]) -> Vec<f64> {
    fields
        .iter()
        .map(|v| f64::from_bits(u64::from_str_radix(v, 16).unwrap()))
        .collect()
}

struct NativeInit {
    attempts: usize,
    aborted: bool,
    infos: Vec<i64>,
    idx: Vec<i64>,
    cfg: Vec<i64>,
    num: Vec<i64>,
    spn: Vec<i64>,
    pf: Vec<f64>,
    inv: Vec<f64>,
    rng: Vec<i64>,
}

fn native_init_case(kind: &str) -> NativeInit {
    let mut lines = NATIVE_C
        .lines()
        .skip_while(|l| !l.starts_with(&format!("init_case {kind} ")));
    let header = lines.next().expect("native init case");
    let field = |name: &str| -> String {
        header
            .split_whitespace()
            .find_map(|t| t.strip_prefix(name))
            .unwrap()
            .to_string()
    };
    let list = |line: &str, label: &str| -> Vec<i64> {
        integers(
            line.strip_prefix(label)
                .unwrap_or_else(|| panic!("{label}")),
        )
    };
    let infos = list(lines.next().unwrap(), "init_infos=");
    let idx = list(lines.next().unwrap(), "eleIdx=");
    let cfg = list(lines.next().unwrap(), "eleCfg=");
    let num = list(lines.next().unwrap(), "eleNum=");
    assert_eq!(lines.next().unwrap(), "eleProjCnt=");
    let spn = list(lines.next().unwrap(), "eleSpn=");
    let pf = lines
        .next()
        .unwrap()
        .split_whitespace()
        .last()
        .unwrap()
        .to_string();
    let inv: Vec<_> = lines.next().unwrap().split_whitespace().skip(2).collect();
    let rng = list(lines.next().unwrap(), "rng624=");
    NativeInit {
        attempts: field("attempts=").parse().unwrap(),
        aborted: field("aborted=") == "1",
        infos,
        idx,
        cfg,
        num,
        spn,
        pf: hex_values(&[pf.as_str()]),
        inv: hex_values(&inv),
        rng,
    }
}

/// `(info, pf for qp 1..3, inverse for qp 1..3)` of native matrix case `case`.
fn native_matrix_case(case: usize) -> (i64, [f64; 2], [Vec<f64>; 2]) {
    let mut lines = NATIVE_C
        .lines()
        .skip_while(|l| !l.starts_with(&format!("matrix_case {case} ")));
    let header = lines.next().expect("native matrix case");
    let info = header.rsplit("info=").next().unwrap().parse().unwrap();
    let mut pf = [0.0; 2];
    let mut inv = [Vec::new(), Vec::new()];
    for k in 0..2 {
        let p: Vec<_> = lines.next().unwrap().split_whitespace().collect();
        assert_eq!(p[0], "matrix_pf");
        assert_eq!(p[1], format!("qp={}", k + 1));
        pf[k] = hex_values(&p[2..])[0];
        let i: Vec<_> = lines.next().unwrap().split_whitespace().collect();
        assert_eq!(i[0], "matrix_inv");
        inv[k] = hex_values(&i[2..]);
    }
    (info, pf, inv)
}

fn sentinels(s: &mut SlaterMatrixData) {
    s.pf_m.fill(Complex64::new(37.0, 11.0));
    s.inv_m.as_mut_slice().fill(Complex64::new(37.0, 11.0));
    s.pf_m_real.fill(23.0);
    s.inv_m_real.as_mut_slice().fill(23.0);
}

/// Rust and native C run the same real kernel sequence on identical operands, so only
/// library rounding can differ. Budget: `8 * n * eps` relative per entry plus the same
/// multiple of the largest entry of the compared object as absolute floor (entries that
/// are exact zeros in one implementation and rounding noise in the other). `n = 2*Ne` is
/// the elimination length; the factor 8 is margin over the O(n*eps) factor/inverse error.
/// Measured Linux difference against the native reference is 0 for every case; the
/// budget exists for other BLAS providers, not to hide a divergence.
fn native_close<'a>(
    actual: impl IntoIterator<Item = &'a f64>,
    expected: &[f64],
    ne: usize,
    label: &str,
) {
    let budget = 8.0 * (2 * ne) as f64 * f64::EPSILON;
    let scale = expected.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    numerical_comparison::assert_values_close(
        actual.into_iter().copied(),
        expected.iter().copied(),
        budget * scale,
        budget,
        label,
    );
}

fn assert_matrix_values(actual: impl IntoIterator<Item = f64>, expected: &[f64]) {
    // Small (<=8) native inverse kernels: allow roundoff amplified by inversion;
    // sentinel planes and scratch slots are checked exactly below.
    numerical_comparison::assert_values_close(
        actual,
        expected.iter().copied(),
        512.0 * f64::EPSILON,
        512.0 * f64::EPSILON,
        "FSZ setup matrix",
    );
}

#[test]
fn real_fsz_calculation_and_initial_retries_match_julia() {
    let fixture = if cfg!(all(
        target_os = "linux",
        target_env = "gnu",
        target_arch = "x86_64"
    )) {
        include_str!("../../../tests/fixtures/linux_gnu_julia/real_fsz/setup.txt")
    } else {
        include_str!("../../../tests/fixtures/real_fsz/setup.txt")
    };
    let mut lines = fixture.lines().filter(|line| !line.starts_with('#'));
    for case in 0..16 {
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
        assert_matrix_values(s.pf_m.iter().flat_map(|z| [z.re, z.im]), &pf);
        // Julia derives its real shadows from the complex computation. That equals C's
        // real kernel only for real-valued Slater input (C's real-mode contract), so the
        // Julia real-shadow expectations apply to those cases; the native-C reference
        // test below covers every case.
        let real_input = header[3] == 0;
        if real_input {
            assert_matrix_values(s.pf_m_real.iter().copied(), &real_pf);
        }
        // Published inverse values use native-library arithmetic; unchanged
        // QP planes remain an exact state-preservation gate.
        for qp in [0, 3] {
            assert!(s
                .inv_m
                .qp_matrix_slice(qp)
                .iter()
                .all(|&z| z == Complex64::new(37.0, 11.0)));
            assert!(s.inv_m_real.qp_matrix_slice(qp).iter().all(|&v| v == 23.0));
        }
        assert_matrix_values(
            (0..4).flat_map(|qp| {
                s.inv_m
                    .qp_matrix_slice(qp)
                    .iter()
                    .flat_map(|z| [z.re, z.im])
            }),
            &inverse,
        );
        if real_input {
            assert_matrix_values(
                (0..4).flat_map(|qp| s.inv_m_real.qp_matrix_slice(qp).iter().copied()),
                &real_inverse,
            );
        }
        // Native C reference for the real shadows (every case, including complex input
        // whose imaginary part C discards via creal()).
        let (info, c_pf, c_inv) = native_matrix_case(case);
        assert_eq!(info, 0, "native INFO");
        for (k, qp) in [1usize, 2].into_iter().enumerate() {
            native_close([&s.pf_m_real[qp]], &[c_pf[k]], ne, "native pf");
            native_close(
                s.inv_m_real.qp_matrix_slice(qp),
                &c_inv[k],
                ne,
                "native inverse",
            );
        }
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
        let julia_expected: Vec<_> = (0..5).map(|_| integers(lines.next().unwrap())).collect();
        let julia_rng = integers(lines.next().unwrap());
        // Native C is the authority. It agrees with the Julia fixture for every case except
        // `zero`: C's DSKTRF reports INFO>0 for the zero matrix, so C retries 101 times and
        // aborts, whereas Julia ignores the zero-pivot status and accepts it.
        let native = native_init_case(kind);
        let expected: Vec<Vec<i64>> = vec![
            native.idx.clone(),
            native.cfg.clone(),
            native.num.clone(),
            Vec::new(),
            native.spn.clone(),
        ];
        let rng_expected = native.rng.clone();
        if kind == "zero" {
            assert_eq!((native.attempts, native.aborted), (101, true));
            assert_ne!(
                rng_expected, julia_rng,
                "documented Julia/C zero-matrix difference"
            );
        } else {
            assert_eq!(
                julia_expected[..3],
                expected[..3],
                "{kind}: Julia/C configuration"
            );
            assert_eq!(julia_expected[4], expected[4], "{kind}: Julia/C spins");
            assert_eq!(julia_rng, rng_expected, "{kind}: Julia/C RNG block");
        }
        assert_eq!(native.aborted, kind == "failure" || kind == "zero");
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
            mvmc_core::sampling::generate_initial_fsz_configuration(
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
        if kind == "failure" || kind == "zero" {
            if kind == "failure" {
                assert_eq!(result, Err(CalcMAllError::NonFinitePfaffian { qp: 0 }));
            } else {
                // C: DSKTRF INFO of the last of the 101 attempts, then abort.
                assert_eq!(
                    result,
                    Err(CalcMAllError::ZeroPivot {
                        qp: 0,
                        info: *native.infos.last().unwrap() as usize
                    })
                );
            }
            assert_eq!(native.infos.len(), 101);
            assert!(native.infos.iter().all(|&info| info > 0));
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
            assert_eq!(native.infos.last(), Some(&0));
            assert_eq!(native.infos.len(), native.attempts);
            // The accepted state is the native kernel's: same Pfaffian and inverse.
            native_close(
                [&state.slater_matrix.pf_m_real[0]],
                &native.pf,
                1,
                "native init pf",
            );
            native_close(
                state.slater_matrix.inv_m_real.qp_matrix_slice(0),
                &native.inv,
                1,
                "native init inverse",
            );
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

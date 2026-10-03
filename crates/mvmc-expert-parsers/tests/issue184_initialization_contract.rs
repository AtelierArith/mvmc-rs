//! Literal initialization scenarios M0161–M0190 (RBM scenario excluded pending authority).
//! Julia source: test_parameter_initialization.jl, SHA c2c46086c95a392e4b33ae1d5fd36fc270c605dafce2b3e88a1b19325dc427a6.
//! Rust keeps declared C Slater coefficients in slater_params, not mapping-term values.
use mvmc_expert_parsers::utils::parameter_init::{
    init_parameter, initialize_parameters, sync_modified_parameter,
};
use mvmc_expert_parsers::{ExpertModeData, GutzwillerTerm, JastrowTerm, OrbitalTerm};
use num_complex::Complex64;
use sfmt19937::Sfmt19937Rng;

const SEED: u32 = 123456789;

fn orbitals(n: usize, complex: bool, active: usize) -> ExpertModeData {
    let mut data = ExpertModeData::new();
    data.modpara.n_orbital_idx = n as i64;
    data.orbital_terms = (0..n)
        .map(|i| OrbitalTerm {
            site1: 0,
            site2: i as i64 + 1,
            idx: i as i64,
            sign: 1,
            is_complex: complex,
        })
        .collect();
    data.optimization_flags = (0..n)
        .flat_map(|i| [i64::from(i < active), i64::from(complex && i < active)])
        .collect();
    data
}

fn same_rng(actual: &Sfmt19937Rng, expected: &Sfmt19937Rng, words: u128) {
    assert_eq!(actual.words_consumed(), words);
    assert_eq!(expected.words_consumed(), words);
    assert_eq!(actual.state_snapshot(), expected.state_snapshot());
    let (mut a, mut b) = (actual.clone(), expected.clone());
    for i in 0..624 {
        assert_eq!(a.gen_rand32(), b.gen_rand32(), "next word {i}");
    }
}

#[test]
fn literal_five_slots_respect_real_and_complex_conditional_draws() {
    for (complex, active) in [(false, 3), (false, 5), (true, 3), (true, 5)] {
        let mut data = orbitals(5, complex, active);
        let mut actual = Sfmt19937Rng::new(SEED);
        let mut expected = Sfmt19937Rng::new(SEED);
        init_parameter(&mut data, &mut actual);
        assert_eq!(data.slater_params.len(), 5);
        for (i, value) in data.slater_params.iter().enumerate() {
            if i >= active {
                assert_eq!(*value, Complex64::new(0.0, 0.0));
                continue;
            }
            let real = 2.0 * (expected.genrand_real2() - 0.5);
            let target = if complex {
                Complex64::new(
                    real / std::f64::consts::SQRT_2,
                    2.0 * (expected.genrand_real2() - 0.5) / std::f64::consts::SQRT_2,
                )
            } else {
                Complex64::new(real, 0.0)
            };
            // Explicit operation-rounding budget, not computed-float bitwise parity.
            assert!((*value - target).norm() <= 4.0 * f64::EPSILON);
            assert!(value.norm() > 1e-10);
            if !complex {
                assert!(value.re >= -1.0 && value.re < 1.0);
                assert_eq!(value.im, 0.0);
            }
        }
        same_rng(
            &actual,
            &expected,
            (active * if complex { 2 } else { 1 }) as u128,
        );
    }
}

#[test]
fn literal_normalization_scales_both_large_and_small_blocks() {
    for (input, expected) in [
        (vec![10.0, 5.0, 8.0], vec![4.0, 2.0, 3.2]),
        (vec![2.0, 3.0], vec![8.0 / 3.0, 4.0]),
    ] {
        let mut data = orbitals(input.len(), false, input.len());
        data.slater_params = input.into_iter().map(|x| Complex64::new(x, 0.0)).collect();
        sync_modified_parameter(&mut data, false);
        for (actual, expected) in data.slater_params.iter().zip(expected) {
            assert!((actual.re - expected).abs() <= 8.0 * f64::EPSILON);
            assert_eq!(actual.im, 0.0);
        }
    }
}

#[test]
fn literal_seed_pair_initializes_then_normalizes_without_extra_draws() {
    let mut results = Vec::new();
    for seed in [SEED, 987654321] {
        let mut data = orbitals(5, false, 5);
        let mut rng = Sfmt19937Rng::new(seed);
        let mut expected = Sfmt19937Rng::new(seed);
        for _ in 0..5 {
            expected.genrand_real2();
        }
        initialize_parameters(&mut data, &mut rng);
        same_rng(&rng, &expected, 5);
        assert!(data.slater_params.iter().all(|v| v.norm() <= 4.0 + 1e-10));
        results.push(data.slater_params);
    }
    assert!(results[0]
        .iter()
        .zip(&results[1])
        .any(|(a, b)| (*a - *b).norm() > 1e-10));
}

#[test]
fn empty_initialization_and_normalization_consume_zero_words() {
    let mut data = ExpertModeData::new();
    let mut rng = Sfmt19937Rng::new(SEED);
    let expected = rng.clone();
    initialize_parameters(&mut data, &mut rng);
    sync_modified_parameter(&mut data, false);
    assert!(data.slater_params.is_empty());
    same_rng(&rng, &expected, 0);
}

#[test]
fn literal_basic_and_ten_slot_workflow_zero_both_projections() {
    // M0161–0166 and M0178–0181: same two projection values and seed as Julia.
    // Only dense Slater storage is adapted to the C-declared Rust representation.
    for (n, normalize) in [(5, false), (10, true)] {
        let mut data = orbitals(n, false, n);
        data.gutzwiller_terms.push(GutzwillerTerm {
            site: 0,
            value: Complex64::new(0.5, 0.0),
            is_complex: false,
        });
        data.jastrow_terms.push(JastrowTerm {
            site1: 0,
            site2: 1,
            value: Complex64::new(0.1, 0.0),
            is_complex: false,
        });
        data.optimization_flags.splice(0..0, [1, 0, 1, 0]);
        let mut rng = Sfmt19937Rng::new(SEED);
        let mut expected_rng = Sfmt19937Rng::new(SEED);
        let mut expected: Vec<f64> = (0..n)
            .map(|_| 2.0 * (expected_rng.genrand_real2() - 0.5))
            .collect();
        if normalize {
            initialize_parameters(&mut data, &mut rng);
            let max = expected.iter().map(|v| v.abs()).fold(0.0, f64::max);
            for value in &mut expected {
                *value *= 4.0 / max;
            }
        } else {
            init_parameter(&mut data, &mut rng);
        }
        assert_eq!(data.gutzwiller_terms[0].value, Complex64::new(0.0, 0.0));
        assert_eq!(data.jastrow_terms[0].value, Complex64::new(0.0, 0.0));
        assert_eq!(data.slater_params.len(), n);
        assert!(data.slater_params.iter().any(|v| v.norm() > 1e-10));
        for (actual, expected) in data.slater_params.iter().zip(expected) {
            assert!((actual.re - expected).abs() <= 8.0 * f64::EPSILON);
            assert_eq!(actual.im, 0.0);
            if normalize {
                assert!(actual.norm() <= 4.0 + 1e-10);
            } else {
                assert!(actual.re >= -1.0 && actual.re < 1.0);
            }
        }
        same_rng(&rng, &expected_rng, n as u128);
    }
}

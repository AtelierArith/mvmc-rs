//! Original A202 public integer resolution, not RNG initialization/trajectory proof.
use mvmc_core::{resolve_rnd_seed, SingleProcessReducer, FALLBACK_SEED};

#[test]
fn public_seed_resolution_matches_original_literal_integer_policy() {
    for (input, explicit, group, expected) in [
        (FALLBACK_SEED, None, 0, 11272),
        (0, None, 0, 0),
        (123, None, 0, 123),
        (123, Some(777), 0, 777),
        (100, None, 3, 103),
    ] {
        assert_eq!(
            resolve_rnd_seed(input, explicit, group, &SingleProcessReducer).unwrap(),
            expected
        );
    }
}

#[test]
fn public_seed_resolution_returns_integer_before_sfmt_conversion() {
    // Existing explicit-override policy: do not invent a conversion/repair here.
    assert_eq!(
        resolve_rnd_seed(-1, Some(-7), 0, &SingleProcessReducer).unwrap(),
        -7
    );
    assert_eq!(
        resolve_rnd_seed(0, Some(i64::MAX), 1, &SingleProcessReducer).unwrap(),
        i64::MIN
    );
    assert_eq!(
        resolve_rnd_seed(i64::from(u32::MAX) + 1, None, 0, &SingleProcessReducer).unwrap(),
        i64::from(u32::MAX) + 1
    );
}

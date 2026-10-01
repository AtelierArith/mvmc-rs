//! Julia test_unit_vmc_main_cal_sr.jl storage/finalization contracts.
use mvmc_core::observables::{
    calculate_oo_store, calculate_oo_store_real, finalize_oo_store, finalize_oo_store_real,
    StoreFinalization,
};
use num_complex::Complex64 as C;

#[test]
fn stored_samples_use_sqrt_weight_and_accumulate_weighted_ho() {
    let mut ho = [0.0; 3];
    let mut store = [123.0; 9];
    calculate_oo_store_real(&mut ho, &mut store, &[1.0, 2.0, -3.0], 9.0, 2.0, 1, 3);
    assert_eq!(
        store,
        [123.0, 123.0, 123.0, 3.0, 6.0, -9.0, 123.0, 123.0, 123.0]
    );
    assert_eq!(ho, [18.0, 36.0, -54.0]);
    let mut ho = [C::new(0.0, 0.0); 2];
    let mut store = [C::new(123.0, 0.0); 6];
    calculate_oo_store(
        &mut ho,
        &mut store,
        &[C::new(1.0, 0.0), C::new(2.0, 1.0)],
        4.0,
        C::new(1.0, -2.0),
        1,
        1,
    );
    assert_eq!(store[2..4], [C::new(2.0, 0.0), C::new(4.0, 2.0)]);
    assert_eq!(store[0], C::new(123.0, 0.0));
    assert_eq!(store[5], C::new(123.0, 0.0));
    assert_eq!(ho, [C::new(4.0, -8.0), C::new(16.0, -12.0)]);
}

#[test]
fn real_gram_uses_active_sample_range_and_preserves_extra_slots() {
    let store = [
        99.0, 99.0, 99.0, 1.0, 2.0, -1.0, 1.0, 4.0, 2.0, 99.0, 99.0, 99.0,
    ];
    let mut oo = [777.0; 15];
    finalize_oo_store_real(
        &mut oo,
        &store,
        3,
        2,
        StoreFinalization {
            sample_start: 1,
            diagonal_only: false,
        },
    );
    assert_eq!(&oo[..9], &[2.0, 6.0, 1.0, 6.0, 20.0, 6.0, 1.0, 6.0, 5.0]);
    assert_eq!(&oo[9..], &[777.0; 6]);
    finalize_oo_store_real(
        &mut oo,
        &store,
        3,
        0,
        StoreFinalization {
            sample_start: 4,
            diagonal_only: false,
        },
    );
    assert_eq!(&oo[..9], &[0.0; 9]);
    assert_eq!(&oo[9..], &[777.0; 6]);
}

#[test]
fn complex_gram_honors_sample_start_and_legacy_flat_order() {
    let store = [
        C::new(99.0, 0.0),
        C::new(99.0, 0.0),
        C::new(1.0, 2.0),
        C::new(-3.0, 4.0),
        C::new(99.0, 0.0),
        C::new(99.0, 0.0),
    ];
    let mut oo = [C::new(777.0, 0.0); 8];
    finalize_oo_store(
        &mut oo,
        &store,
        1,
        1,
        StoreFinalization {
            sample_start: 1,
            diagonal_only: false,
        },
    );
    assert_eq!(
        &oo[..4],
        &[
            C::new(5.0, 0.0),
            C::new(5.0, -10.0),
            C::new(5.0, 10.0),
            C::new(25.0, 0.0)
        ]
    );
    assert_eq!(&oo[4..], &[C::new(777.0, 0.0); 4]);
    finalize_oo_store(
        &mut oo,
        &store,
        1,
        0,
        StoreFinalization {
            sample_start: 3,
            diagonal_only: false,
        },
    );
    assert_eq!(&oo[..4], &[C::new(0.0, 0.0); 4]);
}

#[test]
fn cg_finalization_materializes_only_mean_and_diagonal_blocks() {
    let real = [99.0, 99.0, 99.0, 1.0, 2.0, -1.0, 1.0, 4.0, 2.0];
    let mut oo = [777.0; 15];
    finalize_oo_store_real(
        &mut oo,
        &real,
        3,
        2,
        StoreFinalization {
            sample_start: 1,
            diagonal_only: true,
        },
    );
    assert_eq!(&oo[..6], &[2.0, 6.0, 1.0, 2.0, 20.0, 5.0]);
    assert_eq!(&oo[6..], &[777.0; 9]);
    let complex = [
        C::new(1.0, 0.0),
        C::new(2.0, 1.0),
        C::new(0.0, 1.0),
        C::new(1.0, -2.0),
        C::new(1.0, 0.0),
        C::new(-1.0, 2.0),
        C::new(3.0, 0.0),
        C::new(0.0, 1.0),
    ];
    let mut oo = [C::new(777.0, 0.0); 24];
    finalize_oo_store(
        &mut oo,
        &complex,
        2,
        2,
        StoreFinalization {
            sample_start: 0,
            diagonal_only: true,
        },
    );
    assert_eq!(
        &oo[..8],
        &[
            C::new(2.0, 0.0),
            C::new(1.0, 3.0),
            C::new(3.0, 1.0),
            C::new(1.0, -1.0),
            C::new(2.0, 0.0),
            C::new(10.0, 0.0),
            C::new(10.0, 0.0),
            C::new(6.0, 0.0)
        ]
    );
    assert_eq!(&oo[8..], &[C::new(777.0, 0.0); 16]);
}

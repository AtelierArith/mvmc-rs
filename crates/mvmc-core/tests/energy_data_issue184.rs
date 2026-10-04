//! S180 / M0843–M0854, Julia test_unit_threading.jl:91–117 (8bb1b9).
//! Independent literal energy moments, not a sampling oracle.
//!
//! C vmccal_fsz.c:144–148 accumulates w, w*E, w*conj(E)*E, w*Sz,
//! w*Sz*Sz. Thus the original samples (w=2,E=1+2i,Sz=3) and
//! (w=.5,E=-2+i,Sz=-1) have the immutable moment records below.
//! C average.c:43–75 reduces these five slots, keeps Wc, and multiplies
//! the other four by 1/Wc. Reset uses Rust's public clear_phys_quantity.
//!
//! Julia's private VMCEnergyAccumulator/merge APIs have no Rust public
//! counterpart. The fixture reducer below is a test adapter, NOT proof of
//! MPI execution or the private runner's energy packing/reduction call site.
//! Production public averaging/reset are exercised without adding an API.

use mvmc_core::average::weight_average_we;
use mvmc_core::observables::clear_phys_quantity;
use mvmc_core::state::EnergyData;
use mvmc_core::{Reducer, SingleProcessReducer, VmcOptimizationState};
use num_complex::Complex64;

const fn z(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

// [Wc, Etot, Etot2, Sztot, Sztot2]. These are literals independently
// evaluated from the original two samples, not copied from a Rust run.
const FIRST: [Complex64; 5] = [
    z(2.0, 0.0),
    z(2.0, 4.0),
    z(10.0, 0.0),
    z(6.0, 0.0),
    z(18.0, 0.0),
];
const SECOND: [Complex64; 5] = [
    z(0.5, 0.0),
    z(-1.0, 0.5),
    z(2.5, 0.0),
    z(-0.5, 0.0),
    z(0.5, 0.0),
];
const COMBINED: [Complex64; 5] = [
    z(2.5, 0.0),
    z(1.0, 4.5),
    z(12.5, 0.0),
    z(5.5, 0.0),
    z(18.5, 0.0),
];

fn energy(record: [Complex64; 5]) -> EnergyData {
    let [wc, etot, etot2, sztot, sztot2] = record;
    EnergyData {
        wc,
        etot,
        etot2,
        sztot,
        sztot2,
    }
}

fn fields(e: &EnergyData) -> [Complex64; 5] {
    [e.wc, e.etot, e.etot2, e.sztot, e.sztot2]
}

struct LiteralPeer;
impl Reducer for LiteralPeer {
    fn allreduce_sum_f64(&self, _: &mut [f64]) {
        panic!("energy fixture must use the complex five-slot boundary");
    }
    fn allreduce_sum_i64(&self, _: &mut [i64]) {
        panic!("energy fixture is not an integer reduction");
    }
    fn allreduce_sum_c64(&self, values: &mut [Complex64]) {
        assert_eq!(values, FIRST, "unchanged original first contribution");
        for (value, peer) in values.iter_mut().zip(SECOND) {
            *value += peer;
        }
    }
}

fn state() -> VmcOptimizationState {
    VmcOptimizationState::zeros(2, 1, 0, 0, 1, 1, true, false)
}

#[test]
fn original_s180_literal_moments_compose_through_public_reducer_contract() {
    let a = energy(FIRST);
    let b = energy(SECOND);
    let mut values = fields(&a);
    LiteralPeer.allreduce_sum_c64(&mut values);
    // M0843–0847: five literal binary-rational sums; exact arithmetic here.
    assert_eq!(values, COMBINED);
    let mut serial = values;
    SingleProcessReducer.allreduce_sum_c64(&mut serial);
    assert_eq!(serial, COMBINED);
    let mut output = state();
    output.energy = energy(serial);
    // M0848–0852: public EnergyData receives all five combined moments.
    assert_eq!(fields(&output.energy), COMBINED);
    assert_eq!(fields(&a), FIRST, "source contribution remains immutable");
    assert_eq!(fields(&b), SECOND, "peer contribution remains immutable");
}

#[test]
fn c_energy_normalization_keeps_weight_and_uses_absolute_square_second_moment() {
    let mut output = state();
    output.energy = energy(COMBINED);
    weight_average_we(&mut output);
    // Analytical division by 2.5. Etot2 is <|E|²>, not Etot² and not
    // an energy variance. Five input moments are independent literals.
    let expected = [
        z(2.5, 0.0),
        z(0.4, 1.8),
        z(5.0, 0.0),
        z(2.2, 0.0),
        z(7.4, 0.0),
    ];
    for (column, (actual, expected)) in fields(&output.energy).iter().zip(expected).enumerate() {
        for (a, e) in [(actual.re, expected.re), (actual.im, expected.im)] {
            // One reciprocal/multiply; 4 eps absolute covers rounding at
            // largest expected magnitude 7.4, without a scale-free waiver.
            assert!(
                (a - e).abs() <= 4.0 * f64::EPSILON,
                "moment {column}: {a} != {e}"
            );
        }
    }
    assert_eq!(output.energy.wc, COMBINED[0]);
}

#[test]
fn original_s180_clear_resets_all_five_moments_without_changing_saved_walkers() {
    let mut output = state();
    assert_eq!(fields(&EnergyData::new()), [z(0.0, 0.0); 5]);
    output.energy = energy(COMBINED);
    output.electron_config.ele_idx[0] = 1;
    let before = format!("{:?}", output.electron_config);
    clear_phys_quantity(&mut output);
    // M0853–0854 assert Wc and Etot; also guard the other reset moments.
    assert_eq!(fields(&output.energy), [z(0.0, 0.0); 5]);
    assert_eq!(format!("{:?}", output.electron_config), before);
    clear_phys_quantity(&mut output);
    assert_eq!(fields(&output.energy), [z(0.0, 0.0); 5], "idempotent reset");
}

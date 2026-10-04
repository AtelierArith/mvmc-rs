//! Original Julia S181 literal contracts, without a reference runtime.

use mvmc_core::sr_accumulator::{clear_sr_store, merge_sr_locals, SrAccumulator};
use mvmc_core::SROptData;
use num_complex::Complex64;

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

fn locals(shape: &SROptData) -> [SrAccumulator; 2] {
    let mut a = SrAccumulator::zeros_like(shape);
    let mut b = SrAccumulator::zeros_like(shape);
    a.buffers.sr_opt_oo[..3].copy_from_slice(&[c(1., 1.), c(2., 0.), c(-1., 2.)]);
    b.buffers.sr_opt_oo[..3].copy_from_slice(&[c(0.5, 0.), c(3., -1.), c(1., 0.)]);
    a.buffers.sr_opt_ho[..2].copy_from_slice(&[c(2., 0.), c(1., -1.)]);
    b.buffers.sr_opt_ho[..2].copy_from_slice(&[c(-1., 1.), c(4., 0.)]);
    a.buffers.sr_opt_o_store[..2].copy_from_slice(&[c(3., 0.), c(4., 0.)]);
    b.buffers.sr_opt_o_store[..2].copy_from_slice(&[c(5., 0.), c(6., 0.)]);
    a.buffers.sr_opt_oo_real[..3].copy_from_slice(&[1., 2., 3.]);
    b.buffers.sr_opt_oo_real[..3].copy_from_slice(&[4., 5., 6.]);
    a.buffers.sr_opt_ho_real[..2].copy_from_slice(&[7., 8.]);
    b.buffers.sr_opt_ho_real[..2].copy_from_slice(&[9., 10.]);
    a.buffers.sr_opt_o_store_real[..2].copy_from_slice(&[11., 12.]);
    b.buffers.sr_opt_o_store_real[..2].copy_from_slice(&[13., 14.]);
    [a, b]
}

#[test]
fn original_s181_twelve_literal_merge_and_clear_assertions() {
    // Julia SROptData(2,3,false): two parameters -> sr_opt_size=3.
    let mut destination = SROptData::zeros(3, 3, false);
    let mut local = locals(&destination);
    let before = [local[0].buffers.clone(), local[1].buffers.clone()];
    merge_sr_locals(&mut destination, &local).unwrap();
    // M0855–M0860: literal independent dyadic sums, not generated goldens.
    assert_eq!(
        destination.sr_opt_oo[..3],
        [c(1.5, 1.), c(5., -1.), c(0., 2.)]
    );
    assert_eq!(destination.sr_opt_ho[..2], [c(1., 1.), c(5., -1.)]);
    assert_eq!(destination.sr_opt_o_store[..2], [c(8., 0.), c(10., 0.)]);
    assert_eq!(destination.sr_opt_oo_real[..3], [5., 7., 9.]);
    assert_eq!(destination.sr_opt_ho_real[..2], [16., 18.]);
    assert_eq!(destination.sr_opt_o_store_real[..2], [24., 26.]);
    assert_eq!(local[0].buffers, before[0]);
    assert_eq!(local[1].buffers, before[1]);
    local[0].buffers.sr_opt_o.fill(c(19., -23.));
    local[0].buffers.sr_opt_o_real.fill(29.);
    local[0].clear();
    // M0861–M0864.
    assert!(local[0].buffers.sr_opt_oo.iter().all(|v| *v == c(0., 0.)));
    assert!(local[0].buffers.sr_opt_o.iter().all(|v| *v == c(0., 0.)));
    assert!(local[0].buffers.sr_opt_oo_real.iter().all(|v| *v == 0.));
    assert!(local[0].buffers.sr_opt_o_real.iter().all(|v| *v == 0.));
    destination.sr_opt_o_store[..2].copy_from_slice(&[c(1., 2.), c(3., 4.)]);
    destination.sr_opt_o_store_real[..2].copy_from_slice(&[5., 6.]);
    let aggregate = destination.sr_opt_oo.clone();
    clear_sr_store(&mut destination);
    // M0865–M0866.
    assert!(destination.sr_opt_o_store.iter().all(|v| *v == c(0., 0.)));
    assert!(destination.sr_opt_o_store_real.iter().all(|v| *v == 0.));
    assert_eq!(destination.sr_opt_oo, aggregate);
}

#[test]
fn locals_add_to_existing_destination_without_merging_sample_scratch() {
    let mut destination = SROptData::zeros(3, 3, false);
    destination.sr_opt_oo.fill(c(7., -2.));
    destination.sr_opt_ho.fill(c(7., -2.));
    destination.sr_opt_o.fill(c(11., 12.));
    destination.sr_opt_o_real.fill(13.);
    let mut local = locals(&destination);
    local[0].buffers.sr_opt_o.fill(c(99., 99.));
    local[1].buffers.sr_opt_o_real.fill(99.);
    merge_sr_locals(&mut destination, &local).unwrap();
    assert_eq!(destination.sr_opt_oo[0], c(8.5, -1.));
    assert_eq!(destination.sr_opt_oo[3], c(7., -2.));
    assert_eq!(destination.sr_opt_ho[0], c(8., -1.));
    assert!(destination.sr_opt_o.iter().all(|v| *v == c(11., 12.)));
    assert!(destination.sr_opt_o_real.iter().all(|v| *v == 13.));
    assert_eq!(local[0].buffers.sr_opt_o[0], c(99., 99.));
}

#[test]
fn local_order_is_preserved_and_later_shape_error_is_nonmutating() {
    let mut destination = SROptData::zeros(3, 3, false);
    destination.sr_opt_oo_real[0] = 1.;
    let mut a = SrAccumulator::zeros_like(&destination);
    let mut b = SrAccumulator::zeros_like(&destination);
    a.buffers.sr_opt_oo_real[0] = 9_007_199_254_740_992.;
    b.buffers.sr_opt_oo_real[0] = -9_007_199_254_740_992.;
    merge_sr_locals(&mut destination, &[a, b]).unwrap();
    assert_eq!(destination.sr_opt_oo_real[0], 0.);
    // Independent association control: regrouping locals first would retain 1.
    let valid = SrAccumulator::zeros_like(&destination);
    let mut bad = SrAccumulator::zeros_like(&destination);
    bad.buffers.sr_opt_o.pop();
    let before = destination.clone();
    assert!(merge_sr_locals(&mut destination, &[valid, bad]).is_err());
    assert_eq!(destination, before);
}

#[test]
fn clear_resets_all_local_buffers_and_empty_real_mode_remains_empty() {
    for complex in [false, true] {
        let shape = SROptData::zeros(3, 3, complex);
        let mut a = SrAccumulator::zeros_like(&shape);
        a.buffers.sr_opt_ho.fill(c(1., 2.));
        a.buffers.sr_opt_o.fill(c(3., 4.));
        a.buffers.sr_opt_o_store.fill(c(5., 6.));
        a.buffers.sr_opt_ho_real.fill(7.);
        a.buffers.sr_opt_o_real.fill(8.);
        a.buffers.sr_opt_o_store_real.fill(9.);
        a.clear();
        assert_eq!(a.buffers, shape);
    }
}

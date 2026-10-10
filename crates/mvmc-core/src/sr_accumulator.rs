//! Owned rank-local SR accumulation and ordered publication.
//!
//! These objects are not MPI reductions. Sample stores stay rank-local;
//! independently owned local contributions may be added in caller-supplied order.

use crate::state::SROptData;
use num_complex::Complex64;

/// An owned local SR aggregate, per-sample scratch and sample store.
#[derive(Debug)]
pub struct SrAccumulator {
    /// Local buffers; scratch O is never merged into an aggregate destination.
    pub buffers: SROptData,
}

/// A local/destination layout disagreement, detected before any merge writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SrAccumulatorShapeError;

impl std::fmt::Display for SrAccumulatorShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SR local and destination buffer shapes differ")
    }
}

impl std::error::Error for SrAccumulatorShapeError {}

impl SrAccumulator {
    /// Allocate zero buffers with exactly the template's eight buffer lengths.
    pub fn zeros_like(template: &SROptData) -> Self {
        let complex = |n| vec![Complex64::new(0.0, 0.0); n];
        Self {
            buffers: SROptData {
                sr_opt_size: template.sr_opt_size,
                sr_opt_oo: complex(template.sr_opt_oo.len()),
                sr_opt_ho: complex(template.sr_opt_ho.len()),
                sr_opt_o: complex(template.sr_opt_o.len()),
                sr_opt_o_store: complex(template.sr_opt_o_store.len()),
                sr_opt_oo_real: vec![0.0; template.sr_opt_oo_real.len()],
                sr_opt_ho_real: vec![0.0; template.sr_opt_ho_real.len()],
                sr_opt_o_real: vec![0.0; template.sr_opt_o_real.len()],
                sr_opt_o_store_real: vec![0.0; template.sr_opt_o_store_real.len()],
            },
        }
    }

    /// Take ownership of completed local measurements without copying them.
    pub fn from_buffers(buffers: SROptData) -> Self {
        Self { buffers }
    }

    /// Clear all eight local buffers, including scratch and stores.
    pub fn clear(&mut self) {
        let b = &mut self.buffers;
        for values in [
            &mut b.sr_opt_oo,
            &mut b.sr_opt_ho,
            &mut b.sr_opt_o,
            &mut b.sr_opt_o_store,
        ] {
            values.fill(Complex64::new(0.0, 0.0));
        }
        for values in [
            &mut b.sr_opt_oo_real,
            &mut b.sr_opt_ho_real,
            &mut b.sr_opt_o_real,
            &mut b.sr_opt_o_store_real,
        ] {
            values.fill(0.0);
        }
    }

    /// Add aggregate/store values, leaving destination per-sample scratch intact.
    ///
    /// The caller owns sample-column assignment: this does not redistribute or
    /// sum sample stores across MPI ranks. Source buffers remain unchanged.
    pub fn merge_into(&self, destination: &mut SROptData) -> Result<(), SrAccumulatorShapeError> {
        check_shape(destination, &self.buffers)?;
        add_buffers(destination, &self.buffers);
        Ok(())
    }
}

/// Zero destination stores only, leaving aggregates and scratch intact.
pub fn clear_sr_store(destination: &mut SROptData) {
    destination.sr_opt_o_store.fill(Complex64::new(0.0, 0.0));
    destination.sr_opt_o_store_real.fill(0.0);
}

/// Merge a local slice in its supplied order; validate every shape before writes.
pub fn merge_sr_locals(
    destination: &mut SROptData,
    locals: &[SrAccumulator],
) -> Result<(), SrAccumulatorShapeError> {
    for local in locals {
        check_shape(destination, &local.buffers)?;
    }
    for local in locals {
        add_buffers(destination, &local.buffers);
    }
    Ok(())
}

fn check_shape(a: &SROptData, b: &SROptData) -> Result<(), SrAccumulatorShapeError> {
    if !same_aggregate_scratch_shape(a, b)
        || a.sr_opt_o_store.len() != b.sr_opt_o_store.len()
        || a.sr_opt_o_store_real.len() != b.sr_opt_o_store_real.len()
    {
        Err(SrAccumulatorShapeError)
    } else {
        Ok(())
    }
}

fn same_aggregate_scratch_shape(a: &SROptData, b: &SROptData) -> bool {
    a.sr_opt_size == b.sr_opt_size
        && a.sr_opt_oo.len() == b.sr_opt_oo.len()
        && a.sr_opt_ho.len() == b.sr_opt_ho.len()
        && a.sr_opt_o.len() == b.sr_opt_o.len()
        && a.sr_opt_oo_real.len() == b.sr_opt_oo_real.len()
        && a.sr_opt_ho_real.len() == b.sr_opt_ho_real.len()
        && a.sr_opt_o_real.len() == b.sr_opt_o_real.len()
}

fn add_buffers(destination: &mut SROptData, source: &SROptData) {
    add_aggregates(destination, source);
    for (d, s) in destination
        .sr_opt_o_store
        .iter_mut()
        .zip(&source.sr_opt_o_store)
    {
        *d += *s;
    }
    for (d, s) in destination
        .sr_opt_o_store_real
        .iter_mut()
        .zip(&source.sr_opt_o_store_real)
    {
        *d += *s;
    }
}

fn add_aggregates(destination: &mut SROptData, source: &SROptData) {
    for (dst, src) in [
        (&mut destination.sr_opt_oo, &source.sr_opt_oo),
        (&mut destination.sr_opt_ho, &source.sr_opt_ho),
    ] {
        for (d, s) in dst.iter_mut().zip(src) {
            *d += *s;
        }
    }
    for (dst, src) in [
        (&mut destination.sr_opt_oo_real, &source.sr_opt_oo_real),
        (&mut destination.sr_opt_ho_real, &source.sr_opt_ho_real),
    ] {
        for (d, s) in dst.iter_mut().zip(src) {
            *d += *s;
        }
    }
}

pub(crate) fn publish_measurement_in_place(buffers: &mut SROptData) {
    for values in [
        &mut buffers.sr_opt_oo,
        &mut buffers.sr_opt_ho,
        &mut buffers.sr_opt_o_store,
    ] {
        for v in values {
            *v = Complex64::new(0.0, 0.0) + *v;
        }
    }
    for values in [
        &mut buffers.sr_opt_oo_real,
        &mut buffers.sr_opt_ho_real,
        &mut buffers.sr_opt_o_store_real,
    ] {
        for v in values {
            *v += 0.0;
        }
    }
}

#[cfg(test)]
mod publication_tests {
    use super::*;

    #[test]
    fn publication_preserves_scratch_and_store_ownership_and_normalizes_aggregate_zero() {
        for all_complex in [false, true] {
            let mut buffers = SROptData::zeros(5, 3, all_complex);
            let store = buffers.sr_opt_o_store.as_ptr();
            let real_store = buffers.sr_opt_o_store_real.as_ptr();
            let aggregate = buffers.sr_opt_oo.as_ptr();
            buffers.sr_opt_oo.fill(Complex64::new(-0.0, -0.0));
            buffers.sr_opt_ho.fill(Complex64::new(2.0, -3.0));
            buffers.sr_opt_o.fill(Complex64::new(-0.0, -0.0));
            buffers.sr_opt_o_store.fill(Complex64::new(4.0, -5.0));
            buffers.sr_opt_oo_real.fill(-0.0);
            buffers.sr_opt_ho_real.fill(7.0);
            buffers.sr_opt_o_real.fill(-0.0);
            buffers.sr_opt_o_store_real.fill(9.0);
            publish_measurement_in_place(&mut buffers);
            assert_eq!(buffers.sr_opt_oo.as_ptr(), aggregate);
            assert_eq!(buffers.sr_opt_o_store.as_ptr(), store);
            assert_eq!(buffers.sr_opt_o_store_real.as_ptr(), real_store);
            for v in &buffers.sr_opt_oo {
                assert!(!v.re.is_sign_negative() && !v.im.is_sign_negative());
            }
            for v in &buffers.sr_opt_oo_real {
                assert!(!v.is_sign_negative());
            }
            for v in &buffers.sr_opt_o {
                assert!(v.re.is_sign_negative() && v.im.is_sign_negative());
            }
            for v in &buffers.sr_opt_o_real {
                assert!(v.is_sign_negative());
            }
            assert!(buffers
                .sr_opt_ho
                .iter()
                .all(|v| (*v - Complex64::new(2.0, -3.0)).norm() <= f64::EPSILON));
            assert!(buffers
                .sr_opt_o_store
                .iter()
                .all(|v| (*v - Complex64::new(4.0, -5.0)).norm() <= f64::EPSILON));
            assert!(buffers
                .sr_opt_ho_real
                .iter()
                .all(|v| (*v - 7.0).abs() <= f64::EPSILON));
            assert!(buffers
                .sr_opt_o_store_real
                .iter()
                .all(|v| (*v - 9.0).abs() <= f64::EPSILON));
        }
    }
}

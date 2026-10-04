//! Owned rank-local SR accumulation and ordered publication.
//!
//! These objects are not MPI reductions. Sample stores stay rank-local;
//! independently owned local contributions may be added in caller-supplied order.

use crate::state::{SROptData, VmcOptimizationState};
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

    fn measurement_cache(template: &SROptData) -> Self {
        // Do not allocate stores and discard them: there is one store owner.
        let c = |n| vec![Complex64::new(0.0, 0.0); n];
        Self {
            buffers: SROptData {
                sr_opt_size: template.sr_opt_size,
                sr_opt_oo: c(template.sr_opt_oo.len()),
                sr_opt_ho: c(template.sr_opt_ho.len()),
                sr_opt_o: c(template.sr_opt_o.len()),
                sr_opt_o_store: Vec::new(),
                sr_opt_oo_real: vec![0.0; template.sr_opt_oo_real.len()],
                sr_opt_ho_real: vec![0.0; template.sr_opt_ho_real.len()],
                sr_opt_o_real: vec![0.0; template.sr_opt_o_real.len()],
                sr_opt_o_store_real: Vec::new(),
            },
        }
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

/// One runner-specific measurement scope; no generic guard framework.
pub(crate) struct SrMeasurement<'a> {
    state: &'a mut VmcOptimizationState,
    destination: Option<SrAccumulator>,
}

impl<'a> SrMeasurement<'a> {
    pub(crate) fn begin(state: &'a mut VmcOptimizationState) -> Self {
        let reuse = state
            .sr_local
            .as_ref()
            .is_some_and(|local| same_aggregate_scratch_shape(&local.buffers, &state.sr_opt));
        // Allocate before detaching state buffers; a shape change is explicit.
        let mut local = if reuse {
            state.sr_local.take().expect("matching cache is present")
        } else {
            SrAccumulator::measurement_cache(&state.sr_opt)
        };
        // A replaced cache is discarded only after its successor allocated.
        // Public partial/previous measurement buffers remain in state.sr_opt.
        state.sr_local = None;
        assert!(local.buffers.sr_opt_o_store.is_empty());
        assert!(local.buffers.sr_opt_o_store_real.is_empty());
        local.clear();
        std::mem::swap(
            &mut local.buffers.sr_opt_o_store,
            &mut state.sr_opt.sr_opt_o_store,
        );
        std::mem::swap(
            &mut local.buffers.sr_opt_o_store_real,
            &mut state.sr_opt.sr_opt_o_store_real,
        );
        std::mem::swap(&mut state.sr_opt, &mut local.buffers);
        Self {
            state,
            destination: Some(local),
        }
    }

    pub(crate) fn state(&mut self) -> &mut VmcOptimizationState {
        self.state
    }

    pub(crate) fn finish(mut self) {
        let mut destination = self.destination.take().expect("unfinished measurement");
        std::mem::swap(&mut self.state.sr_opt, &mut destination.buffers);
        // Destination was cleared by the existing runner lifecycle. Scratch
        // is transferred, not added; stores remain uniquely rank/sample owned.
        add_aggregates(&mut self.state.sr_opt, &destination.buffers);
        std::mem::swap(
            &mut self.state.sr_opt.sr_opt_o,
            &mut destination.buffers.sr_opt_o,
        );
        std::mem::swap(
            &mut self.state.sr_opt.sr_opt_o_real,
            &mut destination.buffers.sr_opt_o_real,
        );
        std::mem::swap(
            &mut self.state.sr_opt.sr_opt_o_store,
            &mut destination.buffers.sr_opt_o_store,
        );
        std::mem::swap(
            &mut self.state.sr_opt.sr_opt_o_store_real,
            &mut destination.buffers.sr_opt_o_store_real,
        );
        // Preserve the previous publication zero-add for transferred stores.
        for v in &mut self.state.sr_opt.sr_opt_o_store {
            *v = Complex64::new(0.0, 0.0) + *v;
        }
        for v in &mut self.state.sr_opt.sr_opt_o_store_real {
            *v += 0.0;
        }
        self.state.sr_local = Some(destination);
    }
}

impl Drop for SrMeasurement<'_> {
    fn drop(&mut self) {
        if let Some(destination) = self.destination.take() {
            // On unwind keep the partially measured local in public sr_opt.
            // The old destination is reusable and has no store ownership.
            self.state.sr_local = Some(destination);
        }
    }
}

pub(crate) fn publish_physcal_in_place(buffers: &mut SROptData) {
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
mod lifecycle_tests {
    use super::*;
    use crate::observables::clear_phys_quantity;

    fn state() -> VmcOptimizationState {
        VmcOptimizationState::zeros(2, 1, 0, 2, 1, 3, false, false)
    }

    fn cache_pointers(state: &VmcOptimizationState) -> [usize; 4] {
        let b = &state.sr_local.as_ref().unwrap().buffers;
        [
            b.sr_opt_oo.as_ptr() as usize,
            b.sr_opt_ho.as_ptr() as usize,
            b.sr_opt_oo_real.as_ptr() as usize,
            b.sr_opt_ho_real.as_ptr() as usize,
        ]
    }

    #[test]
    fn owned_local_is_active_during_measurement_and_stores_have_one_owner() {
        let mut state = state();
        assert!(state.sr_local.is_none());
        let store = state.sr_opt.sr_opt_o_store.as_ptr();
        let real_store = state.sr_opt.sr_opt_o_store_real.as_ptr();
        let published_oo = state.sr_opt.sr_opt_oo.as_ptr();
        let mut local = SrMeasurement::begin(&mut state);
        assert_ne!(local.state().sr_opt.sr_opt_oo.as_ptr(), published_oo);
        assert_eq!(local.state().sr_opt.sr_opt_o_store.as_ptr(), store);
        assert_eq!(
            local.state().sr_opt.sr_opt_o_store_real.as_ptr(),
            real_store
        );
        let destination = &local.destination.as_ref().unwrap().buffers;
        assert_eq!(destination.sr_opt_oo.as_ptr(), published_oo);
        assert_eq!(destination.sr_opt_o_store.capacity(), 0);
        assert_eq!(destination.sr_opt_o_store_real.capacity(), 0);
        local.state().sr_opt.sr_opt_oo[0] = Complex64::new(-0.0, -0.0);
        local.state().sr_opt.sr_opt_ho[0] = Complex64::new(2.0, -3.0);
        local.state().sr_opt.sr_opt_o[0] = Complex64::new(-0.0, -0.0);
        local.state().sr_opt.sr_opt_o_real[0] = -0.0;
        local.state().sr_opt.sr_opt_o_store[0] = Complex64::new(4.0, 5.0);
        let scratch = local.state().sr_opt.sr_opt_o.as_ptr();
        local.finish();
        assert_eq!(state.sr_opt.sr_opt_oo.as_ptr(), published_oo);
        assert_eq!(state.sr_opt.sr_opt_o_store.as_ptr(), store);
        assert_eq!(state.sr_opt.sr_opt_o_store_real.as_ptr(), real_store);
        assert_eq!(state.sr_opt.sr_opt_o.as_ptr(), scratch);
        assert!(state.sr_opt.sr_opt_o[0].re.is_sign_negative());
        assert!(state.sr_opt.sr_opt_o[0].im.is_sign_negative());
        assert!(state.sr_opt.sr_opt_o_real[0].is_sign_negative());
        assert!(!state.sr_opt.sr_opt_oo[0].re.is_sign_negative());
        assert!(!state.sr_opt.sr_opt_oo[0].im.is_sign_negative());
        assert_eq!(state.sr_opt.sr_opt_ho[0], Complex64::new(2.0, -3.0));
        assert_eq!(state.sr_opt.sr_opt_o_store[0], Complex64::new(4.0, 5.0));
        assert_eq!(
            state
                .sr_local
                .as_ref()
                .unwrap()
                .buffers
                .sr_opt_o_store
                .capacity(),
            0
        );
    }

    #[test]
    fn fixed_shape_reuses_aggregate_allocations_and_reentry_clears_local_scratch() {
        let mut state = state();
        let mut first = SrMeasurement::begin(&mut state);
        first
            .state()
            .sr_opt
            .sr_opt_o
            .fill(Complex64::new(7.0, -8.0));
        first.finish();
        let pointers = cache_pointers(&state);
        let store = state.sr_opt.sr_opt_o_store.as_ptr();
        for _ in 0..3 {
            clear_phys_quantity(&mut state);
            let mut local = SrMeasurement::begin(&mut state);
            assert!(local
                .state()
                .sr_opt
                .sr_opt_o
                .iter()
                .all(|v| *v == Complex64::new(0.0, 0.0)));
            local.state().sr_opt.sr_opt_oo[0] = Complex64::new(1.5, 1.0);
            local.finish();
            assert_eq!(cache_pointers(&state), pointers);
            assert_eq!(state.sr_opt.sr_opt_o_store.as_ptr(), store);
            assert_eq!(state.sr_opt.sr_opt_oo[0], Complex64::new(1.5, 1.0));
        }
        // Pointer reuse is an ownership test, not measured allocation-count proof.
    }

    #[test]
    fn unwind_keeps_partial_local_visible_and_supports_next_measurement() {
        let mut state = state();
        let store = state.sr_opt.sr_opt_o_store.as_ptr();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut local = SrMeasurement::begin(&mut state);
            local.state().sr_opt.sr_opt_oo[0] = Complex64::new(17.0, -19.0);
            local.state().sr_opt.sr_opt_o[0] = Complex64::new(-0.0, -0.0);
            local.state().sr_opt.sr_opt_o_store[0] = Complex64::new(23.0, 29.0);
            panic!("controlled measurement interruption");
        }));
        assert!(result.is_err());
        assert_eq!(state.sr_opt.sr_opt_oo[0], Complex64::new(17.0, -19.0));
        assert_eq!(state.sr_opt.sr_opt_o_store[0], Complex64::new(23.0, 29.0));
        assert!(state.sr_opt.sr_opt_o[0].im.is_sign_negative());
        assert_eq!(state.sr_opt.sr_opt_o_store.as_ptr(), store);
        assert!(state
            .sr_local
            .as_ref()
            .unwrap()
            .buffers
            .sr_opt_o_store
            .is_empty());
        clear_phys_quantity(&mut state);
        let mut next = SrMeasurement::begin(&mut state);
        next.state().sr_opt.sr_opt_oo[0] = Complex64::new(31.0, 37.0);
        next.finish();
        assert_eq!(state.sr_opt.sr_opt_oo[0], Complex64::new(31.0, 37.0));
        assert_eq!(state.sr_opt.sr_opt_o_store.as_ptr(), store);
    }

    #[test]
    fn cache_is_lazy_and_shape_change_replaces_only_aggregate_scratch_layout() {
        let mut state = state();
        state.phys_quantities = Some(crate::state::PhysicalQuantities::zeros(0, 0, 0));
        publish_physcal_in_place(&mut state.sr_opt);
        assert!(state.sr_local.is_none());
        SrMeasurement::begin(&mut state).finish();
        state.sr_opt = SROptData::zeros(4, 2, true);
        let store = state.sr_opt.sr_opt_o_store.as_ptr();
        let mut local = SrMeasurement::begin(&mut state);
        assert_eq!(local.state().sr_opt.sr_opt_size, 4);
        assert!(local.state().sr_opt.sr_opt_o_real.is_empty());
        assert_eq!(local.state().sr_opt.sr_opt_o_store.as_ptr(), store);
        local.finish();
        assert_eq!(state.sr_local.as_ref().unwrap().buffers.sr_opt_size, 4);
    }
}

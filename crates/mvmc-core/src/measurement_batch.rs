//! Sample-batched measurement staging (issue #422, design: `docs/design/gpu-readiness.md` 5.5).
//!
//! The measurement loop of `VMCMainCal` (C `vmccal.c`) visits the `NVMCSample` saved
//! configurations of a chain. Each visit first builds the Pfaffian/inverse tables
//! (`CalculateMAll`) and then consumes them (IP, local energy, Slater derivative, O store).
//! The table construction of one sample does not depend on any other sample, so the loop is
//! restructured into batches of `B` samples:
//!
//! 1. **stage A (batched)**: `CalculateMAll` for the `B` samples, each result kept in its own
//!    batch slot ([`TableSet`]);
//! 2. **stage B (in sample order)**: for each sample of the batch, the slot's tables are
//!    swapped into the state's working tables (a pointer swap, no copy) and the unchanged
//!    per-sample consumers run.
//!
//! Stage B keeps the C accumulation order (energy, `HO`, `OO`, Green sums are all
//! sample-ordered sums), and no RNG is involved, so every output byte is independent of `B`.
//! `B = 1` is exactly the former serial order.
//!
//! # Layout
//!
//! The CPU slots keep the working layout `[n*n + 1, NQP]` per sample, so the consumers read
//! them in place. [`BatchedPlanes`] is the device-facing batched layout; it is produced from
//! the slots by [`BatchedPlanes::pack`] (per-plane `memcpy`) and is not on the CPU hot path:
//! a first version that copied every sample into and out of this layout cost about 2.7 % of
//! wall time on the L32 Hubbard benchmark, so the CPU path swaps slots instead.
//!
//! Column-major with the batch dimension **trailing**, as tenferro expects, so a batch
//! uploads as one contiguous buffer without a transpose:
//!
//! * `planes`: `[n, n, NQP, B]` (no Pfaffian pad slot, unlike the CPU working table
//!   `[n*n + 1, NQP]`), `n = 2 * Ne`;
//! * `pf`: `[NQP, B]`;
//! * `ele_idx`: `[n, B]`.
//!
//! Each `[n, n]` plane is a plain copy of the working table's plane, so converting between
//! the working layout and this layout is a per-plane `memcpy` (no transposition; the batch
//! axis is last, matching `dot_general` batch placement and `from_vec_col_major` semantics).
//! Callers must not hand such a buffer to `from_vec_col_major` with a row-major shape.
//!
//! # Buffer reuse
//!
//! Following the workspace pattern of `AtelierArith/tenferro-decision-rs`
//! (`crates/tenferro-gated-delta/src/workspace.rs`, `crates/jeff-infer/src/host_opt.rs`):
//! the workspace is owned by the optimization state, slots grow to the largest batch seen
//! and are reused across SR steps (a warmed measurement does not allocate batch buffers), and
//! [`MeasurementBatchWorkspace::retained_bytes`] reports the footprint. No einsum plan is
//! cached because no einsum runs on this path (the C summation order is preserved).
//!
//! # Memory
//!
//! The default batch is [`DEFAULT_BATCH`] samples: one extra table set per batch slot, i.e.
//! about one quarter of the Slater table (`(2 Nsite)^2 * NQP`) per slot at half filling.
//! `MVMC_RS_MEASURE_BATCH=<B>` overrides the batch size (an explicit value is not capped).
//!
//! # Coverage
//!
//! All measurement modes use the batched pipeline: real, complex, FSZ real/complex, with
//! Doublon-Holon, RBM, OptTrans, PhysCal Green functions and Lanczos. Stage B is deliberately
//! per-sample in this issue: local energy, Green functions and the Slater derivative consume
//! mutable shared state and are the subject of a later phase.

use num_complex::Complex64;
use tenferro_tensor::TensorScalar;

use crate::state::InvMColMajor;

/// Default number of samples per batch (see the module docs on memory).
pub const DEFAULT_BATCH: usize = 4;

static OVERRIDE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Process-wide batch-size override (`None` restores the environment/default).
///
/// Intended for tests that sweep batch sizes inside one process; zero is treated as `None`.
pub fn set_measurement_batch_size_override(size: Option<usize>) {
    OVERRIDE.store(size.unwrap_or(0), std::sync::atomic::Ordering::SeqCst);
}

/// Batch size for a chain with `n_samples` saved configurations.
pub fn resolve_batch_size(n_samples: usize) -> usize {
    let explicit = match OVERRIDE.load(std::sync::atomic::Ordering::SeqCst) {
        0 => std::env::var("MVMC_RS_MEASURE_BATCH")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&v| v > 0),
        v => Some(v),
    };
    explicit.unwrap_or(DEFAULT_BATCH).min(n_samples.max(1))
}

/// Contiguous batched Pfaffian/inverse planes, `planes[n, n, NQP, B]` and `pf[NQP, B]`.
#[derive(Debug, Default, Clone)]
pub struct BatchedPlanes<T> {
    n: usize,
    n_qp: usize,
    capacity: usize,
    planes: Vec<T>,
    pf: Vec<T>,
}

impl<T: TensorScalar + Default + Copy> BatchedPlanes<T> {
    /// Grow (never shrink) the buffers for `capacity` samples of `n_qp` planes of side `n`.
    pub fn ensure(&mut self, n: usize, n_qp: usize, capacity: usize) {
        self.n = n;
        self.n_qp = n_qp;
        self.capacity = capacity;
        let plane_len = n * n * n_qp * capacity;
        if self.planes.len() < plane_len {
            self.planes.resize(plane_len, T::default());
        }
        let pf_len = n_qp * capacity;
        if self.pf.len() < pf_len {
            self.pf.resize(pf_len, T::default());
        }
    }

    /// Matrix side `n = 2 * Ne`.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Number of QP planes per sample.
    pub fn n_qp(&self) -> usize {
        self.n_qp
    }

    /// Batch capacity in samples.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Bytes held by the buffers.
    pub fn retained_bytes(&self) -> usize {
        (self.planes.len() + self.pf.len()) * std::mem::size_of::<T>()
    }

    /// The `[n, n]` plane of QP `qp` for batch slot `slot` (column-major).
    pub fn plane(&self, slot: usize, qp: usize) -> &[T] {
        let len = self.n * self.n;
        let start = (slot * self.n_qp + qp) * len;
        &self.planes[start..start + len]
    }

    /// The `[NQP]` Pfaffians of batch slot `slot`.
    pub fn pf(&self, slot: usize) -> &[T] {
        &self.pf[slot * self.n_qp..(slot + 1) * self.n_qp]
    }

    /// Whole `[n, n, NQP, B]` buffer (the first `capacity` slots), for a device upload.
    pub fn planes_slice(&self) -> &[T] {
        &self.planes[..self.n * self.n * self.n_qp * self.capacity]
    }

    /// Copy one sample's working tables into batch slot `slot`.
    pub fn store(&mut self, slot: usize, inv: &InvMColMajor<T>, pf: &[T]) {
        debug_assert!(slot < self.capacity);
        let len = self.n * self.n;
        for qp in 0..self.n_qp {
            let start = (slot * self.n_qp + qp) * len;
            self.planes[start..start + len].copy_from_slice(inv.qp_matrix_slice(qp));
        }
        self.pf[slot * self.n_qp..(slot + 1) * self.n_qp].copy_from_slice(&pf[..self.n_qp]);
    }

    /// Copy batch slot `slot` back into the working tables (matrix planes and Pfaffians only).
    pub fn load(&self, slot: usize, inv: &mut InvMColMajor<T>, pf: &mut [T]) {
        debug_assert!(slot < self.capacity);
        for qp in 0..self.n_qp {
            inv.qp_matrix_slice_mut(qp)
                .copy_from_slice(self.plane(slot, qp));
        }
        pf[..self.n_qp].copy_from_slice(self.pf(slot));
    }
}

impl<T: TensorScalar + Default + Copy> BatchedPlanes<T> {
    /// Pack the first `sets.len()` slots into the batched layout (device upload staging).
    pub fn pack(&mut self, sets: &[TableSet<T>]) {
        if let Some(first) = sets.first() {
            self.ensure(first.inv.n_size(), first.inv.n_qp_full(), sets.len());
        }
        for (slot, set) in sets.iter().enumerate() {
            self.store(slot, &set.inv, &set.pf);
        }
    }
}

/// One batch slot: the inverse table and Pfaffians of one saved configuration, in the CPU
/// working layout (`[n*n + 1, NQP]` and `[NQP]`), swappable with the state's tables.
#[derive(Debug)]
pub struct TableSet<T> {
    /// Inverse planes (column-major, with the per-QP pad slot).
    pub inv: InvMColMajor<T>,
    /// One Pfaffian per QP plane.
    pub pf: Vec<T>,
}

impl<T: TensorScalar + Default + Copy> TableSet<T> {
    fn new(n_qp: usize, n_size: usize) -> Self {
        Self {
            inv: InvMColMajor::zeros(n_qp, n_size / 2),
            pf: vec![T::default(); n_qp],
        }
    }

    /// Exchange this slot's tables with the state's working tables.
    pub fn swap_with(&mut self, inv: &mut InvMColMajor<T>, pf: &mut Vec<T>) {
        std::mem::swap(&mut self.inv, inv);
        std::mem::swap(&mut self.pf, pf);
    }
}

/// Grow the slots to `capacity` tables of the given shape (shape changes rebuild them).
fn ensure_slots<T: TensorScalar + Default + Copy>(
    slots: &mut Vec<TableSet<T>>,
    n_qp: usize,
    n_size: usize,
    capacity: usize,
) {
    if slots
        .first()
        .is_some_and(|s| s.inv.n_qp_full() != n_qp || s.inv.n_size() != n_size)
    {
        slots.clear();
    }
    while slots.len() < capacity {
        slots.push(TableSet::new(n_qp, n_size));
    }
}

/// Outcome of stage A for one batch slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotStatus {
    /// Saved configuration is empty/invalid, no table was built.
    Skipped,
    /// `CalculateMAll` failed (C `info != 0`): the sample is not measured.
    Failed,
    /// Tables are stored in the batch.
    Ready,
}

/// Reusable batched measurement buffers, owned by the optimization state.
#[derive(Debug, Default)]
pub struct MeasurementBatchWorkspace {
    /// Real-mode slots (also used for real FSZ).
    pub real: Vec<TableSet<f64>>,
    /// Complex-mode slots.
    pub complex: Vec<TableSet<Complex64>>,
    /// Electron sites `[n, B]`.
    pub ele_idx: Vec<i64>,
    /// Stage-A outcome per slot.
    pub status: Vec<SlotStatus>,
}

impl MeasurementBatchWorkspace {
    /// Size the buffers needed by the selected mode for `capacity` samples.
    pub fn configure(&mut self, n: usize, n_qp: usize, capacity: usize, all_complex: bool) {
        if all_complex {
            ensure_slots(&mut self.complex, n_qp, n, capacity);
        } else {
            ensure_slots(&mut self.real, n_qp, n, capacity);
        }
        if self.ele_idx.len() < n * capacity {
            self.ele_idx.resize(n * capacity, 0);
        }
        self.status.clear();
        self.status.resize(capacity, SlotStatus::Skipped);
    }

    /// Electron sites of batch slot `slot`.
    pub fn ele_idx(&self, slot: usize, n: usize) -> &[i64] {
        &self.ele_idx[slot * n..(slot + 1) * n]
    }

    /// Total bytes retained by the workspace buffers.
    pub fn retained_bytes(&self) -> usize {
        let slot_bytes = |len: usize, pf: usize, elem: usize| (len + pf) * elem;
        let real: usize = self
            .real
            .iter()
            .map(|s| slot_bytes(s.inv.len(), s.pf.len(), std::mem::size_of::<f64>()))
            .sum();
        let complex: usize = self
            .complex
            .iter()
            .map(|s| slot_bytes(s.inv.len(), s.pf.len(), std::mem::size_of::<Complex64>()))
            .sum();
        real + complex + self.ele_idx.len() * std::mem::size_of::<i64>()
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;

    #[test]
    fn store_load_roundtrip_and_layout_is_trailing_batch() {
        let (n_elec, n_qp, cap) = (2usize, 3usize, 4usize);
        let n = 2 * n_elec;
        let mut ws = BatchedPlanes::<f64>::default();
        ws.ensure(n, n_qp, cap);
        let mut sources = Vec::new();
        for slot in 0..cap {
            let mut inv = InvMColMajor::<f64>::zeros(n_qp, n_elec);
            let mut pf = vec![0.0; n_qp];
            for qp in 0..n_qp {
                for (k, v) in inv.qp_matrix_slice_mut(qp).iter_mut().enumerate() {
                    *v = (slot * 1000 + qp * 100 + k) as f64;
                }
                pf[qp] = (slot * 10 + qp) as f64 + 0.5;
            }
            ws.store(slot, &inv, &pf);
            sources.push((inv, pf));
        }
        // [n, n, NQP, B] column-major: slot is the slowest index.
        for slot in 0..cap {
            for qp in 0..n_qp {
                let start = (slot * n_qp + qp) * n * n;
                assert_eq!(&ws.planes_slice()[start..start + n * n], ws.plane(slot, qp));
                assert_eq!(ws.plane(slot, qp), sources[slot].0.qp_matrix_slice(qp));
            }
        }
        for slot in (0..cap).rev() {
            let mut inv = InvMColMajor::<f64>::zeros(n_qp, n_elec);
            let mut pf = vec![0.0; n_qp];
            ws.load(slot, &mut inv, &mut pf);
            assert_eq!(pf, sources[slot].1);
            for qp in 0..n_qp {
                assert_eq!(inv.qp_matrix_slice(qp), sources[slot].0.qp_matrix_slice(qp));
            }
        }
    }

    #[test]
    fn workspace_grows_to_largest_shape_and_reuses_buffers() {
        let mut ws = MeasurementBatchWorkspace::default();
        ws.configure(8, 4, 4, false);
        let big = ws.retained_bytes();
        assert!(big > 0);
        // A smaller request keeps the already-grown slots (n = 8 is not a valid 2*Ne shape
        // change here: the same shape is requested again).
        ws.configure(8, 4, 2, false);
        assert_eq!(ws.retained_bytes(), big);
        ws.configure(8, 4, 4, false);
        assert_eq!(ws.retained_bytes(), big);
    }

    #[test]
    fn pack_matches_slot_tables_and_swap_is_lossless() {
        let (n_elec, n_qp, cap) = (2usize, 3usize, 3usize);
        let n = 2 * n_elec;
        let mut sets: Vec<TableSet<f64>> = (0..cap).map(|_| TableSet::new(n_qp, n)).collect();
        for (slot, set) in sets.iter_mut().enumerate() {
            for qp in 0..n_qp {
                for (k, v) in set.inv.qp_matrix_slice_mut(qp).iter_mut().enumerate() {
                    *v = (slot * 1000 + qp * 100 + k) as f64;
                }
                set.pf[qp] = (slot * 10 + qp) as f64;
            }
        }
        let mut packed = BatchedPlanes::<f64>::default();
        packed.pack(&sets);
        let mut working = InvMColMajor::<f64>::zeros(n_qp, n_elec);
        let mut working_pf = vec![0.0; n_qp];
        sets[1].swap_with(&mut working, &mut working_pf);
        for qp in 0..n_qp {
            assert_eq!(packed.plane(1, qp), working.qp_matrix_slice(qp));
        }
        assert_eq!(packed.pf(1), &working_pf[..]);
    }

    #[test]
    fn batch_size_is_clamped_to_sample_count() {
        assert_eq!(resolve_batch_size(0), 1);
        assert!(resolve_batch_size(2) <= 2);
    }
}

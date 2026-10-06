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
//! # Stage A through the unified stage backend (issue #422, #437)
//!
//! By default stage A calls the C-order kernels (`calc_m_all_*`) per sample. With
//! `MVMC_RS_MEASURE_PF_BACKEND=c-order|tenferro|cuda[:N]` (or
//! [`set_measurement_backend_override`]) the real, non-FSZ stage A instead assembles the
//! skew-symmetric planes of the whole batch (`[n, n, NQP * B]`, the batched layout) and
//! makes ONE call to `PfaffianStages::pfaffian_inverse_batch` of the selected
//! `StageBackend` (see `stage_backend`). The plane assembly is the exact formula of
//! `assemble_inv_m_real` (`X[msj, msi] = -S[rsi, rsj]`), the stored table is mVMC's
//! `invM = -X^-1` (the stage returns the true inverse `X^-1`; `calc_m_all_child_real` applies
//! the same sign flip), and `pf` is stored unchanged. Per-plane failures (zero pivot,
//! non-finite Pfaffian, all-zero plane, out-of-range site) fail the whole sample exactly as a
//! failing `calc_m_all_real` does. `c-order` through the stage trait uses the same PfaPack
//! sequence as `calc_m_all_real` and is byte-identical to it (tested); other backends are
//! validated with bounds. A backend that does not provide the stage (tenferro 0.7.1 has no
//! Pfaffian) or a complex/FSZ mode is a hard error, never a silent fallback.
//!
//! # Coverage
//!
//! All measurement modes use the batched pipeline: real, complex, FSZ real/complex, with
//! Doublon-Holon, RBM, OptTrans, PhysCal Green functions and Lanczos. Stage B is deliberately
//! per-sample in this issue: local energy, Green functions and the Slater derivative consume
//! mutable shared state and are the subject of a later phase.

use num_complex::Complex64;
use tenferro_tensor::TensorScalar;

use crate::stage_backend::{PfaffianStages, PlaneOutcome, StageBackendKind, StageError};
use crate::state::{InvMColMajor, SlaterElmFlat};

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

/// Where stage A gets its Pfaffian/inverse tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeasurementPfaffian {
    /// The C-order `calc_m_all_*` kernels, per sample (default).
    CalcMAll,
    /// One batched `PfaffianStages` call of the selected backend (real, non-FSZ mode only).
    Stage(StageBackendKind),
}

/// Environment variable selecting the measurement Pfaffian source (`calc-m-all` (default),
/// `c-order`, `tenferro`, `cuda[:N]`).
pub const MEASURE_PF_VARIABLE: &str = "MVMC_RS_MEASURE_PF_BACKEND";

/// Parse a [`MEASURE_PF_VARIABLE`] value.
pub fn parse_measurement_pfaffian(value: &str) -> Result<MeasurementPfaffian, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "" | "calc-m-all" | "default" => Ok(MeasurementPfaffian::CalcMAll),
        other => crate::stage_backend::parse_stage_backend(other)
            .map(MeasurementPfaffian::Stage)
            .map_err(|_| {
                format!(
                    "{MEASURE_PF_VARIABLE}={other:?} is not one of calc-m-all, c-order, tenferro, cuda[:N]"
                )
            }),
    }
}

static MEASURE_OVERRIDE: std::sync::Mutex<Option<MeasurementPfaffian>> =
    std::sync::Mutex::new(None);

/// Process-wide override for tests (`None` restores the environment/default).
pub fn set_measurement_backend_override(source: Option<MeasurementPfaffian>) {
    *MEASURE_OVERRIDE.lock().unwrap_or_else(|e| e.into_inner()) = source;
}

/// The measurement Pfaffian source: override, then environment, then the C-order default.
/// An invalid environment value is a hard error.
pub fn selected_measurement_pfaffian() -> MeasurementPfaffian {
    if let Some(source) = *MEASURE_OVERRIDE.lock().unwrap_or_else(|e| e.into_inner()) {
        return source;
    }
    match std::env::var(MEASURE_PF_VARIABLE) {
        Ok(v) => parse_measurement_pfaffian(&v).unwrap_or_else(|e| panic!("{e}")),
        Err(_) => MeasurementPfaffian::CalcMAll,
    }
}

/// Smallest squared plane entry that `calc_m_all_real` accepts (its `MIN_ABS2`).
const MIN_ABS2: f64 = 1.0e-28;

/// Stage A of the real, non-FSZ measurement through a [`PfaffianStages`] backend.
///
/// `active` lists the batch slots that need tables (their `ele_idx` rows are filled). One
/// batched call evaluates all `active.len() * n_qp` planes (`[n, n, NQP * B]` order, plane
/// `a * n_qp + qp`). On return every active slot is [`SlotStatus::Ready`] with its tables in
/// the working layout (`invM = -X^-1`, `pf`), or [`SlotStatus::Failed`] when any of its planes
/// failed, like a failing `calc_m_all_real`.
///
/// # Errors
///
/// The backend's [`StageError`] (for example `Unsupported`); slot statuses are then
/// unspecified.
pub fn stage_a_pfaffian(
    backend: &mut dyn PfaffianStages,
    slater: &SlaterElmFlat<f64>,
    batch: &mut MeasurementBatchWorkspace,
    active: &[usize],
    n_site: usize,
    n_elec: usize,
    n_qp: usize,
) -> Result<(), StageError> {
    let n = 2 * n_elec;
    let n2 = 2 * n_site;
    let nn = n * n;
    if active.is_empty() {
        return Ok(());
    }
    if n == 0 || n_qp == 0 {
        for &slot in active {
            batch.status[slot] = SlotStatus::Failed;
        }
        return Ok(());
    }
    let planes = active.len() * n_qp;
    let mut x = std::mem::take(&mut batch.stage_x);
    x.clear();
    x.resize(nn * planes, 0.0);
    let src = slater.as_slice();
    let mut sample_ok = vec![true; active.len()];
    let mut cols = vec![0usize; n];
    for (a, &slot) in active.iter().enumerate() {
        let ele = &batch.ele_idx[slot * n..(slot + 1) * n];
        let mut valid = true;
        for (msi, c) in cols.iter_mut().enumerate() {
            let rsi = ele[msi] + ((msi / n_elec) as i64) * (n_site as i64);
            if rsi < 0 || rsi >= n2 as i64 {
                valid = false;
                break;
            }
            *c = rsi as usize;
        }
        if !valid {
            sample_ok[a] = false;
            continue;
        }
        for qp in 0..n_qp {
            let plane = &mut x[(a * n_qp + qp) * nn..(a * n_qp + qp + 1) * nn];
            let mut max_abs2 = 0.0_f64;
            // Same visiting order and formula as `assemble_inv_m_real`.
            for (&rsi, out) in cols.iter().zip(plane.chunks_exact_mut(n)) {
                let start = (qp * n2 + rsi) * n2;
                let row = &src[start..start + n2];
                for (o, &rsj) in out.iter_mut().zip(&cols) {
                    let v = -row[rsj];
                    *o = v;
                    max_abs2 = max_abs2.max(v * v);
                }
            }
            if max_abs2 < MIN_ABS2 {
                sample_ok[a] = false;
            }
        }
    }
    let result = backend.pfaffian_inverse_batch(&x, n, planes);
    batch.stage_x = x;
    let out = result?;
    for (a, &slot) in active.iter().enumerate() {
        let planes_ok = (0..n_qp).all(|qp| out.outcome[a * n_qp + qp] == PlaneOutcome::Ok);
        if !sample_ok[a] || !planes_ok {
            batch.status[slot] = SlotStatus::Failed;
            continue;
        }
        let set = &mut batch.real[slot];
        for qp in 0..n_qp {
            let p = a * n_qp + qp;
            // mVMC invM = -X^-1 (`M_DSCAL(&nsq, &minus_one, invM, &one)`).
            for (dst, &v) in set
                .inv
                .qp_matrix_slice_mut(qp)
                .iter_mut()
                .zip(&out.inv[p * nn..(p + 1) * nn])
            {
                *dst = -v;
            }
            set.pf[qp] = out.pf[p];
        }
        batch.status[slot] = SlotStatus::Ready;
    }
    Ok(())
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
    /// Reusable `[n, n, NQP * B]` plane buffer of the stage-backend path.
    pub stage_x: Vec<f64>,
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
        real + complex
            + self.ele_idx.len() * std::mem::size_of::<i64>()
            + self.stage_x.len() * std::mem::size_of::<f64>()
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

    use crate::stage_backend::{COrderPfaffian, PfInvBatch};
    use crate::state::ThreadedPfaPackWorkspace;

    /// Wraps the C-order stage: scales every Pfaffian by `1 + rel` and marks the planes in
    /// `fail` as zero-pivot, to prove the routing and the failure handling.
    struct Tweaked {
        rel: f64,
        fail: Vec<usize>,
    }

    impl PfaffianStages for Tweaked {
        fn label(&self) -> String {
            "tweaked".into()
        }
        fn provider(&self) -> String {
            "test".into()
        }
        fn pfaffian_inverse_batch(
            &mut self,
            x: &[f64],
            n: usize,
            planes: usize,
        ) -> Result<PfInvBatch, StageError> {
            let mut out = COrderPfaffian.pfaffian_inverse_batch(x, n, planes)?;
            for (p, pf) in out.pf.iter_mut().enumerate() {
                *pf *= 1.0 + self.rel;
                if self.fail.contains(&p) {
                    out.outcome[p] = PlaneOutcome::ZeroPivot { row: 1 };
                }
            }
            Ok(out)
        }
    }

    fn skew_slater(n_qp: usize, n_site: usize) -> SlaterElmFlat<f64> {
        let n2 = 2 * n_site;
        let mut slater = SlaterElmFlat::<f64>::zeros(n_qp, n_site);
        let mut state = 12345u64;
        for qp in 0..n_qp {
            let plane = slater.qp_slice_mut(qp);
            for i in 0..n2 {
                for j in 0..i {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let v = ((state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5;
                    plane[i * n2 + j] = v;
                    plane[j * n2 + i] = -v;
                }
            }
        }
        slater
    }

    fn setup(
        n_site: usize,
        n_elec: usize,
        n_qp: usize,
        configs: &[Vec<i64>],
    ) -> MeasurementBatchWorkspace {
        let mut batch = MeasurementBatchWorkspace::default();
        batch.configure(2 * n_elec, n_qp, configs.len(), false);
        for (slot, ele) in configs.iter().enumerate() {
            batch.ele_idx[slot * 2 * n_elec..(slot + 1) * 2 * n_elec].copy_from_slice(ele);
            batch.status[slot] = SlotStatus::Ready;
        }
        let _ = n_site;
        batch
    }

    #[test]
    fn stage_a_matches_calc_m_all_real_and_applies_the_invm_sign() {
        let (n_site, n_elec, n_qp) = (4usize, 2usize, 3usize);
        let slater = skew_slater(n_qp, n_site);
        let configs = vec![vec![0, 2, 1, 3], vec![3, 1, 0, 2]];
        let mut batch = setup(n_site, n_elec, n_qp, &configs);
        stage_a_pfaffian(
            &mut COrderPfaffian,
            &slater,
            &mut batch,
            &[0, 1],
            n_site,
            n_elec,
            n_qp,
        )
        .unwrap();
        let pool = ThreadedPfaPackWorkspace::new(2 * n_elec, 1);
        for (slot, ele) in configs.iter().enumerate() {
            assert_eq!(batch.status[slot], SlotStatus::Ready);
            let mut inv = InvMColMajor::<f64>::zeros(n_qp, n_elec);
            let mut pf = vec![0.0; n_qp];
            crate::pfaffian::calc_m_all_real(
                ele, &slater, &mut inv, &mut pf, 0, n_qp, n_site, n_elec, &pool,
            )
            .unwrap();
            for qp in 0..n_qp {
                assert_eq!(batch.real[slot].pf[qp], pf[qp]);
                assert_eq!(
                    batch.real[slot].inv.qp_matrix_slice(qp),
                    inv.qp_matrix_slice(qp),
                    "slot {slot} qp {qp}"
                );
            }
        }
    }

    #[test]
    fn stage_a_failure_of_one_plane_fails_only_its_sample() {
        let (n_site, n_elec, n_qp) = (4usize, 2usize, 2usize);
        let slater = skew_slater(n_qp, n_site);
        let configs = vec![vec![0, 2, 1, 3], vec![3, 1, 0, 2], vec![1, 3, 0, 2]];
        let mut batch = setup(n_site, n_elec, n_qp, &configs);
        // plane index = a * n_qp + qp: plane 3 is sample 1, qp 1
        stage_a_pfaffian(
            &mut Tweaked {
                rel: 0.0,
                fail: vec![3],
            },
            &slater,
            &mut batch,
            &[0, 1, 2],
            n_site,
            n_elec,
            n_qp,
        )
        .unwrap();
        assert_eq!(
            batch.status[..3],
            [SlotStatus::Ready, SlotStatus::Failed, SlotStatus::Ready]
        );
    }

    #[test]
    fn stage_a_uses_the_backend_values_and_reports_unsupported() {
        let (n_site, n_elec, n_qp) = (4usize, 2usize, 2usize);
        let slater = skew_slater(n_qp, n_site);
        let configs = vec![vec![0, 2, 1, 3]];
        let mut exact = setup(n_site, n_elec, n_qp, &configs);
        stage_a_pfaffian(
            &mut COrderPfaffian,
            &slater,
            &mut exact,
            &[0],
            n_site,
            n_elec,
            n_qp,
        )
        .unwrap();
        let mut tweaked = setup(n_site, n_elec, n_qp, &configs);
        stage_a_pfaffian(
            &mut Tweaked {
                rel: 1e-6,
                fail: vec![],
            },
            &slater,
            &mut tweaked,
            &[0],
            n_site,
            n_elec,
            n_qp,
        )
        .unwrap();
        for qp in 0..n_qp {
            let (a, b) = (exact.real[0].pf[qp], tweaked.real[0].pf[qp]);
            assert!((b - a * (1.0 + 1e-6)).abs() <= 1e-15 * a.abs().max(1.0));
        }
        let mut unsupported = setup(n_site, n_elec, n_qp, &configs);
        let err = stage_a_pfaffian(
            &mut crate::stage_backend::UnsupportedPfaffian::tenferro(),
            &slater,
            &mut unsupported,
            &[0],
            n_site,
            n_elec,
            n_qp,
        )
        .unwrap_err();
        assert!(matches!(err, StageError::Unsupported(_)));
    }

    #[test]
    fn measurement_selector_parses() {
        assert_eq!(
            parse_measurement_pfaffian(""),
            Ok(MeasurementPfaffian::CalcMAll)
        );
        assert_eq!(
            parse_measurement_pfaffian("c-order"),
            Ok(MeasurementPfaffian::Stage(StageBackendKind::COrder))
        );
        assert_eq!(
            parse_measurement_pfaffian("cuda:1"),
            Ok(MeasurementPfaffian::Stage(StageBackendKind::Cuda(1)))
        );
        assert!(parse_measurement_pfaffian("nope").is_err());
    }

    /// Stage A writes whole planes through the revalidating full-overwrite writer, so a slot
    /// table left stale by a device run (issue #454) is valid again afterwards and readable.
    #[test]
    fn stage_a_overwrites_and_revalidates_stale_slot_tables() {
        let (n_site, n_elec, n_qp) = (4usize, 2usize, 2usize);
        let slater = skew_slater(n_qp, n_site);
        let configs = vec![vec![0, 2, 1, 3]];
        let mut batch = setup(n_site, n_elec, n_qp, &configs);
        batch.real[0]
            .inv
            .mark_stale("test: device-resident sampler");
        assert!(batch.real[0].inv.is_stale());
        stage_a_pfaffian(
            &mut COrderPfaffian,
            &slater,
            &mut batch,
            &[0],
            n_site,
            n_elec,
            n_qp,
        )
        .unwrap();
        assert_eq!(batch.status[0], SlotStatus::Ready);
        assert!(!batch.real[0].inv.is_stale());
        let _ = batch.real[0].inv.qp_matrix_slice(0); // would panic if still stale
    }
}

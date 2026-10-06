//! [`PfaffianStages`] over the batched backends: the Pfaffian slot of the unified
//! `mvmc_core::stage_backend::StageBackend` (issues #437, #424).
//!
//! Convention. The harness oracle (`COrderPfaffian`) returns the true inverse `X^-1`
//! (`X * inv = I`), the output of `utu2inv` without the final sign flip. This crate's `inv`
//! has the same convention, so the mapping is the identity. mVMC's `invM` (C
//! `CalculateMAll`, Rust `calc_m_all_real`) is `-X^-1`: those routines finish with
//! `M_DSCAL(&nsq, &minus_one, invM, &one)` (`mvmc_core::pfaffian`,
//! `calc_m_all_child_real`), so `invM = -inv` and `pf` is unchanged.
//!
//! A call with `planes` matrices is one batch of `planes` planes (`NQP = planes, B = 1`).

use mvmc_core::stage_backend::{
    PfInvBatch, PfaffianStages, PlaneOutcome, StageBackend, StageError,
};

use crate::{pfaffian_inverse_batched, Backend, PlaneStatus};

/// Pfaffian stage backed by [`pfaffian_inverse_batched`] on `backend`.
pub struct BatchedStages<'a> {
    backend: Backend<'a>,
    label: String,
}

impl<'a> BatchedStages<'a> {
    /// Stage on `backend`, reported as `label` (e.g. `batched-cpu-pfapack`).
    pub fn new(backend: Backend<'a>, label: &str) -> Self {
        Self {
            backend,
            label: label.to_string(),
        }
    }

    /// The C-order SR stages (`COrderSr`) composed with this Pfaffian stage in one
    /// [`StageBackend`] labelled like the stage.
    pub fn into_stage_backend(self) -> StageBackend<'a> {
        let label = self.label.clone();
        StageBackend::c_order().with_pfaffian(&label, Box::new(self))
    }
}

impl PfaffianStages for BatchedStages<'_> {
    fn label(&self) -> String {
        self.label.clone()
    }

    fn provider(&self) -> String {
        format!(
            "mvmc-gpu pfaffian_inverse_batched, backend {:?}",
            self.backend
        )
    }

    fn pfaffian_inverse_batch(
        &mut self,
        x: &[f64],
        n: usize,
        planes: usize,
    ) -> Result<PfInvBatch, StageError> {
        let out = pfaffian_inverse_batched(&self.backend, x, n, planes, 1)
            .map_err(|e| StageError::Failed(e.to_string()))?;
        Ok(PfInvBatch {
            pf: out.pf,
            inv: out.inv,
            outcome: out
                .status
                .into_iter()
                .map(|s| match s {
                    PlaneStatus::Ok => PlaneOutcome::Ok,
                    PlaneStatus::ZeroPivot { row } => PlaneOutcome::ZeroPivot { row },
                    PlaneStatus::NonFinite => PlaneOutcome::NonFinite,
                })
                .collect(),
        })
    }
}

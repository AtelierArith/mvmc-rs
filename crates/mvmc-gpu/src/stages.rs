//! [`AcceleratedStages::pfaffian_inverse`] over the batched backends, for the validation
//! harness of issue #424 (`mvmc_core::accel_validation`).
//!
//! Convention. The harness oracle (`CpuOracle`) returns the true inverse `X^-1`
//! (`X * inv = I`), the output of `utu2inv` without the final sign flip. This crate's `inv`
//! has the same convention, so the mapping is the identity. mVMC's `invM` (C
//! `CalculateMAll`, Rust `calc_m_all_real`) is `-X^-1`: those routines finish with
//! `M_DSCAL(&nsq, &minus_one, invM, &one)` (`mvmc_core::pfaffian`,
//! `calc_m_all_child_real`), so `invM = -inv` and `pf` is unchanged.
//!
//! The harness evaluates one matrix per call, so each call is a batch of one plane
//! (`NQP = 1, B = 1`). The SR stage is not provided here and reports `Unsupported`
//! (never a CPU fallback).

use mvmc_core::accel_validation::{AcceleratedStages, PfInv, SrSg, StageError};

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
}

impl AcceleratedStages for BatchedStages<'_> {
    fn label(&self) -> String {
        self.label.clone()
    }
    fn provider(&self) -> String {
        format!(
            "mvmc-gpu pfaffian_inverse_batched, backend {:?}",
            self.backend
        )
    }

    fn pfaffian_inverse(&mut self, x: &[f64], n: usize) -> Result<PfInv, StageError> {
        let out = pfaffian_inverse_batched(&self.backend, x, n, 1, 1)
            .map_err(|e| StageError::Failed(e.to_string()))?;
        match out.status[0] {
            PlaneStatus::Ok => Ok(PfInv {
                pf: out.pf[0],
                inv: out.inv,
            }),
            PlaneStatus::ZeroPivot { row } => {
                Err(StageError::Failed(format!("zero pivot at {row}")))
            }
            PlaneStatus::NonFinite => Err(StageError::Failed("non-finite Pfaffian".into())),
        }
    }

    fn sr_s_g(
        &mut self,
        _o: &[f64],
        _nsample: usize,
        _npara: usize,
        _e: &[f64],
        _w: &[f64],
    ) -> Result<SrSg, StageError> {
        Err(StageError::Unsupported(
            "the batched Pfaffian backends provide no SR stage".to_string(),
        ))
    }
}

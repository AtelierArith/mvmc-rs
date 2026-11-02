//! Ownership of an execution Engine and its optional CPU worker reservation.

use std::ops::{Deref, DerefMut};
use tenferro_runtime::ExecutionEngine;

use crate::cpu_worker_budget::{ReservedResource, WorkerReservation};

/// An Engine together with its resource accounting lease.
///
/// The Engine is destroyed before capacity is returned. CUDA and externally
/// supplied Engines can use [`Self::new`] without a CPU reservation.
pub struct EngineResource {
    owned: ReservedResource<Box<dyn ExecutionEngine + Send>>,
}

impl EngineResource {
    /// Own an externally supplied Engine without reserving CPU workers.
    pub fn new(engine: Box<dyn ExecutionEngine + Send>) -> Self {
        Self {
            owned: ReservedResource::new(engine, None),
        }
    }

    pub(crate) fn reserved(
        engine: Box<dyn ExecutionEngine + Send>,
        reservation: WorkerReservation,
    ) -> Self {
        Self {
            owned: ReservedResource::new(engine, Some(reservation)),
        }
    }
}

impl Deref for EngineResource {
    type Target = Box<dyn ExecutionEngine + Send>;

    fn deref(&self) -> &Self::Target {
        &self.owned.resource
    }
}

impl DerefMut for EngineResource {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.owned.resource
    }
}

//! Nonblocking accounting for additional Engine workers (#583).
//!
//! Callers and other execution pools require separate accounting. A reservation must
//! live until its pool has been destroyed, including on failed construction/unwind.

use std::sync::{Arc, Mutex};

#[derive(Debug)]
pub(crate) struct CpuWorkerBudget {
    capacity: usize,
    used: Mutex<usize>,
}

impl CpuWorkerBudget {
    pub(crate) fn new(capacity: usize) -> Arc<Self> {
        Arc::new(Self {
            capacity,
            used: Mutex::new(0),
        })
    }

    /// Reserve the complete request or return immediately without consuming capacity.
    /// Never wait for capacity: the holder may be blocked in a walker collective.
    pub(crate) fn try_reserve(self: &Arc<Self>, workers: usize) -> Option<WorkerReservation> {
        let mut used = self.used.lock().unwrap_or_else(|e| e.into_inner());
        if workers > self.capacity - *used {
            return None;
        }
        *used += workers;
        Some(WorkerReservation {
            budget: self.clone(),
            workers,
        })
    }
}

#[derive(Debug)]
pub(crate) struct WorkerReservation {
    budget: Arc<CpuWorkerBudget>,
    workers: usize,
}

impl Drop for WorkerReservation {
    fn drop(&mut self) {
        let mut used = self.budget.used.lock().unwrap_or_else(|e| e.into_inner());
        *used -= self.workers;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failure_does_not_consume_capacity_and_drop_restores_it() {
        let budget = CpuWorkerBudget::new(3);
        let first = budget.try_reserve(2).unwrap();
        assert!(budget.try_reserve(2).is_none());
        let last = budget.try_reserve(1).unwrap();
        assert!(budget.try_reserve(1).is_none());
        drop(first);
        assert!(budget.try_reserve(2).is_some());
        drop(last);
        assert!(budget.try_reserve(3).is_some());
        assert!(budget.try_reserve(usize::MAX).is_none());
    }

    #[test]
    fn unwind_releases_reservation() {
        let budget = CpuWorkerBudget::new(3);
        let result = std::panic::catch_unwind({
            let budget = budget.clone();
            move || {
                let _reservation = budget.try_reserve(3).unwrap();
                panic!("simulated pool construction failure");
            }
        });
        assert!(result.is_err());
        assert!(budget.try_reserve(3).is_some());
    }

    #[test]
    fn concurrent_holders_cannot_exceed_capacity() {
        let budget = CpuWorkerBudget::new(3);
        let barrier = Arc::new(std::sync::Barrier::new(8));
        let successful = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8)
                .map(|_| {
                    let budget = budget.clone();
                    let barrier = barrier.clone();
                    scope.spawn(move || {
                        let held = budget.try_reserve(1);
                        barrier.wait();
                        held.is_some()
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| usize::from(h.join().unwrap()))
                .sum::<usize>()
        });
        assert_eq!(successful, 3);
        assert!(budget.try_reserve(3).is_some());
    }
}

/// Own the pool before its accounting lease so field drop order destroys the pool
/// before returning capacity. This also covers early returns and unwinding.
pub(crate) struct ReservedResource<T> {
    pub(crate) resource: T,
    _reservation: Option<WorkerReservation>,
}

impl<T> ReservedResource<T> {
    pub(crate) fn new(resource: T, reservation: Option<WorkerReservation>) -> Self {
        Self {
            resource,
            _reservation: reservation,
        }
    }
}

#[cfg(test)]
mod lifetime_tests {
    use super::*;

    struct AssertStillReserved(Arc<CpuWorkerBudget>);

    impl Drop for AssertStillReserved {
        fn drop(&mut self) {
            assert!(
                self.0.try_reserve(1).is_none(),
                "capacity released before pool destruction"
            );
        }
    }

    #[test]
    fn resource_is_destroyed_before_capacity_is_returned() {
        let budget = CpuWorkerBudget::new(1);
        let reservation = budget.try_reserve(1).unwrap();
        let owner = ReservedResource::new(AssertStillReserved(budget.clone()), Some(reservation));
        let _ = &owner.resource;
        drop(owner);
        assert!(budget.try_reserve(1).is_some());
    }
}

//! Runtime control for deterministic shared-memory inner kernels.

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use num_complex::Complex64;
use rayon::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};

/// Observed kernel entries, not requested pool capacity or install calls.
/// Item counts include entries attempted before a kernel returns an error.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ExecutionSnapshot {
    /// Parallel kernel invocations with at least one entered item.
    pub parallel_calls: usize,
    /// Serial kernel invocations with at least one entered item.
    pub serial_calls: usize,
    /// Total attempted QP child entries.
    pub executed_qp_items: usize,
    /// QP entries actually dispatched through the inner pool.
    pub parallel_qp_items: usize,
    /// QP entries on the serial dispatch path.
    pub serial_qp_items: usize,
    /// Total attempted Transfer term entries.
    pub executed_term_items: usize,
    /// Transfer entries actually dispatched through the inner pool.
    pub parallel_term_items: usize,
    /// Transfer entries on the serial dispatch path.
    pub serial_term_items: usize,
    /// Independent copy/SR entries actually dispatched through the inner pool.
    pub parallel_entry_items: usize,
    /// Independent copy/SR entries on the serial dispatch path.
    pub serial_entry_items: usize,
    /// Observed item entries on named inner-pool workers (not capacity).
    pub worker_entries: usize,
    /// Number of different named inner-pool workers observed.
    pub distinct_workers: usize,
    /// Indices of actual `mvmc-inner-*` workers that entered observed kernels.
    pub worker_ids: Vec<usize>,
}

#[derive(Default)]
struct ExecutionCounters {
    calls: [AtomicUsize; 2],
    items: [[AtomicUsize; 3]; 2],
    worker_entries: AtomicUsize,
    workers: Mutex<BTreeSet<usize>>,
}

impl ExecutionCounters {
    fn snapshot(&self) -> ExecutionSnapshot {
        let get = |parallel: usize, kind: usize| self.items[parallel][kind].load(Ordering::Relaxed);
        let worker_ids: Vec<_> = self.workers.lock().unwrap().iter().copied().collect();
        ExecutionSnapshot {
            serial_calls: self.calls[0].load(Ordering::Relaxed),
            parallel_calls: self.calls[1].load(Ordering::Relaxed),
            serial_qp_items: get(0, 0),
            parallel_qp_items: get(1, 0),
            executed_qp_items: get(0, 0) + get(1, 0),
            serial_term_items: get(0, 1),
            parallel_term_items: get(1, 1),
            executed_term_items: get(0, 1) + get(1, 1),
            serial_entry_items: get(0, 2),
            parallel_entry_items: get(1, 2),
            worker_entries: self.worker_entries.load(Ordering::Relaxed),
            distinct_workers: worker_ids.len(),
            worker_ids,
        }
    }
}

thread_local! {
    static OBSERVER: RefCell<Option<Arc<ExecutionCounters>>> = const { RefCell::new(None) };
}

/// Opt-in, per-run observation. This guard must stay on its initiating thread.
/// Finish after synchronous kernels return, before unrelated capacity probes.
/// Nested observation scopes must be dropped in reverse creation order.
pub struct ObservationGuard {
    counters: Arc<ExecutionCounters>,
    _binding: ObserverBinding,
}

impl ObservationGuard {
    /// Snapshot completed synchronous work and restore the prior observation.
    pub fn finish(self) -> ExecutionSnapshot {
        self.counters.snapshot()
    }
}

/// Begin observational counters without changing configuration, RNG or results.
/// Dropping the guard (including error/panic paths) restores the prior scope.
pub fn start_observation() -> ObservationGuard {
    let counters = Arc::new(ExecutionCounters::default());
    ObservationGuard {
        _binding: bind_observer(Some(counters.clone())),
        counters,
    }
}

pub(crate) struct ObserverBinding {
    previous: Option<Arc<ExecutionCounters>>,
    _same_thread: PhantomData<Rc<()>>,
}

impl Drop for ObserverBinding {
    fn drop(&mut self) {
        OBSERVER.with(|slot| *slot.borrow_mut() = self.previous.take());
    }
}

fn bind_observer(observer: Option<Arc<ExecutionCounters>>) -> ObserverBinding {
    ObserverBinding {
        previous: OBSERVER.with(|slot| slot.replace(observer)),
        _same_thread: PhantomData,
    }
}

#[derive(Clone, Copy)]
pub(crate) enum ObservedWork {
    Qp,
    Term,
    Entry,
}

pub(crate) struct KernelObservation {
    counters: Option<Arc<ExecutionCounters>>,
    parallel: bool,
    kind: ObservedWork,
    entered: AtomicBool,
}

pub(crate) fn observe_kernel(kind: ObservedWork, parallel: bool) -> KernelObservation {
    KernelObservation {
        counters: OBSERVER.with(|slot| slot.borrow().clone()),
        parallel,
        kind,
        entered: AtomicBool::new(false),
    }
}

impl KernelObservation {
    /// Call inside the actual item closure, never for provisioning or install.
    pub(crate) fn enter_item(&self) -> Option<ObserverBinding> {
        let counters = self.counters.as_ref()?;
        let parallel = usize::from(self.parallel);
        if !self.entered.swap(true, Ordering::Relaxed) {
            counters.calls[parallel].fetch_add(1, Ordering::Relaxed);
        }
        counters.items[parallel][self.kind as usize].fetch_add(1, Ordering::Relaxed);
        if self.parallel {
            let thread = std::thread::current();
            if thread
                .name()
                .is_some_and(|name| name.starts_with("mvmc-inner-"))
            {
                if let Some(worker) = rayon::current_thread_index() {
                    counters.worker_entries.fetch_add(1, Ordering::Relaxed);
                    counters.workers.lock().unwrap().insert(worker);
                }
            }
        }
        Some(bind_observer(Some(counters.clone())))
    }
}

/// Default minimum number of independent QP items before inner threading is
/// worthwhile. The sequential path remains the default for small workloads.
pub const DEFAULT_INNER_THRESHOLD: usize = 32;

/// Runtime configuration for independent inner work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InnerThreadConfig {
    /// Number of worker threads requested by `MVMC_RS_INNER_THREADS`.
    pub threads: usize,
    /// Minimum work items required to enable parallel execution.
    pub threshold: usize,
}

/// Read the process-wide inner-kernel controls once.
///
/// `MVMC_RS_INNER_THREADS` defaults to one, preserving the historical
/// sequential execution and RNG/Markov-chain contract. `MVMC_RS_INNER_THRESHOLD`
/// defaults to [`DEFAULT_INNER_THRESHOLD`]. Invalid or zero values fall back
/// to those defaults.
pub fn inner_thread_config() -> InnerThreadConfig {
    static CONFIG: OnceLock<InnerThreadConfig> = OnceLock::new();
    *CONFIG.get_or_init(|| InnerThreadConfig {
        threads: std::env::var("MVMC_RS_INNER_THREADS")
            .ok()
            .and_then(|value| value.parse().ok())
            .filter(|&value: &usize| value > 0)
            .unwrap_or(1),
        threshold: std::env::var("MVMC_RS_INNER_THRESHOLD")
            .ok()
            .and_then(|value| value.parse().ok())
            .filter(|&value: &usize| value > 0)
            .unwrap_or(DEFAULT_INNER_THRESHOLD),
    })
}

/// Return whether a range should use the shared-memory worker pool.
pub fn inner_parallel_enabled(work_items: usize) -> bool {
    let config = inner_thread_config();
    config.threads > 1 && work_items >= config.threshold
}

/// Number of workers to provision for a work range.
pub fn inner_worker_count(work_items: usize) -> usize {
    if inner_parallel_enabled(work_items) {
        inner_thread_config().threads.min(work_items)
    } else {
        1
    }
}

/// Execute a closure on the configured deterministic Rayon pool.
pub fn install<R: Send>(operation: impl FnOnce() -> R + Send) -> R {
    static POOL: OnceLock<ThreadPool> = OnceLock::new();
    let pool = POOL.get_or_init(|| {
        ThreadPoolBuilder::new()
            .num_threads(inner_thread_config().threads.max(1))
            .thread_name(|index| format!("mvmc-inner-{index}"))
            .build()
            .expect("inner Rayon pool must build")
    });
    let observer = OBSERVER.with(|slot| slot.borrow().clone());
    pool.install(|| {
        let _binding = bind_observer(observer);
        operation()
    })
}

/// Visit independent output entries; reductions inside an entry remain serial.
pub fn for_each_mut<T: Send>(items: &mut [T], operation: impl Fn(usize, &mut T) + Send + Sync) {
    let observed = observe_kernel(ObservedWork::Entry, inner_parallel_enabled(items.len()));
    let operation = |i, value| {
        let _entry = observed.enter_item();
        operation(i, value);
    };
    if inner_parallel_enabled(items.len()) {
        install(|| {
            items
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, value)| operation(i, value))
        });
    } else {
        items
            .iter_mut()
            .enumerate()
            .for_each(|(i, value)| operation(i, value));
    }
}

/// Visit matching independent entries without aliasing either output slice.
pub fn for_each_pair_mut<A: Send, B: Send>(
    left: &mut [A],
    right: &mut [B],
    operation: impl Fn(usize, &mut A, &mut B) + Send + Sync,
) {
    assert_eq!(left.len(), right.len());
    let observed = observe_kernel(ObservedWork::Entry, inner_parallel_enabled(left.len()));
    let operation = |i, a, b| {
        let _entry = observed.enter_item();
        operation(i, a, b);
    };
    if inner_parallel_enabled(left.len()) {
        install(|| {
            left.par_iter_mut()
                .zip(right.par_iter_mut())
                .enumerate()
                .for_each(|(i, (a, b))| operation(i, a, b))
        });
    } else {
        left.iter_mut()
            .zip(right)
            .enumerate()
            .for_each(|(i, (a, b))| operation(i, a, b));
    }
}

/// Copy the common prefix with disjoint writes; the destination tail is untouched.
pub fn copy_real_to_complex(dst: &mut [Complex64], src: &[f64]) {
    let n = dst.len().min(src.len());
    for_each_mut(&mut dst[..n], |i, value| {
        *value = Complex64::new(src[i], 0.0)
    });
}

/// Copy real parts of the common prefix, preserving the destination tail.
pub fn copy_complex_realpart(dst: &mut [f64], src: &[Complex64]) {
    let n = dst.len().min(src.len());
    for_each_mut(&mut dst[..n], |i, value| *value = src[i].re);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequential_default_is_safe_for_small_work() {
        // Environment-independent assertion: a zero-sized range never needs
        // a worker, even when a caller configured multiple threads.
        assert_eq!(inner_worker_count(0), 1);
    }

    #[test]
    fn observation_scopes_restore_after_nested_install_panics_and_errors() {
        let outer = start_observation();
        install(|| install(|| ()));
        for_each_mut::<u8>(&mut [], |_, _| panic!("empty range must not enter"));
        assert_eq!(outer.counters.snapshot(), ExecutionSnapshot::default());
        let failed = std::panic::catch_unwind(|| {
            install(|| {
                let _nested = start_observation();
                let task = observe_kernel(ObservedWork::Qp, true);
                let _entry = task.enter_item();
                assert!(rayon::current_thread_index().is_some());
                panic!("observation cleanup probe");
            });
        });
        assert!(failed.is_err());
        let task = observe_kernel(ObservedWork::Term, false);
        {
            let _entry = task.enter_item();
        }
        let snapshot = outer.finish();
        assert_eq!(snapshot.executed_qp_items, 0, "nested scope must not leak");
        assert_eq!(snapshot.serial_term_items, 1);
        assert_eq!(snapshot.parallel_calls, 0);
        assert!(OBSERVER.with(|slot| slot.borrow().is_none()));
        let disabled = observe_kernel(ObservedWork::Qp, true);
        assert!(disabled.counters.is_none());
        assert!(disabled.enter_item().is_none());
        fn failing_scope() -> Result<(), &'static str> {
            let _scope = start_observation();
            Err("early-return cleanup")?;
            Ok(())
        }
        let failure = failing_scope();
        assert!(failure.is_err());
        assert!(OBSERVER.with(|slot| slot.borrow().is_none()));
    }

    #[test]
    fn concurrent_observation_scopes_do_not_share_counters() {
        std::thread::scope(|scope| {
            let handles: Vec<_> = (1..=4)
                .map(|count| {
                    scope.spawn(move || {
                        let observer = start_observation();
                        let task = observe_kernel(ObservedWork::Qp, false);
                        for _ in 0..count {
                            let _entry = task.enter_item();
                        }
                        let snapshot = observer.finish();
                        assert_eq!(snapshot.serial_qp_items, count);
                        assert_eq!(snapshot.serial_calls, 1);
                        assert_eq!(snapshot.worker_entries, 0);
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap();
            }
        });
    }
}

//! Runtime control for deterministic shared-memory inner kernels.

use std::cell::RefCell;
use std::collections::BTreeSet;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

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
    /// Hamiltonian/projection/RBM term-region entries (C `omp for` regions of
    /// `calham*.c`, `projection.c`, `rbm.c`) in total, including serial ones.
    pub executed_region_items: usize,
    /// Region entries actually dispatched through the inner pool.
    pub parallel_region_items: usize,
    /// Region entries on the serial dispatch path.
    pub serial_region_items: usize,
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
    items: [[AtomicUsize; 4]; 2],
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
            serial_region_items: get(0, 3),
            parallel_region_items: get(1, 3),
            executed_region_items: get(0, 3) + get(1, 3),
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
    Region,
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

/// Default minimum estimated serial work (nanoseconds) of one region before the
/// inner pool is used. Waking the pool and joining costs about 15-30 us per region
/// on the reference host (`docs/reference/c-to-julia/performance/`), so a region
/// must carry several times that to gain from two or more workers.
pub const DEFAULT_MIN_PARALLEL_WORK_NS: u64 = 100_000;

/// Estimated serial cost (ns) of one trivially cheap element (a copy or an axpy).
pub const ELEMENT_COST_NS: usize = 2;

/// Estimated serial cost (ns) of one Green-function evaluation with `bodies`
/// creation/annihilation pairs for an `n_size`-electron configuration: the kernels read
/// O(`n_size`) inverse entries per pair (fit to serial `MVMC_RS_INNER_PROFILE=1`
/// timings of 0.26/0.43/0.76 us per term at `n_size` 16/32/64 on the reference host).
pub fn green_cost_ns(n_size: usize, bodies: usize) -> usize {
    scaled_cost_ns(n_size, bodies * (12 * n_size + 60))
}

/// Electron-matrix dimension (`n_size`) at which an infinitely fast pool would break
/// even; the automatic gate requires `DEFAULT_MIN_PARALLEL_SIZE * w / (w - 1)` for `w`
/// workers, because the benefit `(1 - 1/w) * work` has to pay a dispatch cost that does
/// not shrink with `w`.
///
/// Dispatching a region wakes sleeping workers and moves the QP planes between their
/// caches and the caller's. Pooling only some regions of a sample therefore loses
/// (a region that is faster pooled slows the serial regions that follow it), while
/// pooling every region only wins once the per-sample matrices are large. Measured on
/// the Hubbard chain (`docs/reference/c-to-julia/performance/`): with 4 workers
/// `n_size` 16..64 loses up to 2x, 128 is neutral, 160 is about 10% faster and 192
/// about 15% faster; with 2 workers 192 still loses 10% while 256 is 1.35x faster
/// (4 workers 1.7x, 8 workers 2.0x).
pub const DEFAULT_MIN_PARALLEL_SIZE: usize = 120;

/// Default `min_size` for `threads` workers: `DEFAULT_MIN_PARALLEL_SIZE * w / (w - 1)`.
pub fn default_min_size(threads: usize) -> usize {
    let w = threads.max(2);
    (DEFAULT_MIN_PARALLEL_SIZE * w).div_ceil(w - 1)
}

/// Cost estimate of an `n_size`-dependent region: `cost_ns`, or 0 (always serial) when
/// the automatic gate is active and the matrices are below `min_size`. An explicit
/// `MVMC_RS_INNER_THRESHOLD` ignores costs, so it is unaffected.
pub fn scaled_cost_ns(n_size: usize, cost_ns: usize) -> usize {
    if n_size >= inner_thread_config().min_size {
        cost_ns
    } else {
        0
    }
}

/// Runtime configuration for independent inner work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InnerThreadConfig {
    /// Number of worker threads requested by `MVMC_RS_INNER_THREADS`.
    pub threads: usize,
    /// Minimum work items required to enable parallel execution.
    pub threshold: usize,
    /// `MVMC_RS_INNER_THRESHOLD` was set to a valid value: regions then use the
    /// plain item-count gate (`items >= threshold`) instead of the work estimate.
    pub threshold_explicit: bool,
    /// Minimum estimated region work in ns for the automatic gate
    /// (`MVMC_RS_INNER_MIN_WORK_NS`, default [`DEFAULT_MIN_PARALLEL_WORK_NS`]).
    pub min_work_ns: u64,
    /// Minimum electron-matrix dimension for the automatic gate
    /// (`MVMC_RS_INNER_MIN_SIZE`, default [`default_min_size`] of the worker count).
    pub min_size: usize,
}

/// Read the process-wide inner-kernel controls once.
///
/// `MVMC_RS_INNER_THREADS` defaults to one, preserving the historical
/// sequential execution and RNG/Markov-chain contract. `MVMC_RS_INNER_THRESHOLD`
/// defaults to [`DEFAULT_INNER_THRESHOLD`]. Invalid or zero values fall back
/// to those defaults.
pub fn inner_thread_config() -> InnerThreadConfig {
    static CONFIG: OnceLock<InnerThreadConfig> = OnceLock::new();
    *CONFIG.get_or_init(|| {
        let threshold = std::env::var("MVMC_RS_INNER_THRESHOLD")
            .ok()
            .and_then(|value| value.parse().ok())
            .filter(|&value: &usize| value > 0);
        let threads = std::env::var("MVMC_RS_INNER_THREADS")
            .ok()
            .and_then(|value| value.parse().ok())
            .filter(|&value: &usize| value > 0)
            .unwrap_or(1);
        InnerThreadConfig {
            threads,
            threshold: threshold.unwrap_or(DEFAULT_INNER_THRESHOLD),
            threshold_explicit: threshold.is_some(),
            min_work_ns: std::env::var("MVMC_RS_INNER_MIN_WORK_NS")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(DEFAULT_MIN_PARALLEL_WORK_NS),
            min_size: std::env::var("MVMC_RS_INNER_MIN_SIZE")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or_else(|| default_min_size(threads)),
        }
    })
}

/// Return whether a range should use the shared-memory worker pool.
pub fn inner_parallel_enabled(work_items: usize) -> bool {
    let config = inner_thread_config();
    config.threads > 1 && work_items >= config.threshold
}

/// Return whether a region of `items` independent items, each costing about
/// `cost_ns` nanoseconds serially, should use the worker pool.
///
/// With an explicit `MVMC_RS_INNER_THRESHOLD` this is the plain item-count gate
/// [`inner_parallel_enabled`] (used by the worker-invariance tests to force pooled
/// execution of small ranges). Otherwise the pool is used only when the estimated
/// serial work `items * cost_ns` reaches `min_work_ns`: dispatching a tiny region to
/// the pool costs more than running it (issue #361). The choice never changes any
/// result, only which threads form it.
pub fn inner_parallel_work(items: usize, cost_ns: usize) -> bool {
    let config = inner_thread_config();
    if config.threads <= 1 {
        return false;
    }
    if config.threshold_explicit {
        return items >= config.threshold;
    }
    items >= 2 && (items as u64).saturating_mul(cost_ns as u64) >= config.min_work_ns
}

/// Number of workers to provision for a work range already admitted by
/// [`inner_parallel_enabled`] or [`inner_parallel_work`].
pub fn inner_worker_count(work_items: usize) -> usize {
    let config = inner_thread_config();
    let admitted = if config.threshold_explicit {
        inner_parallel_enabled(work_items)
    } else {
        config.threads > 1
    };
    if admitted {
        config.threads.min(work_items).max(1)
    } else {
        1
    }
}

/// One kernel call site, as recorded by `MVMC_RS_INNER_PROFILE=1`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DispatchSite {
    /// `file:line` of the kernel entry.
    pub site: String,
    /// Whether these calls were dispatched to the inner pool (else serial).
    pub parallel: bool,
    /// Number of calls from that site.
    pub calls: u64,
    /// Total work items over those calls (0 when the site does not report it).
    pub items: u64,
    /// Total wall time of those calls, in nanoseconds.
    pub nanos: u128,
}

type DispatchTable = Mutex<std::collections::BTreeMap<(&'static str, u32, bool), (u64, u64, u128)>>;

fn dispatch_table() -> &'static DispatchTable {
    static TABLE: OnceLock<DispatchTable> = OnceLock::new();
    TABLE.get_or_init(Default::default)
}

fn profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("MVMC_RS_INNER_PROFILE").is_ok_and(|value| !value.is_empty() && value != "0")
    })
}

/// Per-call-site counts and wall time of the inner kernels (empty unless
/// `MVMC_RS_INNER_PROFILE=1`), sorted by descending time. Serial calls are recorded
/// too, so a one-worker run calibrates per-item cost and a multi-worker run shows
/// the pool dispatch overhead.
pub fn dispatch_profile() -> Vec<DispatchSite> {
    let mut sites: Vec<_> = dispatch_table()
        .lock()
        .unwrap()
        .iter()
        .map(
            |(&(file, line, parallel), &(calls, items, nanos))| DispatchSite {
                site: format!("{file}:{line}"),
                parallel,
                calls,
                items,
                nanos,
            },
        )
        .collect();
    sites.sort_by_key(|site| std::cmp::Reverse(site.nanos));
    sites
}

/// Records one kernel call on drop when profiling is enabled.
pub struct ProfileScope(Option<(&'static str, u32, bool, u64, Instant)>);

impl ProfileScope {
    #[track_caller]
    fn start(parallel: bool, items: usize) -> Self {
        // `Location::caller` must be read here: closures do not inherit `track_caller`.
        let location = std::panic::Location::caller();
        Self(profile_enabled().then(|| {
            (
                location.file(),
                location.line(),
                parallel,
                items as u64,
                Instant::now(),
            )
        }))
    }
}

impl Drop for ProfileScope {
    fn drop(&mut self) {
        if let Some((file, line, parallel, items, start)) = self.0.take() {
            let mut table = dispatch_table().lock().unwrap();
            let entry = table.entry((file, line, parallel)).or_default();
            entry.0 += 1;
            entry.1 += items;
            entry.2 += start.elapsed().as_nanos();
        }
    }
}

/// Profile a whole kernel call (serial or pooled) at the caller's location.
#[track_caller]
pub fn profile_scope(parallel: bool, items: usize) -> ProfileScope {
    ProfileScope::start(parallel, items)
}

/// Execute a closure on the configured deterministic Rayon pool.
#[track_caller]
pub fn install<R: Send>(operation: impl FnOnce() -> R + Send) -> R {
    let _scope = ProfileScope::start(true, 0);
    install_inner(operation)
}

fn pool() -> &'static ThreadPool {
    static POOL: OnceLock<ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        ThreadPoolBuilder::new()
            .num_threads(inner_thread_config().threads.max(1))
            .thread_name(|index| format!("mvmc-inner-{index}"))
            .build()
            .expect("inner Rayon pool must build")
    })
}

pub(crate) fn install_inner<R: Send>(operation: impl FnOnce() -> R + Send) -> R {
    let pool = pool();
    let observer = OBSERVER.with(|slot| slot.borrow().clone());
    pool.install(|| {
        let _binding = bind_observer(observer);
        operation()
    })
}

/// Static block schedule on the inner pool (the C `omp for` default): worker `k` always
/// handles the same contiguous block `k` of `0..count`, so per-QP data stays in that
/// worker's cache from one call to the next. Returns `body(start, end)` of each
/// non-empty block in block order. Which thread forms which block never affects results.
pub(crate) fn static_blocks<R: Send>(
    count: usize,
    body: impl Fn(usize, usize) -> R + Send + Sync,
) -> Vec<R> {
    let blocks = inner_worker_count(count).min(count).max(1);
    let block = count.div_ceil(blocks);
    let results = Mutex::new(Vec::with_capacity(blocks));
    let observer = OBSERVER.with(|slot| slot.borrow().clone());
    pool().broadcast(|context| {
        let start = context.index() * block;
        if start < count {
            let _binding = bind_observer(observer.clone());
            let value = body(start, (start + block).min(count));
            results.lock().unwrap().push((context.index(), value));
        }
    });
    let mut results = results.into_inner().unwrap();
    results.sort_by_key(|(index, _)| *index);
    results.into_iter().map(|(_, value)| value).collect()
}

/// Visit independent output entries; reductions inside an entry remain serial.
#[track_caller]
pub fn for_each_mut<T: Send>(
    items: &mut [T],
    cost_ns: usize,
    operation: impl Fn(usize, &mut T) + Send + Sync,
) {
    let parallel = inner_parallel_work(items.len(), cost_ns);
    let _scope = ProfileScope::start(parallel, items.len());
    let observed = observe_kernel(ObservedWork::Entry, parallel);
    let operation = |i, value| {
        let _entry = observed.enter_item();
        operation(i, value);
    };
    if parallel {
        install_inner(|| {
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

/// Visit consecutive `chunk_len`-long output windows (`chunk_len == 0` visits nothing).
/// Each window has one producer; reductions inside a window stay serial.
#[track_caller]
pub fn for_each_chunk_mut<T: Send>(
    items: &mut [T],
    chunk_len: usize,
    cost_ns: usize,
    operation: impl Fn(usize, &mut [T]) + Send + Sync,
) {
    if chunk_len == 0 {
        return;
    }
    let count = items.len().div_ceil(chunk_len);
    let parallel = inner_parallel_work(count, cost_ns);
    let _scope = ProfileScope::start(parallel, count);
    let observed = observe_kernel(ObservedWork::Entry, parallel);
    let operation = |i, window: &mut [T]| {
        let _entry = observed.enter_item();
        operation(i, window);
    };
    if parallel {
        install_inner(|| {
            items
                .par_chunks_mut(chunk_len)
                .enumerate()
                .for_each(|(i, window)| operation(i, window))
        });
    } else {
        items
            .chunks_mut(chunk_len)
            .enumerate()
            .for_each(|(i, window)| operation(i, window));
    }
}

/// Like [`for_each_chunk_mut`] over two outputs with their own window lengths;
/// the number of windows is that of the first slice (the second must have at least
/// as many windows).
#[track_caller]
pub fn for_each_chunk_pair_mut<A: Send, B: Send>(
    left: &mut [A],
    left_len: usize,
    right: &mut [B],
    right_len: usize,
    cost_ns: usize,
    operation: impl Fn(usize, &mut [A], &mut [B]) + Send + Sync,
) {
    if left_len == 0 || right_len == 0 {
        return;
    }
    let count = left.len().div_ceil(left_len);
    assert!(right.len().div_ceil(right_len) >= count);
    let parallel = inner_parallel_work(count, cost_ns);
    let _scope = ProfileScope::start(parallel, count);
    let observed = observe_kernel(ObservedWork::Entry, parallel);
    let operation = |i, a: &mut [A], b: &mut [B]| {
        let _entry = observed.enter_item();
        operation(i, a, b);
    };
    if parallel {
        install_inner(|| {
            left.par_chunks_mut(left_len)
                .zip(right.par_chunks_mut(right_len))
                .enumerate()
                .for_each(|(i, (a, b))| operation(i, a, b))
        });
    } else {
        left.chunks_mut(left_len)
            .zip(right.chunks_mut(right_len))
            .enumerate()
            .for_each(|(i, (a, b))| operation(i, a, b));
    }
}

/// Visit matching independent entries without aliasing either output slice.
#[track_caller]
pub fn for_each_pair_mut<A: Send, B: Send>(
    left: &mut [A],
    right: &mut [B],
    cost_ns: usize,
    operation: impl Fn(usize, &mut A, &mut B) + Send + Sync,
) {
    assert_eq!(left.len(), right.len());
    let parallel = inner_parallel_work(left.len(), cost_ns);
    let _scope = ProfileScope::start(parallel, left.len());
    let observed = observe_kernel(ObservedWork::Entry, parallel);
    let operation = |i, a, b| {
        let _entry = observed.enter_item();
        operation(i, a, b);
    };
    if parallel {
        install_inner(|| {
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

/// Fill `out[qp]` for `qp in qp_start..qp_end` (clamped to `out.len()`) with
/// `body(scratch, qp)`. Each output element has exactly one producer, so the
/// result is independent of the worker count. `init` builds the per-task scratch
/// (the C `GetWorkSpaceThread*` buffers of a `#pragma omp for` over `qpidx`).
/// Serial (no pool) when inner threading is off or `qp_end - qp_start` is below
/// the threshold.
#[track_caller]
pub(crate) fn qp_fill<T: Send, S>(
    out: &mut [T],
    qp_start: usize,
    qp_end: usize,
    cost_ns: usize,
    init: impl Fn() -> S + Send + Sync,
    body: impl Fn(&mut S, usize) -> T + Send + Sync,
) {
    let end = qp_end.min(out.len());
    if qp_start >= end {
        return;
    }
    let count = end - qp_start;
    let parallel = inner_parallel_work(count, cost_ns);
    let _scope = ProfileScope::start(parallel, count);
    let observed = observe_kernel(ObservedWork::Qp, parallel);
    if parallel {
        install_inner(|| {
            out[qp_start..end].par_iter_mut().enumerate().for_each_init(
                &init,
                |scratch, (i, slot)| {
                    let _entry = observed.enter_item();
                    *slot = body(scratch, qp_start + i);
                },
            )
        });
    } else {
        let mut scratch = init();
        for (qp, slot) in out.iter_mut().enumerate().take(end).skip(qp_start) {
            let _entry = observed.enter_item();
            *slot = body(&mut scratch, qp);
        }
    }
}

/// Evaluate independent Hamiltonian/Green terms in parallel and return their values
/// in term order, or `None` when the serial path should run (threading off or fewer
/// than the threshold terms). The caller reduces the returned values serially in
/// index order, so the result is bit-identical for every worker count and equal to
/// C's one-thread accumulation order (the `#pragma omp for ... reduction(+:e)` of
/// `calham*.c` / `lslocgrn*.c`). `init` builds the per-task scratch (C's
/// per-thread `myEleIdx`/`myBuffer` copies).
#[track_caller]
pub(crate) fn collect_terms<T: Send, S>(
    count: usize,
    cost_ns: usize,
    init: impl Fn() -> S + Send + Sync,
    body: impl Fn(&mut S, usize) -> T + Send + Sync,
) -> Option<Vec<T>> {
    if !inner_parallel_work(count, cost_ns) {
        return None;
    }
    let _scope = ProfileScope::start(true, count);
    let observed = observe_kernel(ObservedWork::Region, true);
    Some(install_inner(|| {
        (0..count)
            .into_par_iter()
            .map_init(&init, |scratch, index| {
                let _entry = observed.enter_item();
                body(scratch, index)
            })
            .collect()
    }))
}

/// Visit `qp in qp_start..qp_end` (clamped to `values.len()`), giving each QP its
/// own `values[qp]` and its own `stride`-long window `inv[qp * stride ..]` of the
/// flat inverse matrix (windows of different QPs never overlap because
/// `stride >= min_len`). The C `#pragma omp for` over `qpidx` of `updateMAll*`.
#[allow(clippy::too_many_arguments)]
#[track_caller]
pub(crate) fn qp_update<T: Send, U: Send, S>(
    values: &mut [T],
    inv: &mut [U],
    stride: usize,
    min_len: usize,
    qp_start: usize,
    qp_end: usize,
    cost_ns: usize,
    init: impl Fn() -> S + Send + Sync,
    body: impl Fn(&mut S, usize, &mut T, &mut [U]) + Send + Sync,
) {
    let end = qp_end.min(values.len());
    if qp_start >= end {
        return;
    }
    assert!(
        stride >= min_len.max(1),
        "QP windows of the inverse must not overlap"
    );
    let count = end - qp_start;
    let parallel = inner_parallel_work(count, cost_ns);
    let _scope = ProfileScope::start(parallel, count);
    let observed = observe_kernel(ObservedWork::Qp, parallel);
    let inv = &mut inv[qp_start * stride..];
    if parallel {
        install_inner(|| {
            values[qp_start..end]
                .par_iter_mut()
                .zip(inv.par_chunks_mut(stride))
                .enumerate()
                .for_each_init(&init, |scratch, (i, (value, window))| {
                    let _entry = observed.enter_item();
                    body(scratch, qp_start + i, value, window);
                })
        });
    } else {
        let mut scratch = init();
        for (i, (value, window)) in values[qp_start..end]
            .iter_mut()
            .zip(inv.chunks_mut(stride))
            .enumerate()
        {
            let _entry = observed.enter_item();
            body(&mut scratch, qp_start + i, value, window);
        }
    }
}

/// Copy the common prefix with disjoint writes; the destination tail is untouched.
#[track_caller]
pub fn copy_real_to_complex(dst: &mut [Complex64], src: &[f64]) {
    let n = dst.len().min(src.len());
    for_each_mut(&mut dst[..n], ELEMENT_COST_NS, |i, value| {
        *value = Complex64::new(src[i], 0.0)
    });
}

/// Copy real parts of the common prefix, preserving the destination tail.
#[track_caller]
pub fn copy_complex_realpart(dst: &mut [f64], src: &[Complex64]) {
    let n = dst.len().min(src.len());
    for_each_mut(&mut dst[..n], ELEMENT_COST_NS, |i, value| {
        *value = src[i].re
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_blocks_cover_the_range_once_in_order() {
        for count in [1usize, 2, 7, 8, 33] {
            let blocks = static_blocks(count, |start, end| (start, end));
            assert!(!blocks.is_empty());
            assert!(blocks.len() <= inner_thread_config().threads.max(1));
            let mut next = 0;
            for (start, end) in blocks {
                assert_eq!(start, next, "blocks are contiguous and ordered");
                assert!(end > start);
                next = end;
            }
            assert_eq!(next, count);
        }
    }

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
        for_each_mut::<u8>(&mut [], 1, |_, _| panic!("empty range must not enter"));
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

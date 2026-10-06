//! Low-latency fork-join pool for the inner kernels (issue #479).
//!
//! C's `#pragma omp parallel for` regions are cheap because OpenMP workers spin between
//! regions. The Rayon pool used by [`crate::threading`] sleeps between regions, so waking it
//! and joining costs 15-30 us per region, which is more than the work of most per-hop
//! regions (an `UpdateMAll` over 8 QP planes at `n_size` 64 is about 30 us serial).
//!
//! This pool keeps `threads - 1` worker threads that spin for a short while after every
//! region (and park on a condition variable afterwards, so a long serial phase does not burn
//! cores) and lets the calling thread run block 0, like the OpenMP master. A region is
//! `blocks <= threads` static blocks: block `b` runs on worker `b` (worker 0 is the caller).
//! Block boundaries depend only on the item count and the configured worker count, never on
//! timing, and every output element has one producer, so results are independent of the
//! worker count and of the schedule.
//!
//! Dispatch protocol. `word` packs `epoch << 16 | blocks`; a worker that sees a new word
//! runs its block iff its index is below `blocks` and then decrements `remaining`; the caller
//! publishes the job pointer before the word and returns only after `remaining` reaches
//! zero, so the (stack) job outlives every use. Workers that do not take part never touch
//! the job. Concurrent callers (several walker threads) are serialized with `try_lock`; a
//! caller that finds the pool busy, or that is itself running inside a block, executes all
//! blocks inline in order, which yields the same result.

use std::cell::Cell;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

thread_local! {
    /// Worker index while a block runs on this thread (the caller is worker 0).
    static WORKER: Cell<Option<usize>> = const { Cell::new(None) };
}

/// Index of the pool worker executing the current block, if any.
pub(crate) fn current_worker() -> Option<usize> {
    WORKER.with(Cell::get)
}

type Body<'a> = dyn Fn(usize) + Sync + 'a;

struct Job {
    // Lifetime erased: `run_blocks` does not return before every worker is done with it.
    body: *const Body<'static>,
}

struct Shared {
    word: AtomicU64,
    job: AtomicPtr<Job>,
    remaining: AtomicUsize,
    panicked: AtomicBool,
    parked: AtomicUsize,
    gate: Mutex<()>,
    wake: Condvar,
}

/// How long an idle worker spins before parking. About twice the longest regular gap
/// between two regions of one MC step, short enough that serial phases do not burn cores.
const SPIN_BEFORE_PARK: Duration = Duration::from_micros(300);

struct Pool {
    threads: usize,
    shared: &'static Shared,
    caller: Mutex<()>,
}

fn worker_main(index: usize, shared: &'static Shared) {
    WORKER.with(|w| w.set(Some(index)));
    // Start from 0 (the epoch counter starts at 1), so a region published before this thread
    // first ran is still seen.
    let mut seen = 0;
    loop {
        // Wait for a new word: spin, then park.
        let mut spins = 0u32;
        let started = Instant::now();
        let word = loop {
            let word = shared.word.load(Ordering::Acquire);
            if word != seen {
                break word;
            }
            spins += 1;
            if !spins.is_multiple_of(64) {
                std::hint::spin_loop();
            } else if started.elapsed() < SPIN_BEFORE_PARK {
                std::thread::yield_now();
            } else {
                let mut guard = shared.gate.lock().unwrap_or_else(|e| e.into_inner());
                shared.parked.fetch_add(1, Ordering::SeqCst);
                while shared.word.load(Ordering::SeqCst) == seen {
                    guard = shared.wake.wait(guard).unwrap_or_else(|e| e.into_inner());
                }
                shared.parked.fetch_sub(1, Ordering::SeqCst);
                drop(guard);
                spins = 0;
            }
        };
        seen = word;
        let blocks = (word & 0xFFFF) as usize;
        if index < blocks {
            // SAFETY: the caller stored a valid job pointer before publishing `word` and
            // keeps the job alive until `remaining` is zero; this worker takes part in this
            // word, so the pointer cannot have been replaced yet.
            let job = unsafe { &*shared.job.load(Ordering::Acquire) };
            // SAFETY: same lifetime argument as above for the closure behind the pointer.
            let body = unsafe { &*job.body };
            if catch_unwind(AssertUnwindSafe(|| body(index))).is_err() {
                shared.panicked.store(true, Ordering::Relaxed);
            }
            shared.remaining.fetch_sub(1, Ordering::Release);
        }
    }
}

fn pool() -> &'static Pool {
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(|| {
        let threads = crate::threading::inner_thread_config().threads.max(1);
        assert!(
            threads < 0xFFFF,
            "inner thread count must fit the dispatch word"
        );
        let shared: &'static Shared = Box::leak(Box::new(Shared {
            word: AtomicU64::new(0),
            job: AtomicPtr::new(std::ptr::null_mut()),
            remaining: AtomicUsize::new(0),
            panicked: AtomicBool::new(false),
            parked: AtomicUsize::new(0),
            gate: Mutex::new(()),
            wake: Condvar::new(),
        }));
        for index in 1..threads {
            std::thread::Builder::new()
                .name(format!("mvmc-inner-{index}"))
                .spawn(move || worker_main(index, shared))
                .expect("inner spin worker must start");
        }
        Pool {
            threads,
            shared,
            caller: Mutex::new(()),
        }
    })
}

/// Run `body(0..blocks)` with `blocks <= threads` on the pool (block 0 on the caller).
///
/// Falls back to running the blocks inline, in order, when the pool is busy (another
/// caller, or a nested region): the blocks are independent, so the result is the same.
/// Panics of a block are propagated after all blocks finished.
pub(crate) fn run_blocks(blocks: usize, body: &(dyn Fn(usize) + Sync)) {
    if blocks == 0 {
        return;
    }
    let pool = pool();
    let blocks = blocks.min(pool.threads);
    // Inline execution keeps the worker identity of each block, so observation (and any
    // per-worker bookkeeping) sees the same workers as a pooled run.
    let inline = |body: &(dyn Fn(usize) + Sync)| {
        let previous = WORKER.with(Cell::get);
        for b in 0..blocks {
            WORKER.with(|w| w.set(Some(b)));
            body(b);
        }
        WORKER.with(|w| w.set(previous));
    };
    if blocks == 1 || current_worker().is_some() {
        return inline(body);
    }
    let Ok(_caller) = pool.caller.try_lock() else {
        return inline(body);
    };
    let shared = pool.shared;
    // SAFETY: the pointer is only dereferenced between publishing the word and the join
    // below, during which `body` is alive.
    let erased: *const Body<'static> =
        unsafe { std::mem::transmute::<*const Body<'_>, *const Body<'static>>(body) };
    let mut job = Job { body: erased };
    shared.panicked.store(false, Ordering::Relaxed);
    shared.remaining.store(blocks - 1, Ordering::Relaxed);
    shared.job.store(&mut job, Ordering::Release);
    let epoch = (shared.word.load(Ordering::Relaxed) >> 16) + 1;
    shared
        .word
        .store((epoch << 16) | blocks as u64, Ordering::SeqCst);
    if shared.parked.load(Ordering::SeqCst) > 0 {
        let _guard = shared.gate.lock().unwrap_or_else(|e| e.into_inner());
        shared.wake.notify_all();
    }
    // Block 0 on the calling thread, which counts as worker 0 while it runs.
    WORKER.with(|w| w.set(Some(0)));
    let own = catch_unwind(AssertUnwindSafe(|| body(0)));
    WORKER.with(|w| w.set(None));
    let mut spins = 0u32;
    while shared.remaining.load(Ordering::Acquire) != 0 {
        spins += 1;
        if spins.is_multiple_of(256) {
            std::thread::yield_now();
        } else {
            std::hint::spin_loop();
        }
    }
    if let Err(payload) = own {
        std::panic::resume_unwind(payload);
    }
    if shared.panicked.load(Ordering::Relaxed) {
        panic!("a block of the inner spin pool panicked");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_block_runs_exactly_once_for_many_regions() {
        let counters: Vec<AtomicUsize> = (0..16).map(|_| AtomicUsize::new(0)).collect();
        for round in 1..=200usize {
            let blocks = 1 + round % 4;
            run_blocks(blocks, &|b| {
                counters[b].fetch_add(1, Ordering::Relaxed);
            });
        }
        let total: usize = counters.iter().map(|c| c.load(Ordering::Relaxed)).sum();
        let expected: usize = (1..=200usize)
            .map(|r| (1 + r % 4).min(pool().threads))
            .sum();
        assert_eq!(total, expected);
    }

    #[test]
    fn nested_and_concurrent_regions_run_inline_without_deadlock() {
        let hits = AtomicUsize::new(0);
        run_blocks(4, &|_| {
            run_blocks(3, &|_| {
                hits.fetch_add(1, Ordering::Relaxed);
            });
        });
        assert!(hits.load(Ordering::Relaxed) >= 1);
        std::thread::scope(|s| {
            for _ in 0..4 {
                s.spawn(|| {
                    for _ in 0..50 {
                        run_blocks(4, &|_| {
                            hits.fetch_add(1, Ordering::Relaxed);
                        });
                    }
                });
            }
        });
    }

    /// Dispatch latency of an empty region (`cargo nextest run ... --run-ignored all
    /// --no-capture`; set `MVMC_RS_INNER_THREADS`).
    #[test]
    #[ignore = "micro-benchmark"]
    fn dispatch_latency_of_empty_regions() {
        let threads = pool().threads;
        for blocks in [2usize, 4, 8, 16].into_iter().filter(|&b| b <= threads) {
            for gap_ns in [0u64, 2_000, 20_000] {
                let n = 20_000;
                let sink = AtomicUsize::new(0);
                let mut total = Duration::ZERO;
                for _ in 0..n {
                    // serial work between regions keeps the workers inside their spin window
                    let spin = Instant::now();
                    while (spin.elapsed().as_nanos() as u64) < gap_ns {
                        std::hint::spin_loop();
                    }
                    let t = Instant::now();
                    run_blocks(blocks, &|b| {
                        sink.fetch_add(b, Ordering::Relaxed);
                    });
                    total += t.elapsed();
                }
                eprintln!(
                    "blocks={blocks} gap={gap_ns}ns: {:.2} us per empty region",
                    total.as_secs_f64() * 1e6 / n as f64
                );
            }
        }
    }

    /// Region wall time for blocks that each busy-wait `work_us`: the ideal is `work_us`.
    #[test]
    #[ignore = "micro-benchmark"]
    fn region_time_with_busy_blocks() {
        let threads = pool().threads;
        for work_us in [5u64, 20, 90] {
            let n = 5_000;
            let mut total = Duration::ZERO;
            for _ in 0..n {
                let t = Instant::now();
                run_blocks(threads, &|_| {
                    let spin = Instant::now();
                    while (spin.elapsed().as_nanos() as u64) < work_us * 1000 {
                        std::hint::spin_loop();
                    }
                });
                total += t.elapsed();
            }
            eprintln!(
                "threads={threads} work={work_us} us: region {:.2} us (overhead {:.2} us)",
                total.as_secs_f64() * 1e6 / n as f64,
                total.as_secs_f64() * 1e6 / n as f64 - work_us as f64
            );
        }
    }
}

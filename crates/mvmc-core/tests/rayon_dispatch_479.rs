//! Dispatch-latency micro-benchmark of Rayon variants (issue #479); ignored by default:
//! `cargo nextest run -p mvmc-core --cargo-profile test-fast -E 'test(rayon_dispatch)' --run-ignored all --no-capture`.
//!
//! Measured on the reference host (8 workers, quiet): from outside the pool `install` +
//! `scope`/`par_iter` costs 7-18 us per region and `ThreadPool::broadcast` about 4 us while the
//! gap between regions is below Rayon's idle spin window (about 30 us); from inside one hoisted
//! `install` `broadcast` costs about 3 us; after a 60 us gap every variant pays 19-33 us because
//! the workers went to sleep.

use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

fn busy(ns: u64) {
    let s = Instant::now();
    while (s.elapsed().as_nanos() as u64) < ns {
        std::hint::spin_loop();
    }
}

fn run(label: &str, n: usize, gap_ns: u64, work_ns: u64, mut region: impl FnMut()) {
    // warm up
    for _ in 0..200 {
        region();
    }
    let mut total = Duration::ZERO;
    for _ in 0..n {
        busy(gap_ns);
        let t = Instant::now();
        region();
        total += t.elapsed();
    }
    let per = total.as_secs_f64() * 1e6 / n as f64;
    eprintln!(
        "{label:34} gap={gap_ns:6}ns work={work_ns:5}ns/blk: {per:7.2} us (overhead {:6.2} us)",
        per - work_ns as f64 / 1e3
    );
}

#[test]
#[ignore]
fn rayon_dispatch_variants() {
    let threads = 8;
    let pool = ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    let sink = AtomicUsize::new(0);
    for gap in [0u64, 5_000, 20_000, 60_000] {
        for work in [0u64, 5_000] {
            let n = 3000;
            // outside the pool
            run("outside: install+scope", n, gap, work, || {
                pool.install(|| {
                    rayon::scope(|s| {
                        for b in 1..threads {
                            let sink = &sink;
                            s.spawn(move |_| {
                                busy(work);
                                sink.fetch_add(b, Ordering::Relaxed);
                            });
                        }
                        busy(work);
                    })
                })
            });
            run("outside: broadcast", n, gap, work, || {
                pool.broadcast(|ctx| {
                    busy(work);
                    sink.fetch_add(ctx.index(), Ordering::Relaxed);
                });
            });
            run("outside: install+par_iter", n, gap, work, || {
                pool.install(|| {
                    (0..threads).into_par_iter().with_min_len(1).for_each(|b| {
                        busy(work);
                        sink.fetch_add(b, Ordering::Relaxed);
                    })
                })
            });
            // inside one hoisted install
            pool.install(|| {
                run("inside: scope", n, gap, work, || {
                    rayon::scope(|s| {
                        for b in 1..threads {
                            let sink = &sink;
                            s.spawn(move |_| {
                                busy(work);
                                sink.fetch_add(b, Ordering::Relaxed);
                            });
                        }
                        busy(work);
                    })
                });
                run("inside: broadcast", n, gap, work, || {
                    rayon::broadcast(|ctx| {
                        busy(work);
                        sink.fetch_add(ctx.index(), Ordering::Relaxed);
                    });
                });
                run("inside: par_iter min_len 1", n, gap, work, || {
                    (0..threads).into_par_iter().with_min_len(1).for_each(|b| {
                        busy(work);
                        sink.fetch_add(b, Ordering::Relaxed);
                    })
                });
            });
        }
    }
}

//! Standalone scheduling probe, with no mVMC/native/MPI runtime dependency.
use rayon::{prelude::*, ThreadPoolBuilder};
use std::{
    hint::black_box,
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};

#[repr(align(64))]
struct Slot {
    value: AtomicU64,
    calls: AtomicU64,
}

fn payload(block: usize, work: usize) -> u64 {
    let mut value = black_box(block as u64 + 1);
    for _ in 0..work {
        value = value
            .rotate_left(7)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
    }
    black_box(value)
}

fn joins(first: usize, end: usize, body: &(impl Fn(usize) + Sync)) {
    if end - first == 1 {
        body(first);
    } else if end > first {
        let middle = first + (end - first) / 2;
        rayon::join(|| joins(first, middle, body), || joins(middle, end, body));
    }
}

fn region(mode: &str, blocks: usize, body: &(impl Fn(usize) + Sync)) {
    match mode {
        "serial" => (0..blocks).for_each(body),
        "broadcast" => {
            rayon::broadcast(|ctx| {
                if ctx.index() < blocks {
                    body(ctx.index())
                }
            });
        }
        "iterator" => (0..blocks).into_par_iter().for_each(body),
        "join" => joins(0, blocks, body),
        "spindle" | "spindle-budget" => spindle::for_each_raw(blocks, body),
        _ => unreachable!(),
    }
}

fn batch(mode: &str, workers: usize, blocks: usize, work: usize, gap: u64) -> f64 {
    let slots: Vec<_> = (0..blocks)
        .map(|_| Slot {
            value: AtomicU64::new(0),
            calls: AtomicU64::new(0),
        })
        .collect();
    let body = |block: usize| {
        slots[block]
            .value
            .store(payload(block, work), Ordering::Relaxed);
        slots[block].calls.fetch_add(1, Ordering::Relaxed);
    };
    let measure = || {
        for _ in 0..100 {
            region(mode, blocks, &body);
        }
        let mut nanos = 0;
        for _ in 0..200 {
            let waiting = Instant::now();
            while waiting.elapsed().as_nanos() < (gap as u128) {
                std::hint::spin_loop();
            }
            let start = Instant::now();
            region(mode, blocks, &body);
            nanos += start.elapsed().as_nanos();
        }
        nanos as f64 / 200.0
    };
    let ns = match mode {
        "spindle" => spindle::with_lock(workers, measure),
        "spindle-budget" => spindle::with_lock(blocks.min(workers), measure),
        _ => measure(),
    };
    for (block, slot) in slots.iter().enumerate() {
        assert_eq!(slot.value.load(Ordering::Relaxed), payload(block, work));
        assert_eq!(slot.calls.load(Ordering::Relaxed), 300);
    }
    ns
}

fn main() {
    let modes = [
        "serial",
        "broadcast",
        "iterator",
        "join",
        "spindle",
        "spindle-budget",
    ];
    for workers in [4, 16] {
        let pool = ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        pool.install(|| {
            for blocks in [1,2,4,8,16].into_iter().filter(|b| *b<=workers) {
                for work in [0,512,4096] {
                    for gap in [0,60_000] {
                        let mut samples=vec![Vec::new();modes.len()];
                        for trial in 0..5 {
                            for offset in 0..modes.len() {
                                let mode=(offset+trial)%modes.len();
                                samples[mode].push(batch(modes[mode],workers,blocks,work,gap));
                            }
                        }
                        for (mode,values) in modes.iter().zip(samples.iter_mut()) {
                            values.sort_by(f64::total_cmp);
                            println!("REGION workers={workers} blocks={blocks} work={work} gap_ns={gap} mode={mode} median_ns={} raw={values:?}",values[2]);
                        }
                    }
                }
            }
        });
    }
}

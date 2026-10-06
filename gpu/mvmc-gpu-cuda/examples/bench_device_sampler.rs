//! End-to-end benchmark of the device-resident lock-step sampler (issue #434) against the CPU
//! multi-chain sampler.
//!
//! ```text
//! cd gpu/mvmc-gpu-cuda
//! cargo run --release --example bench_device_sampler -- \
//!     [--sizes 16,32,64,128] [--walkers 1,8,64,512] [--hops 3000] [--reps 3] \
//!     [--transfer pinned|pageable|both] [--out results.csv]
//! ```
//!
//! What is timed. One *call* is `vmc_make_sample_real` of every walker: the C `VMCMakeSample`
//! of one sample series (`NVMCWarmUp + NVMCSample` outer steps of `Nsite` hop attempts, plus
//! the initial table construction and the periodic recomputations). Hubbard chain, half filling,
//! `Lsub = 4`, `U = 4`, `NSPGaussLeg = 8` (`NQP = 8`), real normal mode, hopping updates; all
//! walkers share one wavefunction (the parameters of walker 0) and have independent SFMT streams
//! seeded `RndSeed + w`. Each method builds its own walkers, runs one untimed warm-up call (burn-in
//! included), then times `--reps` further calls and reports the median wall time. Measurement
//! (`VMCMainCal`) is not part of the call.
//!
//! * `cpu-1thread`: walkers run one after the other on one thread.
//! * `cpu-multichain`: one thread per walker, up to all cores (the #425 runner's execution
//!   model: independent walkers, single-threaded inside).
//! * `cuda-pinned` / `cuda-pageable`: the lock-step device sampler; the host side of every walker
//!   runs on its own thread as in the multichain runner, the Pfaffian stages run on the GPU.
//!   Thread creation, the initial CPU table construction and the device begin pass are inside
//!   the timed call for the device and (the table construction) for the CPU alike.

use std::io::Write;
use std::time::Instant;

use mvmc_core::device_sampler::{run_lockstep_real, LockstepOptions, LockstepStats};
use mvmc_core::run::{prepare_sampling_walker, PhysCalPreparation, SamplingWalker};
use mvmc_core::sampling::driver::vmc_make_sample_real;
use mvmc_gpu_cuda::device_sampler::{CudaSamplerService, ServiceTimings};
use sfmt19937::Sfmt19937Rng;

fn repo() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn arg<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn list(args: &[String], name: &str, default: &[usize]) -> Vec<usize> {
    arg(args, name).map_or_else(
        || default.to_vec(),
        |s| {
            s.split(',')
                .map(|x| x.trim().parse().expect("number"))
                .collect()
        },
    )
}

/// `w_count` walkers sharing the wavefunction of one preparation, seeds `1 + w`.
fn make_walkers(l: usize, w_count: usize, samples: i64) -> Vec<SamplingWalker> {
    let namelist = repo().join(format!(
        "benchmark/hubbard_chain/inputs/hubbard_chain_L{l}/namelist.def"
    ));
    let mut base =
        mvmc_core::prepare_phys_cal_with_seed_offset(&namelist, None, "real", Some(1), true, 0)
            .expect("prepare");
    base.data.modpara.nvmc_sample = samples;
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(w_count);
    let mut out: Vec<Option<SamplingWalker>> = (0..w_count).map(|_| None).collect();
    std::thread::scope(|scope| {
        for (t, chunk) in out.chunks_mut(w_count.div_ceil(threads)).enumerate() {
            let base = &base;
            let per = w_count.div_ceil(threads);
            scope.spawn(move || {
                for (i, slot) in chunk.iter_mut().enumerate() {
                    let w = t * per + i;
                    let prep = PhysCalPreparation {
                        data: base.data.clone(),
                        rng: Sfmt19937Rng::new(1 + w as u32),
                        n_para_consumed: 0,
                        binary_output: false,
                        initialization_consumed: true,
                    };
                    *slot = Some(prepare_sampling_walker(prep).expect("walker"));
                }
            });
        }
    });
    out.into_iter().map(|w| w.expect("walker built")).collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn cpu_call(walkers: &mut [SamplingWalker], threads: usize) -> f64 {
    let t = Instant::now();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let cells: Vec<std::sync::Mutex<&mut SamplingWalker>> =
        walkers.iter_mut().map(std::sync::Mutex::new).collect();
    std::thread::scope(|scope| {
        for _ in 0..threads.min(cells.len()).max(1) {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if i >= cells.len() {
                    break;
                }
                let mut guard = cells[i].lock().unwrap();
                let w: &mut SamplingWalker = &mut guard;
                vmc_make_sample_real(&w.data, &mut w.state, &mut w.rng).expect("cpu sampler");
            });
        }
    });
    t.elapsed().as_secs_f64()
}

struct Row {
    method: String,
    l: usize,
    w: usize,
    secs: f64,
    hops_per_walker: usize,
    detail: String,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let sizes = list(&args, "--sizes", &[16, 32, 64, 128]);
    let walker_counts = list(&args, "--walkers", &[1, 8, 64, 512]);
    let hops: usize = arg(&args, "--hops").map_or(3000, |s| s.parse().unwrap());
    let reps: usize = arg(&args, "--reps").map_or(3, |s| s.parse().unwrap());
    let transfer = arg(&args, "--transfer").unwrap_or("both").to_string();
    let max_walker_mem_gb: f64 = arg(&args, "--max-gb").map_or(40.0, |s| s.parse().unwrap());
    let profile = args.iter().any(|a| a == "--profile");
    let spin_us: Option<u64> = arg(&args, "--spin-us").map(|s| s.parse().unwrap());
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());

    mvmc_gpu_cuda::install();
    println!("# device sampler benchmark (issue #434)");
    match mvmc_core::backend::device_report(mvmc_core::backend::BackendKind::Cuda(0)) {
        Ok(r) => println!("# {}", r.render().trim_end().replace('\n', "\n# ")),
        Err(e) => println!("# device report unavailable: {e}"),
    }
    println!("# host cores: {cores}; hops per walker per call target: {hops}; reps: {reps}; spin_us: {spin_us:?}");
    println!(
        "# uptime: {}",
        std::fs::read_to_string("/proc/loadavg")
            .unwrap_or_default()
            .trim()
    );
    println!("method,L,W,seconds,hops_per_walker_per_call,hops_per_s,detail");
    let mut csv = arg(&args, "--out").map(|p| std::fs::File::create(p).expect("csv"));
    if let Some(f) = csv.as_mut() {
        writeln!(
            f,
            "method,L,W,seconds,hops_per_walker_per_call,hops_per_s,detail"
        )
        .unwrap();
    }
    let mut emit = |row: &Row| {
        let total = (row.hops_per_walker * row.w) as f64;
        let line = format!(
            "{},{},{},{:.4},{},{:.0},{}",
            row.method,
            row.l,
            row.w,
            row.secs,
            row.hops_per_walker,
            total / row.secs,
            row.detail
        );
        println!("{line}");
        if let Some(f) = csv.as_mut() {
            writeln!(f, "{line}").unwrap();
        }
    };

    for &l in &sizes {
        // NVMCSample chosen so that (NVMCWarmUp 10 + NVMCSample) * Nsite ~ hops
        let samples = ((hops / l) as i64 - 10).max(1);
        let hops_per_walker = (samples as usize + 10) * l;
        let per_walker_gb = {
            let n2 = 2.0 * l as f64;
            // complex master + real Slater table + inverse tables, 8 QP
            8.0 * n2 * n2 * (16.0 + 8.0) / 1e9 + 8.0 * (l * l) as f64 * 8.0 / 1e9
        };
        for &w in &walker_counts {
            if per_walker_gb * w as f64 > max_walker_mem_gb {
                println!(
                    "# skip L={l} W={w}: ~{:.1} GB of walker state exceeds --max-gb",
                    per_walker_gb * w as f64
                );
                continue;
            }
            // CPU, one thread (only for W <= 64: it is W times the single-walker time)
            if w <= 64 {
                let mut ws = make_walkers(l, w, samples);
                cpu_call(&mut ws, 1);
                let t = median((0..reps).map(|_| cpu_call(&mut ws, 1)).collect());
                emit(&Row {
                    method: "cpu-1thread".into(),
                    l,
                    w,
                    secs: t,
                    hops_per_walker,
                    detail: String::new(),
                });
            }
            // CPU multichain (one thread per walker, all cores)
            {
                let mut ws = make_walkers(l, w, samples);
                cpu_call(&mut ws, cores);
                let t = median((0..reps).map(|_| cpu_call(&mut ws, cores)).collect());
                emit(&Row {
                    method: "cpu-multichain".into(),
                    l,
                    w,
                    secs: t,
                    hops_per_walker,
                    detail: format!("threads={}", cores.min(w)),
                });
            }
            // CUDA
            for path in ["pinned", "pageable"] {
                if transfer != "both" && transfer != path {
                    continue;
                }
                let mut ws = make_walkers(l, w, samples);
                let mut times = Vec::new();
                let mut last: Option<(ServiceTimings, LockstepStats)> = None;
                for rep in 0..=reps {
                    let (secs, timings, stats) = if path == "pinned" {
                        device_call(
                            CudaSamplerService::new_pinned(0).expect("service"),
                            &mut ws,
                            spin_us,
                            profile,
                        )
                    } else {
                        device_call(
                            CudaSamplerService::new_pageable(0).expect("service"),
                            &mut ws,
                            spin_us,
                            profile,
                        )
                    };
                    if rep > 0 {
                        times.push(secs);
                        last = Some((timings, stats));
                    }
                }
                let (tm, st) = last.expect("timed call");
                emit(&Row {
                    method: format!("cuda-{path}"),
                    l,
                    w,
                    secs: median(times),
                    hops_per_walker,
                    detail: format!(
                        "passes={} max_batch={} recv_ms={:.1} round_ms={:.1} deliver_ms={:.1} stage_ms={:.1} upload_ms={:.1} launch_ms={:.1} wait_ms={:.1} dev_accept_ms={:.1} slow_wait_ms={:.1} slow_batches={} dev_slow_ms={:.1} dev_propose_ms={:.1}",
                        st.passes, st.max_batch, st.recv_s * 1e3, st.round_s * 1e3, st.deliver_s * 1e3,
                        tm.stage_s * 1e3, tm.upload_s * 1e3, tm.launch_s * 1e3, tm.wait_s * 1e3,
                        tm.dev_accept_s * 1e3, tm.slow_wait_s * 1e3, tm.slow_batches, tm.dev_slow_s * 1e3, tm.dev_propose_s * 1e3
                    ),
                });
            }
        }
    }
}

fn device_call<T: mvmc_gpu_cuda::device_sampler::TransferPath>(
    mut service: CudaSamplerService<T>,
    ws: &mut [SamplingWalker],
    spin_us: Option<u64>,
    profile: bool,
) -> (f64, ServiceTimings, LockstepStats) {
    service.profile = profile;
    let t = Instant::now();
    let (runs, stats) = run_lockstep_real(
        ws,
        &mut service,
        LockstepOptions {
            spin_us,
            ..LockstepOptions::default()
        },
    )
    .expect("device sampler");
    let secs = t.elapsed().as_secs_f64();
    for r in &runs {
        r.stats.as_ref().expect("walker");
    }
    (secs, service.timings, stats)
}

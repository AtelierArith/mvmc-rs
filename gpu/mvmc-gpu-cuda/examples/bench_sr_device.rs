//! Per-SR-step benchmark of the device-resident pipeline (issue #447) against OpenBLAS
//! (one thread and all cores) through the C-order host path.
//!
//! ```text
//! cd gpu/mvmc-gpu-cuda
//! cargo run --release --example bench_sr_device -- \
//!     [--sizes 1000x1000,1000x10000,...] [--reps 3] [--cg-iters 50] [--cpu1-max-flops 2e11] \
//!     [--out results.csv]
//! ```
//!
//! A size `NxM` is `n = NPara + 1` rows of the store and `M` samples. One *direct step* is the
//! host `finalize_oo_store_real` + `build_s_g` + `cholesky_solve` sequence of the C-order path
//! (`COrderSr::gram_real`, `assemble_s_g`, `cholesky_solve`: SYRK, the elementwise assembly,
//! DPOTRF + DPOTRS) against the device pipeline (`DeviceSr::upload_store` + `solve_direct`).
//! One *CG step* is `--cg-iters` iterations of `SampledSrOperator::solve` (tolerance 0, so
//! every iteration runs; two GEMV per product) against `DeviceSr::set_cg_operand` +
//! `solve_cg`. For the device both the end-to-end time (the store is uploaded every step) and
//! the resident time (the store is already on the device, as when it is produced there) are
//! reported. CPU rows: `OPENBLAS_NUM_THREADS` is set through `openblas_set_num_threads`
//! (1, and all cores). All times are medians of `--reps` runs after one warm-up.

use std::io::Write;
use std::time::Instant;

use mvmc_core::sr_backend::{COrderSr, RealView, SrAssembleInput, SrStages};
use mvmc_core::sr_cg::SampledSrOperator;
use mvmc_gpu_cuda::sr_device::DeviceSr;
use mvmc_gpu_cuda::sr_problem::{cg_inputs, make_problem};

extern "C" {
    fn openblas_set_num_threads(threads: i32);
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn time(reps: usize, mut f: impl FnMut()) -> f64 {
    f();
    median(
        (0..reps.max(1))
            .map(|_| {
                let t = Instant::now();
                f();
                t.elapsed().as_secs_f64()
            })
            .collect(),
    )
}

fn arg<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let sizes: Vec<(usize, usize)> = arg(&args, "--sizes")
        .unwrap_or("1000x1000,1000x10000,3000x3000,3000x10000,5000x5000,10000x1000,10000x10000")
        .split(',')
        .map(|s| {
            let (a, b) = s.split_once('x').expect("NxM");
            (a.parse().unwrap(), b.parse().unwrap())
        })
        .collect();
    let reps: usize = arg(&args, "--reps").map_or(3, |s| s.parse().unwrap());
    let cg_iters: usize = arg(&args, "--cg-iters").map_or(50, |s| s.parse().unwrap());
    let cpu1_max_flops: f64 = arg(&args, "--cpu1-max-flops").map_or(2e11, |s| s.parse().unwrap());
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());

    mvmc_gpu_cuda::install();
    println!("# device-resident SR step benchmark (issue #447)");
    match mvmc_core::backend::device_report(mvmc_core::backend::BackendKind::Cuda(0)) {
        Ok(r) => println!("# {}", r.render().trim_end().replace('\n', "\n# ")),
        Err(e) => println!("# device report unavailable: {e}"),
    }
    println!(
        "# host cores {cores}; loadavg {}; reps {reps}; cg iterations {cg_iters}; cpu-1-thread budget {cpu1_max_flops:.1e} flops",
        std::fs::read_to_string("/proc/loadavg").unwrap_or_default().trim()
    );
    let header = "step,n,samples,npara_active,cpu1_s,cpuN_s,gpu_e2e_s,gpu_resident_s,gpu_upload_store_s,gpu_gram_s,gpu_assemble_s,gpu_solve_s,gpu_vec_down_s";
    println!("{header}");
    let mut csv = arg(&args, "--out").map(|p| std::fs::File::create(p).expect("csv"));
    if let Some(f) = csv.as_mut() {
        writeln!(f, "{header}").unwrap();
    }
    let mut emit = |line: String| {
        println!("{line}");
        if let Some(f) = csv.as_mut() {
            writeln!(f, "{line}").unwrap();
        }
    };

    let mut dev = DeviceSr::new(0).expect("device pipeline");
    let fmt = |v: Option<f64>| v.map_or("-".to_string(), |x| format!("{x:.5}"));

    for &(n, samples) in &sizes {
        let (_, total) = dev.mem_info().unwrap();
        let bytes = (3 * n * n + n * samples) * 8;
        if bytes as f64 > 0.9 * total as f64 {
            println!(
                "# skip {n}x{samples}: ~{:.1} GB device memory",
                bytes as f64 / 1e9
            );
            continue;
        }
        let factors = 32.min(n - 1);
        let p = make_problem(n, samples, factors, 0.3, 0, 7);
        let nm = p.map.len();
        let (sta_del, step_dt) = (0.01, 0.003);
        let flops_direct = (n * n) as f64 * samples as f64 + (nm as f64).powi(3) / 3.0;
        let reps_here = if flops_direct > 3e11 { 1 } else { reps };

        // ---------------- direct step ----------------
        let host_step = |threads: i32| {
            // SAFETY: process-wide OpenBLAS setting, as `serial_blas` does.
            unsafe { openblas_set_num_threads(threads) };
            let mut be = COrderSr::default();
            let mut gram = vec![0.0; n * n];
            let mut s = vec![0.0; nm * nm];
            let mut g = vec![0.0; nm];
            time(reps_here, || {
                be.gram_real(&p.store, n, samples, &mut gram).unwrap();
                be.assemble_s_g(
                    &SrAssembleInput {
                        oo: RealView::Real(&gram),
                        ho: RealView::Real(&p.ho),
                        map: &p.map,
                        ld: n,
                        offset: 1,
                        sta_del,
                        step_dt,
                    },
                    &mut s,
                    &mut g,
                )
                .unwrap();
                be.cholesky_solve(&mut s, &mut g, nm).expect("host solve");
            })
        };
        let cpu1 = (flops_direct <= cpu1_max_flops).then(|| host_step(1));
        let cpu_n = host_step(cores as i32);
        // device
        let mut e2e = Vec::new();
        let mut resident = Vec::new();
        let mut tm = None;
        for rep in 0..=reps_here {
            let t = Instant::now();
            dev.upload_store(&p.store, n, samples).unwrap();
            let up = t.elapsed().as_secs_f64();
            let t2 = Instant::now();
            dev.solve_direct(&p.ho, &p.map, 1, sta_del, step_dt)
                .unwrap();
            let comp = t2.elapsed().as_secs_f64();
            if rep > 0 {
                e2e.push(up + comp);
                resident.push(comp);
                tm = Some(dev.timings);
            }
        }
        let tm = tm.unwrap();
        emit(format!(
            "direct,{n},{samples},{nm},{},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5}",
            fmt(cpu1),
            cpu_n,
            median(e2e),
            median(resident),
            tm.upload_store_s,
            tm.gram_s,
            tm.assemble_s,
            tm.solve_s,
            tm.download_s
        ));

        // ---------------- CG step ----------------
        let inp = cg_inputs(&p, step_dt);
        let comp = p.map.len();
        let flops_cg = 4.0 * comp as f64 * samples as f64 * cg_iters as f64;
        let host_cg = |threads: i32| {
            // SAFETY: as above.
            unsafe { openblas_set_num_threads(threads) };
            let mut op = SampledSrOperator::new(comp, samples, false);
            op.mean.copy_from_slice(&inp.mean);
            op.diagonal.copy_from_slice(&inp.diagonal);
            op.real_samples.copy_from_slice(&inp.operand);
            time(reps_here, || {
                let r = op.solve(&inp.gradient, 1.0, 0.5, 0.0, cg_iters);
                std::hint::black_box(&r);
            })
        };
        let cpu1 = (flops_cg <= cpu1_max_flops).then(|| host_cg(1));
        let cpu_n = host_cg(cores as i32);
        let mut e2e = Vec::new();
        let mut resident = Vec::new();
        let mut tm = None;
        for rep in 0..=reps_here {
            let t = Instant::now();
            dev.set_cg_operand(&inp.operand, None, comp, samples)
                .unwrap();
            let up = t.elapsed().as_secs_f64();
            let t2 = Instant::now();
            let out = dev
                .solve_cg(
                    &inp.gradient,
                    &inp.mean,
                    &inp.diagonal,
                    1.0,
                    0.5,
                    0.0,
                    cg_iters,
                )
                .unwrap();
            std::hint::black_box(&out);
            let c = t2.elapsed().as_secs_f64();
            if rep > 0 {
                e2e.push(up + c);
                resident.push(c);
                tm = Some(dev.timings);
            }
        }
        let tm = tm.unwrap();
        emit(format!(
            "cg{cg_iters},{n},{samples},{comp},{},{:.5},{:.5},{:.5},{:.5},-,-,{:.5},{:.5}",
            fmt(cpu1),
            cpu_n,
            median(e2e),
            median(resident),
            tm.upload_store_s,
            tm.solve_s,
            tm.download_s
        ));
    }
}

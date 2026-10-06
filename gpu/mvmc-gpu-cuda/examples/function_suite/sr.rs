//! Family 2: SR stages (issues #421, #437; #447 device-resident via its own hook).
//!
//! Stages: Gram `O O^T`, `S`/`g` assembly, Cholesky solve, CG matvec `O O^T x`, and a full
//! CG solve (fixed iteration count, matvec from the stage backend). Oracle: `COrderSr` (the
//! C-order implementation, OpenBLAS). Variants: the oracle on one core / all cores (separate
//! passes, `--cpu-label`), tenferro CPU (faer) and tenferro CUDA (cuBLAS/cuSOLVER).
//!
//! Numerical bounds (relative max-norm `max|a-b| / max|b|` versus the oracle; `eps = 2^-52`):
//! * Gram and CG matvec: both sides accumulate sums of `k` (resp. `k` then `n`) products of
//!   O(1) terms in a different order: `4 k eps` (resp. `4 (k + n) eps`).
//! * `S`/`g` assembly: elementwise, a few roundings: `8 eps`.
//! * Cholesky solve: `8 n eps kappa`, `kappa` from the Gershgorin discs of the actual (strictly
//!   diagonally dominant) matrix.
//! * CG solve with `K` fixed iterations on `A = O O^T / k + shift I`: matvec errors are
//!   amplified at most by `K kappa`: `2 K kappa 4 (k + n) eps`, `kappa <= (1.2 lambda_max +
//!   shift) / shift` with `lambda_max` from 20 power iterations of the oracle matvec.
//!
//! Uses `SrStages`, the SR layer of `stage_backend::StageBackend` (#437).

use std::time::Instant;

use mvmc_core::sr_backend::{
    COrderSr, CgSamples, Placement, RealView, SrAssembleInput, SrStages, TenferroSr,
};
use mvmc_gpu_cuda::bench::{cpu_runtime, cuda_runtime};

use crate::csv::{measure, Csv, Row, Stat, Verdict};
use crate::Cfg;

const FAMILY: &str = "sr";
const EPS: f64 = f64::EPSILON;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
}

fn rel(a: &[f64], b: &[f64]) -> f64 {
    let scale = b.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let d = a.iter().zip(b).fold(0.0_f64, |m, (x, y)| {
        if x.is_nan() {
            f64::NAN
        } else {
            m.max((x - y).abs())
        }
    });
    d / scale.max(f64::MIN_POSITIVE)
}

/// Strictly diagonally dominant SPD matrix, column-major, plus its Gershgorin condition bound.
fn spd(n: usize) -> (Vec<f64>, f64) {
    let mut st = 5;
    let mut s = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..j {
            let v = lcg(&mut st) * 0.5 / n as f64;
            s[i + j * n] = v;
            s[j + i * n] = v;
        }
        s[j + j * n] = 1.0 + lcg(&mut st).abs();
    }
    let (mut lo, mut hi) = (f64::INFINITY, 0.0_f64);
    for i in 0..n {
        let off: f64 = (0..n).filter(|&j| j != i).map(|j| s[i + j * n].abs()).sum();
        lo = lo.min(s[i + i * n] - off);
        hi = hi.max(s[i + i * n] + off);
    }
    (s, hi / lo)
}

/// Sample store `[n, k]` of O(1) entries.
fn store(n: usize, k: usize) -> Vec<f64> {
    let mut st = 7;
    (0..n * k).map(|_| lcg(&mut st)).collect()
}

struct Cx {
    reps: usize,
    budget: f64,
}

/// Time `f` (seconds per call). Skips the warm-up for very heavy stages (flops > 1e11).
fn timed(cx: &Cx, flops: f64, mut f: impl FnMut() -> Result<(), String>) -> Result<Stat, String> {
    measure(usize::from(flops <= 1e11), cx.reps, cx.budget, || {
        let t = Instant::now();
        f()?;
        Ok::<f64, String>(t.elapsed().as_secs_f64())
    })
}

/// Everything one backend produces for one `(n, k)` case.
struct Out {
    gram: (Stat, Vec<f64>),
    asm: (Stat, Vec<f64>),
    solve: (Stat, Vec<f64>),
    cg_mv: (Stat, Vec<f64>),
    cg_solve: (Stat, Vec<f64>),
    cg_upload: Option<Stat>,
}

/// Inputs shared by every backend of one case.
struct Case<'a> {
    n: usize,
    k: usize,
    store: &'a [f64],
    spd: &'a [f64],
    rhs: &'a [f64],
    /// `[(n+1)^2]` OO matrix, `HO`, and the active map for the assembly.
    oo: &'a [f64],
    ho: &'a [f64],
    cg_iters: usize,
    shift: f64,
    version: std::cell::Cell<u64>,
}

impl Case<'_> {
    fn next_version(&self) -> u64 {
        self.version.set(self.version.get() + 1);
        self.version.get()
    }
}

fn e(x: impl std::fmt::Display) -> String {
    x.to_string()
}

/// CG on `A = O O^T / k + shift I` with the matvec of `backend`, fixed `iters` iterations.
fn cg_solve(backend: &mut dyn SrStages, c: &Case<'_>, version: u64) -> Result<Vec<f64>, String> {
    let (n, k) = (c.n, c.k);
    let view = CgSamples {
        real: c.store,
        imag: &[],
        components: n,
        samples: k,
        version,
    };
    let b = c.rhs;
    let mut x = vec![0.0; n];
    let mut r = b.to_vec();
    let mut p = r.clone();
    let mut ap = vec![0.0; n];
    let mut rs: f64 = r.iter().map(|v| v * v).sum();
    for _ in 0..c.cg_iters {
        let mut z = vec![0.0; n];
        backend.cg_local_product(&view, &p, &mut z).map_err(e)?;
        for i in 0..n {
            ap[i] = z[i] / k as f64 + c.shift * p[i];
        }
        let pap: f64 = p.iter().zip(&ap).map(|(a, b)| a * b).sum();
        let alpha = rs / pap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let rs_new: f64 = r.iter().map(|v| v * v).sum();
        let beta = rs_new / rs;
        rs = rs_new;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
    }
    Ok(x)
}

fn run_backend(
    backend: &mut dyn SrStages,
    c: &Case<'_>,
    cx: &Cx,
    with_upload: bool,
) -> Result<Out, String> {
    let (n, k) = (c.n, c.k);
    let nf = n as f64;
    let kf = k as f64;
    let mut gram = vec![0.0; n * n];
    let s_gram = timed(cx, nf * nf * kf, || {
        backend.gram_real(c.store, n, k, &mut gram).map_err(e)
    })?;

    let size = n + 1;
    let map: Vec<usize> = (0..n).collect();
    let input = SrAssembleInput {
        oo: RealView::Real(c.oo),
        ho: RealView::Real(c.ho),
        map: &map,
        ld: size,
        offset: 1,
        sta_del: 0.02,
        step_dt: 0.05,
    };
    let (mut s, mut g) = (vec![0.0; n * n], vec![0.0; n]);
    let s_asm = timed(cx, nf * nf, || {
        backend.assemble_s_g(&input, &mut s, &mut g).map_err(e)
    })?;
    s.extend_from_slice(&g);

    let mut x = c.rhs.to_vec();
    let s_solve = timed(cx, nf * nf * nf / 3.0, || {
        let (mut sm, mut xv) = (c.spd.to_vec(), c.rhs.to_vec());
        backend
            .cholesky_solve(&mut sm, &mut xv, n)
            .map_err(|()| "cholesky_solve failed".to_string())?;
        x = xv;
        Ok(())
    })?;

    let view = CgSamples {
        real: c.store,
        imag: &[],
        components: n,
        samples: k,
        version: c.next_version(),
    };
    let xin: Vec<f64> = {
        let mut st = 99;
        (0..n).map(|_| lcg(&mut st)).collect()
    };
    let mut z = vec![0.0; n];
    let s_mv = timed(cx, 2.0 * nf * kf, || {
        backend.cg_local_product(&view, &xin, &mut z).map_err(e)
    })?;

    // operand upload (first call with a fresh version) = first-call time minus a cached call
    let cg_upload = if with_upload {
        let fresh = CgSamples {
            version: c.next_version(),
            ..CgSamples {
                real: c.store,
                imag: &[],
                components: n,
                samples: k,
                version: 0,
            }
        };
        let t = Instant::now();
        let mut zz = vec![0.0; n];
        backend.cg_local_product(&fresh, &xin, &mut zz).map_err(e)?;
        Some(Stat::single(
            (t.elapsed().as_secs_f64() - s_mv.median).max(0.0),
        ))
    } else {
        None
    };

    let mut xs = Vec::new();
    let s_cg = timed(cx, c.cg_iters as f64 * 2.0 * nf * kf, || {
        let v = c.next_version();
        xs = cg_solve(backend, c, v)?;
        Ok(())
    })?;
    Ok(Out {
        gram: (s_gram, gram),
        asm: (s_asm, s),
        solve: (s_solve, x),
        cg_mv: (s_mv, z),
        cg_solve: (s_cg, xs),
        cg_upload,
    })
}

/// Largest eigenvalue estimate of `A = O O^T/k + shift I` (power iteration on the oracle).
fn lambda_max(backend: &mut dyn SrStages, c: &Case<'_>) -> Result<f64, String> {
    let view = CgSamples {
        real: c.store,
        imag: &[],
        components: c.n,
        samples: c.k,
        version: c.next_version(),
    };
    let mut v = vec![1.0 / (c.n as f64).sqrt(); c.n];
    let mut lam = 0.0;
    for _ in 0..20 {
        let mut z = vec![0.0; c.n];
        backend.cg_local_product(&view, &v, &mut z).map_err(e)?;
        for i in 0..c.n {
            z[i] = z[i] / c.k as f64 + c.shift * v[i];
        }
        lam = z.iter().map(|x| x * x).sum::<f64>().sqrt();
        for i in 0..c.n {
            v[i] = z[i] / lam;
        }
    }
    Ok(lam)
}

fn variant_rows(
    csv: &mut Csv,
    variant: &str,
    params: &str,
    c: &Case<'_>,
    out: &Out,
    oracle: &Out,
    cg_kappa: f64,
    solve_kappa: f64,
) {
    let (n, k) = (c.n as f64, c.k as f64);
    let stages: [(&str, &(Stat, Vec<f64>), &(Stat, Vec<f64>), f64, &str); 5] = [
        ("gram", &out.gram, &oracle.gram, 4.0 * k * EPS, "4*k*eps"),
        ("assemble_s_g", &out.asm, &oracle.asm, 8.0 * EPS, "8*eps"),
        (
            "cholesky_solve",
            &out.solve,
            &oracle.solve,
            8.0 * n * EPS * solve_kappa,
            "8*n*eps*kappa(gershgorin)",
        ),
        (
            "cg_matvec",
            &out.cg_mv,
            &oracle.cg_mv,
            4.0 * (k + n) * EPS,
            "4*(k+n)*eps",
        ),
        (
            "cg_solve",
            &out.cg_solve,
            &oracle.cg_solve,
            2.0 * c.cg_iters as f64 * cg_kappa * 4.0 * (k + n) * EPS,
            "2*K*kappa*4*(k+n)*eps",
        ),
    ];
    for (name, o, r, bound, why) in stages {
        let row = Row::new(FAMILY, name, variant, "f64", params).stat(o.0);
        csv.push(
            row.dev("rel_maxnorm_vs_corder", rel(&o.1, &r.1), bound)
                .note(&format!("bound={why}")),
        );
    }
    if let Some(u) = out.cg_upload {
        csv.push(
            Row::new(FAMILY, "cg_operand_upload", variant, "f64", params)
                .stat(u)
                .note("once per solve; not numerical"),
        );
    }
}

fn run_case(cfg: &Cfg, cx: &Cx, csv: &mut Csv, n: usize, k: usize, dev_mem: Option<u64>) {
    let params = format!(
        "NPara={n};samples={k};cg_iters={}",
        if cfg.full { 40 } else { 25 }
    );
    let host_bytes = 8.0 * (n * k + 7 * n * n) as f64;
    let host_limit = cfg.max_gb.unwrap_or_else(|| mem_available_gb() * 0.5) * 1e9;
    let one_core = cfg.cpu_only && cfg.cpu_label == "1core";
    let flops = (n * n) as f64 * k as f64 + (n as f64).powi(3) / 3.0;
    if host_bytes > host_limit {
        csv.push(Row::new(FAMILY, "all", "all", "f64", &params).verdict(
            Verdict::Skipped,
            &format!(
                "needs ~{:.1} GB host memory (limit {:.1} GB, --max-gb)",
                host_bytes / 1e9,
                host_limit / 1e9
            ),
        ));
        return;
    }
    if one_core && flops > if cfg.full { 6e11 } else { 3e10 } {
        csv.push(
            Row::new(
                FAMILY,
                "all",
                &format!("cpu-corder-{}", cfg.cpu_label),
                "f64",
                &params,
            )
            .verdict(Verdict::Skipped, "too slow on one core for this profile"),
        );
        return;
    }
    let store = store(n, k);
    let (spd, solve_kappa) = spd(n);
    let mut st = 11;
    let rhs: Vec<f64> = (0..n).map(|_| lcg(&mut st)).collect();
    // OO from the C-order Gram, so every backend assembles identical inputs
    let mut gram = vec![0.0; n * n];
    COrderSr::default()
        .gram_real(&store, n, k, &mut gram)
        .expect("oracle gram");
    let size = n + 1;
    let mut oo = vec![0.0; size * size];
    for j in 0..n {
        for i in 0..n {
            oo[(i + 1) + (j + 1) * size] = gram[i + j * n] / k as f64;
        }
    }
    drop(gram);
    let ho: Vec<f64> = (0..size).map(|_| lcg(&mut st)).collect();
    let case = Case {
        n,
        k,
        store: &store,
        spd: &spd,
        rhs: &rhs,
        oo: &oo,
        ho: &ho,
        cg_iters: if cfg.full { 40 } else { 25 },
        shift: 0.5,
        version: std::cell::Cell::new(100),
    };
    let mut oracle_backend = COrderSr::default();
    let lam = match lambda_max(&mut oracle_backend, &case) {
        Ok(l) => l,
        Err(m) => {
            csv.push(Row::new(FAMILY, "all", "cpu-corder", "f64", &params).error(&m));
            return;
        }
    };
    let cg_kappa = (1.2 * lam + case.shift) / case.shift;
    let oracle = match run_backend(&mut oracle_backend, &case, cx, false) {
        Ok(o) => o,
        Err(m) => {
            csv.push(Row::new(FAMILY, "all", "cpu-corder", "f64", &params).error(&m));
            return;
        }
    };
    let ovar = format!("cpu-corder-{}", cfg.cpu_label);
    for (name, o) in [
        ("gram", &oracle.gram),
        ("assemble_s_g", &oracle.asm),
        ("cholesky_solve", &oracle.solve),
        ("cg_matvec", &oracle.cg_mv),
        ("cg_solve", &oracle.cg_solve),
    ] {
        csv.push(
            Row::new(FAMILY, name, &ovar, "f64", &params)
                .stat(o.0)
                .verdict(Verdict::Oracle, "C-order oracle (OpenBLAS)"),
        );
    }
    let faer_cap = if cfg.full { 2e11 } else { 2e10 };
    if flops <= faer_cap && !one_core {
        match cpu_runtime().and_then(|rt| {
            let mut b = TenferroSr::with_runtime(rt, Placement::Host, "tenferro cpu-faer".into());
            run_backend(&mut b, &case, cx, false)
        }) {
            Ok(o) => variant_rows(
                csv,
                "tenferro-cpu-faer",
                &params,
                &case,
                &o,
                &oracle,
                cg_kappa,
                solve_kappa,
            ),
            Err(m) => {
                csv.push(Row::new(FAMILY, "all", "tenferro-cpu-faer", "f64", &params).error(&m))
            }
        }
    }
    if cfg.cpu_only {
        return;
    }
    let dev_bytes = 8.0 * (n * k + 5 * n * n) as f64;
    if let Some(total) = dev_mem {
        if dev_bytes > total as f64 * 0.8 {
            csv.push(
                Row::new(FAMILY, "all", "tenferro-cuda", "f64", &params).verdict(
                    Verdict::Skipped,
                    &format!(
                        "needs ~{:.1} GB device memory of {:.1} GB",
                        dev_bytes / 1e9,
                        total as f64 / 1e9
                    ),
                ),
            );
            return;
        }
    }
    match cuda_runtime(0).and_then(|rt| {
        let mut b = TenferroSr::with_runtime(rt, Placement::Device, "tenferro cuda".into());
        run_backend(&mut b, &case, cx, true)
    }) {
        Ok(o) => variant_rows(
            csv,
            "tenferro-cuda",
            &params,
            &case,
            &o,
            &oracle,
            cg_kappa,
            solve_kappa,
        ),
        Err(m) => csv.push(Row::new(FAMILY, "all", "tenferro-cuda", "f64", &params).error(&m)),
    }
}

pub fn mem_available_gb() -> f64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("MemAvailable:"))
                .and_then(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok())
        })
        .map_or(16.0, |kb| kb / 1e6)
}

/// `dot_general` / Cholesky micro-benchmark of the tenferro CUDA backend (f64 and c64).
fn gemm_cholesky(cfg: &Cfg, csv: &mut Csv) {
    use mvmc_gpu_cuda::bench::{run_all, Dtype, Op};
    let sizes: &[usize] = if cfg.full {
        &[128, 512, 1024, 2048]
    } else {
        &[128, 512]
    };
    let reps = if cfg.full { 7 } else { 3 };
    match run_all(0, sizes, reps) {
        Ok(rows) => {
            for r in rows {
                let f = match r.op {
                    Op::Dot => "dot_general",
                    Op::Cholesky => "cholesky",
                };
                let dt = if r.dtype == Dtype::F64 { "f64" } else { "c64" };
                let params = format!("n={}", r.n);
                let med = |s: f64| Stat {
                    reps,
                    median: s * 1e-3,
                    min: f64::NAN,
                    max: f64::NAN,
                };
                csv.push(
                    Row::new(FAMILY, f, "tenferro-cpu-faer", dt, &params)
                        .stat(med(r.cpu.compute_ms))
                        .verdict(
                            Verdict::Oracle,
                            "reference for the CUDA result; median only",
                        ),
                );
                let note = format!(
                    "compute only; upload {:.3} ms download {:.3} ms; median only; bound {:.0e} (n*eps scaling of O(1) sums)",
                    r.cuda.upload_ms, r.cuda.download_ms, r.tol
                );
                csv.push(
                    Row::new(FAMILY, f, "tenferro-cuda", dt, &params)
                        .stat(med(r.cuda.compute_ms))
                        .dev("rel_maxnorm_vs_tenferro_cpu", r.rel_err, r.tol)
                        .note(&note),
                );
            }
        }
        Err(m) => {
            csv.push(Row::new(FAMILY, "dot_general+cholesky", "tenferro-cuda", "-", "").error(&m))
        }
    }
}

pub fn run(cfg: &Cfg, csv: &mut Csv) {
    let cx = Cx {
        reps: if cfg.full { 5 } else { 3 },
        budget: if cfg.full { 60.0 } else { 5.0 },
    };
    let cases: &[(usize, usize)] = if cfg.full {
        &[
            (500, 2000),
            (2000, 2000),
            (2000, 20000),
            (5000, 5000),
            (5000, 20000),
            (10000, 10000),
            (20000, 20000),
        ]
    } else {
        &[(200, 1000), (1000, 2000)]
    };
    let dev_mem = if cfg.cpu_only {
        None
    } else {
        mvmc_core::backend::device_report(mvmc_core::backend::BackendKind::Cuda(0))
            .ok()
            .and_then(|r| r.total_memory_bytes)
    };
    for &(n, k) in cases {
        run_case(cfg, &cx, csv, n, k, dev_mem);
    }
    if !cfg.cpu_only {
        gemm_cholesky(cfg, csv);
    }
}

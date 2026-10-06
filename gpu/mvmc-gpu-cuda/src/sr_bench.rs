//! SR stages (issue #421) on CUDA versus the C-order oracle and tenferro CPU.
//!
//! The same `mvmc_core::sr_backend::TenferroSr` code runs on the CUDA eager runtime
//! ([`Placement::Device`]); only the runtime differs. Timings are host-to-host per stage
//! (upload, compute with synchronization, download), the honest cost of the pathfinder where
//! each stage is an independent call; the constant CG operand is uploaded once per solve and
//! reported separately. Correctness uses relative max-norm bounds against the C-order result.

use std::time::Instant;

use mvmc_core::sr_backend::{
    COrderSr, CgSamples, Placement, RealView, SrAssembleInput, SrBackend, TenferroSr,
};

use crate::bench::{cpu_runtime, cuda_runtime};

/// Relative max-norm bounds (CUDA or tenferro CPU versus C order), per stage.
///
/// Gram and CG product: reductions of length `samples` (up to 6000) of O(1) terms,
/// `2 gamma_k` is below `1.4e-12` at `k = 6000`, so `1e-11` keeps a margin while still
/// catching layout/dtype defects. S/g assembly is elementwise (two roundings): `1e-15`.
/// Cholesky solve: `2 c n eps kappa` with the benchmark's `kappa <= 1.5e2` and `n <= 3000`
/// gives `~1e-9`; `1e-8` is used.
pub const TOL: [(&str, f64); 4] = [
    ("gram", 1e-11),
    ("assemble", 1e-15),
    ("solve", 1e-8),
    ("cg", 1e-11),
];

/// One measured size.
#[derive(Debug, Clone)]
pub struct SrRow {
    /// Number of parameters (active components).
    pub n: usize,
    /// Samples in the store.
    pub samples: usize,
    /// Median ms per stage for C order: gram, assemble, solve, cg.
    pub c_order: [f64; 4],
    /// Median ms per stage for tenferro CPU.
    pub tenferro_cpu: [f64; 4],
    /// Median ms per stage for tenferro CUDA.
    pub cuda: [f64; 4],
    /// Relative differences CUDA versus C order, per stage.
    pub cuda_err: [f64; 4],
    /// Relative differences tenferro CPU versus C order, per stage.
    pub cpu_err: [f64; 4],
    /// CUDA CG operand upload (once per solve), ms.
    pub cuda_cg_upload_ms: f64,
}

impl SrRow {
    /// Whether every CUDA and CPU difference is within its bound.
    pub fn ok(&self) -> bool {
        (0..4).all(|i| self.cuda_err[i] <= TOL[i].1 && self.cpu_err[i] <= TOL[i].1)
    }
}

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn time(reps: usize, mut f: impl FnMut()) -> f64 {
    f();
    median(
        (0..reps.max(1))
            .map(|_| {
                let t = Instant::now();
                f();
                t.elapsed().as_secs_f64() * 1e3
            })
            .collect(),
    )
}

fn rel(a: &[f64], b: &[f64]) -> f64 {
    let scale = b.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    a.iter()
        .zip(b)
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()))
        / scale
}

struct StageOutputs {
    gram: Vec<f64>,
    s: Vec<f64>,
    g: Vec<f64>,
    x: Vec<f64>,
    z: Vec<f64>,
    ms: [f64; 4],
}

fn stages(
    backend: &mut dyn SrBackend,
    store: &[f64],
    n: usize,
    samples: usize,
    reps: usize,
    spd_s: Option<&[f64]>,
) -> Result<StageOutputs, String> {
    let mut gram = vec![0.0; n * n];
    let e = |e: mvmc_core::sr_backend::SrBackendError| e.to_string();
    let gram_ms = {
        let mut err = None;
        let ms = time(reps, || {
            if let Err(x) = backend.gram_real(store, n, samples, &mut gram) {
                err = Some(x);
            }
        });
        if let Some(x) = err {
            return Err(format!("gram_real: {}", e(x)));
        }
        ms
    };
    // S/g from the C-order Gram, so every backend assembles identical inputs.
    let size = n + 1;
    let mut full = vec![0.0; size * size];
    let mut reference = vec![0.0; n * n];
    COrderSr::default()
        .gram_real(store, n, samples, &mut reference)
        .map_err(e)?;
    for j in 0..n {
        for i in 0..n {
            full[(i + 1) + (j + 1) * size] = reference[i + j * n] / samples as f64;
        }
    }
    let mut st = 99;
    let ho: Vec<f64> = (0..size).map(|_| lcg(&mut st)).collect();
    let map: Vec<usize> = (0..n).collect();
    let input = SrAssembleInput {
        oo: RealView::Real(&full),
        ho: RealView::Real(&ho),
        map: &map,
        ld: size,
        offset: 1,
        sta_del: 0.02,
        step_dt: 0.05,
    };
    let (mut s, mut g) = (vec![0.0; n * n], vec![0.0; n]);
    let mut err = None;
    let asm_ms = time(reps, || {
        if let Err(x) = backend.assemble_s_g(&input, &mut s, &mut g) {
            err = Some(x);
        }
    });
    if let Some(x) = err {
        return Err(format!("assemble_s_g: {}", e(x)));
    }
    // Solve a well-conditioned SPD system shared by every backend.
    let a = spd_s.expect("shared SPD matrix");
    let b: Vec<f64> = (0..n).map(|_| lcg(&mut st)).collect();
    let mut x = b.clone();
    let mut failed = false;
    let solve_ms = time(reps, || {
        let (mut sm, mut xv) = (a.to_vec(), b.clone());
        if backend.cholesky_solve(&mut sm, &mut xv, n).is_err() {
            failed = true;
        }
        x = xv;
    });
    if failed {
        return Err("cholesky_solve failed".to_string());
    }
    let xin: Vec<f64> = (0..n).map(|_| lcg(&mut st)).collect();
    let view = CgSamples {
        real: store,
        imag: &[],
        components: n,
        samples,
        version: 1,
    };
    let mut z = vec![0.0; n];
    let mut err = None;
    let cg_ms = time(reps, || {
        if let Err(x) = backend.cg_local_product(&view, &xin, &mut z) {
            err = Some(x);
        }
    });
    if let Some(x) = err {
        return Err(format!("cg_local_product: {}", e(x)));
    }
    Ok(StageOutputs {
        gram,
        s,
        g,
        x,
        z,
        ms: [gram_ms, asm_ms, solve_ms, cg_ms],
    })
}

/// SPD `A^T A / n + I` built from random data (column-major), condition number below ~1.5e2.
fn spd(n: usize) -> Vec<f64> {
    // Diagonally dominant instead of A^T A so construction is O(n^2) at n = 3000.
    let mut st = 5;
    let mut s = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..=j {
            let v = lcg(&mut st) / n as f64 * 8.0;
            s[i + j * n] = v;
            s[j + i * n] = v;
        }
        s[j + j * n] += 1.0 + lcg(&mut st).abs();
    }
    s
}

/// Measure one size on C order, tenferro CPU and tenferro CUDA.
pub fn run_case(ordinal: usize, n: usize, samples: usize, reps: usize) -> Result<SrRow, String> {
    let mut st = 7;
    let store: Vec<f64> = (0..n * samples).map(|_| lcg(&mut st)).collect();
    let spd_s = spd(n);
    let mut c = COrderSr::default();
    let mut cpu = TenferroSr::with_runtime(
        cpu_runtime()?,
        Placement::Host,
        "tenferro cpu-faer".to_string(),
    );
    let mut cuda = TenferroSr::with_runtime(
        cuda_runtime(ordinal)?,
        Placement::Device,
        "tenferro cuda".to_string(),
    );
    let reference = stages(&mut c, &store, n, samples, reps, Some(&spd_s))?;
    let on_cpu = stages(&mut cpu, &store, n, samples, reps, Some(&spd_s))?;
    let on_cuda = stages(&mut cuda, &store, n, samples, reps, Some(&spd_s))?;
    let errs = |o: &StageOutputs| {
        [
            rel(&o.gram, &reference.gram),
            rel(&o.s, &reference.s).max(rel(&o.g, &reference.g)),
            rel(&o.x, &reference.x),
            rel(&o.z, &reference.z),
        ]
    };
    // Constant-operand upload for the CG solve, timed on a fresh version.
    let view = CgSamples {
        real: &store,
        imag: &[],
        components: n,
        samples,
        version: 2,
    };
    let xin = vec![0.5; n];
    let mut z = vec![0.0; n];
    let t = Instant::now();
    cuda.cg_local_product(&view, &xin, &mut z)
        .map_err(|e| e.to_string())?;
    let first = t.elapsed().as_secs_f64() * 1e3;
    Ok(SrRow {
        n,
        samples,
        c_order: reference.ms,
        tenferro_cpu: on_cpu.ms,
        cuda: on_cuda.ms,
        cuda_err: errs(&on_cuda),
        cpu_err: errs(&on_cpu),
        cuda_cg_upload_ms: first - on_cuda.ms[3],
    })
}

/// Markdown table.
pub fn render(rows: &[SrRow]) -> String {
    let mut s = String::from(
        "| NPara | samples | stage | C order | tenferro CPU | tenferro CUDA | CUDA rel err | CPU rel err | bound | check |\n\
         |---:|---:|---|---:|---:|---:|---:|---:|---:|---|\n",
    );
    for r in rows {
        for (i, (name, tol)) in TOL.iter().enumerate() {
            s += &format!(
                "| {} | {} | {} | {:.2} | {:.2} | {:.2} | {:.1e} | {:.1e} | {:.0e} | {} |\n",
                r.n,
                r.samples,
                name,
                r.c_order[i],
                r.tenferro_cpu[i],
                r.cuda[i],
                r.cuda_err[i],
                r.cpu_err[i],
                tol,
                if r.cuda_err[i] <= *tol && r.cpu_err[i] <= *tol {
                    "ok"
                } else {
                    "FAIL"
                }
            );
        }
        s += &format!(
            "| {} | {} | cg operand upload (once per solve) | - | - | {:.2} | | | | |\n",
            r.n, r.samples, r.cuda_cg_upload_ms
        );
    }
    s
}

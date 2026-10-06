//! Family 1: batched Pfaffian + inverse (issue #423).
//!
//! Numerical verdict. The oracle is pfapack in C operation order (`Backend::CpuPfapack`).
//! The forward error of a backward-stable inverse is `c * n * eps * cond(A)`; with
//! `cond = ||A||_F ||A^-1||_F` taken from the oracle inverse of the actual plane and `c = 16`
//! (the bound of `tests/pfaffian_gate.rs`) every backend must satisfy, per plane,
//! `max(||inv - inv_oracle||_F / ||inv_oracle||_F, |pf - pf_oracle| / |pf_oracle|) <= bound`.
//! The reported `dev_value` is the worst observed `error / bound` over the planes, so the
//! bound is 1. Plane statuses must be identical. The CUDA result additionally has to satisfy
//! the independent invariants `A inv = I`, skew symmetry of `inv` and `Pf^2 = det` (bounds
//! `16 n eps cond`, `16 n eps cond`, `64 n eps cond`) on the first planes.
//!
//! Planes are independent, so the oracle and the CPU variants run on a prefix of at most
//! `K` planes (`work cap`); the CUDA variants run on all planes and are compared on the same
//! prefix. Timings of CPU variants therefore cover `K` planes; the `params` of every row state
//! the number of planes timed (`planes=`).

use mvmc_gpu::testkit::{invariants, random_planes, TestScalar};
use mvmc_gpu::{pfaffian_inverse_batched, Backend, BatchedPfaffian, PfScalar, PlaneStatus};
use mvmc_gpu_cuda::pfaffian::{with_session, PfSession};
use num_complex::Complex64;

use crate::csv::{measure, wall, Csv, Row, Stat, Verdict};
use crate::Cfg;

const FAMILY: &str = "pfaffian";
const C: f64 = 16.0;

fn fro_lanes<T: PfScalar>(a: &[T]) -> f64 {
    T::as_f64_lanes(a).iter().map(|x| x * x).sum::<f64>().sqrt()
}

struct Dev {
    ratio: f64,
    inv_rel: f64,
    pf_rel: f64,
    max_cond: f64,
    status_ok: bool,
}

/// Compare `(pf, inv, status)` of `k` planes against the oracle.
fn compare<T: TestScalar>(
    planes: &[T],
    n: usize,
    oracle: &BatchedPfaffian<T>,
    pf: &[T],
    inv: &[T],
    status: &[PlaneStatus],
) -> Dev {
    let nn = n * n;
    let k = oracle.planes();
    let mut d = Dev {
        ratio: 0.0,
        inv_rel: 0.0,
        pf_rel: 0.0,
        max_cond: 0.0,
        status_ok: status[..k] == oracle.status[..],
    };
    for p in 0..k {
        if oracle.status[p] != PlaneStatus::Ok {
            continue;
        }
        let a = &planes[p * nn..(p + 1) * nn];
        let ci = &oracle.inv[p * nn..(p + 1) * nn];
        let gi = &inv[p * nn..(p + 1) * nn];
        let cond = fro_lanes(a) * fro_lanes(ci);
        d.max_cond = d.max_cond.max(cond);
        let allowed = C * n as f64 * f64::EPSILON * cond;
        let diff = T::as_f64_lanes(gi)
            .iter()
            .zip(T::as_f64_lanes(ci))
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt();
        let inv_err = diff / fro_lanes(ci);
        let dpf: Vec<f64> = T::as_f64_lanes(&pf[p..p + 1])
            .iter()
            .zip(T::as_f64_lanes(&oracle.pf[p..p + 1]))
            .map(|(x, y)| x - y)
            .collect();
        let pf_err =
            dpf.iter().map(|x| x * x).sum::<f64>().sqrt() / fro_lanes(&oracle.pf[p..p + 1]);
        d.inv_rel = d.inv_rel.max(inv_err);
        d.pf_rel = d.pf_rel.max(pf_err);
        let r = inv_err.max(pf_err) / allowed;
        // NaN compares false: keep it
        d.ratio = if r.is_nan() { f64::NAN } else { d.ratio.max(r) };
    }
    d
}

fn dev_row(base: Row, d: &Dev) -> Row {
    let mut r = base.dev("err/(16*n*eps*cond)", d.ratio, 1.0).note(&format!(
        "max_inv_rel={:.2e} max_pf_rel={:.2e} max_cond={:.2e} status_equal={}",
        d.inv_rel, d.pf_rel, d.max_cond, d.status_ok
    ));
    if !d.status_ok {
        r = r.fail("plane status differs from the oracle");
    }
    r
}

struct Ctx<'a> {
    cfg: &'a Cfg,
    reps: usize,
    budget: f64,
}

fn case<T: TestScalar>(
    cx: &Ctx<'_>,
    csv: &mut Csv,
    session: Option<&PfSession<'_, '_>>,
    n: usize,
    batch: usize,
) {
    const NQP: usize = 8;
    let dtype = if T::COMPLEX { "c64" } else { "f64" };
    let count = NQP * batch;
    let elem = std::mem::size_of::<T>();
    let nn = n * n;
    let pstr = |planes: usize| format!("n={n};NQP={NQP};B={batch};planes={planes}");
    if nn * count * elem * 3 > cx.cfg.max_bytes {
        csv.push(
            Row::new(FAMILY, "pfaffian_inverse", "all", dtype, &pstr(count))
                .verdict(Verdict::Skipped, "exceeds --max-bytes"),
        );
        return;
    }
    let planes = random_planes::<T>(n, count, 1234 + n as u64);
    // oracle prefix: work cap in flops-like units
    let work_cap = if cx.cfg.full { 4.0e9 } else { 4.0e8 } / if T::COMPLEX { 4.0 } else { 1.0 };
    let k = ((work_cap / (n as f64).powi(3)) as usize).clamp(1, count);
    let prefix = &planes[..nn * k];
    let run = |b: &Backend<'_>| pfaffian_inverse_batched(b, prefix, n, k, 1);

    let mut oracle = None;
    let stat = match measure(1, cx.reps, cx.budget, || {
        wall(|| {
            oracle = Some(run(&Backend::CpuPfapack)?);
            Ok::<(), mvmc_gpu::Error>(())
        })
    }) {
        Ok(s) => s,
        Err(e) => {
            csv.push(
                Row::new(
                    FAMILY,
                    "pfaffian_inverse",
                    "cpu-pfapack-1thread",
                    dtype,
                    &pstr(k),
                )
                .error(&e.to_string()),
            );
            return;
        }
    };
    let oracle = oracle.expect("oracle");
    csv.push(
        Row::new(
            FAMILY,
            "pfaffian_inverse",
            "cpu-pfapack-1thread",
            dtype,
            &pstr(k),
        )
        .stat(stat)
        .verdict(Verdict::Oracle, "C-order oracle (pfapack)"),
    );

    let cpu_variant = |name: &str, backend: Backend<'_>, csv: &mut Csv| {
        let mut out = None;
        let r = measure(1, cx.reps, cx.budget, || {
            wall(|| {
                out = Some(run(&backend)?);
                Ok::<(), mvmc_gpu::Error>(())
            })
        });
        let base = Row::new(FAMILY, "pfaffian_inverse", name, dtype, &pstr(k));
        match r {
            Ok(s) => {
                let o = out.unwrap();
                csv.push(dev_row(
                    base.stat(s),
                    &compare(prefix, n, &oracle, &o.pf, &o.inv, &o.status),
                ));
            }
            Err(e) => csv.push(base.error(&e.to_string())),
        }
    };
    cpu_variant("cpu-pfapack-rayon", Backend::CpuPfapackRayon, csv);
    let w = (n as f64).powi(3) * k as f64;
    if w <= 3.5e7 {
        cpu_variant("tenferro-native-cpu", Backend::TenferroNative, csv);
    }
    if w <= 2.0e9 {
        cpu_variant("tenferro-extop-cpu", Backend::TenferroExtension, csv);
    }

    let Some(session) = session else { return };
    let mut last = None;
    let mut kern = Vec::new();
    let r = measure(1, cx.reps, cx.budget, || {
        let t = std::time::Instant::now();
        let (out, tm) = session.run_timed(&planes, n, count)?;
        let s = t.elapsed().as_secs_f64();
        kern.push(tm);
        last = Some(out);
        Ok::<f64, mvmc_gpu::Error>(s)
    });
    let ((pf, inv, status), total) = match (r, last) {
        (Ok(s), Some(o)) => (o, s),
        (Err(e), _) => {
            csv.push(
                Row::new(
                    FAMILY,
                    "pfaffian_inverse",
                    "cuda-total",
                    dtype,
                    &pstr(count),
                )
                .error(&e.to_string()),
            );
            return;
        }
        _ => return,
    };
    // measure() ran warm-up + reps; the kernel vector covers every run, keep the measured ones
    let measured: Vec<_> = kern.iter().skip(1).collect();
    let phase = |f: fn(&mvmc_gpu_cuda::pfaffian::Timings) -> f64| {
        Stat::from_samples(measured.iter().map(|t| f(t)).collect())
    };
    let d = compare(&planes, n, &oracle, &pf, &inv, &status);
    csv.push(dev_row(
        Row::new(
            FAMILY,
            "pfaffian_inverse",
            "cuda-total-with-transfers",
            dtype,
            &pstr(count),
        )
        .stat(total),
        &d,
    ));
    for (name, f) in [
        (
            "cuda-kernel-only",
            (|t: &mvmc_gpu_cuda::pfaffian::Timings| t.kernel_s) as fn(&_) -> f64,
        ),
        ("cuda-upload", |t| t.upload_s),
        ("cuda-download", |t| t.download_s),
    ] {
        csv.push(
            Row::new(FAMILY, "pfaffian_inverse", name, dtype, &pstr(count))
                .stat(phase(f))
                .note("phase of the cuda-total run"),
        );
    }

    // independent invariants on the first planes of the GPU result
    let m = k.min(2);
    let sub = BatchedPfaffian {
        n,
        nqp: m,
        batch: 1,
        pf: pf[..m].to_vec(),
        inv: inv[..nn * m].to_vec(),
        status: status[..m].to_vec(),
    };
    let inv_rep = invariants(&planes[..nn * m], &sub);
    let max_cond = (0..m)
        .map(|p| {
            fro_lanes(&planes[p * nn..(p + 1) * nn]) * fro_lanes(&oracle.inv[p * nn..(p + 1) * nn])
        })
        .fold(0.0, f64::max);
    let scale = C * n as f64 * f64::EPSILON * max_cond;
    let ratios = [
        inv_rep.max_identity_residual / scale,
        inv_rep.max_skew_defect / scale,
        inv_rep.max_pf2_det_rel / (4.0 * scale),
    ];
    let worst = ratios.iter().cloned().fold(
        0.0,
        |a: f64, b| if b.is_nan() { f64::NAN } else { a.max(b) },
    );
    csv.push(
        Row::new(
            FAMILY,
            "pfaffian_inverse_invariants",
            "cuda",
            dtype,
            &pstr(m),
        )
        .dev("invariant/bound", worst, 1.0)
        .note(&format!(
            "A*inv-I={:.2e} skew={:.2e} Pf2/det-1={:.2e} scale={:.2e}",
            inv_rep.max_identity_residual, inv_rep.max_skew_defect, inv_rep.max_pf2_det_rel, scale
        )),
    );
}

/// Zero-pivot / NaN planes must be reported, not silently accepted (CPU oracle semantics).
fn degenerate<T: TestScalar>(session: &PfSession<'_, '_>, csv: &mut Csv) {
    let n = 8;
    let nn = n * n;
    let dtype = if T::COMPLEX { "c64" } else { "f64" };
    let mut p = random_planes::<T>(n, 4, 3);
    p[nn..2 * nn].fill(T::zero());
    p[2 * nn + 4 * n + 1] = T::from_pair(f64::NAN, f64::NAN);
    for k in 0..n {
        p[3 * nn + 3 + k * n] = T::zero();
        p[3 * nn + k + 3 * n] = T::zero();
    }
    let row = Row::new(
        FAMILY,
        "pfaffian_inverse_degenerate",
        "cuda",
        dtype,
        "n=8;planes=4",
    );
    let c = pfaffian_inverse_batched(&Backend::CpuPfapack, &p, n, 2, 2);
    let g = session.run_timed(&p, n, 4);
    match (c, g) {
        (Ok(c), Ok(((_, _, gs), _))) => {
            if gs == c.status && gs[0] == PlaneStatus::Ok && gs[2] != PlaneStatus::Ok {
                csv.push(row.verdict(Verdict::Pass, "status codes identical to the oracle"));
            } else {
                csv.push(row.fail(&format!("cuda {gs:?} vs oracle {:?}", c.status)));
            }
        }
        (c, g) => csv.push(row.error(&format!("{:?} / {:?}", c.err(), g.err()))),
    }
}

pub fn run(cfg: &Cfg, csv: &mut Csv) {
    let (mut ns, mut batches): (Vec<usize>, Vec<usize>) = if cfg.full {
        (vec![16, 32, 64, 128, 256], vec![1, 8, 64, 512])
    } else {
        (vec![16, 64, 256], vec![1, 16])
    };
    if let Some(v) = &cfg.ns {
        ns = v.clone();
    }
    if let Some(v) = &cfg.batches {
        batches = v.clone();
    }
    let cx = Ctx {
        cfg,
        reps: if cfg.full { 7 } else { 3 },
        budget: if cfg.full { 20.0 } else { 4.0 },
    };
    let body = |session: Option<&PfSession<'_, '_>>, csv: &mut Csv| {
        for &n in &ns {
            for &b in &batches {
                case::<f64>(&cx, csv, session, n, b);
                case::<Complex64>(&cx, csv, session, n, b);
            }
        }
        if let Some(s) = session {
            degenerate::<f64>(s, csv);
            degenerate::<Complex64>(s, csv);
        }
    };
    if cfg.cpu_only {
        body(None, csv);
        return;
    }
    let r = with_session(0, |session| {
        // JIT/driver warm-up outside the measurements
        let probe = random_planes::<f64>(16, 8, 1);
        session.run_timed(&probe, 16, 8)?;
        body(Some(session), csv);
        Ok(())
    });
    if let Err(e) = r {
        csv.push(
            Row::new(FAMILY, "pfaffian_inverse", "cuda", "-", "")
                .error(&format!("CUDA session: {e}")),
        );
    }
}

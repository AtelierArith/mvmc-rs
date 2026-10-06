//! Family 3: device-resident lock-step sampler (issue #434) versus the CPU multichain runner.
//!
//! Numerical verdicts (per `L`, `Wc` checked walkers; Hubbard chain, real normal mode,
//! hopping updates; same logic as `tests/sampler_gate.rs`, see the #424 harness):
//! * `sampler_teacher_forced`: the device weights drive the CPU decisions. RNG state, words
//!   consumed and the electron configuration must be bit-identical (RNG parity is exact, never
//!   toleranced); draw mismatches and unexplained flips ("defects") must be 0 (hard gate). The
//!   worst relative weight deviation of each walker must be within a bound derived from the
//!   computation (see `weight_allowed`): `2 * 16 * (n + s) * eps * kappa / sqrt(w_min)` (`w_min`: smallest nonzero reference
//!   weight of the walker, the cancellation factor of the ratio's dot product), with `kappa =
//!   ||A||_F ||A^-1||_F` of the walker's Slater planes (max over planes, start and end of the
//!   run) and `s` the updates between recomputes. `dev_value` is the worst observed/allowed ratio.
//! * `sampler_free_run_vs_cpu`: the device decides. Identical decision bits, RNG state and
//!   configuration pass. A divergence is permitted only when the teacher-forced run located
//!   decision flips (numerical policy of AGENTS.md); otherwise it FAILs.
//! * `sampler_resident_inverse`: device-resident inverse (`<= 1e-6` relative Frobenius,
//!   sampler-gate bound for accumulated updates) and Pfaffians (`<= 1e-8`) against the CPU tables.
//!
//! Timing rows: one call is `vmc_make_sample_real` of every walker (warm-up, then reps); CPU
//! one thread, CPU multichain (all cores), CUDA pinned/pageable, plus device-time stage rows
//! of one profiled call.

use std::time::Instant;

use mvmc_core::device_sampler::{
    run_lockstep_real, LockstepOptions, LockstepStats, Teacher, TeacherReport,
};
use mvmc_core::run::{prepare_sampling_walker, PhysCalPreparation, SamplingWalker};
use mvmc_core::sampling::driver::{trace, vmc_make_sample_real};
use mvmc_gpu_cuda::device_sampler::{CudaSamplerService, ServiceTimings, TransferPath};
use sfmt19937::Sfmt19937Rng;

use crate::csv::{measure, Csv, Row, Stat, Verdict};
use crate::Cfg;

const FAMILY: &str = "sampler";

fn namelist(cfg: &Cfg, l: usize) -> std::path::PathBuf {
    cfg.inputs.join(format!("hubbard_chain_L{l}/namelist.def"))
}

/// `w_count` walkers sharing the wavefunction of one preparation, seeds `1 + w`.
/// `seed = None`: all walkers share the wavefunction of one preparation (seed 1) and have
/// SFMT streams `1 + w` (the benchmark setup). `seed = Some(s)`: every walker is prepared with
/// `prepare_phys_cal_with_seed_offset(seed s, offset w)` (the `sampler_gate` setup).
fn make_walkers(
    cfg: &Cfg,
    l: usize,
    w_count: usize,
    samples: i64,
    seed: Option<i64>,
) -> Result<Vec<SamplingWalker>, String> {
    if let Some(s) = seed {
        return (0..w_count)
            .map(|w| {
                let mut prep = mvmc_core::prepare_phys_cal_with_seed_offset(
                    namelist(cfg, l),
                    None,
                    "real",
                    Some(s),
                    true,
                    w,
                )?;
                prep.data.modpara.nvmc_sample = samples;
                prepare_sampling_walker(prep).map_err(|e| e.to_string())
            })
            .collect();
    }
    let mut base = mvmc_core::prepare_phys_cal_with_seed_offset(
        namelist(cfg, l),
        None,
        "real",
        Some(1),
        true,
        0,
    )
    .map_err(|e| e.to_string())?;
    base.data.modpara.nvmc_sample = samples;
    let threads = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(w_count);
    let per = w_count.div_ceil(threads);
    let mut out: Vec<Option<SamplingWalker>> = (0..w_count).map(|_| None).collect();
    std::thread::scope(|scope| {
        for (t, chunk) in out.chunks_mut(per).enumerate() {
            let base = &base;
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
    Ok(out.into_iter().map(|w| w.expect("walker built")).collect())
}

/// Largest `kappa = ||A||_F ||A^-1||_F` over the QP planes of the walker's current state
/// (`A` is the skew-symmetric matrix of the current electron configuration, `A^-1` the stored
/// inverse table; the same condition number as in the Pfaffian family).
fn plane_kappa_max(w: &SamplingWalker) -> f64 {
    let sm = &w.state.slater_matrix;
    let elm = &sm.slater_elm_real;
    let n_site = w.data.modpara.nsite as usize;
    let n_elec = w.data.modpara.nelec as usize;
    let n = 2 * n_elec;
    let cfg = &w.state.electron_config.tmp_ele_idx;
    let r: Vec<usize> = (0..n)
        .map(|k| cfg[k] as usize + if k < n_elec { 0 } else { n_site })
        .collect();
    let inv = sm.inv_m_real.as_slice();
    let mut worst = 0.0f64;
    for q in 0..elm.n_qp_full() {
        let mut fa = 0.0;
        for &ri in &r {
            for &rj in &r {
                let v = elm.get(q, ri, rj);
                fa += v * v;
            }
        }
        let fi: f64 = inv[q * (n * n + 1)..q * (n * n + 1) + n * n]
            .iter()
            .map(|v| v * v)
            .sum();
        worst = worst.max((fa * fi).sqrt());
    }
    worst
}

/// Allowed relative deviation of a Metropolis weight `w = |Pf ratio|^2` between two
/// implementations of the same recompute plus rank-1/2 updates, derived from the computation:
/// the Pfaffian ratio is a length-`n` dot product of an inverse row with the new Slater row,
/// and each of the `s` accepted updates since the last recompute perturbs the inverse by
/// backward-stable roundoff amplified by `kappa`, so the relative error of the ratio is at most
/// `c (n + s) eps kappa / |ratio|`; the weight squares the ratio (factor 2). `c = 16` as in the Pfaffian
/// family bound. `s` is bounded by the proposals per walker between recomputes
/// (`proposals / max(1, recomputes - walkers)` per walker, rounded up), the worst case in which
/// every proposal in that window is accepted.
fn weight_allowed(
    w: &SamplingWalker,
    kappa: f64,
    st: &LockstepStats,
    walkers: usize,
    decisions: &[(f64, f64)],
) -> f64 {
    // cancellation: the dot product's relative error is (sum |inv_ij a_j|)/|ratio| times eps,
    // and sum |inv_ij a_j| <= ||inv row|| ||a|| <= kappa; |ratio| = sqrt(weight) of the proposal.
    // The worst proposal (smallest nonzero reference weight) bounds every proposal's factor.
    let w_min = decisions
        .iter()
        .map(|d| d.0)
        .filter(|&x| x > 0.0)
        .fold(f64::INFINITY, f64::min);
    let cancel = (1.0 / w_min.sqrt()).max(1.0);
    let n = 2.0 * w.data.modpara.nelec as f64;
    let per_walker = st.proposals.div_ceil(walkers.max(1));
    let windows = (st.recomputes.saturating_sub(walkers))
        .max(1)
        .div_ceil(walkers.max(1));
    let s = per_walker.div_ceil(windows) as f64;
    2.0 * 16.0 * (n + s) * f64::EPSILON * kappa * cancel
}

fn rel_diff(a: &[f64], b: &[f64]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    let den: f64 = b.iter().map(|y| y * y).sum();
    (num / den.max(f64::MIN_POSITIVE)).sqrt()
}

fn cpu_call(walkers: &mut [SamplingWalker], threads: usize) -> Result<f64, String> {
    let t = Instant::now();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let err = std::sync::Mutex::new(None);
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
                if let Err(e) = vmc_make_sample_real(&w.data, &mut w.state, &mut w.rng) {
                    *err.lock().unwrap() = Some(format!("{e:?}"));
                }
            });
        }
    });
    if let Some(e) = err.into_inner().unwrap() {
        return Err(e);
    }
    Ok(t.elapsed().as_secs_f64())
}

type CpuRef = Vec<(Vec<(f64, f64)>, SamplingWalker)>;

fn cpu_reference(
    cfg: &Cfg,
    l: usize,
    walkers: usize,
    samples: i64,
    seed: Option<i64>,
) -> Result<CpuRef, String> {
    let ws = make_walkers(cfg, l, walkers, samples, seed)?;
    ws.into_iter()
        .map(|mut sw| {
            trace::start_decisions();
            vmc_make_sample_real(&sw.data, &mut sw.state, &mut sw.rng)
                .map_err(|e| format!("{e:?}"))?;
            Ok((trace::finish_decisions(), sw))
        })
        .collect()
}

/// Correctness of one transfer path at one `L`; pushes the three verdict rows.
fn check_path<T: TransferPath>(
    csv: &mut Csv,
    cfg: &Cfg,
    l: usize,
    wc: usize,
    samples: i64,
    seed: Option<i64>,
    reference: &CpuRef,
    variant: &str,
    mut service: CudaSamplerService<T>,
    mut fresh: impl FnMut() -> CudaSamplerService<T>,
) {
    let params = format!(
        "L={l};Wc={wc};NVMCSample={samples};setup={}",
        seed.map_or("shared-wavefunction".to_string(), |s| format!("seed{s}"))
    );
    let row = |f: &str| Row::new(FAMILY, f, variant, "f64", &params);
    // teacher forced
    let teachers: Vec<Teacher> = reference
        .iter()
        .map(|(d, _)| Teacher {
            reference: d.clone(),
            weight_abs: 1e-12,
            weight_rel: 1e-10,
        })
        .collect();
    let mut ws = match make_walkers(cfg, l, wc, samples, seed) {
        Ok(w) => w,
        Err(m) => return csv.push(row("sampler_teacher_forced").error(&m)),
    };
    // condition numbers of the Slater planes at the start (before the run)
    let kappa_start: Vec<f64> = ws.iter().map(plane_kappa_max).collect();
    let teach = run_lockstep_real(
        &mut ws,
        &mut service,
        LockstepOptions {
            teachers,
            ..LockstepOptions::default()
        },
    );
    let flips: usize;
    match teach {
        Err(m) => return csv.push(row("sampler_teacher_forced").error(&m)),
        Ok((runs, lstats)) => {
            let mut reports: Vec<TeacherReport> = Vec::new();
            let mut walker_ratio = 0.0f64;
            let (mut kmax, mut allowed_min) = (0.0f64, f64::INFINITY);
            let mut exact = true;
            let mut why = String::new();
            for (w, run) in runs.iter().enumerate() {
                if let Err(m) = &run.stats {
                    exact = false;
                    why += &format!("walker {w}: {m}; ");
                    continue;
                }
                if ws[w].rng.state_snapshot() != reference[w].1.rng.state_snapshot()
                    || ws[w].rng.words_consumed() != reference[w].1.rng.words_consumed()
                {
                    exact = false;
                    why += &format!("walker {w}: RNG state/draw count differs; ");
                }
                if ws[w].state.electron_config.tmp_ele_idx
                    != reference[w].1.state.electron_config.tmp_ele_idx
                {
                    exact = false;
                    why += &format!("walker {w}: configuration differs; ");
                }
                let rep = run.teacher.clone().unwrap_or_default();
                let kappa = kappa_start[w].max(plane_kappa_max(&reference[w].1));
                let allowed = weight_allowed(&ws[w], kappa, &lstats, wc, &reference[w].0);
                walker_ratio = walker_ratio.max(rep.max_weight_rel / allowed);
                kmax = kmax.max(kappa);
                allowed_min = allowed_min.min(allowed);
                reports.push(rep);
            }
            let sum = |f: fn(&TeacherReport) -> usize| reports.iter().map(f).sum::<usize>();
            flips = sum(|r| r.flips);
            let defects = sum(|r| r.defects);
            let draws = sum(|r| r.draw_mismatches) + sum(|r| r.overrun);
            let compared: usize = reports.iter().map(|r| r.compared).sum();
            let expected: usize = reference.iter().map(|(d, _)| d.len()).sum();
            let max_rel = reports.iter().map(|r| r.max_weight_rel).fold(0.0, f64::max);
            let max_abs = reports.iter().map(|r| r.max_weight_abs).fold(0.0, f64::max);
            let min_margin = reports
                .iter()
                .map(|r| r.min_margin)
                .fold(f64::INFINITY, f64::min);
            let note = format!(
                "decisions={compared}/{expected} flips={flips} defects={defects} draw_mismatches={draws} max_abs_dw={max_abs:.2e} min_margin={min_margin:.2e} max_rel_dw={max_rel:.2e} max_kappa={kmax:.2e} min_allowed={allowed_min:.2e}"
            );
            let mut r = row("sampler_teacher_forced")
                .dev("max_walker(rel_weight_dev/allowed)", walker_ratio, 1.0)
                .note(&note);
            if !exact || defects > 0 || draws > 0 || compared != expected {
                r = r.fail(&format!("{note}; {why}"));
            }
            csv.push(r);
        }
    }

    // free run
    let mut ws = match make_walkers(cfg, l, wc, samples, seed) {
        Ok(w) => w,
        Err(m) => return csv.push(row("sampler_free_run_vs_cpu").error(&m)),
    };
    let mut service = fresh();
    match run_lockstep_real(&mut ws, &mut service, LockstepOptions::default()) {
        Err(m) => csv.push(row("sampler_free_run_vs_cpu").error(&m)),
        Ok((runs, stats)) => {
            let mut identical = true;
            let mut first = String::new();
            for (w, run) in runs.iter().enumerate() {
                let (ref_dec, ref_w) = &reference[w];
                let same_dec =
                    run.decisions.len() == ref_dec.len()
                        && run.decisions.iter().zip(ref_dec).all(|(a, b)| {
                            a.1.to_bits() == b.1.to_bits() && (a.0 > a.1) == (b.0 > b.1)
                        });
                let same_state = ws[w].rng.state_snapshot() == ref_w.rng.state_snapshot()
                    && ws[w].state.electron_config.ele_idx == ref_w.state.electron_config.ele_idx;
                if !(same_dec && same_state) {
                    identical = false;
                    first += &format!(
                        "walker {w} decisions_equal={same_dec} rng_and_config_equal={same_state}; "
                    );
                }
            }
            let note = format!(
                "proposals={} accepts={} recomputes={} passes={}",
                stats.proposals, stats.accepts, stats.recomputes, stats.passes
            );
            let r = row("sampler_free_run_vs_cpu");
            csv.push(if identical {
                r.verdict(Verdict::Pass, &format!("bit-identical trajectory; {note}"))
            } else if flips > 0 {
                r.verdict(
                    Verdict::Pass,
                    &format!("diverged after {flips} located teacher-forced decision flips (permitted by the numerical policy); {first}{note}"),
                )
            } else {
                r.fail(&format!("diverged with no located flip; {first}{note}"))
            });
            // resident inverse and pf vs the CPU tables
            let n_qp = ws[0].state.slater_matrix.slater_elm_real.n_qp_full();
            let measure_resident =
                |service: &mut CudaSamplerService<T>| -> Result<(f64, f64, String), String> {
                    let (mut worst_inv, mut worst_pf) = (0.0f64, 0.0f64);
                    let mut per_walker = String::new();
                    for w in 0..wc.min(2) {
                        let (inv, pf) = service.download_walker(w)?;
                        let cpu_inv = reference[w].1.state.slater_matrix.inv_m_real.as_slice();
                        let cpu_pf = &reference[w].1.state.slater_matrix.pf_m_real;
                        let nn = inv.len() / n_qp;
                        let mut flat = Vec::with_capacity(inv.len());
                        for q in 0..n_qp {
                            flat.extend_from_slice(&cpu_inv[q * (nn + 1)..q * (nn + 1) + nn]);
                        }
                        let ri = rel_diff(&inv, &flat);
                        let rp = rel_diff(&pf, cpu_pf);
                        per_walker += &format!(" w{w}:inv={ri:.1e},pf={rp:.1e}");
                        let bad: Vec<usize> = (0..n_qp)
                            .filter(|&q| {
                                rel_diff(&inv[q * nn..(q + 1) * nn], &flat[q * nn..(q + 1) * nn])
                                    > 1e-8
                            })
                            .collect();
                        if let Some(&q) = bad.first() {
                            per_walker += &format!(
                                " [{} of {n_qp} QPs differ; qp{q}: cpu_pf={:.3e} dev_pf={:.3e}]",
                                bad.len(),
                                cpu_pf[q],
                                pf[q]
                            );
                        }
                        worst_inv = worst_inv.max(ri);
                        worst_pf = worst_pf.max(rp);
                    }
                    Ok((worst_inv, worst_pf, per_walker))
                };
            let ratio_of = |wi: f64, wp: f64| (wi / 1e-6).max(wp / 1e-8);
            let metric = "max(inv_rel/1e-6;pf_rel/1e-8)";
            match measure_resident(&mut service) {
                Err(m) => csv.push(row("sampler_resident_inverse").error(&m)),
                Ok((wi, wp, detail)) => {
                    let first = ratio_of(wi, wp);
                    if first <= 1.0 {
                        csv.push(
                            row("sampler_resident_inverse")
                                .dev(metric, first, 1.0)
                                .note(&format!("inv_rel={wi:.2e} pf_rel={wp:.2e};{detail}")),
                        );
                    } else {
                        // Distinguish a numerical defect from an unsynchronized download:
                        // download again after the device had time to finish.
                        std::thread::sleep(std::time::Duration::from_millis(500));
                        match measure_resident(&mut service) {
                            Err(m) => csv.push(row("sampler_resident_inverse").error(&m)),
                            Ok((wi2, wp2, detail2)) => {
                                let settled = ratio_of(wi2, wp2);
                                csv.push(
                                    row("sampler_resident_inverse")
                                        .dev(metric, settled, 1.0)
                                        .note(&format!("settled download (0.5 s later): inv_rel={wi2:.2e} pf_rel={wp2:.2e};{detail2}")),
                                );
                                if settled <= 1.0 {
                                    csv.push(row("sampler_download_synchronization").fail(&format!(
                                        "download_walker right after the run returned stale resident data (inv_rel={wi:.2e} pf_rel={wp:.2e};{detail}) but matched the CPU after 0.5 s: the resident state is not synchronized at the end of the run (race, not a numerical error)"
                                    )));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

fn device_call<T: TransferPath>(
    mut service: CudaSamplerService<T>,
    ws: &mut [SamplingWalker],
    profile: bool,
) -> Result<(f64, ServiceTimings, LockstepStats), String> {
    service.profile = profile;
    let t = Instant::now();
    let (runs, stats) = run_lockstep_real(ws, &mut service, LockstepOptions::default())?;
    let secs = t.elapsed().as_secs_f64();
    for r in &runs {
        r.stats.as_ref().map_err(|e| e.clone())?;
    }
    Ok((secs, service.timings, stats))
}

fn timing_path<T: TransferPath>(
    csv: &mut Csv,
    cfg: &Cfg,
    l: usize,
    w: usize,
    samples: i64,
    reps: usize,
    variant: &str,
    mut mk: impl FnMut() -> Result<CudaSamplerService<T>, String>,
) {
    let params = format!("L={l};W={w};NVMCSample={samples}");
    let base = |f: &str| Row::new(FAMILY, f, variant, "f64", &params);
    let mut ws = match make_walkers(cfg, l, w, samples, None) {
        Ok(v) => v,
        Err(m) => return csv.push(base("sampler_end_to_end").error(&m)),
    };
    let r = measure(1, reps, 120.0, || {
        let svc = mk()?;
        device_call(svc, &mut ws, false).map(|x| x.0)
    });
    match r {
        Ok(s) => csv.push(
            base("sampler_end_to_end")
                .stat(s)
                .note("timing only; correctness rows above"),
        ),
        Err(m) => return csv.push(base("sampler_end_to_end").error(&m)),
    }
    // one profiled call: device-time stage rows
    match mk().and_then(|svc| device_call(svc, &mut ws, true)) {
        Ok((_, tm, st)) => {
            let stage = |name: &str, s: f64, note: &str| {
                base(name)
                    .stat(Stat::single(s))
                    .note(&format!("{note}; one profiled call; passes={}", st.passes))
            };
            csv.push(stage(
                "sampler_propose_ratios",
                tm.dev_propose_s,
                "device event time",
            ));
            csv.push(stage(
                "sampler_accept_updates",
                tm.dev_accept_s,
                "device event time",
            ));
            csv.push(stage(
                "sampler_recompute_slow_lane",
                tm.dev_slow_s,
                "device time of assemble+Pfaffian+commit",
            ));
            csv.push(stage("sampler_host_staging", tm.stage_s, "host"));
            csv.push(stage("sampler_upload", tm.upload_s, "host enqueue"));
            csv.push(stage("sampler_launch", tm.launch_s, "host enqueue"));
            csv.push(stage("sampler_wait_fast_lane", tm.wait_s, "host blocked"));
        }
        Err(m) => csv.push(base("sampler_stages").error(&m)),
    }
}

pub fn run(cfg: &Cfg, csv: &mut Csv) {
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    let (check_ls, wc, check_samples): (&[usize], usize, i64) = if cfg.full {
        (&[16, 32, 64, 128, 256], 4, 20)
    } else {
        (&[16, 32], 2, 20)
    };
    let (time_ls, time_ws, hops, reps): (&[usize], &[usize], usize, usize) = if cfg.full {
        (&[16, 32, 64, 128, 256], &[1, 8, 64, 512, 4096], 3000, 3)
    } else {
        (&[16, 32], &[1, 8, 64], 1500, 2)
    };
    let dev_mem = if cfg.cpu_only {
        None
    } else {
        mvmc_core::backend::device_report(mvmc_core::backend::BackendKind::Cuda(0))
            .ok()
            .and_then(|r| r.total_memory_bytes)
    };
    let host_gb = cfg
        .max_gb
        .unwrap_or_else(|| crate::sr::mem_available_gb() * 0.5);
    let limit_gb = dev_mem.map_or(host_gb, |d| host_gb.min(d as f64 / 1e9 * 0.8));

    // numerical verdicts
    if !cfg.cpu_only {
        for &l in check_ls {
            if !namelist(cfg, l).exists() {
                csv.push(
                    Row::new(
                        FAMILY,
                        "sampler_teacher_forced",
                        "cuda",
                        "f64",
                        &format!("L={l}"),
                    )
                    .verdict(
                        Verdict::Skipped,
                        "input missing (run_all.sh generates it with mvmc --dry-run)",
                    ),
                );
                continue;
            }
            for seed in [Some(7), None] {
                let reference = match cpu_reference(cfg, l, wc, check_samples, seed) {
                    Ok(r) => r,
                    Err(m) => {
                        csv.push(
                            Row::new(
                                FAMILY,
                                "sampler_cpu_reference",
                                "cpu",
                                "f64",
                                &format!("L={l}"),
                            )
                            .error(&m),
                        );
                        continue;
                    }
                };
                match CudaSamplerService::new_pinned(0) {
                    Ok(svc) => check_path(
                        csv,
                        cfg,
                        l,
                        wc,
                        check_samples,
                        seed,
                        &reference,
                        "cuda-pinned",
                        svc,
                        || CudaSamplerService::new_pinned(0).expect("service"),
                    ),
                    Err(m) => csv.push(
                        Row::new(
                            FAMILY,
                            "sampler_teacher_forced",
                            "cuda-pinned",
                            "f64",
                            &format!("L={l}"),
                        )
                        .error(&m),
                    ),
                }
                match CudaSamplerService::new_pageable(0) {
                    Ok(svc) => check_path(
                        csv,
                        cfg,
                        l,
                        wc,
                        check_samples,
                        seed,
                        &reference,
                        "cuda-pageable",
                        svc,
                        || CudaSamplerService::new_pageable(0).expect("service"),
                    ),
                    Err(m) => csv.push(
                        Row::new(
                            FAMILY,
                            "sampler_teacher_forced",
                            "cuda-pageable",
                            "f64",
                            &format!("L={l}"),
                        )
                        .error(&m),
                    ),
                }
            }
        }
    }

    if cfg.checks_only {
        return;
    }
    // timings (reference)
    for &l in time_ls {
        if !namelist(cfg, l).exists() {
            csv.push(
                Row::new(
                    FAMILY,
                    "sampler_end_to_end",
                    "all",
                    "f64",
                    &format!("L={l}"),
                )
                .verdict(Verdict::Skipped, "input missing"),
            );
            continue;
        }
        let samples = ((hops / l) as i64 - 10).max(1);
        let n2 = 2.0 * l as f64;
        let per_walker_gb = (8.0 * n2 * n2 * 24.0 + 8.0 * (l * l) as f64 * 8.0) / 1e9;
        for &w in time_ws {
            let params = format!("L={l};W={w};NVMCSample={samples}");
            if (w * l * l) as f64 > cfg.work_cap {
                csv.push(
                    Row::new(FAMILY, "sampler_end_to_end", "all", "f64", &params).verdict(
                        Verdict::Skipped,
                        &format!(
                            "W*L^2 = {} exceeds the profile work cap {:.1e} (raise with --work-cap; the full profile keeps the 2 h budget)",
                            w * l * l,
                            cfg.work_cap
                        ),
                    ),
                );
                continue;
            }
            if per_walker_gb * w as f64 > limit_gb {
                csv.push(
                    Row::new(FAMILY, "sampler_end_to_end", "all", "f64", &params).verdict(
                        Verdict::Skipped,
                        &format!(
                            "~{:.1} GB walker state exceeds limit {:.1} GB (--max-gb)",
                            per_walker_gb * w as f64,
                            limit_gb
                        ),
                    ),
                );
                continue;
            }
            let hops_pw = (samples as usize + 10) * l;
            let note = format!("hops_per_walker={hops_pw}");
            if w <= 64 {
                match make_walkers(cfg, l, w, samples, None)
                    .and_then(|mut ws| measure(1, reps, 120.0, || cpu_call(&mut ws, 1)))
                {
                    Ok(s) => csv.push(
                        Row::new(FAMILY, "sampler_end_to_end", "cpu-1thread", "f64", &params)
                            .stat(s)
                            .verdict(Verdict::Oracle, &note),
                    ),
                    Err(m) => csv.push(
                        Row::new(FAMILY, "sampler_end_to_end", "cpu-1thread", "f64", &params)
                            .error(&m),
                    ),
                }
            }
            match make_walkers(cfg, l, w, samples, None)
                .and_then(|mut ws| measure(1, reps, 120.0, || cpu_call(&mut ws, cores)))
            {
                Ok(s) => csv.push(
                    Row::new(
                        FAMILY,
                        "sampler_end_to_end",
                        "cpu-multichain",
                        "f64",
                        &params,
                    )
                    .stat(s)
                    .verdict(
                        Verdict::Oracle,
                        &format!("{note}; threads={}", cores.min(w)),
                    ),
                ),
                Err(m) => csv.push(
                    Row::new(
                        FAMILY,
                        "sampler_end_to_end",
                        "cpu-multichain",
                        "f64",
                        &params,
                    )
                    .error(&m),
                ),
            }
            if cfg.cpu_only {
                continue;
            }
            timing_path(csv, cfg, l, w, samples, reps, "cuda-pinned", || {
                CudaSamplerService::new_pinned(0)
            });
            if w <= 512 {
                timing_path(csv, cfg, l, w, samples, reps, "cuda-pageable", || {
                    CudaSamplerService::new_pageable(0)
                });
            }
        }
    }
}

//! Optional CUDA gate for the device-resident lock-step sampler (issue #434). Ignored by default;
//! `MVMC_RS_CUDA_GATE=1` requires a device (hard failure without one), anything else skips
//! explicitly. Runs the real normal-mode sampler of `mvmc_core` with the Pfaffian stages on the
//! device and compares with the CPU sampler: RNG state and draw count exactly, teacher-forced
//! decision flips against the CPU decisions, and the resident inverses after the run.

use std::path::{Path, PathBuf};

use mvmc_core::backend::{
    cuda_device_count, cuda_gate_decision, CudaGateDecision, CUDA_GATE_VARIABLE,
};
use mvmc_core::device_sampler::{
    run_lockstep_real, DeviceService, LockstepOptions, Teacher, TeacherReport,
};
use mvmc_core::run::{prepare_sampling_walker, SamplingWalker};
use mvmc_core::sampling::driver::{trace, vmc_make_sample_real};
use mvmc_gpu_cuda::device_sampler::{CudaSamplerService, PageableTransfer, PinnedTransfer};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn gate() -> bool {
    mvmc_gpu_cuda::install();
    let requested = std::env::var(CUDA_GATE_VARIABLE).ok();
    match cuda_gate_decision(requested.as_deref(), cuda_device_count()) {
        CudaGateDecision::SkippedNoDevice(why) => {
            eprintln!("sampler-gate: ExplicitSkip: skipped, no device ({why})");
            false
        }
        CudaGateDecision::FailNoDevice(why) => {
            panic!("sampler-gate: {CUDA_GATE_VARIABLE} requested but no device: {why}")
        }
        CudaGateDecision::Run => true,
    }
}

struct Case {
    name: &'static str,
    namelist: PathBuf,
    opt: Option<PathBuf>,
    seed: i64,
    samples: i64,
}

fn heisenberg() -> Case {
    let root = repo()
        .join("extern/Julia-mVMC/test/integration/reference/heisenberg_chain_real/physcal_ref");
    Case {
        name: "heisenberg-exchange",
        namelist: root.join("inputs/namelist.def"),
        opt: Some(root.join("zqp_opt.dat")),
        seed: 1,
        samples: 40,
    }
}

fn hubbard(l: usize) -> Case {
    Case {
        name: "hubbard-hopping",
        namelist: repo().join(format!(
            "benchmark/hubbard_chain/inputs/hubbard_chain_L{l}/namelist.def"
        )),
        opt: None,
        seed: 7,
        samples: 20,
    }
}

fn walker(c: &Case, offset: usize) -> SamplingWalker {
    let mut prep = mvmc_core::prepare_phys_cal_with_seed_offset(
        &c.namelist,
        c.opt.as_deref(),
        "real",
        Some(c.seed),
        true,
        offset,
    )
    .unwrap();
    prep.data.modpara.nvmc_sample = c.samples;
    prepare_sampling_walker(prep).unwrap()
}

fn rel_diff(a: &[f64], b: &[f64]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    let den: f64 = b.iter().map(|y| y * y).sum();
    (num / den.max(f64::MIN_POSITIVE)).sqrt()
}

/// CPU references: stats, decisions, final walker (state + rng).
fn cpu_reference(c: &Case, walkers: usize) -> Vec<(Vec<(f64, f64)>, SamplingWalker)> {
    (0..walkers)
        .map(|w| {
            let mut sw = walker(c, w);
            trace::start_decisions();
            vmc_make_sample_real(&sw.data, &mut sw.state, &mut sw.rng).unwrap();
            (trace::finish_decisions(), sw)
        })
        .collect()
}

fn teacher_report<S: DeviceService>(
    c: &Case,
    walkers: usize,
    service: &mut S,
    reference: &[(Vec<(f64, f64)>, SamplingWalker)],
) -> Vec<TeacherReport> {
    let teachers: Vec<Teacher> = reference
        .iter()
        .map(|(d, _)| Teacher {
            reference: d.clone(),
            weight_abs: 1e-12,
            weight_rel: 1e-10,
        })
        .collect();
    let mut ws: Vec<SamplingWalker> = (0..walkers).map(|w| walker(c, w)).collect();
    let (runs, _) = run_lockstep_real(
        &mut ws,
        service,
        LockstepOptions {
            teachers,
            ..LockstepOptions::default()
        },
    )
    .expect("device lock-step run");
    for (w, run) in runs.iter().enumerate() {
        run.stats.as_ref().expect("walker ran");
        // teacher forcing follows the CPU decisions, so the RNG stream must be identical
        assert_eq!(
            ws[w].rng.state_snapshot(),
            reference[w].1.rng.state_snapshot(),
            "walker {w}: RNG state"
        );
        assert_eq!(
            ws[w].rng.words_consumed(),
            reference[w].1.rng.words_consumed()
        );
        assert_eq!(
            ws[w].state.electron_config.tmp_ele_idx,
            reference[w].1.state.electron_config.tmp_ele_idx
        );
    }
    runs.into_iter().map(|r| r.teacher.unwrap()).collect()
}

fn check_case(c: &Case, walkers: usize) {
    let reference = cpu_reference(c, walkers);
    let proposals: usize = reference.iter().map(|(d, _)| d.len()).sum();

    // teacher-forced, pinned and pageable transfer paths
    let mut pinned = CudaSamplerService::new_pinned(0).expect("pinned service");
    let reports = teacher_report(c, walkers, &mut pinned, &reference);
    let mut pageable = CudaSamplerService::new_pageable(0).expect("pageable service");
    let reports_pg = teacher_report(c, walkers, &mut pageable, &reference);
    for (w, (a, b)) in reports.iter().zip(&reports_pg).enumerate() {
        assert_eq!(a.compared, reference[w].0.len());
        assert_eq!(a.draw_mismatches, 0, "walker {w}: draws");
        assert_eq!(a.overrun, 0);
        assert_eq!(a.defects, 0, "walker {w}: defects {a:?}");
        assert_eq!(a.flips, b.flips);
    }
    let flips: usize = reports.iter().map(|r| r.flips).sum();
    let max_abs = reports.iter().map(|r| r.max_weight_abs).fold(0.0, f64::max);
    let max_rel = reports.iter().map(|r| r.max_weight_rel).fold(0.0, f64::max);
    let min_margin = reports
        .iter()
        .map(|r| r.min_margin)
        .fold(f64::INFINITY, f64::min);
    eprintln!(
        "{} W={walkers}: {proposals} decisions, flips {flips}, max |dw| {max_abs:.2e}, max rel dw {max_rel:.2e}, min margin {min_margin:.2e}",
        c.name
    );
    assert!(
        max_rel < 1e-8,
        "device weights differ from CPU by {max_rel}"
    );

    // free run: the device decides; with no flips the trajectories coincide
    let mut ws: Vec<SamplingWalker> = (0..walkers).map(|w| walker(c, w)).collect();
    let mut service = CudaSamplerService::new_pinned(0).expect("service");
    let (runs, stats) = run_lockstep_real(&mut ws, &mut service, LockstepOptions::default())
        .expect("device lock-step run");
    eprintln!(
        "  free run: {} passes, {} proposals, {} accepts, {} recomputes, max batch {}; timings {:?}",
        stats.passes, stats.proposals, stats.accepts, stats.recomputes, stats.max_batch,
        service.timings
    );
    for (w, run) in runs.iter().enumerate() {
        let (ref_dec, ref_walker) = &reference[w];
        assert_eq!(
            run.decisions.len(),
            ref_dec.len(),
            "walker {w}: decision count"
        );
        let same = run
            .decisions
            .iter()
            .zip(ref_dec)
            .all(|(a, b)| a.1.to_bits() == b.1.to_bits() && (a.0 > a.1) == (b.0 > b.1));
        assert!(same, "walker {w}: decision sequence diverged (a flip)");
        assert_eq!(ws[w].rng.state_snapshot(), ref_walker.rng.state_snapshot());
        assert_eq!(
            ws[w].state.electron_config.ele_idx,
            ref_walker.state.electron_config.ele_idx
        );
    }
    // resident inverses vs the CPU tables (mVMC convention); includes accepted-move updates
    let n_qp = ws[0].state.slater_matrix.slater_elm_real.n_qp_full();
    for w in 0..walkers.min(2) {
        let (inv, pf) = service.download_walker(w).expect("download");
        let cpu_inv = reference[w].1.state.slater_matrix.inv_m_real.as_slice();
        let nn = inv.len() / n_qp;
        // the CPU table has one pad slot per QP
        let mut cpu_flat = Vec::with_capacity(inv.len());
        for q in 0..n_qp {
            cpu_flat.extend_from_slice(&cpu_inv[q * (nn + 1)..q * (nn + 1) + nn]);
        }
        let err = rel_diff(&inv, &cpu_flat);
        let pf_err = rel_diff(&pf, &reference[w].1.state.slater_matrix.pf_m_real);
        eprintln!("  walker {w}: resident invM rel diff {err:.2e}, pf rel diff {pf_err:.2e}");
        assert!(err < 1e-6, "walker {w}: resident inverse differs by {err}");
        assert!(pf_err < 1e-8, "walker {w}: pf differs by {pf_err}");
    }
}

#[test]
#[ignore = "optional CUDA gate: needs a device (MVMC_RS_CUDA_GATE=1 to require it)"]
fn device_sampler_hopping_matches_cpu_sampler() {
    if !gate() {
        return;
    }
    check_case(&hubbard(16), 4);
    check_case(&hubbard(32), 3);
}

#[test]
#[ignore = "optional CUDA gate: needs a device (MVMC_RS_CUDA_GATE=1 to require it)"]
fn device_sampler_exchange_matches_cpu_sampler() {
    if !gate() {
        return;
    }
    check_case(&heisenberg(), 4);
}

#[test]
#[ignore = "optional CUDA gate: needs a device (MVMC_RS_CUDA_GATE=1 to require it)"]
fn pinned_helper_raw_copies_work_on_tenferro_raw_session_addresses() {
    use mvmc_gpu_cuda::transfer::{PinnedKind, PinnedPool, TransferStream};
    use tenferro_gpu::cuda::raw::{KernelArg, LaunchConfig, NvrtcOptions};
    use tenferro_gpu::cuda::{cuda_devices, with_cuda_exec_session, CudaBackend};
    use tenferro_tensor::BackendSessionHost;
    if !gate() {
        return;
    }
    const SRC: &str = r#"
extern "C" __global__ void twice(double* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = 2.0 * x[i] + 1.0;
}"#;
    let devices = cuda_devices().expect("devices");
    let mut backend = CudaBackend::new(devices[0].id()).expect("backend");
    let n = 1000usize;
    let outcome = backend.with_backend_session(|session| {
        with_cuda_exec_session(session, |exec| {
            exec.with_raw("sampler-gate.interop", |sess| {
                // device buffer owned by tenferro's allocator; its address from the raw session
                let mut dev = sess.alloc_bytes(n * 8, "interop")?;
                let mut ptr = std::ptr::null_mut::<std::ffi::c_void>();
                dev.with_ptr(|p| ptr = p);
                // a separate cudarc handle of the same primary context performs the copies
                let ctx = cudarc::driver::CudaContext::new(0).expect("ctx");
                // SAFETY: only this test uses the context; ordering is by host waits below.
                unsafe { mvmc_gpu_cuda::transfer::disable_event_tracking(&ctx) };
                let pool = PinnedPool::new(&ctx);
                let ts = TransferStream::new(&ctx).expect("stream");
                let mut up = pool.take(n * 8, PinnedKind::Cached).expect("pinned");
                for (i, v) in up.as_f64_mut()[..n].iter_mut().enumerate() {
                    *v = i as f64;
                }
                // SAFETY: `ptr` is a live device allocation of n*8 bytes owned by `dev`; `up` is
                // pinned and untouched until the wait.
                unsafe { ts.upload_raw(up.as_f64().as_ptr().cast(), ptr as u64, n * 8) }
                    .expect("upload")
                    .wait()
                    .expect("wait");
                // tenferro compiles and launches the kernel on its own stream
                let module = sess.compile_nvrtc(SRC, &NvrtcOptions::default())?;
                let f = module.function("twice")?;
                // SAFETY: ABI (double*, int); `dev` outlives the synchronize.
                unsafe {
                    sess.launch(
                        &f,
                        LaunchConfig::flat(n as u32, 128, 0)?,
                        &[KernelArg::workspace(&dev), KernelArg::i32(n as i32)],
                    )?;
                }
                sess.synchronize()?;
                let mut down = pool.take(n * 8, PinnedKind::Cached).expect("pinned");
                // SAFETY: as above, device to host.
                unsafe {
                    ts.download_raw(ptr as u64, down.as_f64_mut().as_mut_ptr().cast(), n * 8)
                }
                .expect("download")
                .wait()
                .expect("wait");
                for (i, v) in down.as_f64()[..n].iter().enumerate() {
                    assert_eq!(*v, 2.0 * i as f64 + 1.0, "element {i}");
                }
                Ok(())
            })
        })
    });
    outcome.expect("cuda session").expect("interop");
}

#[allow(dead_code)]
fn _link(_: PinnedTransfer, _: PageableTransfer) {}

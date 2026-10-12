//! Single-owner, device-resident variational optimization (Julia GPU contract).
//!
//! Setup uploads one model, then configuration initialization, sampling, local
//! energy/derivatives, direct SR, and parameter refresh execute on one stream.
//! Only initialization status and final parameters/energy history are downloaded.

mod model;
#[cfg(test)]
mod tests;
use crate::interval::{
    IntervalGeometry, IntervalHost, IntervalSampler, ResidentInterval, INTERVAL_KERNELS,
    INTERVAL_MODULE,
};
use model::Model;
use mvmc_core::InitialDef;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use tenferro_ext::raw_exec::{with_raw_exec, Arg, DeviceBuffer, Op, RawExec, RawModule};
use tenferro_gpu::cuda::{cuda_devices, CudaBackend};
use tenferro_tensor::{BackendSession, BackendSessionHost};

const SOURCE: &str = include_str!("opt/kernels.cu");
const KERNELS: &[&str] = &[
    "opt_fill",
    "opt_initial",
    "opt_initial_commit",
    "opt_slater",
    "opt_counts",
    "opt_derivatives",
    "opt_energy_terms",
    "opt_energy",
    "opt_means",
    "opt_variance",
    "opt_cutoff",
    "opt_system",
    "opt_update",
    "opt_gauge",
];

thread_local! {
    // Compiled modules and native library handles are keyed by the exact backend
    // runtime identity. Retain that identity across calls, as CUDA.jl does. Only
    // the execution context is cached: each run still owns fresh RNG and arrays.
    static RUNTIMES: RefCell<HashMap<usize, CudaBackend>> = RefCell::new(HashMap::new());
}

fn runtime(device: usize) -> Result<CudaBackend, String> {
    if let Some(backend) = RUNTIMES.with(|r| r.borrow().get(&device).cloned()) {
        return Ok(backend);
    }
    let devices = cuda_devices().map_err(|e| e.to_string())?;
    let id = devices
        .get(device)
        .ok_or("CUDA device ordinal out of bounds")?
        .id();
    let backend = CudaBackend::new(id).map_err(|e| e.to_string())?;
    RUNTIMES.with(|r| r.borrow_mut().insert(device, backend.clone()));
    Ok(backend)
}

/// Explicit CUDA runner settings. Omitted sampling lengths follow modpara.def.
#[derive(Clone, Debug)]
pub struct OptConfig {
    pub device: usize,
    pub nsteps: usize,
    pub walkers: Option<usize>,
    pub warmup: Option<usize>,
    pub interval: Option<usize>,
    pub seed: Option<i64>,
    pub initial_def: InitialDef,
}

impl Default for OptConfig {
    fn default() -> Self {
        Self {
            device: 0,
            nsteps: 20,
            walkers: None,
            warmup: None,
            interval: None,
            seed: None,
            initial_def: InitialDef::Auto,
        }
    }
}

/// Final host result. Energy is total local energy averaged over the ensemble
/// before each update, as in Julia's GPU Opt runner. Parameters are after the
/// final update and gauge normalization; no CPU optimizer state is constructed.
#[derive(Debug)]
pub struct OptResult {
    pub energies: Vec<f64>,
    pub parameters: Vec<f64>,
    pub seed: u64,
    pub walkers: usize,
}

/// Optimize real normal-orbital Gutzwiller/Jastrow inputs with hopping-only
/// sampling and direct SR. Unsupported modes fail before allocating CUDA state.
pub fn run_gpu_para_opt_from_namelist(
    path: impl AsRef<Path>,
    config: OptConfig,
) -> Result<OptResult, String> {
    let model = Model::prepare(path.as_ref(), &config)?;
    let mut backend = runtime(config.device)?;
    let mut state = backend.with_backend_session(|s| State::prepare(s, model, &config))?;
    backend.with_backend_session(|s| state.initialize(s))?;
    backend.with_backend_session(|s| state.run(s))?;
    backend.with_backend_session(|s| state.result(s))
}

struct Buffers {
    parameters: DeviceBuffer,
    flags: DeviceBuffer,
    pa: DeviceBuffer,
    pb: DeviceBuffer,
    sa: DeviceBuffer,
    sb: DeviceBuffer,
    cs: DeviceBuffer,
    cc: DeviceBuffer,
    ss: DeviceBuffer,
    columns: DeviceBuffer,
    xyz: DeviceBuffer,
    coefficients: DeviceBuffer,
    operators: DeviceBuffer,
    energies: DeviceBuffer,
    lookup: DeviceBuffer,
    counts: DeviceBuffer,
    overlaps: DeviceBuffer,
    logs: DeviceBuffer,
    derivatives: DeviceBuffer,
    values: DeviceBuffer,
    local_energy: DeviceBuffer,
    mean: DeviceBuffer,
    gram: DeviceBuffer,
    ho: DeviceBuffer,
    variance: DeviceBuffer,
    cutoff: DeviceBuffer,
    rhs: DeviceBuffer,
    info: DeviceBuffer,
    workspace: DeviceBuffer,
    workspace_len: usize,
    history: DeviceBuffer,
    initial_tape: DeviceBuffer,
    initial_scratch: DeviceBuffer,
    candidate: DeviceBuffer,
    remaining: DeviceBuffer,
}

struct State {
    model: Model,
    resident: ResidentInterval,
    buffers: Buffers,
    warmup: usize,
    interval: usize,
    steps: usize,
}

fn i(n: usize) -> Arg {
    Arg::I(n as i32)
}
fn gpu_error(message: impl Into<String>) -> tenferro_tensor::Error {
    tenferro_tensor::Error::unsupported("resident Opt", message.into())
}
fn upload<T: Copy>(exec: &RawExec<'_, '_>, values: &[T]) -> tenferro_tensor::Result<DeviceBuffer> {
    let b = exec.alloc_bytes(std::mem::size_of_val(values).max(1))?;
    // SAFETY: allocation has exactly the source slice's size; scope synchronizes.
    unsafe {
        exec.upload(exec.ptr(&b)?, values)?;
    }
    Ok(b)
}

impl State {
    fn prepare<B: BackendSession + ?Sized>(
        backend: &mut B,
        model: Model,
        config: &OptConfig,
    ) -> Result<Self, String> {
        let mp = &model.data.modpara;
        let convert =
            |n: i64| usize::try_from(n).map_err(|_| "negative sampling length".to_string());
        let w = config
            .walkers
            .map(Ok)
            .unwrap_or_else(|| convert(mp.nvmc_sample))?;
        let interval = config.interval.map(Ok).unwrap_or_else(|| {
            convert(mp.nvmc_interval)
                .and_then(|n| n.checked_mul(model.ns).ok_or("interval overflow".into()))
        })?;
        let warmup = config.warmup.map(Ok).unwrap_or_else(|| {
            convert(mp.nvmc_warmup).and_then(|n| {
                convert(mp.nvmc_interval).and_then(|s| {
                    n.checked_mul(s)
                        .and_then(|x| x.checked_mul(model.ns))
                        .ok_or("warmup overflow".into())
                })
            })
        })?;
        if w == 0
            || config.nsteps == 0
            || [
                w,
                interval,
                warmup,
                config.nsteps,
                model.np,
                model.nq,
                model.ns,
            ]
            .iter()
            .any(|&n| n > i32::MAX as usize)
        {
            return Err("invalid GPU Opt dimensions".into());
        }
        let n = 2 * model.ne;
        // Kernel indexing is i32 within a walker; allocations use checked usize.
        for dims in [
            [w, model.np, 1],
            [w, model.energies.len(), 1],
            [model.nq, 2 * model.ns, 2 * model.ns],
            [w, model.nq, 1],
            [w, warmup.max(interval), 5],
            [model.np, model.np, 1],
            [w, 2 * model.ne + 2 * model.ns + 16, 1],
        ] {
            if dims
                .iter()
                .try_fold(1usize, |a, &b| a.checked_mul(b))
                .filter(|&v| v <= i32::MAX as usize)
                .is_none()
            {
                return Err("GPU Opt index capacity exceeded".into());
            }
        }
        w.checked_mul(model.nq)
            .and_then(|v| v.checked_mul(n * n * 8))
            .ok_or("GPU inverse size overflow")?;
        let geom = IntervalGeometry {
            walkers: w,
            n,
            ne: model.ne,
            nsite: model.ns,
            nqp: model.nq,
            ngutz: model.ng,
            nproj: model.ng + model.nj,
        };
        if geom.shared_bytes() > 48 * 1024 {
            return Err("GPU sampler shared-memory capacity exceeded".into());
        }
        let placeholder: Vec<i32> = (0..w * n).map(|a| (a % model.ne) as i32).collect();
        let slater = vec![0.; model.nq * 4 * model.ns * model.ns];
        let host = IntervalHost {
            ele: &placeholder,
            inv: &[],
            pf: &[],
            slater: &slater,
            weights: &model.weights,
            gutz: &model.gutz,
            jast: &model.jast,
            params: &model.parameters[..geom.nproj],
        };
        let sampler =
            IntervalSampler::prepare_for_rebuild(backend, geom, &host, warmup.max(interval))?;
        let resident = ResidentInterval::from_sampler(backend, sampler, model.seed)?;
        let buffers = with_raw_exec(backend, "mvmc.opt.prepare", |exec| {
            let np = model.np;
            let alloc = |count: usize, size: usize| exec.alloc_bytes((count * size).max(1));
            let gram = alloc(np * np, 8)?;
            // SAFETY: this is the allocated square FP64 matrix on this context.
            let workspace_len = unsafe { exec.dpotrf_workspace(exec.ptr(&gram)?, np)? };
            Ok(Buffers {
                parameters: upload(exec, &model.parameters)?,
                flags: upload(exec, &model.flags)?,
                pa: upload(exec, &model.pa)?,
                pb: upload(exec, &model.pb)?,
                sa: upload(exec, &model.sa)?,
                sb: upload(exec, &model.sb)?,
                cs: upload(exec, &model.cs)?,
                cc: upload(exec, &model.cc)?,
                ss: upload(exec, &model.ss)?,
                columns: upload(exec, &model.columns)?,
                xyz: upload(exec, &model.coordinates)?,
                coefficients: upload(exec, &model.coefficients)?,
                operators: upload(exec, &model.operators)?,
                energies: upload(exec, &model.energies)?,
                lookup: alloc(w * 2 * model.ns, 4)?,
                counts: alloc(w * geom.nproj, 4)?,
                overlaps: alloc(w, 8)?,
                logs: alloc(w, 8)?,
                derivatives: alloc(w * np, 8)?,
                values: alloc(w * model.energies.len(), 8)?,
                local_energy: alloc(w, 8)?,
                mean: alloc(np + 1, 8)?,
                gram,
                ho: alloc(np, 8)?,
                variance: alloc(np, 8)?,
                cutoff: alloc(1, 8)?,
                rhs: alloc(np, 8)?,
                info: alloc(2, 4)?,
                workspace: alloc(workspace_len, 8)?,
                workspace_len,
                history: alloc(config.nsteps, 8)?,
                initial_tape: alloc(w * (n + 2 * model.ns + 16), 4)?,
                initial_scratch: alloc(w * (n + model.ns), 4)?,
                candidate: alloc(w, 4)?,
                remaining: alloc(1, 4)?,
            })
        })
        .map_err(|e| e.to_string())?;
        Ok(Self {
            model,
            resident,
            buffers,
            warmup,
            interval,
            steps: config.nsteps,
        })
    }

    fn slater(&self, exec: &RawExec<'_, '_>, module: &RawModule) -> tenferro_tensor::Result<()> {
        let b = &self.buffers;
        let m = &self.model;
        let p = |b: &DeviceBuffer| exec.ptr(b).map(Arg::P);
        unsafe {
            exec.launch(
                module.function("opt_slater")?,
                ((m.nq * m.ns * m.ns).div_ceil(256) as u32, 1, 1),
                256,
                0,
                &[
                    p(&self.resident.sampler.slater)?,
                    p(&b.parameters)?,
                    p(&b.pa)?,
                    p(&b.pb)?,
                    p(&b.sa)?,
                    p(&b.sb)?,
                    p(&b.cs)?,
                    p(&b.cc)?,
                    p(&b.ss)?,
                    i(m.ns),
                    i(m.nq),
                ],
            )
        }
    }

    fn counts(&self, exec: &RawExec<'_, '_>, module: &RawModule) -> tenferro_tensor::Result<()> {
        let b = &self.buffers;
        let m = &self.model;
        let s = &self.resident.sampler;
        let p = |b: &DeviceBuffer| exec.ptr(b).map(Arg::P);
        unsafe {
            exec.launch(
                module.function("opt_counts")?,
                (s.geom.walkers as u32, 1, 1),
                256,
                0,
                &[
                    p(&s.ele)?,
                    p(&s.occ)?,
                    p(&b.lookup)?,
                    p(&b.counts)?,
                    p(&s.gutz)?,
                    p(&s.jast)?,
                    p(&b.parameters)?,
                    p(&s.pf)?,
                    p(&s.weights)?,
                    p(&b.overlaps)?,
                    p(&b.logs)?,
                    i(m.ns),
                    i(m.ne),
                    i(m.nq),
                    i(m.ng),
                    i(m.nj),
                ],
            )
        }
    }

    fn initialize<B: BackendSession + ?Sized>(&mut self, backend: &mut B) -> Result<(), String> {
        let w = self.resident.sampler.geom.walkers;
        with_raw_exec(backend, "mvmc.opt.initial_model", |exec| {
            let module = exec.module(SOURCE, KERNELS)?;
            self.slater(exec, &module)?;
            unsafe {
                exec.launch(
                    module.function("opt_fill")?,
                    (w.div_ceil(256) as u32, 1, 1),
                    256,
                    0,
                    &[
                        Arg::P(exec.ptr(&self.resident.sampler.rebuild)?),
                        i(1),
                        i(w),
                    ],
                )
            }
        })
        .map_err(|e| e.to_string())?;
        for _ in 0..100 {
            let remaining = with_raw_exec(backend, "mvmc.opt.initialize", |exec| {
                let module = exec.module(SOURCE, KERNELS)?;
                let interval = exec.module(INTERVAL_MODULE, INTERVAL_KERNELS)?;
                let m = &self.model;
                let words = w * (2 * m.ne + 2 * m.ns + 16);
                self.resident
                    .rng
                    .fill_u32(exec, &mut self.buffers.initial_tape, words)?;
                let s = &self.resident.sampler;
                let b = &self.buffers;
                let g = s.geom;
                let p = |b: &DeviceBuffer| exec.ptr(b).map(Arg::P);
                let (nt, shared) = crate::pfaffian::launch_geometry(g.n, 8, 1);
                let mut remaining = [0i32];
                unsafe {
                    exec.zero_bytes(exec.ptr(&b.remaining)?, 4)?;
                    exec.launch(
                        module.function("opt_initial")?,
                        (w.div_ceil(128) as u32, 1, 1),
                        128,
                        0,
                        &[
                            p(&s.ele)?,
                            p(&b.initial_scratch)?,
                            p(&b.initial_tape)?,
                            p(&s.rebuild)?,
                            p(&b.candidate)?,
                            i(m.ns),
                            i(m.ne),
                            i(w),
                        ],
                    )?;
                    self.counts(exec, &module)?;
                    exec.launch(
                        interval.function("interval_resident_factor")?,
                        ((w * m.nq) as u32, 1, 1),
                        nt,
                        shared,
                        &[
                            p(&s.rebuild)?,
                            p(&self.resident.error)?,
                            p(&s.ele)?,
                            p(&s.slater)?,
                            p(&s.planes)?,
                            p(&s.pfinv_w)?,
                            p(&s.pfinv_out)?,
                            p(&s.pfinv_pf)?,
                            p(&s.pfinv_st)?,
                            i(g.n),
                            i(m.ne),
                            i(m.ns),
                            i(m.nq),
                        ],
                    )?;
                    exec.launch(
                        module.function("opt_initial_commit")?,
                        (w as u32, 1, 1),
                        256,
                        0,
                        &[
                            p(&s.rebuild)?,
                            p(&b.candidate)?,
                            p(&s.pfinv_st)?,
                            p(&s.pfinv_out)?,
                            p(&s.pfinv_pf)?,
                            p(&s.weights)?,
                            p(&b.logs)?,
                            p(&s.inv)?,
                            p(&s.pf)?,
                            p(&s.logip)?,
                            p(&s.aw)?,
                            p(&b.remaining)?,
                            i(g.n),
                            i(m.nq),
                        ],
                    )?;
                    exec.download(&mut remaining, exec.ptr(&b.remaining)?)?;
                }
                Ok(remaining[0])
            })
            .map_err(|e| e.to_string())?;
            if remaining == 0 {
                return Ok(());
            }
        }
        Err("GPU initialization failed after 100 attempts".into())
    }

    fn observe(&self, exec: &RawExec<'_, '_>, module: &RawModule) -> tenferro_tensor::Result<()> {
        self.counts(exec, module)?;
        let b = &self.buffers;
        let m = &self.model;
        let s = &self.resident.sampler;
        let w = s.geom.walkers;
        let p = |b: &DeviceBuffer| exec.ptr(b).map(Arg::P);
        unsafe {
            exec.launch(
                module.function("opt_derivatives")?,
                ((m.np * w).div_ceil(8) as u32, 1, 1),
                256,
                0,
                &[
                    p(&b.derivatives)?,
                    p(&b.counts)?,
                    p(&b.lookup)?,
                    p(&b.columns)?,
                    p(&b.xyz)?,
                    p(&b.coefficients)?,
                    p(&s.inv)?,
                    p(&s.pf)?,
                    p(&s.weights)?,
                    p(&b.overlaps)?,
                    i(m.np),
                    i(m.ng + m.nj),
                    i(m.ns),
                    i(m.ne),
                    i(m.nq),
                    i(w),
                    p(&self.resident.error)?,
                ],
            )?;
            let terms = m.energies.len();
            if terms > 0 {
                exec.launch(
                    module.function("opt_energy_terms")?,
                    ((terms * w).div_ceil(8) as u32, 1, 1),
                    256,
                    0,
                    &[
                        p(&b.values)?,
                        p(&b.operators)?,
                        p(&s.ele)?,
                        p(&s.occ)?,
                        p(&b.lookup)?,
                        p(&b.overlaps)?,
                        p(&s.gutz)?,
                        p(&s.jast)?,
                        p(&b.parameters)?,
                        p(&s.slater)?,
                        p(&s.weights)?,
                        p(&s.inv)?,
                        p(&s.pf)?,
                        i(m.ns),
                        i(m.ne),
                        i(m.nq),
                        i(m.ng),
                        i(m.nj),
                        i(terms),
                        i(w),
                        p(&self.resident.error)?,
                    ],
                )?;
            }
            exec.launch(
                module.function("opt_energy")?,
                (w.div_ceil(128) as u32, 1, 1),
                128,
                0,
                &[
                    p(&b.local_energy)?,
                    p(&b.values)?,
                    p(&b.energies)?,
                    i(terms),
                    i(w),
                    p(&self.resident.error)?,
                ],
            )?;
        }
        Ok(())
    }

    fn update(
        &self,
        exec: &RawExec<'_, '_>,
        module: &RawModule,
        step: usize,
    ) -> tenferro_tensor::Result<()> {
        let b = &self.buffers;
        let m = &self.model;
        let np = m.np;
        let s = &self.resident.sampler;
        let w = s.geom.walkers;
        let p = |b: &DeviceBuffer| exec.ptr(b).map(Arg::P);
        let mp = &m.data.modpara;
        unsafe {
            exec.launch(
                module.function("opt_means")?,
                ((np + 1) as u32, 1, 1),
                256,
                0,
                &[
                    p(&b.mean)?,
                    p(&b.derivatives)?,
                    p(&b.local_energy)?,
                    i(np),
                    i(w),
                ],
            )?;
            exec.dgemm(
                Op::N,
                Op::T,
                np,
                np,
                w,
                1.,
                exec.ptr(&b.derivatives)?,
                np,
                exec.ptr(&b.derivatives)?,
                np,
                0.,
                exec.ptr(&b.gram)?,
                np,
            )?;
            exec.dgemm(
                Op::N,
                Op::N,
                np,
                1,
                w,
                1.,
                exec.ptr(&b.derivatives)?,
                np,
                exec.ptr(&b.local_energy)?,
                w,
                0.,
                exec.ptr(&b.ho)?,
                np,
            )?;
            exec.launch(
                module.function("opt_variance")?,
                (np.div_ceil(256) as u32, 1, 1),
                256,
                0,
                &[p(&b.variance)?, p(&b.gram)?, p(&b.mean)?, i(np), i(w)],
            )?;
            exec.launch(
                module.function("opt_cutoff")?,
                (1, 1, 1),
                1,
                0,
                &[
                    p(&b.cutoff)?,
                    p(&b.variance)?,
                    p(&b.mean)?,
                    p(&b.history)?,
                    i(step),
                    i(np),
                    Arg::D(mp.dsr_opt_red_cut),
                ],
            )?;
            exec.launch(
                module.function("opt_system")?,
                (np.div_ceil(128) as u32, 1, 1),
                128,
                0,
                &[
                    p(&b.gram)?,
                    p(&b.rhs)?,
                    p(&b.ho)?,
                    p(&b.mean)?,
                    p(&b.variance)?,
                    p(&b.cutoff)?,
                    p(&b.flags)?,
                    i(np),
                    i(w),
                    Arg::D(mp.dsr_opt_sta_del),
                    Arg::D(mp.dsr_opt_step_dt),
                ],
            )?;
            exec.dpotrf_solve(
                exec.ptr(&b.gram)?,
                exec.ptr(&b.rhs)?,
                np,
                exec.ptr(&b.workspace)?,
                b.workspace_len,
                exec.ptr(&b.info)?,
            )?;
            exec.launch(
                module.function("opt_update")?,
                (np.div_ceil(128) as u32, 1, 1),
                128,
                0,
                &[
                    p(&b.parameters)?,
                    p(&b.rhs)?,
                    p(&b.info)?,
                    p(&self.resident.error)?,
                    i(np),
                ],
            )?;
            exec.launch(
                module.function("opt_gauge")?,
                (1, 1, 1),
                1,
                0,
                &[
                    p(&b.parameters)?,
                    p(&s.params)?,
                    i(np),
                    i(m.ng + m.nj),
                    Arg::I(i32::from(m.shift)),
                    p(&self.resident.error)?,
                ],
            )?;
            self.slater(exec, module)?;
            exec.launch(
                module.function("opt_fill")?,
                (w.div_ceil(256) as u32, 1, 1),
                256,
                0,
                &[p(&s.rebuild)?, i(1), i(w)],
            )?;
        }
        let interval = exec.module(INTERVAL_MODULE, INTERVAL_KERNELS)?;
        self.resident.enqueue_rebuild(exec, &interval)
    }

    fn run<B: BackendSession + ?Sized>(&mut self, backend: &mut B) -> Result<(), String> {
        // One raw scope for the complete warmup/optimization series. No intermediate
        // host reads, transfers, or per-step stream synchronization.
        with_raw_exec(backend, "mvmc.opt.run", |exec| {
            let module = exec.module(SOURCE, KERNELS)?;
            self.resident.enqueue_random(exec, self.warmup)?;
            for step in 0..self.steps {
                self.resident.enqueue_random(exec, self.interval)?;
                self.observe(exec, &module)?;
                self.update(exec, &module, step)?;
            }
            Ok(())
        })
        .map_err(|e| e.to_string())
    }

    fn result<B: BackendSession + ?Sized>(&mut self, backend: &mut B) -> Result<OptResult, String> {
        with_raw_exec(backend, "mvmc.opt.result", |exec| {
            let mut error = [0i32];
            unsafe {
                exec.download(&mut error, exec.ptr(&self.resident.error)?)?;
            }
            if error[0] != 0 {
                return Err(gpu_error(format!(
                    "GPU Opt failed (device status {}: 1=rebuild, 2=energy, 3=SR, 4=parameters)",
                    error[0]
                )));
            }
            let mut energies = vec![0.; self.steps];
            let mut parameters = vec![0.; self.model.np];
            unsafe {
                exec.download(&mut energies, exec.ptr(&self.buffers.history)?)?;
                exec.download(&mut parameters, exec.ptr(&self.buffers.parameters)?)?;
            }
            Ok(OptResult {
                energies,
                parameters,
                seed: self.model.seed,
                walkers: self.resident.sampler.geom.walkers,
            })
        })
        .map_err(|e| e.to_string())
    }
}

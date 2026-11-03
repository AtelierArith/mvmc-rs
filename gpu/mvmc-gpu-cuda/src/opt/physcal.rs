//! Device-resident fixed-parameter PhysCal with the Julia GPU walker contract.
//!
//! This explicit API uses independently initialized XORWOW walkers. It does not
//! replace the CPU/C-order or legacy single-SFMT-chain `vmc_phys_cal` runner.

use super::*;
use std::path::PathBuf;

#[cfg(test)]
mod tests;

const PHYS_SOURCE: &str = include_str!("physcal.cu");
const PHYS_KERNELS: &[&str] = &["physcal_reduce"];

/// Sampling lengths count proposals per walker; omitted lengths follow modpara.
#[derive(Clone, Debug, Default)]
pub struct PhysCalConfig {
    pub device: usize,
    pub walkers: Option<usize>,
    pub nsamples: Option<usize>,
    pub warmup: Option<usize>,
    pub interval: Option<usize>,
    pub seed: Option<i64>,
    pub initial_def: InitialDef,
    pub opt_para: Option<PathBuf>,
}

/// Ensemble means in parsed operator order, averaged over measurement windows.
/// With TwoBodyGEx, the shared parser supplies C's canonical one-body list.
#[derive(Debug)]
pub struct PhysCalResult {
    pub energy: f64,
    pub green_one: Vec<f64>,
    pub green_two: Vec<f64>,
    pub green_two_ex: Vec<f64>,
    pub seed: u64,
    pub walkers: usize,
    pub nsamples: usize,
}

/// Run real normal-orbital Gutzwiller/Jastrow PhysCal without Lanczos or SR.
/// Input parsing, fresh initialization, sampling and final host results belong
/// to this call. RNG and intermediate configurations/observables stay on CUDA.
pub fn run_gpu_phys_cal_from_namelist(
    path: impl AsRef<Path>,
    config: PhysCalConfig,
) -> Result<PhysCalResult, String> {
    let mut sampling = OptConfig {
        device: config.device,
        walkers: config.walkers,
        warmup: config.warmup,
        interval: config.interval,
        seed: config.seed,
        initial_def: config.initial_def,
        nsteps: 1,
    };
    let model = Model::prepare_physcal(path.as_ref(), &sampling, config.opt_para.as_deref())?;
    sampling.nsteps = config.nsamples.map(Ok).unwrap_or_else(|| {
        usize::try_from(model.data.modpara.n_data_qty_smp)
            .map_err(|_| "negative PhysCal window count".to_string())
    })?;
    let operators = Operators::prepare(&model)?;
    with_runtime(config.device, |engine| {
        let mut state = State::prepare_with_derivatives(engine, model, &sampling, false)?;
        // Julia's PhysCal runner uses immediate updates; its Opt runner opts
        // into delayed queues. Retain that distinction for proposal decisions.
        state.resident.delayed_updates = false;
        state.initialize(engine)?;
        let mut measurement = Measurement::prepare(engine, operators, &state)?;
        measurement.run(engine, &mut state)?;
        measurement.result(engine, &state)
    })
}

#[derive(Debug)]
struct Operators {
    ops: Vec<i32>,
    n_energy: usize,
    n_one: usize,
    n_two: usize,
    n_ex: usize,
}

impl Operators {
    fn prepare(model: &Model) -> Result<Self, String> {
        let data = &model.data;
        let mut ops = model.operators.clone();
        let mut add = |op: [i64; 7]| -> Result<(), String> {
            if [0, 1, 3, 4]
                .iter()
                .any(|&i| op[i] < 0 || op[i] >= model.ns as i64)
                || ![op[2], op[5]].iter().all(|&s| (0..=1).contains(&s))
            {
                return Err("invalid GPU PhysCal operator site/spin".into());
            }
            ops.extend(op.map(|x| x as i32));
            Ok(())
        };
        for t in &data.green_one_terms {
            if t.spin1 != t.spin2 {
                return Err("GPU PhysCal requires spin-conserving OneBodyG pairs".into());
            }
            add([t.site1, t.site2, t.spin1.as_code() as i64, 0, 0, 0, 1])?;
        }
        for t in &data.green_two_terms {
            if t.spin1 != t.spin2 || t.spin3 != t.spin4 {
                return Err("GPU PhysCal requires spin-conserving TwoBodyG pairs".into());
            }
            add([
                t.site1,
                t.site2,
                t.spin1.as_code() as i64,
                t.site3,
                t.site4,
                t.spin3.as_code() as i64,
                2,
            ])?;
        }
        for t in &data.green_two_ex_terms {
            if t.spin1 != t.spin2 || t.spin3 != t.spin4 {
                return Err("GPU PhysCal requires spin-conserving TwoBodyGEx pairs".into());
            }
            add([t.site1, t.site2, t.spin1.as_code() as i64, 0, 0, 0, 1])?;
            add([t.site3, t.site4, t.spin3.as_code() as i64, 0, 0, 0, 1])?;
        }
        Ok(Self {
            ops,
            n_energy: model.energies.len(),
            n_one: data.green_one_terms.len(),
            n_two: data.green_two_terms.len(),
            n_ex: data.green_two_ex_terms.len(),
        })
    }

    fn terms(&self) -> usize {
        self.ops.len() / 7
    }

    fn rows(&self) -> usize {
        1 + self.n_one + self.n_two + self.n_ex
    }
}

struct Measurement {
    operators: Operators,
    ops: DeviceBuffer,
    values: DeviceBuffer,
    means: DeviceBuffer,
}

impl Measurement {
    fn prepare<B: Engine + ?Sized>(
        backend: &mut B,
        operators: Operators,
        state: &State,
    ) -> Result<Self, String> {
        let w = state.resident.sampler.geom.walkers;
        let terms = operators.terms();
        let rows = operators.rows();
        let entries = w
            .checked_mul(terms)
            .filter(|&v| v <= i32::MAX as usize)
            .ok_or("GPU PhysCal operator capacity exceeded")?;
        if rows > i32::MAX as usize {
            return Err("GPU PhysCal output capacity exceeded".into());
        }
        with_raw_exec(backend, "mvmc.physcal.prepare", |exec| {
            let ops = upload(exec, &operators.ops)?;
            let values = exec.alloc_bytes((entries * 8).max(1))?;
            let means = exec.alloc_bytes(rows * 8)?;
            // SAFETY: means owns exactly rows FP64 values.
            unsafe {
                exec.zero_bytes(exec.ptr(&means)?, rows * 8)?;
            }
            Ok(Self {
                operators,
                ops,
                values,
                means,
            })
        })
        .map_err(|e| e.to_string())
    }

    fn observe(&self, exec: &RawExec<'_, '_>, state: &State) -> tenferro_tensor::Result<()> {
        let module = exec.module(SOURCE, KERNELS)?;
        state.counts(exec, &module)?;
        let b = &state.buffers;
        let m = &state.model;
        let s = &state.resident.sampler;
        let w = s.geom.walkers;
        let terms = self.operators.terms();
        let p = |b: &DeviceBuffer| exec.ptr(b).map(Arg::P);
        // SAFETY: all dimensions were checked before allocation; the common
        // operator evaluator reads resident tables and writes disjoint slots.
        unsafe {
            if terms != 0 {
                exec.launch(
                    module.function("opt_energy_terms")?,
                    ((terms * w).div_ceil(8) as u32, 1, 1),
                    256,
                    0,
                    &[
                        p(&self.values)?,
                        p(&self.ops)?,
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
                        p(&state.resident.error)?,
                    ],
                )?;
            }
            let reduce = exec.module(PHYS_SOURCE, PHYS_KERNELS)?;
            exec.launch(
                reduce.function("physcal_reduce")?,
                (self.operators.rows() as u32, 1, 1),
                256,
                0,
                &[
                    p(&self.means)?,
                    p(&self.values)?,
                    p(&b.energies)?,
                    i(self.operators.n_energy),
                    i(self.operators.n_one + self.operators.n_two),
                    i(self.operators.n_ex),
                    i(terms),
                    i(w),
                    p(&state.resident.error)?,
                ],
            )?;
        }
        Ok(())
    }

    fn run<B: Engine + ?Sized>(
        &mut self,
        backend: &mut B,
        state: &mut State,
    ) -> Result<(), String> {
        with_raw_exec(backend, "mvmc.physcal.run", |exec| {
            state.resident.enqueue_random(exec, state.warmup)?;
            for _ in 0..state.steps {
                state.resident.enqueue_random(exec, state.interval)?;
                self.observe(exec, state)?;
            }
            Ok(())
        })
        .map_err(|e| e.to_string())
    }

    fn result<B: Engine + ?Sized>(
        &self,
        backend: &mut B,
        state: &State,
    ) -> Result<PhysCalResult, String> {
        with_raw_exec(backend, "mvmc.physcal.result", |exec| {
            let mut error = [0i32];
            let mut means = vec![0.0f64; self.operators.rows()];
            // SAFETY: allocations have these exact lengths, execution has
            // completed before the public API returns host observations.
            unsafe {
                exec.download(&mut error, exec.ptr(&state.resident.error)?)?;
                exec.download(&mut means, exec.ptr(&self.means)?)?;
            }
            if error[0] != 0 || means.iter().any(|x| !x.is_finite()) {
                return Err(gpu_error(format!(
                    "GPU PhysCal failed (device status {})",
                    error[0]
                )));
            }
            for mean in &mut means {
                *mean /= state.steps as f64;
            }
            let one_end = 1 + self.operators.n_one;
            let two_end = one_end + self.operators.n_two;
            Ok(PhysCalResult {
                energy: means[0],
                green_one: means[1..one_end].to_vec(),
                green_two: means[one_end..two_end].to_vec(),
                green_two_ex: means[two_end..].to_vec(),
                seed: state.model.seed,
                walkers: state.resident.sampler.geom.walkers,
                nsamples: state.steps,
            })
        })
        .map_err(|e| e.to_string())
    }
}

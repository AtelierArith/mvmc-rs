//! Device-resident fixed-parameter PhysCal with the Julia GPU walker contract.
//!
//! This explicit API uses independently initialized XORWOW walkers. It does not
//! replace the CPU/C-order or legacy single-SFMT-chain `vmc_phys_cal` runner.

use super::*;
use std::path::PathBuf;

#[cfg(test)]
mod complex_tests;
#[cfg(test)]
mod dh_tests;
#[cfg(test)]
mod fsz_tests;
#[cfg(test)]
mod localspin_tests;
#[cfg(test)]
mod opttrans_tests;
#[cfg(test)]
mod rbm_tests;
#[cfg(test)]
mod tests;

const PHYS_SOURCE: &str = concat!(
    include_str!("../resident_status.cuh"),
    "\n",
    include_str!("physcal.cu")
);
const PHYS_KERNELS: &[&str] = &["physcal_reduce", "physcal_reduce_c"];

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
/// The `*_imag` fields are the imaginary parts: zeros for real inputs, C's
/// complex observables for complex normal orbitals (#678).
#[derive(Debug)]
pub struct PhysCalResult {
    pub energy: f64,
    pub green_one: Vec<f64>,
    pub green_two: Vec<f64>,
    pub green_two_ex: Vec<f64>,
    pub energy_imag: f64,
    pub green_one_imag: Vec<f64>,
    pub green_two_imag: Vec<f64>,
    pub green_two_ex_imag: Vec<f64>,
    pub seed: u64,
    pub walkers: usize,
    pub nsamples: usize,
}

/// Run normal-orbital Gutzwiller/Jastrow(/DH2/DH4/RBM/OptTrans) PhysCal without Lanczos or SR;
/// complex orbitals follow C's complex path (#678).
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
        // `[dest1, s1, src1, t1, dest2, s2, src2, t2, pairs]` (encode_operator).
        // FSZ walkers measure spin-flip pairs (C calgrn_fsz.c GreenFunc1_fsz2,
        // GreenFunc2_fsz2); normal walkers require spin-conserving pairs.
        let mut add = |op: [i64; 9], what: &str| -> Result<(), String> {
            ops.extend(
                super::model::encode_operator(model.fsz, model.ns, op)
                    .map_err(|_| format!("GPU PhysCal requires spin-conserving {what} pairs"))?,
            );
            Ok(())
        };
        let spin = |s: mvmc_expert_parsers::types::Spin| s.as_code() as i64;
        for t in &data.green_one_terms {
            let op = [
                t.site1,
                spin(t.spin1),
                t.site2,
                spin(t.spin2),
                0,
                0,
                0,
                0,
                1,
            ];
            add(op, "OneBodyG")?;
        }
        for t in &data.green_two_terms {
            let op = [
                t.site1,
                spin(t.spin1),
                t.site2,
                spin(t.spin2),
                t.site3,
                spin(t.spin3),
                t.site4,
                spin(t.spin4),
                2,
            ];
            add(op, "TwoBodyG")?;
        }
        for t in &data.green_two_ex_terms {
            let one = [
                t.site1,
                spin(t.spin1),
                t.site2,
                spin(t.spin2),
                0,
                0,
                0,
                0,
                1,
            ];
            let two = [
                t.site3,
                spin(t.spin3),
                t.site4,
                spin(t.spin4),
                0,
                0,
                0,
                0,
                1,
            ];
            add(one, "TwoBodyGEx")?;
            add(two, "TwoBodyGEx")?;
        }
        Ok(Self {
            ops,
            n_energy: model.energies.len() / model.lanes(),
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
        let lanes = state.model.lanes();
        let entries = w
            .checked_mul(terms)
            .filter(|&v| v <= i32::MAX as usize)
            .ok_or("GPU PhysCal operator capacity exceeded")?;
        if rows > i32::MAX as usize {
            return Err("GPU PhysCal output capacity exceeded".into());
        }
        with_raw_exec(backend, "mvmc.physcal.prepare", |exec| {
            let ops = upload(exec, &operators.ops)?;
            let values = exec.alloc_bytes((entries * 8 * lanes).max(1))?;
            let means = exec.alloc_bytes(rows * 8 * lanes)?;
            // SAFETY: means owns exactly rows FP64 values (pairs for complex).
            unsafe {
                exec.zero_bytes(exec.ptr(&means)?, rows * 8 * lanes)?;
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
        let w = state.resident.sampler.geom.walkers;
        let terms = self.operators.terms();
        let p = |b: &DeviceBuffer| exec.ptr(b).map(Arg::P);
        // The common operator evaluator reads resident tables and writes disjoint slots.
        if terms != 0 {
            state.energy_terms(exec, &module, &self.values, &self.ops, terms)?;
        }
        // SAFETY: all dimensions were checked before allocation.
        unsafe {
            let reduce = exec.module(PHYS_SOURCE, PHYS_KERNELS)?;
            exec.launch(
                reduce.function(if state.model.complex {
                    "physcal_reduce_c"
                } else {
                    "physcal_reduce"
                })?,
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
            let lanes = state.model.lanes();
            let mut means = vec![0.0f64; self.operators.rows() * lanes];
            // SAFETY: allocations have these exact lengths, execution has
            // completed before the public API returns host observations. A
            // failed run exposes only the status word.
            unsafe {
                exec.download(&mut error, exec.ptr(&state.resident.error)?)?;
            }
            if let Some(failure) = ResidentFailure::from_status(error[0]) {
                return Err(gpu_error(format!("GPU PhysCal failed: {failure}")));
            }
            unsafe {
                exec.download(&mut means, exec.ptr(&self.means)?)?;
            }
            if means.iter().any(|x| !x.is_finite()) {
                return Err(gpu_error("GPU PhysCal produced nonfinite means"));
            }
            for mean in &mut means {
                *mean /= state.steps as f64;
            }
            let (re, im): (Vec<f64>, Vec<f64>) = if lanes == 2 {
                means
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|z| (z[0], z[1]))
                    .unzip()
            } else {
                let zeros = vec![0.0; means.len()];
                (means, zeros)
            };
            let one_end = 1 + self.operators.n_one;
            let two_end = one_end + self.operators.n_two;
            Ok(PhysCalResult {
                energy: re[0],
                green_one: re[1..one_end].to_vec(),
                green_two: re[one_end..two_end].to_vec(),
                green_two_ex: re[two_end..].to_vec(),
                energy_imag: im[0],
                green_one_imag: im[1..one_end].to_vec(),
                green_two_imag: im[one_end..two_end].to_vec(),
                green_two_ex_imag: im[two_end..].to_vec(),
                seed: state.model.seed,
                walkers: state.resident.sampler.geom.walkers,
                nsamples: state.steps,
            })
        })
        .map_err(|e| e.to_string())
    }
}

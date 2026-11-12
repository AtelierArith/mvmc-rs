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
mod fsz_complex_tests;
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
#[cfg(test)]
mod window_tests;

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
    /// Save Green-function means for every window (potentially large).
    pub window_greens: bool,
}

/// One walker-averaged measurement window. For complex local energies the
/// second moment is mean(|E_loc|²), as in C vmccal.c, not mean(E_loc²).
#[derive(Debug)]
pub struct EnergyWindow {
    pub energy: f64,
    pub energy_imag: f64,
    pub energy_second_moment: f64,
}

/// One window's Green-function means, in parsed operator order.
#[derive(Debug, Default, Clone)]
pub struct GreenWindow {
    pub green_one: Vec<f64>,
    pub green_two: Vec<f64>,
    pub green_two_ex: Vec<f64>,
    pub green_one_imag: Vec<f64>,
    pub green_two_imag: Vec<f64>,
    pub green_two_ex_imag: Vec<f64>,
}

impl GreenWindow {
    fn from_flat(values: &[f64], ops: &Operators, lanes: usize) -> Self {
        let one = ops.n_one * lanes;
        let two = one + ops.n_two * lanes;
        let re = |s: &[f64]| s.chunks_exact(lanes).map(|x| x[0]).collect();
        let im = |s: &[f64]| {
            s.chunks_exact(lanes)
                .map(|x| if lanes == 2 { x[1] } else { 0. })
                .collect()
        };
        Self {
            green_one: re(&values[..one]),
            green_two: re(&values[one..two]),
            green_two_ex: re(&values[two..]),
            green_one_imag: im(&values[..one]),
            green_two_imag: im(&values[one..two]),
            green_two_ex_imag: im(&values[two..]),
        }
    }
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
    pub warmup: usize,
    pub interval: usize,
    pub energy_second_moment: f64,
    pub energy_windows: Vec<EnergyWindow>,
    /// None unless PhysCalConfig::window_greens was enabled.
    pub green_windows: Option<Vec<GreenWindow>>,
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
        let mut measurement =
            Measurement::prepare(engine, operators, &state, config.window_greens)?;
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
    output_len: usize,
    window_greens: bool,
}

impl Measurement {
    fn prepare<B: Engine + ?Sized>(
        backend: &mut B,
        operators: Operators,
        state: &State,
        window_greens: bool,
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
        // One allocation/download: overall means, then three energy fields
        // per window, then optional window Green means. No per-window host I/O.
        let window_width = if window_greens { (rows - 1) * lanes } else { 0 };
        let output_len = state
            .steps
            .checked_mul(3 + window_width)
            .and_then(|n| n.checked_add(rows * lanes))
            .ok_or("GPU PhysCal window capacity exceeded")?;
        let output_bytes = output_len
            .checked_mul(8)
            .ok_or("GPU PhysCal window capacity exceeded")?;
        with_raw_exec(backend, "mvmc.physcal.prepare", |exec| {
            let ops = upload(exec, &operators.ops)?;
            let values = exec.alloc_bytes((entries * 8 * lanes).max(1))?;
            let means = exec.alloc_bytes(output_bytes)?;
            // SAFETY: the checked allocation covers overall and window outputs.
            unsafe {
                exec.zero_bytes(exec.ptr(&means)?, output_bytes)?;
            }
            Ok(Self {
                operators,
                ops,
                values,
                means,
                output_len,
                window_greens,
            })
        })
        .map_err(|e| e.to_string())
    }

    fn observe(
        &self,
        exec: &RawExec<'_, '_>,
        state: &State,
        window: usize,
    ) -> tenferro_tensor::Result<()> {
        debug_assert!(window < state.steps);
        let module = exec.module(SOURCE, KERNELS)?;
        state.counts(exec, &module)?;
        let b = &state.buffers;
        let w = state.resident.sampler.geom.walkers;
        let terms = self.operators.terms();
        let p = |b: &DeviceBuffer| exec.ptr(b).map(Arg::P);
        let rows = self.operators.rows();
        let lanes = state.model.lanes();
        let base = exec.ptr(&self.means)?;
        let energy_out = base + ((rows * lanes + 3 * window) * 8) as u64;
        let green_out = if self.window_greens && rows > 1 {
            base + ((rows * lanes + 3 * state.steps + window * (rows - 1) * lanes) * 8) as u64
        } else {
            0
        };
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
                    Arg::P(energy_out),
                    Arg::P(green_out),
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
            for window in 0..state.steps {
                state.resident.enqueue_random(exec, state.interval)?;
                self.observe(exec, state, window)?;
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
            let mut output = vec![0.0f64; self.output_len];
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
                exec.download(&mut output, exec.ptr(&self.means)?)?;
            }
            if output.iter().any(|x| !x.is_finite()) {
                return Err(gpu_error("GPU PhysCal produced nonfinite means"));
            }
            let mean_len = self.operators.rows() * lanes;
            let (means, windows) = output.split_at_mut(mean_len);
            for mean in means.iter_mut() {
                *mean /= state.steps as f64;
            }
            let greens = GreenWindow::from_flat(&means[lanes..], &self.operators, lanes);
            let energy_windows: Vec<_> = windows[..3 * state.steps]
                .as_chunks::<3>()
                .0
                .iter()
                .map(|v| EnergyWindow {
                    energy: v[0],
                    energy_imag: v[1],
                    energy_second_moment: v[2],
                })
                .collect();
            let energy_second_moment = energy_windows
                .iter()
                .map(|v| v.energy_second_moment)
                .sum::<f64>()
                / state.steps as f64;
            let green_windows = self.window_greens.then(|| {
                let width = (self.operators.rows() - 1) * lanes;
                if width == 0 {
                    vec![GreenWindow::default(); state.steps]
                } else {
                    windows[3 * state.steps..]
                        .chunks_exact(width)
                        .map(|v| GreenWindow::from_flat(v, &self.operators, lanes))
                        .collect()
                }
            });
            Ok(PhysCalResult {
                energy: means[0],
                energy_imag: if lanes == 2 { means[1] } else { 0. },
                green_one: greens.green_one,
                green_two: greens.green_two,
                green_two_ex: greens.green_two_ex,
                green_one_imag: greens.green_one_imag,
                green_two_imag: greens.green_two_imag,
                green_two_ex_imag: greens.green_two_ex_imag,
                seed: state.model.seed,
                walkers: state.resident.sampler.geom.walkers,
                nsamples: state.steps,
                warmup: state.warmup,
                interval: state.interval,
                energy_second_moment,
                energy_windows,
                green_windows,
            })
        })
        .map_err(|e| e.to_string())
    }
}

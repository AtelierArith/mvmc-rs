//! Complex normal orbitals against the Rust CPU C-order complex path (#678, #572).
//!
//! Julia GPU rejects complex parameters (`ext/cuda_state.jl:39`), so the
//! references are mvmc-core's C-order complex kernels evaluated on the
//! configurations the GPU sampled: `calc_m_all_complex` (C CalculateMAll_fcmp),
//! `calculate_ip_complex`, the complex `calculate_local_energy` (C calham.c),
//! `slater_elm_diff` (C SlaterElmDiff_fcmp) and `sync_modified_parameter_local`
//! (C SyncModifiedParameter). The supplied-tape trajectory is replayed on the
//! host with the GPU tape contract, C `UpdateProjCnt`/`LogProjRatio` and full
//! CPU complex Pfaffians (`log|ip|`). Integers (configurations, acceptance
//! counts, counters) are exact; FP64 bounds are stated at each assertion.
use super::dh_tests::{arrays, tape};
use super::tests::close;
use super::*;
use mvmc_core::observables::{calculate_ip_complex, calculate_local_energy, slater_elm_diff};
use mvmc_core::sampling::projection::{log_proj_ratio, make_proj_cnt, update_proj_cnt};
use mvmc_core::{ThreadedPfaPackWorkspace, VmcOptimizationState};
use num_complex::Complex64;
use tenferro_ext::raw_exec::raw_exec_counters;

pub(crate) fn input(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/complex")
        .join(name)
        .join("namelist.def")
}

/// Interleaved device values as complex numbers.
pub(crate) fn complex(values: &[f64]) -> Vec<Complex64> {
    values
        .as_chunks::<2>()
        .0
        .iter()
        .map(|z| Complex64::new(z[0], z[1]))
        .collect()
}

/// CPU C-order complex evaluation of one walker configuration.
pub(crate) struct Evaluation {
    pub counts: Vec<i64>,
    pub ip: Complex64,
    /// `sum_q |QPFullWeight_q Pf_q|`, the scale of the overlap sum.
    pub ip_scale: f64,
    /// Largest Frobenius condition number `|A|_F |A^-1|_F` over QP planes.
    pub kappa: f64,
    pub energy: Complex64,
    /// C SROptO slots `2(1+p)` (d/dRe p_p) and `2(1+p)+1` (d/dIm p_p).
    pub derivatives: Vec<Complex64>,
}

/// The Rust CPU C-order complex model with the given parameters.
pub(crate) struct Cpu {
    pub data: mvmc_core::ExpertModeData,
    pub state: VmcOptimizationState,
    pool: ThreadedPfaPackWorkspace,
    ns: usize,
    ne: usize,
    nq: usize,
    nproj: usize,
    np: usize,
}

impl Cpu {
    pub fn new(model: &Model, parameters: &[f64]) -> Self {
        let mut data = model.data.clone();
        mvmc_core::unpack_parameters(&mut data, &complex(parameters)).unwrap();
        Self::from_data(model, data)
    }

    /// As [`Self::new`], with the imaginary parts of the QP weights dropped
    /// (what Julia GPU does with a complex `ParaQPTrans`, `cuda_state.jl:70`).
    pub fn real_weights(model: &Model, parameters: &[f64]) -> Self {
        let mut data = model.data.clone();
        mvmc_core::unpack_parameters(&mut data, &complex(parameters)).unwrap();
        let weights = data.qp_weights.as_mut().unwrap();
        for w in &mut weights.qp_full_weight {
            w.im = 0.0;
        }
        Self::from_data(model, data)
    }

    fn from_data(model: &Model, mut data: mvmc_core::ExpertModeData) -> Self {
        let (ns, ne, nq, nproj, np) = (model.ns, model.ne, model.nq, model.nproj, model.np);
        let mut state = VmcOptimizationState::zeros(ns, ne, nproj, np, nq, 1, true, false);
        state.validate_declared_mode(&data).unwrap();
        mvmc_core::slater_update::update_slater_elm(&mut data, &mut state);
        Self {
            data,
            state,
            pool: ThreadedPfaPackWorkspace::new(2 * ne, 1),
            ns,
            ne,
            nq,
            nproj,
            np,
        }
    }

    pub fn counts(&self, num: &[i64]) -> Vec<i64> {
        let mut counts = vec![0i64; self.nproj];
        make_proj_cnt(&mut counts, num, &self.data);
        counts
    }

    /// C CalculateMAll_fcmp + CalculateIP_fcmp; the tables remain in `state`.
    pub fn factor(&mut self, idx: &[i64]) -> Complex64 {
        let m = &mut self.state.slater_matrix;
        mvmc_core::calc_m_all_complex(
            idx,
            &m.slater_elm,
            &mut m.inv_m,
            &mut m.pf_m,
            0,
            self.nq,
            self.ns,
            self.ne,
            &self.pool,
        )
        .unwrap();
        calculate_ip_complex(&m.pf_m, 0, self.nq, &self.data)
    }

    fn kappa(&self, idx: &[i64]) -> f64 {
        let m = &self.state.slater_matrix;
        let n = idx.len();
        let site = |i: usize| idx[i] as usize + if i >= self.ne { self.ns } else { 0 };
        (0..self.nq)
            .map(|q| {
                let mut a = 0.;
                let mut b = 0.;
                for i in 0..n {
                    for j in 0..n {
                        a += m.slater_elm.get(q, site(i), site(j)).norm_sqr();
                        b += m.inv_m.get(q, i, j).norm_sqr();
                    }
                }
                (a * b).sqrt()
            })
            .fold(0f64, f64::max)
    }

    pub fn evaluate(&mut self, cfg: &[i32]) -> Evaluation {
        let (idx, ele_cfg, num) = arrays(cfg, self.ns, self.ne);
        let counts = self.counts(&num);
        let ip = self.factor(&idx);
        let kappa = self.kappa(&idx);
        let weights = &self.data.qp_weights.as_ref().unwrap().qp_full_weight;
        let ip_scale = (0..self.nq)
            .map(|q| (weights[q] * self.state.slater_matrix.pf_m[q]).norm())
            .sum();
        let energy = calculate_local_energy(
            ip,
            &self.data,
            &mut self.state,
            &idx,
            &ele_cfg,
            &num,
            &counts,
        );
        let mut o = vec![Complex64::new(0., 0.); 2 * (1 + self.np)];
        mvmc_core::observables::set_projection_diff(&mut o, &counts, self.nproj);
        slater_elm_diff(
            &mut o[2 * (1 + self.nproj)..],
            ip,
            &idx,
            &self.data,
            &self.state,
        );
        Evaluation {
            counts,
            ip,
            ip_scale,
            kappa,
            energy,
            derivatives: o[2..].to_vec(),
        }
    }
}

/// Host replay of the resident complex tape contract (`interval_complex_hop`):
/// proposal words, exchange threshold and the log-space decision on
/// `2 (LogProjRatio + log|ip_new| - log|ip_old|)` with the 1e-100 floor.
pub(crate) struct Replay {
    pub cfg: Vec<i32>,
    pub accepted: Vec<i32>,
    /// Smallest |log u - log ratio| over all decisions.
    pub margin: f64,
    /// Decisions that dropping the imaginary QP weight parts would change.
    pub flips: usize,
}

pub(crate) fn replay(
    cpu: &mut Cpu,
    plain: &mut Cpu,
    model: &Model,
    initial: &[i32],
    tape: &[u32],
    walkers: usize,
    steps: usize,
) -> Replay {
    let (ns, ne) = (model.ns, model.ne);
    let n = 2 * ne;
    let mut cfg = initial.to_vec();
    let mut accepted = vec![0i32; walkers];
    let mut margin = f64::INFINITY;
    let mut flips = 0;
    let scale = 4294967296.0;
    let log_abs = |z: Complex64| (z.norm() + 1e-100).ln();
    for w in 0..walkers {
        let e = &mut cfg[w * n..(w + 1) * n];
        let (_, _, mut num) = arrays(e, ns, ne);
        let mut counts = cpu.counts(&num);
        let idx = |e: &[i32]| e.iter().map(|&r| r as i64).collect::<Vec<_>>();
        let mut log_old = log_abs(cpu.factor(&idx(e)));
        let mut plain_old = log_abs(plain.factor(&idx(e)));
        for step in 0..steps {
            let t = &tape[5 * (w + walkers * step)..][..5];
            let mi = (t[0] % ne as u32) as usize;
            let spin = (t[1] >> 31) as usize;
            let ri = e[mi + spin * ne] as usize;
            let exchange = (t[3] as f64) / scale < model.exchange;
            let (rj, partner, valid) = if exchange {
                let partner = (t[2] % ne as u32) as usize + (1 - spin) * ne;
                let rj = e[partner] as usize;
                let valid = ri != rj && num[rj + spin * ns] == 0 && num[ri + (1 - spin) * ns] == 0;
                (rj, partner, valid)
            } else {
                let empty: Vec<usize> = (0..ns).filter(|&r| num[r + spin * ns] == 0).collect();
                let rj = if empty.is_empty() {
                    0
                } else {
                    empty[(t[2] % empty.len() as u32) as usize]
                };
                (rj, usize::MAX, !empty.is_empty())
            };
            if !valid {
                continue;
            }
            let previous = (e.to_vec(), num.clone());
            e[mi + spin * ne] = rj as i32;
            num[ri + spin * ns] = 0;
            num[rj + spin * ns] = 1;
            let mut proposed = vec![0i64; counts.len()];
            if exchange {
                e[partner] = ri as i32;
                num[rj + (1 - spin) * ns] = 0;
                num[ri + (1 - spin) * ns] = 1;
                make_proj_cnt(&mut proposed, &num, &cpu.data);
            } else {
                update_proj_cnt(
                    ri as i64,
                    rj as i64,
                    spin as u8,
                    &mut proposed,
                    &counts,
                    &num,
                    &cpu.data,
                );
            }
            let proj = if exchange {
                0.0
            } else {
                log_proj_ratio(&proposed, &counts, &cpu.data)
            };
            let ip_new = cpu.factor(&idx(e));
            let log_new = log_abs(ip_new);
            let plain_new = log_abs(plain.factor(&idx(e)));
            let ratio = 2.0 * (proj + log_new - log_old);
            let without = 2.0 * (proj + plain_new - plain_old);
            let u = (t[4] as f64 / scale).ln();
            if u.is_finite() {
                margin = margin.min((u - ratio).abs());
            }
            let ok = ip_new.re.is_finite() && ip_new.im.is_finite() && ip_new.norm() != 0.0;
            flips += usize::from(ok && (u < ratio) != (u < without));
            if ok && u < ratio {
                accepted[w] += 1;
                log_old = log_new;
                plain_old = plain_new;
                counts = proposed;
            } else {
                e.copy_from_slice(&previous.0);
                num = previous.1;
            }
        }
    }
    Replay {
        cfg,
        accepted,
        margin,
        flips,
    }
}

pub(crate) fn download<T: Copy + Default>(
    exec: &RawExec<'_, '_>,
    buffer: &DeviceBuffer,
    len: usize,
) -> tenferro_tensor::Result<Vec<T>> {
    let mut out = vec![T::default(); len];
    // SAFETY: every caller passes a length within the buffer's allocation.
    unsafe {
        exec.download(&mut out, exec.ptr(buffer)?)?;
    }
    Ok(out)
}

struct Measured {
    counts: Vec<i32>,
    /// SR energy vector `[Re e; -Im e]`.
    energy: Vec<f64>,
    /// Stacked SR rows `[Re O; Im O]`, `2w x 2np`.
    derivatives: Vec<f64>,
    overlaps: Vec<Complex64>,
    terms: Vec<Complex64>,
    aw: Vec<i32>,
}

fn measure(backend: &mut CudaEngine, state: &State) -> Measured {
    with_raw_exec(backend, "complex.measure", |exec| {
        let module = exec.module(SOURCE, KERNELS)?;
        state.observe(exec, &module)?;
        let w = state.resident.sampler.geom.walkers;
        let m = &state.model;
        Ok(Measured {
            counts: download(exec, &state.buffers.counts, w * m.nproj)?,
            energy: download(exec, &state.buffers.local_energy, 2 * w)?,
            derivatives: download(exec, &state.buffers.derivatives, 4 * w * m.np)?,
            overlaps: complex(&download::<f64>(exec, &state.buffers.overlaps, 2 * w)?),
            terms: complex(&download::<f64>(
                exec,
                &state.buffers.values,
                w * m.energies.len(),
            )?),
            aw: download(exec, &state.resident.sampler.aw, w)?,
        })
    })
    .map_err(|e| e.to_string())
    .unwrap()
}

/// Per-walker #656 interval budget `16 (aw+1) n eps kappa_F`: the relative
/// error of a Pfaffian ratio from an inverse after `aw` rank-two updates of
/// a plane with Frobenius condition `kappa_F` (docs/NUMERICAL_COMPARISONS.md).
pub(crate) fn budget(aw: i32, n: usize, kappa: f64) -> f64 {
    16.0 * (aw + 1) as f64 * n as f64 * f64::EPSILON * kappa
}

/// Compare one stage with the CPU C-order complex evaluation of the same walkers.
fn assert_stage(name: &str, stage: &str, cpu: &mut Cpu, state: &State, backend: &mut CudaEngine) {
    let cfg = with_raw_exec(backend, "complex.cfg", |exec| {
        let g = state.resident.sampler.geom;
        download::<i32>(exec, &state.resident.sampler.ele, g.walkers * g.n)
    })
    .map_err(|e| e.to_string())
    .unwrap();
    let g = measure(backend, state);
    let m = &state.model;
    let n = 2 * m.ne;
    let walkers = cfg.len() / n;
    let width = 2 * m.np;
    let terms = m.energies.len() / 2;
    let coefficients = complex(&m.energies);
    let (mut energy_error, mut derivative_error, mut overlap_error) = (0f64, 0f64, 0f64);
    let (mut energy_abs, mut imag_energy, mut kappa_max) = (0f64, 0f64, 0f64);
    for w in 0..walkers {
        let walker = &cfg[w * n..(w + 1) * n];
        let e = cpu.evaluate(walker);
        let gpu: Vec<i64> = g.counts[w * m.nproj..(w + 1) * m.nproj]
            .iter()
            .map(|&c| c as i64)
            .collect();
        assert_eq!(gpu, e.counts, "{name} {stage} walker {w}: counters");
        let re = &g.derivatives[w * width..(w + 1) * width];
        let im = &g.derivatives[(walkers + w) * width..(walkers + w + 1) * width];
        // Projection: C writes the counter into the real slot and 0 into the
        // imaginary one (vmccal.c); exact.
        for p in 0..m.nproj {
            assert_eq!(
                [re[2 * p], im[2 * p], re[2 * p + 1], im[2 * p + 1]],
                [e.counts[p] as f64, 0.0, 0.0, 0.0],
                "{name} {stage} walker {w}: projection derivative {p}"
            );
        }
        let budget = budget(g.aw[w], n, e.kappa);
        kappa_max = kappa_max.max(e.kappa);
        // Overlap: each plane Pfaffian carries the budget, relative to the
        // magnitude sum of the weighted terms.
        let d = (g.overlaps[w] - e.ip).norm();
        let bound = 1e-13 + budget * e.ip_scale;
        assert!(
            d <= bound,
            "{name} {stage} walker {w}: overlap {} vs {}",
            g.overlaps[w],
            e.ip
        );
        overlap_error = overlap_error.max(d / (budget * e.ip_scale / 16.0));
        // Energy: every operator ratio carries the budget (#656), relative
        // to sum_t |c_t v_t| of the GPU terms.
        let v = &g.terms[w * terms..(w + 1) * terms];
        let scale: f64 = (0..terms).map(|t| (coefficients[t] * v[t]).norm()).sum();
        let energy = Complex64::new(g.energy[w], -g.energy[walkers + w]);
        let d = (energy - e.energy).norm();
        let bound = 1e-13 + budget * scale;
        assert!(
            d <= bound,
            "{name} {stage} walker {w}: energy {energy} vs {} (kappa {:e})",
            e.energy,
            e.kappa
        );
        energy_abs = energy_abs.max(d);
        energy_error = energy_error.max(d / (budget * scale / 16.0));
        imag_energy = imag_energy.max(e.energy.im.abs());
        // Slater derivatives: the stacked rows hold O (column 2p) and i O
        // (column 2p+1) exactly; both against C's slot pair within the
        // budget relative to the largest Slater derivative of the walker.
        let largest = e.derivatives[2 * m.nproj..]
            .iter()
            .fold(0f64, |a, x| a.max(x.norm()));
        for p in m.nproj..m.np {
            let o = Complex64::new(re[2 * p], im[2 * p]);
            assert_eq!(
                [re[2 * p + 1], im[2 * p + 1]],
                [-o.im, o.re],
                "{name} {stage} walker {w}: column 2p+1 = i O"
            );
            for (slot, gpu) in [(2 * p, o), (2 * p + 1, Complex64::new(-o.im, o.re))] {
                let d = (gpu - e.derivatives[slot]).norm();
                assert!(
                    d <= 1e-13 + budget * largest,
                    "{name} {stage} walker {w} Slater slot {slot}: {gpu} vs {}",
                    e.derivatives[slot]
                );
                derivative_error = derivative_error.max(d / (budget * largest / 16.0));
            }
        }
    }
    assert!(
        imag_energy > 1e-6,
        "{name} {stage}: complex inputs must give complex local energies ({imag_energy:e})"
    );
    println!(
        "{name} {stage}: max kappa_F={kappa_max:.3e}; max |Im e|={imag_energy:.3e}; max abs diff \
         energy={energy_abs:e}; normalized by (aw+1) n eps kappa_F scale: overlap={overlap_error:.3} \
         energy={energy_error:.3} Slater derivative={derivative_error:.3} (bound 16)"
    );
}

/// C's complex SR system (stcopt.c, stcopt_dposv.c) rebuilt on the host from
/// the device's stacked rows through C's SROptO layout: slots
/// `O_{2k} + i O_{2k}'` with `S_ij = Re<conj(O_i) O_j> - Re<O_i> Re<O_j>` and
/// `g_i = -2 dt (Re<e O_i> - Re<e> Re<O_i>)` in complex arithmetic, the
/// resident diagonal cut and flags (`opt_system`), and the normalized
/// backward error of the device solution.
pub(crate) fn complex_sr_backward_error(
    model: &Model,
    derivatives: &[f64],
    energy: &[f64],
    update: &[f64],
) -> f64 {
    let np = 2 * model.np;
    let w = energy.len() / 2;
    let o: Vec<Vec<Complex64>> = (0..w)
        .map(|b| {
            (0..np)
                .map(|i| Complex64::new(derivatives[b * np + i], derivatives[(w + b) * np + i]))
                .collect()
        })
        .collect();
    let e: Vec<Complex64> = (0..w)
        .map(|b| Complex64::new(energy[b], -energy[w + b]))
        .collect();
    let wf = w as f64;
    let mean: Vec<Complex64> = (0..np)
        .map(|i| o.iter().map(|x| x[i]).sum::<Complex64>() / wf)
        .collect();
    let s = |i: usize, j: usize| {
        o.iter().map(|x| x[i].conj() * x[j]).sum::<Complex64>().re / wf - mean[i].re * mean[j].re
    };
    let variance: Vec<f64> = (0..np).map(|i| s(i, i)).collect();
    let mp = &model.data.modpara;
    let cutoff = variance.iter().map(|v| v.abs()).fold(0f64, f64::max) * mp.dsr_opt_red_cut;
    let active: Vec<bool> = (0..np)
        .map(|i| model.flags[i] != 0 && variance[i] > cutoff)
        .collect();
    let mean_e = e.iter().sum::<Complex64>() / wf;
    let (mut residual, mut norm_a, mut norm_b) = (0f64, 0f64, 0f64);
    for row in 0..np {
        let rhs = if active[row] {
            let he = o.iter().zip(&e).map(|(x, y)| y * x[row]).sum::<Complex64>() / wf;
            -2. * mp.dsr_opt_step_dt * (he.re - mean_e.re * mean[row].re)
        } else {
            0.
        };
        let (mut value, mut row_norm) = (0., 0.);
        for col in 0..np {
            let a = if active[row] && active[col] {
                s(row, col)
                    * if row == col {
                        1. + mp.dsr_opt_sta_del
                    } else {
                        1.
                    }
            } else if row == col {
                1.
            } else {
                0.
            };
            value += a * update[col];
            row_norm += a.abs();
        }
        residual = residual.max((value - rhs).abs());
        norm_a = norm_a.max(row_norm);
        norm_b = norm_b.max(rhs.abs());
    }
    residual / (norm_a * update.iter().map(|v| v.abs()).fold(0f64, f64::max) + norm_b)
}

fn stage_config(steps: usize) -> OptConfig {
    OptConfig {
        nsteps: 1,
        walkers: Some(8),
        warmup: Some(0),
        interval: Some(steps),
        seed: Some(79),
        initial_def: InitialDef::None,
        ..Default::default()
    }
}

fn assert_opt_stages(name: &str, steps_per_site: usize) {
    let probe = Model::prepare(&input(name), &stage_config(1)).unwrap();
    let steps = steps_per_site * probe.ns;
    let config = stage_config(steps);
    let model = Model::prepare(&input(name), &config).unwrap();
    assert!(model.complex && model.weights.iter().skip(1).step_by(2).any(|&x| x != 0.0));
    let walkers = 8;
    let mut backend = CudaEngine::open(cuda_devices().unwrap()[0].id()).unwrap();
    let backend: &mut CudaEngine = &mut backend;
    let mut cpu = Cpu::new(&model, &model.parameters);
    let mut plain = Cpu::real_weights(&model, &model.parameters);
    let mut state = State::prepare(backend, model, &config).unwrap();
    state.initialize(backend).unwrap();
    let initial = state.resident.snapshot(backend).unwrap().ele;
    assert_stage(name, "initial", &mut cpu, &state, backend);
    let tape = tape(walkers, steps);
    state.resident.advance_with_tape(backend, &tape).unwrap();
    let last = state.resident.snapshot(backend).unwrap();
    let accepted = state.resident.accepted(backend).unwrap();
    let host = replay(
        &mut cpu,
        &mut plain,
        &state.model,
        &initial,
        &tape,
        walkers,
        steps,
    );
    assert_eq!(last.ele, host.cfg, "{name}: supplied-tape configurations");
    assert_eq!(accepted, host.accepted, "{name}: acceptance counts");
    assert!(accepted.iter().any(|&a| a > 0));
    assert!(
        host.flips > 0,
        "{name}: the complex QP weights changed no decision"
    );
    println!(
        "{name}: {steps} proposals x {walkers} walkers, accepted {accepted:?}, {} decisions \
         changed by Im QPFullWeight, smallest decision margin {:e}",
        host.flips, host.margin
    );
    assert_stage(name, "final", &mut cpu, &state, backend);
    // One SR update plus the gauge, against C's complex system and the
    // C-order complex sync; then the rebuilt state against a CPU model with
    // the new parameters.
    let (before, update, after, energy, derivatives) =
        with_raw_exec(backend, "complex.update", |exec| {
            let module = exec.module(SOURCE, KERNELS)?;
            let m = &state.model;
            let w = state.resident.sampler.geom.walkers;
            state.observe(exec, &module)?;
            let before = download::<f64>(exec, &state.buffers.parameters, 2 * m.np)?;
            let energy = download::<f64>(exec, &state.buffers.local_energy, 2 * w)?;
            let derivatives = download::<f64>(exec, &state.buffers.derivatives, 4 * w * m.np)?;
            state.update(exec, &module, 0)?;
            Ok((
                before,
                download(exec, &state.buffers.rhs, 2 * m.np)?,
                download(exec, &state.buffers.parameters, 2 * m.np)?,
                energy,
                derivatives,
            ))
        })
        .unwrap();
    state.resident.check_status(backend).unwrap();
    let m = &state.model;
    let relative = complex_sr_backward_error(m, &derivatives, &energy, &update);
    assert!(relative < 1e-11, "{name}: SR backward error {relative:e}");
    assert!(
        (m.nproj..m.np).any(|p| update[2 * p + 1] != 0.0),
        "{name}: SR moved no imaginary Slater component"
    );
    let pre: Vec<Complex64> = complex(&before)
        .iter()
        .zip(complex(&update))
        .map(|(p, u)| p + u)
        .collect();
    let mut data = m.data.clone();
    mvmc_core::unpack_parameters(&mut data, &pre).unwrap();
    mvmc_core::sync::sync_modified_parameter_local(&mut data, true);
    let expected: Vec<f64> = mvmc_core::pack_parameters(&data)
        .unwrap()
        .iter()
        .flat_map(|p| [p.re, p.im])
        .collect();
    // Same IEEE operations as the C-order sync, except CUDA's hypot for C's
    // cabs (at most one ulp), which moves the Slater ratio by <= 2 eps.
    let (ratio, diff) = super::dh_tests::worst(&after, &expected, 1e-15, 4.0 * f64::EPSILON);
    assert!(ratio <= 1.0, "{name}: gauge ratio {ratio:e}");
    println!("{name}: SR backward error={relative:e}; gauge max abs diff={diff:e}");
    let mut updated = Cpu::new(&state.model, &after);
    assert_stage(name, "after SR", &mut updated, &state, backend);
}

#[test]
#[ignore = "requires CUDA; complex L16 doped chain with exchanges against the CPU C-order path"]
fn resident_opt_complex_l16_matches_cpu_c_order() {
    assert_opt_stages("hubbard_complex_L16", 8);
}

#[test]
#[ignore = "requires CUDA; complex L128 chain against the CPU C-order path"]
fn resident_opt_complex_l128_matches_cpu_c_order() {
    assert_opt_stages("hubbard_complex_L128", 1);
}

/// #573 per-plane recovery on the complex path. Marking one plane of every
/// walker invalid routes the energies through `opt_energy_fallback_c` (full
/// complex Pfaffians of every moved configuration) and the derivatives
/// through `opt_cofactors_c` (exact complex minors); both must still match
/// the CPU C-order evaluation. Sampling then scores every proposal with full
/// Pfaffians and refactors on accept (`rc_refactor_walker`), and the
/// supplied-tape trajectory must still equal the host replay exactly.
#[test]
#[ignore = "requires CUDA; complex invalid-plane fallbacks against the CPU C-order path"]
fn resident_opt_complex_invalid_plane_fallbacks_match_cpu() {
    let name = "hubbard_complex_L16";
    let probe = Model::prepare(&input(name), &stage_config(1)).unwrap();
    let steps = 4 * probe.ns;
    let config = stage_config(steps);
    let model = Model::prepare(&input(name), &config).unwrap();
    let walkers = 8;
    let mut backend = CudaEngine::open(cuda_devices().unwrap()[0].id()).unwrap();
    let backend: &mut CudaEngine = &mut backend;
    let mut cpu = Cpu::new(&model, &model.parameters);
    let mut plain = Cpu::real_weights(&model, &model.parameters);
    let mut state = State::prepare(backend, model, &config).unwrap();
    state.initialize(backend).unwrap();
    let nq = state.model.nq;
    let mut valid = vec![1i32; walkers * nq];
    for w in 0..walkers {
        valid[w * nq + (3 * w) % nq] = 0;
    }
    with_raw_exec(backend, "complex.invalidate", |exec| unsafe {
        exec.upload(exec.ptr(&state.resident.valid)?, &valid)
    })
    .unwrap();
    let initial = state.resident.snapshot(backend).unwrap().ele;
    assert_stage(name, "invalid planes", &mut cpu, &state, backend);
    let tape = tape(walkers, steps);
    state.resident.advance_with_tape(backend, &tape).unwrap();
    let last = state.resident.snapshot(backend).unwrap();
    let accepted = state.resident.accepted(backend).unwrap();
    let host = replay(
        &mut cpu,
        &mut plain,
        &state.model,
        &initial,
        &tape,
        walkers,
        steps,
    );
    assert_eq!(last.ele, host.cfg, "{name}: full-Pfaffian configurations");
    assert_eq!(accepted, host.accepted, "{name}: full-Pfaffian acceptance");
    assert!(accepted.iter().all(|&a| a > 0), "every walker refactored");
    let restored = with_raw_exec(backend, "complex.valid", |exec| {
        download::<i32>(exec, &state.resident.valid, walkers * nq)
    })
    .unwrap();
    assert!(
        restored.iter().all(|&v| v == 1),
        "the refactor restores validity"
    );
    assert_stage(name, "after refactor", &mut cpu, &state, backend);
}

/// Complete 20-step Opt runs repeat exactly within one implementation, after
/// an intervening differently seeded run, with zero explicit transfers and one
/// stream synchronization for the whole warmup/optimization series.
#[test]
#[ignore = "requires CUDA; 20-step complex repeatability, transfers and timing"]
fn resident_opt_complex_repeats_twenty_steps() {
    for (name, walkers) in [("hubbard_complex_L16", 64), ("hubbard_complex_L128", 64)] {
        let config = OptConfig {
            nsteps: 20,
            walkers: Some(walkers),
            warmup: Some(128),
            interval: Some(128),
            seed: Some(79),
            initial_def: InitialDef::None,
            ..Default::default()
        };
        let start = std::time::Instant::now();
        let first = run_gpu_para_opt_from_namelist(input(name), config.clone()).unwrap();
        let first_seconds = start.elapsed().as_secs_f64();
        run_gpu_para_opt_from_namelist(
            input(name),
            OptConfig {
                nsteps: 1,
                walkers: Some(8),
                warmup: Some(0),
                interval: Some(16),
                seed: Some(23),
                ..config.clone()
            },
        )
        .unwrap();
        let start = std::time::Instant::now();
        let second = run_gpu_para_opt_from_namelist(input(name), config.clone()).unwrap();
        let second_seconds = start.elapsed().as_secs_f64();
        // Same implementation, inputs and seed: bitwise repeatability is the
        // contract, so the bounds are only formatting slack.
        close("energies", &first.energies, &second.energies, 1e-12, 1e-12);
        close(
            "parameters",
            &first.parameters,
            &second.parameters,
            1e-12,
            1e-12,
        );
        close(
            "imaginary parameters",
            &first.parameters_imag,
            &second.parameters_imag,
            1e-12,
            1e-12,
        );
        assert!(first.parameters_imag.iter().any(|&x| x != 0.0));
        let (traffic, run_seconds) = with_runtime(0, |backend| {
            let model = Model::prepare(&input(name), &config)?;
            let mut state = State::prepare(backend, model, &config)?;
            state.initialize(backend)?;
            let before = raw_exec_counters();
            let start = std::time::Instant::now();
            state.run(backend)?;
            let seconds = start.elapsed().as_secs_f64();
            let traffic = raw_exec_counters() - before;
            let result = state.result(backend)?;
            close(
                "resident run",
                &result.energies,
                &first.energies,
                1e-12,
                1e-12,
            );
            Ok((traffic, seconds))
        })
        .unwrap();
        assert_eq!(
            (
                traffic.upload_calls,
                traffic.download_calls,
                traffic.synchronizations
            ),
            (0, 0, 1),
            "{name}: sampling/optimization must stay resident"
        );
        println!(
            "{name}: walkers={walkers} e2e first={first_seconds:.3}s warm={second_seconds:.3}s \
             run={run_seconds:.3}s traffic={traffic:?} energies[0,19]=({:.6},{:.6})",
            first.energies[0], first.energies[19]
        );
    }
}

//! Doublon-holon factors against the Rust CPU C-order path (#665, #572).
//!
//! Julia GPU has no DH support, so the references are mvmc-core's C-order
//! kernels evaluated on the configurations the GPU sampled:
//! `make_proj_cnt` / `update_proj_cnt` / `log_proj_ratio` (C `projection.c`),
//! `calc_m_all_real`, `calculate_local_energy`, `slater_elm_diff` and
//! `sync_modified_parameter_local` (C `parameter.c`). The supplied-tape
//! trajectory is replayed on the host with the GPU tape contract
//! (`interval_kernels.cu`) and full CPU Pfaffians. Integers (counters,
//! configurations, acceptance counts) are exact; FP64 bounds are stated at
//! each assertion.
use super::tests::{close, sr_backward_error};
use super::*;
use mvmc_core::observables::{calculate_ip_real, calculate_local_energy, slater_elm_diff};
use mvmc_core::sampling::projection::{log_proj_ratio, make_proj_cnt, update_proj_cnt};
use mvmc_core::sampling::rbm::{make_rbm_cnt, set_rbm_diff, RbmConfig};
use mvmc_core::{ThreadedPfaPackWorkspace, VmcOptimizationState};
use num_complex::Complex64;
use tenferro_ext::raw_exec::raw_exec_counters;

pub(crate) fn input(name: &str) -> std::path::PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    match name {
        "dh2_production" => root.join("../../tests/fixtures/dh2/production_real/namelist.def"),
        "dh4_production" => root.join("../../tests/fixtures/dh4/production_dh4_real/namelist.def"),
        _ => root
            .join("tests/fixtures/dh")
            .join(name)
            .join("namelist.def"),
    }
}

/// Deterministic tape of the local-spin fixtures (`[word][walker][step]`).
pub(crate) fn tape(walkers: usize, steps: usize) -> Vec<u32> {
    (1..=(5 * walkers * steps) as u64)
        .map(|k| {
            (k.wrapping_mul(0x9e3779b97f4a7c15)
                .wrapping_add(0xbf58476d1ce4e5b9)
                >> 32) as u32
        })
        .collect()
}

/// Maximum `|a-b| / (atol + rtol*max(|a|,|b|))`; at most 1 passes.
pub(crate) fn worst(actual: &[f64], expected: &[f64], atol: f64, rtol: f64) -> (f64, f64) {
    assert_eq!(actual.len(), expected.len());
    let mut ratio = 0f64;
    let mut diff = 0f64;
    for (&a, &b) in actual.iter().zip(expected) {
        assert!(a.is_finite() && b.is_finite(), "{a} vs {b}");
        let d = (a - b).abs();
        diff = diff.max(d);
        ratio = ratio.max(d / (atol + rtol * a.abs().max(b.abs())));
    }
    (ratio, diff)
}

/// One walker configuration as mvmc-core's (`ele_idx`, `ele_cfg`, `ele_num`).
pub(crate) fn arrays(cfg: &[i32], ns: usize, ne: usize) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
    let idx: Vec<i64> = cfg.iter().map(|&r| r as i64).collect();
    let mut ele_cfg = vec![-1i64; 2 * ns];
    let mut num = vec![0i64; 2 * ns];
    for (slot, &site) in cfg.iter().enumerate() {
        let k = site as usize + if slot >= ne { ns } else { 0 };
        num[k] = 1;
        ele_cfg[k] = (slot % ne) as i64;
    }
    (idx, ele_cfg, num)
}

/// CPU C-order evaluation of one walker configuration.
pub(crate) struct Evaluation {
    pub counts: Vec<i64>,
    pub ip: f64,
    /// Largest Frobenius condition number `|A|_F |A^-1|_F` over QP planes.
    pub kappa: f64,
    pub energy: f64,
    pub derivatives: Vec<f64>,
    /// C MakeRBMCnt hidden fields (empty without RBM).
    pub theta: Vec<f64>,
}

/// The Rust CPU C-order model with the given (real) parameters.
pub(crate) struct Cpu {
    pub data: mvmc_core::ExpertModeData,
    pub state: VmcOptimizationState,
    pool: ThreadedPfaPackWorkspace,
    ns: usize,
    ne: usize,
    nq: usize,
    nproj: usize,
    nrbm: usize,
    np: usize,
}

impl Cpu {
    pub fn new(model: &Model, parameters: &[f64]) -> Self {
        let (ns, ne, nq, nproj, np) = (model.ns, model.ne, model.nq, model.nproj, model.np);
        let mut data = model.data.clone();
        let values: Vec<Complex64> = parameters.iter().map(|&x| Complex64::new(x, 0.)).collect();
        mvmc_core::unpack_parameters(&mut data, &values).unwrap();
        let mut state = VmcOptimizationState::zeros(ns, ne, nproj, np, nq, 1, false, false);
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
            nrbm: model.nrbm,
            np,
        }
    }

    pub fn counts(&self, num: &[i64]) -> Vec<i64> {
        let mut counts = vec![0i64; self.nproj];
        make_proj_cnt(&mut counts, num, &self.data);
        counts
    }

    /// C CalculateMAll + CalculateIP; the inverse tables remain in `state`.
    pub fn factor(&mut self, idx: &[i64]) -> f64 {
        let m = &mut self.state.slater_matrix;
        mvmc_core::calc_m_all_real(
            idx,
            &m.slater_elm_real,
            &mut m.inv_m_real,
            &mut m.pf_m_real,
            0,
            self.nq,
            self.ns,
            self.ne,
            &self.pool,
        )
        .unwrap();
        calculate_ip_real(&m.pf_m_real, 0, self.nq, &self.data)
    }

    /// Frobenius condition of the factored planes of `idx` (after `factor`).
    pub fn kappa(&self, idx: &[i64]) -> f64 {
        let m = &self.state.slater_matrix;
        let n = idx.len();
        let site = |i: usize| idx[i] as usize + if i >= self.ne { self.ns } else { 0 };
        (0..self.nq)
            .map(|q| {
                let mut a = 0.;
                let mut b = 0.;
                for i in 0..n {
                    for j in 0..n {
                        a += m.slater_elm_real.get(q, site(i), site(j)).powi(2);
                        b += m.inv_m_real.get(q, i, j).powi(2);
                    }
                }
                (a * b).sqrt()
            })
            .fold(0f64, f64::max)
    }

    pub fn evaluate(&mut self, cfg: &[i32]) -> Evaluation {
        self.evaluate_with(cfg, None)
    }

    /// As [`Self::evaluate`]; RBM derivatives use the given hidden fields
    /// (C RBMDiff reads the sampled rbmCnt) instead of a fresh MakeRBMCnt.
    pub fn evaluate_with(&mut self, cfg: &[i32], theta: Option<&[f64]>) -> Evaluation {
        let (idx, ele_cfg, num) = arrays(cfg, self.ns, self.ne);
        let counts = self.counts(&num);
        let ip = self.factor(&idx);
        let kappa = self.kappa(&idx);
        let z = Complex64::new(ip, 0.);
        let energy = calculate_local_energy(
            z,
            &self.data,
            &mut self.state,
            &idx,
            &ele_cfg,
            &num,
            &counts,
        );
        assert_eq!(energy.im, 0.);
        // The public `slater_elm_diff` reads the complex tables (C
        // SlaterElmDiff_fcmp); mirror the real C-order tables exactly.
        let m = &mut self.state.slater_matrix;
        for (dst, &src) in m
            .inv_m
            .as_mut_slice()
            .iter_mut()
            .zip(m.inv_m_real.as_slice())
        {
            *dst = Complex64::new(src, 0.);
        }
        for (dst, &src) in m.pf_m.iter_mut().zip(&m.pf_m_real) {
            *dst = Complex64::new(src, 0.);
        }
        for (dst, &src) in m
            .slater_elm
            .as_mut_slice()
            .iter_mut()
            .zip(m.slater_elm_real.as_slice())
        {
            *dst = Complex64::new(src, 0.);
        }
        let mut o = vec![Complex64::new(0., 0.); 2 * (1 + self.np)];
        mvmc_core::observables::set_projection_diff(&mut o, &counts, self.nproj);
        // Normal C VMCMainCal: RBMDiff between projection and Slater.
        let mut fresh = vec![];
        if self.nrbm > 0 {
            let config = RbmConfig::from(&self.data);
            let mut cnt = make_rbm_cnt(&num, &config);
            let nphys = cnt.len()
                - self.data.modpara.nneuron_charge as usize
                - self.data.modpara.nneuron_spin as usize
                - self.data.modpara.nneuron_general as usize;
            fresh = cnt[nphys..].iter().map(|z| z.re).collect();
            if let Some(theta) = theta {
                for (dst, &t) in cnt[nphys..].iter_mut().zip(theta) {
                    *dst = Complex64::new(t, 0.);
                }
            }
            let offset = 2 * (1 + self.nproj);
            set_rbm_diff(&mut o[offset..offset + 2 * self.nrbm], &cnt, &num, &config);
        }
        slater_elm_diff(
            &mut o[2 * (1 + self.nproj + self.nrbm)..],
            z,
            &idx,
            &self.data,
            &self.state,
        );
        let derivatives = (0..self.np).map(|p| o[2 * (1 + p)].re).collect();
        Evaluation {
            counts,
            ip,
            kappa,
            energy: energy.re,
            derivatives,
            theta: fresh,
        }
    }
}

/// Replay the resident sampler's tape contract (`interval_hop_impl`) on the
/// host: proposal words, local-spin mask, exchange threshold and log-space
/// Metropolis decision. Projection ratios use C `UpdateProjCnt` and
/// `LogProjRatio`; overlaps are full CPU Pfaffians of every candidate.
/// Returns (final configurations, acceptance counts, smallest
/// |log u - log ratio| over all decisions). Asserts that some decisions
/// changed a DH counter, so the trajectory depends on the DH ratio.
pub(crate) fn replay(
    cpu: &mut Cpu,
    model: &Model,
    initial: &[i32],
    tape: &[u32],
    walkers: usize,
    steps: usize,
) -> (Vec<i32>, Vec<i32>, f64) {
    let (ns, ne) = (model.ns, model.ne);
    let n = 2 * ne;
    let mut cfg = initial.to_vec();
    let mut accepted = vec![0i32; walkers];
    let mut margin = f64::INFINITY;
    let mut dh_moves = 0;
    let dh = model.ng + model.nj;
    let scale = 4294967296.0;
    for w in 0..walkers {
        let e = &mut cfg[w * n..(w + 1) * n];
        let (_, _, mut num) = arrays(e, ns, ne);
        let mut counts = cpu.counts(&num);
        let ip = cpu.factor(&e.iter().map(|&r| r as i64).collect::<Vec<_>>());
        let mut log_old = (ip.abs() + 1e-100).ln();
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
                let empty: Vec<usize> = (0..ns)
                    .filter(|&r| num[r + spin * ns] == 0 && model.localspins[r] == 0)
                    .collect();
                let valid = !empty.is_empty() && model.localspins[ri] == 0;
                let rj = if empty.is_empty() {
                    0
                } else {
                    empty[(t[2] % empty.len() as u32) as usize]
                };
                (rj, usize::MAX, valid)
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
                // C's incremental update agrees with a full recount.
                assert_eq!(proposed, cpu.counts(&num), "UpdateProjCnt");
            }
            if exchange {
                assert_eq!(proposed, counts, "an exchange keeps every DH counter");
            }
            dh_moves += usize::from(proposed[dh..] != counts[dh..]);
            let proj = log_proj_ratio(&proposed, &counts, &cpu.data);
            let ip_new = cpu.factor(&e.iter().map(|&r| r as i64).collect::<Vec<_>>());
            let log_new = (ip_new.abs() + 1e-100).ln();
            let ratio = 2.0 * (proj + log_new - log_old);
            let u = (t[4] as f64 / scale).ln();
            if u.is_finite() {
                margin = margin.min((u - ratio).abs());
            }
            if ip_new.is_finite() && ip_new != 0.0 && u < ratio {
                accepted[w] += 1;
                log_old = log_new;
                counts = proposed;
            } else {
                e.copy_from_slice(&previous.0);
                num = previous.1;
            }
        }
    }
    assert!(dh_moves > 0, "no proposal changed a DH counter");
    (cfg, accepted, margin)
}

fn download<T: Copy + Default>(
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

/// GPU observation of the current walkers.
struct Measured {
    counts: Vec<i32>,
    energy: Vec<f64>,
    derivatives: Vec<f64>,
    overlaps: Vec<f64>,
    /// Hamiltonian term values `[walker][term]` (ratios, before coefficients).
    terms: Vec<f64>,
    /// Accepted updates since each walker's last refactor.
    aw: Vec<i32>,
}

fn measure(backend: &mut CudaEngine, state: &State) -> Measured {
    with_raw_exec(backend, "dh.measure", |exec| {
        let module = exec.module(SOURCE, KERNELS)?;
        state.observe(exec, &module)?;
        let w = state.resident.sampler.geom.walkers;
        let m = &state.model;
        Ok(Measured {
            counts: download(exec, &state.buffers.counts, w * m.nproj)?,
            energy: download(exec, &state.buffers.local_energy, w)?,
            derivatives: download(exec, &state.buffers.derivatives, w * m.np)?,
            overlaps: download(exec, &state.buffers.overlaps, w)?,
            terms: download(exec, &state.buffers.values, w * m.energies.len())?,
            aw: download(exec, &state.resident.sampler.aw, w)?,
        })
    })
    .map_err(|e| e.to_string())
    .unwrap()
}

/// Compare one stage with the CPU C-order evaluation of the same walkers.
fn assert_stage(name: &str, stage: &str, cpu: &mut Cpu, state: &State, backend: &mut CudaEngine) {
    let cfg = {
        let s: &mut CudaEngine = backend;
        with_raw_exec(s, "dh.cfg", |exec| {
            let g = state.resident.sampler.geom;
            download::<i32>(exec, &state.resident.sampler.ele, g.walkers * g.n)
        })
        .map_err(|e| e.to_string())
        .unwrap()
    };
    let g = measure(backend, state);
    let (counts, derivatives) = (&g.counts, &g.derivatives);
    let m = &state.model;
    let n = 2 * m.ne;
    let walkers = cfg.len() / n;
    let mut cpu_overlaps = vec![];
    let mut dh_counts = 0;
    let (mut energy_error, mut derivative_error, mut energy_abs, mut derivative_abs) =
        (0f64, 0f64, 0f64, 0f64);
    let mut kappa_max = 0f64;
    for w in 0..walkers {
        let e = cpu.evaluate(&cfg[w * n..(w + 1) * n]);
        let gpu: Vec<i64> = counts[w * m.nproj..(w + 1) * m.nproj]
            .iter()
            .map(|&c| c as i64)
            .collect();
        assert_eq!(
            gpu, e.counts,
            "{name} {stage} walker {w}: C MakeProjCnt counters"
        );
        dh_counts += e.counts[m.ng + m.nj..].iter().sum::<i64>();
        // Projection derivatives are the counters themselves.
        for p in 0..m.nproj {
            assert_eq!(derivatives[w * m.np + p], e.counts[p] as f64);
        }
        cpu_overlaps.push(e.ip);
        // First divergence: the device inverse (paired/blocked factor plus
        // rank-two or queued updates since the last refactor) and the CPU
        // PFAPACK inverse differ by forward error ~ n eps kappa_F per update
        // (the #656 interval bound). Every ratio inherits it, so a local energy
        // is bounded by 16 (aw+1) n eps kappa_F sum_t |c_t r_t|, and a Slater
        // derivative by the same factor times the walker's largest one.
        let budget = 16.0 * (g.aw[w] + 1) as f64 * n as f64 * f64::EPSILON * e.kappa;
        kappa_max = kappa_max.max(e.kappa);
        let terms = m.energies.len();
        let scale: f64 = (0..terms)
            .map(|t| (m.energies[t] * g.terms[w * terms + t]).abs())
            .sum();
        let d = (g.energy[w] - e.energy).abs();
        assert!(
            d <= 1e-13 + budget * scale,
            "{name} {stage} walker {w}: energy {} vs {} (kappa {:e})",
            g.energy[w],
            e.energy,
            e.kappa
        );
        energy_abs = energy_abs.max(d);
        energy_error = energy_error.max(d / (budget * scale / 16.0));
        let row = &derivatives[w * m.np..(w + 1) * m.np];
        let largest = e.derivatives.iter().fold(0f64, |a, x| a.max(x.abs()));
        for (p, (&a, &b)) in row.iter().zip(&e.derivatives).enumerate().skip(m.nproj) {
            let d = (a - b).abs();
            assert!(
                d <= 1e-13 + budget * largest,
                "{name} {stage} walker {w} derivative {p}: {a} vs {b}"
            );
            derivative_abs = derivative_abs.max(d);
            derivative_error = derivative_error.max(d / (budget * largest / 16.0));
        }
    }
    assert!(
        dh_counts > 0,
        "{name} {stage}: no doublon or holon was sampled"
    );
    // Overlaps use the resident stage-fixture bound (opt_julia.md).
    let (ratio, diff) = worst(&g.overlaps, &cpu_overlaps, 1e-12, 5e-11);
    assert!(ratio <= 1.0, "{name} {stage} overlaps: ratio {ratio:e}");
    println!(
        "{name} {stage}: max kappa_F={kappa_max:.3e}; max abs diff overlap={diff:e} \
         energy={energy_abs:e} derivative={derivative_abs:e}; \
         normalized by (aw+1) n eps kappa_F scale: energy={energy_error:.3} \
         derivative={derivative_error:.3} (bound 16)"
    );
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
    let walkers = 8;
    let mut backend = CudaEngine::open(cuda_devices().unwrap()[0].id()).unwrap();
    let backend: &mut CudaEngine = &mut backend;
    let mut cpu = Cpu::new(&model, &model.parameters);
    let mut state = State::prepare(backend, model, &config).unwrap();
    state.initialize(backend).unwrap();
    let initial = state.resident.snapshot(backend).unwrap().ele;
    assert_stage(name, "initial", &mut cpu, &state, backend);
    let tape = tape(walkers, steps);
    state.resident.advance_with_tape(backend, &tape).unwrap();
    let last = state.resident.snapshot(backend).unwrap();
    let accepted = state.resident.accepted(backend).unwrap();
    let (cfg, host_accepted, margin) =
        replay(&mut cpu, &state.model, &initial, &tape, walkers, steps);
    assert_eq!(last.ele, cfg, "{name}: supplied-tape configurations");
    assert_eq!(accepted, host_accepted, "{name}: acceptance counts");
    assert!(accepted.iter().any(|&a| a > 0));
    println!(
        "{name}: {steps} proposals x {walkers} walkers, accepted {accepted:?}, \
         smallest decision margin {margin:e}"
    );
    assert_stage(name, "final", &mut cpu, &state, backend);
    // One SR update plus the C DH gauge, against the C-order sync.
    let (before, update, after, energy, derivatives) =
        with_raw_exec(backend, "dh.update", |exec| {
            let module = exec.module(SOURCE, KERNELS)?;
            let m = &state.model;
            let w = state.resident.sampler.geom.walkers;
            state.observe(exec, &module)?;
            let before = download::<f64>(exec, &state.buffers.parameters, m.np)?;
            let energy = download::<f64>(exec, &state.buffers.local_energy, w)?;
            let derivatives = download::<f64>(exec, &state.buffers.derivatives, w * m.np)?;
            state.update(exec, &module, 0)?;
            Ok((
                before,
                download(exec, &state.buffers.rhs, m.np)?,
                download(exec, &state.buffers.parameters, m.np)?,
                energy,
                derivatives,
            ))
        })
        .unwrap();
    state.resident.check_status(backend).unwrap();
    let relative = sr_backward_error(&state.model, &derivatives, &energy, &update);
    assert!(relative < 1e-11, "{name}: SR backward error {relative:e}");
    let pre: Vec<Complex64> = before
        .iter()
        .zip(&update)
        .map(|(p, u)| Complex64::new(p + u, 0.))
        .collect();
    let mut data = state.model.data.clone();
    mvmc_core::unpack_parameters(&mut data, &pre).unwrap();
    mvmc_core::sync::sync_modified_parameter_local(&mut data, true);
    let expected: Vec<f64> = mvmc_core::pack_parameters(&data)
        .unwrap()
        .iter()
        .map(|p| p.re)
        .collect();
    // Same IEEE operations in the same order as the C-order sync (no FMA):
    // a few ulps of headroom only.
    let (ratio, diff) = worst(&after, &expected, 1e-15, 4.0 * f64::EPSILON);
    assert!(ratio <= 1.0, "{name}: gauge ratio {ratio:e}");
    let m = &state.model;
    let dh_mean_zero = |offset: usize, count: usize, bins: usize| {
        (0..2 * count)
            .map(|x| {
                (0..bins)
                    .map(|b| after[offset + x + 2 * b * count])
                    .sum::<f64>()
                    .abs()
            })
            .fold(0f64, f64::max)
    };
    println!(
        "{name}: SR backward error={relative:e}; gauge max abs diff={diff:e}; \
         DH2/DH4 bin-mean residual={:e}/{:e}",
        dh_mean_zero(m.ng + m.nj, m.n2, 3),
        dh_mean_zero(m.ng + m.nj + 6 * m.n2, m.n4, 5)
    );
}

#[test]
#[ignore = "requires CUDA; DH2 six-site production input against the CPU C-order path"]
fn resident_opt_dh2_production_matches_cpu_c_order() {
    assert_opt_stages("dh2_production", 8);
}

#[test]
#[ignore = "requires CUDA; DH4 six-site production input against the CPU C-order path"]
fn resident_opt_dh4_production_matches_cpu_c_order() {
    assert_opt_stages("dh4_production", 8);
}

#[test]
#[ignore = "requires CUDA; DH2+DH4 L16 doped chain with exchanges against the CPU C-order path"]
fn resident_opt_dh_l16_matches_cpu_c_order() {
    assert_opt_stages("hubbard_dh_L16", 8);
}

#[test]
#[ignore = "requires CUDA; DH2+DH4 L128 chain (delayed queue, paired inverse) against the CPU C-order path"]
fn resident_opt_dh_l128_matches_cpu_c_order() {
    assert_opt_stages("hubbard_dh_L128", 1);
}

/// Complete 20-step Opt runs repeat exactly within one implementation, after
/// an intervening differently seeded run, with zero explicit transfers and one
/// stream synchronization for the whole warmup/optimization series.
#[test]
#[ignore = "requires CUDA; 20-step DH repeatability, transfers and timing"]
fn resident_opt_dh_repeats_twenty_steps() {
    for (name, walkers) in [("hubbard_dh_L16", 64), ("hubbard_dh_L128", 64)] {
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

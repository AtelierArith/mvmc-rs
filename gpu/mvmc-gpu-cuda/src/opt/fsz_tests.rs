//! General orbitals / FSZ against the Rust CPU C-order FSZ path (#714, #572).
//!
//! Julia GPU rejects FSZ ("CUDA requires sz-conserved orbitals",
//! `ext/cuda_state.jl:38`), so the references are mvmc-core's C-order FSZ
//! kernels evaluated on the configurations the GPU sampled:
//! `update_slater_elm_fsz` (C UpdateSlaterElm_fsz), `calc_m_all_fsz_real`
//! (CalculateMAll_fsz_real), `calculate_ip_real`,
//! `calculate_local_energy_fsz` (CalculateHamiltonian_fsz_real),
//! `slater_elm_diff_fsz` (SlaterElmDiff_fsz) and
//! `sync_modified_parameter_local` (SyncModifiedParameter). The supplied-tape
//! trajectory is replayed on the host with the FSZ tape contract of
//! `interval_kernels.cu`, C `UpdateProjCnt_fsz`/`LogProjRatio` and full CPU
//! FSZ Pfaffians of every candidate. Integers (configurations, spins,
//! acceptance counts, counters) are exact; FP64 bounds are stated at each
//! assertion.
use super::complex_tests::{budget, download};
use super::dh_tests::{tape, worst};
use super::scope_tests::fsz_input as input;
use super::tests::{close, sr_backward_error};
use super::*;
use mvmc_core::observables::{
    calculate_ip_real, calculate_local_energy_fsz, set_projection_diff, slater_elm_diff_fsz,
};
use mvmc_core::sampling::projection::{log_proj_ratio, make_proj_cnt, update_proj_cnt};
use mvmc_core::{ThreadedPfaPackWorkspace, VmcOptimizationState};
use num_complex::Complex64;
use tenferro_ext::raw_exec::raw_exec_counters;

/// One FSZ walker (`rs` per slot) as mvmc-core's (`ele_idx`, `ele_spn`,
/// `ele_cfg`, `ele_num`): C EleIdx, EleSpn, EleCfg and EleNum.
pub(crate) fn arrays(cfg: &[i32], ns: usize) -> (Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>) {
    let idx = cfg.iter().map(|&rs| (rs as usize % ns) as i64).collect();
    let spn = cfg.iter().map(|&rs| (rs as usize / ns) as i64).collect();
    let mut ele_cfg = vec![-1i64; 2 * ns];
    let mut num = vec![0i64; 2 * ns];
    for (slot, &rs) in cfg.iter().enumerate() {
        assert_eq!(ele_cfg[rs as usize], -1, "doubly occupied rs {rs}");
        ele_cfg[rs as usize] = slot as i64;
        num[rs as usize] = 1;
    }
    (idx, spn, ele_cfg, num)
}

/// CPU C-order FSZ evaluation of one walker configuration.
pub(crate) struct Evaluation {
    pub counts: Vec<i64>,
    pub ip: f64,
    /// `sum_q |QPFullWeight_q Pf_q|`, the scale of the overlap sum.
    pub ip_scale: f64,
    /// Largest Frobenius condition number `|A|_F |A^-1|_F` over QP planes.
    pub kappa: f64,
    pub energy: f64,
    /// C SROptO real slots `2(1+p)`, one per parameter.
    pub derivatives: Vec<f64>,
}

/// The Rust CPU C-order FSZ model with the given (real) parameters.
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
        let (ns, ne, nq, nproj, np) = (model.ns, model.ne, model.nq, model.nproj, model.np);
        let mut data = model.data.clone();
        let values: Vec<Complex64> = parameters.iter().map(|&x| Complex64::new(x, 0.)).collect();
        mvmc_core::unpack_parameters(&mut data, &values).unwrap();
        let mut state = VmcOptimizationState::zeros(ns, ne, nproj, np, nq, 1, false, true);
        state.validate_declared_mode(&data).unwrap();
        mvmc_core::slater_update::update_slater_elm_fsz(&mut data, &mut state);
        // C vmcmain.c:369-372 (and vmc_make_sample_fsz_real): the real FSZ
        // kernels read SlaterElm_real = creal(SlaterElm).
        let m = &mut state.slater_matrix;
        for (dst, src) in m
            .slater_elm_real
            .as_mut_slice()
            .iter_mut()
            .zip(m.slater_elm.as_slice())
        {
            *dst = src.re;
        }
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

    /// C CalculateMAll_fsz_real + CalculateIP_real. As C VMCMainCal_fsz,
    /// the complex shadow (InvM, PfM) is refreshed for the Green and
    /// derivative kernels; the tables remain in `state`.
    pub fn factor(&mut self, cfg: &[i32]) -> f64 {
        let (idx, spn, _, _) = arrays(cfg, self.ns);
        let m = &mut self.state.slater_matrix;
        mvmc_core::calc_m_all_fsz_real(&idx, &spn, m, 0, self.nq, self.ns, self.ne, &self.pool)
            .unwrap();
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
        calculate_ip_real(&m.pf_m_real, 0, self.nq, &self.data)
    }

    /// Frobenius condition of the factored planes of `cfg` (after `factor`).
    pub fn kappa(&self, cfg: &[i32]) -> f64 {
        let m = &self.state.slater_matrix;
        let n = cfg.len();
        (0..self.nq)
            .map(|q| {
                let (mut a, mut b) = (0., 0.);
                for i in 0..n {
                    for j in 0..n {
                        a += m
                            .slater_elm
                            .get(q, cfg[i] as usize, cfg[j] as usize)
                            .re
                            .powi(2);
                        b += m.inv_m_real.get(q, i, j).powi(2);
                    }
                }
                (a * b).sqrt()
            })
            .fold(0f64, f64::max)
    }

    pub fn evaluate(&mut self, cfg: &[i32]) -> Evaluation {
        let (idx, spn, ele_cfg, num) = arrays(cfg, self.ns);
        let counts = self.counts(&num);
        let ip = self.factor(cfg);
        let kappa = self.kappa(cfg);
        let weights = &self.data.qp_weights.as_ref().unwrap().qp_full_weight;
        let ip_scale = (0..self.nq)
            .map(|q| (weights[q].re * self.state.slater_matrix.pf_m_real[q]).abs())
            .sum();
        let z = Complex64::new(ip, 0.);
        let energy = calculate_local_energy_fsz(
            z,
            &self.data,
            &mut self.state,
            &idx,
            &ele_cfg,
            &num,
            &counts,
            &spn,
        );
        assert_eq!(energy.im, 0., "real FSZ energy");
        let mut o = vec![Complex64::new(0., 0.); 2 * (1 + self.np)];
        set_projection_diff(&mut o, &counts, self.nproj);
        slater_elm_diff_fsz(
            &mut o[2 * (1 + self.nproj)..],
            z,
            &idx,
            &spn,
            &self.data,
            &self.state,
        );
        Evaluation {
            counts,
            ip,
            ip_scale,
            kappa,
            energy: energy.re,
            derivatives: (0..self.np).map(|p| o[2 * (1 + p)].re).collect(),
        }
    }
}

/// Host replay of the resident FSZ tape contract (`interval_hop_impl` with
/// `fsz`): slot `w0 % 2Ne`, the update type from `w3` (exchange, local spin
/// flip or hop, C getUpdateType), the `2Sz=-1` hop kind `w1 >> 31`, the
/// destination rank or partner `w2`, and the log-space decision on
/// `2 (LogProjRatio + log|ip_new| - log|ip_old|)` with the 1e-100 floor.
pub(crate) struct Replay {
    pub cfg: Vec<i32>,
    pub accepted: Vec<i32>,
    /// Smallest |log u - log ratio| over all decisions.
    pub margin: f64,
    /// Accepted moves that changed an electron's spin (hops and flips).
    pub spin_moves: usize,
    /// Accepted exchanges.
    pub exchanges: usize,
    /// Accepted local spin flips (C makeCandidate_LocalSpinFlip_localspin).
    pub local_flips: usize,
}

pub(crate) fn replay(
    cpu: &mut Cpu,
    model: &Model,
    initial: &[i32],
    tape: &[u32],
    walkers: usize,
    steps: usize,
) -> Replay {
    let ns = model.ns;
    let n = 2 * model.ne;
    let loc = |r: usize| model.localspins[r % ns] == 1;
    let mut cfg = initial.to_vec();
    let mut accepted = vec![0i32; walkers];
    let (mut margin, mut spin_moves, mut exchanges, mut local_flips) = (f64::INFINITY, 0, 0, 0);
    let scale = 4294967296.0;
    let log_abs = |x: f64| (x.abs() + 1e-100).ln();
    for w in 0..walkers {
        let e = &mut cfg[w * n..(w + 1) * n];
        let (_, _, _, mut num) = arrays(e, ns);
        let mut counts = cpu.counts(&num);
        let mut log_old = log_abs(cpu.factor(e));
        for step in 0..steps {
            let t = &tape[5 * (w + walkers * step)..][..5];
            let mi = (t[0] % n as u32) as usize;
            let from = e[mi] as usize;
            let (ri, s) = (from % ns, from / ns);
            let u3 = (t[3] as f64) / scale;
            let exchange = u3 < model.exchange;
            let local_flip = !exchange && u3 < model.exchange + model.flip;
            let (mut partner, mut pfrom, mut pto) = (usize::MAX, 0, 0);
            let (valid, to) = if local_flip {
                let to = ri + (1 - s) * ns;
                (loc(ri) && num[to] == 0, to)
            } else if exchange {
                partner = (t[2] % n as u32) as usize;
                pfrom = e[partner] as usize;
                let rj = pfrom % ns;
                pto = ri + (1 - s) * ns;
                let valid = pfrom / ns == 1 - s && num[pto] == 0 && num[rj + s * ns] == 0;
                (valid, rj + s * ns)
            } else if model.spin_free && t[1] >> 31 == 1 {
                let to = ri + (1 - s) * ns;
                (!loc(ri) && num[to] == 0, to)
            } else {
                let range = if model.spin_free {
                    0..2 * ns
                } else {
                    s * ns..(s + 1) * ns
                };
                let empty: Vec<usize> = range.filter(|&r| num[r] == 0 && !loc(r)).collect();
                if empty.is_empty() || loc(ri) {
                    (false, 0)
                } else {
                    (true, empty[(t[2] % empty.len() as u32) as usize])
                }
            };
            if !valid {
                continue;
            }
            let previous = (e.to_vec(), num.clone());
            e[mi] = to as i32;
            num[from] = 0;
            num[to] = 1;
            let mut proposed = vec![0i64; counts.len()];
            if exchange {
                e[partner] = pto as i32;
                num[pfrom] = 0;
                num[pto] = 1;
                make_proj_cnt(&mut proposed, &num, &cpu.data);
                assert_eq!(proposed, counts, "an exchange changes no counter");
            } else {
                // C UpdateProjCnt_fsz: spin-agnostic post-move totals, and no
                // change for an on-site flip (ri == rj).
                update_proj_cnt(
                    ri as i64,
                    (to % ns) as i64,
                    s as u8,
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
            let ip_new = cpu.factor(e);
            let log_new = log_abs(ip_new);
            let ratio = 2.0 * (proj + log_new - log_old);
            let u = (t[4] as f64 / scale).ln();
            if u.is_finite() {
                margin = margin.min((u - ratio).abs());
            }
            if ip_new.is_finite() && ip_new != 0.0 && u < ratio {
                accepted[w] += 1;
                log_old = log_new;
                counts = proposed;
                spin_moves += usize::from(!exchange && to / ns != s);
                exchanges += usize::from(exchange);
                local_flips += usize::from(local_flip);
            } else {
                e.copy_from_slice(&previous.0);
                num = previous.1;
            }
        }
        // C makeInitialSample_fsz spins: 2Sz >= 0 keeps the up count.
        if !model.spin_free {
            assert_eq!(
                e.iter().filter(|&&rs| (rs as usize) < ns).count(),
                model.nup,
                "walker {w}: fixed 2Sz sector"
            );
        }
    }
    Replay {
        cfg,
        accepted,
        margin,
        spin_moves,
        exchanges,
        local_flips,
    }
}

/// Exact initial-configuration contract (#714): every slot holds a distinct
/// `rs`, the labels below `nup` are up (C makeInitialSample_fsz), and every
/// local-spin site holds exactly one electron.
pub(crate) fn assert_initial(model: &Model, cfg: &[i32]) {
    let (ns, n) = (model.ns, 2 * model.ne);
    for (w, walker) in cfg.chunks_exact(n).enumerate() {
        let (_, spn, _, num) = arrays(walker, ns);
        for (a, &s) in spn.iter().enumerate() {
            assert_eq!(
                s,
                i64::from(a >= model.nup),
                "walker {w} slot {a}: spin label"
            );
        }
        for r in (0..ns).filter(|&r| model.localspins[r] == 1) {
            assert_eq!(num[r] + num[r + ns], 1, "walker {w}: local spin {r}");
        }
    }
}

struct Measured {
    counts: Vec<i32>,
    energy: Vec<f64>,
    derivatives: Vec<f64>,
    overlaps: Vec<f64>,
    terms: Vec<f64>,
    aw: Vec<i32>,
}

fn measure(backend: &mut CudaEngine, state: &State) -> Measured {
    with_raw_exec(backend, "fsz.measure", |exec| {
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

/// Compare one stage with the CPU C-order FSZ evaluation of the same walkers.
fn assert_stage(name: &str, stage: &str, cpu: &mut Cpu, state: &State, backend: &mut CudaEngine) {
    let cfg = with_raw_exec(backend, "fsz.cfg", |exec| {
        let g = state.resident.sampler.geom;
        download::<i32>(exec, &state.resident.sampler.ele, g.walkers * g.n)
    })
    .map_err(|e| e.to_string())
    .unwrap();
    let g = measure(backend, state);
    let m = &state.model;
    let n = 2 * m.ne;
    let walkers = cfg.len() / n;
    let terms = m.energies.len();
    let (mut overlap_error, mut energy_error, mut derivative_error) = (0f64, 0f64, 0f64);
    let (mut energy_abs, mut kappa_max, mut spin_flip_terms) = (0f64, 0f64, 0usize);
    for w in 0..walkers {
        let walker = &cfg[w * n..(w + 1) * n];
        let e = cpu.evaluate(walker);
        let gpu: Vec<i64> = g.counts[w * m.nproj..(w + 1) * m.nproj]
            .iter()
            .map(|&c| c as i64)
            .collect();
        assert_eq!(gpu, e.counts, "{name} {stage} walker {w}: counters");
        let row = &g.derivatives[w * m.np..(w + 1) * m.np];
        for (p, (&o, &c)) in row.iter().zip(&e.counts).enumerate() {
            assert_eq!(o, c as f64, "{name} {stage}: projection {p}");
        }
        // First divergence: the device inverse (factor plus rank-two or
        // queued updates since the last refactor) and the CPU PFAPACK inverse
        // differ by ~ n eps kappa_F per update, the #656 interval bound;
        // every ratio inherits it.
        let budget = budget(g.aw[w], n, e.kappa);
        kappa_max = kappa_max.max(e.kappa);
        let d = (g.overlaps[w] - e.ip).abs();
        assert!(
            d <= 1e-13 + budget * e.ip_scale,
            "{name} {stage} walker {w}: overlap {} vs {}",
            g.overlaps[w],
            e.ip
        );
        overlap_error = overlap_error.max(d / (budget * e.ip_scale / 16.0));
        let v = &g.terms[w * terms..(w + 1) * terms];
        let scale: f64 = (0..terms).map(|t| (m.energies[t] * v[t]).abs()).sum();
        // Spin-flip operators that act nontrivially on this walker.
        spin_flip_terms += m
            .operators
            .as_chunks::<7>()
            .0
            .iter()
            .zip(v)
            .filter(|(op, &x)| op[0] / m.ns as i32 != op[1] / m.ns as i32 && x != 0.0)
            .count();
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
        let largest = e.derivatives[m.nproj..]
            .iter()
            .fold(0f64, |a, x| a.max(x.abs()));
        for (p, (&a, &b)) in row.iter().zip(&e.derivatives).enumerate().skip(m.nproj) {
            let d = (a - b).abs();
            assert!(
                d <= 1e-13 + budget * largest,
                "{name} {stage} walker {w} Slater derivative {p}: {a} vs {b}"
            );
            derivative_error = derivative_error.max(d / (budget * largest / 16.0));
        }
    }
    if m.spin_free {
        assert!(
            spin_flip_terms > 0,
            "{name} {stage}: no spin-flip operator contributed"
        );
    }
    println!(
        "{name} {stage}: max kappa_F={kappa_max:.3e}; max abs diff energy={energy_abs:e}; \
         nonzero spin-flip terms={spin_flip_terms}; normalized by (aw+1) n eps kappa_F scale: \
         overlap={overlap_error:.3} energy={energy_error:.3} Slater \
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
    assert!(model.fsz);
    let walkers = 8;
    let mut backend = CudaEngine::open(cuda_devices().unwrap()[0].id()).unwrap();
    let backend: &mut CudaEngine = &mut backend;
    let mut cpu = Cpu::new(&model, &model.parameters);
    let mut state = State::prepare(backend, model, &config).unwrap();
    state.initialize(backend).unwrap();
    let initial = state.resident.snapshot(backend).unwrap().ele;
    assert_initial(&state.model, &initial);
    assert_stage(name, "initial", &mut cpu, &state, backend);
    let tape = tape(walkers, steps);
    state.resident.advance_with_tape(backend, &tape).unwrap();
    let last = state.resident.snapshot(backend).unwrap();
    let accepted = state.resident.accepted(backend).unwrap();
    let host = replay(&mut cpu, &state.model, &initial, &tape, walkers, steps);
    assert_eq!(last.ele, host.cfg, "{name}: supplied-tape configurations");
    assert_eq!(accepted, host.accepted, "{name}: acceptance counts");
    assert!(accepted.iter().any(|&a| a > 0));
    assert_eq!(
        host.spin_moves > 0,
        state.model.spin_free,
        "{name}: spin-changing moves"
    );
    if state.model.exchange > 0.0 {
        assert!(host.exchanges > 0, "{name}: no exchange accepted");
    }
    if state.model.flip > 0.0 {
        assert!(host.local_flips > 0, "{name}: no local spin flip accepted");
    }
    println!(
        "{name}: {steps} proposals x {walkers} walkers, accepted {accepted:?}, {} spin-changing \
         moves ({} local spin flips) and {} exchanges accepted, smallest decision margin {:e}",
        host.spin_moves, host.local_flips, host.exchanges, host.margin
    );
    assert_stage(name, "final", &mut cpu, &state, backend);
    // One SR update plus the gauge, against the CPU SR system and the
    // C-order sync; then the rebuilt state against a CPU model with the new
    // parameters.
    let (before, update, after, energy, derivatives) =
        with_raw_exec(backend, "fsz.update", |exec| {
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
    let m = &state.model;
    let relative = sr_backward_error(m, &derivatives, &energy, &update);
    assert!(relative < 1e-11, "{name}: SR backward error {relative:e}");
    assert!(
        (m.nproj..m.np).any(|p| update[p] != 0.0),
        "{name}: SR moved no orbital"
    );
    let pre: Vec<Complex64> = before
        .iter()
        .zip(&update)
        .map(|(p, u)| Complex64::new(p + u, 0.))
        .collect();
    let mut data = m.data.clone();
    mvmc_core::unpack_parameters(&mut data, &pre).unwrap();
    mvmc_core::sync::sync_modified_parameter_local(&mut data, true);
    let expected: Vec<f64> = mvmc_core::pack_parameters(&data)
        .unwrap()
        .iter()
        .map(|p| p.re)
        .collect();
    // Same IEEE operations in the same order as the C-order sync (no FMA).
    let (ratio, diff) = worst(&after, &expected, 1e-15, 4.0 * f64::EPSILON);
    assert!(ratio <= 1.0, "{name}: gauge ratio {ratio:e}");
    println!("{name}: SR backward error={relative:e}; gauge max abs diff={diff:e}");
    let mut updated = Cpu::new(&state.model, &after);
    assert_stage(name, "after SR", &mut updated, &state, backend);
}

#[test]
#[ignore = "requires CUDA; FSZ L16 (2Sz=-1, spin-flip terms, exchanges) against the CPU C-order path"]
fn resident_opt_fsz_l16_matches_cpu_c_order() {
    assert_opt_stages("hubbard_fsz_L16", 8);
}

#[test]
#[ignore = "requires CUDA; FSZ Kondo chain (2Sz=2, local spins, AP+P) against the CPU C-order path"]
fn resident_opt_fsz_kondo_2sz2_matches_cpu_c_order() {
    assert_opt_stages("kondo_fsz_2sz2", 8);
}

#[test]
#[ignore = "requires CUDA; FSZ L128 (delayed queue, paired factor) against the CPU C-order path"]
fn resident_opt_fsz_l128_matches_cpu_c_order() {
    assert_opt_stages("hubbard_fsz_L128", 1);
}

#[test]
#[ignore = "requires CUDA; FSZ KondoGC chain (NExUpdatePath=3) against the CPU C-order path"]
fn resident_opt_fsz_kondo_kgc_matches_cpu_c_order() {
    assert_opt_stages("kondo_fsz_kgc", 8);
}

#[test]
#[ignore = "requires CUDA; FSZ KondoGC 64+64 (local spin flips, delayed queue) against the CPU"]
fn resident_opt_fsz_kondo_kgc_l128_matches_cpu_c_order() {
    assert_opt_stages("kondo_fsz_kgc_L128", 1);
}

#[test]
#[ignore = "requires CUDA; FSZ Heisenberg chain (exchange and local spin flips) against the CPU"]
fn resident_opt_fsz_heisenberg_matches_cpu_c_order() {
    assert_opt_stages("heisenberg_fsz_L16", 8);
}

/// #573 per-plane recovery with FSZ walkers. Marking one plane of every
/// walker invalid routes the energies through `opt_energy_fallback` (full
/// Pfaffians of every moved rs configuration) and the derivatives through
/// `opt_cofactors` (exact minors); both must still match the CPU C-order FSZ
/// evaluation. Sampling then scores every proposal with full Pfaffians and
/// refactors on accept, and the supplied-tape trajectory must still equal the
/// host replay exactly.
#[test]
#[ignore = "requires CUDA; FSZ invalid-plane fallbacks against the CPU C-order path"]
fn resident_opt_fsz_invalid_plane_fallbacks_match_cpu() {
    let name = "hubbard_fsz_L16";
    let probe = Model::prepare(&input(name), &stage_config(1)).unwrap();
    let steps = 4 * probe.ns;
    let config = stage_config(steps);
    let model = Model::prepare(&input(name), &config).unwrap();
    let walkers = 8;
    let mut backend = CudaEngine::open(cuda_devices().unwrap()[0].id()).unwrap();
    let backend: &mut CudaEngine = &mut backend;
    let mut cpu = Cpu::new(&model, &model.parameters);
    let mut state = State::prepare(backend, model, &config).unwrap();
    state.initialize(backend).unwrap();
    let nq = state.model.nq;
    let mut valid = vec![1i32; walkers * nq];
    for w in 0..walkers {
        valid[w * nq + w % nq] = 0;
    }
    with_raw_exec(backend, "fsz.invalidate", |exec| unsafe {
        exec.upload(exec.ptr(&state.resident.valid)?, &valid)
    })
    .unwrap();
    let initial = state.resident.snapshot(backend).unwrap().ele;
    assert_stage(name, "invalid planes", &mut cpu, &state, backend);
    let tape = tape(walkers, steps);
    state.resident.advance_with_tape(backend, &tape).unwrap();
    let last = state.resident.snapshot(backend).unwrap();
    let accepted = state.resident.accepted(backend).unwrap();
    let host = replay(&mut cpu, &state.model, &initial, &tape, walkers, steps);
    assert_eq!(last.ele, host.cfg, "{name}: full-Pfaffian configurations");
    assert_eq!(accepted, host.accepted, "{name}: full-Pfaffian acceptance");
    assert!(accepted.iter().all(|&a| a > 0), "every walker refactored");
    let restored = with_raw_exec(backend, "fsz.valid", |exec| {
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
#[ignore = "requires CUDA; 20-step FSZ repeatability, transfers and timing"]
fn resident_opt_fsz_repeats_twenty_steps() {
    for (name, walkers) in [
        ("hubbard_fsz_L16", 64),
        ("kondo_fsz_2sz2", 64),
        ("hubbard_fsz_L128", 64),
        ("kondo_fsz_kgc", 64),
        ("heisenberg_fsz_L16", 64),
        ("kondo_fsz_kgc_L128", 64),
    ] {
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

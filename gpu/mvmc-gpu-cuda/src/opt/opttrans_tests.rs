//! OptTrans weights against the Rust CPU C-order path (#674, #572).
//!
//! Julia GPU rejects variational OptTrans (`ext/cuda_state.jl:43`), so the
//! references are mvmc-core's C-order kernels evaluated on the configurations
//! the GPU sampled: `calc_m_all_real`/`calculate_ip_real` over the
//! `NQPFix * NQPOptTrans` planes, `calculate_local_energy`, `slater_elm_diff`
//! with the full weights, `opt_trans_diff` (C calculateOptTransDiff),
//! `sync_modified_parameter_local` (C SyncModifiedParameter, including the
//! OptTrans normalization) and `update_qp_weight_for` (C UpdateQPWeight). The
//! supplied-tape trajectory is replayed on the host with the GPU tape contract
//! and full CPU Pfaffians. Integers (configurations, acceptance counts,
//! counters) and the device QP weights are compared exactly; FP64 bounds are
//! stated at each assertion.
use super::dh_tests::{arrays, tape, worst, Cpu};
use super::rbm_tests::download;
use super::tests::{close, sr_backward_error};
use super::*;
use mvmc_core::sampling::projection::{log_proj_ratio, make_proj_cnt, update_proj_cnt};
use num_complex::Complex64;
use tenferro_ext::raw_exec::raw_exec_counters;

pub(crate) fn input(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/opttrans")
        .join(name)
        .join("namelist.def")
}

/// First OptTrans parameter (C `OptTrans = Para + NProj + NRBM + NSlater`).
pub(crate) fn tail(model: &Model) -> usize {
    model.np - model.nopt
}

/// CPU evaluator with only the identity sector weighted (OptTrans = 1, 0, ...):
/// the model without the optimized translations.
pub(crate) fn identity_only(model: &Model, parameters: &[f64]) -> Cpu {
    let mut parameters = parameters.to_vec();
    let t = tail(model);
    parameters[t..].fill(0.0);
    parameters[t] = 1.0;
    Cpu::new(model, &parameters)
}

/// Host replay of the resident tape contract (`interval_hop_impl`), as
/// `dh_tests::replay`, with C UpdateProjCnt/LogProjRatio and the full CPU
/// overlap `sum_q QPFullWeight[q] Pf_q` over every OptTrans plane. Returns
/// (configurations, acceptance counts, smallest |log u - log ratio|, number
/// of decisions that the identity-only weights would have decided otherwise).
pub(crate) fn replay(
    cpu: &mut Cpu,
    plain: &mut Cpu,
    model: &Model,
    initial: &[i32],
    tape: &[u32],
    walkers: usize,
    steps: usize,
) -> (Vec<i32>, Vec<i32>, f64, usize) {
    let (ns, ne) = (model.ns, model.ne);
    let n = 2 * ne;
    let mut cfg = initial.to_vec();
    let mut accepted = vec![0i32; walkers];
    let mut margin = f64::INFINITY;
    let mut flips = 0;
    let scale = 4294967296.0;
    let idx = |e: &[i32]| e.iter().map(|&r| r as i64).collect::<Vec<_>>();
    for w in 0..walkers {
        let e = &mut cfg[w * n..(w + 1) * n];
        let (_, _, mut num) = arrays(e, ns, ne);
        let mut counts = cpu.counts(&num);
        let mut log_old = (cpu.factor(&idx(e)).abs() + 1e-100).ln();
        let mut plain_old = (plain.factor(&idx(e)).abs() + 1e-100).ln();
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
            }
            let proj = log_proj_ratio(&proposed, &counts, &cpu.data);
            let ip_new = cpu.factor(&idx(e));
            let log_new = (ip_new.abs() + 1e-100).ln();
            let plain_new = (plain.factor(&idx(e)).abs() + 1e-100).ln();
            let ratio = 2.0 * (proj + log_new - log_old);
            let without = 2.0 * (proj + plain_new - plain_old);
            let u = (t[4] as f64 / scale).ln();
            if u.is_finite() {
                margin = margin.min((u - ratio).abs());
            }
            let ok = ip_new.is_finite() && ip_new != 0.0;
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
    (cfg, accepted, margin, flips)
}

struct Measured {
    counts: Vec<i32>,
    energy: Vec<f64>,
    derivatives: Vec<f64>,
    overlaps: Vec<f64>,
    terms: Vec<f64>,
    aw: Vec<i32>,
    weights: Vec<f64>,
}

fn measure(backend: &mut CudaEngine, state: &State) -> Measured {
    with_raw_exec(backend, "opttrans.measure", |exec| {
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
            weights: download(exec, &state.resident.sampler.weights, m.nq)?,
        })
    })
    .map_err(|e| e.to_string())
    .unwrap()
}

/// Compare one stage with the CPU C-order evaluation of the same walkers
/// (`cpu` holds the parameters the device state was built from).
fn assert_stage(
    name: &str,
    stage: &str,
    cpu: &mut Cpu,
    plain: &mut Cpu,
    state: &State,
    backend: &mut CudaEngine,
) {
    let cfg = with_raw_exec(backend, "opttrans.cfg", |exec| {
        let g = state.resident.sampler.geom;
        download::<i32>(exec, &state.resident.sampler.ele, g.walkers * g.n)
    })
    .map_err(|e| e.to_string())
    .unwrap();
    let g = measure(backend, state);
    let m = &state.model;
    let n = 2 * m.ne;
    let walkers = cfg.len() / n;
    let t0 = tail(m);
    let nfix = m.fix.len();
    // The device full weights are C UpdateQPWeight of the current OptTrans
    // (one IEEE product each, as the CPU), bit for bit.
    let full = cpu.data.qp_weights.as_ref().unwrap().qp_full_weight.clone();
    let opt: Vec<f64> = cpu.data.opt_trans.iter().map(|z| z.re).collect();
    assert_eq!(opt.len(), m.nopt);
    for (q, (&a, b)) in g.weights.iter().zip(&full).enumerate() {
        assert_eq!(a.to_bits(), b.re.to_bits(), "{name} {stage}: QP weight {q}");
    }
    let mut cpu_overlaps = vec![];
    let (mut energy_error, mut derivative_error, mut opt_error) = (0f64, 0f64, 0f64);
    let (mut energy_abs, mut opt_abs, mut kappa_max) = (0f64, 0f64, 0f64);
    let (mut effect, mut spread, mut bound_max) = (0f64, 0f64, 0f64);
    for w in 0..walkers {
        let walker = &cfg[w * n..(w + 1) * n];
        let e = cpu.evaluate(walker);
        let pf = cpu.state.slater_matrix.pf_m_real.clone();
        let gpu: Vec<i64> = g.counts[w * m.nproj..(w + 1) * m.nproj]
            .iter()
            .map(|&c| c as i64)
            .collect();
        assert_eq!(
            gpu, e.counts,
            "{name} {stage} walker {w}: projection counters"
        );
        let row = &g.derivatives[w * m.np..(w + 1) * m.np];
        for (&a, &c) in row.iter().zip(&e.counts).take(m.nproj) {
            assert_eq!(
                a, c as f64,
                "{name} {stage} walker {w}: projection derivative"
            );
        }
        cpu_overlaps.push(e.ip);
        kappa_max = kappa_max.max(e.kappa);
        // #656 interval bound (dh_tests::assert_stage): the relative error of
        // every sampled Pfaffian and Hamiltonian ratio.
        let budget = 16.0 * (g.aw[w] + 1) as f64 * n as f64 * f64::EPSILON * e.kappa;
        let terms = m.energies.len();
        let scale: f64 = (0..terms)
            .map(|t| (m.energies[t] * g.terms[w * terms + t]).abs())
            .sum();
        let d = (g.energy[w] - e.energy).abs();
        let bound = 1e-13 + budget * scale;
        bound_max = bound_max.max(bound);
        assert!(
            d <= bound,
            "{name} {stage} walker {w}: energy {} vs {} (kappa {:e})",
            g.energy[w],
            e.energy,
            e.kappa
        );
        energy_abs = energy_abs.max(d);
        energy_error = energy_error.max(d / (budget * scale / 16.0));
        // The reference weights every OptTrans plane: identity-only weights
        // move the local energy.
        effect = effect.max((e.energy - plain.evaluate(walker).energy).abs());
        let largest = e.derivatives[m.nproj + m.nrbm..t0]
            .iter()
            .fold(0f64, |a, x| a.max(x.abs()));
        for p in m.nproj + m.nrbm..t0 {
            let (a, b) = (row[p], e.derivatives[p]);
            let d = (a - b).abs();
            assert!(
                d <= 1e-13 + budget * largest,
                "{name} {stage} walker {w} Slater derivative {p}: {a} vs {b}"
            );
            derivative_error = derivative_error.max(d / (budget * largest / 16.0));
        }
        // OptTrans derivative O_i = A_i / ip, A_i = sum_j fix_j Pf_ij. With
        // per-plane relative errors <= budget, |dO_i| <= budget (s_i + |O_i| c),
        // s_i = sum_j |fix_j Pf_ij| / |ip|, c = sum_q |w_q Pf_q| / |ip|; the
        // fixed-order sums and the division add a few roundings of the same
        // magnitudes (the CPU divides in complex arithmetic).
        let ip = e.ip.abs();
        let c: f64 = full
            .iter()
            .zip(&pf)
            .map(|(w, p)| (w.re * p).abs())
            .sum::<f64>()
            / ip;
        let mut sum = 0f64;
        let (mut low, mut high) = (f64::INFINITY, f64::NEG_INFINITY);
        for i in 0..m.nopt {
            let p = t0 + i;
            let (a, b) = (row[p], e.derivatives[p]);
            let s: f64 = (0..nfix)
                .map(|j| (m.fix[j] * pf[i * nfix + j]).abs())
                .sum::<f64>()
                / ip;
            let rounding = 4.0 * (nfix + 2) as f64 * f64::EPSILON * (s + b.abs() * c);
            let bound = 1e-14 + budget * (s + b.abs() * c) + rounding;
            let d = (a - b).abs();
            assert!(
                d <= bound,
                "{name} {stage} walker {w} OptTrans derivative {i}: {a} vs {b} (bound {bound:e})"
            );
            opt_abs = opt_abs.max(d);
            opt_error = opt_error.max(d / bound);
            sum += opt[i] * a;
            low = low.min(b);
            high = high.max(b);
        }
        // Distinct sector overlaps: the derivative columns are not degenerate.
        spread = spread.max(high - low);
        // sum_i OptTrans_i O_i = 1 for each sample (C's layout breaks this, #370).
        assert!(
            (sum - 1.0).abs() <= 1e-10 * c.max(1.0),
            "{name} {stage} walker {w}: sum_i OptTrans_i O_i = {sum}"
        );
    }
    assert!(
        effect > 1e2 * bound_max,
        "{name} {stage}: OptTrans must change the local energy ({effect:e})"
    );
    assert!(
        spread > 1e-3,
        "{name} {stage}: degenerate OptTrans sectors ({spread:e})"
    );
    let (ratio, diff) = worst(&g.overlaps, &cpu_overlaps, 1e-12, 5e-11);
    assert!(ratio <= 1.0, "{name} {stage} overlaps: ratio {ratio:e}");
    println!(
        "{name} {stage}: max kappa_F={kappa_max:.3e}; OptTrans energy effect={effect:.3e}; \
         sector derivative spread={spread:.3e}; max abs diff overlap={diff:e} \
         energy={energy_abs:e} OptTrans derivative={opt_abs:e} (normalized {opt_error:.2e}, bound 1); normalized by (aw+1) n eps kappa_F scale: \
         energy={energy_error:.2e} Slater derivative={derivative_error:.2e} (bound 16)"
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
    assert!(model.nopt > 1 && model.nq == model.nopt * model.fix.len());
    let walkers = 8;
    let mut backend = CudaEngine::open(cuda_devices().unwrap()[0].id()).unwrap();
    let backend: &mut CudaEngine = &mut backend;
    let mut cpu = Cpu::new(&model, &model.parameters);
    let mut plain = identity_only(&model, &model.parameters);
    let mut state = State::prepare(backend, model, &config).unwrap();
    state.initialize(backend).unwrap();
    let initial = state.resident.snapshot(backend).unwrap().ele;
    assert_stage(name, "initial", &mut cpu, &mut plain, &state, backend);
    let tape = tape(walkers, steps);
    state.resident.advance_with_tape(backend, &tape).unwrap();
    let last = state.resident.snapshot(backend).unwrap();
    let accepted = state.resident.accepted(backend).unwrap();
    let (cfg, host_accepted, margin, flips) = replay(
        &mut cpu,
        &mut plain,
        &state.model,
        &initial,
        &tape,
        walkers,
        steps,
    );
    assert_eq!(last.ele, cfg, "{name}: supplied-tape configurations");
    assert_eq!(accepted, host_accepted, "{name}: acceptance counts");
    assert!(accepted.iter().any(|&a| a > 0));
    assert!(
        flips > 0,
        "{name}: the OptTrans weights changed no decision"
    );
    println!(
        "{name}: {steps} proposals x {walkers} walkers, accepted {accepted:?}, \
         {flips} decisions changed by OptTrans, smallest decision margin {margin:e}"
    );
    assert_stage(name, "final", &mut cpu, &mut plain, &state, backend);
    // One SR update plus the gauge, against the C-order sync and C
    // UpdateQPWeight; then the rebuilt state against the new parameters.
    let (before, update, after, energy, derivatives) =
        with_raw_exec(backend, "opttrans.update", |exec| {
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
    let m = &state.model;
    let t0 = tail(m);
    assert!(
        update[t0..].iter().all(|&x| x != 0.0),
        "{name}: SR left an OptTrans weight unchanged"
    );
    let pre: Vec<Complex64> = before
        .iter()
        .zip(&update)
        .map(|(p, u)| Complex64::new(p + u, 0.))
        .collect();
    let largest = pre[t0..].iter().fold(0f64, |a, x| a.max(x.re.abs()));
    assert_ne!(
        largest, 1.0,
        "{name}: the OptTrans normalization is not exercised"
    );
    let mut data = m.data.clone();
    mvmc_core::unpack_parameters(&mut data, &pre).unwrap();
    mvmc_core::sync::sync_modified_parameter_local(&mut data, true);
    let expected: Vec<f64> = mvmc_core::pack_parameters(&data)
        .unwrap()
        .iter()
        .map(|p| p.re)
        .collect();
    // Same IEEE operations in the same order as the C-order sync (no FMA):
    // Slater rescale over the Slater block only, then OptTrans *= 1/max|.|.
    let (ratio, diff) = worst(&after, &expected, 1e-15, 4.0 * f64::EPSILON);
    assert!(ratio <= 1.0, "{name}: gauge ratio {ratio:e}");
    for (a, b) in after[t0..].iter().zip(&expected[t0..]) {
        assert_eq!(a.to_bits(), b.to_bits(), "{name}: OptTrans normalization");
    }
    // The rebuild used the refreshed weights: the device state equals a CPU
    // model built from the updated parameters (weights compared bit for bit).
    let mut cpu = Cpu::new(m, &after);
    let mut plain = identity_only(m, &after);
    assert_stage(name, "after SR", &mut cpu, &mut plain, &state, backend);
    println!("{name}: SR backward error={relative:e}; gauge max abs diff={diff:e}");
}

#[test]
#[ignore = "requires CUDA; OptTrans L16 doped chain with exchanges against the CPU C-order path"]
fn resident_opt_opttrans_l16_matches_cpu_c_order() {
    assert_opt_stages("hubbard_opttrans_L16", 8);
}

#[test]
#[ignore = "requires CUDA; OptTrans L128 chain (delayed queue, paired inverse) against the CPU C-order path"]
fn resident_opt_opttrans_l128_matches_cpu_c_order() {
    assert_opt_stages("hubbard_opttrans_L128", 1);
}

/// Complete 20-step Opt runs repeat exactly within one implementation, after
/// an intervening differently seeded run, with zero explicit transfers and one
/// stream synchronization for the whole warmup/optimization series. The
/// OptTrans weights move and stay normalized. With the OptTrans flags cleared
/// (C's effective `-o` contract: `GetInfoOptTrans` never writes the OptTrans
/// component flags, `readdef.c:2400`) they keep their initial values.
#[test]
#[ignore = "requires CUDA; 20-step OptTrans repeatability, transfers and timing"]
fn resident_opt_opttrans_repeats_twenty_steps() {
    for (name, walkers) in [("hubbard_opttrans_L16", 64), ("hubbard_opttrans_L128", 64)] {
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
        let model = Model::prepare(&input(name), &config).unwrap();
        let t0 = tail(&model);
        let opt = &first.parameters[t0..];
        assert_eq!(opt.iter().fold(0f64, |a, x| a.max(x.abs())), 1.0);
        assert!(
            opt.iter()
                .zip(&model.parameters[t0..])
                .any(|(a, b)| (a - b).abs() > 1e-6),
            "{name}: 20 SR steps left the OptTrans weights unchanged: {opt:?}"
        );
        let (traffic, run_seconds, frozen) = with_runtime(0, |backend| {
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
            let mut model = Model::prepare(&input(name), &config)?;
            model.flags[t0..].fill(0);
            let mut state = State::prepare(backend, model, &config)?;
            state.initialize(backend)?;
            state.run(backend)?;
            Ok((traffic, seconds, state.result(backend)?))
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
        for (a, b) in frozen.parameters[t0..].iter().zip(&model.parameters[t0..]) {
            assert_eq!(a.to_bits(), b.to_bits(), "{name}: inactive OptTrans moved");
        }
        println!(
            "{name}: walkers={walkers} e2e first={first_seconds:.3}s warm={second_seconds:.3}s \
             run={run_seconds:.3}s traffic={traffic:?} energies[0,19]=({:.6},{:.6}) \
             OptTrans={opt:?} frozen energies[19]={:.6}",
            first.energies[0], first.energies[19], frozen.energies[19]
        );
    }
}

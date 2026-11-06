//! Complex general orbitals / FSZ against the Rust CPU C-order complex FSZ
//! path (#714 (c), #572).
//!
//! C's complex FSZ path (`AllComplexFlag != 0`, `iFlgOrbitalGeneral=1`):
//! `UpdateSlaterElm_fsz`, `CalculateMAll_fsz` (`calc_m_all_fsz_complex`),
//! `CalculateIP_fcmp`, `vmcmake_fsz.c`, `CalculateHamiltonian_fsz`
//! (`calculate_local_energy_fsz` on a complex state), `SlaterElmDiff_fsz` and
//! the complex `StochasticOpt` over `2*NPara` real components. The device
//! complex sampler uses the FSZ proposals of `resident_propose`; the host
//! replay is the FSZ replay of `fsz_tests` with complex Pfaffians (`log|ip|`).
//! The stage checks are those of `complex_tests`.
use super::complex_tests::{
    assert_stage, complex, complex_sr_backward_error, download, ComplexReference, Evaluation,
};
use super::dh_tests::{tape, worst};
use super::fsz_tests::{arrays, assert_initial, replay, FszOverlap};
use super::scope_tests::fsz_input as input;
use super::tests::close;
use super::*;
use mvmc_core::observables::{
    calculate_ip_complex, calculate_local_energy_fsz, set_projection_diff, slater_elm_diff_fsz,
};
use mvmc_core::sampling::projection::make_proj_cnt;
use mvmc_core::{ThreadedPfaPackWorkspace, VmcOptimizationState};
use num_complex::Complex64;
use tenferro_ext::raw_exec::raw_exec_counters;

/// The Rust CPU C-order complex FSZ model with the given interleaved
/// parameters.
pub(crate) struct ComplexFszCpu {
    pub data: mvmc_core::ExpertModeData,
    pub state: VmcOptimizationState,
    pool: ThreadedPfaPackWorkspace,
    ns: usize,
    ne: usize,
    nq: usize,
    nproj: usize,
    np: usize,
}

impl ComplexFszCpu {
    pub fn new(model: &Model, parameters: &[f64]) -> Self {
        let (ns, ne, nq, nproj, np) = (model.ns, model.ne, model.nq, model.nproj, model.np);
        let mut data = model.data.clone();
        mvmc_core::unpack_parameters(&mut data, &complex(parameters)).unwrap();
        let mut state = VmcOptimizationState::zeros(ns, ne, nproj, np, nq, 1, true, true);
        state.validate_declared_mode(&data).unwrap();
        mvmc_core::slater_update::update_slater_elm_fsz(&mut data, &mut state);
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

    /// C CalculateMAll_fsz + CalculateIP_fcmp; the tables remain in `state`.
    pub fn factor(&mut self, cfg: &[i32]) -> Complex64 {
        let (idx, spn, _, _) = arrays(cfg, self.ns);
        let m = &mut self.state.slater_matrix;
        mvmc_core::calc_m_all_fsz_complex(
            &idx,
            &spn,
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

    fn kappa(&self, cfg: &[i32]) -> f64 {
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
                            .norm_sqr();
                        b += m.inv_m.get(q, i, j).norm_sqr();
                    }
                }
                (a * b).sqrt()
            })
            .fold(0f64, f64::max)
    }

    /// C GreenFunc1_fsz/_fsz2 and GreenFunc2_fsz/_fsz2 (`conj(z/ip)`) on the
    /// factored tables of `cfg`.
    pub fn green(
        &self,
        cfg: &[i32],
        ip: Complex64,
        counts: &[i64],
        op: [usize; 4],
        spins: [u8; 4],
        pairs: usize,
    ) -> Complex64 {
        let (idx, spn, ele_cfg, num) = arrays(cfg, self.ns);
        let [i, j, k, l] = op;
        let [s, t, u, v] = spins;
        if pairs == 1 {
            mvmc_core::observables::green_func1_fsz_complex(
                i,
                j,
                s,
                t,
                ip,
                &self.data,
                &self.state,
                &idx,
                &ele_cfg,
                &num,
                counts,
                &spn,
            )
        } else {
            mvmc_core::observables::green_func2_fsz_complex(
                i,
                j,
                k,
                l,
                s,
                t,
                u,
                v,
                ip,
                &self.data,
                &self.state,
                &idx,
                &ele_cfg,
                &num,
                counts,
                &spn,
            )
        }
    }
}

impl ComplexReference for ComplexFszCpu {
    fn evaluate(&mut self, cfg: &[i32]) -> Evaluation {
        let (idx, spn, ele_cfg, num) = arrays(cfg, self.ns);
        let counts = self.counts(&num);
        let ip = self.factor(cfg);
        let kappa = self.kappa(cfg);
        let weights = &self.data.qp_weights.as_ref().unwrap().qp_full_weight;
        let ip_scale = (0..self.nq)
            .map(|q| (weights[q] * self.state.slater_matrix.pf_m[q]).norm())
            .sum();
        let energy = calculate_local_energy_fsz(
            ip,
            &self.data,
            &mut self.state,
            &idx,
            &ele_cfg,
            &num,
            &counts,
            &spn,
        );
        let mut o = vec![Complex64::new(0., 0.); 2 * (1 + self.np)];
        set_projection_diff(&mut o, &counts, self.nproj);
        slater_elm_diff_fsz(
            &mut o[2 * (1 + self.nproj)..],
            ip,
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
            energy,
            derivatives: o[2..].to_vec(),
        }
    }
}

impl FszOverlap for ComplexFszCpu {
    fn data(&self) -> &mvmc_core::ExpertModeData {
        &self.data
    }

    fn log_overlap(&mut self, cfg: &[i32]) -> (f64, bool) {
        let ip = self.factor(cfg);
        let ok = ip.re.is_finite() && ip.im.is_finite() && ip.norm() != 0.0;
        ((ip.norm() + 1e-100).ln(), ok)
    }
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
    assert!(model.fsz && model.complex);
    assert!(model.weights.iter().skip(1).step_by(2).any(|&x| x != 0.0));
    let walkers = 8;
    let mut backend = CudaEngine::open(cuda_devices().unwrap()[0].id()).unwrap();
    let backend: &mut CudaEngine = &mut backend;
    let mut cpu = ComplexFszCpu::new(&model, &model.parameters);
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
    assert!(
        host.spin_moves > 0,
        "{name}: no spin-changing move accepted"
    );
    if state.model.exchange > 0.0 {
        assert!(host.exchanges > 0, "{name}: no exchange accepted");
    }
    println!(
        "{name}: {steps} proposals x {walkers} walkers, accepted {accepted:?}, {} spin-changing \
         moves and {} exchanges accepted, smallest decision margin {:e}",
        host.spin_moves, host.exchanges, host.margin
    );
    assert_stage(name, "final", &mut cpu, &state, backend);
    let (before, update, after, energy, derivatives) =
        with_raw_exec(backend, "fsz.complex.update", |exec| {
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
        "{name}: SR moved no imaginary orbital component"
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
    // As complex_tests: CUDA's hypot for C's cabs is within one ulp.
    let (ratio, diff) = worst(&after, &expected, 1e-15, 4.0 * f64::EPSILON);
    assert!(ratio <= 1.0, "{name}: gauge ratio {ratio:e}");
    println!("{name}: SR backward error={relative:e}; gauge max abs diff={diff:e}");
    let mut updated = ComplexFszCpu::new(&state.model, &after);
    assert_stage(name, "after SR", &mut updated, &state, backend);
}

#[test]
#[ignore = "requires CUDA; complex FSZ L16 against the CPU C-order complex FSZ path"]
fn resident_opt_fsz_complex_l16_matches_cpu_c_order() {
    assert_opt_stages("hubbard_fsz_complex_L16", 8);
}

#[test]
#[ignore = "requires CUDA; complex FSZ L128 against the CPU C-order complex FSZ path"]
fn resident_opt_fsz_complex_l128_matches_cpu_c_order() {
    assert_opt_stages("hubbard_fsz_complex_L128", 1);
}

/// #573 per-plane recovery on the complex FSZ path: full complex Pfaffians of
/// every moved configuration, exact complex minors, refactor on accept.
#[test]
#[ignore = "requires CUDA; complex FSZ invalid-plane fallbacks against the CPU C-order path"]
fn resident_opt_fsz_complex_invalid_plane_fallbacks_match_cpu() {
    let name = "hubbard_fsz_complex_L16";
    let probe = Model::prepare(&input(name), &stage_config(1)).unwrap();
    let steps = 4 * probe.ns;
    let config = stage_config(steps);
    let model = Model::prepare(&input(name), &config).unwrap();
    let walkers = 8;
    let mut backend = CudaEngine::open(cuda_devices().unwrap()[0].id()).unwrap();
    let backend: &mut CudaEngine = &mut backend;
    let mut cpu = ComplexFszCpu::new(&model, &model.parameters);
    let mut state = State::prepare(backend, model, &config).unwrap();
    state.initialize(backend).unwrap();
    let nq = state.model.nq;
    let mut valid = vec![1i32; walkers * nq];
    for w in 0..walkers {
        valid[w * nq + w % nq] = 0;
    }
    with_raw_exec(backend, "fsz.complex.invalidate", |exec| unsafe {
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
    let restored = with_raw_exec(backend, "fsz.complex.valid", |exec| {
        download::<i32>(exec, &state.resident.valid, walkers * nq)
    })
    .unwrap();
    assert!(
        restored.iter().all(|&v| v == 1),
        "the refactor restores validity"
    );
    assert_stage(name, "after refactor", &mut cpu, &state, backend);
}

/// Complete 20-step complex FSZ Opt runs repeat exactly, with zero explicit
/// transfers and one stream synchronization.
#[test]
#[ignore = "requires CUDA; 20-step complex FSZ repeatability, transfers and timing"]
fn resident_opt_fsz_complex_repeats_twenty_steps() {
    for name in ["hubbard_fsz_complex_L16", "hubbard_fsz_complex_L128"] {
        let config = OptConfig {
            nsteps: 20,
            walkers: Some(64),
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
            "{name}: walkers=64 e2e first={first_seconds:.3}s warm={second_seconds:.3}s \
             run={run_seconds:.3}s traffic={traffic:?} energies[0,19]=({:.6},{:.6})",
            first.energies[0], first.energies[19]
        );
    }
}

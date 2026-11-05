//! Sampled SR operator contracts from Julia test_unit_stochastic_opt.jl.
#[path = "../../../tests/support/julia_fixture.rs"]
mod julia_fixture;
#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;
use mvmc_core::sr_cg::{sequential_dot, SampledSrOperator};

#[derive(Default)]
struct FixedCgDiagnostic(std::cell::RefCell<String>);
impl mvmc_core::sr_cg::CgObserver for FixedCgDiagnostic {
    fn product(&self, phase: mvmc_core::sr_cg::CgProductPhase, x: &[f64], z: &[f64]) {
        use std::fmt::Write;
        writeln!(self.0.borrow_mut(), "{phase:?} search={x:?} product={z:?}").unwrap();
    }
    fn iteration(&self, state: mvmc_core::sr_cg::CgIterationView<'_>) {
        use std::fmt::Write;
        writeln!(
            self.0.borrow_mut(),
            "iteration={} delta={:.17e} alpha={:?} solution={:?} residual={:?} direction={:?}",
            state.iteration,
            state.delta,
            state.alpha,
            state.solution,
            state.residual,
            state.direction
        )
        .unwrap();
    }
}

#[test]
fn cg_step_reads_real_store_and_normalizes_by_weight_count() {
    use mvmc_core::{ExpertModeData, VmcOptimizationState};
    use mvmc_expert_parsers::OrbitalTerm;
    use num_complex::Complex64 as C;
    let mut data = ExpertModeData::new();
    data.modpara.nsite = 1;
    data.modpara.nelec = 1;
    data.modpara.nvmc_sample = 2;
    data.modpara.n_orbital_idx = 1;
    data.slater_params = vec![C::new(10.0, 0.0)];
    data.modpara.nsrcg = 1;
    data.modpara.dsr_opt_red_cut = 0.0;
    data.modpara.dsr_opt_sta_del = 0.0;
    data.modpara.dsr_opt_step_dt = 0.5;
    data.orbital_terms.push(OrbitalTerm {
        site1: 0,
        site2: 0,
        idx: 0,
        is_complex: false,
        sign: 1,
    });
    let mut state = VmcOptimizationState::zeros(1, 1, 0, 1, 1, 2, false, false);
    state.energy.wc = C::new(4.0, 0.0);
    state.sr_opt.sr_opt_oo_real[3] = 2.0;
    state.sr_opt.sr_opt_ho_real[1] = 1.0;
    state.sr_opt.sr_opt_o_store_real[1] = 2.0;
    state.sr_opt.sr_opt_o_store_real[3] = 2.0;
    assert_eq!(
        mvmc_core::sr_cg::stochastic_opt_cg(&mut data, &state, None).unwrap(),
        0
    );
    numerical_comparison::assert_values_close(
        [data.slater_params[0].re, data.slater_params[0].im],
        [9.5, 0.0],
        16.0 * f64::EPSILON,
        16.0 * f64::EPSILON,
        "condition-one CG increment",
    );
    let dir = std::env::temp_dir().join(format!("mvmc-cg-srinfo-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    data.modpara.c_data_file_head = "custom".into();
    // mVMC 1.4.0 CG row: "%5d %d" with the iteration count (stcopt_cg_impl.c:285).
    let row = "    1     1     0     0  2.00000e+00  2.00000e+00 -5.00000e-01     0 1\n";
    let mut files = mvmc_core::output_files::RunFiles::init(
        &data,
        &dir,
        mvmc_core::output_files::RunKind::ParaOpt,
        None,
    )
    .unwrap();
    for _ in 0..2 {
        assert_eq!(
            mvmc_core::sr_cg::stochastic_opt_cg(&mut data, &state, Some(&mut files)).unwrap(),
            0
        );
    }
    files.close().unwrap();
    assert_eq!(
        std::fs::read_to_string(dir.join("custom_SRinfo.dat")).unwrap(),
        format!(
            "#Npara Msize optCut diagCut sDiagMax  sDiagMin    absRmax       imax info\n{row}{row}"
        )
    );
    data.optimization_flags = vec![0, 0];
    let before = data.slater_params.clone();
    assert_eq!(
        mvmc_core::sr_cg::stochastic_opt_cg(&mut data, &state, None).unwrap(),
        0
    );
    assert_eq!(data.slater_params, before);
    // A NaN gradient makes the initial delta non-finite: C 1.4.0 returns
    // -(0+1) and, with the default NSRCGAbortOnFail = 1 and no fallback,
    // aborts (stcopt_cg_impl.c:252-262).
    data.optimization_flags = vec![1, 0];
    state.sr_opt.sr_opt_ho_real[1] = f64::NAN;
    let error = mvmc_core::sr_cg::stochastic_opt_cg(&mut data, &state, None).unwrap_err();
    assert_eq!(
        error.to_string(),
        "error: SR-CG fallback is disabled. Abort."
    );
    assert_eq!(data.slater_params, before);
    // NSRCGAbortOnFail = 0 continues with the approximate solution x = 0 and
    // reports info = -(0+1) in zqp_SRinfo.dat.
    data.modpara.nsrcg_abort_on_fail = false;
    let mut files = mvmc_core::output_files::RunFiles::init(
        &data,
        &dir,
        mvmc_core::output_files::RunKind::ParaOpt,
        None,
    )
    .unwrap();
    assert_eq!(
        mvmc_core::sr_cg::stochastic_opt_cg(&mut data, &state, Some(&mut files)).unwrap(),
        0
    );
    files.close().unwrap();
    assert_eq!(data.slater_params, before);
    let info = std::fs::read_to_string(dir.join("custom_SRinfo.dat")).unwrap();
    assert!(
        info.lines().nth(1).unwrap().ends_with("     0 -1"),
        "{info}"
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn cg_controls_default_to_julia_tolerance_and_active_dimension_limit() {
    use mvmc_expert_parsers::parsers::modpara::parse_modpara_content;
    // C SetDefaultValuesModPara: DSROptCGTol=1e-10, NSROptCGMaxIter=0. The
    // eight-line header is part of the C file contract.
    let header = "-\nModel_Parameters 0\n-\nVMC_Cal_Parameters\n-\nCDataFileHead zvo\nCParaFileHead zqp\n-\n";
    let p = parse_modpara_content(header).unwrap();
    assert_eq!(p.dsr_opt_cg_tol, 1e-10);
    assert_eq!(p.nsr_opt_cg_max_iter, 0);
}

#[test]
fn cg_complex_component_flags_and_variance_cut_preserve_fixed_components() {
    use mvmc_core::{ExpertModeData, VmcOptimizationState};
    use mvmc_expert_parsers::OrbitalTerm;
    use num_complex::Complex64 as C;
    for (flags, cut, expected) in [
        (vec![1, 1], 0.0, C::new(9.5, 4.75)),
        (vec![1, 0], 0.0, C::new(9.5, 5.0)),
        (vec![1, 1], 0.5, C::new(10.0, 4.75)),
    ] {
        let mut data = ExpertModeData::new();
        data.modpara.nvmc_sample = 2;
        data.modpara.n_orbital_idx = 1;
        data.slater_params = vec![C::new(10.0, 5.0)];
        data.modpara.dsr_opt_step_dt = 0.5;
        data.modpara.dsr_opt_sta_del = 0.0;
        data.modpara.dsr_opt_red_cut = cut;
        data.optimization_flags = flags;
        data.orbital_terms.push(OrbitalTerm {
            site1: 0,
            site2: 0,
            idx: 0,
            is_complex: true,
            sign: 1,
        });
        let mut state = VmcOptimizationState::zeros(1, 1, 0, 1, 1, 2, true, false);
        state.energy.wc = C::new(4.0, 0.0);
        state.sr_opt.sr_opt_oo[6] = C::new(2.0, 0.0);
        state.sr_opt.sr_opt_oo[7] = C::new(8.0, 0.0);
        state.sr_opt.sr_opt_ho[2] = C::new(1.0, 0.0);
        state.sr_opt.sr_opt_ho[3] = C::new(2.0, 0.0);
        for s in 0..2 {
            state.sr_opt.sr_opt_o_store[4 * s + 2] = C::new(2.0, 0.0);
            state.sr_opt.sr_opt_o_store[4 * s + 3] = C::new(0.0, 4.0);
        }
        assert_eq!(
            mvmc_core::sr_cg::stochastic_opt_cg(&mut data, &state, None).unwrap(),
            0
        );
        assert!((data.slater_params[data.orbital_terms[0].idx as usize] - expected).norm() < 1e-14);
    }
}

#[test]
fn dot_preserves_source_sequential_accumulation() {
    let mut p = vec![1e16];
    p.extend([1.0; 100]);
    p.push(-1e16);
    // Preserve cancellation/order sensitivity: an absolute epsilon budget,
    // not a relative budget scaled by the cancelling 1e16 inputs.
    numerical_comparison::assert_close(
        sequential_dot(&p, &vec![1.0; p.len()]),
        0.0,
        4.0 * f64::EPSILON,
        0.0,
        "sequential cancellation",
    );
}

#[test]
fn real_operator_applies_sampled_product_before_global_correction() {
    let mut op = SampledSrOperator::new(2, 2, false);
    op.real_samples.copy_from_slice(&[1.0, 3.0, 2.0, 4.0]);
    op.mean.copy_from_slice(&[0.25, -0.5]);
    op.diagonal.copy_from_slice(&[2.0, 3.0]);
    let x = [0.5, -1.0];
    let mut z = [0.0; 2];
    op.apply(&mut z, &x, 0.25, 0.1);
    assert_eq!(x, [0.5, -1.0]);
    for (actual, expected) in z.into_iter().zip([-2.18125, -4.8625]) {
        assert!((actual - expected).abs() < 1e-14);
    }
}

#[test]
fn complex_operator_adds_imaginary_sample_gram() {
    let mut op = SampledSrOperator::new(2, 2, true);
    op.real_samples.copy_from_slice(&[1.0, 3.0, 2.0, 4.0]);
    op.imag_samples.copy_from_slice(&[2.0, -1.0, 0.0, 3.0]);
    op.mean.copy_from_slice(&[0.25, -0.5]);
    op.diagonal.copy_from_slice(&[2.0, 3.0]);
    let mut z = [0.0; 2];
    op.apply(&mut z, &[0.5, -1.0], 0.25, 0.1);
    // Re(O O^H) = [[9,9],[9,35]], before mean/diagonal correction.
    for (actual, expected) in z.into_iter().zip([-1.18125, -7.6125]) {
        assert!((actual - expected).abs() < 1e-14);
    }
}

#[test]
fn standard_cg_solves_sampled_spd_matrix() {
    let mut op = SampledSrOperator::new(2, 2, false);
    // Sample Gram [[4,2],[2,2]], no means or regularization.
    op.real_samples.copy_from_slice(&[2.0, 1.0, 0.0, 1.0]);
    let result = op.solve(&[6.0, 4.0], 1.0, 0.0, 1e-14, 10);
    assert_eq!(result.iterations, 2);
    assert!((result.solution[0] - 1.0).abs() < 1e-14);
    assert!((result.solution[1] - 1.0).abs() < 1e-14);
}

#[test]
fn cg_reports_c_1_4_0_breakdown_and_non_convergence_status() {
    use mvmc_core::sr_cg::CgVariant;
    let mut op = SampledSrOperator::new(2, 2, false);
    // Zero gradient: delta = 0 is below the threshold before any iteration.
    let zero = op.solve(&[0.0, 0.0], 1.0, 0.0, 1e-6, 10);
    assert_eq!((zero.status, zero.iterations), (0, 0));
    assert_eq!(zero.solution, [0.0, 0.0]);
    // Zero operator, nonzero gradient: d.q = 0 < 1e-20, so C 1.4.0 returns
    // -(0+1) before forming alpha (stcopt_cg_impl.c:539-540); 1.3.0 divided by
    // zero and exhausted the iteration limit with NaN iterates.
    let breakdown = op.solve(&[1.0, 0.0], 1.0, 0.0, 1e-6, 10);
    assert_eq!((breakdown.status, breakdown.iterations), (-1, 0));
    assert_eq!(breakdown.solution, [0.0, 0.0]);
    assert!(!breakdown.converged());
    // No iteration allowed: the limit is reached with delta >= threshold, so
    // C returns -(max_iter+1) (stcopt_cg_impl.c:583).
    let no_iterations = op.solve(&[1.0, 0.0], 1.0, 0.0, 1e-6, 0);
    assert_eq!(no_iterations.status, -1);
    assert_eq!(no_iterations.solution, [0.0, 0.0]);
    // An SPD system that needs two iterations does not converge in one.
    let mut spd = SampledSrOperator::new(2, 2, false);
    spd.real_samples.copy_from_slice(&[2.0, 1.0, 0.0, 1.0]);
    let limited = spd.solve(&[6.0, 4.0], 1.0, 0.0, 1e-14, 1);
    assert_eq!((limited.status, limited.iterations), (-2, 1));
    // The same rules apply to the DiagScale variant.
    let diag = op.solve_variant(CgVariant::DiagScale, &[1.0, 0.0], 1.0, 0.0, 1e-6, 10);
    assert_eq!((diag.status, diag.iterations), (-1, 0));
}

#[test]
fn cg_rejects_tiny_search_products_like_c_1_4_0() {
    // One-dimensional SPD system S = (1e-16)^2, g = 1e-16 (x = 1e16). Its first
    // d.S.d = 1e-64 is below C 1.4.0's 1e-20 guard (stcopt_cg_impl.c:540), so
    // the solve stops with status -(0+1) and x = 0; 1.3.0 divided by it.
    let mut op = SampledSrOperator::new(1, 1, false);
    op.real_samples[0] = 1e-16;
    let result = op.solve(&[1e-16], 1.0, 0.0, 1e-30, 1);
    assert_eq!((result.status, result.iterations), (-1, 0));
    assert_eq!(result.solution, [0.0]);
}

// The optional C oracle restarts the verbatim upstream 1.4.0 Main/operator
// (plain and DiagScale) at every limit from 1 to 41. Historical Julia records remain in sr_cg/*.txt;
// they assign delta_new directly and are not C recurrence expectations.
// Compare each iterate's complete state, including both residual refreshes,
// with roundoff budgets and an independent explicit Gram residual check.
// Iteration counts and refresh limits remain exact.
#[test]
fn cg_fixed_input_matches_c_through_residual_refresh() {
    let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures");
    use mvmc_core::sr_cg::CgVariant;
    for (name, file, variant) in [
        ("real", "real.txt", CgVariant::Plain),
        ("complex", "complex.txt", CgVariant::Plain),
        ("sampled complex", "sampled_complex.txt", CgVariant::Plain),
        (
            "real DiagScale",
            "real_diag_scale.txt",
            CgVariant::DiagScale,
        ),
        (
            "complex DiagScale",
            "complex_diag_scale.txt",
            CgVariant::DiagScale,
        ),
        (
            "sampled complex DiagScale",
            "sampled_complex_diag_scale.txt",
            CgVariant::DiagScale,
        ),
    ] {
        let path = julia_fixture::fixture_path(&fixtures, format!("sr_cg/c_refresh/{file}"));
        // The oracle runs on Linux x86_64 OpenBLAS (Haswell lineage). Hosts without a
        // platform-specific C 1.4.0 copy (the 1.3.0 Apple Silicon copies were removed in
        // #682) keep exact statuses and the backward-error check at every limit but
        // compare forward iterates only through limit 5, like other unverified kernels.
        let fallback = cfg!(not(all(
            target_os = "linux",
            target_env = "gnu",
            target_arch = "x86_64"
        ))) && path == fixtures.join(format!("sr_cg/c_refresh/{file}"));
        let fixture = julia_fixture::read_text(path).unwrap();
        let mut lines = fixture.lines().filter(|line| !line.starts_with('#'));
        let shape: Vec<usize> = lines
            .next()
            .unwrap()
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        assert_eq!(shape.len(), 3, "{name}: shape must have three fields");
        let (n, samples, complex) = (shape[0], shape[1], shape[2] != 0);
        assert!(n > 0 && samples > 0, "{name}: nonempty fixed operands");
        assert!(shape[2] <= 1, "{name}: complex flag must be zero or one");
        let parse = |line: &str| -> Vec<f64> {
            line.split_whitespace()
                .map(|v| f64::from_bits(u64::from_str_radix(v, 16).unwrap()))
                .collect()
        };
        let mut op = SampledSrOperator::new(n, samples, complex);
        op.mean = parse(lines.next().unwrap());
        op.diagonal = parse(lines.next().unwrap());
        op.real_samples = parse(lines.next().unwrap());
        op.imag_samples = parse(lines.next().unwrap());
        let g = parse(lines.next().unwrap());
        assert_eq!(op.mean.len(), n, "{name}: mean width");
        assert_eq!(op.diagonal.len(), n, "{name}: diagonal width");
        assert_eq!(g.len(), n, "{name}: gradient width");
        assert_eq!(op.real_samples.len(), n * samples, "{name}: real samples");
        assert_eq!(
            op.imag_samples.len(),
            if complex { n * samples } else { 0 },
            "{name}: imaginary samples"
        );
        let expected = parse(lines.next().unwrap());
        assert_eq!(expected.len(), n);
        let mut z = vec![0.0; n];
        op.apply(&mut z, &g, 1.0 / samples as f64, 1e-5);
        for (i, (&a, &b)) in z.iter().zip(&expected).enumerate() {
            // Two GEMVs plus mean/diagonal corrections: O(n+samples) rounding.
            let budget = 8.0 * (n + samples) as f64 * f64::EPSILON;
            numerical_comparison::assert_close(
                a,
                b,
                budget,
                budget,
                format!("{name} operator component {i}"),
            );
        }
        // Materialize the covariance from the fixed samples independently of
        // the two-GEMV implementation. Row sums give the residual scale without
        // inventing a global forward tolerance for ill-conditioned components.
        let mut covariance = vec![0.0; n * n];
        for row in 0..n {
            for column in 0..n {
                let gram: f64 = (0..samples)
                    .map(|sample| {
                        let r = sample * n + row;
                        let c = sample * n + column;
                        op.real_samples[r] * op.real_samples[c]
                            + if complex {
                                op.imag_samples[r] * op.imag_samples[c]
                            } else {
                                0.0
                            }
                    })
                    .sum();
                covariance[row * n + column] = gram / samples as f64
                    - op.mean[row] * op.mean[column]
                    + if row == column {
                        1e-5 * op.diagonal[row]
                    } else {
                        0.0
                    };
            }
        }
        let mut checked_limits = 0;
        while let Some(line) = lines.next() {
            let mut words = line.splitn(3, ' ');
            let limit: usize = words.next().unwrap().parse().unwrap();
            checked_limits += 1;
            assert_eq!(limit, checked_limits);
            // C 1.4.0 return value: iterations, or -(iter+1) on breakdown or
            // non-convergence (tol = 0 never converges).
            let status: i64 = words.next().unwrap().parse().unwrap();
            assert!(
                status < 0 && -status - 1 <= limit as i64,
                "{name}: reference status {status} at limit {limit}"
            );
            let expected = parse(words.next().unwrap());
            let expected_residual = parse(lines.next().unwrap());
            let expected_direction = parse(lines.next().unwrap());
            assert_eq!(expected.len(), n, "{name} limit {limit}: solution width");
            assert_eq!(
                expected_residual.len(),
                n,
                "{name} limit {limit}: residual width"
            );
            assert_eq!(
                expected_direction.len(),
                n,
                "{name} limit {limit}: direction width"
            );
            let diagnostic = std::env::var_os("MVMC_CG_DIAGNOSTICS")
                .map(|_| std::rc::Rc::new(FixedCgDiagnostic::default()));
            let guard = diagnostic
                .as_ref()
                .map(|observer| mvmc_core::sr_cg::install_cg_observer(observer.clone()).unwrap());
            let result = op.solve_variant(variant, &g, 1.0 / samples as f64, 1e-5, 0.0, limit);
            drop(guard);
            if let Some(diagnostic) = diagnostic {
                let path = std::env::temp_dir().join(format!(
                    "mvmc-cg-fixed-{}-{}-{limit}.txt",
                    std::process::id(),
                    name.replace(' ', "_")
                ));
                let mut text = format!("n={n} samples={samples} complex={complex} limit={limit} shift=1e-5 tolerance=0 inv_weight={:?}\nmean={:?}\ndiagonal={:?}\nreal_samples={:?}\nimag_samples={:?}\ngradient={g:?}\nexpected_solution={expected:?}\nexpected_residual={expected_residual:?}\nexpected_direction={expected_direction:?}\ncovariance={covariance:?}\n",
                    1.0 / samples as f64, op.mean, op.diagonal, op.real_samples, op.imag_samples);
                text.push_str(&diagnostic.0.borrow());
                std::fs::write(&path, text).unwrap();
                eprintln!("ACTUAL fixed CG diagnostic: {}", path.display());
            }
            assert_eq!(
                result.solution.len(),
                n,
                "{name} limit {limit}: actual solution width"
            );
            assert_eq!(
                result.residual.len(),
                n,
                "{name} limit {limit}: actual residual width"
            );
            assert_eq!(
                result.direction.len(),
                n,
                "{name} limit {limit}: actual direction width"
            );
            assert_eq!(result.status, status, "{name} limit {limit}: C status");
            for row in 0..n {
                let terms = covariance[row * n..(row + 1) * n]
                    .iter()
                    .zip(&result.solution);
                let ax: f64 = terms.clone().map(|(a, x)| a * x).sum();
                let scale = g[row].abs() + terms.map(|(a, x)| (a * x).abs()).sum::<f64>();
                // Recurrent residuals incur iteration rounding between the exact
                // 20-step refreshes; bound it by the actual |A| |x| + |g| scale.
                let budget = 16.0 * (n + samples) as f64 * limit as f64 * f64::EPSILON;
                numerical_comparison::assert_close(
                    result.residual[row],
                    g[row] - ax,
                    budget * scale,
                    0.0,
                    format!("{name} limit {limit} explicit residual {row}"),
                );
            }
            // The reference trajectory is a chaotic recurrence (tol = 0 on a 32 x 48
            // operand, 41 iterations without convergence). Bitwise-compatible FMA
            // kernels (OpenBLAS Haswell, the checked-in ARM cores) reproduce it at every
            // limit. A kernel with different GEMV rounding differs by one ulp at
            // iteration 2 and grows about 1e4 per two iterations from iteration 5
            // (measured on Sandybridge, Nehalem, Prescott: max |dx|/|x| 2e-16 at limit 5,
            // 1e-15 at 6, 2e-14 at 7, O(1e-2) after 13), so only the first limits are a
            // meaningful forward comparison there. The explicit-residual (backward error)
            // check above holds at every limit on every kernel.
            // DiagScale divides by the clamped S diagonal (down to 1e-12), which
            // multiplies a one-ulp GEMV difference by up to 1/min(diag): on a host
            // that reads the Linux oracle (Apple: 3.5e-12 at limit 1 for complex
            // DiagScale) only the exact status and the backward error are portable.
            let forward_limit = match julia_fixture::kernel_class(&fixtures) {
                julia_fixture::KernelClass::Reference if !fallback => usize::MAX,
                _ if fallback && variant == CgVariant::DiagScale => 0,
                _ => 5,
            };
            if limit > forward_limit {
                continue;
            }
            for (stage, actual, expected) in [
                ("solution", &result.solution, &expected),
                ("residual", &result.residual, &expected_residual),
                ("direction", &result.direction, &expected_direction),
            ] {
                assert_eq!(actual.len(), expected.len());
                for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
                    // Budget grows with actual iteration count and short GEMV lengths.
                    // An independent backward-residual check below protects against
                    // a loose forward comparison on this scaled/conditioned system.
                    let budget = 8.0 * (n + samples) as f64 * limit as f64 * f64::EPSILON;
                    numerical_comparison::assert_close(
                        a,
                        b,
                        budget,
                        budget,
                        format!("{name} limit {limit} {stage} component {i}"),
                    );
                }
            }
        }
        assert_eq!(checked_limits, 41);
    }
}

/// One real Gutzwiller and one Slater parameter with three stored samples.
/// Projection O = (1, -1, 0), the given Slater O, energies (1, 2, 4).
fn two_parameter_case(
    slater: [f64; 3],
) -> (mvmc_core::ExpertModeData, mvmc_core::VmcOptimizationState) {
    use mvmc_core::{ExpertModeData, VmcOptimizationState};
    use mvmc_expert_parsers::{GutzwillerTerm, OrbitalTerm};
    use num_complex::Complex64 as C;
    let mut data = ExpertModeData::new();
    data.modpara.nsite = 1;
    data.modpara.nelec = 1;
    data.modpara.nvmc_sample = 3;
    data.modpara.n_orbital_idx = 1;
    data.modpara.nsrcg = 1;
    data.modpara.dsr_opt_red_cut = 0.0;
    data.modpara.dsr_opt_sta_del = 0.0;
    data.modpara.dsr_opt_step_dt = 0.25;
    data.modpara.dsr_opt_cg_tol = 1e-14;
    data.n_gutzwiller_idx = 1;
    data.gutzwiller_terms.push(GutzwillerTerm {
        site: 0,
        value: C::new(0.5, 0.0),
        is_complex: false,
    });
    data.slater_params = vec![C::new(10.0, 0.0)];
    data.orbital_terms.push(OrbitalTerm {
        site1: 0,
        site2: 0,
        idx: 0,
        is_complex: false,
        sign: 1,
    });
    let mut state = VmcOptimizationState::zeros(1, 1, 1, 2, 1, 3, false, false);
    let (proj, energy) = ([1.0, -1.0, 0.0], [1.0, 2.0, 4.0]);
    let mean = |f: &dyn Fn(usize) -> f64| (0..3).map(f).sum::<f64>() / 3.0;
    let sr = &mut state.sr_opt;
    // Real CG layout: O at [0, size), the OO diagonal at [size, 2 size), size = 3.
    sr.sr_opt_oo_real[0] = 1.0;
    sr.sr_opt_oo_real[1] = mean(&|s| proj[s]);
    sr.sr_opt_oo_real[2] = mean(&|s| slater[s]);
    sr.sr_opt_oo_real[3] = 1.0;
    sr.sr_opt_oo_real[4] = mean(&|s| proj[s] * proj[s]);
    sr.sr_opt_oo_real[5] = mean(&|s| slater[s] * slater[s]);
    sr.sr_opt_ho_real[0] = mean(&|s| energy[s]);
    sr.sr_opt_ho_real[1] = mean(&|s| energy[s] * proj[s]);
    sr.sr_opt_ho_real[2] = mean(&|s| energy[s] * slater[s]);
    for s in 0..3 {
        sr.sr_opt_o_store_real[3 * s] = 1.0;
        sr.sr_opt_o_store_real[3 * s + 1] = proj[s];
        sr.sr_opt_o_store_real[3 * s + 2] = slater[s];
    }
    state.energy.wc = C::new(3.0, 0.0);
    (data, state)
}

#[test]
fn rescale_smat_reparametrizes_the_slater_block_without_changing_the_update() {
    // C 1.4.0 Rescale4SRCG (stcopt_cg_impl.c:772-875) scales the Slater O by
    // s = sqrt(max|S_proj| / max|S_Slater|) and divides the Slater parameters
    // by s. Without the diagonal shift or cut the solved step is the same
    // wavefunction update: projection increments are unchanged and the
    // rescaled Slater parameter is the unscaled result divided by s.
    let (mut plain, state) = two_parameter_case([4.0, 0.0, 3.0]);
    let mut rescaled = plain.clone();
    rescaled.modpara.rescale_smat = true;
    assert_eq!(
        mvmc_core::sr_cg::stochastic_opt_cg(&mut plain, &state, None).unwrap(),
        0
    );
    assert_eq!(
        mvmc_core::sr_cg::stochastic_opt_cg(&mut rescaled, &state, None).unwrap(),
        0
    );
    // S_proj = 2/3, S_Slater = 25/3 - (7/3)^2 = 26/9.
    let s = ((2.0_f64 / 3.0) / (26.0 / 9.0)).sqrt();
    // Two-component CG in at most two iterations: roundoff of O(10) operations
    // on O(10) values; 1e-12 relative leaves a wide margin.
    numerical_comparison::assert_values_close(
        [rescaled.gutzwiller_terms[0].value.re],
        [plain.gutzwiller_terms[0].value.re],
        1e-12,
        1e-12,
        "projection parameter",
    );
    numerical_comparison::assert_values_close(
        [rescaled.slater_params[0].re * s],
        [plain.slater_params[0].re],
        1e-12,
        1e-12,
        "rescaled Slater parameter times s",
    );
    assert_ne!(rescaled.slater_params[0], plain.slater_params[0]);
}

#[test]
fn nsrcg_fallback_retries_with_diag_scale_and_reports_the_first_failure() {
    // Slater O = (1, 1, 4) is uncorrelated with the projection O, so S is
    // diagonal, diag(2/3, 2). Plain CG needs two iterations and fails at the
    // limit of one (status -(1+1)); DiagScale CG with M = diag(S) is exact in
    // one. C 1.4.0 then reports info = -(1+1) and applies the DiagScale step
    // (stcopt_cg_impl.c:208-251).
    let (mut data, state) = two_parameter_case([1.0, 1.0, 4.0]);
    data.modpara.nsr_opt_cg_max_iter = 1;
    data.modpara.nsrcg_fallback = true;
    let (proj, slater) = (data.gutzwiller_terms[0].value.re, data.slater_params[0].re);
    let dir = std::env::temp_dir().join(format!("mvmc-cg-fallback-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut files = mvmc_core::output_files::RunFiles::init(
        &data,
        &dir,
        mvmc_core::output_files::RunKind::ParaOpt,
        None,
    )
    .unwrap();
    assert_eq!(
        mvmc_core::sr_cg::stochastic_opt_cg(&mut data, &state, Some(&mut files)).unwrap(),
        0
    );
    files.close().unwrap();
    let info = std::fs::read_to_string(dir.join("zvo_SRinfo.dat")).unwrap();
    assert!(info.lines().nth(1).unwrap().ends_with(" -2"), "{info}");
    // x_k = g_k / S_kk with g = -2 dt (<HO> - <H><O>), dt = 0.25, <H> = 7/3.
    let g_proj = -0.5 * ((1.0 - 2.0) / 3.0 - (7.0 / 3.0) * 0.0);
    let g_slater = -0.5 * ((1.0 + 2.0 + 16.0) / 3.0 - (7.0 / 3.0) * 2.0);
    numerical_comparison::assert_values_close(
        [
            data.gutzwiller_terms[0].value.re - proj,
            data.slater_params[0].re - slater,
        ],
        [g_proj / (2.0 / 3.0), g_slater / 2.0],
        1e-14,
        1e-13,
        "DiagScale fallback increments",
    );
    std::fs::remove_dir_all(dir).unwrap();
    // Without the fallback the default NSRCGAbortOnFail aborts.
    let (mut data, state) = two_parameter_case([1.0, 1.0, 4.0]);
    data.modpara.nsr_opt_cg_max_iter = 1;
    let error = mvmc_core::sr_cg::stochastic_opt_cg(&mut data, &state, None).unwrap_err();
    assert_eq!(
        error.to_string(),
        "error: SR-CG fallback is disabled. Abort."
    );
    // DiagScale first: converged, no fallback needed.
    let (mut data, state) = two_parameter_case([1.0, 1.0, 4.0]);
    data.modpara.nsr_opt_cg_max_iter = 1;
    data.modpara.use_diag_scale = true;
    assert_eq!(
        mvmc_core::sr_cg::stochastic_opt_cg(&mut data, &state, None).unwrap(),
        0
    );
}

//! Sampled SR operator contracts from Julia test_unit_stochastic_opt.jl.
#[path = "../../../tests/support/julia_fixture.rs"]
mod julia_fixture;
#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;
use mvmc_core::sr_cg::{sequential_dot, SampledSrOperator};

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
    let row = "    1     1     0     0  2.00000e+00  2.00000e+00 -5.00000e-01     0, 1\n";
    for _ in 0..2 {
        assert_eq!(
            mvmc_core::sr_cg::stochastic_opt_cg(&mut data, &state, Some(&dir)).unwrap(),
            0
        );
    }
    assert_eq!(
        std::fs::read_to_string(dir.join("custom_SRinfo.dat")).unwrap(),
        format!("#Npara Msize optCut diagCut sDiagMax  sDiagMin    absRmax       imax\n{row}{row}")
    );
    data.optimization_flags = vec![0, 0];
    let before = data.slater_params.clone();
    assert_eq!(
        mvmc_core::sr_cg::stochastic_opt_cg(&mut data, &state, Some(&dir)).unwrap(),
        0
    );
    assert_eq!(data.slater_params, before);
    data.optimization_flags = vec![1, 0];
    state.sr_opt.sr_opt_ho_real[1] = f64::NAN;
    assert_eq!(
        mvmc_core::sr_cg::stochastic_opt_cg(&mut data, &state, None).unwrap(),
        1
    );
    assert_eq!(data.slater_params, before);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn cg_controls_default_to_julia_tolerance_and_active_dimension_limit() {
    use mvmc_expert_parsers::parsers::modpara::parse_modpara_content;
    for input in ["", "DSROptCGTol bad\nNSROptCGMaxIter bad"] {
        let p = parse_modpara_content(input);
        assert_eq!(p.dsr_opt_cg_tol, 1e-10);
        assert_eq!(p.nsr_opt_cg_max_iter, 0);
    }
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
fn cg_preserves_c_zero_gradient_and_ieee_breakdown_iteration_counts() {
    let mut op = SampledSrOperator::new(2, 2, false);
    let zero = op.solve(&[0.0, 0.0], 1.0, 0.0, 1e-6, 10);
    assert_eq!(zero.iterations, 0);
    assert_eq!(zero.solution, [0.0, 0.0]);
    let breakdown = op.solve(&[1.0, 0.0], 1.0, 0.0, 1e-6, 10);
    // stcopt_cg_impl.c:306-310 has only the residual-norm stop test,
    // not Julia's abs(dq) < 1e-30 extension. With a zero operator and a
    // nonzero gradient, alpha is infinite and the subsequent IEEE NaNs
    // do not satisfy delta < threshold; C exhausts the iteration limit.
    assert_eq!(breakdown.iterations, 10);
    assert!(breakdown.solution.iter().all(|value| value.is_nan()));
    assert!(breakdown.residual.iter().all(|value| value.is_nan()));
    let no_iterations = op.solve(&[1.0, 0.0], 1.0, 0.0, 1e-6, 0);
    assert_eq!(no_iterations.iterations, 0);
    assert_eq!(no_iterations.solution, [0.0, 0.0]);
}

#[test]
fn cg_solves_small_positive_operator_without_julia_denominator_guard() {
    // Independent one-dimensional SPD system: S = (1e-16)^2,
    // g = 1e-16, hence x = g/S = 1e16. The first d*S*d = 1e-64
    // is nonzero and C:310 divides by it without a magnitude cutoff.
    let mut op = SampledSrOperator::new(1, 1, false);
    op.real_samples[0] = 1e-16;
    let result = op.solve(&[1e-16], 1.0, 0.0, 1e-30, 1);
    assert_eq!(result.iterations, 1);
    // Four binary64 operations in the scalar Gram/alpha/update path give a
    // conservative two-ULP forward allowance at 1e16 (ULP=2); residual
    // cancellation is bounded by roughly 2*epsilon*|g| = 4.45e-32.
    assert!((result.solution[0] - 1e16).abs() <= 4.0);
    assert!(result.residual[0].abs() <= 5e-32);
}

// The optional C oracle restarts the verbatim upstream Main/operator at
// every limit from 1 to 41. Historical Julia records remain in sr_cg/*.txt;
// they assign delta_new directly and are not C recurrence expectations.
// Compare each iterate's complete state, including both residual refreshes,
// with roundoff budgets and an independent explicit Gram residual check.
// Iteration counts and refresh limits remain exact.
#[test]
fn cg_fixed_input_matches_c_through_residual_refresh() {
    let fixtures = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures");
    for (name, file) in [
        ("real", "real.txt"),
        ("complex", "complex.txt"),
        ("sampled complex", "sampled_complex.txt"),
    ] {
        let fixture = julia_fixture::read_text(julia_fixture::fixture_path(
            &fixtures,
            format!("sr_cg/c_refresh/{file}"),
        ))
        .unwrap();
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
            let iter: usize = words.next().unwrap().parse().unwrap();
            assert!(
                iter <= limit,
                "{name}: reference iterations exceed limit {limit}"
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
            let result = op.solve(&g, 1.0 / samples as f64, 1e-5, 0.0, limit);
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
            assert_eq!(result.iterations, iter);
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

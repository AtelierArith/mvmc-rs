//! Offline C-window fixture/aggregation audit, not model runner execution.
//! Expectations are standalone native C outputs; no oracle runtime is used.
//! Archived 50-step cases are historical aggregation evidence, not execution
//! of the current 20-step long-run acceptance baseline.
use num_complex::Complex64;
use std::fs;
use std::path::Path;

struct OutputDirectory(std::path::PathBuf);
impl OutputDirectory {
    fn create() -> Self {
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "mvmc-ctest-window-fixture-{}-{id}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for OutputDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;

#[test]
fn standalone_c_window_fixtures_preserve_effective_window_and_aggregation() {
    let root =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/ctest_model_prefixes");
    let mut cases = 0;
    let mut maximum_error = 0.0_f64;
    let output = OutputDirectory::create();
    for model in fs::read_dir(&root).unwrap() {
        let model = model.unwrap();
        if !model.file_type().unwrap().is_dir() {
            continue;
        }
        for steps in [1, 2, 3, 50] {
            let case = model.path().join(format!("step-{steps}"));
            let settings = fs::read_to_string(case.join("model-settings.txt")).unwrap();
            assert!(settings.contains(&format!("effective_NSROptItrStep={steps}")));
            assert!(settings.contains(&format!("effective_NSROptItrSmp={steps}")));
            assert!(settings.contains("override=both_no_clamp"));
            let provenance = fs::read_to_string(case.join("c-window-provenance.txt")).unwrap();
            assert!(provenance.contains("actual C avevar.c bodies; no Rust results"));
            let text = fs::read_to_string(case.join("c-window-declared-input.txt")).unwrap();
            let mut lines = text.lines();
            let header: Vec<usize> = lines
                .next()
                .unwrap()
                .split_whitespace()
                .map(|word| word.parse().unwrap())
                .collect();
            assert_eq!(header.len(), 17);
            assert_eq!(header[0], steps);
            let npara = header[1];
            let total: usize = header[2..]
                .iter()
                .enumerate()
                .map(|(index, count)| {
                    count
                        * match index {
                            2 => 6,
                            3 => 10,
                            _ => 1,
                        }
                })
                .sum();
            assert_eq!(total, npara);
            let rows: Vec<Vec<Complex64>> = lines
                .map(|line| {
                    let values: Vec<f64> = line
                        .split_whitespace()
                        .map(|word| word.parse().unwrap())
                        .collect();
                    assert_eq!(values.len(), 2 * (npara + 2));
                    assert!(values.iter().all(|value| value.is_finite()));
                    let (pairs, remainder) = values.as_chunks::<2>();
                    assert!(remainder.is_empty());
                    pairs
                        .iter()
                        .map(|pair| Complex64::new(pair[0], pair[1]))
                        .collect()
                })
                .collect();
            assert_eq!(rows.len(), steps);
            // Supply only declared output layout and independently recorded
            // history to the production writer. No initialization, sampler,
            // oracle runtime, or Rust-generated expectation is involved.
            let mut data = mvmc_expert_parsers::ExpertModeData::new();
            data.n_gutzwiller_idx = header[2] as i64;
            data.n_jastrow_idx = header[3] as i64;
            assert_eq!(
                (header[4], header[5]),
                (0, 0),
                "ctest models have no DH groups"
            );
            data.rbm_section_widths.copy_from_slice(&header[6..15]);
            data.modpara.n_orbital_idx = header[15] as i64;
            data.slater_params = vec![Complex64::new(0.0, 0.0); header[15]];
            data.opt_trans = vec![Complex64::new(0.0, 0.0); header[16]];
            data.modpara.nsr_opt_itr_step = steps as i64;
            data.modpara.nsr_opt_itr_smp = steps as i64;
            // Flag selection/mapped-term absence must not truncate the
            // all-Para output stream, including reserved declared slots.
            data.optimization_flags = (0..2 * npara)
                .map(|index| i64::from(index % 3 == 0))
                .collect();
            assert_eq!(data.count_variational_parameters(), npara);
            let mut state =
                mvmc_core::VmcOptimizationState::zeros(1, 1, 0, npara, 1, 0, true, false);
            state.opt_data = rows
                .iter()
                .map(|row| mvmc_core::OptDataPoint {
                    energy: row[0],
                    energy_squared: row[1],
                    parameters: row[2..].to_vec(),
                })
                .collect();
            let case_output = output
                .0
                .join(model.file_name())
                .join(format!("step-{steps}"));
            fs::create_dir_all(&case_output).unwrap();
            mvmc_core::io::output_opt_data(&data, &state, Some(&case_output)).unwrap();
            let actual_text = fs::read_to_string(case_output.join("zqp_opt.dat")).unwrap();
            let expected_text = fs::read_to_string(case.join("zqp_c_window_opt.dat")).unwrap();
            numerical_comparison::assert_numeric_text(
                &actual_text,
                &expected_text,
                1e-11,
                1e-11,
                &[],
                format!(
                    "C-window production writer {} step={steps}",
                    model.file_name().to_string_lossy()
                ),
            );
            let actual_fields: Vec<_> = actual_text.split_whitespace().collect();
            if steps == 1 {
                for (actual, expected) in actual_fields
                    .iter()
                    .skip(1)
                    .step_by(2)
                    .zip(expected_text.split_whitespace().skip(1).step_by(2))
                {
                    // Exact literal formatting contract, not a computed-float
                    // bitwise assertion or a tolerance around zero.
                    assert_eq!(*actual, expected, "one-window literal imaginary zero");
                }
            } else {
                for deviation in actual_fields.iter().skip(2).step_by(3) {
                    let deviation: f64 = deviation.parse().unwrap();
                    assert!(
                        deviation.is_finite() && deviation >= 0.0,
                        "finite fixture requires nonnegative finite C sample deviation"
                    );
                }
            }
            let expected: Vec<f64> = fs::read_to_string(case.join("zqp_c_window_opt.dat"))
                .unwrap()
                .split_whitespace()
                .map(|word| word.parse().unwrap())
                .collect();
            assert!(expected.iter().all(|value| value.is_finite()));
            let mut actual = Vec::new();
            for field in 0..npara + 2 {
                if steps == 1 {
                    // Actual C special branch: no CalcAveVar, no deviation
                    // column, and a literal zero imaginary output field.
                    actual.extend([rows[0][field].re, 0.0]);
                } else {
                    let mut mean = Complex64::new(0.0, 0.0);
                    for row in &rows {
                        mean += row[field];
                    }
                    mean /= steps as f64;
                    let mut squared = 0.0;
                    for row in &rows {
                        let delta = row[field] - mean;
                        squared += (delta * delta.conj()).re;
                    }
                    assert!(squared.is_finite() && squared >= 0.0);
                    actual.extend([mean.re, mean.im, (squared / (steps - 1) as f64).sqrt()]);
                }
            }
            assert_eq!(actual.len(), expected.len());
            for (index, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
                maximum_error = maximum_error.max((actual - expected).abs());
                numerical_comparison::assert_close(
                    actual,
                    expected,
                    1e-11,
                    1e-11,
                    format!(
                        "offline C-window aggregation {} step={steps} field={index}",
                        model.file_name().to_string_lossy()
                    ),
                );
            }
            cases += 1;
        }
    }
    assert_eq!(cases, 56);
    eprintln!("Offline C-window fixture audit: {cases} cases, maximum absolute arithmetic error={maximum_error:.5e}; not model execution coverage");
}

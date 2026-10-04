//! Same-input callback non-interference, complementary to independent references.
#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;

use mvmc_core::{ExpertModeData, PhysCalPreparation};
use num_complex::Complex64;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Output(PathBuf);

impl Output {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        loop {
            let path = std::env::temp_dir().join(format!(
                "physcal-callback-exact175-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("exclusive temporary output: {error}"),
            }
        }
    }
}

impl Drop for Output {
    fn drop(&mut self) {
        // This object exclusively created this exact directory.
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn preparation() -> PhysCalPreparation {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/physcal_181/two-samples/heisenberg_chain_real");
    let mut prepared = mvmc_core::prepare_phys_cal_from_namelist(
        root.join("inputs/namelist.def"),
        root.join("zqp_opt.dat"),
        "real",
        Some(1),
    )
    .unwrap();
    // Bounded same-input observation test, not a regenerated trajectory oracle.
    // Preserve the parsed interval, projection and index settings on both sides.
    prepared.data.modpara.nvmc_sample = 3;
    prepared.data.modpara.nvmc_warmup = 1;
    prepared
}

fn bits(values: impl IntoIterator<Item = Complex64>) -> Vec<(u64, u64)> {
    values
        .into_iter()
        .map(|z| (z.re.to_bits(), z.im.to_bits()))
        .collect()
}

fn fixed(data: &ExpertModeData) -> Vec<(u64, u64)> {
    bits(
        data.projection_parameters()
            .into_iter()
            .chain(data.rbm_parameters())
            .chain(data.slater_params.iter().copied())
            .chain(data.opt_trans.iter().copied()),
    )
}

fn weights(data: &ExpertModeData) -> Vec<Vec<(u64, u64)>> {
    let q = data.qp_weights.as_ref().unwrap();
    [
        &q.qp_full_weight,
        &q.qp_fix_weight,
        &q.spgl_cos,
        &q.spgl_sin,
        &q.spgl_cos_sin,
        &q.spgl_cos_cos,
        &q.spgl_sin_sin,
    ]
    .into_iter()
    .map(|buffer| bits(buffer.iter().copied()))
    .collect()
}

fn assert_weight_values(actual: &ExpertModeData, expected: &ExpertModeData) {
    let actual = weights(actual);
    let expected = weights(expected);
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(&expected) {
        assert_eq!(actual.len(), expected.len());
        // Existing QP coefficient/quadrature budget from physcal_callback.rs.
        // Separate preparations compute these buffers independently.
        numerical_comparison::assert_values_close(
            actual
                .iter()
                .flat_map(|&(re, im)| [f64::from_bits(re), f64::from_bits(im)]),
            expected
                .iter()
                .flat_map(|&(re, im)| [f64::from_bits(re), f64::from_bits(im)]),
            32.0 * f64::EPSILON,
            32.0 * f64::EPSILON,
            "cross-run initialized QP values",
        );
    }
}

fn inventory(path: &Path) -> Vec<String> {
    let mut files = fs::read_dir(path)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .collect::<Vec<_>>();
    files.sort();
    files
}

#[test]
fn serial_callback_preserves_raw_rng_fixed_bits_and_all_weight_buffers() {
    let baseline_output = Output::new();
    let observed_output = Output::new();
    let baseline = mvmc_core::vmc_phys_cal_to_dir(preparation(), &baseline_output.0).unwrap();
    let prepared = preparation();
    assert_eq!(prepared.data.modpara.n_data_qty_smp, 2);
    assert_eq!(prepared.data.modpara.n_data_idx_start, 7);
    assert_eq!(prepared.data.i_flg_orbital_general, 0);
    assert_eq!(prepared.data.modpara.lanczos_mode, 0);
    assert_eq!(prepared.data.modpara.nsp_gauss_leg, 8);
    assert_eq!(prepared.data.modpara.nmp_trans, -1);
    let original_fixed = fixed(&prepared.data);
    let original_flags = prepared.data.optimization_flags.clone();
    let original_complex_flags = prepared.data.complex_flags.clone();
    let mut calls = Vec::new();
    let mut own_initialized_weights = None;
    let mut callback = |sample, data: &ExpertModeData, energy: Complex64, status| {
        assert_eq!(sample, calls.len());
        assert_eq!(status, 0);
        assert!(energy.re.is_finite() && energy.im.is_finite());
        assert_eq!(fixed(data), original_fixed);
        assert_eq!(data.optimization_flags, original_flags);
        assert_eq!(data.complex_flags, original_complex_flags);
        assert_weight_values(data, &baseline.data);
        // Capture this execution's initialized buffers at its first callback.
        // Only subsequent reads in this same execution are bitwise copy checks.
        if let Some(saved) = &own_initialized_weights {
            assert_eq!(&weights(data), saved);
        } else {
            own_initialized_weights = Some(weights(data));
        }
        let index = sample + 7;
        assert!(observed_output
            .0
            .join(format!("zvo_out_{index:03}.dat"))
            .is_file());
        assert!(!observed_output
            .0
            .join(format!("zvo_out_{:03}.dat", index + 1))
            .exists());
        calls.push(sample);
        Ok(())
    };
    let observed =
        mvmc_core::vmc_phys_cal_to_dir_with_callback(prepared, &observed_output.0, &mut callback)
            .unwrap();
    assert_eq!(calls, [0, 1]);
    assert_eq!(observed.iterations, baseline.iterations);
    assert_eq!(fixed(&observed.data), original_fixed);
    assert_weight_values(&observed.data, &baseline.data);
    assert_eq!(Some(weights(&observed.data)), own_initialized_weights);
    assert_eq!(
        observed.state.electron_config,
        baseline.state.electron_config
    );
    assert_eq!(
        observed.final_rng.words_consumed(),
        baseline.final_rng.words_consumed()
    );
    assert_eq!(
        observed.final_rng.state_snapshot(),
        baseline.final_rng.state_snapshot()
    );
    let mut actual = observed.final_rng.clone();
    let mut expected = baseline.final_rng.clone();
    for word in 0..624 {
        assert_eq!(actual.gen_rand32(), expected.gen_rand32(), "word {word}");
    }
    let names = inventory(&baseline_output.0);
    assert_eq!(inventory(&observed_output.0), names);
    // This untimed public PhysCal entry writes only the indexed measurement
    // files. No timer output is compared as a deterministic byte contract.
    let mut expected_names = [7, 8]
        .into_iter()
        .flat_map(|index| {
            ["out", "var", "cisajs", "cisajscktalt", "cisajscktaltex"]
                .map(|stem| format!("zvo_{stem}_{index:03}.dat"))
        })
        .collect::<Vec<_>>();
    expected_names.sort();
    assert_eq!(names, expected_names);
    for name in names {
        let indices: &[usize] = if name.starts_with("zvo_cisajscktalt_") {
            &[0, 1, 2, 3, 4, 5, 6, 7]
        } else if name.starts_with("zvo_cisajs_") {
            &[0, 1, 2, 3]
        } else {
            &[]
        };
        // Inherited same-implementation output budget; no independent oracle
        // or cross-backend numerical bound is changed by this observation test.
        numerical_comparison::assert_numeric_text(
            &fs::read_to_string(observed_output.0.join(&name)).unwrap(),
            &fs::read_to_string(baseline_output.0.join(&name)).unwrap(),
            1e-12,
            1e-12,
            indices,
            &name,
        );
    }
}

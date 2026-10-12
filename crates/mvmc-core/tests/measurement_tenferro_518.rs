//! Issue #518 milestone 1: the tenferro measurement stage (local energy + Slater
//! derivative of one real, non-FSZ sample) against the C-order oracle.
//!
//! (a) On a representative small real non-FSZ Hubbard sample, the tenferro CPU stage
//!     agrees with `COrderMeasurement` within the documented reordering bound.
//! (b) Inputs outside the tensor formulation (complex, FSZ, two-body Green terms) are
//!     `StageError::Unsupported`, never a silent CPU result.
//! (c) A short optimization with `MVMC_RS_MEASURE_BACKEND=tenferro` matches the C-order
//!     run within the propagated bound, so the selection path is exercised end to end.
//! (d) Selecting the tenferro measurement stage in a complex/FSZ run is a hard error, never
//!     a silent CPU fallback (the runner guard).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use mvmc_core::measurement_batch::{set_measurement_consumer_override, MeasurementConsumer};
use mvmc_core::observables::calculate_ip_real;
use mvmc_core::pfaffian::calc_m_all_real;
use mvmc_core::stage_backend::{
    COrderMeasurement, MeasurementInput, MeasurementStages, StageBackendKind, StageError,
};
use mvmc_core::TenferroMeasurement;
use mvmc_core::{ExpertModeData, SlaterMatrixData, ThreadedPfaPackWorkspace, VmcOptimizationState};
use mvmc_expert_parsers::utils::qp_weight::init_qp_weight;
use mvmc_expert_parsers::PairHopTerm;
use num_complex::Complex64;

#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;

/// The selection override is process-wide; serialize the tests that set it (nextest runs
/// each test in its own process, but `cargo test` shares threads).
static SERIAL: Mutex<()> = Mutex::new(());

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A genuine measurement sample of the L16 Hubbard chain: opaque quantum-projection
/// weights from the input, a random skew-symmetric Slater matrix, and the Pfaffian and
/// inverse tables of a chosen half-filled configuration built by the production
/// `calc_m_all_real`.
struct Sample {
    data: ExpertModeData,
    state: VmcOptimizationState,
    ip: Complex64,
    ele_idx: Vec<i64>,
    ele_cfg: Vec<i64>,
    ele_num: Vec<i64>,
    n_slater: usize,
}

fn hubbard_sample() -> Sample {
    let namelist =
        repo_root().join("benchmark/hubbard_chain/inputs/hubbard_chain_L16/namelist.def");
    let mut data = mvmc_expert_parsers::parse_expert_mode_files(namelist).unwrap();
    init_qp_weight(&mut data);
    assert_eq!(data.i_flg_orbital_general, 0);
    let n_site = data.modpara.nsite as usize;
    let n_elec = data.modpara.nelec as usize;
    let n_size = 2 * n_elec;
    let n_2site = 2 * n_site;
    let n_sp = data.modpara.nsp_gauss_leg.max(1) as usize;
    let n_mp = data.modpara.nmp_trans.unsigned_abs() as usize;
    let n_opt = data.n_qp_opt_trans.max(1) as usize;
    let n_qp_full = n_sp * n_mp * n_opt;
    let n_slater = mvmc_expert_parsers::utils::parameter_init::n_slater(&data);

    // Random skew-symmetric Slater planes (values are irrelevant for kernel equivalence;
    // the tables are made consistent with `ele_idx` by `calc_m_all_real`).
    let mut matrix = SlaterMatrixData::zeros(n_qp_full, n_site, n_elec, false);
    let mut seed = 20_260_518_u64;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    };
    for qp in 0..n_qp_full {
        let plane = matrix.slater_elm_real.qp_slice_mut(qp);
        for i in 0..n_2site {
            for j in 0..i {
                let value = next();
                plane[i * n_2site + j] = value;
                plane[j * n_2site + i] = -value;
            }
        }
    }
    // Half filling: even sites for spin up, odd sites for spin down.
    let ele_idx: Vec<i64> = (0..n_size)
        .map(|slot| {
            let site = 2 * (slot % n_elec) + usize::from(slot >= n_elec);
            site as i64
        })
        .collect();
    let pool = ThreadedPfaPackWorkspace::new(n_size, 1);
    calc_m_all_real(
        &ele_idx,
        &matrix.slater_elm_real,
        &mut matrix.inv_m_real,
        &mut matrix.pf_m_real,
        0,
        n_qp_full,
        n_site,
        n_elec,
        &pool,
    )
    .unwrap();
    let ip = Complex64::new(
        calculate_ip_real(&matrix.pf_m_real, 0, n_qp_full, &data),
        0.0,
    );
    assert!(ip.re != 0.0, "the test sample must have a nonzero overlap");

    let mut ele_num = vec![0_i64; n_2site];
    let mut ele_cfg = vec![-1_i64; n_2site];
    for (slot, &site) in ele_idx.iter().enumerate() {
        let site_slot = site as usize + usize::from(slot >= n_elec) * n_site;
        ele_num[site_slot] = 1;
        ele_cfg[site_slot] = (slot % n_elec) as i64;
    }

    let n_proj = data.projection_layout().n_proj;
    let n_para = data.count_variational_parameters();
    let mut state =
        VmcOptimizationState::zeros(n_site, n_elec, n_proj, n_para, n_qp_full, 1, false, false);
    state.slater_matrix = matrix;
    Sample {
        data,
        state,
        ip,
        ele_idx,
        ele_cfg,
        ele_num,
        n_slater,
    }
}

fn measure_both(
    sample: &mut Sample,
) -> (
    mvmc_core::stage_backend::MeasurementOutput,
    mvmc_core::stage_backend::MeasurementOutput,
) {
    let slater_len = 2 * sample.n_slater;
    let mut timing = mvmc_core::c_timer::CTimer::<false>::new();
    let mut c_order = COrderMeasurement::default();
    let c_out = {
        let mut input = MeasurementInput {
            data: &sample.data,
            state: &mut sample.state,
            ip: sample.ip,
            ele_idx: &sample.ele_idx,
            ele_cfg: &sample.ele_cfg,
            ele_num: &sample.ele_num,
            ele_proj_cnt: &[],
            sample: 0,
            batch_slot: 0,
        };
        c_order
            .measure_sample(&mut input, slater_len, &mut timing)
            .expect("the C-order oracle must run the sample")
    };
    let mut tenferro = TenferroMeasurement::new_cpu().unwrap();
    let t_out = {
        let mut input = MeasurementInput {
            data: &sample.data,
            state: &mut sample.state,
            ip: sample.ip,
            ele_idx: &sample.ele_idx,
            ele_cfg: &sample.ele_cfg,
            ele_num: &sample.ele_num,
            ele_proj_cnt: &[],
            sample: 0,
            batch_slot: 0,
        };
        tenferro
            .measure_sample(&mut input, slater_len, &mut timing)
            .expect("the tenferro stage must run the sample")
    };
    (c_out, t_out)
}

/// Worst componentwise deviation: absolute, and normalized by the compared magnitude.
fn worst_complex(actual: &[Complex64], expected: &[Complex64]) -> (f64, f64, usize) {
    let mut worst_abs = 0.0_f64;
    let mut worst_rel = 0.0_f64;
    let mut at = 0;
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        for (x, y) in [(a.re, e.re), (a.im, e.im)] {
            let scale = x.abs().max(y.abs()).max(1e-300);
            let difference = (x - y).abs();
            if difference > worst_abs {
                worst_abs = difference;
                at = i;
            }
            worst_rel = worst_rel.max(difference / scale);
        }
    }
    (worst_abs, worst_rel, at)
}

#[test]
fn tenferro_measurement_matches_c_order_on_a_hubbard_sample() {
    let mut sample = hubbard_sample();
    let (c_out, t_out) = measure_both(&mut sample);
    // Energy: the transfer section re-associates only the electron-slot dot product of
    // each `new_ip` (n = 2*Ne entries) and the QP fold; the diagonal and the
    // term-ordered accumulation are identical host operations.
    let energy_scale = c_out.local_energy.re.abs().max(t_out.local_energy.re.abs());
    let energy_deviation =
        (c_out.local_energy.re - t_out.local_energy.re).abs() / energy_scale.max(1e-300);
    println!(
        "energy: c-order {:.17e}, tenferro {:.17e}, relative deviation {:.2e}",
        c_out.local_energy.re, t_out.local_energy.re, energy_deviation
    );
    numerical_comparison::assert_close(
        t_out.local_energy.re,
        c_out.local_energy.re,
        1e-13,
        1e-13,
        "tenferro local energy",
    );
    numerical_comparison::assert_close(
        t_out.local_energy.im,
        c_out.local_energy.im,
        0.0,
        0.0,
        "tenferro local energy imag",
    );

    // Slater derivative: the per-pair QP fold is re-associated into per-sector QP
    // reductions (at most n_qp = 8 weighted plane entries per reduction for this input),
    // so the deviation is bounded by the reordering error of two sums of at most
    // `n_qp * n_size^2` separately rounded products (see the module docs).
    assert_eq!(c_out.slater_diff.len(), t_out.slater_diff.len());
    let (worst_abs, worst_rel, at) = worst_complex(&t_out.slater_diff, &c_out.slater_diff);
    println!(
        "slater derivative: {} components, worst absolute deviation {:.3e} (scale-relative {:.2e}) at {at}",
        t_out.slater_diff.len(),
        worst_abs,
        worst_rel
    );
    numerical_comparison::assert_values_close(
        t_out.slater_diff.iter().flat_map(|z| [z.re, z.im]),
        c_out.slater_diff.iter().flat_map(|z| [z.re, z.im]),
        1e-13,
        1e-13,
        "tenferro Slater derivative",
    );
}

#[test]
fn tenferro_measurement_reports_unsupported_modes() {
    let mut sample = hubbard_sample();
    let slater_len = 2 * sample.n_slater;
    let mut timing = mvmc_core::c_timer::CTimer::<false>::new();
    let mut stage = TenferroMeasurement::new_cpu().unwrap();

    // FSZ is outside the real, non-FSZ contract.
    let mut fsz_data = sample.data.clone();
    fsz_data.i_flg_orbital_general = 1;
    let err = {
        let mut input = MeasurementInput {
            data: &fsz_data,
            state: &mut sample.state,
            ip: sample.ip,
            ele_idx: &sample.ele_idx,
            ele_cfg: &sample.ele_cfg,
            ele_num: &sample.ele_num,
            ele_proj_cnt: &[],
            sample: 0,
            batch_slot: 0,
        };
        stage
            .measure_sample(&mut input, slater_len, &mut timing)
            .unwrap_err()
    };
    assert!(
        matches!(err, StageError::Unsupported(ref r) if r.contains("FSZ")),
        "{err}"
    );

    // Complex mode: a state allocated for the all-complex layout has no real tables.
    let err = {
        let mut complex_state = VmcOptimizationState::zeros(
            sample.data.modpara.nsite as usize,
            sample.data.modpara.nelec as usize,
            0,
            0,
            sample.state.slater_matrix.pf_m.len(),
            1,
            true,
            false,
        );
        let mut input = MeasurementInput {
            data: &sample.data,
            state: &mut complex_state,
            ip: sample.ip,
            ele_idx: &sample.ele_idx,
            ele_cfg: &sample.ele_cfg,
            ele_num: &sample.ele_num,
            ele_proj_cnt: &[],
            sample: 0,
            batch_slot: 0,
        };
        stage
            .measure_sample(&mut input, slater_len, &mut timing)
            .unwrap_err()
    };
    assert!(
        matches!(err, StageError::Unsupported(ref r) if r.contains("complex")),
        "{err}"
    );

    // Two-body Green terms.
    let mut two_body_data = sample.data.clone();
    two_body_data.pair_hop_terms.push(PairHopTerm {
        site1: 0,
        site2: 1,
        value: 0.5,
    });
    let err = {
        let mut input = MeasurementInput {
            data: &two_body_data,
            state: &mut sample.state,
            ip: sample.ip,
            ele_idx: &sample.ele_idx,
            ele_cfg: &sample.ele_cfg,
            ele_num: &sample.ele_num,
            ele_proj_cnt: &[],
            sample: 0,
            batch_slot: 0,
        };
        stage
            .measure_sample(&mut input, slater_len, &mut timing)
            .unwrap_err()
    };
    assert!(
        matches!(err, StageError::Unsupported(ref r) if r.contains("two-body")),
        "{err}"
    );
}

// ---- end-to-end selection path ---------------------------------------------------------

fn copy_inputs(fixture: &str, tag: &str) -> (PathBuf, PathBuf) {
    let source = repo_root().join("tests/fixtures/physcal_181").join(fixture);
    let root = std::env::temp_dir().join(format!(
        "mvmc-issue518-{}-{fixture}-{tag}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    let inputs = root.join("inputs");
    std::fs::create_dir_all(&inputs).unwrap();
    for entry in std::fs::read_dir(source.join("inputs")).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), inputs.join(entry.file_name())).unwrap();
    }
    (root, inputs)
}

fn rewrite_modpara(inputs: &Path, overrides: &[(&str, &str)]) {
    let modpara = std::fs::read_to_string(inputs.join("modpara.def")).unwrap();
    let mut out = String::new();
    let mut seen = Vec::new();
    for line in modpara.lines() {
        let key = line.split_whitespace().next().unwrap_or("");
        match overrides.iter().find(|(k, _)| k.eq_ignore_ascii_case(key)) {
            Some((k, v)) => {
                seen.push(*k);
                out.push_str(&format!("{k} {v}\n"));
            }
            None => {
                out.push_str(line);
                out.push('\n');
            }
        }
    }
    for (k, v) in overrides {
        if !seen.contains(k) {
            out.push_str(&format!("{k} {v}\n"));
        }
    }
    std::fs::write(inputs.join("modpara.def"), out).unwrap();
}

fn optimization(consumer: Option<MeasurementConsumer>, tag: &str) -> BTreeMap<String, String> {
    set_measurement_consumer_override(consumer);
    let (root, inputs) = copy_inputs("hubbard_chain_real", tag);
    rewrite_modpara(
        &inputs,
        &[
            ("NVMCCalMode", "0"),
            ("NSROptItrStep", "2"),
            ("NSROptItrSmp", "2"),
            ("NVMCSample", "20"),
            ("NVMCWarmUp", "5"),
            ("NSRCG", "0"),
            ("NStore", "1"),
            ("NDataIdxStart", "1"),
            ("NDataQtySmp", "1"),
            ("NLanczosMode", "0"),
        ],
    );
    let output = root.join("output");
    let mut config = mvmc_core::RunConfig::new(2, "real");
    config.output_dir = Some(output.clone());
    config.seed = Some(1);
    config.enable_opt_trans = Some(false);
    mvmc_core::run_para_opt_from_namelist(inputs.join("namelist.def"), config)
        .unwrap_or_else(|e| panic!("{tag}: {e}"));
    let mut files = BTreeMap::new();
    for entry in std::fs::read_dir(&output).unwrap() {
        let name = entry.unwrap().file_name().to_string_lossy().into_owned();
        if name.contains("Timer") || name.starts_with("zvo_time") {
            continue;
        }
        files.insert(
            name.clone(),
            std::fs::read_to_string(output.join(&name)).unwrap(),
        );
    }
    let _ = std::fs::remove_dir_all(&root);
    files
}

/// The selected tenferro measurement stage runs in production and every numeric output
/// token matches the C-order run within the propagated bound (the stage changes only the
/// last bits of each sample's energy/derivative, then two SR steps amplify them). Bound
/// `abs 1e-12 / rel 1e-11` against a measured worst 1.49e-13 absolute and 2.60e-13
/// relative; any layout or sign defect would show up as an O(1) deviation instead.
#[test]
fn tenferro_measurement_selection_matches_c_order_optimization() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let reference = optimization(
        Some(MeasurementConsumer::Stage(StageBackendKind::COrder)),
        "ref",
    );
    let staged = optimization(
        Some(MeasurementConsumer::Stage(StageBackendKind::TenferroCpu)),
        "tenferro",
    );
    set_measurement_consumer_override(None);
    assert_eq!(
        reference.keys().collect::<Vec<_>>(),
        staged.keys().collect::<Vec<_>>(),
        "file set"
    );
    let mut worst_abs = 0.0_f64;
    let mut worst_rel_large = 0.0_f64;
    for (name, expected) in &reference {
        for (row, (a, e)) in staged[name].lines().zip(expected.lines()).enumerate() {
            if e.trim_start().starts_with('#') {
                continue;
            }
            let a: Vec<&str> = a.split_whitespace().collect();
            let e: Vec<&str> = e.split_whitespace().collect();
            assert_eq!(a.len(), e.len(), "{name}: row {row} column count");
            for (column, (x, y)) in a.iter().zip(&e).enumerate() {
                match (x.parse::<f64>(), y.parse::<f64>()) {
                    (Ok(x), Ok(y)) => {
                        let scale = x.abs().max(y.abs()).max(1e-300);
                        let difference = (x - y).abs();
                        worst_abs = worst_abs.max(difference);
                        if scale > 1e-6 {
                            worst_rel_large = worst_rel_large.max(difference / scale);
                        }
                        numerical_comparison::assert_close(
                            x,
                            y,
                            1e-12,
                            1e-11,
                            format!("{name}: row {row} column {column}"),
                        );
                    }
                    _ => assert_eq!(x, y, "{name}: row {row} column {column}"),
                }
            }
        }
    }
    println!(
        "end-to-end worst absolute difference {worst_abs:.2e}, worst relative (|value| > 1e-6) {worst_rel_large:.2e}"
    );
}

/// Selecting the tenferro measurement stage for a complex run is a hard error in the
/// runner (complex/FSZ have no stage implementation): never a silent fallback to the
/// inline C-order kernels.
#[test]
#[should_panic(expected = "supports only real, non-FSZ mode")]
fn tenferro_measurement_in_complex_mode_is_a_hard_error() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    set_measurement_consumer_override(Some(MeasurementConsumer::Stage(
        StageBackendKind::TenferroCpu,
    )));
    let source = repo_root().join("tests/fixtures/physcal_181/heisenberg_chain_cmp");
    let root = std::env::temp_dir().join(format!("mvmc-issue518-complex-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let inputs = root.join("inputs");
    std::fs::create_dir_all(&inputs).unwrap();
    for entry in std::fs::read_dir(source.join("inputs")).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), inputs.join(entry.file_name())).unwrap();
    }
    std::fs::copy(source.join("zqp_opt.dat"), root.join("zqp_opt.dat")).unwrap();
    let preparation = mvmc_core::prepare_phys_cal_from_namelist(
        inputs.join("namelist.def"),
        root.join("zqp_opt.dat"),
        "cmp",
        Some(1),
    )
    .unwrap();
    let output = root.join("output");
    let _ = mvmc_core::vmc_phys_cal_to_dir(preparation, &output);
    set_measurement_consumer_override(None);
}

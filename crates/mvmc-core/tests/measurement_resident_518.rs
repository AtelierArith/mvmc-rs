//! Issue #518 milestone 2: the device-resident measurement protocol through the production
//! runner, exercised on the host by a mock backend with C-order arithmetic.
//!
//! The mock implements the same seam the CUDA provider uses: [`PfaffianStages::stage_a_resident`]
//! fills a shared, server-side cache and [`MeasurementStages::measure_sample`] consumes it by
//! `batch_slot` through [`mvmc_core::measurement_backend::resident_measure_sample`]. The
//! production optimization run with `MVMC_RS_MEASURE_PF_BACKEND=cuda` +
//! `MVMC_RS_MEASURE_BACKEND=cuda` must match the C-order run within the stage reordering bound,
//! and the resident traffic counters must show that no inverse plane was downloaded (the Opt
//! path does not need host tables) and that exactly one Slater upload happened per content
//! version. Non-float output tokens are compared exactly.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use mvmc_core::measurement_backend::{
    resident_measure_sample, ResidentMeasurementOps, ResidentRatioTerms, ResidentSlaterDiff,
};
use mvmc_core::measurement_batch::{
    set_measurement_backend_override, set_measurement_consumer_override, MeasurementConsumer,
    MeasurementPfaffian,
};
use mvmc_core::stage_backend::{
    register_stage_backend, COrderMeasurement, COrderPfaffian, MeasurementInput, MeasurementOutput,
    MeasurementStages, MeasurementTiming, PfInvBatch, PfaffianStages, PlaneOutcome,
    ResidentInputRequest, ResidentStageAInput, ResidentStageAOutput, ResidentStats, StageBackend,
    StageBackendKind, StageError,
};
use mvmc_core::state::VmcOptimizationState;
use mvmc_core::ExpertModeData;

static SERIAL: Mutex<()> = Mutex::new(());

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

// ---------------------------------------------------------------------------------------------
// Host mock of the device-resident measurement (C-order arithmetic)
// ---------------------------------------------------------------------------------------------

#[derive(Default, Clone)]
struct MockSlot {
    pf: Vec<f64>,
    /// mVMC `invM = -X^-1`, `[n_qp][n*n]`.
    inv: Vec<f64>,
}

#[derive(Default)]
struct MockDevice {
    geom: Option<(usize, usize, usize, usize)>, // n_site, n_elec, n_qp, n_slater
    slots: Vec<MockSlot>,
    slater: Vec<f64>,
    slater_version: u64,
    orbital_idx: Vec<Vec<i64>>,
    orbital_sgn: Vec<Vec<i64>>,
    qp_weight: Vec<f64>,
    cossin: Vec<f64>,
    coscos: Vec<f64>,
    sinsin: Vec<f64>,
    n_sp: usize,
    stats: ResidentStats,
}

impl MockDevice {
    fn stage_a(
        &mut self,
        input: &ResidentStageAInput<'_>,
    ) -> Result<ResidentStageAOutput, StageError> {
        let n = 2 * input.n_elec;
        let nn = n * n;
        let n2 = 2 * input.n_site;
        let planes = input.slots * input.n_qp;
        self.geom = Some((
            input.n_site,
            input.n_elec,
            input.n_qp,
            self.geom.map_or(0, |g| g.3),
        ));
        if self.slater.len() != input.slater.as_slice().len()
            || self.slater_version != input.slater.version()
        {
            self.slater = input.slater.as_slice().to_vec();
            self.slater_version = input.slater.version();
            self.stats.slater_uploads += 1;
            self.stats.slater_upload_bytes += (self.slater.len() * 8) as u64;
        }
        let mut x = vec![0.0_f64; planes * nn];
        let mut sample_ok = vec![true; input.slots];
        for a in 0..input.slots {
            let ele = &input.ele_idx[a * n..(a + 1) * n];
            let mut cols = vec![0usize; n];
            let mut valid = true;
            for (msi, c) in cols.iter_mut().enumerate() {
                let rsi = ele[msi] + ((msi / input.n_elec) as i64) * (input.n_site as i64);
                if rsi < 0 || rsi >= n2 as i64 {
                    valid = false;
                    break;
                }
                *c = rsi as usize;
            }
            if !valid {
                sample_ok[a] = false;
                continue;
            }
            for qp in 0..input.n_qp {
                let plane = &mut x[(a * input.n_qp + qp) * nn..(a * input.n_qp + qp + 1) * nn];
                for (&rsi, out) in cols.iter().zip(plane.chunks_exact_mut(n)) {
                    let start = (qp * n2 + rsi) * n2;
                    let row = &self.slater[start..start + n2];
                    for (o, &rsj) in out.iter_mut().zip(&cols) {
                        *o = -row[rsj];
                    }
                }
            }
        }
        let out = COrderPfaffian.pfaffian_inverse_batch(&x, n, planes)?;
        self.slots.clear();
        self.slots.resize(input.slots, MockSlot::default());
        let mut pf = Vec::with_capacity(planes);
        let mut outcome = Vec::with_capacity(planes);
        #[allow(clippy::needless_range_loop)]
        for a in 0..input.slots {
            self.slots[a].pf = vec![0.0; input.n_qp];
            self.slots[a].inv = vec![0.0; input.n_qp * nn];
            for qp in 0..input.n_qp {
                let p = a * input.n_qp + qp;
                let o = if sample_ok[a] {
                    out.outcome[p]
                } else {
                    PlaneOutcome::NonFinite
                };
                outcome.push(o);
                pf.push(out.pf[p]);
                self.slots[a].pf[qp] = out.pf[p];
                for (dst, &v) in self.slots[a].inv[qp * nn..(qp + 1) * nn]
                    .iter_mut()
                    .zip(&out.inv[p * nn..(p + 1) * nn])
                {
                    *dst = -v;
                }
            }
        }
        self.stats.stage_a_calls += 1;
        self.stats.planes_factorized += planes as u64;
        self.stats.pf_download_bytes += (planes * 12) as u64;
        let inv = if input.download_inv {
            self.stats.inv_download_planes += planes as u64;
            self.stats.inv_download_bytes += (planes * nn * 8) as u64;
            let mut all = Vec::with_capacity(planes * nn);
            for a in 0..input.slots {
                for qp in 0..input.n_qp {
                    all.extend_from_slice(&self.slots[a].inv[qp * nn..(qp + 1) * nn]);
                }
            }
            Some(all)
        } else {
            None
        };
        Ok(ResidentStageAOutput { pf, inv, outcome })
    }
}

impl ResidentMeasurementOps for MockDevice {
    fn prepare_input(&mut self, request: &ResidentInputRequest<'_>) -> Result<(), StageError> {
        self.orbital_idx = request.orbital_idx.to_vec();
        self.orbital_sgn = request.orbital_sgn.to_vec();
        self.qp_weight = request.qp_weight.to_vec();
        self.cossin = request.spgl_cos_sin.to_vec();
        self.coscos = request.spgl_cos_cos.to_vec();
        self.sinsin = request.spgl_sin_sin.to_vec();
        self.n_sp = request.n_sp_gauss_leg;
        let geom = (
            request.n_site,
            request.n_elec,
            request.n_qp,
            request.n_slater,
        );
        // Every sweep starts without resident slots (its own stage A fills them), mirroring the
        // CUDA device's `prepare_input` reset.
        self.slots.clear();
        if self.geom != Some(geom) {
            self.stats = ResidentStats {
                slater_uploads: self.stats.slater_uploads,
                ..ResidentStats::default()
            };
        }
        self.geom = Some(geom);
        Ok(())
    }

    fn stage_a(
        &mut self,
        input: &ResidentStageAInput<'_>,
    ) -> Result<ResidentStageAOutput, StageError> {
        self.stage_a(input)
    }

    fn has_slot(&self, slot: usize) -> bool {
        slot < self.slots.len()
    }

    fn ratio_terms(
        &mut self,
        slot: usize,
        input: &ResidentRatioTerms<'_>,
    ) -> Result<Vec<f64>, StageError> {
        let (n_site, n_elec, n_qp, _) = self
            .geom
            .ok_or_else(|| StageError::Failed("mock: prepare_input was not called".into()))?;
        if slot >= self.slots.len() {
            return Err(StageError::Failed(format!(
                "mock: slot {slot} is not resident"
            )));
        }
        let n = 2 * n_elec;
        let nn = n * n;
        let n2 = 2 * n_site;
        let pf = &self.slots[slot].pf;
        let inv = &self.slots[slot].inv;
        let mut values = Vec::with_capacity(input.msa.len());
        for t in 0..input.msa.len() {
            let (a, r, moved) = (
                input.msa[t] as usize,
                input.rsa[t] as usize,
                input.ri[t] as usize,
            );
            let mut acc = 0.0;
            for qp in 0..n_qp {
                let inv_plane = &inv[qp * nn..(qp + 1) * nn];
                let mut ratio = 0.0;
                for msj in 0..n {
                    let site = if msj == a {
                        moved
                    } else {
                        input.ele_idx[msj] as usize
                    };
                    let rsj = site + if msj < n_elec { 0 } else { n_site };
                    ratio += inv_plane[a * n + msj] * self.slater[(qp * n2 + r) * n2 + rsj];
                }
                acc += self.qp_weight[qp] * (-(ratio * pf[qp]));
            }
            values.push(acc);
        }
        self.stats.ratio_tables += 1;
        Ok(values)
    }

    fn slater_diff(
        &mut self,
        slot: usize,
        input: &ResidentSlaterDiff<'_>,
    ) -> Result<Vec<f64>, StageError> {
        let (_, n_elec, n_qp, n_slater) = self
            .geom
            .ok_or_else(|| StageError::Failed("mock: prepare_input was not called".into()))?;
        if slot >= self.slots.len() {
            return Err(StageError::Failed(format!(
                "mock: slot {slot} is not resident"
            )));
        }
        let n = 2 * n_elec;
        let nn = n * n;
        let nsp = input.n_sp_gauss_leg;
        let ntrans = input.n_trans;
        let pf = &self.slots[slot].pf;
        let inv = &self.slots[slot].inv;
        // The exact scatter of `accumulate_slater_diff_real` plus the QP-weighted fold.
        let mut qp_orb = vec![0.0_f64; n_qp * n_slater];
        for qp in 0..n_qp {
            let trans = qp / nsp;
            if trans >= ntrans {
                continue;
            }
            let spidx = qp % nsp;
            let cs = pf[qp] * self.cossin[spidx];
            let cc = pf[qp] * self.coscos[spidx];
            let ss = pf[qp] * self.sinsin[spidx];
            let plane = &inv[qp * nn..(qp + 1) * nn];
            for msi in 0..n {
                let tri = input.tri[trans * n + msi];
                if tri < 0 {
                    continue;
                }
                let tri = tri as usize;
                let sgni = input.sgn[trans * n + msi];
                for msj in 0..n {
                    let trj = input.tri[trans * n + msj];
                    if trj < 0 {
                        continue;
                    }
                    let trj = trj as usize;
                    let orb = self.orbital_idx[tri][trj];
                    if orb < 0 || orb as usize >= n_slater {
                        continue;
                    }
                    let sign = sgni * input.sgn[trans * n + msj] * self.orbital_sgn[tri][trj];
                    let (first, second) = if msi < n_elec { (cs, cc) } else { (ss, cs) };
                    let value = if msj < n_elec {
                        plane[msj + msi * n] * first
                    } else {
                        -plane[msj + msi * n] * second
                    };
                    qp_orb[qp * n_slater + orb as usize] += value * sign as f64;
                }
            }
        }
        let mut acc = vec![0.0_f64; n_slater];
        for (orb, out) in acc.iter_mut().enumerate() {
            let mut sum = 0.0;
            for qp in 0..n_qp {
                sum += self.qp_weight[qp] * qp_orb[qp * n_slater + orb];
            }
            *out = sum;
        }
        self.stats.slater_diffs += 1;
        Ok(acc)
    }

    fn stats(&self) -> Option<ResidentStats> {
        Some(self.stats)
    }
}

/// `PfaffianStages` adapter of the mock (device-resident stage A + host-plane fallback).
struct MockStageA {
    device: Arc<Mutex<MockDevice>>,
}

impl PfaffianStages for MockStageA {
    fn label(&self) -> String {
        "cuda-mock-resident".to_string()
    }
    fn provider(&self) -> String {
        "host mock of the device-resident measurement (C-order arithmetic)".to_string()
    }
    fn resident_stage_a(&self) -> bool {
        true
    }
    fn stage_a_resident(
        &mut self,
        input: &ResidentStageAInput<'_>,
    ) -> Result<ResidentStageAOutput, StageError> {
        self.device
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .stage_a(input)
    }
    fn pfaffian_inverse_batch(
        &mut self,
        x: &[f64],
        n: usize,
        planes: usize,
    ) -> Result<PfInvBatch, StageError> {
        COrderPfaffian.pfaffian_inverse_batch(x, n, planes)
    }
}

/// `MeasurementStages` adapter of the mock.
struct MockStageB {
    inner: COrderMeasurement,
    device: Arc<Mutex<MockDevice>>,
}

impl MeasurementStages for MockStageB {
    fn label(&self) -> String {
        "cuda-mock-resident".to_string()
    }
    fn provider(&self) -> String {
        "host mock of the device-resident measurement (C-order arithmetic)".to_string()
    }
    fn resident_measurement(&self) -> bool {
        true
    }
    fn resident_prepare(
        &mut self,
        data: &ExpertModeData,
        state: &VmcOptimizationState,
    ) -> Result<(), StageError> {
        let mut device = self.device.lock().unwrap_or_else(|e| e.into_inner());
        mvmc_core::measurement_backend::resident_prepare(&mut *device, data, state)
    }
    fn resident_stats(&self) -> Option<ResidentStats> {
        Some(self.device.lock().unwrap_or_else(|e| e.into_inner()).stats)
    }
    fn measure_sample(
        &mut self,
        input: &mut MeasurementInput<'_>,
        slater_diff_len: usize,
        timing: &mut dyn MeasurementTiming,
    ) -> Result<MeasurementOutput, StageError> {
        let mut device = self.device.lock().unwrap_or_else(|e| e.into_inner());
        if device.has_slot(input.batch_slot) {
            let mut timer = mvmc_core::c_timer::CTimer::<false>::new();
            timer.diagnostics = timing.diagnostics();
            let out = resident_measure_sample(&mut *device, input, slater_diff_len, &mut timer)?;
            timing.merge_elapsed(&timer.elapsed_ns);
            Ok(out)
        } else {
            drop(device);
            self.inner.measure_sample(input, slater_diff_len, timing)
        }
    }
}

fn register_mock() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        let device = Arc::new(Mutex::new(MockDevice::default()));
        let backend = StageBackend::new(
            "cuda-mock",
            Box::new(mvmc_core::sr_backend::COrderSr::default()),
            Box::new(MockStageA {
                device: device.clone(),
            }),
        )
        .with_measurement(Box::new(MockStageB {
            inner: COrderMeasurement::default(),
            device,
        }));
        register_stage_backend(StageBackendKind::Cuda(0), backend);
    });
}

/// The traffic counters of the registered mock (the production selects `Cuda(0)`).
fn mock_stats() -> ResidentStats {
    use mvmc_core::stage_backend::acquire_kind;
    let mut handle = acquire_kind(StageBackendKind::Cuda(0));
    handle
        .backend()
        .measurement()
        .resident_stats()
        .expect("the registered mock reports stats")
}

// ---------------------------------------------------------------------------------------------
// Optimization comparison
// ---------------------------------------------------------------------------------------------

fn copy_inputs(fixture: &str, tag: &str) -> (PathBuf, PathBuf) {
    let source = repo_root().join("tests/fixtures/physcal_181").join(fixture);
    let root = std::env::temp_dir().join(format!(
        "mvmc-issue518-resident-{}-{fixture}-{tag}",
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

fn optimization(cuda: bool, tag: &str) -> BTreeMap<String, String> {
    if cuda {
        set_measurement_backend_override(Some(MeasurementPfaffian::Stage(StageBackendKind::Cuda(
            0,
        ))));
        set_measurement_consumer_override(Some(MeasurementConsumer::Stage(
            StageBackendKind::Cuda(0),
        )));
    } else {
        set_measurement_backend_override(None);
        set_measurement_consumer_override(None);
    }
    let (root, inputs) = copy_inputs("hubbard_chain_real", tag);
    rewrite_modpara(
        &inputs,
        &[
            ("NVMCCalMode", "0"),
            ("NSROptItrStep", "2"),
            ("NSROptItrSmp", "2"),
            ("NVMCSample", "20"),
            ("NVMCWarmUp", "5"),
            ("DSROptRedCut", "0.001"),
            ("DSROptStaDel", "0.02"),
            ("DSROptStepDt", "0.02"),
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

/// Production optimization with both measurement stages on the resident mock vs the C-order
/// path: every numeric output token inside `abs 1e-12 / rel 1e-11` (the stage re-associates a
/// few sums; the milestone-1 stage measured 1.5e-13), every non-float token exactly, and the
/// residency counters show one table download per batch for the Pfaffians and none for the
/// inverse planes (the Opt path leaves the inverse tables resident, issue #518).
#[test]
fn resident_measurement_selection_matches_c_order_optimization() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    register_mock();
    let reference = optimization(false, "ref");
    let resident = optimization(true, "resident");
    let stats = mock_stats();
    set_measurement_backend_override(None);
    set_measurement_consumer_override(None);

    assert_eq!(
        reference.keys().collect::<Vec<_>>(),
        resident.keys().collect::<Vec<_>>(),
        "file set"
    );
    let mut worst_abs = 0.0_f64;
    let mut worst_rel_large = 0.0_f64;
    for (name, expected) in &reference {
        for (row, (a, e)) in resident[name].lines().zip(expected.lines()).enumerate() {
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
                        assert!(
                            difference <= 1e-12 + 1e-11 * scale,
                            "{name}: row {row} column {column}: {x} vs {y}"
                        );
                    }
                    // Exact non-float quantities (integer counters, column labels).
                    _ => assert_eq!(x, y, "{name}: row {row} column {column}"),
                }
            }
        }
    }
    println!(
        "resident mock worst absolute {worst_abs:.2e}, worst relative (|value| > 1e-6) \
         {worst_rel_large:.2e}; stats {stats:?}"
    );
    assert!(stats.stage_a_calls >= 2, "{stats:?}");
    assert!(stats.planes_factorized >= 40, "{stats:?}");
    assert!(stats.ratio_tables >= 40, "{stats:?}");
    assert!(stats.slater_diffs >= 40, "{stats:?}");
    assert_eq!(
        stats.inv_download_planes, 0,
        "the Opt path must not download inverse planes: {stats:?}"
    );
    // The Slater table is uploaded once per content version, far below one per sample.
    assert!(stats.slater_uploads <= 4, "{stats:?}");
}

/// A selected resident backend in a complex/FSZ run is a hard error, never a silent fallback
/// (the runner guard); this mirrors the tenferro milestone-1 check for the same selector.
#[test]
#[should_panic(expected = "supports only real, non-FSZ mode")]
fn resident_measurement_in_complex_mode_is_a_hard_error() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    register_mock();
    set_measurement_consumer_override(Some(MeasurementConsumer::Stage(StageBackendKind::Cuda(0))));
    let source = repo_root().join("tests/fixtures/physcal_181/heisenberg_chain_cmp");
    let root =
        std::env::temp_dir().join(format!("mvmc-issue518-resident-cmp-{}", std::process::id()));
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

/// The resident stage A leaves the host inverse table intentionally stale when the inverse
/// planes are not downloaded; reading it must fail loudly instead of returning wrong numbers
/// (issue #454, the residency contract of #518).
#[test]
fn resident_stage_a_marks_the_host_inverse_stale() {
    let mut device = MockDevice::default();
    let (n_site, n_elec, n_qp, n_slater) = (2usize, 1usize, 1usize, 1usize);
    device
        .prepare_input(&ResidentInputRequest {
            n_site,
            n_elec,
            n_qp,
            n_slater,
            n_sp_gauss_leg: 1,
            orbital_idx: &[vec![0, -1], vec![-1, -1]],
            orbital_sgn: &[vec![1, 1], vec![1, 1]],
            qp_weight: &[1.0],
            spgl_cos_sin: &[1.0],
            spgl_cos_cos: &[1.0],
            spgl_sin_sin: &[1.0],
        })
        .unwrap();
    let mut batch = mvmc_core::measurement_batch::MeasurementBatchWorkspace::default();
    batch.configure(2 * n_elec, n_qp, 1, false);
    batch.ele_idx[0] = 1;
    batch.ele_idx[1] = 0;
    let mut slater = mvmc_core::state::SlaterElmFlat::<f64>::zeros(n_qp, n_site);
    slater.set(0, 1, 2, 3.0);
    slater.set(0, 2, 1, -3.0);
    let mut stage_a = MockStageA {
        device: Arc::new(Mutex::new(device)),
    };
    mvmc_core::measurement_batch::stage_a_pfaffian(
        &mut stage_a,
        &slater,
        &mut batch,
        &[0],
        n_site,
        n_elec,
        n_qp,
        false,
    )
    .unwrap();
    assert_eq!(
        batch.status[0],
        mvmc_core::measurement_batch::SlotStatus::Ready
    );
    assert!(batch.real[0].inv.is_stale());
}

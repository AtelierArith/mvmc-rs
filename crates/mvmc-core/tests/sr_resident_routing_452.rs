//! Issue #452: production SR routes the direct and CG steps to a resident backend.
//!
//! The resident backend here is a host implementation of the resident stage contract
//! (`SrStages::resident_direct` / `direct_begin` / `direct_assemble` / `direct_factor_solve` and
//! `resident_cg` / `cg_step`) built from the C-order stages with exactly the arithmetic the
//! per-stage path uses. It runs in normal CI without a device and proves the routing and the
//! bookkeeping: with it the whole optimizer run must equal the C-order run bit for bit (the same
//! operations in the same order), while the call counters show that the Gram stage was never
//! asked for a host `OO`, the O store was handed over once per step, and a complex direct run
//! (not supported by the resident direct step) stays on the per-stage path. The CUDA backend's
//! numerics against C order are the job of `gpu/mvmc-gpu-cuda/tests/sr_routing_gate.rs`.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use mvmc_core::sr_backend::{
    cg_step_with, COrderSr, CgSamples, CgStepInput, CgStepResult, DirectSolveInput, GramSummary,
    RealView, SrAssembleInput, SrStages,
};
use mvmc_core::stage_backend::{
    register_stage_backend, set_stage_backend_override, COrderPfaffian, SrSg, StageBackend,
    StageBackendKind, StageError,
};
use num_complex::Complex64;

static BEGIN: AtomicUsize = AtomicUsize::new(0);
static ASSEMBLE: AtomicUsize = AtomicUsize::new(0);
static FACTOR: AtomicUsize = AtomicUsize::new(0);
static CG_STEP: AtomicUsize = AtomicUsize::new(0);
static GRAM_REAL: AtomicUsize = AtomicUsize::new(0);
static GRAM_COMPLEX: AtomicUsize = AtomicUsize::new(0);
static STORE_VALUES: AtomicUsize = AtomicUsize::new(0);

fn reset() {
    for c in [
        &BEGIN,
        &ASSEMBLE,
        &FACTOR,
        &CG_STEP,
        &GRAM_REAL,
        &GRAM_COMPLEX,
        &STORE_VALUES,
    ] {
        c.store(0, Ordering::SeqCst);
    }
}

fn count(c: &AtomicUsize) -> usize {
    c.load(Ordering::SeqCst)
}

/// Host model of a resident SR backend: C-order arithmetic behind the resident stage contract.
#[derive(Default)]
struct ResidentHostSr {
    inner: COrderSr,
    n: usize,
    gram: Vec<f64>,
    scaled: Vec<f64>,
    s: Vec<f64>,
    g: Vec<f64>,
}

impl SrStages for ResidentHostSr {
    fn label(&self) -> String {
        "resident-host-model".to_string()
    }
    fn provider(&self) -> String {
        "COrderSr behind the resident stage contract".to_string()
    }
    fn sr_s_g(
        &mut self,
        o: &[f64],
        nsample: usize,
        npara: usize,
        e: &[f64],
        w: &[f64],
    ) -> Result<SrSg, StageError> {
        self.inner.sr_s_g(o, nsample, npara, e, w)
    }
    fn gram_real(
        &mut self,
        store: &[f64],
        n: usize,
        samples: usize,
        out: &mut [f64],
    ) -> Result<(), StageError> {
        GRAM_REAL.fetch_add(1, Ordering::SeqCst);
        self.inner.gram_real(store, n, samples, out)
    }
    fn gram_complex(
        &mut self,
        store: &[Complex64],
        n: usize,
        samples: usize,
    ) -> Result<Vec<Complex64>, StageError> {
        GRAM_COMPLEX.fetch_add(1, Ordering::SeqCst);
        self.inner.gram_complex(store, n, samples)
    }
    fn assemble_s_g(
        &mut self,
        input: &SrAssembleInput<'_>,
        s: &mut [f64],
        g: &mut [f64],
    ) -> Result<(), StageError> {
        self.inner.assemble_s_g(input, s, g)
    }
    fn cholesky_solve(&mut self, s: &mut [f64], rhs: &mut [f64], n: usize) -> Result<(), ()> {
        self.inner.cholesky_solve(s, rhs, n)
    }
    fn cg_local_product(
        &mut self,
        samples: &CgSamples<'_>,
        x: &[f64],
        z: &mut [f64],
    ) -> Result<(), StageError> {
        self.inner.cg_local_product(samples, x, z)
    }

    fn resident_direct(&self) -> bool {
        true
    }
    fn resident_cg(&self) -> bool {
        true
    }

    fn direct_begin(
        &mut self,
        store: &[f64],
        n: usize,
        samples: usize,
    ) -> Result<GramSummary, StageError> {
        BEGIN.fetch_add(1, Ordering::SeqCst);
        STORE_VALUES.fetch_add(store.len(), Ordering::SeqCst);
        self.n = n;
        self.gram = vec![0.0; n * n];
        self.inner.gram_real(store, n, samples, &mut self.gram)?;
        Ok(GramSummary {
            diag: (0..n).map(|i| self.gram[i + i * n]).collect(),
            col0: (0..n).map(|i| self.gram[i]).collect(),
        })
    }

    fn direct_assemble(&mut self, input: &DirectSolveInput<'_>) -> Result<(), StageError> {
        ASSEMBLE.fetch_add(1, Ordering::SeqCst);
        // the `OO *= 1/wc` of `weight_average_sr_opt_real`, applied once per entry
        self.scaled = self.gram.iter().map(|&x| x * input.gram_scale).collect();
        let nm = input.map.len();
        self.s = vec![0.0; nm * nm];
        self.g = vec![0.0; nm];
        self.inner.assemble_s_g(
            &SrAssembleInput {
                oo: RealView::Real(&self.scaled),
                ho: RealView::Real(input.ho),
                map: input.map,
                ld: self.n,
                offset: input.offset,
                sta_del: input.sta_del,
                step_dt: input.step_dt,
            },
            &mut self.s,
            &mut self.g,
        )
    }

    fn direct_download_s_g(&mut self, _nmap: usize) -> Result<(Vec<f64>, Vec<f64>), StageError> {
        Ok((self.s.clone(), self.g.clone()))
    }

    fn direct_factor_solve(&mut self, nmap: usize) -> Result<Vec<f64>, StageError> {
        FACTOR.fetch_add(1, Ordering::SeqCst);
        let (mut s, mut g) = (std::mem::take(&mut self.s), std::mem::take(&mut self.g));
        self.inner
            .cholesky_solve(&mut s, &mut g, nmap)
            .map_err(|()| StageError::Failed("Cholesky solve failed".into()))?;
        Ok(g)
    }

    fn cg_step(&mut self, input: &CgStepInput<'_>) -> Result<CgStepResult, StageError> {
        CG_STEP.fetch_add(1, Ordering::SeqCst);
        // the host loop on the C-order stages (what the device loop is validated against)
        cg_step_with(&mut self.inner, input)
    }
}

static SERIAL: Mutex<()> = Mutex::new(());

fn install() {
    // Opened by `shared` on first use; the kind needs no provider because it is pre-registered.
    let _ = register_stage_backend(
        StageBackendKind::Cuda(0),
        StageBackend::new(
            "resident-host-model",
            Box::new(ResidentHostSr::default()),
            Box::new(COrderPfaffian),
        ),
    );
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Short optimization run of a `physcal_181` fixture; returns the output files without timers.
fn run_optimization(
    fixture: &str,
    mode: &str,
    cg: bool,
    kind: StageBackendKind,
) -> std::collections::BTreeMap<String, String> {
    set_stage_backend_override(Some(kind));
    let source = repo_root().join("tests/fixtures/physcal_181").join(fixture);
    let root = std::env::temp_dir().join(format!(
        "mvmc-issue452-{}-{fixture}-{}-{kind:?}",
        std::process::id(),
        u8::from(cg)
    ));
    let _ = std::fs::remove_dir_all(&root);
    let inputs = root.join("inputs");
    std::fs::create_dir_all(&inputs).unwrap();
    for entry in std::fs::read_dir(source.join("inputs")).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), inputs.join(entry.file_name())).unwrap();
    }
    let modpara = std::fs::read_to_string(inputs.join("modpara.def")).unwrap();
    let overrides = [
        ("NVMCCalMode", "0"),
        ("NSROptItrStep", "4"),
        ("NSROptItrSmp", "4"),
        ("NVMCSample", "60"),
        ("NVMCWarmUp", "10"),
        ("NSRCG", if cg { "1" } else { "0" }),
        ("NStore", "1"),
        ("NDataIdxStart", "1"),
        ("NDataQtySmp", "1"),
        ("NLanczosMode", "0"),
    ];
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
        if !seen.contains(&k) {
            out.push_str(&format!("{k} {v}\n"));
        }
    }
    std::fs::write(inputs.join("modpara.def"), out).unwrap();
    let output = root.join("output");
    let mut config = mvmc_core::RunConfig::new(4, mode);
    config.output_dir = Some(output.clone());
    config.seed = Some(1);
    let result = mvmc_core::run_para_opt_from_namelist(inputs.join("namelist.def"), config);
    set_stage_backend_override(None);
    result.unwrap_or_else(|e| panic!("{fixture} {kind:?}: {e}"));
    let mut files = std::collections::BTreeMap::new();
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

fn assert_identical(
    label: &str,
    reference: &std::collections::BTreeMap<String, String>,
    routed: &std::collections::BTreeMap<String, String>,
) {
    assert_eq!(
        reference.keys().collect::<Vec<_>>(),
        routed.keys().collect::<Vec<_>>(),
        "{label}: file set"
    );
    for (name, text) in reference {
        assert_eq!(text, &routed[name], "{label}: {name}");
    }
}

/// Real direct SR: the resident step takes over (Gram never requested from the host stage, one
/// store hand-over per SR step) and, with C-order arithmetic behind it, the run is identical.
#[test]
fn real_direct_sr_is_routed_to_the_resident_step() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    install();
    let reference = run_optimization(
        "hubbard_chain_dh_real",
        "real",
        false,
        StageBackendKind::COrder,
    );
    reset();
    let routed = run_optimization(
        "hubbard_chain_dh_real",
        "real",
        false,
        StageBackendKind::Cuda(0),
    );
    assert_identical("real direct", &reference, &routed);
    let steps = 4;
    assert_eq!(
        count(&BEGIN),
        steps,
        "one resident Gram/store upload per step"
    );
    assert_eq!(count(&ASSEMBLE), steps);
    assert_eq!(count(&FACTOR), steps);
    assert_eq!(
        count(&GRAM_REAL),
        0,
        "the host OO must not be materialized on the resident path"
    );
    assert!(count(&STORE_VALUES) > 0);
}

/// Complex direct SR is not covered by the resident direct step: it stays on the per-stage path
/// of the selected backend (Gram through the stage, not the resident step).
#[test]
fn complex_direct_sr_stays_on_the_per_stage_path() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    install();
    let reference = run_optimization(
        "heisenberg_chain_cmp",
        "cmp",
        false,
        StageBackendKind::COrder,
    );
    reset();
    let routed = run_optimization(
        "heisenberg_chain_cmp",
        "cmp",
        false,
        StageBackendKind::Cuda(0),
    );
    assert_identical("complex direct", &reference, &routed);
    assert_eq!(count(&BEGIN), 0);
    assert!(count(&GRAM_COMPLEX) > 0);
}

/// CG SR (real and complex samples): the whole solve goes through the resident `cg_step`, once
/// per SR step.
#[test]
fn cg_sr_is_routed_to_the_resident_solve() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    install();
    for (fixture, mode) in [
        ("hubbard_chain_dh_real", "real"),
        ("heisenberg_chain_cmp", "cmp"),
    ] {
        let reference = run_optimization(fixture, mode, true, StageBackendKind::COrder);
        reset();
        let routed = run_optimization(fixture, mode, true, StageBackendKind::Cuda(0));
        assert_identical(&format!("{fixture} cg"), &reference, &routed);
        assert_eq!(
            count(&CG_STEP),
            4,
            "{fixture}: one resident CG solve per step"
        );
        assert_eq!(count(&BEGIN), 0);
    }
}

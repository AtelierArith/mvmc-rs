//! Accelerated-backend validation harness (issue #424, design `docs/design/gpu-readiness.md`).
//!
//! An accelerated implementation of a tensor-shaped stage changes summation order, so it
//! cannot be compared bitwise with the C-order CPU path and must not be validated by averaging
//! Monte Carlo runs. This harness compares it against the C-order CPU oracle stage by stage:
//!
//! * **Teacher-forced replay** ([`replay`]): the oracle drives a Metropolis trajectory (the
//!   oracle's decisions advance the configuration and consume the RNG); at every step the
//!   backend under test evaluates the *same* candidate configuration, so deviations never
//!   compound through diverging trajectories. It reports the maximum deviation of the
//!   Pfaffian `pf`, the inverse `invM`, the acceptance weights, the O store, `S` and `g`.
//! * **Decision recording**: every proposal records the oracle and backend weights, the draw
//!   and the decision margin `|w - u|`. A decision flip is legitimate only when its margin is
//!   within the weight error that the tolerance allows and not larger than the measured weight
//!   error; a flip with a larger margin is a **defect** ([`ProposalRecord::defect`]).
//! * **Repeatability** ([`repeatability`]): same implementation, same input, same seed: the
//!   discrete state (moves, decisions, draws, configuration, RNG state) is exact and the
//!   computed fields agree within the numerical bound (20 steps by default).
//! * **Metadata** ([`BenchMetadata`], [`bench_stages`]): every benchmark records revision,
//!   hardware, OS, rustc, tenferro version, provider, threads, dtype, batch, warmups,
//!   iterations and whether upload/download are included.
//!
//! The harness drives the production backend object ([`StageBackend`], issue #437): the same
//! `COrderSr`/`TenferroSr`/batched-Pfaffian stage objects that `stage_backend::acquire` hands
//! to production, so a validated backend is the deployed one. A stage a
//! backend does not provide returns [`StageError::Unsupported`]; it is listed as
//! `Unsupported` in the report and never silently replaced by the CPU result. The model is a
//! synthetic pairing-orbital system (`L` sites, `n` electrons, real skew-symmetric `F`) that
//! exercises exactly the linear-algebra stages (Pfaffian + inverse, O store from the inverse,
//! SR `S` and `g`); it is a validation fixture, not an mVMC physics model. The CPU oracle is
//! `StageBackend::c_order()` (the production PfaPack sequence `dsktf2`, `utu2pfa_real`,
//! `utu2inv_real` and the scalar SR loops).
//!
//! RNG contract: one electron draw, one empty-site draw and one acceptance draw per proposal,
//! always in that order and always the same count (also for rejected proposals); the draw
//! sequence and final RNG state are independent of the backend under test.

use std::time::Instant;

use sfmt19937::Sfmt19937Rng;

use crate::stage_backend::{StageBackend, StageError};

// ---------------------------------------------------------------------------------------------
// Tolerances and deviations
// ---------------------------------------------------------------------------------------------

/// Absolute and relative bound: `|a - b| <= abs + rel * max(|a|, |b|)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bound {
    /// Absolute part.
    pub abs: f64,
    /// Relative part.
    pub rel: f64,
}

impl Bound {
    /// Whether `a` and `b` agree within the bound (NaN matches only NaN).
    pub fn within(&self, a: f64, b: f64) -> bool {
        if a.is_nan() || b.is_nan() {
            return a.is_nan() && b.is_nan();
        }
        if !a.is_finite() || !b.is_finite() {
            return a == b;
        }
        (a - b).abs() <= self.abs + self.rel * a.abs().max(b.abs())
    }
}

/// Per-quantity bounds. Defaults are justified by problem scale: `n = 6..12` electrons with
/// O(1) entries, a Pfaffian condition number below about 1e3 for the seeded `F`, and sums of at
/// most a few hundred terms, so reordering errors are O(n * eps) to O(sqrt(ns) * eps) with
/// eps = 2.2e-16; the bounds leave a margin of about 1e4 and still catch a wrong layout, sign
/// or dtype defect (O(1) errors).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tolerances {
    /// Pfaffian.
    pub pf: Bound,
    /// Inverse.
    pub inv: Bound,
    /// Acceptance weight `(pf_new/pf_old)^2`.
    pub weight: Bound,
    /// O store.
    pub o: Bound,
    /// SR matrix.
    pub s: Bound,
    /// SR force.
    pub g: Bound,
}

impl Default for Tolerances {
    fn default() -> Self {
        let b = Bound {
            abs: 1e-12,
            rel: 1e-10,
        };
        Self {
            pf: b,
            inv: b,
            weight: b,
            o: b,
            s: b,
            g: b,
        }
    }
}

/// Maximum deviation of one quantity.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Dev {
    /// Largest absolute difference.
    pub max_abs: f64,
    /// Largest relative difference `|a-b| / max(|a|,|b|)` (over entries with magnitude above
    /// 1e-300).
    pub max_rel: f64,
    /// Entries outside the bound.
    pub violations: usize,
    /// Entries compared.
    pub compared: usize,
}

impl Dev {
    fn add(&mut self, oracle: f64, other: f64, bound: &Bound) {
        let d = (oracle - other).abs();
        self.max_abs = self.max_abs.max(if d.is_nan() { f64::INFINITY } else { d });
        let scale = oracle.abs().max(other.abs());
        if scale > 1e-300 {
            self.max_rel = self.max_rel.max(d / scale);
        }
        self.compared += 1;
        if !bound.within(oracle, other) {
            self.violations += 1;
        }
    }
    fn add_slice(&mut self, oracle: &[f64], other: &[f64], bound: &Bound) {
        assert_eq!(oracle.len(), other.len());
        for (a, b) in oracle.iter().zip(other) {
            self.add(*a, *b, bound);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Synthetic model
// ---------------------------------------------------------------------------------------------

/// Replay configuration.
#[derive(Debug, Clone, Copy)]
pub struct ReplayConfig {
    /// SFMT seed (orbital coefficients, moves and draws).
    pub seed: u32,
    /// Number of sites `L` (`F` is `L x L`).
    pub sites: usize,
    /// Number of electrons `n` (even, `< sites`).
    pub electrons: usize,
    /// Number of proposals.
    pub steps: usize,
    /// Bounds.
    pub tol: Tolerances,
}

impl Default for ReplayConfig {
    fn default() -> Self {
        Self {
            seed: 11272,
            sites: 12,
            electrons: 6,
            steps: 200,
            tol: Tolerances::default(),
        }
    }
}

fn pair_index(i: usize, j: usize, l: usize) -> usize {
    debug_assert!(i < j);
    // index of (i, j), i < j, in row-major upper-triangle order
    i * l - i * (i + 1) / 2 + (j - i - 1)
}

/// Number of parameters `L (L - 1) / 2`.
pub fn npara(sites: usize) -> usize {
    sites * (sites - 1) / 2
}

fn slater_matrix(f: &[f64], l: usize, config: &[usize]) -> Vec<f64> {
    let n = config.len();
    let mut x = vec![0.0; n * n];
    for (b, &rb) in config.iter().enumerate() {
        for (a, &ra) in config.iter().enumerate() {
            if ra < rb {
                x[a + b * n] = f[pair_index(ra, rb, l)];
            } else if ra > rb {
                x[a + b * n] = -f[pair_index(rb, ra, l)];
            }
        }
    }
    x
}

/// `O_p = d ln pf / d F_ij` for one configuration (`(X^-1)_{ba}` at the pair `r_a < r_b`).
pub fn o_row(config: &[usize], inv: &[f64], l: usize) -> Vec<f64> {
    let n = config.len();
    let mut o = vec![0.0; npara(l)];
    for a in 0..n {
        for b in 0..n {
            if config[a] < config[b] {
                o[pair_index(config[a], config[b], l)] = inv[b + a * n];
            }
        }
    }
    o
}

/// Synthetic local energy: a deterministic function of the configuration.
fn local_energy(config: &[usize]) -> f64 {
    let mut e = 0.0;
    for (a, &ra) in config.iter().enumerate() {
        e += 0.37 * ra as f64;
        for &rb in &config[a + 1..] {
            e += (0.5 * (ra as f64 - rb as f64)).cos();
        }
    }
    e
}

/// One recorded proposal.
#[derive(Debug, Clone, PartialEq)]
pub struct ProposalRecord {
    /// Proposal index.
    pub step: usize,
    /// Electron moved.
    pub electron: usize,
    /// Source site.
    pub from: usize,
    /// Target site.
    pub to: usize,
    /// Acceptance draw `u` in `[0, 1)`.
    pub draw: f64,
    /// Oracle weight `(pf_new / pf_old)^2`.
    pub w_oracle: f64,
    /// Backend weight (`None` when the backend does not provide the Pfaffian).
    pub w_backend: Option<f64>,
    /// Oracle decision (`draw < w`).
    pub accept_oracle: bool,
    /// Backend decision.
    pub accept_backend: Option<bool>,
    /// Decision margin `|w_oracle - draw|`.
    pub margin: f64,
    /// Measured weight error `|w_backend - w_oracle|`.
    pub weight_error: Option<f64>,
    /// The two decisions differ.
    pub flip: bool,
    /// A flip whose margin exceeds the measured weight error or the allowed weight bound.
    pub defect: bool,
}

/// Result of [`replay`].
#[derive(Debug, Clone)]
pub struct ReplayReport {
    /// Backend label.
    pub label: String,
    /// Configuration replayed.
    pub steps: usize,
    /// Every proposal.
    pub records: Vec<ProposalRecord>,
    /// Pfaffian deviation (current and candidate configurations).
    pub pf: Dev,
    /// Inverse deviation.
    pub inv: Dev,
    /// Weight deviation.
    pub weight: Dev,
    /// O store deviation.
    pub o: Dev,
    /// `S` deviation.
    pub s: Dev,
    /// `g` deviation.
    pub g: Dev,
    /// Stages the backend reported as unsupported, with the reason (not compared).
    pub unsupported: Vec<String>,
    /// Decision flips.
    pub flips: usize,
    /// Flips classified as defects.
    pub defects: usize,
    /// Smallest decision margin over all proposals (how close the run came to a flip).
    pub min_margin: f64,
    /// Proposals whose margin is within the allowed weight bound (a flip was possible).
    pub at_risk: usize,
    /// Oracle acceptance count.
    pub accepted: usize,
    /// Final configuration of the oracle trajectory.
    pub final_config: Vec<usize>,
    /// Final RNG state words and position.
    pub final_rng: ([u32; 624], usize),
    /// Words consumed from the RNG.
    pub rng_words: u128,
    /// The O store, `S` and `g` computed by the oracle (for repeatability checks).
    pub oracle_s: Vec<f64>,
    /// Oracle `g`.
    pub oracle_g: Vec<f64>,
}

impl ReplayReport {
    /// Violations of the configured bounds and of the decision rule; empty means validated.
    pub fn violations(&self) -> Vec<String> {
        let mut v = Vec::new();
        for (name, d) in [
            ("pf", &self.pf),
            ("invM", &self.inv),
            ("weight", &self.weight),
            ("O store", &self.o),
            ("S", &self.s),
            ("g", &self.g),
        ] {
            if d.violations > 0 {
                v.push(format!(
                    "{name}: {} of {} entries outside the bound (max abs {:.3e}, max rel {:.3e})",
                    d.violations, d.compared, d.max_abs, d.max_rel
                ));
            }
        }
        if self.defects > 0 {
            v.push(format!(
                "{} decision flip(s) classified as defect",
                self.defects
            ));
        }
        v
    }

    /// Human-readable report (markdown).
    pub fn render(&self) -> String {
        let row = |name: &str, d: &Dev| {
            format!(
                "| {name} | {} | {:.3e} | {:.3e} | {} |\n",
                d.compared, d.max_abs, d.max_rel, d.violations
            )
        };
        let mut s = format!(
            "### Replay report: {}\n\nsteps {}, accepted {}, flips {}, defects {}, at-risk {}, \
             min margin {:.3e}, RNG words {}\n\n",
            self.label,
            self.steps,
            self.accepted,
            self.flips,
            self.defects,
            self.at_risk,
            self.min_margin,
            self.rng_words
        );
        s += "| quantity | compared | max abs dev | max rel dev | outside bound |\n|---|---|---|---|---|\n";
        s += &row("pf", &self.pf);
        s += &row("invM", &self.inv);
        s += &row("weight", &self.weight);
        s += &row("O store", &self.o);
        s += &row("S", &self.s);
        s += &row("g", &self.g);
        for u in &self.unsupported {
            s += &format!("\nUnsupported (not compared): {u}\n");
        }
        s
    }
}

fn build_orbitals(rng: &mut Sfmt19937Rng, l: usize) -> Vec<f64> {
    (0..npara(l)).map(|_| rng.genrand_res53() - 0.5).collect()
}

fn initial_config(rng: &mut Sfmt19937Rng, l: usize, n: usize) -> Vec<usize> {
    // partial Fisher-Yates over the sites; fixed draw count `n`
    let mut sites: Vec<usize> = (0..l).collect();
    for i in 0..n {
        let j = i + rng.gen_rand_mod((l - i) as u32) as usize;
        sites.swap(i, j);
    }
    let mut c = sites[..n].to_vec();
    c.sort_unstable();
    c
}

/// Teacher-forced replay of `cfg.steps` proposals: the C-order oracle drives, `backend` is
/// evaluated on the same candidate configurations and compared with the oracle.
pub fn replay(
    backend: &mut StageBackend<'_>,
    cfg: &ReplayConfig,
) -> Result<ReplayReport, StageError> {
    assert!(cfg.electrons.is_multiple_of(2) && cfg.electrons < cfg.sites);
    let (l, n) = (cfg.sites, cfg.electrons);
    let mut oracle = StageBackend::c_order();
    let mut rng = Sfmt19937Rng::new(cfg.seed);
    let f = build_orbitals(&mut rng, l);
    let mut config = initial_config(&mut rng, l, n);

    let mut report = ReplayReport {
        label: backend.label(),
        steps: cfg.steps,
        records: Vec::with_capacity(cfg.steps),
        pf: Dev::default(),
        inv: Dev::default(),
        weight: Dev::default(),
        o: Dev::default(),
        s: Dev::default(),
        g: Dev::default(),
        unsupported: Vec::new(),
        flips: 0,
        defects: 0,
        min_margin: f64::INFINITY,
        at_risk: 0,
        accepted: 0,
        final_config: Vec::new(),
        final_rng: ([0; 624], 0),
        rng_words: 0,
        oracle_s: Vec::new(),
        oracle_g: Vec::new(),
    };
    let note_unsupported = |report: &mut ReplayReport, stage: &str, e: &StageError| {
        let msg = format!("{stage}: {e}");
        if !report.unsupported.contains(&msg) {
            report.unsupported.push(msg);
        }
    };

    let mut cur_oracle = oracle
        .pfaffian()
        .pfaffian_inverse(&slater_matrix(&f, l, &config), n)?;
    let mut o_oracle: Vec<Vec<f64>> = Vec::with_capacity(cfg.steps);
    let mut o_backend: Vec<Option<Vec<f64>>> = Vec::with_capacity(cfg.steps);
    let mut energies = Vec::with_capacity(cfg.steps);

    for step in 0..cfg.steps {
        // RNG protocol: electron, empty-site index, acceptance draw; always three draws.
        let electron = rng.gen_rand_mod(n as u32) as usize;
        let empty_idx = rng.gen_rand_mod((l - n) as u32) as usize;
        let draw = rng.genrand_res53();
        let to = (0..l)
            .filter(|s| !config.contains(s))
            .nth(empty_idx)
            .expect("empty site exists");
        let from = config[electron];
        let mut cand = config.clone();
        cand[electron] = to;
        // keep the configuration sorted so that matrices are canonical; the Pfaffian sign of
        // the reordering is applied through the permutation parity
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&a| cand[a]);
        let sorted: Vec<usize> = order.iter().map(|&a| cand[a]).collect();
        let cur_x = slater_matrix(&f, l, &config);
        let cand_x = slater_matrix(&f, l, &sorted);

        let new_oracle = oracle.pfaffian().pfaffian_inverse(&cand_x, n)?;
        let w_oracle = (new_oracle.pf / cur_oracle.pf).powi(2);
        let accept_oracle = draw < w_oracle;
        let margin = (w_oracle - draw).abs();

        // backend evaluation on the oracle's own configurations (teacher forcing)
        let cur_b = backend.pfaffian().pfaffian_inverse(&cur_x, n);
        let new_b = backend.pfaffian().pfaffian_inverse(&cand_x, n);
        let (w_backend, accept_backend, weight_error) = match (&cur_b, &new_b) {
            (Ok(c), Ok(nw)) => {
                report.pf.add(cur_oracle.pf, c.pf, &cfg.tol.pf);
                report.pf.add(new_oracle.pf, nw.pf, &cfg.tol.pf);
                report.inv.add_slice(&cur_oracle.inv, &c.inv, &cfg.tol.inv);
                report.inv.add_slice(&new_oracle.inv, &nw.inv, &cfg.tol.inv);
                let wb = (nw.pf / c.pf).powi(2);
                report.weight.add(w_oracle, wb, &cfg.tol.weight);
                (Some(wb), Some(draw < wb), Some((wb - w_oracle).abs()))
            }
            (Err(e), _) | (_, Err(e)) => {
                if matches!(e, StageError::Failed(_)) {
                    return Err(e.clone());
                }
                note_unsupported(&mut report, "pfaffian_inverse", e);
                (None, None, None)
            }
        };
        let allowed = cfg.tol.weight.abs + cfg.tol.weight.rel * w_oracle.abs();
        let flip = accept_backend.is_some_and(|a| a != accept_oracle);
        let defect = flip && weight_error.is_some_and(|e| margin > e || margin > allowed);
        if flip {
            report.flips += 1;
        }
        if defect {
            report.defects += 1;
        }
        report.min_margin = report.min_margin.min(margin);
        if margin <= allowed {
            report.at_risk += 1;
        }
        report.records.push(ProposalRecord {
            step,
            electron,
            from,
            to,
            draw,
            w_oracle,
            w_backend,
            accept_oracle,
            accept_backend,
            margin,
            weight_error,
            flip,
            defect,
        });

        if accept_oracle {
            config = sorted;
            cur_oracle = new_oracle;
            report.accepted += 1;
        }

        // measurement after the decision: O row and local energy of the current configuration
        o_oracle.push(o_row(&config, &cur_oracle.inv, l));
        energies.push(local_energy(&config));
        let ob = match backend
            .pfaffian()
            .pfaffian_inverse(&slater_matrix(&f, l, &config), n)
        {
            Ok(p) => {
                let row = o_row(&config, &p.inv, l);
                report
                    .o
                    .add_slice(o_oracle.last().unwrap(), &row, &cfg.tol.o);
                Some(row)
            }
            Err(e) => {
                if matches!(e, StageError::Failed(_)) {
                    return Err(e);
                }
                None
            }
        };
        o_backend.push(ob);
    }

    // SR stage: the backend's S/g on its own O store (or the oracle's when it has none)
    let np = npara(l);
    let ns = cfg.steps;
    let flatten = |rows: &[&Vec<f64>]| {
        let mut o = vec![0.0; ns * np];
        for (k, row) in rows.iter().enumerate() {
            for p in 0..np {
                o[k + p * ns] = row[p];
            }
        }
        o
    };
    let w = vec![1.0 / ns as f64; ns];
    let o_ref = flatten(&o_oracle.iter().collect::<Vec<_>>());
    let ref_sg = oracle.sr().sr_s_g(&o_ref, ns, np, &energies, &w)?;
    let o_test = if o_backend.iter().all(Option::is_some) {
        flatten(
            &o_backend
                .iter()
                .map(|r| r.as_ref().unwrap())
                .collect::<Vec<_>>(),
        )
    } else {
        o_ref.clone()
    };
    match backend.sr().sr_s_g(&o_test, ns, np, &energies, &w) {
        Ok(sg) => {
            report.s.add_slice(&ref_sg.s, &sg.s, &cfg.tol.s);
            report.g.add_slice(&ref_sg.g, &sg.g, &cfg.tol.g);
        }
        Err(e @ StageError::Unsupported(_)) => note_unsupported(&mut report, "sr_s_g", &e),
        Err(e) => return Err(e),
    }
    report.oracle_s = ref_sg.s;
    report.oracle_g = ref_sg.g;
    report.final_config = config;
    report.final_rng = rng.state_snapshot();
    report.rng_words = rng.words_consumed();
    Ok(report)
}

/// Same-implementation repeatability: two fresh runs with the same input and seed must agree
/// exactly on every discrete quantity (moves, decisions, draw bits, configuration, RNG state)
/// and within `cfg.tol` on the computed fields. Returns the first mismatch.
pub fn repeatability<'a>(
    mut make: impl FnMut() -> StageBackend<'a>,
    cfg: &ReplayConfig,
) -> Result<(), String> {
    let a = replay(&mut make(), cfg).map_err(|e| e.to_string())?;
    let b = replay(&mut make(), cfg).map_err(|e| e.to_string())?;
    if a.final_config != b.final_config {
        return Err("final configuration differs".to_string());
    }
    if a.final_rng != b.final_rng || a.rng_words != b.rng_words {
        return Err("final RNG state differs".to_string());
    }
    for (x, y) in a.records.iter().zip(&b.records) {
        let discrete = x.step == y.step
            && x.electron == y.electron
            && x.from == y.from
            && x.to == y.to
            && x.draw.to_bits() == y.draw.to_bits()
            && x.accept_oracle == y.accept_oracle
            && x.accept_backend == y.accept_backend;
        if !discrete {
            return Err(format!("discrete state differs at step {}", x.step));
        }
        if !cfg.tol.weight.within(x.w_oracle, y.w_oracle) {
            return Err(format!("oracle weight differs at step {}", x.step));
        }
        if let (Some(p), Some(q)) = (x.w_backend, y.w_backend) {
            if !cfg.tol.weight.within(p, q) {
                return Err(format!("backend weight differs at step {}", x.step));
            }
        }
    }
    for (p, q) in a.oracle_s.iter().zip(&b.oracle_s) {
        if !cfg.tol.s.within(*p, *q) {
            return Err("S differs".to_string());
        }
    }
    for (p, q) in a.oracle_g.iter().zip(&b.oracle_g) {
        if !cfg.tol.g.within(*p, *q) {
            return Err("g differs".to_string());
        }
    }
    if a.pf.violations + a.inv.violations + a.o.violations + a.s.violations + a.g.violations
        != b.pf.violations + b.inv.violations + b.o.violations + b.s.violations + b.g.violations
    {
        return Err("deviation classification differs".to_string());
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Benchmark metadata and stage benchmark
// ---------------------------------------------------------------------------------------------

/// Metadata recorded with every benchmark/validation result (the list used by
/// tenferro-decision-rs `docs/agents/specs/docs/05_TESTING_BENCHMARKS.md` section 5).
#[derive(Debug, Clone)]
pub struct BenchMetadata {
    /// Source revision (`MVMC_RS_REVISION`, else `git rev-parse HEAD`, else `unknown`).
    pub revision: String,
    /// CPU model.
    pub cpu: String,
    /// Logical CPUs.
    pub cpu_threads_available: usize,
    /// GPU description (`none` for CPU runs).
    pub gpu: String,
    /// Driver/CUDA/cuBLAS/cuSOLVER versions (`none` for CPU runs).
    pub gpu_libraries: String,
    /// Operating system and architecture.
    pub os: String,
    /// `rustc --version` used to build this crate.
    pub rustc: String,
    /// tenferro version.
    pub tenferro: String,
    /// Backend provider description.
    pub provider: String,
    /// Thread settings in effect (`RAYON_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, ...).
    pub threads: String,
    /// Scalar type.
    pub dtype: String,
    /// Batch size (independent problems per call).
    pub batch: usize,
    /// Warm-up iterations.
    pub warmups: usize,
    /// Timed iterations (median reported).
    pub iterations: usize,
    /// Whether upload and download are inside the timed region.
    pub transfers_included: bool,
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new(program)
        .args(args)
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
}

fn cpu_model() -> String {
    if let Ok(text) = std::fs::read_to_string("/proc/cpuinfo") {
        if let Some(line) = text.lines().find(|l| l.starts_with("model name")) {
            if let Some((_, v)) = line.split_once(':') {
                return v.trim().to_string();
            }
        }
    }
    command_output("sysctl", &["-n", "machdep.cpu.brand_string"])
        .unwrap_or_else(|| "unknown".to_string())
}

impl BenchMetadata {
    /// Collect the metadata for a run. `gpu` is the device report for CUDA runs.
    pub fn collect(
        provider: &str,
        dtype: &str,
        batch: usize,
        warmups: usize,
        iterations: usize,
        transfers_included: bool,
        gpu: Option<&crate::backend::DeviceReport>,
    ) -> Self {
        let env = |k: &str| std::env::var(k).unwrap_or_else(|_| "unset".to_string());
        let (gpu_name, gpu_libs) = match gpu {
            Some(r) => (
                format!(
                    "{} (cc {}, {} bytes)",
                    r.device.as_deref().unwrap_or("unavailable"),
                    r.compute_capability.as_deref().unwrap_or("unavailable"),
                    r.total_memory_bytes
                        .map_or("unavailable".to_string(), |b| b.to_string())
                ),
                format!(
                    "driver {} cuda-driver-api {} nvrtc {} cublas {} cusolver {}",
                    r.driver_version.as_deref().unwrap_or("unavailable"),
                    r.cuda_driver_api.as_deref().unwrap_or("unavailable"),
                    r.nvrtc_version.as_deref().unwrap_or("unavailable"),
                    r.cublas_version.as_deref().unwrap_or("unavailable"),
                    r.cusolver_version.as_deref().unwrap_or("unavailable"),
                ),
            ),
            None => ("none".to_string(), "none".to_string()),
        };
        let os_release = std::fs::read_to_string("/etc/os-release")
            .ok()
            .and_then(|t| {
                t.lines().find_map(|l| {
                    l.strip_prefix("PRETTY_NAME=")
                        .map(|v| v.trim_matches('"').to_string())
                })
            })
            .or_else(|| command_output("uname", &["-sr"]))
            .unwrap_or_else(|| "unknown".to_string());
        Self {
            revision: std::env::var("MVMC_RS_REVISION")
                .ok()
                .or_else(|| command_output("git", &["rev-parse", "HEAD"]))
                .unwrap_or_else(|| "unknown".to_string()),
            cpu: cpu_model(),
            cpu_threads_available: std::thread::available_parallelism().map_or(1, |n| n.get()),
            gpu: gpu_name,
            gpu_libraries: gpu_libs,
            os: format!(
                "{os_release} ({}-{})",
                std::env::consts::OS,
                std::env::consts::ARCH
            ),
            rustc: env!("MVMC_RS_RUSTC_VERSION").to_string(),
            tenferro: "0.7.1".to_string(),
            provider: provider.to_string(),
            threads: format!(
                "RAYON_NUM_THREADS={} OPENBLAS_NUM_THREADS={} OMP_NUM_THREADS={}",
                env("RAYON_NUM_THREADS"),
                env("OPENBLAS_NUM_THREADS"),
                env("OMP_NUM_THREADS")
            ),
            dtype: dtype.to_string(),
            batch,
            warmups,
            iterations,
            transfers_included,
        }
    }

    /// `key=value` lines.
    pub fn render(&self) -> String {
        format!(
            "revision={}\ncpu={}\ncpu_threads_available={}\ngpu={}\ngpu_libraries={}\nos={}\n\
             rustc={}\ntenferro={}\nprovider={}\nthreads={}\ndtype={}\nbatch={}\nwarmups={}\n\
             iterations={}\ntransfers_included={}\n",
            self.revision,
            self.cpu,
            self.cpu_threads_available,
            self.gpu,
            self.gpu_libraries,
            self.os,
            self.rustc,
            self.tenferro,
            self.provider,
            self.threads,
            self.dtype,
            self.batch,
            self.warmups,
            self.iterations,
            self.transfers_included
        )
    }
}

/// Default warm-ups and iterations (tenferro-decision-rs: median of 30 after 5 warm-ups).
pub const DEFAULT_WARMUPS: usize = 5;
/// Default timed iterations.
pub const DEFAULT_ITERATIONS: usize = 30;

/// Median wall time in milliseconds of `f` after `warmups` unmeasured calls.
pub fn median_ms<E>(
    warmups: usize,
    iterations: usize,
    mut f: impl FnMut() -> Result<(), E>,
) -> Result<f64, E> {
    for _ in 0..warmups {
        f()?;
    }
    let mut t = Vec::with_capacity(iterations);
    for _ in 0..iterations.max(1) {
        let start = Instant::now();
        f()?;
        t.push(start.elapsed().as_secs_f64() * 1e3);
    }
    t.sort_by(|a, b| a.total_cmp(b));
    Ok(t[t.len() / 2])
}

/// Median stage times (ms) of the Pfaffian+inverse (`sites`-electron matrix) and the SR stage.
/// A stage the backend does not provide is `None` (reported as unsupported, not as 0).
pub fn bench_stages(
    backend: &mut StageBackend<'_>,
    cfg: &ReplayConfig,
    warmups: usize,
    iterations: usize,
) -> (Option<f64>, Option<f64>) {
    let (l, n) = (cfg.sites, cfg.electrons);
    let mut rng = Sfmt19937Rng::new(cfg.seed);
    let f = build_orbitals(&mut rng, l);
    let config = initial_config(&mut rng, l, n);
    let x = slater_matrix(&f, l, &config);
    let pf = median_ms(warmups, iterations, || {
        backend.pfaffian().pfaffian_inverse(&x, n).map(|_| ())
    })
    .ok();
    let np = npara(l);
    let ns = cfg.steps;
    let mut s = 1u64;
    let o: Vec<f64> = (0..ns * np)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        })
        .collect();
    let e: Vec<f64> = (0..ns).map(|k| (k as f64).sin()).collect();
    let w = vec![1.0 / ns as f64; ns];
    let sr = median_ms(warmups, iterations, || {
        backend.sr().sr_s_g(&o, ns, np, &e, &w).map(|_| ())
    })
    .ok();
    (pf, sr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stage_backend::{COrderPfaffian, PfInvBatch, PfaffianStages};

    /// A Pfaffian stage that perturbs the oracle result; used to prove the harness detects
    /// defects.
    struct Perturbed {
        inner: COrderPfaffian,
        pf_rel: f64,
        flip_all_weights: bool,
        calls: usize,
    }

    impl Perturbed {
        fn backend(pf_rel: f64, flip_all_weights: bool) -> StageBackend<'static> {
            StageBackend::c_order().with_pfaffian(
                "perturbed",
                Box::new(Perturbed {
                    inner: COrderPfaffian,
                    pf_rel,
                    flip_all_weights,
                    calls: 0,
                }),
            )
        }
    }

    impl PfaffianStages for Perturbed {
        fn label(&self) -> String {
            "perturbed".to_string()
        }
        fn provider(&self) -> String {
            "test".to_string()
        }
        fn pfaffian_inverse_batch(
            &mut self,
            x: &[f64],
            n: usize,
            planes: usize,
        ) -> Result<PfInvBatch, StageError> {
            let mut r = self.inner.pfaffian_inverse_batch(x, n, planes)?;
            for pf in &mut r.pf {
                // perturb every second call only: a common factor would cancel in the weight
                self.calls += 1;
                if self.calls % 2 == 1 {
                    *pf *= 1.0 + self.pf_rel;
                }
                if self.flip_all_weights {
                    *pf = 1e-3 * pf.signum(); // grossly wrong Pfaffian
                }
            }
            Ok(r)
        }
    }

    #[test]
    fn oracle_inverse_and_o_row_are_correct_independent_of_the_harness() {
        let (l, n) = (10, 4);
        let mut rng = Sfmt19937Rng::new(5);
        let f = build_orbitals(&mut rng, l);
        let config = vec![1usize, 3, 6, 8];
        let x = slater_matrix(&f, l, &config);
        let r = COrderPfaffian.pfaffian_inverse(&x, n).unwrap();
        // X * invM = I
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += x[i + k * n] * r.inv[k + j * n];
                }
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!((acc - expect).abs() < 1e-10, "X*inv[{i},{j}] = {acc}");
            }
        }
        // O_p = d ln pf / d F_p by central finite difference of the 4x4 Pfaffian
        // pf = x01 x23 - x02 x13 + x03 x12 (closed form, independent of PfaPack)
        let pf4 = |f: &[f64]| {
            let m = slater_matrix(f, l, &config);
            m[4] * m[3 * 4 + 2] - m[2 * 4] * m[3 * 4 + 1] + m[3 * 4] * m[2 * 4 + 1]
        };
        assert!((pf4(&f) - r.pf).abs() < 1e-12 * r.pf.abs().max(1.0));
        let o = o_row(&config, &r.inv, l);
        for p in 0..npara(l) {
            let h = 1e-6;
            let (mut fp, mut fm) = (f.clone(), f.clone());
            fp[p] += h;
            fm[p] -= h;
            let fd = (pf4(&fp).abs().ln() - pf4(&fm).abs().ln()) / (2.0 * h);
            assert!((fd - o[p]).abs() < 1e-6, "O[{p}] {} vs fd {fd}", o[p]);
        }
    }

    #[test]
    fn oracle_against_itself_has_zero_deviation_and_no_flip() {
        let rep = replay(&mut StageBackend::c_order(), &ReplayConfig::default()).unwrap();
        assert!(rep.violations().is_empty(), "{:?}", rep.violations());
        assert_eq!(rep.pf.max_abs, 0.0);
        assert_eq!(rep.flips, 0);
        assert!(rep.accepted > 0 && rep.accepted < rep.steps);
        assert!(rep.pf.compared > 0 && rep.s.compared > 0);
    }

    #[test]
    fn tenferro_cpu_sr_matches_oracle_and_pfaffian_is_unsupported() {
        let rep = replay(
            &mut StageBackend::tenferro_cpu().unwrap(),
            &ReplayConfig::default(),
        )
        .unwrap();
        assert!(rep.violations().is_empty(), "{:?}", rep.violations());
        assert!(rep.s.compared > 0 && rep.g.compared > 0);
        assert_eq!(rep.pf.compared, 0);
        assert!(rep
            .unsupported
            .iter()
            .any(|u| u.starts_with("pfaffian_inverse")));
        // reordered summation: deviation is tiny but the comparison used the bounds, not bits
        assert!(rep.s.max_rel < 1e-12);
    }

    #[test]
    fn small_reordering_noise_is_within_bounds_and_flips_are_not_defects() {
        let mut b = Perturbed::backend(1e-13, false);
        let rep = replay(&mut b, &ReplayConfig::default()).unwrap();
        assert!(rep.violations().is_empty(), "{:?}", rep.violations());
    }

    #[test]
    fn grossly_wrong_pfaffian_is_reported_with_decision_flip_defects() {
        let mut b = Perturbed::backend(0.0, true);
        let rep = replay(&mut b, &ReplayConfig::default()).unwrap();
        assert!(rep.pf.violations > 0);
        assert!(rep.flips > 0);
        assert!(
            rep.defects > 0,
            "a flip with margin >> error must be a defect"
        );
        assert!(!rep.violations().is_empty());
    }

    #[test]
    fn flip_within_weight_error_is_classified_legitimate() {
        // weight error larger than the bound but flip margin inside the error: only the
        // bound violation is reported through `weight`, the flip itself is consistent
        let mut b = Perturbed::backend(5e-3, false);
        let rep = replay(&mut b, &ReplayConfig::default()).unwrap();
        assert!(rep.weight.violations > 0);
        for r in rep.records.iter().filter(|r| r.flip) {
            assert!(r.margin <= r.weight_error.unwrap());
        }
    }

    #[test]
    fn twenty_step_repeatability_of_both_variants() {
        let cfg = ReplayConfig {
            steps: 20,
            ..ReplayConfig::default()
        };
        repeatability(StageBackend::c_order, &cfg).unwrap();
        repeatability(|| StageBackend::tenferro_cpu().unwrap(), &cfg).unwrap();
    }

    #[test]
    fn rng_trajectory_is_independent_of_the_backend() {
        let cfg = ReplayConfig::default();
        let a = replay(&mut StageBackend::c_order(), &cfg).unwrap();
        let b = replay(&mut StageBackend::tenferro_cpu().unwrap(), &cfg).unwrap();
        assert_eq!(a.final_rng, b.final_rng);
        assert_eq!(a.rng_words, b.rng_words);
        assert_eq!(a.final_config, b.final_config);
        // three draws per proposal plus orbital and initial-configuration draws
        let words_a = a.rng_words;
        let c = replay(&mut Perturbed::backend(0.0, true), &cfg).unwrap();
        assert_eq!(
            words_a, c.rng_words,
            "backend errors must not change RNG consumption"
        );
    }

    #[test]
    fn metadata_lists_every_required_field() {
        let m = BenchMetadata::collect(
            "p",
            "f64",
            1,
            DEFAULT_WARMUPS,
            DEFAULT_ITERATIONS,
            true,
            None,
        );
        let text = m.render();
        for key in [
            "revision=",
            "cpu=",
            "gpu=",
            "os=",
            "rustc=",
            "tenferro=",
            "provider=",
            "threads=",
            "dtype=",
            "batch=",
            "warmups=5",
            "iterations=30",
            "transfers_included=true",
        ] {
            assert!(text.contains(key), "missing {key}");
        }
        assert!(m.rustc.starts_with("rustc"));
    }

    #[test]
    fn stage_bench_reports_unsupported_as_none() {
        let cfg = ReplayConfig {
            steps: 20,
            ..ReplayConfig::default()
        };
        let (pf, sr) = bench_stages(&mut StageBackend::tenferro_cpu().unwrap(), &cfg, 1, 3);
        assert!(pf.is_none());
        assert!(sr.is_some());
    }
}

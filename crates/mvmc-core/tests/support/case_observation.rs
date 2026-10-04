//! Isolated TEST-ONLY proposal. No invocation occurs unless explicitly enabled.
//! A child outcome is not an independent comparison or parent-worker assertion.
#[path = "thread_case_recorder.rs"]
mod recorder;
use recorder::Recorder;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Context {
    recorder: Recorder,
    boundaries: BoundaryState,
    expected_cases: BTreeSet<String>,
    worker: usize,
    repeat: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct JobBinding {
    run: String,
    id: usize,
    job: String,
    worker: usize,
    threshold: usize,
    repeat: usize,
    directory: PathBuf,
}

#[derive(Default, Debug, PartialEq, Eq)]
enum JobPhase {
    #[default]
    Idle,
    Active,
    Finished,
    Poisoned,
}

#[derive(Default)]
struct JobMemory {
    phase: JobPhase,
    binding: Option<JobBinding>,
    seeds: BTreeMap<String, u32>,
    dimensions: BTreeMap<String, usize>,
    layouts: BTreeSet<String>,
}

fn layout_label(label: &str) {
    assert!(
        !label.is_empty()
            && label
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"_-".contains(&c)),
        "exact safe layout label"
    );
}

impl JobMemory {
    fn enter(&mut self, binding: JobBinding) {
        assert_eq!(
            self.phase,
            JobPhase::Idle,
            "one recorder job per fresh child process"
        );
        assert!(
            self.binding.is_none()
                && self.seeds.is_empty()
                && self.dimensions.is_empty()
                && self.layouts.is_empty(),
            "stale recorder memory"
        );
        self.binding = Some(binding);
        self.phase = JobPhase::Active;
    }
    fn assert_binding(&self, binding: &JobBinding) {
        assert_eq!(self.phase, JobPhase::Active, "active job guard required");
        assert_eq!(
            self.binding.as_ref(),
            Some(binding),
            "run/invocation/job/worker/repeat/root binding changed"
        );
    }
    fn seed(&mut self, label: &str, seed: u32) {
        assert_eq!(self.phase, JobPhase::Active, "active job guard required");
        layout_label(label);
        assert!(
            !self.seeds.contains_key(label),
            "duplicate launch-seed label"
        );
        self.seeds.insert(label.to_owned(), seed);
    }
    fn dimension(&mut self, label: &str, dimension: usize) {
        assert_eq!(self.phase, JobPhase::Active, "active job guard required");
        layout_label(label);
        assert!(
            dimension > 0 && self.seeds.contains_key(label),
            "positive retained dimension needs launch binding"
        );
        assert!(
            !self.dimensions.contains_key(label),
            "duplicate retained system binding"
        );
        self.dimensions.insert(label.to_owned(), dimension);
    }
    fn layout(&mut self, label: &str) {
        assert_eq!(self.phase, JobPhase::Active, "active job guard required");
        assert!(
            self.seeds.contains_key(label),
            "missing launch-seed binding"
        );
        assert!(!self.layouts.contains(label), "duplicate layout claim");
        self.layouts.insert(label.to_owned());
    }
    fn finish(&mut self, binding: &JobBinding) {
        self.assert_binding(binding);
        assert!(
            !self.seeds.is_empty()
                && self.seeds.keys().cloned().collect::<BTreeSet<_>>() == self.layouts,
            "missing/unexpected layout claims"
        );
        assert!(
            self.dimensions
                .keys()
                .all(|label| self.layouts.contains(label)),
            "unbound retained dimensions"
        );
        self.phase = JobPhase::Finished;
    }
    fn release(&mut self, unwinding: bool) {
        if unwinding || self.phase != JobPhase::Finished {
            self.phase = JobPhase::Poisoned;
        }
        self.binding = None;
        self.seeds.clear();
        self.dimensions.clear();
        self.layouts.clear();
        // Finished/Poisoned deliberately remains: a second job is rejected.
    }
}

fn only_when_enabled<T>(enabled: bool, factory: impl FnOnce() -> T) -> Option<T> {
    if enabled {
        Some(factory())
    } else {
        None
    }
}

fn current_binding() -> JobBinding {
    let run = std::env::var("ISSUE182_CASE_RUN_UUID").unwrap();
    run_uuid(&run);
    let job = std::env::var("ISSUE182_CHILD").unwrap();
    let _ = cases_for_job(&job);
    JobBinding {
        run,
        id: decimal(&std::env::var("ISSUE182_CASE_INVOCATION").unwrap(), false),
        job,
        worker: decimal(&std::env::var("ISSUE182_CASE_WORKER").unwrap(), true),
        threshold: decimal(&std::env::var("ISSUE182_CASE_THRESHOLD").unwrap(), true),
        repeat: decimal(&std::env::var("ISSUE182_CASE_REPEAT").unwrap(), true),
        directory: root().unwrap(),
    }
}

pub struct JobGuard {
    _not_send: std::marker::PhantomData<std::rc::Rc<()>>,
}

pub fn enter_job() -> Option<JobGuard> {
    only_when_enabled(enabled(), || {
        let binding = current_binding();
        let config = super::inner_thread_config();
        // Only worker/threshold are observed from the cached runtime config.
        // Repeat is invocation metadata, not an independently observed repeat.
        settings_match(
            (binding.worker, binding.threshold, binding.repeat),
            (config.threads, config.threshold, binding.repeat),
        );
        CONTEXT.with(|cell| assert!(cell.borrow().is_none(), "stale boundary context"));
        JOB_MEMORY.with(|cell| cell.borrow_mut().enter(binding));
        JobGuard {
            _not_send: std::marker::PhantomData,
        }
    })
}

impl Drop for JobGuard {
    fn drop(&mut self) {
        CONTEXT.with(|cell| {
            cell.borrow_mut().take();
        });
        JOB_MEMORY.with(|cell| cell.borrow_mut().release(std::thread::panicking()));
        // Never emits COMPLETE, terminal, or a successful finish on drop.
    }
}

#[derive(Default)]
struct BoundaryState {
    pending: BTreeSet<(String, String)>,
    completed: BTreeSet<(String, String)>,
}

impl BoundaryState {
    fn begin(&mut self, case: &str, boundary: &str) {
        let key = (case.to_owned(), boundary.to_owned());
        assert!(
            !self.completed.contains(&key) && self.pending.insert(key),
            "duplicate boundary"
        );
    }
    fn complete(&mut self, case: &str, boundary: &str) {
        let key = (case.to_owned(), boundary.to_owned());
        assert!(
            self.pending.remove(&key) && self.completed.insert(key),
            "missing/duplicate START"
        );
    }
    fn finish(&self) {
        assert!(
            self.pending.is_empty() && !self.completed.is_empty(),
            "missing completion/cases"
        );
    }
}

fn decimal(value: &str, positive: bool) -> usize {
    assert!(
        !value.is_empty() && value.bytes().all(|c| c.is_ascii_digit()),
        "invalid invocation/repeat/threshold atom"
    );
    let number: usize = value.parse().expect("metadata overflow");
    assert_eq!(
        number.to_string(),
        value,
        "canonical decimal metadata required"
    );
    assert!(!positive || number > 0, "positive metadata required");
    number
}

fn run_uuid(value: &str) {
    assert_eq!(value.len(), 36, "canonical run UUID required");
    for (index, byte) in value.bytes().enumerate() {
        assert!(
            if [8, 13, 18, 23].contains(&index) {
                byte == b'-'
            } else {
                byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)
            },
            "invalid UUID atom"
        );
    }
}

fn settings_match(requested: (usize, usize, usize), actual: (usize, usize, usize)) {
    assert!(
        [1, 2, 4].contains(&actual.0) && actual.1 > 0 && actual.2 > 0,
        "worker/threshold/repeat"
    );
    assert_eq!(
        actual, requested,
        "actual worker/threshold/repeat differs from parent binding"
    );
}

fn cases_for_job(job: &str) -> BTreeSet<String> {
    let mut cases = BTreeSet::new();
    match job {
        "runners-long20-observed" => {
            for model in [
                "heisenberg_chain_real",
                "heisenberg_chain_cmp",
                "heisenberg_chain_fsz",
                "real-fsz",
                "hubbard_chain_real",
            ] {
                for size in [31, 32, 33] {
                    for (store, cg) in [(0, 0), (1, 0), (0, 1)] {
                        cases.insert(format!("long20/{model}/qp{size}/store{store}/cg{cg}"));
                    }
                    cases.insert(format!("long20/{model}/qp{size}/physcal"));
                }
            }
            for model in super::INDEPENDENT_PREFIX_CASES {
                cases.insert(format!("long20/prefix/{}", model.0));
            }
        }
        "independent-prefixes" => {
            for model in super::INDEPENDENT_PREFIX_CASES {
                cases.insert(format!("prefix/{}", model.0));
            }
        }
        "independent-physcal" => {
            for model in super::INDEPENDENT_PHYSCAL_CASES {
                cases.insert(format!("physcal/{}", model.0));
            }
        }
        "direct-sr-failure-boundary" => {
            cases.insert("failure/hubbard_chain_real/qp32/store0/cg0".to_owned());
        }
        "transfer-site" => {
            for size in [31, 32, 33] {
                cases.insert(format!("transfer/hubbard_chain_real/terms{size}"));
            }
        }
        "reviewed-cg-long20" => {
            cases.insert("cg20/general_rbm_cmp/steps20/window20/store0/cg1".to_owned());
        }
        "independent-real-fsz" => {
            for stage in ["seeded", "initialized", "pre-sr", "final"] {
                cases.insert(format!("real-fsz/{stage}"));
            }
        }
        _ => panic!("recorder supports seven explicit selections only"),
    }
    cases
}

pub struct Invocation {
    id: usize,
    run: String,
    job: String,
    worker: usize,
    threshold: usize,
    repeat: usize,
}

impl Invocation {
    fn metadata(&self) -> String {
        format!(
            "run={}\ninvocation={}\njob={}\nworkers={}\nthreshold={}\nrepeat={}\n",
            self.run, self.id, self.job, self.worker, self.threshold, self.repeat
        )
    }
}

thread_local! {
    static CONTEXT: RefCell<Option<Context>> = const { RefCell::new(None) };
    static JOB_MEMORY: RefCell<JobMemory> = RefCell::new(JobMemory::default());
}
static INVOCATIONS: AtomicUsize = AtomicUsize::new(0);
static COMPARISONS: AtomicUsize = AtomicUsize::new(0);

fn root() -> Option<PathBuf> {
    std::env::var_os("ISSUE182_CASE_RECORD_DIR").map(PathBuf::from)
}

pub fn enabled() -> bool {
    std::env::var_os("ISSUE182_CASE_RECORD_DIR").is_some()
}

pub fn begin(case: &str, boundary: &str) {
    if root().is_none() {
        return;
    }
    JOB_MEMORY.with(|cell| cell.borrow().assert_binding(&current_binding()));
    let job = std::env::var("ISSUE182_CHILD").expect("child job binding required");
    assert!(
        cases_for_job(&job).contains(case),
        "unknown/wrong-job case ID"
    );
    CONTEXT.with(|cell| {
        let mut value = cell.borrow_mut();
        if value.is_none() {
            let path = root().unwrap();
            assert!(
                path.is_dir() && !path.is_symlink(),
                "existing exclusive recorder directory required"
            );
            let run = std::env::var("ISSUE182_CASE_RUN_UUID").expect("explicit recorder run ID");
            run_uuid(&run);
            let invocation =
                std::env::var("ISSUE182_CASE_INVOCATION").expect("parent invocation binding");
            let id = decimal(&invocation, false);
            // The existing lifecycle probe intentionally changes the public env
            // to 99 AFTER freezing configuration. Never label that as actual work.
            let worker = decimal(&std::env::var("ISSUE182_CASE_WORKER").unwrap(), true);
            let threshold = decimal(&std::env::var("ISSUE182_CASE_THRESHOLD").unwrap(), true);
            let repeat = decimal(&std::env::var("ISSUE182_CASE_REPEAT").unwrap(), true);
            let config = super::inner_thread_config();
            settings_match(
                (worker, threshold, repeat),
                (config.threads, config.threshold, repeat),
            );
            let binding = Invocation {
                id,
                run: run.clone(),
                job: std::env::var("ISSUE182_CHILD").unwrap(),
                worker,
                threshold,
                repeat,
            };
            let requested = path.join(format!("child-{id}.requested"));
            assert!(
                requested.is_file() && !requested.is_symlink(),
                "missing parent binding artifact"
            );
            assert_eq!(
                std::fs::read_to_string(requested).unwrap(),
                binding.metadata(),
                "parent invocation/repeat binding"
            );
            write(
                &format!("child-{id}.actual-settings"),
                binding.metadata().as_bytes(),
            );
            *value = Some(Context {
                recorder: Recorder::create(&path.join(format!("child-{invocation}.events")), &run)
                    .unwrap(),
                boundaries: BoundaryState::default(),
                expected_cases: cases_for_job(&job),
                worker,
                repeat,
            });
        }
        let context = value.as_mut().unwrap();
        context.boundaries.begin(case, boundary);
        context
            .recorder
            .event("START", case, boundary, context.worker, context.repeat)
            .unwrap();
    });
}

pub fn complete(case: &str, boundary: &str) {
    if root().is_none() {
        return;
    }
    JOB_MEMORY.with(|cell| cell.borrow().assert_binding(&current_binding()));
    CONTEXT.with(|cell| {
        let mut value = cell.borrow_mut();
        let context = value.as_mut().expect("START before COMPLETE");
        context.boundaries.complete(case, boundary);
        context
            .recorder
            .event("COMPLETE", case, boundary, context.worker, context.repeat)
            .unwrap();
    });
}

pub fn configure_child(
    command: &mut std::process::Command,
    job: &str,
    workers: usize,
    threshold: usize,
) -> Option<Invocation> {
    let directory = root()?;
    assert!(
        directory.is_dir() && !directory.is_symlink(),
        "exclusive existing recorder directory required"
    );
    let run = std::env::var("ISSUE182_CASE_RUN_UUID").expect("explicit run UUID");
    run_uuid(&run);
    settings_match((workers, threshold, 1), (workers, threshold, 1));
    assert!(
        matches!(
            job,
            "runners-long20-observed"
                | "direct-sr-failure-boundary"
                | "independent-prefixes"
                | "independent-physcal"
                | "transfer-site"
                | "reviewed-cg-long20"
                | "independent-real-fsz"
        ),
        "recorder supports seven explicit selections only"
    );
    // Count actual child invocations separately from planned groups and repeats.
    static COUNTS: std::sync::Mutex<Vec<(String, usize, usize)>> =
        std::sync::Mutex::new(Vec::new());
    let mut counts = COUNTS.lock().unwrap();
    let index = counts
        .iter()
        .position(|(name, worker, _)| name == job && *worker == workers);
    let repeat = if let Some(index) = index {
        counts[index].2 += 1;
        counts[index].2
    } else {
        counts.push((job.to_owned(), workers, 1));
        1
    };
    let invocation = INVOCATIONS.fetch_add(1, Ordering::Relaxed);
    command
        .env("ISSUE182_CASE_INVOCATION", invocation.to_string())
        .env("ISSUE182_CASE_WORKER", workers.to_string())
        .env("ISSUE182_CASE_THRESHOLD", threshold.to_string())
        .env("ISSUE182_CASE_REPEAT", repeat.to_string());
    let binding = Invocation {
        id: invocation,
        run,
        job: job.to_owned(),
        worker: workers,
        threshold,
        repeat,
    };
    write(
        &format!("child-{invocation}.requested"),
        binding.metadata().as_bytes(),
    );
    Some(binding)
}

fn write(name: &str, bytes: &[u8]) {
    use std::io::Write;
    let root = root().unwrap();
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(root.join(name))
        .unwrap();
    file.write_all(bytes).unwrap();
    file.flush().unwrap();
}

/// Actual borrowed layout only; never creates independently reviewed expectations.
/// Disabled observation returns before allocating, formatting or copying arrays.
pub fn retain_launch_seed(label: &str, seed: u32) {
    if root().is_none() {
        return;
    }
    JOB_MEMORY.with(|cell| {
        let mut memory = cell.borrow_mut();
        memory.assert_binding(&current_binding());
        memory.seed(label, seed);
    });
}

/// Borrow an actual early-stage RNG without constructing a VMC state.
/// Next words are read from a clone only after observation is enabled.
fn initial_stage_shape(
    label: &str,
    stage: &str,
    parameters: Option<&[num_complex::Complex64]>,
    weights: Option<&[num_complex::Complex64]>,
) {
    match stage {
        "seeded" => {
            assert_eq!(label, "real-fsz-seeded");
            assert!(parameters.is_none() && weights.is_none());
        }
        "initialized" => {
            assert_eq!(label, "real-fsz-initialized");
            for array in [parameters, weights] {
                let values = array.expect("initialized actual array");
                assert!(!values.is_empty(), "zero active initial array");
                assert!(values.iter().all(|v| v.re.is_finite() && v.im.is_finite()));
            }
        }
        _ => panic!("unknown initial stage"),
    }
}

pub fn retain_initial_stage(
    label: &str,
    stage: &str,
    rng: &sfmt19937::Sfmt19937Rng,
    parameters: Option<&[num_complex::Complex64]>,
    weights: Option<&[num_complex::Complex64]>,
) {
    if root().is_none() {
        return;
    }
    assert!(label
        .bytes()
        .all(|c| c.is_ascii_alphanumeric() || b"_-".contains(&c)));
    assert!(!label.is_empty());
    initial_stage_shape(label, stage, parameters, weights);
    let seed = JOB_MEMORY.with(|cell| {
        let memory = cell.borrow();
        memory.assert_binding(&current_binding());
        assert!(!memory.layouts.contains(label), "duplicate initial stage");
        *memory.seeds.get(label).expect("explicit launch seed")
    });
    let id = decimal(&std::env::var("ISSUE182_CASE_INVOCATION").unwrap(), false);
    let run = std::env::var("ISSUE182_CASE_RUN_UUID").unwrap();
    run_uuid(&run);
    let (raw, index) = rng.state_snapshot();
    let count = rng.words_consumed();
    let mut shadow = rng.clone();
    let next: Vec<u32> = (0..624).map(|_| shadow.gen_rand32()).collect();
    write(
        &format!("child-{id}.initial-stage-{label}.json"),
        format!("{{\"run_uuid\":\"{run}\",\"invocation\":{id},\"label\":\"{label}\",\"stage\":\"{stage}\",\"launch_seed\":{seed},\"raw624\":{raw:?},\"index\":{index},\"draw_count\":{count},\"next624\":{next:?},\"npara\":{},\"nqp\":{}}}\n", parameters.map_or(0, |v| v.len()), weights.map_or(0, |v| v.len())).as_bytes(),
    );
    super::discrete(&format!("{label}-raw624"), raw);
    super::discrete(&format!("{label}-rng-index"), index);
    super::discrete(&format!("{label}-draw-count"), count);
    super::discrete(&format!("{label}-rng"), next);
    if let (Some(parameters), Some(weights)) = (parameters, weights) {
        super::complex(&format!("{label}-parameters"), parameters);
        super::complex(&format!("{label}-qpweights"), weights);
    }
    JOB_MEMORY.with(|cell| cell.borrow_mut().layout(label));
}

pub fn retain_sr_dimension(label: &str, dimension: usize) {
    if root().is_none() {
        return;
    }
    JOB_MEMORY.with(|cell| {
        let mut memory = cell.borrow_mut();
        memory.assert_binding(&current_binding());
        memory.dimension(label, dimension);
    });
}

pub fn retain_layout(
    label: &str,
    data: &mvmc_core::ExpertModeData,
    state: &mvmc_core::VmcOptimizationState,
) {
    if root().is_none() {
        return;
    }
    assert!(
        !label.is_empty()
            && label
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"_-".contains(&c)),
        "exact safe layout label required"
    );
    let id = decimal(&std::env::var("ISSUE182_CASE_INVOCATION").unwrap(), false);
    let run = std::env::var("ISSUE182_CASE_RUN_UUID").unwrap();
    run_uuid(&run);
    let (launch_seed, retained_dimension) = JOB_MEMORY.with(|cell| {
        let memory = cell.borrow();
        memory.assert_binding(&current_binding());
        assert!(!memory.layouts.contains(label), "duplicate layout artifact");
        (
            *memory
                .seeds
                .get(label)
                .expect("missing explicit constructor/preparation seed binding"),
            memory.dimensions.get(label).copied(),
        )
    });
    let mut settings = format!(
        "{{\"npara\":{},\"nelec\":{},\"nqp\":{},\"samples\":{},\"onebody\":{},\"twobody\":{},\"store\":{},\"cg\":{},\"input_seed\":{},\"steps\":{},\"window\":{},\"groups\":1,\"ranks\":1,\"all_complex\":{},\"physical\":{},\"launch_seed\":{launch_seed}}}",
        data.count_variational_parameters(),
        data.modpara.nelec,
        state.slater_matrix.inv_m.n_qp_full(),
        data.modpara.nvmc_sample,
        data.green_one_terms.len(),
        data.green_two_ex_terms.len(),
        data.modpara.nstore_o,
        data.modpara.nsrcg,
        data.modpara.rnd_seed,
        data.modpara.nsr_opt_itr_step,
        data.modpara.nsr_opt_itr_smp,
        mvmc_core::get_all_complex_flag(data).unwrap(),
        state.phys_quantities.is_some(),
    );
    if let Some(dimension) = retained_dimension {
        assert_eq!(settings.pop(), Some('}'));
        settings.push_str(&format!(",\"retained_dimension\":{dimension}}}"));
    }
    // A layout is not a case completion or numeric comparison. Exact aliases
    // between emitted prefix and case IDs require a separately reviewed schema.
    write(
        &format!("child-{id}.layout-{label}.json"),
        format!("{{\"run_uuid\":\"{run}\",\"invocation\":{id},\"label\":\"{label}\",\"settings\":{settings}}}\n").as_bytes(),
    );
    JOB_MEMORY.with(|cell| cell.borrow_mut().layout(label));
}

pub fn retain_child(
    invocation: Option<Invocation>,
    job: &str,
    workers: usize,
    threshold: usize,
    output: &std::process::Output,
) {
    if let Some(binding) = invocation {
        assert_eq!(
            (job, workers, threshold),
            (binding.job.as_str(), binding.worker, binding.threshold)
        );
        let id = binding.id;
        write(&format!("child-{id}.stdout"), &output.stdout);
        write(&format!("child-{id}.stderr"), &output.stderr);
        // This is ONLY a helper terminal. It cannot complete a case assertion.
        write(
            &format!("child-{id}.terminal"),
            format!("{}status={:?}\n", binding.metadata(), output.status.code()).as_bytes(),
        );
    }
}

pub fn finish_child() {
    if root().is_none() {
        return;
    }
    CONTEXT.with(|cell| {
        let value = cell.borrow();
        let context = value.as_ref().expect("missing recorder cases");
        context.boundaries.finish();
        let actual: BTreeSet<_> = context
            .boundaries
            .completed
            .iter()
            .map(|(case, _)| case.clone())
            .collect();
        assert_eq!(
            actual, context.expected_cases,
            "missing/unexpected case inventory"
        );
        assert_eq!(
            context.boundaries.completed.len(),
            context.expected_cases.len(),
            "duplicate case boundary"
        );
    });
    // No new COMPLETE record; this only rejects pending/absent boundaries.
    JOB_MEMORY.with(|cell| cell.borrow_mut().finish(&current_binding()));
}

pub fn comparison(
    expected: &std::collections::BTreeMap<String, (char, String)>,
    actual: &std::collections::BTreeMap<String, (char, String)>,
) {
    if root().is_none() {
        return;
    }
    let id = COMPARISONS.fetch_add(1, Ordering::Relaxed);
    // Called AFTER all original compare assertions. Preserve full operands.
    // It is not a generic test-PASS marker and is not independent oracle proof.
    write(
        &format!("comparison-{id}.expected"),
        format!("{expected:?}\n").as_bytes(),
    );
    write(
        &format!("comparison-{id}.actual"),
        format!("{actual:?}\n").as_bytes(),
    );
    let run = std::env::var("ISSUE182_CASE_RUN_UUID").unwrap();
    write(
        &format!("comparison-{id}.terminal"),
        format!("run={run}\ncomparison={id}\nboundary=original-worker-compare-returned\n")
            .as_bytes(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    fn binding() -> JobBinding {
        JobBinding {
            run: "00000000-0000-4000-8000-000000000001".to_owned(),
            id: 0,
            job: "independent-prefixes".to_owned(),
            worker: 1,
            threshold: 32,
            repeat: 1,
            directory: PathBuf::from("synthetic-exclusive-directory"),
        }
    }

    #[test]
    fn initial_stage_missing_wrong_stage_and_zero_active_rejected() {
        let values = [num_complex::Complex64::new(1.0, 0.0)];
        initial_stage_shape("real-fsz-seeded", "seeded", None, None);
        initial_stage_shape(
            "real-fsz-initialized",
            "initialized",
            Some(&values),
            Some(&values),
        );
        for (label, stage, parameters, weights) in [
            (
                "real-fsz-initialized",
                "initialized",
                None,
                Some(values.as_slice()),
            ),
            (
                "real-fsz-initialized",
                "initialized",
                Some(values.as_slice()),
                None,
            ),
            (
                "real-fsz-initialized",
                "initialized",
                Some([].as_slice()),
                Some(values.as_slice()),
            ),
            (
                "real-fsz-seeded",
                "initialized",
                Some(values.as_slice()),
                Some(values.as_slice()),
            ),
            ("real-fsz-seeded", "seeded", Some(values.as_slice()), None),
            ("real-fsz-seeded", "final", None, None),
        ] {
            assert!(
                catch_unwind(|| initial_stage_shape(label, stage, parameters, weights)).is_err()
            );
        }
    }

    #[test]
    fn initial_stage_duplicate_and_missing_seed_memory_rejected() {
        let mut memory = JobMemory::default();
        let binding = binding();
        memory.enter(binding.clone());
        assert!(catch_unwind(AssertUnwindSafe(|| memory.layout("real-fsz-seeded"))).is_err());
        memory.seed("real-fsz-seeded", 1);
        assert!(catch_unwind(AssertUnwindSafe(|| memory.finish(&binding))).is_err());
        memory.layout("real-fsz-seeded");
        assert!(catch_unwind(AssertUnwindSafe(|| memory.layout("real-fsz-seeded"))).is_err());
        memory.seed("real-fsz-initialized", 1);
        memory.layout("real-fsz-initialized");
        memory.finish(&binding);
    }

    #[test]
    fn initial_stage_rng_snapshot_clone_is_nonconsuming() {
        let rng = sfmt19937::Sfmt19937Rng::new(1);
        let before = rng.state_snapshot();
        let count = rng.words_consumed();
        let mut shadow = rng.clone();
        for _ in 0..624 {
            shadow.gen_rand32();
        }
        assert_eq!(rng.state_snapshot(), before);
        assert_eq!(rng.words_consumed(), count);
    }

    #[test]
    fn duplicate_launch_seed_rejected_without_replacement() {
        let mut memory = JobMemory::default();
        memory.enter(binding());
        memory.seed("exact", 1);
        assert!(catch_unwind(AssertUnwindSafe(|| memory.seed("exact", 99))).is_err());
        assert_eq!(memory.seeds["exact"], 1);
    }

    #[test]
    fn duplicate_retained_dimension_rejected_without_replacement() {
        let mut memory = JobMemory::default();
        memory.enter(binding());
        memory.seed("exact", 1);
        memory.dimension("exact", 5);
        assert!(catch_unwind(AssertUnwindSafe(|| memory.dimension("exact", 99))).is_err());
        assert_eq!(memory.dimensions["exact"], 5);
    }

    #[test]
    fn zero_or_unseeded_retained_dimension_rejected() {
        let mut memory = JobMemory::default();
        memory.enter(binding());
        assert!(catch_unwind(AssertUnwindSafe(|| memory.dimension("exact", 5))).is_err());
        memory.seed("exact", 1);
        assert!(catch_unwind(AssertUnwindSafe(|| memory.dimension("exact", 0))).is_err());
        assert!(memory.dimensions.is_empty());
    }

    #[test]
    fn repeated_active_or_finished_job_rejected() {
        let mut memory = JobMemory::default();
        let binding = binding();
        memory.enter(binding.clone());
        assert!(catch_unwind(AssertUnwindSafe(|| memory.enter(binding.clone()))).is_err());
        memory.seed("exact", 1);
        memory.layout("exact");
        memory.finish(&binding);
        memory.release(false);
        assert!(memory.seeds.is_empty() && memory.binding.is_none());
        assert!(catch_unwind(AssertUnwindSafe(|| memory.enter(binding.clone()))).is_err());
    }

    #[test]
    fn disabled_job_does_not_construct_binding_or_maps() {
        let constructed = std::cell::Cell::new(false);
        let result: Option<JobMemory> = only_when_enabled(false, || {
            constructed.set(true);
            panic!("disabled binding/format/heap bookkeeping factory must not run")
        });
        assert!(result.is_none());
        assert!(!constructed.get());
        // This proves our factory is not called, not a global allocator bound
        // on the standard library's environment lookup implementation.
    }

    #[test]
    fn panic_release_clears_maps_and_poison_rejects_reuse() {
        let result = catch_unwind(AssertUnwindSafe(|| {
            let _guard = JobGuard {
                _not_send: std::marker::PhantomData,
            };
            JOB_MEMORY.with(|cell| {
                let mut state = cell.borrow_mut();
                state.enter(binding());
                state.seed("exact", 1);
                state.dimension("exact", 5);
            });
            panic!("synthetic original assertion failure");
        }));
        assert!(result.is_err());
        CONTEXT.with(|cell| assert!(cell.borrow().is_none()));
        JOB_MEMORY.with(|cell| {
            let mut state = cell.borrow_mut();
            assert_eq!(state.phase, JobPhase::Poisoned);
            assert!(
                state.binding.is_none()
                    && state.seeds.is_empty()
                    && state.dimensions.is_empty()
                    && state.layouts.is_empty()
            );
            assert!(catch_unwind(AssertUnwindSafe(|| state.enter(binding()))).is_err());
        });
    }

    #[test]
    fn early_normal_release_is_not_successful_finish() {
        let mut memory = JobMemory::default();
        memory.enter(binding());
        memory.seed("exact", 1);
        memory.release(false);
        assert_eq!(memory.phase, JobPhase::Poisoned);
        assert!(memory.seeds.is_empty());
    }

    #[test]
    fn finish_requires_exact_bound_identity_and_all_layouts() {
        let mut memory = JobMemory::default();
        let binding = binding();
        memory.enter(binding.clone());
        memory.seed("exact", 1);
        assert!(catch_unwind(AssertUnwindSafe(|| memory.finish(&binding))).is_err());
        memory.layout("exact");
        for field in 0..7 {
            let mut changed = binding.clone();
            match field {
                0 => changed.run.push('x'),
                1 => changed.id += 1,
                2 => changed.job.push('x'),
                3 => changed.worker = 4,
                4 => changed.threshold = 99,
                5 => changed.repeat += 1,
                _ => changed.directory.push("other"),
            }
            assert!(catch_unwind(AssertUnwindSafe(|| memory.finish(&changed))).is_err());
        }
        memory.finish(&binding);
        assert_eq!(memory.phase, JobPhase::Finished);
    }

    #[test]
    fn duplicate_or_unseeded_layout_rejected() {
        let mut memory = JobMemory::default();
        memory.enter(binding());
        assert!(catch_unwind(AssertUnwindSafe(|| memory.layout("exact"))).is_err());
        memory.seed("exact", 1);
        memory.layout("exact");
        assert!(catch_unwind(AssertUnwindSafe(|| memory.layout("exact"))).is_err());
    }

    #[test]
    fn invocation_filename_atoms_reject_traversal() {
        for value in [
            "../0",
            "0/other",
            "01",
            "-1",
            "",
            "1\n",
            "999999999999999999999999999999999",
        ] {
            assert!(catch_unwind(|| decimal(value, false)).is_err());
        }
        assert_eq!(decimal("0", false), 0);
    }

    #[test]
    fn run_uuid_rejects_filename_injection() {
        for value in [
            "../run",
            "",
            "00000000-0000-4000-8000-00000000000/",
            "00000000-0000-4000-8000-00000000000A",
        ] {
            assert!(catch_unwind(|| run_uuid(value)).is_err());
        }
        run_uuid("00000000-0000-4000-8000-000000000001");
    }

    #[test]
    fn actual_worker_threshold_repeat_binding_rejects_mismatch() {
        for actual in [(4, 32, 1), (2, 99, 1), (2, 32, 2)] {
            assert!(catch_unwind(|| settings_match((2, 32, 1), actual)).is_err());
        }
        settings_match((2, 32, 1), (2, 32, 1));
    }

    #[test]
    fn repeat_zero_or_noncanonical_rejected() {
        for value in ["0", "01", "../1"] {
            assert!(catch_unwind(|| decimal(value, true)).is_err());
        }
    }

    #[test]
    fn complete_without_pending_rejected() {
        let mut state = BoundaryState::default();
        assert!(catch_unwind(AssertUnwindSafe(|| state.complete("case", "assertions"))).is_err());
    }

    #[test]
    fn duplicate_pending_case_rejected() {
        let mut state = BoundaryState::default();
        state.begin("case", "assertions");
        assert!(catch_unwind(AssertUnwindSafe(|| state.begin("case", "assertions"))).is_err());
    }

    #[test]
    fn duplicate_completed_case_rejected() {
        let mut state = BoundaryState::default();
        state.begin("case", "assertions");
        state.complete("case", "assertions");
        assert!(catch_unwind(AssertUnwindSafe(|| state.begin("case", "assertions"))).is_err());
    }

    #[test]
    fn pending_or_missing_cases_reject_finish() {
        let mut state = BoundaryState::default();
        assert!(catch_unwind(|| state.finish()).is_err());
        state.begin("case", "assertions");
        assert!(catch_unwind(|| state.finish()).is_err());
        state.complete("case", "assertions");
        state.finish();
    }

    #[test]
    fn case_ids_are_exactly_the_full_83_group_inventory() {
        let mut all = BTreeSet::new();
        for job in [
            "runners-long20-observed",
            "independent-prefixes",
            "independent-physcal",
            "direct-sr-failure-boundary",
            "transfer-site",
            "reviewed-cg-long20",
            "independent-real-fsz",
        ] {
            for case in cases_for_job(job) {
                assert!(all.insert(case), "duplicate group inventory");
            }
        }
        assert_eq!(all.len(), 83);
    }

    #[test]
    fn wrong_job_case_and_unknown_selection_rejected() {
        let cases = cases_for_job("independent-prefixes");
        assert!(!cases.contains("long20/prefix/heisenberg_chain_real"));
        assert!(!cases.contains("prefix/../../file"));
        assert!(catch_unwind(|| cases_for_job("invented-pass")).is_err());
    }
}

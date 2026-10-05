//! Issue #342: native negative INFO for the normal initializer.
//!
//! The checked-in `tests/fixtures/issue342_negative_info/*.stdout` files are
//! C-derived: the unmodified original `makeInitialSample[_real]`, `matrix.c`
//! (`CalculateMAll_*`), `projection.c`, `SFMT.c` and the pfapack Fortran
//! `dsktrf`/`zsktrf` linked against OpenBLAS (whose `XERBLA` reports and
//! returns). See `PROVENANCE.txt` there. Ordinary tests read only those files.
//!
//! `Ne = 0` is admitted by C's readers and drives `Nsize = LDA = 0` into
//! `DSKTRF`/`ZSKTRF`, whose argument check `LDA < MAX(1, N)` yields INFO = -5.
//! The initializer's `while (flag > 0)` accepts a negative flag, so it exits
//! after one attempt with an empty electron configuration and no RNG draw.
use mvmc_core::c_timer::CTimer;
use mvmc_core::reducer::Reducer;
use mvmc_core::sampling::driver::{
    vmc_make_sample_real_with_reducer_timed, vmc_make_sample_with_reducer_timed,
};
use mvmc_core::sampling::normal_initial::make_initial_sample_normal_with_info;
use mvmc_core::state::{
    InvMColMajor, SlaterElmFlat, SlaterMatrixData, ThreadedPfaPackWorkspace, VmcOptimizationState,
};
use mvmc_expert_parsers::{ExpertModeData, QuantumProjectionWeights};
use num_complex::Complex64;
use sfmt19937::Sfmt19937Rng;
use std::cell::RefCell;
use std::collections::HashMap;

const SENTINEL: f64 = 12345.0;

struct Native {
    kernel_infos: Vec<i32>,
    returned: i32,
    ele_idx: Vec<i64>,
    ele_cfg: Vec<i64>,
    ele_num: Vec<i64>,
    pf_shared_unchanged: bool,
    setup_info: i32,
    pf_setup_unchanged: bool,
    rng_after: [u32; 4],
    rng_fresh: [u32; 4],
    xerbla: Vec<String>,
}

fn ints<T: std::str::FromStr>(value: &str) -> Vec<T>
where
    T::Err: std::fmt::Debug,
{
    value
        .split_whitespace()
        .map(|t| t.parse().unwrap())
        .collect()
}

fn native(mode: &str, case: &str) -> Native {
    let path = format!(
        "{}/../../tests/fixtures/issue342_negative_info/{mode}.{case}.stdout",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(path).unwrap();
    let mut map = HashMap::new();
    let mut xerbla = Vec::new();
    for line in text.lines() {
        if line.starts_with(" ** On entry to ") {
            xerbla.push(line.trim().to_string());
        } else if let Some((k, v)) = line.split_once('=') {
            map.insert(k.to_string(), v.to_string());
        }
    }
    assert_eq!(map["mode"], mode);
    assert_eq!(map["case"], case);
    let kernel_infos = ints::<i32>(&map["kernel_infos"]);
    assert_eq!(
        map["attempts"].parse::<usize>().unwrap(),
        kernel_infos.len()
    );
    let rng = |k: &str| -> [u32; 4] { ints::<u32>(&map[k]).try_into().unwrap() };
    Native {
        kernel_infos,
        returned: map["returned"].parse().unwrap(),
        ele_idx: ints(&map["eleIdx"]),
        ele_cfg: ints(&map["eleCfg"]),
        ele_num: ints(&map["eleNum"]),
        pf_shared_unchanged: map["pf_shared_sentinel_unchanged"] == "1",
        setup_info: map["setup_info"].parse().unwrap(),
        pf_setup_unchanged: map["pf_setup_sentinel_unchanged"] == "1",
        rng_after: rng("rng_next4_after"),
        rng_fresh: rng("rng_next4_fresh"),
        xerbla,
    }
}

fn model(n_elec: i64) -> ExpertModeData {
    let mut data = ExpertModeData::new();
    data.modpara.nsite = 2;
    data.modpara.nelec = n_elec;
    // Zero proposal iterations: C's `% Ne` proposal code is outside the
    // initializer contract (and would divide by zero for Ne = 0).
    data.modpara.nvmc_sample = 0;
    data.modpara.nvmc_warmup = 0;
    data.modpara.nvmc_interval = 0;
    let mut weights = QuantumProjectionWeights::new();
    weights.qp_full_weight = vec![Complex64::new(1.0, 0.0)];
    data.qp_weights = Some(weights);
    data
}

fn matrix(n_elec: usize) -> SlaterMatrixData {
    let mut matrix = SlaterMatrixData {
        slater_elm: SlaterElmFlat::zeros(1, 2),
        inv_m: InvMColMajor::zeros(1, n_elec),
        pf_m: vec![Complex64::new(SENTINEL, 0.0)],
        slater_elm_real: SlaterElmFlat::zeros(1, 2),
        inv_m_real: InvMColMajor::zeros(1, n_elec),
        pf_m_real: vec![SENTINEL],
    };
    if n_elec == 1 {
        for up in 0..2 {
            for down in 0..2 {
                for (row, col, value) in [(up, down + 2, 1.0), (down + 2, up, -1.0)] {
                    matrix
                        .slater_elm
                        .set(0, row, col, Complex64::new(value, 0.0));
                    matrix.slater_elm_real.set(0, row, col, value);
                }
            }
        }
    }
    matrix
}

fn next4(rng: &mut Sfmt19937Rng) -> [u32; 4] {
    std::array::from_fn(|_| rng.gen_rand32())
}

fn check_fresh(native: &Native) {
    let mut fresh = Sfmt19937Rng::new(1);
    assert_eq!(next4(&mut fresh), native.rng_fresh);
}

/// XERBLA report for argument 5 (`LDA < MAX(1, N)`) of an SKTRF routine.
fn xerbla_lda(routine: &str) -> String {
    format!("** On entry to {routine} parameter number  5 had an illegal value")
}

/// Shared initializer: typed INFO, RNG words/count, configuration and storage.
fn check_shared(mode: &str, case: &str) {
    let complex = mode == "complex";
    let c = native(mode, case);
    check_fresh(&c);
    let n_elec = c.ele_idx.len() / 2;
    let data = model(n_elec as i64);
    let mut m = matrix(n_elec);
    let pool = ThreadedPfaPackWorkspace::new(2 * n_elec, 1);
    let mut rng = Sfmt19937Rng::new(1);
    let (mut idx, mut cfg, mut num, mut proj) = (
        vec![-7; 2 * n_elec],
        vec![-7; 4],
        vec![-7; 4],
        Vec::<i64>::new(),
    );
    let mut calls = Vec::new();
    let info = make_initial_sample_normal_with_info(
        &mut idx,
        &mut cfg,
        &mut num,
        &mut proj,
        &data,
        &[0, 0],
        &mut rng,
        &mut m,
        0..1,
        &pool,
        |status| {
            calls.push(status);
            Ok(status)
        },
    )
    .expect("negative native INFO is not a typed failure");
    // Preflight status 0, then exactly the native kernel INFO list.
    let mut expected_calls = vec![0];
    expected_calls.extend_from_slice(&c.kernel_infos);
    assert_eq!(calls, expected_calls, "{mode}/{case}");
    assert_eq!(info.attempts, c.kernel_infos.len());
    assert_eq!(info.local_info, *c.kernel_infos.last().unwrap());
    assert_eq!(info.comm1_info, info.local_info);
    assert_eq!(c.returned, 0, "C's initializer always returns 0");
    assert_eq!((idx, cfg, num), (c.ele_idx, c.ele_cfg, c.ele_num));
    // RNG: identical consumption as C, observed through the next 4 words.
    let consumed = rng.words_consumed();
    assert_eq!(next4(&mut rng), c.rng_after, "{mode}/{case}");
    if case == "ne0" {
        assert_eq!(info.local_info, -5);
        assert_eq!(c.kernel_infos, [-5]);
        // Native argument error: parameter 5 of ZSKTRF in the shared complex
        // loop, then of ZSKTRF/DSKTRF in the separate caller setup.
        let setup = if complex { "ZSKTRF" } else { "DSKTRF" };
        let count = |routine: &str| {
            c.xerbla
                .iter()
                .filter(|l| **l == xerbla_lda(routine))
                .count()
        };
        assert_eq!(count("ZSKTRF"), if complex { 2 } else { 1 });
        assert_eq!(count(setup), if complex { 2 } else { 1 });
        assert_eq!(c.setup_info, -5, "separate setup status is ignored by C");
        assert_eq!(c.rng_after, c.rng_fresh, "no placement draw for Ne = 0");
        assert_eq!(consumed, 0);
    } else {
        assert_eq!(info.local_info, 0);
        assert_eq!(c.setup_info, 0);
        assert!(c.xerbla.is_empty());
        assert_eq!(
            consumed, 2,
            "two placement words, as the native stream skips two"
        );
    }
    // The shared loop validates with the complex master in both modes.
    assert_eq!(
        m.pf_m == [Complex64::new(SENTINEL, 0.0)],
        c.pf_shared_unchanged,
        "{mode}/{case}"
    );
}

#[test]
fn native342_shared_ne0_negative_info_real() {
    check_shared("real", "ne0");
}
#[test]
fn native342_shared_ne0_negative_info_complex() {
    check_shared("complex", "ne0");
}
#[test]
fn native342_shared_ne1_control_real() {
    check_shared("real", "ne1");
}
#[test]
fn native342_shared_ne1_control_complex() {
    check_shared("complex", "ne1");
}

#[derive(Default)]
struct Audit {
    info: RefCell<Vec<i32>>,
}
impl Reducer for Audit {
    fn allreduce_sum_f64(&self, _: &mut [f64]) {}
    fn allreduce_sum_c64(&self, _: &mut [Complex64]) {}
    fn allreduce_sum_i64(&self, _: &mut [i64]) {}
    fn sampling_max_info(&self, info: i32) -> Result<i32, String> {
        self.info.borrow_mut().push(info);
        Ok(info)
    }
}

/// Public sampler with zero proposal iterations, as the full C program run
/// (`vmc.out`, Ne = 0, NVMCWarmUp = NVMCSample = 0) completes the initializer.
fn check_sampler(mode: &str) {
    let complex = mode == "complex";
    let c = native(mode, "ne0");
    let data = model(0);
    let mut state = VmcOptimizationState::zeros(2, 0, 0, 0, 1, 0, complex, false);
    state.slater_matrix = matrix(0);
    let mut rng = Sfmt19937Rng::new(1);
    let audit = Audit::default();
    let mut timer = CTimer::<false>::new();
    let result = if complex {
        vmc_make_sample_with_reducer_timed(&data, &mut state, &mut rng, &mut timer, &audit)
    } else {
        vmc_make_sample_real_with_reducer_timed(&data, &mut state, &mut rng, &mut timer, &audit)
    };
    let stats = result.expect("native Ne=0 negative INFO does not reject the sampler");
    assert_eq!((stats.accepted, stats.saved), (0, 0));
    assert_eq!(rng.words_consumed(), 0);
    assert_eq!(next4(&mut rng), c.rng_after);
    let config = &state.electron_config;
    assert!(config.tmp_ele_idx.is_empty());
    assert_eq!(config.tmp_ele_cfg, c.ele_cfg);
    assert_eq!(config.tmp_ele_num, c.ele_num);
    // sampler preflight, shared preflight, native INFO, separate setup status.
    assert_eq!(*audit.info.borrow(), [0, 0, -5, 0]);
    // C leaves the published Pfaffian untouched on this negative INFO.
    let pf_unchanged = if complex {
        state.slater_matrix.pf_m == [Complex64::new(SENTINEL, 0.0)]
    } else {
        state.slater_matrix.pf_m_real == [SENTINEL]
    };
    assert_eq!(pf_unchanged, c.pf_setup_unchanged);
    assert_eq!(c.setup_info, -5);
}

#[test]
fn native342_sampler_ne0_negative_info_real() {
    check_sampler("real");
}
#[test]
fn native342_sampler_ne0_negative_info_complex() {
    check_sampler("complex");
}

/// Smallest admitted sizes (`1 <= Ne <= Nsite`) never produce a negative INFO:
/// SKTRF's argument checks (`N >= 0`, `LDA = N >= MAX(1, N)`, `LWORK = N*N >= 1`)
/// all hold for `N = 2*Ne >= 2`, so only zero/positive statuses are possible.
#[test]
fn native342_positive_electron_counts_never_report_negative_info() {
    for (n_site, n_elec) in [(1usize, 1usize), (2, 1), (2, 2), (3, 1)] {
        for fill in [0.0, 1.0, f64::NAN] {
            let mut data = model(n_elec as i64);
            data.modpara.nsite = n_site as i64;
            let mut m = SlaterMatrixData {
                slater_elm: SlaterElmFlat::zeros(2, n_site),
                inv_m: InvMColMajor::zeros(2, n_elec),
                pf_m: vec![Complex64::new(SENTINEL, 0.0); 2],
                slater_elm_real: SlaterElmFlat::zeros(2, n_site),
                inv_m_real: InvMColMajor::zeros(2, n_elec),
                pf_m_real: vec![SENTINEL; 2],
            };
            for qp in 0..2 {
                for row in 0..2 * n_site {
                    for col in 0..2 * n_site {
                        let sign = if row < col { 1.0 } else { -1.0 };
                        let value = if row == col { 0.0 } else { sign * fill };
                        m.slater_elm.set(qp, row, col, Complex64::new(value, 0.0));
                    }
                }
            }
            let pool = ThreadedPfaPackWorkspace::new(2 * n_elec, 1);
            let mut rng = Sfmt19937Rng::new(1);
            let mut statuses = Vec::new();
            let _ = make_initial_sample_normal_with_info(
                &mut vec![0; 2 * n_elec],
                &mut vec![0; 2 * n_site],
                &mut vec![0; 2 * n_site],
                &mut Vec::new(),
                &data,
                &vec![0; n_site],
                &mut rng,
                &mut m,
                0..2,
                &pool,
                |status| {
                    statuses.push(status);
                    Ok(status)
                },
            );
            assert!(
                statuses.len() >= 2 && statuses.iter().all(|&s| s >= 0),
                "Nsite={n_site} Ne={n_elec} fill={fill}: {statuses:?}"
            );
        }
    }
}

//! Independently C-derived retained fixtures; serialization and refusal controls accepted.
//! No toolbox/reference runtime is invoked by this ordinary Rust test.
use mvmc_core::c_timer::CTimer;
use mvmc_core::reducer::Reducer;
use mvmc_core::sampling::driver::{
    trace, vmc_make_sample_real_with_reducer_timed, vmc_make_sample_with_reducer_timed,
};
use mvmc_core::sampling::initial::make_initial_sample;
use mvmc_core::sampling::normal_initial::NormalInitializationError;
use mvmc_core::state::VmcOptimizationState;
use mvmc_expert_parsers::{ExpertModeData, QuantumProjectionWeights};
use num_complex::Complex64;
use serde_json::Value;
use sfmt19937::Sfmt19937Rng;
use std::cell::RefCell;

fn unsigned(v: &Value) -> u64 {
    v.as_u64().expect("fixture unsigned numeric type")
}
fn signed(v: &Value) -> i64 {
    v.as_i64().expect("fixture signed numeric type")
}
fn count(v: &Value) -> u128 {
    let s = v.as_str().expect("canonical-decimal-u128-string");
    assert!(!s.is_empty() && (s == "0" || !s.starts_with('0')));
    assert!(s.bytes().all(|b| b.is_ascii_digit()));
    s.parse().expect("u128 count")
}
#[derive(Default)]
struct CallerAudit {
    info: RefCell<Vec<i32>>,
    zero_ip: RefCell<Vec<bool>>,
}
impl Reducer for CallerAudit {
    fn allreduce_sum_f64(&self, _: &mut [f64]) {
        panic!("IP must use comm1");
    }
    fn allreduce_sum_c64(&self, _: &mut [Complex64]) {
        panic!("IP must use comm1");
    }
    fn allreduce_sum_i64(&self, _: &mut [i64]) {}
    fn sampling_max_info(&self, info: i32) -> Result<i32, String> {
        self.info.borrow_mut().push(info);
        Ok(info)
    }
    fn sampling_sum_f64(&self, ip: &mut [f64]) {
        assert_eq!(ip.len(), 1);
        assert!(ip[0].is_finite());
        self.zero_ip.borrow_mut().push(ip[0] == 0.0);
    }
    fn sampling_sum_c64(&self, ip: &mut [Complex64]) {
        assert_eq!(ip.len(), 1);
        assert!(ip[0].re.is_finite() && ip[0].im.is_finite());
        self.zero_ip
            .borrow_mut()
            .push(ip[0] == Complex64::new(0.0, 0.0));
    }
}
struct TraceGuard;
impl Drop for TraceGuard {
    fn drop(&mut self) {
        let _ = trace::finish();
    }
}
fn rng_checkpoint(c: &Value, rng: &Sfmt19937Rng) {
    assert_eq!(rng.words_consumed(), count(&c["count"]), "{}", c["stage"]);
    assert_eq!(
        rng.state_snapshot(),
        (words(&c["raw"]), unsigned(&c["cursor"]) as usize)
    );
    let before = rng.state_snapshot();
    let consumed = rng.words_consumed();
    let mut next = [0; 624];
    rng.dump_rand32(&mut next);
    assert_eq!(
        next,
        words(&c["future"]),
        "common-observer full tail, not independent full624 oracle"
    );
    assert_eq!(rng.state_snapshot(), before);
    assert_eq!(rng.words_consumed(), consumed);
}
fn words(v: &Value) -> [u32; 624] {
    let a = v.as_array().expect("array");
    assert_eq!(a.len(), 624);
    std::array::from_fn(|i| u32::try_from(unsigned(&a[i])).unwrap())
}
fn fixture(case: &str, mode: &str) -> Value {
    let path = format!(
        "{}/../../tests/fixtures/issue274_native_boundaries/{case}.{mode}.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let v: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    assert_eq!(v["schema"], "native274-retained-discrete-fixture-v2");
    assert_eq!(v["count_encoding"], "canonical-decimal-u128-string");
    assert_eq!(v["case"], case);
    assert_eq!(v["mode"], mode);
    v
}

fn check_boundary(case: &str, complex: bool) {
    let mode = if complex { "complex" } else { "real" };
    let f = fixture(case, mode);
    let checkpoints = f["lineage"]["observed"]["checkpoints"].as_array().unwrap();
    let seeded = &checkpoints[0];
    assert_eq!(seeded["stage"], "seeded");
    let qps = if case == "recover" { 2 } else { 1 };
    let mut data = ExpertModeData::new();
    data.modpara.nsite = 2;
    data.modpara.nelec = 1;
    // Zero proposal iterations: exercise the actual caller prefix and
    // burn/recovery branches, not a replacement shared initializer.
    data.modpara.nvmc_sample = 0;
    data.modpara.nvmc_warmup = 0;
    data.modpara.nvmc_interval = 0;
    data.n_gutzwiller_idx = 2;
    data.gutzwiller_idx = vec![0, 1];
    let mut weights = QuantumProjectionWeights::new();
    weights.qp_full_weight = (0..qps)
        .map(|q| Complex64::new(if q == 0 { 1.0 } else { -1.0 }, 0.0))
        .collect();
    data.qp_weights = Some(weights);
    let mut state = VmcOptimizationState::zeros(2, 1, 2, 0, qps, 0, complex, false);
    for q in 0..qps {
        for up in 0..2 {
            for down in 0..2 {
                let value = if case == "exhaust" || (case == "retry" && up != down) {
                    0.0
                } else {
                    1.0
                };
                state
                    .slater_matrix
                    .slater_elm
                    .set(q, up, down + 2, Complex64::new(value, 0.0));
                state
                    .slater_matrix
                    .slater_elm
                    .set(q, down + 2, up, Complex64::new(-value, 0.0));
            }
        }
    }
    if case == "burn" {
        let c = &mut state.electron_config;
        c.tmp_ele_idx.copy_from_slice(&[0, 1]);
        c.tmp_ele_cfg.copy_from_slice(&[0, -1, -1, 0]);
        c.tmp_ele_num.copy_from_slice(&[1, 0, 0, 1]);
        c.tmp_ele_proj_cnt.copy_from_slice(&[0, 0]);
        // Same canonical contiguous public burn storage as C, not a
        // call to the crate-private save_burn helper.
        c.burn_ele_idx = vec![0, 1, 0, -1, -1, 0, 1, 0, 0, 1, 0, 0];
        c.counter[9] = 1;
    }
    let mut rng = Sfmt19937Rng::new(1);
    assert_eq!(rng.state_snapshot(), (words(&seeded["raw"]), 624));
    let mut future = [0; 624];
    rng.dump_rand32(&mut future);
    assert_eq!(future, words(&seeded["future"]));
    let audit = CallerAudit::default();
    trace::start_with_raw_checkpoints();
    let guard = TraceGuard;
    let result = if complex {
        vmc_make_sample_with_reducer_timed(
            &data,
            &mut state,
            &mut rng,
            &mut CTimer::<false>::new(),
            &audit,
        )
    } else {
        vmc_make_sample_real_with_reducer_timed(
            &data,
            &mut state,
            &mut rng,
            &mut CTimer::<false>::new(),
            &audit,
        )
    };
    std::mem::forget(guard);
    let events = trace::finish();
    if case == "exhaust" {
        let Err(NormalInitializationError::RetryLimit(info)) = result else {
            panic!("expected typed exhaustion: {result:?}");
        };
        assert_eq!(
            (info.attempts, info.local_info, info.comm1_info),
            (101, 1, 1)
        );
    } else {
        let stats = result.unwrap();
        assert_eq!((stats.accepted, stats.saved), (0, 0));
    }
    let actual_words: Vec<u32> = events
        .iter()
        .filter(|e| e[0] == 9)
        .map(|e| {
            assert_eq!(e.len(), 2);
            u32::try_from(e[1]).unwrap()
        })
        .collect();
    let expected_words: Vec<u32> = f["lineage"]["observed"]["discrete"]["primitive"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| u32::try_from(unsigned(v)).unwrap())
        .collect();
    assert_eq!(
        actual_words, expected_words,
        "first actual primitive divergence {case}/{mode}"
    );
    assert!(
        events.iter().all(|e| [8, 9, 10].contains(&e[0])),
        "no proposals/acceptance"
    );
    let native_infos: Vec<i32> = checkpoints
        .iter()
        .filter(|c| c["stage"] == "local-info")
        .map(|c| i32::try_from(signed(&c["info"])).unwrap())
        .collect();
    let mut expected_callbacks = vec![0]; // sampler typed preflight
    if case != "burn" {
        expected_callbacks.push(0); // shared typed preflight
        let end = if case == "recover" {
            1
        } else {
            native_infos.len()
        };
        expected_callbacks.extend_from_slice(&native_infos[..end]);
    }
    if case != "exhaust" {
        expected_callbacks.push(0); // separate setup supported result, NOT its native INFO
        if case == "recover" {
            expected_callbacks.push(0);
            expected_callbacks.extend_from_slice(&native_infos[1..]);
            expected_callbacks.push(0);
        }
    }
    assert_eq!(
        *audit.info.borrow(),
        expected_callbacks,
        "actual supported-callback ordinal/native INFO"
    );
    let ips: Vec<bool> = f["lineage"]["observed"]["sidecars"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|s| s["type"] == "IP")
        .map(|s| {
            let tokens = s["tokens"].as_array().unwrap();
            assert_eq!(tokens.len(), 8);
            assert_eq!(tokens[6], "native_recovery_predicate");
            match tokens[7].as_str().unwrap() {
                "0" => false,
                "1" => true,
                _ => panic!("IP predicate"),
            }
        })
        .collect();
    assert_eq!(
        *audit.zero_ip.borrow(),
        ips,
        "actual finite-literal overlap cancellation/recovery"
    );
    // Intermediate raw states reconstructed from the EXACT actual
    // caller primitive stream: not new live intermediate snapshots.
    let mut replay = Sfmt19937Rng::new(1);
    for (i, c) in checkpoints.iter().enumerate() {
        assert_eq!(unsigned(&c["sequence"]), (i + 1) as u64);
        assert_eq!(c["dims"], serde_json::json!([2, 1, 2, 4, 2]));
        assert_eq!(c["qp"], serde_json::json!([0, qps]));
        while replay.words_consumed() < count(&c["count"]) {
            let word = actual_words[usize::try_from(replay.words_consumed()).unwrap()];
            assert_eq!(replay.gen_rand32(), word);
        }
        rng_checkpoint(c, &replay);
    }
    rng_checkpoint(checkpoints.last().unwrap(), &rng);
    // Independent public placement replay at all defined C checkpoints.
    // This is explicitly NOT a live borrow of actual caller buffers.
    let mut placement_rng = Sfmt19937Rng::new(1);
    let (mut idx, mut cfg, mut num, mut proj) = (vec![0; 2], vec![0; 4], vec![0; 4], vec![0; 2]);
    if case == "burn" {
        idx = vec![0, 1];
        cfg = vec![0, -1, -1, 0];
        num = vec![1, 0, 0, 1];
    }
    for c in checkpoints
        .iter()
        .filter(|c| c["configuration_defined"] == 1)
    {
        while placement_rng.words_consumed() < count(&c["count"]) {
            make_initial_sample(
                &mut idx,
                &mut cfg,
                &mut num,
                &mut proj,
                &data,
                &[0, 0],
                &mut placement_rng,
            )
            .unwrap();
        }
        assert_eq!(placement_rng.words_consumed(), count(&c["count"]));
        for (name, actual) in [("idx", &idx), ("cfg", &cfg), ("num", &num), ("proj", &proj)] {
            let expected: Vec<i64> = c[name].as_array().unwrap().iter().map(signed).collect();
            assert_eq!(
                actual, &expected,
                "public placement replay {case}/{mode}/{}",
                c["stage"]
            );
        }
    }
    let last = checkpoints
        .iter()
        .rev()
        .find(|c| {
            c["configuration_defined"] == 1 && count(&c["count"]) == actual_words.len() as u128
        })
        .unwrap();
    assert_eq!(rng.words_consumed(), count(&last["count"]), "{case}/{mode}");
    assert_eq!(
        rng.state_snapshot(),
        (words(&last["raw"]), unsigned(&last["cursor"]) as usize),
        "{case}/{mode}"
    );
    rng.dump_rand32(&mut future);
    assert_eq!(
        future,
        words(&last["future"]),
        "off/on-origin tail; {case}/{mode}"
    );
    for (name, actual) in [
        ("idx", &state.electron_config.tmp_ele_idx),
        ("cfg", &state.electron_config.tmp_ele_cfg),
        ("num", &state.electron_config.tmp_ele_num),
        ("proj", &state.electron_config.tmp_ele_proj_cnt),
    ] {
        let expected: Vec<i64> = last[name].as_array().unwrap().iter().map(signed).collect();
        assert_eq!(actual, &expected, "{case}/{mode}/{name}");
    }
}

#[test]
fn native274_first_real() {
    check_boundary("first", false);
}
#[test]
fn native274_first_complex() {
    check_boundary("first", true);
}
#[test]
fn native274_retry_real() {
    check_boundary("retry", false);
}
#[test]
fn native274_retry_complex() {
    check_boundary("retry", true);
}
#[test]
fn native274_burn_real() {
    check_boundary("burn", false);
}
#[test]
fn native274_burn_complex() {
    check_boundary("burn", true);
}
#[test]
fn native274_recover_real() {
    check_boundary("recover", false);
}
#[test]
fn native274_recover_complex() {
    check_boundary("recover", true);
}
#[test]
fn native274_exhaust_real() {
    check_boundary("exhaust", false);
}
#[test]
fn native274_exhaust_complex() {
    check_boundary("exhaust", true);
}

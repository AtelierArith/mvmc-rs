//! Opt-in, observational per-thread sampling events for reference verification.
//! No RNG draws, numerical operations, or configuration mutations are added.

use std::cell::{Cell, RefCell};

thread_local! {
    /// Opt-in log of Metropolis decisions `(weight, draw)` for the multi-chain runner (#425).
    static DECISIONS: RefCell<Option<Vec<(f64, f64)>>> = const { RefCell::new(None) };
    static EVENTS: RefCell<Option<Vec<Vec<i64>>>> = const { RefCell::new(None) };
    static RAW_CHECKPOINTS: Cell<bool> = const { Cell::new(false) };
}

/// Start recording `(weight, draw)` of every Metropolis decision on this thread.
/// Purely observational: no RNG draw, no numerical operation and no state change.
pub fn start_decisions() {
    DECISIONS.with(|slot| *slot.borrow_mut() = Some(Vec::new()));
}

/// Stop recording and return the decisions in execution order (empty if not started).
pub fn finish_decisions() -> Vec<(f64, f64)> {
    DECISIONS.with(|slot| slot.borrow_mut().take().unwrap_or_default())
}

pub(crate) fn record_decision(weight: f64, draw: f64) {
    DECISIONS.with(|slot| {
        if let Some(log) = slot.borrow_mut().as_mut() {
            log.push((weight, draw));
        }
    });
}

/// Enable capture on the sampler's calling thread. Nested captures are errors.
pub fn start() {
    EVENTS.with(|slot| {
        let mut slot = slot.borrow_mut();
        assert!(slot.is_none(), "sampling trace already active");
        *slot = Some(Vec::new());
    });
}

/// Enable the separately versioned raw-checkpoint extension for this capture.
/// Legacy start() remains event0..9 only. Nested requests leave it unchanged.
pub fn start_with_raw_checkpoints() {
    start();
    RAW_CHECKPOINTS.with(|enabled| enabled.set(true));
}

/// End capture and return actual events, in execution order.
pub fn finish() -> Vec<Vec<i64>> {
    let events = EVENTS.with(|slot| slot.borrow_mut().take().expect("sampling trace not active"));
    RAW_CHECKPOINTS.with(|enabled| enabled.set(false));
    events
}

/// Event schema: first integer is kind (0 update, 1 normal hop, 2 normal
/// exchange, 3 FSZ hop, 4 conduction flip, 5 local flip, 6 FSZ exchange,
/// 7 Metropolis). Candidate fields follow C's mi/ri/rj/spin order; exchange
/// adds mj and other spin. The final candidate field is its reject flag.
pub(crate) fn record(kind: i64, fields: &[i64]) {
    EVENTS.with(|slot| {
        if let Some(events) = slot.borrow_mut().as_mut() {
            let mut event = Vec::with_capacity(fields.len() + 1);
            event.push(kind);
            event.extend_from_slice(fields);
            events.push(event);
        }
    });
}

pub(crate) fn draw_real2(rng: &mut sfmt19937::Sfmt19937Rng) -> f64 {
    let value = rng.genrand_real2();
    // SFMT real2 is an exact dyadic mapping of one consumed u32.
    record(9, &[(value * 4294967296.0) as i64]);
    value
}

pub(crate) fn draw_mod(rng: &mut sfmt19937::Sfmt19937Rng, n: u32) -> u32 {
    debug_assert!(n > 0);
    let word = rng.gen_rand32();
    record(9, &[i64::from(word)]);
    word % n
}

/// Sampling boundary before measurement overwrites scratch. Seven length-
/// prefixed fields: saved idx/cfg/num/proj/spin, counters, nonconsuming next624.
pub(crate) fn checkpoint(
    state: &crate::state::VmcOptimizationState,
    rng: &sfmt19937::Sfmt19937Rng,
) {
    EVENTS.with(|slot| {
        if slot.borrow().is_none() {
            return;
        }
        let ec = &state.electron_config;
        let mut words = [0_u32; 624];
        rng.dump_rand32(&mut words);
        let next: Vec<_> = words.into_iter().map(i64::from).collect();
        let mut fields = Vec::new();
        let arrays: [&[i64]; 7] = [
            &ec.ele_idx,
            &ec.ele_cfg,
            &ec.ele_num,
            &ec.ele_proj_cnt,
            &ec.ele_spn,
            &ec.counter,
            &next,
        ];
        for values in arrays {
            fields.push(values.len() as i64);
            fields.extend_from_slice(values);
        }
        record(8, &fields);
        if RAW_CHECKPOINTS.with(Cell::get) {
            let (raw, cursor) = rng.state_snapshot();
            let mut observation = Vec::with_capacity(626);
            observation.push(i64::try_from(cursor).expect("SFMT cursor fits i64"));
            observation.push(i64::try_from(rng.words_consumed()).expect("draw count fits i64"));
            observation.extend(raw.map(i64::from));
            record(10, &observation);
        }
    });
}

#[cfg(test)]
mod tests {
    fn frame() -> crate::state::VmcOptimizationState {
        crate::state::VmcOptimizationState::zeros(2, 1, 0, 1, 1, 1, false, false)
    }

    #[test]
    fn raw_checkpoint_disabled_legacy_and_enabled_are_nonmutating() {
        let state = frame();
        let mut rng = sfmt19937::Sfmt19937Rng::new(17);
        rng.gen_rand32();
        let original = rng.state_snapshot();
        let count = rng.words_consumed();
        super::checkpoint(&state, &rng);
        assert!(super::EVENTS.with(|events| events.borrow().is_none()));
        super::start();
        super::checkpoint(&state, &rng);
        let legacy = super::finish();
        assert_eq!(legacy.len(), 1);
        assert_eq!(legacy[0][0], 8);
        super::start_with_raw_checkpoints();
        super::checkpoint(&state, &rng);
        let observed = super::finish();
        assert_eq!(observed[0], legacy[0]);
        assert_eq!(observed.len(), 2);
        assert_eq!(observed[1].len(), 627);
        assert_eq!(observed[1][0], 10);
        assert_eq!(observed[1][1], original.1 as i64);
        assert_eq!(observed[1][2], count as i64);
        assert_eq!(observed[1][3..], original.0.map(i64::from));
        assert_eq!(rng.state_snapshot(), original);
        assert_eq!(rng.words_consumed(), count);
        // checkpoint accepts state immutably; compare all saved planes explicitly.
        let fresh = frame();
        assert_eq!(state.electron_config.ele_idx, fresh.electron_config.ele_idx);
        assert_eq!(state.electron_config.ele_cfg, fresh.electron_config.ele_cfg);
        assert_eq!(state.electron_config.ele_num, fresh.electron_config.ele_num);
        assert_eq!(
            state.electron_config.ele_proj_cnt,
            fresh.electron_config.ele_proj_cnt
        );
        assert_eq!(state.electron_config.ele_spn, fresh.electron_config.ele_spn);
        assert_eq!(state.electron_config.counter, fresh.electron_config.counter);
    }

    #[test]
    fn nested_raw_request_cannot_change_legacy_contract_and_panic_can_be_harvested() {
        super::start();
        assert!(std::panic::catch_unwind(super::start_with_raw_checkpoints).is_err());
        assert!(!super::RAW_CHECKPOINTS.with(std::cell::Cell::get));
        assert!(super::finish().is_empty());
        super::start_with_raw_checkpoints();
        let panic = std::panic::catch_unwind(|| {
            super::checkpoint(&frame(), &sfmt19937::Sfmt19937Rng::new(1));
            panic!("test diagnostic panic");
        });
        assert!(panic.is_err());
        assert_eq!(super::finish().len(), 2);
        assert!(!super::RAW_CHECKPOINTS.with(std::cell::Cell::get));
        // Existing trace is NOT RAII. Explicit finish after caught panic is required.
        super::start();
        assert!(super::finish().is_empty());
    }
    #[test]
    fn observation_preserves_actual_candidate_decision_and_next624() {
        use crate::sampling::{get_update_type, make_candidate_hopping, metropolis_decision};
        use num_complex::Complex64;
        use sfmt19937::Sfmt19937Rng;
        fn run(rng: &mut Sfmt19937Rng) {
            get_update_type(0, 0, 0, rng);
            make_candidate_hopping(&[0, 2], &[0, -1, -1, -1, -1, -1, 0, -1], 4, 1, &[0; 4], rng);
            metropolis_decision(
                0.0,
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                rng,
            );
        }
        let mut baseline = Sfmt19937Rng::new(11272);
        let mut observed = baseline.clone();
        run(&mut baseline);
        super::start();
        run(&mut observed);
        let events = super::finish();
        let semantic: Vec<_> = events.iter().filter(|event| event[0] != 9).collect();
        assert_eq!(semantic.len(), 3);
        assert_eq!(semantic[0], &[0, 0]);
        assert_eq!(semantic[1][0], 1);
        assert_eq!(semantic[2][..2], [7, 1]);
        let mut before = [0_u32; 624];
        let mut after = [0_u32; 624];
        baseline.dump_rand32(&mut before);
        observed.dump_rand32(&mut after);
        assert_eq!(before, after);
    }

    #[test]
    fn capture_is_opt_in_ordered_and_thread_local() {
        super::record(0, &[99]);
        super::start();
        super::record(1, &[2, 3]);
        std::thread::spawn(|| super::record(1, &[99]))
            .join()
            .unwrap();
        super::record(7, &[0, 123]);
        assert_eq!(super::finish(), vec![vec![1, 2, 3], vec![7, 0, 123]]);
    }
}

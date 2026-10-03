//! Opt-in, observational per-thread sampling events for reference verification.
//! No RNG draws, numerical operations, or configuration mutations are added.

use std::cell::RefCell;

thread_local! {
    static EVENTS: RefCell<Option<Vec<Vec<i64>>>> = const { RefCell::new(None) };
}

/// Enable capture on the sampler's calling thread. Nested captures are errors.
pub fn start() {
    EVENTS.with(|slot| {
        let mut slot = slot.borrow_mut();
        assert!(slot.is_none(), "sampling trace already active");
        *slot = Some(Vec::new());
    });
}

/// End capture and return actual events, in execution order.
pub fn finish() -> Vec<Vec<i64>> {
    EVENTS.with(|slot| slot.borrow_mut().take().expect("sampling trace not active"))
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
    });
}

#[cfg(test)]
mod tests {
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

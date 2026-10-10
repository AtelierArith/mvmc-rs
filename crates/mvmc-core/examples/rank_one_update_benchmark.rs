//! Optional dispatch diagnostic; synthetic inputs, never full VMC benchmark timings.
use mvmc_core::{sampling::updates::update_m_all_real_flat, state::SlaterElmFlat};
use std::time::{Duration, Instant};

fn benchmark() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    assert_eq!(args.len(), 2, "usage: rank_one_update_benchmark SIZE REPS");
    let n: usize = args[0].parse().unwrap();
    let reps: usize = args[1].parse().unwrap();
    assert!(n >= 2 && n.is_multiple_of(2) && reps > 0);
    let (ne, nq, stride) = (n / 2, 8, n * n + 1);
    let mut slater = SlaterElmFlat::<f64>::zeros(nq, n);
    // Deterministic data, independent of the production SFMT stream. Reset the
    // inverse before each update so repeated calls do not accumulate conditioning.
    for (i, x) in slater.as_mut_slice().iter_mut().enumerate() {
        *x = ((i * 13 % 101) as f64 - 50.0) / 101.0;
    }
    let initial: Vec<f64> = (0..nq * stride)
        .map(|i| ((i * 17 % 97) as f64 - 48.0) / 97.0)
        .collect();
    let mut inverse = initial.clone();
    let mut pf = vec![1.0; nq];
    let mut idx: Vec<i64> = (0..n).map(|i| (i % ne) as i64).collect();
    idx[0] = (n - 1) as i64;
    let mut elapsed = Duration::ZERO;
    for iteration in 0..100 + reps {
        inverse.copy_from_slice(&initial);
        pf.fill(1.0);
        let start = Instant::now();
        update_m_all_real_flat(
            0,
            0,
            &idx,
            &slater,
            &mut inverse,
            stride,
            &mut pf,
            0,
            nq,
            n,
            ne,
        );
        let duration = start.elapsed();
        if iteration >= 100 {
            elapsed += duration;
        }
        std::hint::black_box((&inverse, &pf));
    }
    assert!(inverse.iter().chain(&pf).all(|x| x.is_finite()));
    println!(
        "SIZE {n} THREADS {} QPS {nq} REPS {reps} NS_PER_CALL {:.3}",
        mvmc_core::threading::inner_thread_config().threads,
        elapsed.as_nanos() as f64 / reps as f64,
    );
}

fn main() {
    if mvmc_core::threading::inner_thread_config().threads > 1 {
        mvmc_core::threading::install(benchmark);
    } else {
        benchmark();
    }
}

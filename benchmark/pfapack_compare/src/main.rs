use std::time::Instant;

use num_complex::Complex64;
use pfapack::{
    dsktf2, pfaffian_ltl_complex, pfaffian_ltl_real, utu2inv_complex, utu2inv_real,
    utu2pfa_complex, utu2pfa_real, zsktf2, PivotIndex1Based, SqMat,
};

fn next_f64(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let bits = (*s >> 32) as u32;
    (bits as f64 / u32::MAX as f64) * 2.0 - 1.0
}

fn skew_real(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed.wrapping_add(0xdeadbeef);
    let mut a = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..j {
            let v = next_f64(&mut s);
            a[j * n + i] = v;
            a[i * n + j] = -v;
        }
    }
    a
}

fn skew_complex(n: usize, seed: u64) -> Vec<Complex64> {
    let mut s = seed.wrapping_add(0xdeadbeef);
    let mut a = vec![Complex64::new(0.0, 0.0); n * n];
    for j in 0..n {
        for i in 0..j {
            let v = Complex64::new(next_f64(&mut s), next_f64(&mut s));
            a[j * n + i] = v;
            a[i * n + j] = -v;
        }
    }
    a
}

fn median_ms(mut samples: Vec<u128>) -> f64 {
    samples.sort_unstable();
    samples[samples.len() / 2] as f64 / 1e6
}

fn bench<F: FnMut()>(iters: usize, mut f: F) -> f64 {
    for _ in 0..3 {
        f();
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        samples.push(t0.elapsed().as_nanos());
    }
    median_ms(samples)
}

fn iters_for(n: usize) -> usize {
    match n {
        0..=64 => 200,
        65..=128 => 80,
        129..=256 => 25,
        _ => 8,
    }
}

fn print_result(kind: &str, n: usize, op: &str, t: f64) {
    println!("rust,{kind},{n},{op},{t:.6}");
}

fn run_benchmark() {
    for &n in &[32usize, 64, 128, 256] {
        let iters = iters_for(n);

        let orig = skew_real(n, 42);
        let mut a = vec![0.0; n * n];
        let t = bench(iters, || {
            a.copy_from_slice(&orig);
            let mut sm = SqMat::new(&mut a, n);
            std::hint::black_box(pfaffian_ltl_real(&mut sm));
        });
        print_result("real", n, "pfaffian_ltl", t);

        let mut a = vec![0.0; n * n];
        let mut piv = vec![PivotIndex1Based(0); n];
        let t = bench(iters, || {
            a.copy_from_slice(&orig);
            piv.fill(PivotIndex1Based(0));
            let mut sm = SqMat::new(&mut a, n);
            dsktf2(&mut sm, &mut piv).unwrap();
            std::hint::black_box(piv[0]);
        });
        print_result("real", n, "ltl", t);

        let mut a = vec![0.0; n * n];
        let mut piv = vec![PivotIndex1Based(0); n];
        let t = bench(iters, || {
            a.copy_from_slice(&orig);
            piv.fill(PivotIndex1Based(0));
            let mut sm = SqMat::new(&mut a, n);
            dsktf2(&mut sm, &mut piv).unwrap();
            std::hint::black_box(utu2pfa_real(&sm, &piv));
        });
        print_result("real", n, "ltl_utu2pfa", t);

        let mut a = vec![0.0; n * n];
        let mut piv = vec![PivotIndex1Based(0); n];
        let mut vt = vec![0.0; n - 1];
        let mut m_buf = vec![0.0; n * n];
        let t = bench(iters, || {
            a.copy_from_slice(&orig);
            piv.fill(PivotIndex1Based(0));
            {
                let mut sm = SqMat::new(&mut a, n);
                dsktf2(&mut sm, &mut piv).unwrap();
            }
            let mut sm = SqMat::new(&mut a, n);
            let mut mm = SqMat::new(&mut m_buf, n);
            utu2inv_real(&mut sm, &piv, &mut vt, &mut mm);
            std::hint::black_box(sm.get(0, 0));
        });
        print_result("real", n, "ltl_utu2inv", t);

        let orig = skew_complex(n, 42);
        let mut a = vec![Complex64::new(0.0, 0.0); n * n];
        let t = bench(iters, || {
            a.copy_from_slice(&orig);
            let mut sm = SqMat::new(&mut a, n);
            std::hint::black_box(pfaffian_ltl_complex(&mut sm));
        });
        print_result("complex", n, "pfaffian_ltl", t);

        let mut a = vec![Complex64::new(0.0, 0.0); n * n];
        let mut piv = vec![PivotIndex1Based(0); n];
        let t = bench(iters, || {
            a.copy_from_slice(&orig);
            piv.fill(PivotIndex1Based(0));
            let mut sm = SqMat::new(&mut a, n);
            zsktf2(&mut sm, &mut piv).unwrap();
            std::hint::black_box(piv[0]);
        });
        print_result("complex", n, "ltl", t);

        let mut a = vec![Complex64::new(0.0, 0.0); n * n];
        let mut piv = vec![PivotIndex1Based(0); n];
        let t = bench(iters, || {
            a.copy_from_slice(&orig);
            piv.fill(PivotIndex1Based(0));
            let mut sm = SqMat::new(&mut a, n);
            zsktf2(&mut sm, &mut piv).unwrap();
            std::hint::black_box(utu2pfa_complex(&sm, &piv));
        });
        print_result("complex", n, "ltl_utu2pfa", t);

        let mut a = vec![Complex64::new(0.0, 0.0); n * n];
        let mut piv = vec![PivotIndex1Based(0); n];
        let mut vt = vec![Complex64::new(0.0, 0.0); n - 1];
        let mut m_buf = vec![Complex64::new(0.0, 0.0); n * n];
        let t = bench(iters, || {
            a.copy_from_slice(&orig);
            piv.fill(PivotIndex1Based(0));
            {
                let mut sm = SqMat::new(&mut a, n);
                zsktf2(&mut sm, &mut piv).unwrap();
            }
            let mut sm = SqMat::new(&mut a, n);
            let mut mm = SqMat::new(&mut m_buf, n);
            utu2inv_complex(&mut sm, &piv, &mut vt, &mut mm);
            std::hint::black_box(sm.get(0, 0));
        });
        print_result("complex", n, "ltl_utu2inv", t);
    }
}

fn main() {
    println!("impl,kind,n,op,median_ms");
    run_benchmark();
    bench_transfer_green();
}

fn transfer_state() -> (
    mvmc_expert_parsers::ExpertModeData,
    mvmc_core::VmcOptimizationState,
    [i64; 2],
    [i64; 4],
    [i64; 4],
    [i64; 2],
) {
    let mut data = mvmc_expert_parsers::ExpertModeData::new();
    data.modpara.nsite = 2;
    data.modpara.nelec = 1;
    data.n_gutzwiller_idx = 2;
    data.gutzwiller_idx = vec![0, 1];
    data.gutzwiller_terms = vec![
        mvmc_expert_parsers::GutzwillerTerm {
            site: 0,
            value: Complex64::new(0.25, 0.0),
            is_complex: false,
        },
        mvmc_expert_parsers::GutzwillerTerm {
            site: 1,
            value: Complex64::new(-0.15, 0.0),
            is_complex: false,
        },
    ];
    data.transfer_terms.push(mvmc_expert_parsers::TransferTerm {
        site1: 1,
        spin1: mvmc_expert_parsers::Spin::Up,
        site2: 0,
        spin2: mvmc_expert_parsers::Spin::Up,
        value: Complex64::new(0.75, 0.0),
    });
    let mut state = mvmc_core::VmcOptimizationState::zeros(2, 1, 0, 0, 1, 1, false, false);
    state.slater_matrix.pf_m_real[0] = 1.0;
    state.slater_matrix.inv_m_real.as_mut_slice()[0] = 1.0;
    state.slater_matrix.slater_elm_real.set(0, 1, 0, 1.0);
    (data, state, [0, 0], [0, -1, -1, -1], [1, 0, 0, 0], [1, 0])
}

fn bench_transfer_green() {
    use mvmc_core::c_timer::CTimer;

    let iters = std::env::var("MVMC_RS_TRANSFER_BENCH_ITERS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(10_000usize);
    let (data_fast, mut fast_state, idx, cfg, num, counts) = transfer_state();
    let ip = Complex64::new(1.0, 0.0);
    let _ = mvmc_core::observables::calculate_local_energy(
        ip,
        &data_fast,
        &mut fast_state,
        &idx,
        &cfg,
        &num,
        &counts,
    );
    let (data_generic, mut generic_state, _, _, _, _) = transfer_state();

    let mut generic_samples = Vec::with_capacity(7);
    let mut fast_samples = Vec::with_capacity(7);
    for _ in 0..7 {
        let t0 = Instant::now();
        let mut checksum = Complex64::new(0.0, 0.0);
        for _ in 0..iters {
            checksum += mvmc_core::observables::green_func1(
                1,
                0,
                0,
                0,
                ip,
                &data_generic,
                &mut generic_state,
                &idx,
                &cfg,
                &num,
                &counts,
            );
        }
        std::hint::black_box(checksum);
        generic_samples.push(t0.elapsed().as_nanos());

        let t0 = Instant::now();
        let mut checksum = Complex64::new(0.0, 0.0);
        let mut timer = CTimer::<false>::new();
        for _ in 0..iters {
            checksum += mvmc_core::observables::green_func1_timed(
                1,
                0,
                0,
                0,
                ip,
                &data_fast,
                &mut fast_state,
                &idx,
                &cfg,
                &num,
                &counts,
                &mut timer,
            );
        }
        std::hint::black_box(checksum);
        fast_samples.push(t0.elapsed().as_nanos());
    }
    generic_samples.sort_unstable();
    fast_samples.sort_unstable();
    println!(
        "transfer_green,iterations={},generic_median_ms={:.6},fast_median_ms={:.6},speedup={:.3}",
        iters,
        generic_samples[3] as f64 / 1e6,
        fast_samples[3] as f64 / 1e6,
        generic_samples[3] as f64 / fast_samples[3] as f64
    );
}

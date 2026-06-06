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
}

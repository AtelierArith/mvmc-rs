//! SR stage micro-benchmark, C-order versus tenferro (issue #421).
//!
//! `cargo run --release -p mvmc-core --example sr_backend_bench -- [n_para ...]`
//!
//! Prints the median of 7 runs (after one warm-up) of each stage for `samples = 300` and
//! `samples = 2 n_para`, plus the maximum relative differences. Use
//! `OPENBLAS_NUM_THREADS`, `RAYON_NUM_THREADS` to fix threads on both sides; set
//! `TENFERRO_PROFILE_EAGER_OP_AGG=1` for tenferro's per-op aggregate profile on stderr.

use std::time::Instant;

use mvmc_core::sr_backend::{COrderSr, CgSamples, RealView, SrAssembleInput, SrStages, TenferroSr};

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn time<T>(mut f: impl FnMut() -> T) -> f64 {
    f();
    median(
        (0..7)
            .map(|_| {
                let t = Instant::now();
                std::hint::black_box(f());
                t.elapsed().as_secs_f64() * 1e3
            })
            .collect(),
    )
}

fn rel(a: &[f64], b: &[f64]) -> f64 {
    let scale = b.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    a.iter()
        .zip(b)
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()))
        / scale
}

fn run(n: usize, samples: usize) {
    let mut st = 7;
    let store: Vec<f64> = (0..n * samples).map(|_| lcg(&mut st)).collect();
    let mut c = COrderSr::default();
    let mut t = TenferroSr::new_cpu().expect("tenferro cpu");

    let mut oo_c = vec![0.0; n * n];
    let mut oo_t = vec![0.0; n * n];
    let g_c = time(|| c.gram_real(&store, n, samples, &mut oo_c).unwrap());
    let g_t = time(|| t.gram_real(&store, n, samples, &mut oo_t).unwrap());
    let gram_err = rel(&oo_t, &oo_c);

    // S/g from the Gram (all components active), then SPD by the regularization.
    let size = n + 1;
    let mut full = vec![0.0; size * size];
    for j in 0..n {
        for i in 0..n {
            full[(i + 1) + (j + 1) * size] = oo_c[i + j * n] / samples as f64;
        }
    }
    let ho: Vec<f64> = (0..size).map(|_| lcg(&mut st)).collect();
    let map: Vec<usize> = (0..n).collect();
    let input = SrAssembleInput {
        oo: RealView::Real(&full),
        ho: RealView::Real(&ho),
        map: &map,
        ld: size,
        offset: 1,
        sta_del: 0.02,
        step_dt: 0.05,
    };
    let (mut s_c, mut g_vec_c) = (vec![0.0; n * n], vec![0.0; n]);
    let (mut s_t, mut g_vec_t) = (vec![0.0; n * n], vec![0.0; n]);
    let a_c = time(|| c.assemble_s_g(&input, &mut s_c, &mut g_vec_c).unwrap());
    let a_t = time(|| t.assemble_s_g(&input, &mut s_t, &mut g_vec_t).unwrap());
    let asm_err = rel(&s_t, &s_c).max(rel(&g_vec_t, &g_vec_c));

    let (mut x_c, mut x_t) = (g_vec_c.clone(), g_vec_t.clone());
    let s_c_keep = s_c.clone();
    let solve_c = time(|| {
        let (mut s, mut x) = (s_c_keep.clone(), g_vec_c.clone());
        c.cholesky_solve(&mut s, &mut x, n).unwrap();
        x_c.copy_from_slice(&x);
    });
    let solve_t = time(|| {
        let (mut s, mut x) = (s_c_keep.clone(), g_vec_c.clone());
        t.cholesky_solve(&mut s, &mut x, n).unwrap();
        x_t.copy_from_slice(&x);
    });
    let solve_err = rel(&x_t, &x_c);

    let x: Vec<f64> = (0..n).map(|_| lcg(&mut st)).collect();
    let view = CgSamples {
        real: &store,
        imag: &[],
        components: n,
        samples,
        version: 1,
    };
    let (mut z_c, mut z_t) = (vec![0.0; n], vec![0.0; n]);
    let cg_c = time(|| c.cg_local_product(&view, &x, &mut z_c).unwrap());
    let cg_t = time(|| t.cg_local_product(&view, &x, &mut z_t).unwrap());
    let cg_err = rel(&z_t, &z_c);

    println!(
        "| {n} | {samples} | {g_c:.2} / {g_t:.2} | {a_c:.2} / {a_t:.2} | {solve_c:.2} / {solve_t:.2} | {cg_c:.3} / {cg_t:.3} | {gram_err:.1e} {asm_err:.1e} {solve_err:.1e} {cg_err:.1e} |"
    );
}

fn main() {
    let sizes: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let sizes = if sizes.is_empty() {
        vec![100, 388, 1000, 3000]
    } else {
        sizes
    };
    println!("ms, median of 7, C-order / tenferro (cpu-faer)");
    println!("| NPara | samples | Gram | S,g assembly | Cholesky solve | CG matvec | rel diff gram,asm,solve,cg |");
    println!("|---:|---:|---|---|---|---|---|");
    for n in sizes {
        run(n, 300);
        run(n, 2 * n);
    }
}

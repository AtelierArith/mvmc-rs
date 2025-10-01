//! Benchmarks for linear algebra operations
//!
//! This module provides benchmarks for various linear algebra operations
//! to measure performance and compare with C implementation.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ndarray::Array2;
use num_complex::Complex64;
use mvmc_math::linear_algebra::ComplexMatrix;
use mvmc_bindings::wrappers::linear_algebra::{
    solve_linear_system, lu_decomposition, invert_matrix,
    eigenvalue_decomposition, singular_value_decomposition,
};

/// Generate a random complex matrix for benchmarking
fn generate_random_matrix(size: usize) -> Array2<Complex64> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let mut data = Vec::with_capacity(size * size);
    for _ in 0..size * size {
        let re = rng.gen_range(-10.0..10.0);
        let im = rng.gen_range(-10.0..10.0);
        data.push(Complex64::new(re, im));
    }

    Array2::from_shape_vec((size, size), data).unwrap()
}

/// Generate a random complex vector for benchmarking
fn generate_random_vector(size: usize) -> Array2<Complex64> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let mut data = Vec::with_capacity(size);
    for _ in 0..size {
        let re = rng.gen_range(-10.0..10.0);
        let im = rng.gen_range(-10.0..10.0);
        data.push(Complex64::new(re, im));
    }

    Array2::from_shape_vec((size, 1), data).unwrap()
}

/// Benchmark matrix multiplication
fn bench_matrix_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_multiplication");
    group.sample_size(10); // サンプル数を削減

    for size in [10, 50, 100].iter() { // 200を削除
        let a = generate_random_matrix(*size);
        let b = generate_random_matrix(*size);

        group.bench_with_input(BenchmarkId::new("ndarray", size), size, |bencher, _| {
            bencher.iter(|| {
                let result = a.dot(&b);
                black_box(result);
            });
        });

        let matrix_a = ComplexMatrix::new(a.clone());
        let matrix_b = ComplexMatrix::new(b.clone());

        group.bench_with_input(BenchmarkId::new("mvmc_math", size), size, |bencher, _| {
            bencher.iter(|| {
                let result = matrix_a.clone() * matrix_b.clone();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark LU decomposition
fn bench_lu_decomposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("lu_decomposition");
    group.sample_size(10); // サンプル数を削減

    for size in [10, 50, 100].iter() { // 200を削除
        let mut matrix = generate_random_matrix(*size);

        group.bench_with_input(BenchmarkId::new("mvmc_math", size), size, |bencher, _| {
            bencher.iter(|| {
                let matrix_clone = matrix.clone();
                let result = ComplexMatrix::new(matrix_clone).lu_decomposition();
                black_box(result);
            });
        });

        group.bench_with_input(BenchmarkId::new("mvmc_bindings", size), size, |bencher, _| {
            bencher.iter(|| {
                let mut matrix_clone = matrix.clone();
                let result = lu_decomposition(&mut matrix_clone);
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark matrix inversion
fn bench_matrix_inversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_inversion");
    group.sample_size(10); // サンプル数を削減

    for size in [10, 50].iter() { // 100を削除
        let mut matrix = generate_random_matrix(*size);

        group.bench_with_input(BenchmarkId::new("mvmc_math", size), size, |bencher, _| {
            bencher.iter(|| {
                let matrix_clone = matrix.clone();
                let result = ComplexMatrix::new(matrix_clone).lu_decomposition();
                black_box(result);
            });
        });

        group.bench_with_input(BenchmarkId::new("mvmc_bindings", size), size, |bencher, _| {
            bencher.iter(|| {
                let mut matrix_clone = matrix.clone();
                let result = invert_matrix(&mut matrix_clone);
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark eigenvalue decomposition
fn bench_eigenvalue_decomposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("eigenvalue_decomposition");
    group.sample_size(10); // サンプル数を削減

    for size in [10, 50].iter() { // 100を削除
        let mut matrix = generate_random_matrix(*size);

        group.bench_with_input(BenchmarkId::new("mvmc_bindings", size), size, |bencher, _| {
            bencher.iter(|| {
                let mut matrix_clone = matrix.clone();
                let result = eigenvalue_decomposition(&mut matrix_clone, true, true);
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark singular value decomposition
fn bench_singular_value_decomposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("singular_value_decomposition");
    group.sample_size(10); // サンプル数を削減

    for size in [10, 50].iter() { // 100を削除
        let mut matrix = generate_random_matrix(*size);

        group.bench_with_input(BenchmarkId::new("mvmc_bindings", size), size, |bencher, _| {
            bencher.iter(|| {
                let mut matrix_clone = matrix.clone();
                let result = singular_value_decomposition(&mut matrix_clone);
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark linear system solving
fn bench_linear_system_solving(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_system_solving");
    group.sample_size(10); // サンプル数を削減

    for size in [10, 50, 100].iter() { // 200を削除
        let mut a = generate_random_matrix(*size);
        let mut b = generate_random_vector(*size);

        group.bench_with_input(BenchmarkId::new("mvmc_bindings", size), size, |bencher, _| {
            bencher.iter(|| {
                let mut a_clone = a.clone();
                let mut b_clone = b.clone();
                let result = solve_linear_system(&mut a_clone, &mut b_clone);
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark memory usage
fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");
    group.sample_size(5); // サンプル数を大幅削減

    for size in [100, 500].iter() { // 1000, 2000を削除
        group.bench_with_input(BenchmarkId::new("matrix_allocation", size), size, |bencher, _| {
            bencher.iter(|| {
                let matrix = generate_random_matrix(*size);
                black_box(matrix);
            });
        });

        group.bench_with_input(BenchmarkId::new("matrix_operations", size), size, |bencher, _| {
            bencher.iter(|| {
                let a = generate_random_matrix(*size);
                let b = generate_random_matrix(*size);
                let result = a.dot(&b);
                black_box(result);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_matrix_multiplication,
    bench_lu_decomposition,
    bench_linear_system_solving
    // 時間のかかるベンチマークは一時的に除外
    // bench_matrix_inversion,
    // bench_eigenvalue_decomposition,
    // bench_singular_value_decomposition,
    // bench_memory_usage
);

criterion_main!(benches);

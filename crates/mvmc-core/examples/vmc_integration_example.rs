//! VMC Integration Test Example
//!
//! This example demonstrates how to run comprehensive integration tests
//! for the VMC system.

use mvmc_core::vmc::integration_test::{VmcIntegrationTest, VmcBenchmark};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== VMC Integration Test Example ===");

    // Run integration tests
    println!("1. Running Integration Tests");
    let test_suite = VmcIntegrationTest::new();
    let results = test_suite.run_all_tests()?;
    results.print_summary();

    // Run performance benchmark
    println!("2. Running Performance Benchmark");
    let benchmark = VmcBenchmark::new();
    let benchmark_results = benchmark.run_benchmark()?;
    benchmark_results.print_summary();

    println!();
    println!("=== Example completed successfully ===");

    Ok(())
}

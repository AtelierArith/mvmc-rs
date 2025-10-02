//! Comprehensive VMC Test Suite
//!
//! This example demonstrates comprehensive testing of the VMC system
//! including all implementations and various physical models.

use mvmc_core::vmc::{
    SimpleHeisenbergVMC,
    ImprovedHeisenbergVMC,
    AdaptiveHeisenbergVMC,
    VmcIntegrationTest,
    VmcBenchmark
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Comprehensive VMC Test Suite ===");

    // Test 1: Basic functionality tests
    println!("\n1. Basic Functionality Tests");
    test_basic_functionality()?;

    // Test 2: Energy calculation accuracy
    println!("\n2. Energy Calculation Accuracy Tests");
    test_energy_accuracy()?;

    // Test 3: Monte Carlo sampling efficiency
    println!("\n3. Monte Carlo Sampling Efficiency Tests");
    test_sampling_efficiency()?;

    // Test 4: Temperature dependence
    println!("\n4. Temperature Dependence Tests");
    test_temperature_dependence()?;

    // Test 5: System size scaling
    println!("\n5. System Size Scaling Tests");
    test_system_scaling()?;

    // Test 6: Integration test suite
    println!("\n6. Integration Test Suite");
    let test_suite = VmcIntegrationTest::new();
    let results = test_suite.run_all_tests()?;
    results.print_summary();

    // Test 7: Performance benchmark
    println!("\n7. Performance Benchmark");
    let benchmark = VmcBenchmark::new();
    let benchmark_results = benchmark.run_benchmark()?;
    benchmark_results.print_summary();

    println!("\n=== All tests completed successfully ===");

    Ok(())
}

fn test_basic_functionality() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Testing basic VMC functionality...");

    // Test Simple VMC
    let vmc = SimpleHeisenbergVMC::new(4, 0, 1.0)?;
    let config = vmc.generate_simple_config();
    let energy = vmc.calculate_energy(&config);
    println!("    Simple VMC: Energy = {:.6}", energy);

    // Test Improved VMC
    let mut improved_vmc = ImprovedHeisenbergVMC::new(4, 0, 1.0, 1.0)?;
    let config = improved_vmc.generate_random_config();
    let energy = improved_vmc.calculate_energy(&config);
    println!("    Improved VMC: Energy = {:.6}", energy);

    // Test Adaptive VMC
    let mut adaptive_vmc = AdaptiveHeisenbergVMC::new(4, 0, 1.0, 1.0, 0.5)?;
    let config = adaptive_vmc.generate_random_config();
    let energy = adaptive_vmc.calculate_energy(&config);
    println!("    Adaptive VMC: Energy = {:.6}", energy);

    Ok(())
}

fn test_energy_accuracy() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Testing energy calculation accuracy...");

    let test_cases = vec![
        (4, 0, 1.0, "4-site chain, Sz=0"),
        (6, 0, 1.0, "6-site chain, Sz=0"),
        (8, 0, 1.0, "8-site chain, Sz=0"),
        (4, 2, 1.0, "4-site chain, Sz=1"),
        (4, 0, -1.0, "4-site chain, FM coupling"),
    ];

    for (nsite, two_sz, exchange, description) in test_cases {
        println!("    Testing {}...", description);

        // Test Simple VMC
        let vmc = SimpleHeisenbergVMC::new(nsite, two_sz, exchange)?;
        let config = vmc.generate_simple_config();
        let energy = vmc.calculate_energy(&config);
        println!("      Simple VMC: {:.6}", energy);

        // Test Improved VMC
        let mut improved_vmc = ImprovedHeisenbergVMC::new(nsite, two_sz, exchange, 1.0)?;
        let stats = improved_vmc.run_vmc(100, 200)?;
        println!("      Improved VMC: {:.6} ± {:.6}", stats.average_energy, stats.variance.sqrt());

        // Test Adaptive VMC
        let mut adaptive_vmc = AdaptiveHeisenbergVMC::new(nsite, two_sz, exchange, 1.0, 0.5)?;
        let stats = adaptive_vmc.run_adaptive_vmc(300)?;
        println!("      Adaptive VMC: {:.6} ± {:.6}", stats.average_energy, stats.variance.sqrt());
    }

    Ok(())
}

fn test_sampling_efficiency() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Testing Monte Carlo sampling efficiency...");

    let nsite = 6;
    let two_sz = 0;
    let exchange = 1.0;

    // Test Simple VMC
    let vmc = SimpleHeisenbergVMC::new(nsite, two_sz, exchange)?;
    let (avg_energy, variance, acceptance_rate) = vmc.run_vmc(1000, 12345)?;
    println!("    Simple VMC: E={:.6}, Var={:.6}, Acc={:.3}", avg_energy, variance, acceptance_rate);

    // Test Improved VMC
    let mut improved_vmc = ImprovedHeisenbergVMC::new(nsite, two_sz, exchange, 1.0)?;
    let stats = improved_vmc.run_vmc(200, 800)?;
    println!("    Improved VMC: E={:.6}, Var={:.6}, Acc={:.3}",
        stats.average_energy, stats.variance, stats.acceptance_rate);

    // Test Adaptive VMC
    let mut adaptive_vmc = AdaptiveHeisenbergVMC::new(nsite, two_sz, exchange, 1.0, 0.5)?;
    let stats = adaptive_vmc.run_adaptive_vmc(1000)?;
    println!("    Adaptive VMC: E={:.6}, Var={:.6}, Acc={:.3}",
        stats.average_energy, stats.variance, stats.final_acceptance_rate);

    Ok(())
}

fn test_temperature_dependence() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Testing temperature dependence...");

    let nsite = 4;
    let two_sz = 0;
    let exchange = 1.0;
    let temperatures = vec![0.1, 0.5, 1.0, 2.0, 5.0];

    println!("    Temperature | Simple VMC | Improved VMC | Adaptive VMC");
    println!("    ------------|------------|--------------|-------------");

    for temp in temperatures {
        // Simple VMC (no temperature control)
        let vmc = SimpleHeisenbergVMC::new(nsite, two_sz, exchange)?;
        let (simple_energy, _, _) = vmc.run_vmc(200, 12345)?;

        // Improved VMC
        let mut improved_vmc = ImprovedHeisenbergVMC::new(nsite, two_sz, exchange, temp)?;
        let improved_stats = improved_vmc.run_vmc(50, 150)?;

        // Adaptive VMC
        let mut adaptive_vmc = AdaptiveHeisenbergVMC::new(nsite, two_sz, exchange, temp, 0.5)?;
        let adaptive_stats = adaptive_vmc.run_adaptive_vmc(200)?;

        println!("    {:>10.1} | {:>10.6} | {:>12.6} | {:>11.6}",
            temp, simple_energy, improved_stats.average_energy, adaptive_stats.average_energy);
    }

    Ok(())
}

fn test_system_scaling() -> Result<(), Box<dyn std::error::Error>> {
    println!("  Testing system size scaling...");

    let system_sizes = vec![4, 6, 8, 10, 12];
    let two_sz = 0;
    let exchange = 1.0;
    let temperature = 1.0;

    println!("    Size | Simple VMC | Improved VMC | Adaptive VMC");
    println!("    -----|------------|--------------|-------------");

    for nsite in system_sizes {
        // Simple VMC
        let vmc = SimpleHeisenbergVMC::new(nsite, two_sz, exchange)?;
        let (simple_energy, _, _) = vmc.run_vmc(200, 12345)?;

        // Improved VMC
        let mut improved_vmc = ImprovedHeisenbergVMC::new(nsite, two_sz, exchange, temperature)?;
        let improved_stats = improved_vmc.run_vmc(50, 150)?;

        // Adaptive VMC
        let mut adaptive_vmc = AdaptiveHeisenbergVMC::new(nsite, two_sz, exchange, temperature, 0.5)?;
        let adaptive_stats = adaptive_vmc.run_adaptive_vmc(200)?;

        println!("    {:>4} | {:>10.6} | {:>12.6} | {:>11.6}",
            nsite, simple_energy, improved_stats.average_energy, adaptive_stats.average_energy);
    }

    Ok(())
}

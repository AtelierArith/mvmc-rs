//! Improved Heisenberg VMC test
//!
//! This tests the improved Heisenberg VMC implementation with better
//! Monte Carlo sampling techniques.

use mvmc_core::vmc::{ImprovedHeisenbergVMC, VmcStatistics, MultipleVmcStatistics};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Improved Heisenberg VMC Test ===");

    // Test parameters
    let nsite = 4;
    let two_sz = 0;
    let exchange = 1.0;
    let temperature = 1.0;

    println!("Parameters:");
    println!("  Sites: {}", nsite);
    println!("  2*Sz: {}", two_sz);
    println!("  Exchange coupling J: {}", exchange);
    println!("  Temperature T: {}", temperature);
    println!();

    // Create improved VMC calculator
    let mut vmc = ImprovedHeisenbergVMC::new(nsite, two_sz, exchange, temperature)?;

    // Test 1: Single VMC run
    println!("Test 1: Single VMC run");
    let stats = vmc.run_vmc(200, 1000)?;

    println!("Results:");
    println!("  Average energy: {:.6}", stats.average_energy);
    println!("  Variance: {:.6}", stats.variance);
    println!("  Acceptance rate: {:.3}", stats.acceptance_rate);
    println!("  Autocorrelation time: {:.3}", stats.autocorrelation_time);
    println!("  Final configuration: {:?}", stats.final_configuration);
    println!();

    // Test 2: Multiple VMC runs for error estimation
    println!("Test 2: Multiple VMC runs for error estimation");
    let multi_stats = vmc.run_multiple_vmc(5, 100, 500)?;

    println!("Results across {} runs:", multi_stats.individual_energies.len());
    println!("  Mean energy: {:.6} ± {:.6}", multi_stats.mean_energy, multi_stats.energy_error);
    println!("  Mean acceptance rate: {:.3}", multi_stats.mean_acceptance_rate);
    println!("  Individual energies: {:?}", multi_stats.individual_energies);
    println!();

    // Test 3: Different temperatures
    println!("Test 3: Different temperatures");
    let temperatures = vec![0.5, 1.0, 2.0, 5.0];

    for temp in temperatures {
        println!("  Temperature T = {}", temp);
        let mut vmc_temp = ImprovedHeisenbergVMC::new(nsite, two_sz, exchange, temp)?;
        let stats_temp = vmc_temp.run_vmc(100, 300)?;

        println!("    Average energy: {:.6}", stats_temp.average_energy);
        println!("    Acceptance rate: {:.3}", stats_temp.acceptance_rate);
    }
    println!();

    // Test 4: Different system sizes
    println!("Test 4: Different system sizes");
    let system_sizes = vec![4, 6, 8];

    for size in system_sizes {
        println!("  System size N = {}", size);
        let mut vmc_size = ImprovedHeisenbergVMC::new(size, 0, 1.0, 1.0)?;
        let stats_size = vmc_size.run_vmc(100, 300)?;

        println!("    Average energy: {:.6}", stats_size.average_energy);
        println!("    Acceptance rate: {:.3}", stats_size.acceptance_rate);
    }
    println!();

    // Test 5: Energy vs temperature analysis
    println!("Test 5: Energy vs temperature analysis");
    let temp_range = vec![0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0];

    println!("  Temperature  |  Average Energy  |  Acceptance Rate");
    println!("  ------------|------------------|------------------");

    for temp in temp_range {
        let mut vmc_temp = ImprovedHeisenbergVMC::new(nsite, two_sz, exchange, temp)?;
        let stats_temp = vmc_temp.run_vmc(100, 300)?;

        println!("  {:>10.1}  |  {:>14.6}  |  {:>14.3}",
            temp, stats_temp.average_energy, stats_temp.acceptance_rate);
    }

    println!();
    println!("=== Test completed successfully ===");

    Ok(())
}

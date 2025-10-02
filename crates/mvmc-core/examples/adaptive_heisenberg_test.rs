//! Adaptive Heisenberg VMC test
//!
//! This tests the adaptive Heisenberg VMC implementation with
//! self-adjusting Monte Carlo parameters.

use mvmc_core::vmc::adaptive_heisenberg::{AdaptiveHeisenbergVMC, AdaptiveVmcStatistics};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Adaptive Heisenberg VMC Test ===");

    // Test parameters
    let nsite = 4;
    let two_sz = 0;
    let exchange = 1.0;
    let initial_temperature = 1.0;
    let target_acceptance_rate = 0.5;

    println!("Parameters:");
    println!("  Sites: {}", nsite);
    println!("  2*Sz: {}", two_sz);
    println!("  Exchange coupling J: {}", exchange);
    println!("  Initial temperature T: {}", initial_temperature);
    println!("  Target acceptance rate: {}", target_acceptance_rate);
    println!();

    // Test 1: Single adaptive VMC run
    println!("Test 1: Single adaptive VMC run");
    let mut vmc = AdaptiveHeisenbergVMC::new(nsite, two_sz, exchange, initial_temperature, target_acceptance_rate)?;
    let stats = vmc.run_adaptive_vmc(1000)?;

    println!("Results:");
    println!("  Average energy: {:.6}", stats.average_energy);
    println!("  Variance: {:.6}", stats.variance);
    println!("  Final acceptance rate: {:.3}", stats.final_acceptance_rate);
    println!("  Final temperature: {:.3}", stats.final_temperature);
    println!("  Autocorrelation time: {:.3}", stats.autocorrelation_time);
    println!("  Final configuration: {:?}", stats.final_configuration);
    println!();

    // Test 2: Different target acceptance rates
    println!("Test 2: Different target acceptance rates");
    let target_rates = vec![0.3, 0.5, 0.7];

    for target_rate in target_rates {
        println!("  Target acceptance rate: {}", target_rate);
        let mut vmc = AdaptiveHeisenbergVMC::new(nsite, two_sz, exchange, initial_temperature, target_rate)?;
        let stats = vmc.run_adaptive_vmc(500)?;

        println!("    Final acceptance rate: {:.3}", stats.final_acceptance_rate);
        println!("    Final temperature: {:.3}", stats.final_temperature);
        println!("    Average energy: {:.6}", stats.average_energy);
    }
    println!();

    // Test 3: Different initial temperatures
    println!("Test 3: Different initial temperatures");
    let initial_temps = vec![0.1, 1.0, 10.0];

    for initial_temp in initial_temps {
        println!("  Initial temperature: {}", initial_temp);
        let mut vmc = AdaptiveHeisenbergVMC::new(nsite, two_sz, exchange, initial_temp, target_acceptance_rate)?;
        let stats = vmc.run_adaptive_vmc(500)?;

        println!("    Final temperature: {:.3}", stats.final_temperature);
        println!("    Final acceptance rate: {:.3}", stats.final_acceptance_rate);
        println!("    Average energy: {:.6}", stats.average_energy);
    }
    println!();

    // Test 4: Different system sizes
    println!("Test 4: Different system sizes");
    let system_sizes = vec![4, 6, 8];

    for size in system_sizes {
        println!("  System size N = {}", size);
        let mut vmc = AdaptiveHeisenbergVMC::new(size, 0, 1.0, 1.0, 0.5)?;
        let stats = vmc.run_adaptive_vmc(500)?;

        println!("    Final temperature: {:.3}", stats.final_temperature);
        println!("    Final acceptance rate: {:.3}", stats.final_acceptance_rate);
        println!("    Average energy: {:.6}", stats.average_energy);
    }
    println!();

    // Test 5: Temperature adaptation analysis
    println!("Test 5: Temperature adaptation analysis");
    let mut vmc = AdaptiveHeisenbergVMC::new(nsite, two_sz, exchange, 10.0, 0.5)?;
    let stats = vmc.run_adaptive_vmc(1000)?;

    println!("  Initial temperature: 10.0");
    println!("  Final temperature: {:.3}", stats.final_temperature);
    println!("  Temperature adaptation successful: {}",
        (stats.final_temperature - 10.0).abs() > 1.0);
    println!("  Final acceptance rate: {:.3}", stats.final_acceptance_rate);
    println!("  Target acceptance rate: {:.3}", target_acceptance_rate);
    println!("  Acceptance rate convergence: {}",
        (stats.final_acceptance_rate - target_acceptance_rate).abs() < 0.1);

    println!();
    println!("=== Test completed successfully ===");

    Ok(())
}

//! Simple Heisenberg VMC test
//!
//! This is a minimal test program to verify that our simple Heisenberg VMC
//! implementation works correctly.

use mvmc_core::vmc::SimpleHeisenbergVMC;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Simple Heisenberg VMC Test ===");

    // Test parameters
    let nsite = 4;
    let two_sz = 0;  // Equal number of up and down spins
    let exchange = 1.0;  // Antiferromagnetic coupling

    println!("Parameters:");
    println!("  Sites: {}", nsite);
    println!("  2*Sz: {}", two_sz);
    println!("  Exchange coupling J: {}", exchange);
    println!();

    // Create VMC calculator
    let vmc = SimpleHeisenbergVMC::new(nsite, two_sz, exchange)?;

    // Test 1: Simple alternating configuration
    println!("Test 1: Simple alternating configuration");
    let config = vmc.generate_simple_config();
    println!("  Configuration: {:?}", config);
    let energy = vmc.calculate_energy(&config);
    println!("  Energy: {:.6}", energy);
    println!();

    // Test 2: Random configuration
    println!("Test 2: Random configuration");
    let config = vmc.generate_random_config(12345);
    println!("  Configuration: {:?}", config);
    let energy = vmc.calculate_energy(&config);
    println!("  Energy: {:.6}", energy);
    println!();

    // Test 3: Monte Carlo sampling
    println!("Test 3: Monte Carlo sampling");
    let (avg_energy, variance, acceptance_rate) = vmc.run_vmc(1000, 12345)?;
    println!("  Average energy: {:.6}", avg_energy);
    println!("  Variance: {:.6}", variance);
    println!("  Acceptance rate: {:.3}", acceptance_rate);
    println!();

    // Test 4: Multiple random seeds
    println!("Test 4: Multiple random seeds");
    for seed in 0..5 {
        let config = vmc.generate_random_config(seed);
        let energy = vmc.calculate_energy(&config);
        println!("  Seed {}: {:?} -> Energy: {:.6}", seed, config, energy);
    }

    println!();
    println!("=== Test completed successfully ===");

    Ok(())
}

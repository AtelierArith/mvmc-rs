//! Larger Heisenberg VMC test
//!
//! This tests the Heisenberg VMC implementation on larger systems
//! to verify scalability and correctness.

use mvmc_core::vmc::SimpleHeisenbergVMC;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Larger Heisenberg VMC Test ===");

    // Test different system sizes
    let test_cases = vec![
        (4, 0, 1.0, "4-site chain, Sz=0"),
        (6, 0, 1.0, "6-site chain, Sz=0"),
        (8, 0, 1.0, "8-site chain, Sz=0"),
        (4, 2, 1.0, "4-site chain, Sz=1"),
        (6, 2, 1.0, "6-site chain, Sz=1"),
    ];

    for (nsite, two_sz, exchange, description) in test_cases {
        println!("\n--- {} ---", description);
        println!("Parameters: nsite={}, 2*Sz={}, J={}", nsite, two_sz, exchange);

        // Create VMC calculator
        let vmc = SimpleHeisenbergVMC::new(nsite, two_sz, exchange)?;

        // Test 1: Simple alternating configuration
        let config = vmc.generate_simple_config();
        let energy = vmc.calculate_energy(&config);
        println!("  Alternating config: {:?}", config);
        println!("  Energy: {:.6}", energy);

        // Test 2: Random configuration
        let config = vmc.generate_random_config(12345);
        let energy = vmc.calculate_energy(&config);
        println!("  Random config: {:?}", config);
        println!("  Energy: {:.6}", energy);

        // Test 3: Monte Carlo sampling (fewer steps for larger systems)
        let num_steps = if nsite <= 4 { 1000 } else { 500 };
        let (avg_energy, variance, acceptance_rate) = vmc.run_vmc(num_steps, 12345)?;
        println!("  MC sampling ({} steps):", num_steps);
        println!("    Average energy: {:.6}", avg_energy);
        println!("    Variance: {:.6}", variance);
        println!("    Acceptance rate: {:.3}", acceptance_rate);
    }

    println!("\n=== Test completed successfully ===");

    Ok(())
}

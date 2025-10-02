//! Integration tests for VMC calculations
//!
//! This module provides comprehensive integration tests for the VMC system,
//! testing the complete workflow from configuration to results.

use mvmc_core::vmc::{
    SimpleHeisenbergVMC,
    ImprovedHeisenbergVMC,
    AdaptiveHeisenbergVMC,
    VmcStatistics,
    MultipleVmcStatistics,
    AdaptiveVmcStatistics
};
use mvmc_physics::hamiltonian::Spin;
use std::collections::HashMap;

/// Integration test suite for VMC calculations
#[derive(Debug)]
pub struct VmcIntegrationTest {
    test_cases: Vec<VmcTestCase>,
}

/// Individual test case for VMC calculation
#[derive(Debug, Clone)]
pub struct VmcTestCase {
    pub name: String,
    pub nsite: usize,
    pub two_sz: i32,
    pub exchange: f64,
    pub temperature: f64,
    pub expected_energy_range: (f64, f64),
    pub expected_acceptance_range: (f64, f64),
}

impl VmcIntegrationTest {
    /// Creates a new integration test suite
    pub fn new() -> Self {
        Self {
            test_cases: vec![
                VmcTestCase {
                    name: "4-site chain, Sz=0, J=1.0".to_string(),
                    nsite: 4,
                    two_sz: 0,
                    exchange: 1.0,
                    temperature: 1.0,
                    expected_energy_range: (-5.0, -2.0),
                    expected_acceptance_range: (0.3, 0.8),
                },
                VmcTestCase {
                    name: "6-site chain, Sz=0, J=1.0".to_string(),
                    nsite: 6,
                    two_sz: 0,
                    exchange: 1.0,
                    temperature: 1.0,
                    expected_energy_range: (-8.0, -4.0),
                    expected_acceptance_range: (0.3, 0.8),
                },
                VmcTestCase {
                    name: "8-site chain, Sz=0, J=1.0".to_string(),
                    nsite: 8,
                    two_sz: 0,
                    exchange: 1.0,
                    temperature: 1.0,
                    expected_energy_range: (-10.0, -6.0),
                    expected_acceptance_range: (0.3, 0.8),
                },
                VmcTestCase {
                    name: "4-site chain, Sz=1, J=1.0".to_string(),
                    nsite: 4,
                    two_sz: 1,
                    exchange: 1.0,
                    temperature: 1.0,
                    expected_energy_range: (-2.0, 2.0),
                    expected_acceptance_range: (0.3, 0.8),
                },
                VmcTestCase {
                    name: "4-site chain, Sz=0, J=-1.0 (FM)".to_string(),
                    nsite: 4,
                    two_sz: 0,
                    exchange: -1.0,
                    temperature: 1.0,
                    expected_energy_range: (-4.0, -2.0),
                    expected_acceptance_range: (0.3, 0.8),
                },
            ],
        }
    }

    /// Runs all integration tests
    pub fn run_all_tests(&self) -> Result<IntegrationTestResults, Box<dyn std::error::Error>> {
        let mut results = IntegrationTestResults::new();

        println!("=== VMC Integration Test Suite ===");
        println!("Running {} test cases...", self.test_cases.len());
        println!();

        for (i, test_case) in self.test_cases.iter().enumerate() {
            println!("Test {}/{}: {}", i + 1, self.test_cases.len(), test_case.name);

            // Test Simple VMC
            let simple_result = self.test_simple_vmc(test_case)?;
            results.add_result("Simple VMC", test_case.name.clone(), simple_result);

            // Test Improved VMC
            let improved_result = self.test_improved_vmc(test_case)?;
            results.add_result("Improved VMC", test_case.name.clone(), improved_result);

            // Test Adaptive VMC
            let adaptive_result = self.test_adaptive_vmc(test_case)?;
            results.add_result("Adaptive VMC", test_case.name.clone(), adaptive_result);

            println!();
        }

        Ok(results)
    }

    /// Tests Simple VMC implementation
    fn test_simple_vmc(&self, test_case: &VmcTestCase) -> Result<TestResult, Box<dyn std::error::Error>> {
        let vmc = SimpleHeisenbergVMC::new(test_case.nsite, test_case.two_sz, test_case.exchange)?;

        // Test basic functionality
        let config = vmc.generate_simple_config();
        let energy = vmc.calculate_energy(&config);

        // Test Monte Carlo sampling
        let (avg_energy, variance, acceptance_rate) = vmc.run_vmc(500, 12345)?;

        let success = self.validate_results(
            avg_energy,
            acceptance_rate,
            test_case.expected_energy_range,
            test_case.expected_acceptance_range
        );

        Ok(TestResult {
            success,
            average_energy: avg_energy,
            variance,
            acceptance_rate,
            error_message: if success { None } else {
                Some(format!("Energy {:.3} not in range {:?}, Acceptance {:.3} not in range {:?}",
                    avg_energy, test_case.expected_energy_range,
                    acceptance_rate, test_case.expected_acceptance_range))
            },
        })
    }

    /// Tests Improved VMC implementation
    fn test_improved_vmc(&self, test_case: &VmcTestCase) -> Result<TestResult, Box<dyn std::error::Error>> {
        let mut vmc = ImprovedHeisenbergVMC::new(test_case.nsite, test_case.two_sz, test_case.exchange, test_case.temperature)?;

        // Test basic functionality
        let config = vmc.generate_random_config(12345);
        let energy = vmc.calculate_energy(&config);

        // Test Monte Carlo sampling with warmup
        let stats = vmc.run_vmc(100, 400)?;

        let success = self.validate_results(
            stats.average_energy,
            stats.acceptance_rate,
            test_case.expected_energy_range,
            test_case.expected_acceptance_range
        );

        Ok(TestResult {
            success,
            average_energy: stats.average_energy,
            variance: stats.variance,
            acceptance_rate: stats.acceptance_rate,
            error_message: if success { None } else {
                Some(format!("Energy {:.3} not in range {:?}, Acceptance {:.3} not in range {:?}",
                    stats.average_energy, test_case.expected_energy_range,
                    stats.acceptance_rate, test_case.expected_acceptance_range))
            },
        })
    }

    /// Tests Adaptive VMC implementation
    fn test_adaptive_vmc(&self, test_case: &VmcTestCase) -> Result<TestResult, Box<dyn std::error::Error>> {
        let mut vmc = AdaptiveHeisenbergVMC::new(
            test_case.nsite,
            test_case.two_sz,
            test_case.exchange,
            test_case.temperature,
            0.5 // target acceptance rate
        )?;

        // Test basic functionality
        let config = vmc.generate_random_config();
        let energy = vmc.calculate_energy(&config);

        // Test adaptive Monte Carlo sampling
        let stats = vmc.run_adaptive_vmc(500)?;

        let success = self.validate_results(
            stats.average_energy,
            stats.final_acceptance_rate,
            test_case.expected_energy_range,
            test_case.expected_acceptance_range
        );

        Ok(TestResult {
            success,
            average_energy: stats.average_energy,
            variance: stats.variance,
            acceptance_rate: stats.final_acceptance_rate,
            error_message: if success { None } else {
                Some(format!("Energy {:.3} not in range {:?}, Acceptance {:.3} not in range {:?}",
                    stats.average_energy, test_case.expected_energy_range,
                    stats.final_acceptance_rate, test_case.expected_acceptance_range))
            },
        })
    }

    /// Validates test results against expected ranges
    fn validate_results(
        &self,
        energy: f64,
        acceptance_rate: f64,
        expected_energy_range: (f64, f64),
        expected_acceptance_range: (f64, f64),
    ) -> bool {
        let energy_ok = energy >= expected_energy_range.0 && energy <= expected_energy_range.1;
        let acceptance_ok = acceptance_rate >= expected_acceptance_range.0 && acceptance_rate <= expected_acceptance_range.1;

        energy_ok && acceptance_ok
    }
}

/// Results from a single test
#[derive(Debug)]
pub struct TestResult {
    pub success: bool,
    pub average_energy: f64,
    pub variance: f64,
    pub acceptance_rate: f64,
    pub error_message: Option<String>,
}

/// Results from all integration tests
#[derive(Debug)]
pub struct IntegrationTestResults {
    pub results: HashMap<String, HashMap<String, TestResult>>,
    pub total_tests: usize,
    pub passed_tests: usize,
    pub failed_tests: usize,
}

impl IntegrationTestResults {
    pub fn new() -> Self {
        Self {
            results: HashMap::new(),
            total_tests: 0,
            passed_tests: 0,
            failed_tests: 0,
        }
    }

    pub fn add_result(&mut self, implementation: String, test_case: String, result: TestResult) {
        self.results.entry(implementation).or_insert_with(HashMap::new).insert(test_case, result);
        self.total_tests += 1;
        if result.success {
            self.passed_tests += 1;
        } else {
            self.failed_tests += 1;
        }
    }

    pub fn print_summary(&self) {
        println!("=== Integration Test Summary ===");
        println!("Total tests: {}", self.total_tests);
        println!("Passed: {}", self.passed_tests);
        println!("Failed: {}", self.failed_tests);
        println!("Success rate: {:.1}%", (self.passed_tests as f64 / self.total_tests as f64) * 100.0);
        println!();

        for (implementation, tests) in &self.results {
            println!("{}:", implementation);
            for (test_case, result) in tests {
                let status = if result.success { "PASS" } else { "FAIL" };
                println!("  {}: {} (E={:.3}, A={:.3})", status, test_case, result.average_energy, result.acceptance_rate);
                if let Some(ref error) = result.error_message {
                    println!("    Error: {}", error);
                }
            }
            println!();
        }
    }
}

/// Performance benchmark for VMC calculations
pub struct VmcBenchmark {
    pub system_sizes: Vec<usize>,
    pub temperatures: Vec<f64>,
}

impl VmcBenchmark {
    pub fn new() -> Self {
        Self {
            system_sizes: vec![4, 6, 8, 10, 12],
            temperatures: vec![0.5, 1.0, 2.0],
        }
    }

    pub fn run_benchmark(&self) -> Result<BenchmarkResults, Box<dyn std::error::Error>> {
        let mut results = BenchmarkResults::new();

        println!("=== VMC Performance Benchmark ===");

        for &nsite in &self.system_sizes {
            println!("System size: {} sites", nsite);

            for &temperature in &self.temperatures {
                println!("  Temperature: {}", temperature);

                // Benchmark Simple VMC
                let start = std::time::Instant::now();
                let vmc = SimpleHeisenbergVMC::new(nsite, 0, 1.0)?;
                let (_, _, _) = vmc.run_vmc(1000, 12345)?;
                let simple_time = start.elapsed();

                // Benchmark Improved VMC
                let start = std::time::Instant::now();
                let mut improved_vmc = ImprovedHeisenbergVMC::new(nsite, 0, 1.0, temperature)?;
                let _ = improved_vmc.run_vmc(100, 400)?;
                let improved_time = start.elapsed();

                // Benchmark Adaptive VMC
                let start = std::time::Instant::now();
                let mut adaptive_vmc = AdaptiveHeisenbergVMC::new(nsite, 0, 1.0, temperature, 0.5)?;
                let _ = adaptive_vmc.run_adaptive_vmc(500)?;
                let adaptive_time = start.elapsed();

                results.add_benchmark(nsite, temperature, simple_time, improved_time, adaptive_time);

                println!("    Simple: {:.2}ms, Improved: {:.2}ms, Adaptive: {:.2}ms",
                    simple_time.as_millis(), improved_time.as_millis(), adaptive_time.as_millis());
            }
        }

        Ok(results)
    }
}

/// Benchmark results
#[derive(Debug)]
pub struct BenchmarkResults {
    pub benchmarks: Vec<BenchmarkEntry>,
}

#[derive(Debug)]
pub struct BenchmarkEntry {
    pub nsite: usize,
    pub temperature: f64,
    pub simple_time: std::time::Duration,
    pub improved_time: std::time::Duration,
    pub adaptive_time: std::time::Duration,
}

impl BenchmarkResults {
    pub fn new() -> Self {
        Self {
            benchmarks: Vec::new(),
        }
    }

    pub fn add_benchmark(&mut self, nsite: usize, temperature: f64, simple_time: std::time::Duration, improved_time: std::time::Duration, adaptive_time: std::time::Duration) {
        self.benchmarks.push(BenchmarkEntry {
            nsite,
            temperature,
            simple_time,
            improved_time,
            adaptive_time,
        });
    }

    pub fn print_summary(&self) {
        println!("=== Benchmark Summary ===");
        for entry in &self.benchmarks {
            println!("N={}, T={}: Simple={:.2}ms, Improved={:.2}ms, Adaptive={:.2}ms",
                entry.nsite, entry.temperature,
                entry.simple_time.as_millis(),
                entry.improved_time.as_millis(),
                entry.adaptive_time.as_millis());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integration_test_creation() {
        let test_suite = VmcIntegrationTest::new();
        assert!(!test_suite.test_cases.is_empty());
    }

    #[test]
    fn test_single_vmc_test() {
        let test_suite = VmcIntegrationTest::new();
        let test_case = &test_suite.test_cases[0];

        let result = test_suite.test_simple_vmc(test_case).unwrap();
        assert!(result.success);
    }

    #[test]
    fn test_benchmark_creation() {
        let benchmark = VmcBenchmark::new();
        assert!(!benchmark.system_sizes.is_empty());
        assert!(!benchmark.temperatures.is_empty());
    }
}

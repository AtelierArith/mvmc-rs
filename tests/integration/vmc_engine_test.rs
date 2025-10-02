//! Integration tests for VMC engine
//!
//! This module contains integration tests that verify the VMC engine
//! works correctly with all its components.

use mvmc_core::{
    vmc::VmcEngine,
    config::VmcParameters,
    types::{SiteCount, ElectronCount, TwoSz, RandomSeed, CalcMode, LanczosMode},
};
use mvmc_physics::wavefunction::CombinedWavefunction;
use mvmc_physics::hamiltonian::{HeisenbergHamiltonian, Hamiltonian};
use mvmc_physics::lattice::ChainLattice;

/// Test basic VMC engine creation and initialization
#[test]
fn test_vmc_engine_creation() {
    let params = VmcParameters::builder()
        .nsite(SiteCount::new(4))
        .ne(ElectronCount::new(2))
        .two_sz(TwoSz::new(0))
        .calc_mode(CalcMode::Optimization)
        .lanczos_mode(LanczosMode::None)
        .random_seed(RandomSeed::new(12345))
        .build()
        .unwrap();

    let wavefunction = CombinedWavefunction::new(4, 2).unwrap();
    let lattice = ChainLattice::new(4, true).unwrap();
    let hamiltonian = HeisenbergHamiltonian::new(lattice, -1.0, 0.0).unwrap();
    let engine = VmcEngine::new(params, wavefunction, Box::new(hamiltonian));
    assert!(engine.is_ok());

    let engine = engine.unwrap();
    assert_eq!(engine.params().nsite.get(), 4);
    assert_eq!(engine.params().ne.get(), 2);
    assert_eq!(engine.params().two_sz.get(), 0);
}

/// Test VMC optimization with a small system
#[test]
fn test_vmc_optimization_small_system() {
    let params = VmcParameters::builder()
        .nsite(SiteCount::new(2))
        .ne(ElectronCount::new(1))
        .two_sz(TwoSz::new(1))
        .calc_mode(CalcMode::Optimization)
        .lanczos_mode(LanczosMode::None)
        .random_seed(RandomSeed::new(12345))
        .build()
        .unwrap();

    let wavefunction = CombinedWavefunction::new(2, 1).unwrap();
    let lattice = ChainLattice::new(2, true).unwrap();
    let hamiltonian = HeisenbergHamiltonian::new(lattice, -1.0, 0.0).unwrap();
    let mut engine = VmcEngine::new(params, wavefunction, Box::new(hamiltonian)).unwrap();

    // Run optimization (this should complete without error)
    let result = engine.run(None::<fn(usize, num_complex::Complex64, &mvmc_core::monte_carlo::SamplingStatistics)>);
    assert!(result.is_ok());

    let result = result.unwrap();
    assert!(result.optimization.is_some());
    if let Some(opt) = result.optimization {
        assert!(opt.iterations > 0);
    }
}

/// Test VMC expectation calculation with a small system
#[test]
fn test_vmc_expectation_small_system() {
    let params = VmcParameters::builder()
        .nsite(SiteCount::new(2))
        .ne(ElectronCount::new(1))
        .two_sz(TwoSz::new(1))
        .calc_mode(CalcMode::Expectation)
        .lanczos_mode(LanczosMode::None)
        .random_seed(RandomSeed::new(12345))
        .build()
        .unwrap();

    let wavefunction = CombinedWavefunction::new(2, 1).unwrap();
    let lattice = ChainLattice::new(2, true).unwrap();
    let hamiltonian = HeisenbergHamiltonian::new(lattice, -1.0, 0.0).unwrap();
    let mut engine = VmcEngine::new(params, wavefunction, Box::new(hamiltonian)).unwrap();

    // Run expectation calculation (this should complete without error)
    let result = engine.run(None::<fn(usize, num_complex::Complex64, &mvmc_core::monte_carlo::SamplingStatistics)>);
    assert!(result.is_ok());

    let result = result.unwrap();
    assert!(result.energy.re.is_finite());
    assert!(result.energy_error.is_finite());
}

/// Test energy calculation
#[test]
fn test_energy_calculation() {
    let params = VmcParameters::builder()
        .nsite(SiteCount::new(4))
        .ne(ElectronCount::new(2))
        .two_sz(TwoSz::new(0))
        .calc_mode(CalcMode::Expectation)
        .lanczos_mode(LanczosMode::None)
        .random_seed(RandomSeed::new(12345))
        .build()
        .unwrap();

    let wavefunction = CombinedWavefunction::new(4, 2).unwrap();
    let lattice = ChainLattice::new(4, true).unwrap();
    let hamiltonian = HeisenbergHamiltonian::new(lattice, -1.0, 0.0).unwrap();
    let mut engine = VmcEngine::new(params, wavefunction, Box::new(hamiltonian)).unwrap();

    // Run expectation calculation to get energy
    let result = engine.run(None::<fn(usize, num_complex::Complex64, &mvmc_core::monte_carlo::SamplingStatistics)>);
    assert!(result.is_ok());

    let result = result.unwrap();
    assert!(result.energy.re.is_finite());
    assert!(result.energy.im.is_finite());
}

/// Test wavefunction access
#[test]
fn test_wavefunction_access() {
    let params = VmcParameters::builder()
        .nsite(SiteCount::new(4))
        .ne(ElectronCount::new(2))
        .two_sz(TwoSz::new(0))
        .calc_mode(CalcMode::Optimization)
        .lanczos_mode(LanczosMode::None)
        .random_seed(RandomSeed::new(12345))
        .build()
        .unwrap();

    let wavefunction = CombinedWavefunction::new(4, 2).unwrap();
    let lattice = ChainLattice::new(4, true).unwrap();
    let hamiltonian = HeisenbergHamiltonian::new(lattice, -1.0, 0.0).unwrap();
    let engine = VmcEngine::new(params, wavefunction, Box::new(hamiltonian)).unwrap();

    // Test wavefunction access
    let wavefunction = engine.wavefunction();
    assert_eq!(wavefunction.nsite(), 4);
    assert_eq!(wavefunction.ne(), 2);
}

/// Test current configuration access
#[test]
fn test_current_configuration_access() {
    let params = VmcParameters::builder()
        .nsite(SiteCount::new(4))
        .ne(ElectronCount::new(2))
        .two_sz(TwoSz::new(0))
        .calc_mode(CalcMode::Optimization)
        .lanczos_mode(LanczosMode::None)
        .random_seed(RandomSeed::new(12345))
        .build()
        .unwrap();

    let wavefunction = CombinedWavefunction::new(4, 2).unwrap();
    let lattice = ChainLattice::new(4, true).unwrap();
    let hamiltonian = HeisenbergHamiltonian::new(lattice, -1.0, 0.0).unwrap();
    let engine = VmcEngine::new(params, wavefunction, Box::new(hamiltonian)).unwrap();

    // Test sampler access
    let _sampler = engine.sampler();
    // TODO: Add proper sampler method tests when implemented
}

/// Test sampling statistics
#[test]
fn test_sampling_statistics() {
    let params = VmcParameters::builder()
        .nsite(SiteCount::new(4))
        .ne(ElectronCount::new(2))
        .two_sz(TwoSz::new(0))
        .calc_mode(CalcMode::Optimization)
        .lanczos_mode(LanczosMode::None)
        .random_seed(RandomSeed::new(12345))
        .build()
        .unwrap();

    let wavefunction = CombinedWavefunction::new(4, 2).unwrap();
    let lattice = ChainLattice::new(4, true).unwrap();
    let hamiltonian = HeisenbergHamiltonian::new(lattice, -1.0, 0.0).unwrap();
    let engine = VmcEngine::new(params, wavefunction, Box::new(hamiltonian)).unwrap();

    // Test sampling statistics access
    let _stats = engine.sampler();
    // TODO: Add proper statistics tests when implemented
}

/// Test parameter validation
#[test]
fn test_parameter_validation() {
    // Test with invalid parameters (too many electrons)
    let params = VmcParameters::builder()
        .nsite(SiteCount::new(2))
        .ne(ElectronCount::new(10)) // More electrons than sites * 2
        .two_sz(TwoSz::new(0))
        .calc_mode(CalcMode::Optimization)
        .lanczos_mode(LanczosMode::None)
        .random_seed(RandomSeed::new(12345))
        .build();

    // This should fail validation at parameter building stage
    assert!(params.is_err());

    // Check that the error message contains the expected validation error
    if let Err(e) = params {
        assert!(e.to_string().contains("Too many electrons"));
    } else {
        panic!("Expected validation error but got success");
    }
}

/// Test different calculation modes
#[test]
fn test_different_calculation_modes() {
    let base_params = || VmcParameters::builder()
        .nsite(SiteCount::new(2))
        .ne(ElectronCount::new(1))
        .two_sz(TwoSz::new(1))
        .lanczos_mode(LanczosMode::None)
        .random_seed(RandomSeed::new(12345));

    // Test optimization mode
    let opt_params = base_params()
        .calc_mode(CalcMode::Optimization)
        .build()
        .unwrap();

    let wavefunction = CombinedWavefunction::new(2, 1).unwrap();
    let lattice1 = ChainLattice::new(2, true).unwrap();
    let hamiltonian1 = HeisenbergHamiltonian::new(lattice1, -1.0, 0.0).unwrap();
    let opt_engine = VmcEngine::new(opt_params, wavefunction, Box::new(hamiltonian1)).unwrap();
    assert_eq!(opt_engine.params().calc_mode, CalcMode::Optimization);

    // Test expectation mode
    let exp_params = base_params()
        .calc_mode(CalcMode::Expectation)
        .build()
        .unwrap();

    let wavefunction2 = CombinedWavefunction::new(2, 1).unwrap();
    let lattice2 = ChainLattice::new(2, true).unwrap();
    let hamiltonian2 = HeisenbergHamiltonian::new(lattice2, -1.0, 0.0).unwrap();
    let exp_engine = VmcEngine::new(exp_params, wavefunction2, Box::new(hamiltonian2)).unwrap();
    assert_eq!(exp_engine.params().calc_mode, CalcMode::Expectation);
}

/// Test different system sizes
#[test]
fn test_different_system_sizes() {
    let sizes = vec![(1, 1), (2, 1), (4, 2), (6, 3)];

    for (nsite, ne) in sizes {
        let params = VmcParameters::builder()
            .nsite(SiteCount::new(nsite))
            .ne(ElectronCount::new(ne))
            .two_sz(TwoSz::new(0))
            .calc_mode(CalcMode::Expectation)
            .lanczos_mode(LanczosMode::None)
            .random_seed(RandomSeed::new(12345))
            .build()
            .unwrap();

        let wavefunction = CombinedWavefunction::new(nsite, ne).unwrap();
        let lattice = ChainLattice::new(nsite, true).unwrap();
        let hamiltonian = HeisenbergHamiltonian::new(lattice, -1.0, 0.0).unwrap();
        let engine = VmcEngine::new(params, wavefunction, Box::new(hamiltonian));
        assert!(engine.is_ok(), "Failed for nsite={}, ne={}", nsite, ne);
    }
}

/// Test different spin configurations
#[test]
fn test_different_spin_configurations() {
    let spins = vec![0, 1, -1, 2, -2];

    for spin in spins {
        let params = VmcParameters::builder()
            .nsite(SiteCount::new(4))
            .ne(ElectronCount::new(2))
            .two_sz(TwoSz::new(spin))
            .calc_mode(CalcMode::Expectation)
            .lanczos_mode(LanczosMode::None)
            .random_seed(RandomSeed::new(12345))
            .build()
            .unwrap();

        let wavefunction = CombinedWavefunction::new(4, 2).unwrap();
        let lattice = ChainLattice::new(4, true).unwrap();
        let hamiltonian = HeisenbergHamiltonian::new(lattice, -1.0, 0.0).unwrap();
        let engine = VmcEngine::new(params, wavefunction, Box::new(hamiltonian));
        assert!(engine.is_ok(), "Failed for spin={}", spin);
    }
}

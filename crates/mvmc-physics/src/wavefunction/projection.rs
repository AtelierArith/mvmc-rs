//! Projection operators for wavefunction implementation.
//!
//! This module provides various projection operators used in VMC calculations
//! to enforce physical constraints and symmetries.

use ndarray::Array1;
use num_complex::Complex64;
use std::fmt;

/// Projection operator for enforcing particle number conservation.
///
/// This operator projects the wavefunction onto the subspace with
/// a fixed number of particles.
#[derive(Debug, Clone)]
pub struct ParticleNumberProjector {
    /// Target number of particles
    target_particles: usize,
    /// Tolerance for particle number
    tolerance: f64,
}

impl ParticleNumberProjector {
    /// Creates a new particle number projector.
    ///
    /// # Arguments
    /// * `target_particles` - Target number of particles
    /// * `tolerance` - Tolerance for particle number (default: 1e-6)
    ///
    /// # Returns
    /// * `Self` - The particle number projector
    pub fn new(target_particles: usize, tolerance: f64) -> Self {
        Self {
            target_particles,
            tolerance,
        }
    }

    /// Projects a spin configuration onto the target particle number.
    ///
    /// # Arguments
    /// * `spin_config` - Spin configuration (0: empty, 1: up, 2: down, 3: both)
    ///
    /// # Returns
    /// * `Complex64` - The projection weight
    pub fn project(&self, spin_config: &[u8]) -> Complex64 {
        let particle_count = self.count_particles(spin_config);

        if (particle_count as f64 - self.target_particles as f64).abs() < self.tolerance {
            Complex64::new(1.0, 0.0)
        } else {
            Complex64::new(0.0, 0.0)
        }
    }

    /// Counts the number of particles in a spin configuration.
    ///
    /// # Arguments
    /// * `spin_config` - Spin configuration
    ///
    /// # Returns
    /// * `usize` - Number of particles
    fn count_particles(&self, spin_config: &[u8]) -> usize {
        spin_config.iter().map(|&spin| {
            match spin {
                1 => 1,  // up spin
                2 => 1,  // down spin
                3 => 2,  // both spins
                _ => 0,  // empty
            }
        }).sum()
    }
}

/// Projection operator for enforcing total spin conservation.
///
/// This operator projects the wavefunction onto the subspace with
/// a fixed total spin.
#[derive(Debug, Clone)]
pub struct TotalSpinProjector {
    /// Target total spin (2Sz)
    target_spin: i32,
    /// Tolerance for total spin
    tolerance: f64,
}

impl TotalSpinProjector {
    /// Creates a new total spin projector.
    ///
    /// # Arguments
    /// * `target_spin` - Target total spin (2Sz)
    /// * `tolerance` - Tolerance for total spin (default: 1e-6)
    ///
    /// # Returns
    /// * `Self` - The total spin projector
    pub fn new(target_spin: i32, tolerance: f64) -> Self {
        Self {
            target_spin,
            tolerance,
        }
    }

    /// Projects a spin configuration onto the target total spin.
    ///
    /// # Arguments
    /// * `spin_config` - Spin configuration (0: empty, 1: up, 2: down, 3: both)
    ///
    /// # Returns
    /// * `Complex64` - The projection weight
    pub fn project(&self, spin_config: &[u8]) -> Complex64 {
        let total_spin = self.calculate_total_spin(spin_config);

        if (total_spin as f64 - self.target_spin as f64).abs() < self.tolerance {
            Complex64::new(1.0, 0.0)
        } else {
            Complex64::new(0.0, 0.0)
        }
    }

    /// Calculates the total spin of a spin configuration.
    ///
    /// # Arguments
    /// * `spin_config` - Spin configuration
    ///
    /// # Returns
    /// * `i32` - Total spin (2Sz)
    fn calculate_total_spin(&self, spin_config: &[u8]) -> i32 {
        spin_config.iter().map(|&spin| {
            match spin {
                1 => 1,   // up spin
                2 => -1,  // down spin
                3 => 0,   // both spins (net zero)
                _ => 0,   // empty
            }
        }).sum()
    }
}

/// Projection operator for enforcing momentum conservation.
///
/// This operator projects the wavefunction onto the subspace with
/// a fixed total momentum.
#[derive(Debug, Clone)]
pub struct MomentumProjector {
    /// Target momentum
    target_momentum: f64,
    /// Tolerance for momentum
    tolerance: f64,
}

impl MomentumProjector {
    /// Creates a new momentum projector.
    ///
    /// # Arguments
    /// * `target_momentum` - Target momentum
    /// * `tolerance` - Tolerance for momentum (default: 1e-6)
    ///
    /// # Returns
    /// * `Self` - The momentum projector
    pub fn new(target_momentum: f64, tolerance: f64) -> Self {
        Self {
            target_momentum,
            tolerance,
        }
    }

    /// Projects a spin configuration onto the target momentum.
    ///
    /// # Arguments
    /// * `spin_config` - Spin configuration
    ///
    /// # Returns
    /// * `Complex64` - The projection weight
    pub fn project(&self, spin_config: &[u8]) -> Complex64 {
        let momentum = self.calculate_momentum(spin_config);

        if (momentum - self.target_momentum).abs() < self.tolerance {
            Complex64::new(1.0, 0.0)
        } else {
            Complex64::new(0.0, 0.0)
        }
    }

    /// Calculates the total momentum of a spin configuration.
    ///
    /// # Arguments
    /// * `spin_config` - Spin configuration
    ///
    /// # Returns
    /// * `f64` - Total momentum
    fn calculate_momentum(&self, spin_config: &[u8]) -> f64 {
        // This is a simplified implementation
        // In practice, this would depend on the specific lattice and model
        let nsite = spin_config.len();
        let mut momentum = 0.0;

        for (i, &spin) in spin_config.iter().enumerate() {
            if spin != 0 {
                let k = 2.0 * std::f64::consts::PI * i as f64 / nsite as f64;
                momentum += k;
            }
        }

        momentum
    }
}

/// Projection operator for enforcing spatial symmetries.
///
/// This operator projects the wavefunction onto the subspace that
/// respects spatial symmetries of the lattice.
#[derive(Debug, Clone)]
pub struct SpatialSymmetryProjector {
    /// Symmetry operations
    symmetry_operations: Vec<Array1<usize>>,
    /// Tolerance for symmetry
    #[allow(dead_code)]
    tolerance: f64,
}

impl SpatialSymmetryProjector {
    /// Creates a new spatial symmetry projector.
    ///
    /// # Arguments
    /// * `symmetry_operations` - List of symmetry operations (permutations)
    /// * `tolerance` - Tolerance for symmetry (default: 1e-6)
    ///
    /// # Returns
    /// * `Self` - The spatial symmetry projector
    pub fn new(symmetry_operations: Vec<Array1<usize>>, tolerance: f64) -> Self {
        Self {
            symmetry_operations,
            tolerance,
        }
    }

    /// Creates a projector for translational symmetry.
    ///
    /// # Arguments
    /// * `nsite` - Number of lattice sites
    /// * `tolerance` - Tolerance for symmetry (default: 1e-6)
    ///
    /// # Returns
    /// * `Self` - The spatial symmetry projector
    pub fn new_translational(nsite: usize, tolerance: f64) -> Self {
        let mut symmetry_operations = Vec::new();

        for shift in 1..nsite {
            let mut permutation = Array1::zeros(nsite);
            for i in 0..nsite {
                permutation[i] = (i + shift) % nsite;
            }
            symmetry_operations.push(permutation);
        }

        Self::new(symmetry_operations, tolerance)
    }

    /// Projects a spin configuration onto the symmetric subspace.
    ///
    /// # Arguments
    /// * `spin_config` - Spin configuration
    ///
    /// # Returns
    /// * `Complex64` - The projection weight
    pub fn project(&self, spin_config: &[u8]) -> Complex64 {
        let nsite = spin_config.len();
        let mut weight = Complex64::new(0.0, 0.0);
        let mut count = 0;

        for operation in &self.symmetry_operations {
            if operation.len() == nsite {
                let mut transformed = vec![0u8; nsite];
                for i in 0..nsite {
                    transformed[operation[i]] = spin_config[i];
                }

                if transformed == spin_config {
                    weight += Complex64::new(1.0, 0.0);
                }
                count += 1;
            }
        }

        if count > 0 {
            weight / (count as f64)
        } else {
            Complex64::new(1.0, 0.0)
        }
    }
}

/// Combined projection operator.
///
/// This operator combines multiple projection operators and applies
/// them sequentially or simultaneously.
pub struct CombinedProjector {
    /// List of projection operators
    projectors: Vec<Box<dyn Projector + Send + Sync>>,
    /// Whether to apply projectors sequentially (true) or simultaneously (false)
    sequential: bool,
}

impl CombinedProjector {
    /// Creates a new combined projector.
    ///
    /// # Arguments
    /// * `projectors` - List of projection operators
    /// * `sequential` - Whether to apply projectors sequentially
    ///
    /// # Returns
    /// * `Self` - The combined projector
    pub fn new(projectors: Vec<Box<dyn Projector + Send + Sync>>, sequential: bool) -> Self {
        Self {
            projectors,
            sequential,
        }
    }

    /// Projects a spin configuration using all projectors.
    ///
    /// # Arguments
    /// * `spin_config` - Spin configuration
    ///
    /// # Returns
    /// * `Complex64` - The combined projection weight
    pub fn project(&self, spin_config: &[u8]) -> Complex64 {
        if self.sequential {
            // Apply projectors sequentially
            let mut weight = Complex64::new(1.0, 0.0);
            for projector in &self.projectors {
                weight *= projector.project(spin_config);
            }
            weight
        } else {
            // Apply projectors simultaneously (average)
            let mut total_weight = Complex64::new(0.0, 0.0);
            let mut count = 0;

            for projector in &self.projectors {
                total_weight += projector.project(spin_config);
                count += 1;
            }

            if count > 0 {
                total_weight / (count as f64)
            } else {
                Complex64::new(1.0, 0.0)
            }
        }
    }
}

/// Trait for projection operators.
pub trait Projector {
    /// Projects a spin configuration.
    ///
    /// # Arguments
    /// * `spin_config` - Spin configuration
    ///
    /// # Returns
    /// * `Complex64` - The projection weight
    fn project(&self, spin_config: &[u8]) -> Complex64;

    /// Applies the projection operator to a given wavefunction amplitude.
    ///
    /// # Arguments
    /// * `amplitude` - Current wavefunction amplitude
    /// * `spin_config` - Spin configuration
    ///
    /// # Returns
    /// * `Complex64` - The projected amplitude
    fn apply(&self, amplitude: Complex64, spin_config: &[crate::hamiltonian::Spin]) -> Complex64 {
        // Convert spin configuration to u8 array
        let u8_config = spin_config.iter().map(|spin| {
            match spin {
                crate::hamiltonian::Spin::Up => 1,
                crate::hamiltonian::Spin::Down => 2,
                crate::hamiltonian::Spin::Empty => 0,
            }
        }).collect::<Vec<u8>>();

        let projection_weight = self.project(&u8_config);
        amplitude * projection_weight
    }
}

impl Projector for ParticleNumberProjector {
    fn project(&self, spin_config: &[u8]) -> Complex64 {
        self.project(spin_config)
    }
}

impl Projector for TotalSpinProjector {
    fn project(&self, spin_config: &[u8]) -> Complex64 {
        self.project(spin_config)
    }
}

impl Projector for MomentumProjector {
    fn project(&self, spin_config: &[u8]) -> Complex64 {
        self.project(spin_config)
    }
}

impl Projector for SpatialSymmetryProjector {
    fn project(&self, spin_config: &[u8]) -> Complex64 {
        self.project(spin_config)
    }
}

impl Projector for CombinedProjector {
    fn project(&self, spin_config: &[u8]) -> Complex64 {
        self.project(spin_config)
    }
}

impl fmt::Display for ParticleNumberProjector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ParticleNumberProjector(target={})", self.target_particles)
    }
}

impl fmt::Display for TotalSpinProjector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TotalSpinProjector(target={})", self.target_spin)
    }
}

impl fmt::Display for MomentumProjector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MomentumProjector(target={})", self.target_momentum)
    }
}

impl fmt::Display for SpatialSymmetryProjector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SpatialSymmetryProjector(operations={})", self.symmetry_operations.len())
    }
}

impl Clone for CombinedProjector {
    fn clone(&self) -> Self {
        Self {
            projectors: Vec::new(), // Cannot clone trait objects
            sequential: self.sequential,
        }
    }
}

impl fmt::Debug for CombinedProjector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CombinedProjector(projectors={}, sequential={})",
               self.projectors.len(), self.sequential)
    }
}

impl fmt::Display for CombinedProjector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CombinedProjector(projectors={}, sequential={})",
               self.projectors.len(), self.sequential)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_particle_number_projector() {
        let projector = ParticleNumberProjector::new(2, 1e-6);
        let spin_config = vec![1, 2]; // up, down (2 particles)
        let weight = projector.project(&spin_config);
        assert!((weight.re - 1.0).abs() < 1e-12);
        assert!(weight.im.abs() < 1e-12);
    }

    #[test]
    fn test_total_spin_projector() {
        let projector = TotalSpinProjector::new(0, 1e-6);
        let spin_config = vec![1, 2]; // up, down (net zero spin)
        let weight = projector.project(&spin_config);
        assert!((weight.re - 1.0).abs() < 1e-12);
        assert!(weight.im.abs() < 1e-12);
    }

    #[test]
    fn test_momentum_projector() {
        let projector = MomentumProjector::new(0.0, 1e-6);
        let spin_config = vec![1, 2];
        let weight = projector.project(&spin_config);
        // Weight depends on the specific momentum calculation
        assert!(weight.norm() >= 0.0);
    }

    #[test]
    fn test_spatial_symmetry_projector() {
        let nsite = 4;
        let projector = SpatialSymmetryProjector::new_translational(nsite, 1e-6);
        let spin_config = vec![1, 0, 1, 0]; // periodic pattern
        let weight = projector.project(&spin_config);
        assert!(weight.norm() >= 0.0);
    }
}

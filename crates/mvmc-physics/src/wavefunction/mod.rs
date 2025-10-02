//! Wavefunction implementations for VMC calculations.
//!
//! This module provides various wavefunction implementations including
//! Slater determinants, Pfaffians, and projection operators.

pub mod slater;
pub mod pfaffian;
pub mod projection;

// Re-export commonly used types
pub use slater::SlaterDeterminant;
pub use pfaffian::PfaffianWavefunction;
pub use projection::{
    ParticleNumberProjector, TotalSpinProjector, MomentumProjector,
    SpatialSymmetryProjector, CombinedProjector, Projector,
};

use crate::Result;
use num_complex::Complex64;

/// Trait for wavefunction implementations.
///
/// This trait defines the common interface for all wavefunction types,
/// allowing for generic algorithms that work with any wavefunction.
pub trait Wavefunction: Send + Sync {
    /// Calculates the wavefunction amplitude for a given configuration.
    ///
    /// # Arguments
    /// * `config` - Spin configuration as u8 array
    ///
    /// # Returns
    /// * `Result<Complex64>` - The wavefunction amplitude
    fn calculate(&self, config: &[u8]) -> Result<Complex64>;

    /// Calculates the wavefunction amplitude for a given spin configuration.
    ///
    /// # Arguments
    /// * `config` - Spin configuration
    ///
    /// # Returns
    /// * `Result<Complex64>` - The wavefunction amplitude
    fn calculate_spin(&self, config: &[crate::hamiltonian::Spin]) -> Result<Complex64> {
        let u8_config = config.iter().map(|spin| {
            match spin {
                crate::hamiltonian::Spin::Up => 1,
                crate::hamiltonian::Spin::Down => 2,
                crate::hamiltonian::Spin::Empty => 0,
            }
        }).collect::<Vec<u8>>();
        self.calculate(&u8_config)
    }

    /// Returns the number of lattice sites.
    fn nsite(&self) -> usize;

    /// Returns the number of electrons.
    fn ne(&self) -> usize;
}

/// Combined wavefunction for VMC calculations.
///
/// This structure combines different wavefunction components
/// to represent the full many-body wavefunction.
pub struct CombinedWavefunction {
    /// Slater determinant component
    slater: Option<SlaterDeterminant>,
    /// Pfaffian component
    pfaffian: Option<PfaffianWavefunction>,
    /// Projection operators
    projectors: Vec<Box<dyn Projector + Send + Sync>>,
    /// Number of lattice sites
    nsite: usize,
    /// Number of electrons
    ne: usize,
    /// Gutzwiller projection parameters (one per site, for double occupancy)
    gutzwiller_params: Vec<f64>,
    /// Jastrow correlation parameters (one per site pair)
    /// Indexed as jastrow_params[i * nsite + j] for i < j
    jastrow_params: Vec<f64>,
}

impl CombinedWavefunction {
    /// Creates a new combined wavefunction.
    ///
    /// # Arguments
    /// * `nsite` - Number of lattice sites
    /// * `ne` - Number of electrons
    ///
    /// # Returns
    /// * `Result<Self>` - The combined wavefunction
    ///
    /// # Note
    /// This creates a wavefunction with a plane wave Slater determinant by default.
    /// Reference: mVMC/src/mVMC/vmcmake.c - makeInitialSample()
    pub fn new(nsite: usize, ne: usize) -> Result<Self> {
        // Initialize with plane wave basis (like C implementation)
        // Reference: mVMC/src/mVMC/vmcmake.c - makeInitialSlaterElm()
        let slater = SlaterDeterminant::new_plane_wave(nsite, ne)?;

        // Initialize Gutzwiller and Jastrow parameters to zero
        let gutzwiller_params = vec![0.0; nsite];
        let n_jastrow = nsite * (nsite - 1) / 2;  // Number of unique pairs i < j
        let jastrow_params = vec![0.0; n_jastrow];

        Ok(Self {
            slater: Some(slater),
            pfaffian: None,
            projectors: Vec::new(),
            nsite,
            ne,
            gutzwiller_params,
            jastrow_params,
        })
    }

    /// Creates a combined wavefunction with Slater determinant.
    ///
    /// # Arguments
    /// * `nsite` - Number of lattice sites
    /// * `ne` - Number of electrons
    /// * `slater` - Slater determinant
    ///
    /// # Returns
    /// * `Result<Self>` - The combined wavefunction
    pub fn with_slater(nsite: usize, ne: usize, slater: SlaterDeterminant) -> Result<Self> {
        if slater.nsite() != nsite || slater.ne() != ne {
            return Err(anyhow::anyhow!(
                "Slater determinant dimensions mismatch: expected ({}, {}), got ({}, {})",
                nsite, ne, slater.nsite(), slater.ne()
            ));
        }

        let gutzwiller_params = vec![0.0; nsite];
        let n_jastrow = nsite * (nsite - 1) / 2;
        let jastrow_params = vec![0.0; n_jastrow];

        Ok(Self {
            slater: Some(slater),
            pfaffian: None,
            projectors: Vec::new(),
            nsite,
            ne,
            gutzwiller_params,
            jastrow_params,
        })
    }

    /// Creates a combined wavefunction with Pfaffian.
    ///
    /// # Arguments
    /// * `nsite` - Number of lattice sites
    /// * `ne` - Number of electrons
    /// * `pfaffian` - Pfaffian wavefunction
    ///
    /// # Returns
    /// * `Result<Self>` - The combined wavefunction
    pub fn with_pfaffian(nsite: usize, ne: usize, pfaffian: PfaffianWavefunction) -> Result<Self> {
        if pfaffian.nsite() != nsite || pfaffian.ne() != ne {
            return Err(anyhow::anyhow!(
                "Pfaffian wavefunction dimensions mismatch: expected ({}, {}), got ({}, {})",
                nsite, ne, pfaffian.nsite(), pfaffian.ne()
            ));
        }

        let gutzwiller_params = vec![0.0; nsite];
        let n_jastrow = nsite * (nsite - 1) / 2;
        let jastrow_params = vec![0.0; n_jastrow];

        Ok(Self {
            slater: None,
            pfaffian: Some(pfaffian),
            projectors: Vec::new(),
            nsite,
            ne,
            gutzwiller_params,
            jastrow_params,
        })
    }

    /// Creates a combined wavefunction with both Slater and Pfaffian.
    ///
    /// # Arguments
    /// * `nsite` - Number of lattice sites
    /// * `ne` - Number of electrons
    /// * `slater` - Slater determinant
    /// * `pfaffian` - Pfaffian wavefunction
    ///
    /// # Returns
    /// * `Result<Self>` - The combined wavefunction
    pub fn with_both(
        nsite: usize,
        ne: usize,
        slater: SlaterDeterminant,
        pfaffian: PfaffianWavefunction,
    ) -> Result<Self> {
        if slater.nsite() != nsite || slater.ne() != ne {
            return Err(anyhow::anyhow!(
                "Slater determinant dimensions mismatch: expected ({}, {}), got ({}, {})",
                nsite, ne, slater.nsite(), slater.ne()
            ));
        }

        if pfaffian.nsite() != nsite || pfaffian.ne() != ne {
            return Err(anyhow::anyhow!(
                "Pfaffian wavefunction dimensions mismatch: expected ({}, {}), got ({}, {})",
                nsite, ne, pfaffian.nsite(), pfaffian.ne()
            ));
        }

        let gutzwiller_params = vec![0.0; nsite];
        let n_jastrow = nsite * (nsite - 1) / 2;
        let jastrow_params = vec![0.0; n_jastrow];

        Ok(Self {
            slater: Some(slater),
            pfaffian: Some(pfaffian),
            projectors: Vec::new(),
            nsite,
            ne,
            gutzwiller_params,
            jastrow_params,
        })
    }

    /// Returns the number of lattice sites.
    pub fn nsite(&self) -> usize {
        self.nsite
    }

    /// Returns the number of electrons.
    pub fn ne(&self) -> usize {
        self.ne
    }

    /// Returns a reference to the Slater determinant.
    pub fn slater(&self) -> Option<&SlaterDeterminant> {
        self.slater.as_ref()
    }

    /// Returns a mutable reference to the Slater determinant.
    pub fn slater_mut(&mut self) -> Option<&mut SlaterDeterminant> {
        self.slater.as_mut()
    }

    /// Returns a reference to the Pfaffian wavefunction.
    pub fn pfaffian(&self) -> Option<&PfaffianWavefunction> {
        self.pfaffian.as_ref()
    }

    /// Returns a mutable reference to the Pfaffian wavefunction.
    pub fn pfaffian_mut(&mut self) -> Option<&mut PfaffianWavefunction> {
        self.pfaffian.as_mut()
    }

    /// Adds a projection operator.
    ///
    /// # Arguments
    /// * `projector` - Projection operator to add
    pub fn add_projector(&mut self, projector: Box<dyn Projector + Send + Sync>) {
        self.projectors.push(projector);
    }

    /// Updates wavefunction parameters using SR method.
    ///
    /// Applies parameter updates computed from SR optimization:
    /// f_{ij} ← f_{ij} + δf_{ij}
    ///
    /// # Arguments
    /// * `param_updates` - Parameter updates δf_{ij} from SR equation
    /// * `_learning_rate` - Learning rate (not used when updates are provided)
    ///
    /// # Reference
    /// C implementation: mVMC/src/mVMC/stcopt.c - lines 174-186
    pub fn update_parameters(&mut self, param_updates: &[f64], learning_rate: f64) {
        if param_updates.is_empty() {
            // Fallback to random perturbation if no SR updates provided
            if let Some(slater) = &mut self.slater {
                slater.add_noise(0.01);
            }
        } else {
            // Apply SR-calculated parameter updates to all components
            let mut offset = 0;

            // Update Slater determinant parameters
            if let Some(slater) = &mut self.slater {
                let n_slater_params = slater.nsite() * slater.ne() * 2;
                if param_updates.len() >= n_slater_params {
                    slater.apply_parameter_updates(&param_updates[offset..offset+n_slater_params]);
                    offset += n_slater_params;
                }
            }

            // Update Gutzwiller and Jastrow parameters
            self.update_projection_parameters(param_updates, learning_rate);
        }
    }

    /// Calculates O-operators for SR optimization.
    ///
    /// Computes O_k = (1/ψ) ∂ψ/∂f_k for each variational parameter.
    ///
    /// # Arguments
    /// * `spin_config` - Spin configuration
    ///
    /// # Returns
    /// * `Vec<Complex64>` - O-operators for all parameters
    ///
    /// # Reference
    /// C implementation: mVMC/src/mVMC/slater.c - SlaterElmDiff_fcmp
    pub fn calculate_o_operators(&self, spin_config: &[u8]) -> Vec<Complex64> {
        // Use the new method that includes all parameters
        self.calculate_all_parameter_derivatives(spin_config)
    }

    /// Returns the number of variational parameters.
    pub fn num_parameters(&self) -> usize {
        let mut n_params = 0;

        // Slater determinant parameters
        if let Some(slater) = &self.slater {
            n_params += slater.nsite() * slater.ne() * 2;  // real + imaginary
        }

        // Gutzwiller parameters (one per site)
        n_params += self.nsite;

        // Jastrow parameters (one per unique site pair i < j)
        n_params += self.nsite * (self.nsite - 1) / 2;

        n_params
    }

    /// Calculates the wavefunction value for a given spin configuration.
    ///
    /// # Arguments
    /// * `spin_config` - Spin configuration (0: empty, 1: up, 2: down, 3: both)
    ///
    /// # Returns
    /// * `Complex64` - The wavefunction value
    pub fn calculate(&self, spin_config: &[u8]) -> Complex64 {
        if spin_config.len() != self.nsite {
            return Complex64::new(0.0, 0.0);
        }

        // Calculate Slater determinant component
        let slater_value = if let Some(slater) = &self.slater {
            slater.calculate_determinant(spin_config)
        } else {
            Complex64::new(1.0, 0.0)
        };

        // Calculate Pfaffian component
        let pfaffian_value = if let Some(pfaffian) = &self.pfaffian {
            pfaffian.calculate_pfaffian(spin_config)
        } else {
            Complex64::new(1.0, 0.0)
        };

        // Calculate projection weight
        let projection_weight = self.calculate_projection_weight(spin_config);

        // Combine all components
        slater_value * pfaffian_value * projection_weight
    }

    /// Calculates the ratio of wavefunction values after a spin flip.
    ///
    /// # Arguments
    /// * `spin_config` - Current spin configuration
    /// * `flip_site` - Site to flip
    /// * `flip_from` - Current spin state
    /// * `flip_to` - New spin state
    ///
    /// # Returns
    /// * `Complex64` - The ratio of wavefunction values
    pub fn calculate_ratio(&self, spin_config: &[u8], flip_site: usize, flip_from: u8, flip_to: u8) -> Complex64 {
        // Calculate Slater determinant ratio
        let slater_ratio = if let Some(slater) = &self.slater {
            slater.calculate_ratio(spin_config, flip_site, flip_from, flip_to)
        } else {
            Complex64::new(1.0, 0.0)
        };

        // Calculate Pfaffian ratio
        let pfaffian_ratio = if let Some(pfaffian) = &self.pfaffian {
            pfaffian.calculate_ratio(spin_config, flip_site, flip_from, flip_to)
        } else {
            Complex64::new(1.0, 0.0)
        };

        // Calculate projection weight ratio
        let mut new_config = spin_config.to_vec();
        new_config[flip_site] = flip_to;
        let old_projection = self.calculate_projection_weight(spin_config);
        let new_projection = self.calculate_projection_weight(&new_config);

        let projection_ratio = if old_projection.norm() < 1e-12 {
            Complex64::new(0.0, 0.0)
        } else {
            new_projection / old_projection
        };

        // Combine all ratios
        slater_ratio * pfaffian_ratio * projection_ratio
    }

    /// Calculates the projection weight for a spin configuration.
    ///
    /// # Arguments
    /// * `spin_config` - Spin configuration
    ///
    /// # Returns
    /// * `Complex64` - The projection weight
    fn calculate_projection_weight(&self, spin_config: &[u8]) -> Complex64 {
        let mut weight = Complex64::new(1.0, 0.0);

        for projector in &self.projectors {
            weight *= projector.project(spin_config);
        }

        weight
    }

    /// Normalizes the wavefunction components.
    pub fn normalize(&mut self) {
        if let Some(slater) = &mut self.slater {
            slater.normalize();
        }

        if let Some(pfaffian) = &mut self.pfaffian {
            pfaffian.normalize();
        }
    }

    /// Updates the Slater determinant orbitals.
    ///
    /// # Arguments
    /// * `new_orbitals` - New orbital coefficients
    pub fn update_slater_orbitals(&mut self, new_orbitals: ndarray::Array2<Complex64>) {
        if let Some(slater) = &mut self.slater {
            slater.update_orbitals(new_orbitals);
        }
    }

    /// Updates the Pfaffian pairing amplitudes.
    ///
    /// # Arguments
    /// * `new_amplitudes` - New pairing amplitude matrix
    pub fn update_pfaffian_amplitudes(&mut self, new_amplitudes: ndarray::Array2<Complex64>) {
        if let Some(pfaffian) = &mut self.pfaffian {
            pfaffian.update_pairing_amplitudes(new_amplitudes);
        }
    }

    /// Calculates projection counts (Gutzwiller + Jastrow) for a configuration.
    ///
    /// Reference: mVMC/src/mVMC/projection.c:MakeProjCnt()
    ///
    /// # Arguments
    /// * `config` - Spin configuration (0: empty, 1: up, 2: down, 3: both)
    ///
    /// # Returns
    /// * `Vec<i32>` - Projection counts [Gutzwiller counts, Jastrow counts]
    fn calculate_proj_counts(&self, config: &[u8]) -> Vec<i32> {
        let nsite = self.nsite;
        let n_gutzwiller = nsite;
        let n_jastrow = nsite * (nsite - 1) / 2;
        let mut proj_counts = vec![0i32; n_gutzwiller + n_jastrow];

        // Calculate occupation numbers: n0 (up), n1 (down)
        let mut n0 = vec![0i32; nsite];  // up spin
        let mut n1 = vec![0i32; nsite];  // down spin

        for site in 0..nsite {
            match config[site] {
                1 => n0[site] = 1,  // up only
                2 => n1[site] = 1,  // down only
                3 => { n0[site] = 1; n1[site] = 1; }  // both (doubly occupied)
                _ => {}  // empty
            }
        }

        // Gutzwiller factor: counts double occupancy n0[ri] * n1[ri]
        // Reference: mVMC/src/mVMC/projection.c:77-81
        for ri in 0..nsite {
            proj_counts[ri] = n0[ri] * n1[ri];
        }

        // Jastrow factor: counts (n_i - 1) * (n_j - 1) for i < j
        // Reference: mVMC/src/mVMC/projection.c:84-95
        let offset = n_gutzwiller;
        let mut idx_count = 0;
        for ri in 0..nsite {
            let xi = n0[ri] + n1[ri] - 1;

            for rj in (ri+1)..nsite {
                let xj = n0[rj] + n1[rj] - 1;
                proj_counts[offset + idx_count] = xi * xj;
                idx_count += 1;
            }
        }

        proj_counts
    }

    /// Calculates parameter derivatives (O-operators) for optimization.
    ///
    /// For Gutzwiller/Jastrow: O_k = ∂log(ψ)/∂θ_k = projCnt[k]
    /// For Slater/Pfaffian: O_k = Tr[Inv[M] * ∂M/∂θ_k] / ψ
    ///
    /// # Arguments
    /// * `config` - Spin configuration
    ///
    /// # Returns
    /// * `Vec<Complex64>` - Parameter derivatives for all parameters
    pub fn calculate_all_parameter_derivatives(&self, config: &[u8]) -> Vec<Complex64> {
        let mut derivatives = Vec::new();

        // 1. Slater determinant parameters (orbital parameters)
        if let Some(ref slater) = self.slater {
            let slater_derivs = slater.calculate_parameter_derivatives(config);
            eprintln!("DEBUG: Slater derivatives: {}", slater_derivs.len());
            derivatives.extend(slater_derivs);
        }

        // 2. Pfaffian parameters (not implemented yet)
        // if let Some(ref pfaffian) = self.pfaffian {
        //     let pfaffian_derivs = pfaffian.calculate_parameter_derivatives(config);
        //     derivatives.extend(pfaffian_derivs);
        // }

        // 3. Gutzwiller and Jastrow parameters
        // For these, O_k = ∂log(ψ)/∂θ_k = projCnt[k]
        // Reference: Gutzwiller factor = exp(Σ g_k * n_k) => ∂log/∂g_k = n_k
        let proj_counts = self.calculate_proj_counts(config);
        eprintln!("DEBUG: Proj counts: {}", proj_counts.len());
        eprintln!("DEBUG: Expected - Gutzwiller: {}, Jastrow: {}", self.nsite, self.nsite * (self.nsite - 1) / 2);
        for count in proj_counts {
            derivatives.push(Complex64::new(count as f64, 0.0));
        }

        eprintln!("DEBUG: Total derivatives: {}", derivatives.len());
        derivatives
    }

    /// Updates Gutzwiller and Jastrow parameters.
    ///
    /// # Arguments
    /// * `param_updates` - Parameter updates (same length as total parameters)
    /// * `learning_rate` - Learning rate for updates
    pub fn update_projection_parameters(&mut self, param_updates: &[f64], learning_rate: f64) {
        // Calculate offset (skip Slater parameters)
        let n_slater_params = if let Some(ref slater) = self.slater {
            slater.nsite() * slater.ne() * 2  // 2 for real + imaginary
        } else {
            0
        };

        let n_gutzwiller = self.nsite;
        let n_jastrow = self.nsite * (self.nsite - 1) / 2;

        // Update Gutzwiller parameters
        for k in 0..n_gutzwiller {
            let param_idx = n_slater_params + k;
            if param_idx < param_updates.len() {
                self.gutzwiller_params[k] -= learning_rate * param_updates[param_idx];
            }
        }

        // Update Jastrow parameters
        for k in 0..n_jastrow {
            let param_idx = n_slater_params + n_gutzwiller + k;
            if param_idx < param_updates.len() {
                self.jastrow_params[k] -= learning_rate * param_updates[param_idx];
            }
        }
    }
}

impl Clone for CombinedWavefunction {
    fn clone(&self) -> Self {
        Self {
            slater: self.slater.clone(),
            pfaffian: self.pfaffian.clone(),
            projectors: Vec::new(), // Cannot clone trait objects
            nsite: self.nsite,
            ne: self.ne,
            gutzwiller_params: self.gutzwiller_params.clone(),
            jastrow_params: self.jastrow_params.clone(),
        }
    }
}

impl std::fmt::Debug for CombinedWavefunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CombinedWavefunction(nsite={}, ne={}, slater={}, pfaffian={}, projectors={})",
               self.nsite, self.ne,
               self.slater.is_some(),
               self.pfaffian.is_some(),
               self.projectors.len())
    }
}

impl std::fmt::Display for CombinedWavefunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CombinedWavefunction(nsite={}, ne={}, slater={}, pfaffian={}, projectors={})",
               self.nsite, self.ne,
               self.slater.is_some(),
               self.pfaffian.is_some(),
               self.projectors.len())
    }
}

impl Wavefunction for CombinedWavefunction {
    fn calculate(&self, config: &[u8]) -> Result<Complex64> {
        if config.len() != self.nsite {
            return Err(anyhow::anyhow!("Configuration length mismatch"));
        }

        let mut amplitude = Complex64::new(1.0, 0.0);

        // Calculate Slater determinant contribution
        if let Some(ref slater) = self.slater {
            let slater_amp = slater.calculate(config)?;
            amplitude *= slater_amp;
        }

        // Calculate Pfaffian contribution
        if let Some(ref pfaffian) = self.pfaffian {
            let pfaffian_amp = pfaffian.calculate(config)?;
            amplitude *= pfaffian_amp;
        }

        // Apply projection operators
        for projector in &self.projectors {
            let spin_config = config.iter().map(|&s| {
                match s {
                    1 => crate::hamiltonian::Spin::Up,
                    2 => crate::hamiltonian::Spin::Down,
                    _ => crate::hamiltonian::Spin::Empty,
                }
            }).collect::<Vec<_>>();

            amplitude = projector.apply(amplitude, &spin_config);
        }

        // Apply Gutzwiller and Jastrow factors
        // ψ = ψ_0 * exp(Σ g_k * projCnt_k)
        // Reference: mVMC/src/mVMC/projection.c:LogProjVal()
        let proj_counts = self.calculate_proj_counts(config);
        let n_gutzwiller = self.nsite;
        let n_jastrow = self.nsite * (self.nsite - 1) / 2;

        let mut log_proj_val = 0.0;

        // Gutzwiller contribution
        for k in 0..n_gutzwiller {
            log_proj_val += self.gutzwiller_params[k] * proj_counts[k] as f64;
        }

        // Jastrow contribution
        for k in 0..n_jastrow {
            log_proj_val += self.jastrow_params[k] * proj_counts[n_gutzwiller + k] as f64;
        }

        // Apply exponential factor
        amplitude *= Complex64::from_polar(log_proj_val.exp(), 0.0);

        Ok(amplitude)
    }

    fn nsite(&self) -> usize {
        self.nsite
    }

    fn ne(&self) -> usize {
        self.ne
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use num_complex::Complex64;

    #[test]
    fn test_combined_wavefunction_creation() {
        let nsite = 4;
        let ne = 2;
        let wavefunction = CombinedWavefunction::new(nsite, ne).unwrap();

        assert_eq!(wavefunction.nsite(), nsite);
        assert_eq!(wavefunction.ne(), ne);
        assert!(wavefunction.slater().is_some()); // Now initialized by default
        assert!(wavefunction.pfaffian().is_none());
    }

    #[test]
    fn test_combined_wavefunction_with_slater() {
        let nsite = 4;
        let ne = 2;
        let orbitals = Array2::from_shape_fn((nsite, ne), |(i, j)| {
            Complex64::new(i as f64 + j as f64, 0.0)
        });
        let slater = SlaterDeterminant::new(nsite, ne, orbitals).unwrap();
        let wavefunction = CombinedWavefunction::with_slater(nsite, ne, slater).unwrap();

        assert_eq!(wavefunction.nsite(), nsite);
        assert_eq!(wavefunction.ne(), ne);
        assert!(wavefunction.slater().is_some());
        assert!(wavefunction.pfaffian().is_none());
    }

    #[test]
    fn test_wavefunction_calculation() {
        let nsite = 2;
        let ne = 2;
        let orbitals = Array2::from_shape_fn((nsite, ne), |(i, j)| {
            if i == j {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            }
        });
        let slater = SlaterDeterminant::new(nsite, ne, orbitals).unwrap();
        let wavefunction = CombinedWavefunction::with_slater(nsite, ne, slater).unwrap();

        let spin_config = vec![1, 2]; // up, down
        let value = wavefunction.calculate(&spin_config);

        assert!(value.norm() > 0.0);
    }
}

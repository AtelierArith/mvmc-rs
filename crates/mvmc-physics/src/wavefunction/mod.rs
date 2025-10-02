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
    /// For Heisenberg models (ne=0), this creates a wavefunction without Slater determinant.
    /// For fermionic models (ne>0), this creates a wavefunction with a plane wave Slater determinant.
    /// Reference: mVMC/src/mVMC/vmcmake.c - makeInitialSample()
    pub fn new(nsite: usize, ne: usize) -> Result<Self> {
        if ne == 0 {
            // Heisenberg model (ne=0): No Slater determinant needed
            let gutzwiller_params = vec![0.0; nsite];
            let n_jastrow = nsite * (nsite - 1) / 2;  // Number of unique pairs i < j
            let jastrow_params = vec![0.0; n_jastrow];

            Ok(Self {
                slater: None,  // No Slater determinant for spin models
                pfaffian: None,
                projectors: Vec::new(),
                nsite,
                ne,
                gutzwiller_params,
                jastrow_params,
            })
        } else {
            // Fermionic model (ne>0): Initialize with plane wave basis
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
            if self.ne == 0 {
                // Heisenberg model (ne=0): Update Jastrow parameters only
                let n_jastrow_params = self.nsite * (self.nsite - 1) / 2;
                if param_updates.len() >= n_jastrow_params {
                    for i in 0..n_jastrow_params {
                        self.jastrow_params[i] -= learning_rate * param_updates[i];
                    }
                }
            } else {
                // Fermionic model (ne>0): Update all parameters
                // Update Slater determinant parameters
                if let Some(slater) = &mut self.slater {
                    let n_slater_params = slater.nsite() * slater.ne() * 2;
                    if param_updates.len() >= n_slater_params {
                        slater.apply_parameter_updates(&param_updates[0..n_slater_params]);
                    }
                }

                // Update Gutzwiller and Jastrow parameters
                self.update_projection_parameters(param_updates, learning_rate);
            }
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
        if self.ne == 0 {
            // Heisenberg model (ne=0): Only Jastrow parameters
            self.nsite * (self.nsite - 1) / 2
        } else {
            // Fermionic model (ne>0): Full combined wavefunction
            let mut n_params = 0;

            // Slater determinant parameters
            if let Some(slater) = &self.slater {
                n_params += slater.nsite() * slater.ne() * 2;  // real + imaginary
            }

            // Gutzwiller/Jastrow (electron) parameters (kept for compatibility)
            n_params += self.nsite; // gutzwiller
            n_params += self.nsite * (self.nsite - 1) / 2; // jastrow (electron)

            // Spin-Jastrow projector parameters (one per projector)
            for _p in &self.projectors {
                // if projector exposes spin-jastrow, add one parameter
                if _p.spin_jastrow_pairs().is_some() {
                    n_params += 1;
                }
            }

            n_params
        }
    }

    /// Exports current variational parameters in C-compatible order.
    ///
    /// Order:
    /// - Slater orbitals flattened as (i,j) over sites and electrons
    /// - Gutzwiller parameters (per site)
    /// - Jastrow parameters (i<j pairs)
    pub fn export_parameters(&self) -> Vec<num_complex::Complex64> {
        use num_complex::Complex64;
        let mut params = Vec::new();

        if self.ne == 0 {
            // Heisenberg model (ne=0): Only Jastrow parameters
            let mut idx = 0;
            for _i in 0..self.nsite {
                for _j in (_i + 1)..self.nsite {
                    if idx < self.jastrow_params.len() {
                        params.push(Complex64::new(self.jastrow_params[idx], 0.0));
                        idx += 1;
                    }
                }
            }
        } else {
            // Fermionic model (ne>0): Full combined wavefunction
            // Slater determinant parameters
            if let Some(slater) = &self.slater {
                let orbs = slater.orbitals();
                let (ns, ne) = (slater.nsite(), slater.ne());
                for i in 0..ns {
                    for j in 0..ne {
                        params.push(orbs[[i, j]]);
                    }
                }
            }

            // Gutzwiller parameters
            for &g in &self.gutzwiller_params {
                params.push(Complex64::new(g, 0.0));
            }

            // Jastrow parameters (i<j)
            let mut idx = 0usize;
            for i in 0..self.nsite {
                for _j in (i + 1)..self.nsite {
                    if idx < self.jastrow_params.len() {
                        params.push(Complex64::new(self.jastrow_params[idx], 0.0));
                        idx += 1;
                    }
                }
            }
        }

        params
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

        if self.ne == 0 {
            // Heisenberg model (ne=0): Use Jastrow factor only
            // ψ = exp(Σ_{i<j} v_{ij} S_i^z S_j^z)
            self.calculate_heisenberg_amplitude(spin_config)
        } else {
            // Fermionic model (ne>0): Use full combined wavefunction
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
        if self.ne == 0 {
            // Heisenberg model (ne=0): Use Jastrow factor only
            self.calculate_heisenberg_ratio(spin_config, flip_site, flip_from, flip_to)
        } else {
            // Fermionic model (ne>0): Use full combined wavefunction
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
    }

    /// Calculates the projection weight for a spin configuration.
    ///
    /// # Arguments
    /// * `spin_config` - Spin configuration
    ///
    /// # Returns
    /// * `Complex64` - The projection weight
    fn calculate_projection_weight(&self, spin_config: &[u8]) -> Complex64 {
        use num_complex::Complex64 as C;
        let mut weight = C::new(1.0, 0.0);

        // Built-in Gutzwiller/Jastrow factors
        let mut expo = 0.0f64;
        let mut n_site = vec![0.0f64; self.nsite];
        for i in 0..self.nsite {
            let s = spin_config[i];
            n_site[i] = match s { 3 => 2.0, 1|2 => 1.0, _ => 0.0 };
        }
        for i in 0..self.nsite { expo += self.gutzwiller_params[i] * n_site[i]; }
        let mut idx = 0usize;
        for i in 0..self.nsite {
            for j in (i+1)..self.nsite {
                expo += self.jastrow_params[idx] * (n_site[i]*n_site[j]);
                idx += 1;
            }
        }
        // Doublon-holon (simplified site-local placeholders)
        // If needed, can be generalized to bond-based terms
        // Here, we just modulate doubles/holons by site parameters
        // to reflect DH-like correlation weights.
        // (Zeros by default, so no effect unless set.)
        // self.doublon_holon2_params / doublon_holon4_params applied additively in exponent
        // TODO: Add doublon/holon parameters if needed
        // if false { /* reserved for future bond-based DH */ }
        // for i in 0..self.nsite {
        //     if n_site[i] >= 1.5 { expo += self.doublon_holon2_params[i]; }
        //     if n_site[i] <= 0.5 { expo += self.doublon_holon4_params[i]; }
        // }
        weight *= C::new(expo.exp(), 0.0);

        // External projectors (e.g., spin-Jastrow)
        for projector in &self.projectors { weight *= projector.project(spin_config); }
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

    /// Calculates the Heisenberg model amplitude using Jastrow factor.
    ///
    /// For Heisenberg models (ne=0), the wavefunction is:
    /// ψ = exp(Σ_{i<j} v_{ij} S_i^z S_j^z)
    ///
    /// # Arguments
    /// * `spin_config` - Spin configuration (1: up, 2: down)
    ///
    /// # Returns
    /// * `Complex64` - The wavefunction amplitude
    fn calculate_heisenberg_amplitude(&self, spin_config: &[u8]) -> Complex64 {
        // Calculate Jastrow factor: exp(Σ_{i<j} v_{ij} S_i^z S_j^z)
        let mut log_amplitude = 0.0;
        let mut idx = 0;

        for i in 0..self.nsite {
            for _j in (i + 1)..self.nsite {
                let s_i = self.spin_value(spin_config[i]);
                let s_j = self.spin_value(spin_config[_j]);
                if idx < self.jastrow_params.len() {
                    let v_ij = self.jastrow_params[idx];
                    log_amplitude += v_ij * s_i * s_j;
                    idx += 1;
                }
            }
        }

        Complex64::from_polar(log_amplitude.exp(), 0.0)
    }

    /// Calculates the Heisenberg model ratio for a spin flip.
    ///
    /// For Heisenberg models (ne=0), the ratio is:
    /// ψ_new / ψ_old = exp(Σ_{j≠i} v_{ij} (S_i^new - S_i^old) S_j)
    ///
    /// # Arguments
    /// * `spin_config` - Current spin configuration
    /// * `flip_site` - Site to flip
    /// * `flip_from` - Current spin state
    /// * `flip_to` - New spin state
    ///
    /// # Returns
    /// * `Complex64` - The ratio of wavefunction values
    fn calculate_heisenberg_ratio(&self, spin_config: &[u8], flip_site: usize, flip_from: u8, flip_to: u8) -> Complex64 {
        let s_old = self.spin_value(flip_from);
        let s_new = self.spin_value(flip_to);
        let delta_s = s_new - s_old;

        // Calculate the change in log amplitude
        let mut delta_log_amplitude = 0.0;

        // Iterate over all pairs (i, j) with i < j
        let mut idx = 0;
        for i in 0..self.nsite {
            for j in (i + 1)..self.nsite {
                if i == flip_site || j == flip_site {
                    // This pair involves the flipped site
                    let other_site = if i == flip_site { j } else { i };
                    let s_other = self.spin_value(spin_config[other_site]);
                    if idx < self.jastrow_params.len() {
                        let v_ij = self.jastrow_params[idx];
                        delta_log_amplitude += v_ij * delta_s * s_other;
                    }
                }
                idx += 1;
            }
        }

        Complex64::from_polar(delta_log_amplitude.exp(), 0.0)
    }

    /// Converts spin configuration to spin value for Heisenberg models.
    ///
    /// # Arguments
    /// * `spin_state` - Spin state (1: up, 2: down)
    ///
    /// # Returns
    /// * `f64` - Spin value (+1.0 for up, -1.0 for down)
    fn spin_value(&self, spin_state: u8) -> f64 {
        match spin_state {
            1 => 1.0,   // spin up
            2 => -1.0,  // spin down
            _ => 0.0,   // empty or invalid (should not happen in Heisenberg models)
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

        // For spin models (ne==0), use spin correlation Jastrow: S_i^z S_j^z
        // Set on-site (Gutzwiller-like) counts to zero for spins
        // For fermionic models (ne>0), use standard counts
        for ri in 0..nsite {
            if self.ne == 0 {
                proj_counts[ri] = 0; // no on-site projection for pure spin model
            } else {
                proj_counts[ri] = n0[ri] * n1[ri]; // double occupancy
            }
        }

        // Pair terms (Jastrow):
        // - Spin model: S_i^z S_j^z with S^z=+1 (Up), -1 (Down)
        // - Fermion model: (n_i - 1) * (n_j - 1)
        let offset = n_gutzwiller;
        let mut idx_count = 0;
        for ri in 0..nsite {
            let xi = if self.ne == 0 {
                // Spin value +1 (Up) or -1 (Down)
                if n0[ri] == 1 && n1[ri] == 0 { 1 } else if n0[ri] == 0 && n1[ri] == 1 { -1 } else { 0 }
            } else {
                // Fermionic: (n - 1)
                n0[ri] + n1[ri] - 1
            };

            for rj in (ri+1)..nsite {
                let xj = if self.ne == 0 {
                    if n0[rj] == 1 && n1[rj] == 0 { 1 } else if n0[rj] == 0 && n1[rj] == 1 { -1 } else { 0 }
                } else {
                    n0[rj] + n1[rj] - 1
                };
                proj_counts[offset + idx_count] = xi * xj; // spin correlation or density correlation
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

        if self.ne == 0 {
            // Heisenberg model (ne=0): Only Jastrow parameters
            // For Jastrow parameters: O_k = ∂log(ψ)/∂v_k = S_i^z S_j^z
            for i in 0..self.nsite {
                for j in (i + 1)..self.nsite {
                    let s_i = self.spin_value(config[i]);
                    let s_j = self.spin_value(config[j]);
                    let derivative = s_i * s_j;
                    derivatives.push(Complex64::new(derivative, 0.0));
                }
            }
        } else {
            // Fermionic model (ne>0): Full combined wavefunction
            // 1. Slater determinant parameters (orbital parameters)
            if let Some(ref slater) = self.slater {
                let slater_derivs = slater.calculate_parameter_derivatives(config);
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

            for count in proj_counts {
                derivatives.push(Complex64::new(count as f64, 0.0));
            }
        }

        // 4. Spin-Jastrow projector parameter(s) (only for fermionic models)
        if self.ne > 0 {
            // For ψ = exp(α Σ_{<i,j>} σ_i σ_j), ∂logψ/∂α = Σ σ_i σ_j
            for p in &self.projectors {
                if let Some(pairs) = p.spin_jastrow_pairs() {
                    let mut s = 0.0;
                    for &(i, j) in pairs {
                        let si = match config.get(i).copied().unwrap_or(0) { 1 => 1.0, 2 => -1.0, _ => 0.0 };
                        let sj = match config.get(j).copied().unwrap_or(0) { 1 => 1.0, 2 => -1.0, _ => 0.0 };
                        s += si * sj;
                    }
                    derivatives.push(Complex64::new(s, 0.0));
                }
            }
        }

        derivatives
    }

    /// Updates Gutzwiller and Jastrow parameters.
    ///
    /// # Arguments
    /// * `param_updates` - Parameter updates (same length as total parameters)
    /// * `learning_rate` - Learning rate for updates
    pub fn update_projection_parameters(&mut self, param_updates: &[f64], learning_rate: f64) {
        if self.ne == 0 {
            // Heisenberg model (ne=0): Update Jastrow parameters only
            let n_jastrow = self.nsite * (self.nsite - 1) / 2;
            for k in 0..n_jastrow {
                if k < param_updates.len() {
                    self.jastrow_params[k] -= learning_rate * param_updates[k];
                }
            }
        } else {
            // Fermionic model (ne>0): Update all parameters
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

            // Update Jastrow parameters (electron)
            for k in 0..n_jastrow {
                let param_idx = n_slater_params + n_gutzwiller + k;
                if param_idx < param_updates.len() {
                    self.jastrow_params[k] -= learning_rate * param_updates[param_idx];
                }
            }

            // Spin-Jastrow projector parameters (only for fermionic models)
            let n_slater_params = if let Some(ref slater) = self.slater {
                slater.nsite() * slater.ne() * 2  // 2 for real + imaginary
            } else {
                0
            };
            let n_gutzwiller = self.nsite;
            let n_jastrow = self.nsite * (self.nsite - 1) / 2;
            let mut offset = n_slater_params + n_gutzwiller + n_jastrow;

            for p in &mut self.projectors {
                if p.spin_jastrow_pairs().is_some() {
                    if offset < param_updates.len() {
                        let delta = -learning_rate * param_updates[offset];
                        p.update_param(delta);
                    }
                    offset += 1;
                }
            }
        }

        // debug logs removed for clean CLI output
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

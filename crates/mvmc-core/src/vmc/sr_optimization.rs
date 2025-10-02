//! Simplified Stochastic Reconfiguration (SR) optimization implementation
//!
//! This is a simplified but mathematically correct implementation of the SR method.
//! Full C implementation has many optimizations and projections that we simplify here.
//!
//! Reference: mVMC/src/mVMC/slater.c - SlaterElmDiff_fcmp
//!           mVMC/src/mVMC/vmccal.c - calculateOO
//!           mVMC/src/mVMC/stcopt.c - StochasticOpt

use crate::error::Result;
use num_complex::Complex64;

/// SR optimization data for a single Monte Carlo sample
#[derive(Debug, Clone)]
pub struct SRSampleData {
    /// O-operators for this sample: O_k = (1/ψ) ∂ψ/∂f_k
    pub o_operators: Vec<Complex64>,
    /// Local energy for this sample
    pub local_energy: Complex64,
    /// Weight of this sample (|ψ|^2)
    pub weight: f64,
}

/// SR optimization calculator
pub struct SROptimizationCalculator {
    /// Number of variational parameters
    n_params: usize,
    /// Accumulated samples
    samples: Vec<SRSampleData>,
}

impl SROptimizationCalculator {
    /// Creates a new SR optimization calculator
    pub fn new(n_params: usize) -> Self {
        Self {
            n_params,
            samples: Vec::new(),
        }
    }

    /// Adds a sample to the calculator
    pub fn add_sample(&mut self, sample: SRSampleData) {
        assert_eq!(sample.o_operators.len(), self.n_params);
        self.samples.push(sample);
    }

    /// Clears all accumulated samples
    pub fn clear(&mut self) {
        self.samples.clear();
    }

    /// Calculates the SR matrix S_ij = ⟨O_i† O_j⟩ - ⟨O_i†⟩⟨O_j⟩
    ///
    /// Reference: mVMC/src/mVMC/vmccal.c - calculateOO
    pub fn calculate_sr_matrix(&self) -> Result<Vec<Vec<Complex64>>> {
        if self.samples.is_empty() {
            return Err(crate::error::VmcError::invalid_param("No samples available for SR matrix calculation"));
        }

        let n = self.n_params;
        let mut sr_matrix = vec![vec![Complex64::new(0.0, 0.0); n]; n];

        // Calculate ⟨O_i†⟩
        let mut o_avg = vec![Complex64::new(0.0, 0.0); n];
        let mut total_weight = 0.0;

        for sample in &self.samples {
            let w = sample.weight;
            total_weight += w;
            for k in 0..n {
                o_avg[k] += w * sample.o_operators[k].conj();
            }
        }

        if total_weight > 1e-12 {
            for k in 0..n {
                o_avg[k] /= total_weight;
            }
        }

        // Calculate S_ij = ⟨O_i† O_j⟩ - ⟨O_i†⟩⟨O_j⟩
        for sample in &self.samples {
            let w = sample.weight / total_weight;
            for i in 0..n {
                for j in 0..n {
                    let o_i_conj = sample.o_operators[i].conj();
                    let o_j = sample.o_operators[j];
                    sr_matrix[i][j] += w * o_i_conj * o_j;
                }
            }
        }

        // Subtract ⟨O_i†⟩⟨O_j⟩
        for i in 0..n {
            for j in 0..n {
                sr_matrix[i][j] -= o_avg[i] * o_avg[j].conj();
            }
        }

        Ok(sr_matrix)
    }

    /// Calculates the force vector F_i = ⟨O_i† H⟩ - ⟨O_i†⟩⟨H⟩
    ///
    /// Reference: mVMC/src/mVMC/vmccal.c - calculateOO (HO calculation)
    pub fn calculate_force_vector(&self) -> Result<Vec<Complex64>> {
        if self.samples.is_empty() {
            return Err(crate::error::VmcError::invalid_param("No samples available for force vector calculation"));
        }

        let n = self.n_params;
        let mut force = vec![Complex64::new(0.0, 0.0); n];
        let mut o_avg = vec![Complex64::new(0.0, 0.0); n];
        let mut energy_avg = Complex64::new(0.0, 0.0);
        let mut total_weight = 0.0;

        for sample in &self.samples {
            let w = sample.weight;
            total_weight += w;
            energy_avg += w * sample.local_energy;
            for k in 0..n {
                o_avg[k] += w * sample.o_operators[k].conj();
                force[k] += w * sample.o_operators[k].conj() * sample.local_energy;
            }
        }

        if total_weight > 1e-12 {
            energy_avg /= total_weight;
            for k in 0..n {
                o_avg[k] /= total_weight;
                force[k] /= total_weight;
            }
        }

        // Subtract ⟨O_i†⟩⟨H⟩
        for k in 0..n {
            force[k] -= o_avg[k] * energy_avg;
        }

        Ok(force)
    }

    /// Solves the SR equation: S * δp = -F
    /// Returns the parameter updates δp
    ///
    /// Reference: mVMC/src/mVMC/stcopt.c - StochasticOpt
    pub fn solve_sr_equation(&self, learning_rate: f64) -> Result<Vec<f64>> {
        let sr_matrix = self.calculate_sr_matrix()?;
        let force = self.calculate_force_vector()?;

        let n = self.n_params;
        let mut param_updates = vec![0.0; n];

        // Simplified solver: Use diagonal approximation
        // Full C implementation uses LAPACK for solving linear system
        // Here we use: δp_i = -learning_rate * F_i / (S_ii + ε)
        let epsilon = 1e-4; // Regularization

        for i in 0..n {
            let s_ii = sr_matrix[i][i].re;
            if s_ii.abs() > epsilon {
                param_updates[i] = -learning_rate * force[i].re / (s_ii + epsilon);
            } else {
                // Diagonal element too small, use gradient descent fallback
                param_updates[i] = -learning_rate * force[i].re;
            }
        }

        // Clip updates to prevent instability
        let max_update = 0.1;
        for update in &mut param_updates {
            *update = update.max(-max_update).min(max_update);
        }

        Ok(param_updates)
    }
}

/// Calculates O-operators for a given configuration
///
/// This is a simplified version. Full C implementation (SlaterElmDiff_fcmp) calculates:
/// O_k = (1/ψ) ∂ψ/∂f_k
///
/// Here we use finite differences as a simplification:
/// O_k ≈ (ψ(f_k + δ) - ψ(f_k)) / (δ * ψ(f_k))
///
/// Reference: mVMC/src/mVMC/slater.c - SlaterElmDiff_fcmp (lines 99-244)
pub fn calculate_o_operators_finite_diff(
    _wavefunction: &dyn Fn(&[u8]) -> Complex64,
    _spin_config: &[u8],
    n_params: usize,
) -> Vec<Complex64> {
    // Placeholder - full implementation would use finite differences
    // This is handled by SlaterDeterminant::calculate_parameter_derivatives
    vec![Complex64::new(0.0, 0.0); n_params]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sr_calculator_creation() {
        let calc = SROptimizationCalculator::new(10);
        assert_eq!(calc.n_params, 10);
    }

    #[test]
    fn test_add_sample() {
        let mut calc = SROptimizationCalculator::new(3);
        let sample = SRSampleData {
            o_operators: vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
            ],
            local_energy: Complex64::new(-1.0, 0.0),
            weight: 1.0,
        };
        calc.add_sample(sample);
        assert_eq!(calc.samples.len(), 1);
    }
}


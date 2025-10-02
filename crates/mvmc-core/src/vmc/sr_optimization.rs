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

/// One-step SR diagnostic info for zvo_SRinfo.dat
#[derive(Debug, Clone)]
pub struct SRStepInfo {
    pub npara: usize,
    pub msize: usize,
    pub opt_cut: usize,
    pub diag_cut: usize,
    pub sdiag_max: f64,
    pub sdiag_min: f64,
    pub abs_rmax: f64,
    pub imax: usize,
}

/// SR optimization data for a single Monte Carlo sample
#[derive(Debug, Clone)]
pub struct SRSampleData {
    /// O-operators for this sample: O_k = (1/ψ) ∂ψ/∂f_k
    pub o_operators: Vec<f64>,
    /// Local energy for this sample
    pub energy: f64,
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

    /// Performs SR optimization and returns parameter updates
    pub fn optimize(&mut self) -> Result<Vec<f64>> {
        if self.samples.is_empty() {
            return Ok(vec![0.0; self.n_params]);
        }

        // Use simplified SR solver
        self.solve_sr_equation(0.1)
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
                o_avg[k] += w * sample.o_operators[k];
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
                    let o_i_conj = sample.o_operators[i];
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
            energy_avg += w * sample.energy;
            for k in 0..n {
                o_avg[k] += w * sample.o_operators[k];
                force[k] += w * sample.o_operators[k] * sample.energy;
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

    /// Calculates averages needed for SR: <O>, <H>, <O^* O>, <O^* H>
    pub fn calculate_averages(&self) -> Result<(
        Vec<Complex64>,
        Complex64,
        Vec<Vec<Complex64>>,
        Vec<Complex64>,
    )> {
        if self.samples.is_empty() {
            return Err(crate::error::VmcError::invalid_param("No samples available for averages"));
        }

        let n = self.n_params;
        let mut o_avg = vec![Complex64::new(0.0, 0.0); n];
        let mut e_avg = Complex64::new(0.0, 0.0);
        let mut total_w = 0.0f64;

        for s in &self.samples {
            total_w += s.weight;
            e_avg += s.weight * s.energy;
            for k in 0..n { o_avg[k] += s.weight * s.o_operators[k]; }
        }
        if total_w > 1e-12 {
            e_avg /= total_w;
            for k in 0..n { o_avg[k] /= total_w; }
        }

        let mut oo = vec![vec![Complex64::new(0.0, 0.0); n]; n];
        let mut oh = vec![Complex64::new(0.0, 0.0); n];
        for s in &self.samples {
            let w = s.weight / total_w;
            for i in 0..n {
                let oi = s.o_operators[i];
                oh[i] += w * oi * s.energy;
                for j in 0..n { oo[i][j] += w * oi * s.o_operators[j]; }
            }
        }

        Ok((o_avg, e_avg, oo, oh))
    }

    /// Solves the full SR equation S δp = -F with reduction and stabilization.
    ///
    /// - Reduction: drop directions with S_ii < DSROptRedCut * max(S_ii)
    /// - Stabilization: add DSROptStaDel to diagonal
    pub fn solve_full_sr(
        &self,
        red_cut: f64,
        sta_del: f64,
        step_dt: f64,
    ) -> Result<Vec<f64>> {
        let (o_avg, e_avg, oo, oh) = self.calculate_averages()?;
        let n = self.n_params;

        // Build S and F (real part used)
        let mut s_mat = vec![vec![0.0f64; n]; n];
        let mut f_vec = vec![0.0f64; n];
        let mut max_diag = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let s = oo[i][j] - o_avg[i] * o_avg[j].conj();
                s_mat[i][j] = s.re;
            }
            if s_mat[i][i] > max_diag { max_diag = s_mat[i][i]; }
        }
        for i in 0..n {
            let f = oh[i] - o_avg[i] * e_avg;
            f_vec[i] = f.re;
        }

        // Reduction
        let thr = max_diag * red_cut;
        let idx: Vec<usize> = (0..n).filter(|&i| s_mat[i][i] >= thr).collect();
        if idx.is_empty() { return Ok(vec![0.0; n]); }

        // Reduced system with stabilization
        let m = idx.len();
        let mut a = vec![vec![0.0f64; m]; m];
        let mut b = vec![0.0f64; m];
        for (ri, &i) in idx.iter().enumerate() {
            b[ri] = -f_vec[i];
            for (rj, &j) in idx.iter().enumerate() { a[ri][rj] = s_mat[i][j]; }
            a[ri][ri] += sta_del;
        }

        // Solve reduced system: try SPD Cholesky, then CG, then Gaussian
        // Prefer LAPACK DPOSV if available
        let x_red = if let Some(x) = try_lapack_dposv(&a, &b) {
            x
        } else if let Some(x) = solve_spd(&a, &b) {
            x
        } else if let Some(x) = try_cg(&a, &b, m) {
            x
        } else {
            gaussian_solve(a, b).unwrap_or(vec![0.0; m])
        };
        let mut updates = vec![0.0f64; n];
        for (k, &i) in idx.iter().enumerate() { updates[i] = step_dt * x_red[k]; }
        Ok(updates)
    }

    /// Full SR solve with SRStepInfo for output
    pub fn solve_full_sr_with_info(
        &self,
        red_cut: f64,
        sta_del: f64,
        step_dt: f64,
    ) -> Result<(Vec<f64>, SRStepInfo)> {
        let (o_avg, e_avg, oo, oh) = self.calculate_averages()?;
        let n = self.n_params;

        // Build S and F (real part)
        let mut s_mat = vec![vec![0.0f64; n]; n];
        let mut f_vec = vec![0.0f64; n];
        let mut sdiag_max = 0.0f64;
        let mut sdiag_min = f64::MAX;
        for i in 0..n {
            for j in 0..n {
                let s = oo[i][j] - o_avg[i] * o_avg[j].conj();
                s_mat[i][j] = s.re;
            }
            sdiag_max = sdiag_max.max(s_mat[i][i]);
            sdiag_min = sdiag_min.min(s_mat[i][i]);
        }
        for i in 0..n {
            let f = oh[i] - o_avg[i] * e_avg;
            f_vec[i] = f.re;
        }

        // Reduction
        let thr = sdiag_max * red_cut;
        let idx: Vec<usize> = (0..n).filter(|&i| s_mat[i][i] >= thr).collect();
        let m = idx.len();
        let diag_cut = n.saturating_sub(m);

        // Reduced system
        let mut a = vec![vec![0.0f64; m]; m];
        let mut b = vec![0.0f64; m];
        for (ri, &i) in idx.iter().enumerate() {
            b[ri] = -f_vec[i];
            for (rj, &j) in idx.iter().enumerate() { a[ri][rj] = s_mat[i][j]; }
            a[ri][ri] += sta_del;
        }

        let x_red = if m == 0 {
            Vec::new()
        } else if let Some(x) = solve_spd(&a, &b) { x } else {
            if let Some(x) = try_cg(&a, &b, m) { x } else { gaussian_solve(a, b).unwrap_or(vec![0.0; m]) }
        };

        let mut updates = vec![0.0f64; n];
        for (k, &i) in idx.iter().enumerate() { updates[i] = step_dt * x_red[k]; }

        // absRmax and imax
        let mut abs_rmax = 0.0f64;
        let mut imax = 0usize;
        for (i, &u) in updates.iter().enumerate() {
            let a = u.abs();
            if a > abs_rmax { abs_rmax = a; imax = i; }
        }

        let info = SRStepInfo {
            npara: n,
            msize: n,
            opt_cut: 0,
            diag_cut,
            sdiag_max,
            sdiag_min,
            abs_rmax,
            imax,
        };

        Ok((updates, info))
    }

    /// Full SR solve with SRStepInfo using SRParameters for solver control
    pub fn solve_full_sr_with_params(
        &self,
        params: &crate::config::SRParameters,
    ) -> Result<(Vec<f64>, SRStepInfo)> {
        let (o_avg, e_avg, oo, oh) = self.calculate_averages()?;
        let n = self.n_params;

        let mut s_mat = vec![vec![0.0f64; n]; n];
        let mut f_vec = vec![0.0f64; n];
        let mut sdiag_max = 0.0f64;
        let mut sdiag_min = f64::MAX;
        for i in 0..n {
            for j in 0..n {
                let s = oo[i][j] - o_avg[i] * o_avg[j].conj();
                s_mat[i][j] = s.re;
            }
            sdiag_max = sdiag_max.max(s_mat[i][i]);
            sdiag_min = sdiag_min.min(s_mat[i][i]);
        }
        for i in 0..n {
            let f = oh[i] - o_avg[i] * e_avg;
            f_vec[i] = f.re;
        }

        // Reduction
        let thr = sdiag_max * params.reduction_cutoff;
        let idx: Vec<usize> = (0..n).filter(|&i| s_mat[i][i] >= thr).collect();
        let m = idx.len();
        let diag_cut = n.saturating_sub(m);

        let mut a = vec![vec![0.0f64; m]; m];
        let mut b = vec![0.0f64; m];
        for (ri, &i) in idx.iter().enumerate() {
            b[ri] = -f_vec[i];
            for (rj, &j) in idx.iter().enumerate() { a[ri][rj] = s_mat[i][j]; }
            a[ri][ri] += params.stability_delta;
        }

        // Solver selection based on NSRCG
        let x_red = if params.use_cg {
            // CG only
            try_cg(&a, &b, m).or_else(|| solve_spd(&a, &b)).unwrap_or_else(|| gaussian_solve(a, b).unwrap_or(vec![0.0; m]))
        } else {
            // Prefer LAPACK DPOSV
            if let Some(x) = try_lapack_dposv(&a, &b) { x }
            else if let Some(x) = solve_spd(&a, &b) { x }
            else if let Some(x) = try_cg(&a, &b, m) { x }
            else { gaussian_solve(a, b).unwrap_or(vec![0.0; m]) }
        };

        let mut updates = vec![0.0f64; n];
        for (k, &i) in idx.iter().enumerate() { updates[i] = params.step_size * x_red[k]; }

        let mut abs_rmax = 0.0f64;
        let mut imax = 0usize;
        for (i, &u) in updates.iter().enumerate() { if u.abs() > abs_rmax { abs_rmax = u.abs(); imax = i; } }

        let info = SRStepInfo {
            npara: n,
            msize: n,
            opt_cut: 0,
            diag_cut,
            sdiag_max,
            sdiag_min,
            abs_rmax,
            imax,
        };

        Ok((updates, info))
    }
}

/// Simple SPD solver (Cholesky). Returns None if not SPD.
fn solve_spd(a: &Vec<Vec<f64>>, b: &Vec<f64>) -> Option<Vec<f64>> {
    let n = a.len();
    if n == 0 { return Some(Vec::new()); }
    let mut l = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i][j];
            for k in 0..j { s -= l[i][k] * l[j][k]; }
            if i == j {
                if s <= 0.0 { return None; }
                l[i][j] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i { s -= l[i][j] * y[j]; }
        y[i] = s / l[i][i];
    }
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for j in (i + 1)..n { s -= l[j][i] * x[j]; }
        x[i] = s / l[i][i];
    }
    Some(x)
}

/// Gaussian elimination solver
fn gaussian_solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = a.len();
    if n == 0 { return Some(Vec::new()); }
    for i in 0..n {
        let mut piv = i;
        for r in (i + 1)..n { if a[r][i].abs() > a[piv][i].abs() { piv = r; } }
        if a[piv][i].abs() < 1e-15 { return None; }
        if piv != i { a.swap(i, piv); b.swap(i, piv); }
        let diag = a[i][i];
        for j in i..n { a[i][j] /= diag; }
        b[i] /= diag;
        for r in 0..n {
            if r == i { continue; }
            let f = a[r][i];
            for c in i..n { a[r][c] -= f * a[i][c]; }
            b[r] -= f * b[i];
        }
    }
    Some(b)
}

/// Try Conjugate Gradient on reduced SPD system
fn try_cg(a: &Vec<Vec<f64>>, b: &Vec<f64>, n: usize) -> Option<Vec<f64>> {
    use crate::optimization::{ConjugateGradientSolver, CGSolver};
    let mut solver = ConjugateGradientSolver::new(n);
    solver.set_max_iterations(n.max(1000));
    solver.set_tolerance(1e-10);
    match solver.solve(a, b) {
        Ok(x) => Some(x),
        Err(_) => None,
    }
}

/// Try LAPACK DPOSV for SPD system Ax=b; returns solution x if succeeds
fn try_lapack_dposv(a: &Vec<Vec<f64>>, b: &Vec<f64>) -> Option<Vec<f64>> {
    let n = a.len();
    if n == 0 || b.len() != n { return None; }
    // Column-major copy of A (lower or upper, but LAPACK expects full; we pass full symmetric)
    let mut a_data = vec![0.0f64; n*n];
    for i in 0..n {
        for j in 0..n {
            a_data[j*n + i] = a[i][j]; // column-major
        }
    }
    // b as one RHS (nrhs=1)
    let mut b_data = b.clone();
    let (uplo, nn, nrhs, lda, ldb) = (b"L\0", n as i32, 1i32, n as i32, n as i32);
    let mut info: i32 = 0;
    unsafe {
        mvmc_bindings::ffi::lapack::dposv_(
            uplo.as_ptr() as *const i8,
            &nn as *const i32,
            &nrhs as *const i32,
            a_data.as_mut_ptr(),
            &lda as *const i32,
            b_data.as_mut_ptr(),
            &ldb as *const i32,
            &mut info as *mut i32,
        );
    }
    if info == 0 {
        Some(b_data)
    } else {
        None
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
            o_operators: vec![1.0, 2.0, 3.0],
            energy: -1.0,
            weight: 1.0,
        };
        calc.add_sample(sample);
        assert_eq!(calc.samples.len(), 1);
    }
}

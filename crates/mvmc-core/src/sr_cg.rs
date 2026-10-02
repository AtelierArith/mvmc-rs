//! Sampled matrix operator for Julia's standard stochastic-reconfiguration CG.

use crate::{ExpertModeData, VmcOptimizationState};
use std::fs::OpenOptions;
use std::io::{self, Write};
use std::path::Path;

/// Apply Julia's standard sampled SR-CG step, writing diagnostics before updates.
/// Returns zero for finite increments, including iteration-limit/breakdown exits.
pub fn stochastic_opt_cg(
    data: &mut ExpertModeData,
    state: &VmcOptimizationState,
    output_dir: Option<&Path>,
) -> io::Result<i32> {
    let n_proj = data.projection_layout().n_proj;
    let n_para = data.count_variational_parameters();
    let complex = crate::run::get_all_complex_flag(data);
    let offset = if complex { 2 } else { 1 };
    let full = offset * n_para;
    if full == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "SR-CG requires a nonempty parameter variance array",
        ));
    }
    if data.optimization_flags.is_empty() {
        data.optimization_flags = vec![1; 2 * n_para];
    }
    let size = offset * state.sr_opt.sr_opt_size;
    let oo: Vec<f64> = if complex {
        state.sr_opt.sr_opt_oo.iter().map(|z| z.re).collect()
    } else {
        state.sr_opt.sr_opt_oo_real.clone()
    };
    let ho: Vec<f64> = if complex {
        state.sr_opt.sr_opt_ho.iter().map(|z| z.re).collect()
    } else {
        state.sr_opt.sr_opt_ho_real.clone()
    };
    let variance: Vec<f64> = (0..full)
        .map(|pi| {
            let idx = pi + offset;
            oo.get(size + idx).copied().unwrap_or(0.0) - oo.get(idx).copied().unwrap_or(0.0).powi(2)
        })
        .collect();
    let (maximum, minimum) = if variance.iter().any(|v| v.is_nan()) {
        (f64::NAN, f64::NAN)
    } else {
        (
            variance.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            variance.iter().copied().fold(f64::INFINITY, f64::min),
        )
    };
    let threshold = maximum * data.modpara.dsr_opt_red_cut;
    let mut mapping = Vec::new();
    let mut opt_cut = 0;
    let mut diag_cut = 0;
    for (pi, &v) in variance.iter().enumerate() {
        let flag = if complex { pi } else { 2 * pi };
        if data.optimization_flags.get(flag).copied().unwrap_or(0) != 1 {
            opt_cut += 1;
        } else if v < threshold {
            diag_cut += 1;
        } else {
            mapping.push(pi);
        }
    }
    if mapping.is_empty() {
        return Ok(0);
    }
    let samples = data.modpara.nvmc_sample.max(0) as usize;
    let mut operator = SampledSrOperator::new(mapping.len(), samples, complex);
    let mut gradient = vec![0.0; mapping.len()];
    let dt = 2.0 * data.modpara.dsr_opt_step_dt;
    for (si, &pi) in mapping.iter().enumerate() {
        let idx = pi + offset;
        operator.mean[si] = oo[idx];
        operator.diagonal[si] = variance[pi];
        gradient[si] = -dt * (ho[idx] - ho[0] * oo[idx]);
        for s in 0..samples {
            let src = s * size + idx;
            let dst = s * mapping.len() + si;
            if complex {
                if let Some(o) = state.sr_opt.sr_opt_o_store.get(src) {
                    operator.real_samples[dst] = o.re;
                    operator.imag_samples[dst] = o.im;
                }
            } else if let Some(&o) = state.sr_opt.sr_opt_o_store_real.get(src) {
                operator.real_samples[dst] = o;
            }
        }
    }
    let max_iterations = if data.modpara.nsr_opt_cg_max_iter > 0 {
        data.modpara.nsr_opt_cg_max_iter as usize
    } else {
        mapping.len()
    };
    let result = operator.solve(
        &gradient,
        1.0 / state.energy.wc.re,
        data.modpara.dsr_opt_sta_del,
        data.modpara.dsr_opt_cg_tol,
        max_iterations,
    );
    let info = i32::from(result.solution.iter().any(|x| !x.is_finite()));
    if let Some(dir) = output_dir {
        let prefix = if data.modpara.c_data_file_head.is_empty() {
            "zvo"
        } else {
            &data.modpara.c_data_file_head
        };
        let path = dir.join(format!("{prefix}_SRinfo.dat"));
        let header = !path.exists() || path.metadata()?.len() == 0;
        let mut file = OpenOptions::new().create(true).append(true).open(path)?;
        if header {
            writeln!(
                file,
                "#Npara Msize optCut diagCut sDiagMax  sDiagMin    absRmax       imax"
            )?;
        }
        let mut imax = 0;
        for i in 1..result.solution.len() {
            if result.solution[imax].abs() < result.solution[i].abs() {
                imax = i;
            }
        }
        writeln!(
            file,
            "{full:5} {:5} {opt_cut:5} {diag_cut:5} {} {} {} {:5}, {}",
            mapping.len(),
            cg_number(maximum),
            cg_number(minimum),
            cg_number(result.solution[imax]),
            mapping[imax],
            result.iterations
        )?;
    }
    if info == 0 {
        for (&pi, &x) in mapping.iter().zip(&result.solution) {
            let para = if complex { pi / 2 } else { pi };
            let (re, im) = if complex && pi % 2 != 0 {
                (0.0, x)
            } else {
                (x, 0.0)
            };
            crate::sr::update_parameter_value(data, para, re, im, n_proj);
        }
    }
    Ok(info)
}

fn cg_number(x: f64) -> String {
    let raw = format!("{x:+.5e}");
    let signed = if let Some(rest) = raw.strip_prefix('+') {
        format!(" {rest}")
    } else {
        raw
    };
    if let Some((mantissa, exp)) = signed.split_once('e') {
        let exponent: i32 = exp.parse().expect("numeric exponent");
        format!("{mantissa}e{exponent:+03}")
    } else {
        signed
    }
}

/// Dot product with Julia/C's explicit sequential accumulation order.
pub fn sequential_dot(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len());
    let mut sum = 0.0;
    for i in 0..p.len() {
        sum += p[i] * q[i];
    }
    sum
}

/// Result of the standard CG iteration loop (not a convergence status).
#[derive(Debug)]
pub struct CgSolution {
    /// Solution increment, initialized to zero for every solve.
    pub solution: Vec<f64>,
    /// Julia iteration count, including the iteration that detects breakdown.
    pub iterations: usize,
    /// Final residual, including Julia's periodic explicit recomputation.
    pub residual: Vec<f64>,
    /// Final search direction, retained for deterministic solver diagnostics.
    pub direction: Vec<f64>,
}

/// Apply the covariance matrix from saved samples without materializing S.
/// Sample matrices have shape [active component, sample] in column-major order.
pub struct SampledSrOperator {
    /// Real part of the normalized component means.
    pub mean: Vec<f64>,
    /// Normalized component variances before diagonal regularization.
    pub diagonal: Vec<f64>,
    /// Raw real parts of sqrt(weight)*O for active components.
    pub real_samples: Vec<f64>,
    /// Raw imaginary parts; empty for a real parameter layout.
    pub imag_samples: Vec<f64>,
    components: usize,
    samples: usize,
    complex: bool,
    y_real: Vec<f64>,
    y_imag: Vec<f64>,
}

impl SampledSrOperator {
    /// Solve with Julia's convergence threshold and 20-iteration residual refresh.
    /// A small search-direction product ends the loop without an error, as in
    /// Julia. The caller must check the returned increments for finiteness.
    pub fn solve(
        &mut self,
        gradient: &[f64],
        inv_weight: f64,
        shift: f64,
        tolerance: f64,
        max_iterations: usize,
    ) -> CgSolution {
        let n = self.components;
        assert_eq!(gradient.len(), n);
        let threshold = tolerance.powi(2) * (n as f64).powi(2);
        let mut solution = vec![0.0; n];
        let mut direction = gradient.to_vec();
        let mut residual = gradient.to_vec();
        let mut product = vec![0.0; n];
        let mut delta = sequential_dot(&residual, &residual);
        let mut iterations = 0;
        for iteration in 1..=max_iterations {
            iterations = iteration;
            if delta < threshold {
                iterations = iteration - 1;
                break;
            }
            self.apply(&mut product, &direction, inv_weight, shift);
            let dq = sequential_dot(&direction, &product);
            if dq.abs() < 1e-30 {
                break;
            }
            let alpha = delta / dq;
            for i in 0..n {
                solution[i] += alpha * direction[i];
            }
            if iteration % 20 == 0 {
                self.apply(&mut residual, &solution, inv_weight, shift);
                for i in 0..n {
                    residual[i] = gradient[i] - residual[i];
                }
            } else {
                for i in 0..n {
                    residual[i] -= alpha * product[i];
                }
            }
            let delta_new = sequential_dot(&residual, &residual);
            let beta = delta_new / delta;
            delta = delta_new;
            for i in 0..n {
                direction[i] = residual[i] + beta * direction[i];
            }
        }
        CgSolution {
            solution,
            iterations,
            residual,
            direction,
        }
    }

    /// Allocate the sampled operator and its reusable intermediate vectors.
    pub fn new(components: usize, samples: usize, complex: bool) -> Self {
        Self {
            mean: vec![0.0; components],
            diagonal: vec![0.0; components],
            real_samples: vec![0.0; components * samples],
            imag_samples: vec![0.0; if complex { components * samples } else { 0 }],
            components,
            samples,
            complex,
            y_real: vec![0.0; samples],
            y_imag: vec![0.0; if complex { samples } else { 0 }],
        }
    }

    /// Compute z = inv_weight*Re(O Oᴴ)*x - mean*(meanᵀx) + shift*diag*x.
    /// The sampled products use Julia's GEMV order; corrections use its
    /// sequential dot product. MPI reduction belongs before the corrections.
    pub fn apply(&mut self, z: &mut [f64], x: &[f64], inv_weight: f64, shift: f64) {
        let n = self.components;
        assert_eq!(z.len(), n);
        assert_eq!(x.len(), n);
        assert_eq!(self.mean.len(), n);
        assert_eq!(self.diagonal.len(), n);
        assert_eq!(self.real_samples.len(), n * self.samples);
        if self.complex {
            assert_eq!(self.imag_samples.len(), n * self.samples);
        }
        if n == 0 {
            return;
        }
        if self.samples == 0 {
            z.fill(0.0);
        } else {
            crate::serial_blas::initialize();
            let rows = i32::try_from(n).expect("CG component count must fit BLAS LP64");
            let cols = i32::try_from(self.samples).expect("CG sample count must fit BLAS LP64");
            // SAFETY: matrix buffers have exactly rows*cols entries, leading
            // dimensions are rows, and every input/output vector has the
            // required length. Mutable outputs never alias matrix/input views.
            unsafe {
                blas::dgemv(
                    b'T',
                    rows,
                    cols,
                    1.0,
                    &self.real_samples,
                    rows,
                    x,
                    1,
                    0.0,
                    &mut self.y_real,
                    1,
                );
                if self.complex {
                    blas::dgemv(
                        b'T',
                        rows,
                        cols,
                        1.0,
                        &self.imag_samples,
                        rows,
                        x,
                        1,
                        0.0,
                        &mut self.y_imag,
                        1,
                    );
                }
                blas::dgemv(
                    b'N',
                    rows,
                    cols,
                    1.0,
                    &self.real_samples,
                    rows,
                    &self.y_real,
                    1,
                    0.0,
                    z,
                    1,
                );
                if self.complex {
                    blas::dgemv(
                        b'N',
                        rows,
                        cols,
                        1.0,
                        &self.imag_samples,
                        rows,
                        &self.y_imag,
                        1,
                        1.0,
                        z,
                        1,
                    );
                }
            }
        }
        let coef = sequential_dot(&self.mean, x);
        for i in 0..n {
            z[i] = inv_weight * z[i] - coef * self.mean[i] + shift * self.diagonal[i] * x[i];
        }
    }
}

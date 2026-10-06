//! Synthetic SR problems for the device pipeline gate and benchmark (issue #447).
//!
//! The store has the structure of the real `sr_opt_o_store_real`: `[n, samples]` column-major,
//! row 0 is the constant `O_0 = 1` (so that `G[p, 0]` is the mean of `O_p`), the other rows are
//! sampled derivatives, all scaled by `1/sqrt(samples)` (equal weights), so `G` is the
//! normalized second-moment matrix and `S = G - mean mean^T` the sample covariance. A latent
//! factor model gives a controllable condition number: `O_p = sum_f L_pf z_f + noise`.

/// Deterministic generator (uniform in `(-1, 1)`, sum of three is close to normal).
pub struct Lcg(pub u64);

impl Lcg {
    /// Next uniform value in `(-1, 1)`.
    pub fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    /// Approximately standard normal.
    pub fn normal(&mut self) -> f64 {
        (self.uniform() + self.uniform() + self.uniform()) * 1.0
    }
}

/// One synthetic SR problem.
pub struct SrProblem {
    /// Rows of the store (`1 + NPara`).
    pub n: usize,
    /// Samples.
    pub samples: usize,
    /// Store `[n, samples]` column-major.
    pub store: Vec<f64>,
    /// `HO` vector (length `n`, `ho[0]` the energy-like constant).
    pub ho: Vec<f64>,
    /// Active components (indices of parameters, offset 1 added by the pipeline).
    pub map: Vec<usize>,
}

/// Build a problem with `n - 1` parameters, a `factors`-dimensional latent structure and
/// independent noise of relative size `noise` (smaller means worse conditioning when
/// `factors < n`). Every `skip`-th parameter is dropped from the active map (`skip == 0`
/// keeps all).
pub fn make_problem(
    n: usize,
    samples: usize,
    factors: usize,
    noise: f64,
    skip: usize,
    seed: u64,
) -> SrProblem {
    let mut rng = Lcg(seed ^ 0x5851_f42d_4c95_7f2d);
    let npara = n - 1;
    let factors = factors.max(1);
    let loading: Vec<f64> = (0..npara * factors).map(|_| rng.normal()).collect();
    let scale = 1.0 / (samples as f64).sqrt();
    let mut store = vec![0.0; n * samples];
    let mut z = vec![0.0; factors];
    for s in 0..samples {
        for f in z.iter_mut() {
            *f = rng.normal();
        }
        let col = &mut store[s * n..(s + 1) * n];
        col[0] = scale;
        for p in 0..npara {
            let mut v = 0.0;
            for f in 0..factors {
                v += loading[p + f * npara] * z[f];
            }
            v += noise * rng.normal();
            // a nonzero mean makes the mean-subtraction in S matter
            col[p + 1] = scale * (v + 0.3 * ((p % 7) as f64 - 3.0));
        }
    }
    let ho: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    let map: Vec<usize> = (0..npara)
        .filter(|p| skip == 0 || p % skip != skip - 1)
        .collect();
    SrProblem {
        n,
        samples,
        store,
        ho,
        map,
    }
}

/// Active-component sample matrix `[components, samples]` for the CG operand, as
/// `stochastic_opt_cg` fills it, plus the mean and the variance (`diagonal`) vectors and the
/// gradient, from the Gram of the store.
pub struct CgInputs {
    /// `[components, samples]`.
    pub operand: Vec<f64>,
    /// Means `O_p` of the active components.
    pub mean: Vec<f64>,
    /// Variances `E[O_p^2] - mean^2`.
    pub diagonal: Vec<f64>,
    /// Gradient `-2 dt (HO_p - HO_0 mean_p)`.
    pub gradient: Vec<f64>,
}

/// CG inputs of `problem` (`offset` 1; weights equal, `inv_weight = 1` because the store is
/// pre-scaled by `1/sqrt(samples)`).
pub fn cg_inputs(problem: &SrProblem, step_dt: f64) -> CgInputs {
    let n = problem.n;
    let comp = problem.map.len();
    let mut operand = vec![0.0; comp * problem.samples];
    let mut mean = vec![0.0; comp];
    let mut second = vec![0.0; comp];
    for s in 0..problem.samples {
        let col = &problem.store[s * n..(s + 1) * n];
        for (si, &p) in problem.map.iter().enumerate() {
            let o = col[p + 1];
            operand[si + s * comp] = o;
            // store is O/sqrt(ns): sum_s O_s * (1/sqrt ns) = mean; sum O^2 = second moment
            mean[si] += o * col[0];
            second[si] += o * o;
        }
    }
    let diagonal: Vec<f64> = second
        .iter()
        .zip(&mean)
        .map(|(sec, m)| sec - m * m)
        .collect();
    let ho0 = problem.ho[0];
    let gradient = problem
        .map
        .iter()
        .zip(&mean)
        .map(|(&p, m)| -2.0 * step_dt * (problem.ho[p + 1] - ho0 * m))
        .collect();
    CgInputs {
        operand,
        mean,
        diagonal,
        gradient,
    }
}

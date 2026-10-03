//! Metropolis accept/reject logic shared by production samplers.
//!
//! Port target: the repeated acceptance block in `vmc_make_sample!`
//! (`vmc_sampling.jl`):
//!
//! ```text
//! x = log_proj_ratio(...)
//! if use_rbm
//!     x += real(log_rbm_ratio(...))
//! end
//! w = exp(2.0 * real(x + log_ip_new - log_ip_old))
//! if !isfinite(w)
//!     w = -1.0
//! end
//! r_metro = rng_real2(rng)
//! accept if w > r_metro
//! ```
//!
//! The production sampler supplies candidate generation, projection/RBM counters,
//! Pfaffian updates and accepted/rejected configuration handling.

use num_complex::Complex64;
use sfmt19937::Sfmt19937Rng;

/// Result of a Metropolis decision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetropolisDecision {
    /// Acceptance weight `w = exp(2*real(delta))`, or `-1.0` if non-finite.
    pub weight: f64,
    /// Draw from `genrand_real2()` used for the comparison.
    pub draw: f64,
    /// True iff `weight > draw`.
    pub accepted: bool,
}

/// Compute the upstream Metropolis acceptance weight.
///
/// `log_proj_delta` is real-valued. `log_rbm_delta` is complex in the
/// upstream but only its real part is included in `x`. `log_ip_new` and
/// `log_ip_old` are the complex projected inner-product logs.
pub fn metropolis_weight(
    log_proj_delta: f64,
    log_rbm_delta: Complex64,
    log_ip_new: Complex64,
    log_ip_old: Complex64,
) -> f64 {
    let exponent = 2.0 * ((log_proj_delta + log_rbm_delta.re + log_ip_new.re) - log_ip_old.re);
    let weight = mvmc_expert_parsers::utils::julia_exp::exp(exponent);
    if weight.is_finite() {
        weight
    } else {
        -1.0
    }
}

/// Compute [`metropolis_weight`], draw `genrand_real2()`, and return
/// the comparison result. The RNG draw order matches Julia's
/// `r_metro = rng_real2(rng)` exactly.
pub fn metropolis_decision(
    log_proj_delta: f64,
    log_rbm_delta: Complex64,
    log_ip_new: Complex64,
    log_ip_old: Complex64,
    rng: &mut Sfmt19937Rng,
) -> MetropolisDecision {
    let weight = metropolis_weight(log_proj_delta, log_rbm_delta, log_ip_new, log_ip_old);
    let draw = rng.genrand_real2();
    MetropolisDecision {
        weight,
        draw,
        accepted: weight > draw,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rbm_acceptance_preserves_julia_left_associative_log_additions() {
        let actual = metropolis_weight(
            1.0e16,
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0e16, 0.0),
            Complex64::new(-1.0, 0.0),
        );
        crate::numerical_comparison::assert_close(
            actual,
            mvmc_expert_parsers::utils::julia_exp::exp(2.0),
            0.0,
            4.0 * f64::EPSILON,
            "acceptance exponential",
        );
    }

    #[test]
    fn nonfinite_weight_is_rejected() {
        let w = metropolis_weight(
            1.0e308,
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        );
        assert_eq!(w, -1.0);
    }

    #[test]
    fn decision_draws_from_sfmt_real2() {
        let mut rng = Sfmt19937Rng::new(42);
        let decision = metropolis_decision(
            0.0,
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            &mut rng,
        );
        assert_eq!(decision.weight, 1.0);
        assert!(decision.draw >= 0.0 && decision.draw < 1.0);
        assert!(decision.accepted);
    }
}

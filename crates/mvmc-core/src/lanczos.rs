//! Lanczos post-processing shared by the PhysCal path.

use num_complex::Complex64;

/// The selected two-step Lanczos estimate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LanczosEnergy {
    /// Variational energy after the Lanczos correction.
    pub energy: f64,
    /// Normalized energy variance of the selected state.
    pub variance: f64,
    /// Mixing coefficient selected from the two stationary points.
    pub alpha: f64,
}

/// Why a Lanczos estimate could not be computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LanczosEnergyError {
    /// The flattened QQQQ accumulator does not contain all 16 moments.
    QqqqTooShort,
    /// The stationary-point equation contains a non-finite coefficient.
    NonFiniteEquation,
    /// The stationary-point equation has a negative discriminant.
    NegativeDiscriminant,
    /// The stationary-point equation is singular.
    SingularEquation,
    /// A stationary point or its energy is not finite or has an illegal norm.
    InvalidStationaryPoint,
}

/// Accumulate one sample's 16 flattened QQQQ moments.
pub fn accumulate_lanczos_qqqq(
    dst: &mut [Complex64],
    weight: f64,
    h1: Complex64,
    h2: Complex64,
    all_complex: bool,
) -> Result<(), LanczosEnergyError> {
    if dst.len() != 16 {
        return Err(LanczosEnergyError::QqqqTooShort);
    }
    let lslq = [Complex64::new(1.0, 0.0), h1, h1, h2];
    for (i0, slot) in dst.iter_mut().enumerate() {
        let rj = i0 % 2;
        let ri = (i0 / 2) % 2;
        let rp = (i0 / 4) % 2;
        let rq = (i0 / 8) % 2;
        let left = lslq[rq * 2 + ri];
        let right = lslq[rp * 2 + rj];
        let left = if all_complex { left.conj() } else { left };
        *slot += weight * left * right;
    }
    Ok(())
}

/// Calculate the lower-energy two-step Lanczos estimate from flattened QQQQ moments.
///
/// The indexing and operation order follow Julia-mVMC's `_lanczos_energy`; the
/// moments are zero-based here, so Julia entries 3, 4, 11, 12, and 16 are
/// Rust entries 2, 3, 10, 11, and 15.
pub fn lanczos_energy(qqqq: &[Complex64]) -> Result<LanczosEnergy, LanczosEnergyError> {
    if qqqq.len() < 16 {
        return Err(LanczosEnergyError::QqqqTooShort);
    }

    let h1 = qqqq[2].re;
    let h2_1 = qqqq[3].re;
    let h2_2 = qqqq[10].re;
    let h3 = qqqq[11].re;
    let h4 = qqqq[15].re;

    let tmp_aa = h2_1 * (h2_1 + h2_2) - 2.0 * h1 * h3;
    let tmp_bb = -h1 * h2_1 + h3;
    let tmp_cc = h2_1 * (h2_1 + h2_2).powi(2) - h1.powi(2) * h2_1 * (h2_1 + 2.0 * h2_2)
        + 4.0 * h1.powi(3) * h3
        - 2.0 * h1 * (2.0 * h2_1 + h2_2) * h3
        + h3.powi(2);
    if !(tmp_aa.is_finite() && tmp_bb.is_finite() && tmp_cc.is_finite()) {
        return Err(LanczosEnergyError::NonFiniteEquation);
    }
    if tmp_cc < 0.0 {
        return Err(LanczosEnergyError::NegativeDiscriminant);
    }
    if tmp_aa.abs() < f64::EPSILON {
        return Err(LanczosEnergyError::SingularEquation);
    }

    let root = tmp_cc.sqrt();
    let alpha_p = (tmp_bb + root) / tmp_aa;
    let alpha_m = (tmp_bb - root) / tmp_aa;
    if !(alpha_p.is_finite() && alpha_m.is_finite()) {
        return Err(LanczosEnergyError::InvalidStationaryPoint);
    }

    let energy_p = energy_by_alpha(h1, h2_1, h2_2, h3, h4, alpha_p);
    let energy_m = energy_by_alpha(h1, h2_1, h2_2, h3, h4, alpha_m);
    let (energy_p, energy_m) = match (energy_p, energy_m) {
        (Some(p), Some(m)) => (p, m),
        _ => return Err(LanczosEnergyError::InvalidStationaryPoint),
    };

    if energy_p.0 > energy_m.0 {
        Ok(LanczosEnergy {
            energy: energy_m.0,
            variance: energy_m.1,
            alpha: alpha_m,
        })
    } else {
        Ok(LanczosEnergy {
            energy: energy_p.0,
            variance: energy_p.1,
            alpha: alpha_p,
        })
    }
}

fn energy_by_alpha(
    h1: f64,
    h2_1: f64,
    h2_2: f64,
    h3: f64,
    h4: f64,
    alpha: f64,
) -> Option<(f64, f64)> {
    let tmp_ene = h1 + alpha * (h2_1 + h2_2) + alpha.powi(2) * h3;
    let dnorm = 1.0 + 2.0 * alpha * h1 + alpha.powi(2) * h2_1;
    let tmp_ene_v = h2_1 + 2.0 * alpha * h3 + alpha.powi(2) * h4;
    if !h1.is_finite() || h1.abs() < f64::EPSILON {
        return None;
    }
    let norm_ratio = dnorm / h1;
    if !norm_ratio.is_finite() || norm_ratio.abs() < 1.0e-12 {
        return None;
    }
    let ene = tmp_ene / dnorm;
    let ene_v = ((tmp_ene_v / dnorm) - ene.powi(2)) / ene.powi(2);
    if !ene.is_finite() || !ene_v.is_finite() {
        return None;
    }
    Some((ene, ene_v))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn moments(h1: f64, h2_1: f64, h2_2: f64, h3: f64, h4: f64) -> Vec<Complex64> {
        let mut values = vec![Complex64::new(0.0, 0.0); 16];
        values[2] = Complex64::new(h1, 7.0);
        values[3] = Complex64::new(h2_1, -3.0);
        values[10] = Complex64::new(h2_2, 2.0);
        values[11] = Complex64::new(h3, 1.0);
        values[15] = Complex64::new(h4, -1.0);
        values
    }

    #[test]
    fn selects_lower_stationary_energy() {
        let result = lanczos_energy(&moments(1.0, 2.0, 3.0, 4.0, 5.0)).unwrap();
        assert!(result.energy.is_finite());
        assert!(result.variance.is_finite());
        assert!(result.alpha.is_finite());
        let p = energy_by_alpha(1.0, 2.0, 3.0, 4.0, 5.0, result.alpha).unwrap();
        assert_eq!(p.0, result.energy);
        assert_eq!(p.1, result.variance);
    }

    #[test]
    fn rejects_short_or_invalid_moments() {
        assert_eq!(
            lanczos_energy(&[Complex64::default(); 15]),
            Err(LanczosEnergyError::QqqqTooShort)
        );
        assert_eq!(
            lanczos_energy(&moments(0.0, 0.0, 0.0, 0.0, 0.0)),
            Err(LanczosEnergyError::SingularEquation)
        );
        let mut values = moments(1.0, 2.0, 3.0, 4.0, 5.0);
        values[2].re = f64::NAN;
        assert_eq!(
            lanczos_energy(&values),
            Err(LanczosEnergyError::NonFiniteEquation)
        );

        let values = moments(
            -2.152457446309748,
            -2.9568890514902515,
            0.6960797422130414,
            3.1910420776032336,
            2.2685266679515284,
        );
        assert_eq!(
            lanczos_energy(&values),
            Err(LanczosEnergyError::NegativeDiscriminant)
        );

        assert_eq!(
            lanczos_energy(&moments(0.0, 1.0, 0.0, 0.0, 1.0)),
            Err(LanczosEnergyError::InvalidStationaryPoint)
        );
    }

    #[test]
    fn accumulates_qqqq_in_julia_flattened_order() {
        let mut actual = vec![Complex64::new(0.0, 0.0); 16];
        accumulate_lanczos_qqqq(
            &mut actual,
            2.0,
            Complex64::new(2.0, 3.0),
            Complex64::new(5.0, 7.0),
            true,
        )
        .unwrap();
        assert_eq!(actual[0], Complex64::new(2.0, 0.0));
        assert_eq!(actual[1], Complex64::new(4.0, 6.0));
        assert_eq!(actual[2], Complex64::new(4.0, -6.0));
        assert_eq!(actual[15], Complex64::new(148.0, 0.0));
    }

    #[test]
    #[allow(clippy::excessive_precision)] // Retain the C fixture's decimal literals.
    fn c_calculate_ene_reference_is_preserved() {
        // Values from the C mVMC physcal_lanczos.c reference fixture
        // (hubbard_chain_lanczos/physcal_ref).  Keep all five moments as
        // independent literals so a change in operation order cannot be
        // hidden by deriving one moment from another.
        let result = lanczos_energy(&moments(
            -3.218953137198691916,
            10.80845047645121149,
            11.11256143424133391,
            -34.35245926247226391,
            138.1493179027195026,
        ))
        .unwrap();

        assert!((result.energy - (-3.301741343987198540)).abs() <= 1.0e-12);
        assert!((result.variance - 2.240411016743763806e-2).abs() <= 1.0e-12);
        assert!((result.alpha - (-2.490954930208576223e-1)).abs() <= 1.0e-12);
    }
}

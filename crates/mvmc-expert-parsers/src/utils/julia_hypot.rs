//! Julia 1.13.1 Base/math.jl Float64 hypot for parameter normalization.
//! MIT Julia contributors; full notice in ../../LICENSE-julia-math.
#[cfg(test)]
use crate::numerical_comparison;

/// Compute the complex amplitude with Julia's rounding and scaling order.
pub fn hypot(x: f64, y: f64) -> f64 {
    let (mut ax, mut ay) = (x.abs(), y.abs());
    if ax.is_infinite() || ay.is_infinite() {
        return f64::INFINITY;
    }
    if ay > ax {
        std::mem::swap(&mut ax, &mut ay);
    }
    if ay <= ax * (f64::EPSILON / 2.0).sqrt() {
        return ax;
    }
    let tiny_scale = f64::EPSILON * f64::MIN_POSITIVE.sqrt();
    let scale = if ax > (f64::MAX / 2.0).sqrt() {
        ax *= tiny_scale;
        ay *= tiny_scale;
        1.0 / tiny_scale
    } else if ay < f64::MIN_POSITIVE.sqrt() {
        ax /= tiny_scale;
        ay /= tiny_scale;
        tiny_scale
    } else {
        1.0
    };
    let mut h = ax.mul_add(ax, ay * ay).sqrt();
    let hsquared = h * h;
    let axsquared = ax * ax;
    h -= ((-ay).mul_add(ay, hsquared - axsquared) + h.mul_add(h, -hsquared)
        - ax.mul_add(ax, -axsquared))
        / (2.0 * h);
    h * scale
}

#[cfg(test)]
mod tests {
    #[test]
    fn normalization_magnitude_matches_julia_with_roundoff_bound() {
        for line in include_str!("../../../../tests/fixtures/hypot.txt").lines() {
            let values: Vec<f64> = line
                .split_whitespace()
                .map(|v| f64::from_bits(u64::from_str_radix(v, 16).unwrap()))
                .collect();
            let actual = super::hypot(values[0], values[1]);
            // Scaled squares, square root and correction: eight rounding units.
            super::numerical_comparison::assert_close(
                actual,
                values[2],
                2.0 * f64::from_bits(1),
                8.0 * f64::EPSILON,
                format!("hypot({}, {})", values[0], values[1]),
            );
        }
    }
}

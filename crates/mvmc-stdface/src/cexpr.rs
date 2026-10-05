//! The compound `double complex` expressions of `StdFace_HubbardLocal`, `StdFace_MagField` and
//! `StdFace_GeneralJ`, evaluated exactly as compiled C does.
//!
//! GCC keeps IEEE signed zeros, and a mixed `double`/`double complex` operation is not a full
//! complex operation: products scale both components, `real + complex` copies the imaginary
//! part, `real - complex` negates it (see [`C64::scale`] and friends). Each
//! function mirrors one source expression, operand order and associativity preserved;
//! `tests/complex_expressions.rs` checks them bitwise against the C compiler on a grid that
//! includes `-0.0`.

use crate::ccomplex::C64;

/// `I * x` for a real `x` (`I` is the constant `(0, 1)`).
fn i_times(x: f64) -> C64 {
    C64::I.scale(x)
}

/// `-0.5 * I * Gamma0_y` (`HubbardLocal`).
pub fn hubbard_local_im_neg(gamma_y: f64) -> C64 {
    C64::I.scale(-0.5).scale(gamma_y)
}

/// `0.5 * I * Gamma0_y` (`HubbardLocal`).
pub fn hubbard_local_im_pos(gamma_y: f64) -> C64 {
    C64::I.scale(0.5).scale(gamma_y)
}

fn mag_root(s: f64, sz: f64) -> f64 {
    (s * (s + 1.0) - sz * (sz + 1.0)).sqrt()
}

/// `-0.5*Gamma*sqrt(..) - 0.5*I*Gamma_y*sqrt(..)` (`MagField`).
pub fn mag_field_minus(gamma: f64, gamma_y: f64, s: f64, sz: f64) -> C64 {
    let real = -0.5 * gamma * mag_root(s, sz);
    let imag = C64::I.scale(0.5).scale(gamma_y).scale(mag_root(s, sz));
    C64::real_minus(real, imag)
}

/// `-0.5*Gamma*sqrt(..) + 0.5*I*Gamma_y*sqrt(..)` (`MagField`).
pub fn mag_field_plus(gamma: f64, gamma_y: f64, s: f64, sz: f64) -> C64 {
    let real = -0.5 * gamma * mag_root(s, sz);
    let imag = C64::I.scale(0.5).scale(gamma_y).scale(mag_root(s, sz));
    C64::real_plus(real, imag)
}

/// `intr0 = J[2][2] * Siz * Sjz` (`GeneralJ`, term 1).
pub fn general_j_zz(j22: f64, siz: f64, sjz: f64) -> C64 {
    C64::real(j22 * siz * sjz)
}

/// `0.25*(Jx + Jy + I*(Jxy - Jyx)) * sqrt(..i) * sqrt(..j)` (`GeneralJ`, term 2).
pub fn general_j_pm(j: &[[f64; 3]; 3], si: f64, siz: f64, sj: f64, sjz: f64) -> C64 {
    C64::real_plus(j[0][0] + j[1][1], i_times(j[0][1] - j[1][0]))
        .scale(0.25)
        .scale(mag_root(si, siz))
        .scale(mag_root(sj, sjz))
}

/// `0.5*0.5*(Jx - Jy - I*(Jxy + Jyx)) * sqrt(..i) * sqrt(..j)` (`GeneralJ`, term 3).
pub fn general_j_pp(j: &[[f64; 3]; 3], si: f64, siz: f64, sj: f64, sjz: f64) -> C64 {
    C64::real_minus(j[0][0] - j[1][1], i_times(j[0][1] + j[1][0]))
        .scale(0.5 * 0.5)
        .scale(mag_root(si, siz))
        .scale(mag_root(sj, sjz))
}

/// `0.5*(Jxz - I*Jyz) * sqrt(..i) * Sjz` (`GeneralJ`, term 4).
pub fn general_j_pz(j: &[[f64; 3]; 3], si: f64, siz: f64, sjz: f64) -> C64 {
    C64::real_minus(j[0][2], i_times(j[1][2]))
        .scale(0.5)
        .scale(mag_root(si, siz))
        .scale(sjz)
}

/// `0.5*(Jzx - I*Jzy) * Siz * sqrt(..j)` (`GeneralJ`, term 5).
pub fn general_j_zp(j: &[[f64; 3]; 3], sj: f64, siz: f64, sjz: f64) -> C64 {
    C64::real_minus(j[2][0], i_times(j[2][1]))
        .scale(0.5)
        .scale(siz)
        .scale(mag_root(sj, sjz))
}

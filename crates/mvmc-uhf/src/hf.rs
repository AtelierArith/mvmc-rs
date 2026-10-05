//! Hartree-Fock iteration: ports of `makeham.c`, `diag.c`, `green.c`,
//! `cal_energy.c` and `initial.c`.
//!
//! Matrices are `2*Nsite` square, stored row-major (`m[row * n2 + col]`) like
//! the C `double complex **` arrays. Spin-orbital index is `site + spin*Nsite`.
//! Complex products follow C99: complex*complex uses
//! `(ac - bd, ad + bc)`, real*complex scales each component.

// The energy expressions deliberately keep the C spelling (`-1.0 * tmp * ...`).
#![allow(clippy::neg_multiply, clippy::identity_op)]

use num_complex::Complex64;
use sfmt19937::Sfmt19937Rng;

use crate::definition::UhfDefinition;
use crate::linalg;
use crate::UhfError;

/// Real times complex, component-wise (C mixed real/complex multiplication).
fn rc(x: f64, z: Complex64) -> Complex64 {
    Complex64::new(x * z.re, x * z.im)
}

/// Complex times complex, the `__muldc3` main path.
fn cm(a: Complex64, b: Complex64) -> Complex64 {
    Complex64::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
}

/// Working state of one UHF run (`LargeList` + `PhysList`).
pub struct HfState {
    /// `2*Nsite`.
    pub n2: usize,
    /// Current Green function `G[a][b] = <c^+_a c_b>`.
    pub g: Vec<Complex64>,
    /// Previous Green function.
    pub g_old: Vec<Complex64>,
    /// Mean-field Hamiltonian.
    pub ham: Vec<Complex64>,
    /// `R_SLT[2N][Nsize]` (conjugated occupied eigenvectors).
    pub r_slt: Vec<Complex64>,
    /// `L_SLT[Nsize][2N]` (occupied eigenvectors).
    pub l_slt: Vec<Complex64>,
    /// Eigenvalues of the last diagonalisation.
    pub eigen_values: Vec<f64>,
    /// Total energy of the last `cal_energy`.
    pub energy: f64,
    /// Mean absolute Green-function residual.
    pub rest: f64,
    /// Particle number.
    pub num: f64,
}

impl HfState {
    /// Allocate zeroed arrays (`xsetmem_large.c`).
    pub fn new(def: &UhfDefinition) -> Self {
        let n2 = 2 * def.nsite;
        let zero = Complex64::new(0.0, 0.0);
        Self {
            n2,
            g: vec![zero; n2 * n2],
            g_old: vec![zero; n2 * n2],
            ham: vec![zero; n2 * n2],
            r_slt: vec![zero; n2 * def.nsize],
            l_slt: vec![zero; def.nsize * n2],
            eigen_values: vec![0.0; n2],
            energy: 0.0,
            rest: 0.0,
            num: 0.0,
        }
    }

    fn at(&self, row: usize, col: usize) -> Complex64 {
        self.g[row * self.n2 + col]
    }
}

/// `initial()`: random start (`NInitial == 0`) or the `Initial` file entries.
pub fn initial(def: &UhfDefinition, state: &mut HfState, rng: &mut Sfmt19937Rng) {
    let ns = def.nsite;
    let n2 = state.n2;
    if def.n_initial == 0 {
        for row in 0..n2 {
            for col in 0..n2 {
                // uniform distribution [0,1) shifted to [-0.005, 0.005)
                state.g[row * n2 + col] = Complex64::new(0.01 * (rng.genrand_real2() - 0.5), 0.0);
            }
        }
    }
    if def.print == 1 {
        println!("#[s]output initial Green functions");
    }
    for (k, term) in def.initial.iter().enumerate() {
        let [site0, spin0, site1, spin1] = term.index;
        if def.print == 1 {
            println!("{} {:.6} {:.6} ", k, term.value.re, term.value.im);
        }
        let t0 = site0 as usize + spin0 as usize * ns;
        let t1 = site1 as usize + spin1 as usize * ns;
        state.g[t0 * n2 + t1] = term.value;
    }
    if def.print == 1 {
        println!("#[e]output initial Green functions");
    }
}

/// `makeham()`: mean-field Hamiltonian from the current Green function.
pub fn makeham(def: &UhfDefinition, state: &mut HfState) {
    let ns = def.nsite;
    let n2 = state.n2;
    let HfState { ham, g, .. } = state;
    let at = |a: usize, b: usize| g[a * n2 + b];
    ham.iter_mut().for_each(|h| *h = Complex64::new(0.0, 0.0));
    let site = |s: i32, spin: i32| s as usize + spin as usize * ns;

    // Transfer input
    for term in &def.transfer {
        let [s1, p1, s2, p2] = term.index;
        let tmp = -term.value;
        ham[site(s1, p1) * n2 + site(s2, p2)] += tmp;
    }
    // Intra U input
    for &(s, u) in &def.coulomb_intra {
        let tmp = Complex64::new(u, 0.0);
        let (us, ds) = (site(s, 0), site(s, 1));
        let g_dd = at(ds, ds);
        let g_uu = at(us, us);
        ham[us * n2 + us] += cm(tmp, g_dd);
        ham[ds * n2 + ds] += cm(tmp, g_uu);
        // Off-Diagonal Fock term
        let g_du = at(ds, us);
        let g_ud = at(us, ds);
        ham[us * n2 + ds] += cm(rc(-1.0, tmp), g_du);
        ham[ds * n2 + us] += cm(rc(-1.0, tmp), g_ud);
    }
    // Inter U input
    for &(s1, s2, v) in &def.coulomb_inter {
        let tmp = Complex64::new(v, 0.0);
        let (u1, d1) = (site(s1, 0), site(s1, 1));
        let (u2, d2) = (site(s2, 0), site(s2, 1));
        let mut charge = at(u2, u2).re + at(d2, d2).re;
        ham[u1 * n2 + u1] += rc(charge, tmp);
        ham[d1 * n2 + d1] += rc(charge, tmp);
        charge = at(u1, u1).re + at(d1, d1).re;
        ham[u2 * n2 + u2] += rc(charge, tmp);
        ham[d2 * n2 + d2] += rc(charge, tmp);
        // Diagonal Fock term
        let m_tmp = -tmp;
        let (g21, g12) = (at(u2, u1), at(u1, u2));
        ham[u1 * n2 + u2] += cm(m_tmp, g21);
        ham[u2 * n2 + u1] += cm(m_tmp, g12);
        let (g21, g12) = (at(d2, d1), at(d1, d2));
        ham[d1 * n2 + d2] += cm(m_tmp, g21);
        ham[d2 * n2 + d1] += cm(m_tmp, g12);
        // Off-Diagonal Fock term
        ham[u1 * n2 + d2] += cm(m_tmp, at(d2, u1));
        ham[d2 * n2 + u1] += cm(m_tmp, at(u1, d2));
        ham[u2 * n2 + d1] += cm(m_tmp, at(d1, u2));
        ham[d1 * n2 + u2] += cm(m_tmp, at(u2, d1));
    }
    // Hund input
    for &(s1, s2, j) in &def.hund {
        let tmp = Complex64::new(-j, 0.0);
        let (u1, d1) = (site(s1, 0), site(s1, 1));
        let (u2, d2) = (site(s2, 0), site(s2, 1));
        ham[u1 * n2 + u1] += cm(tmp, at(u2, u2));
        ham[d1 * n2 + d1] += cm(tmp, at(d2, d2));
        ham[u2 * n2 + u2] += cm(tmp, at(u1, u1));
        ham[d2 * n2 + d2] += cm(tmp, at(d1, d1));
        // Diagonal Fock term
        let m_tmp = -tmp;
        ham[u1 * n2 + u2] += cm(m_tmp, at(u2, u1));
        ham[u2 * n2 + u1] += cm(m_tmp, at(u1, u2));
        ham[d1 * n2 + d2] += cm(m_tmp, at(d2, d1));
        ham[d2 * n2 + d1] += cm(m_tmp, at(d1, d2));
    }
    // Exchange input
    for &(s1, s2, j) in &def.exchange {
        let tmp = Complex64::new(j, 0.0);
        let (u1, d1) = (site(s1, 0), site(s1, 1));
        let (u2, d2) = (site(s2, 0), site(s2, 1));
        // Diagonal Fock term
        ham[u1 * n2 + u2] += cm(tmp, at(d2, d1));
        ham[d2 * n2 + d1] += cm(tmp, at(u1, u2));
        ham[d1 * n2 + d2] += cm(tmp, at(u2, u1));
        ham[u2 * n2 + u1] += cm(tmp, at(d1, d2));
        // Off-Diagonal Fock term
        let m_tmp = -tmp;
        ham[u1 * n2 + d1] += cm(m_tmp, at(d2, u2));
        ham[u2 * n2 + d2] += cm(m_tmp, at(d1, u1));
        ham[d1 * n2 + u1] += cm(m_tmp, at(u2, d2));
        ham[d2 * n2 + u2] += cm(m_tmp, at(u1, d1));
    }
    // PairHopping input
    for &(s1, s2, t) in &def.pair_hopping {
        let tmp = Complex64::new(t, 0.0);
        let (u1, d1) = (site(s1, 0), site(s1, 1));
        let (u2, d2) = (site(s2, 0), site(s2, 1));
        // Diagonal Fock term
        ham[u1 * n2 + u2] += cm(tmp, at(d1, d2));
        ham[d1 * n2 + d2] += cm(tmp, at(u1, u2));
        // Off-Diagonal Fock term
        let m_tmp = -tmp;
        ham[u1 * n2 + d2] += cm(m_tmp, at(d1, u2));
        ham[d1 * n2 + u2] += cm(m_tmp, at(u1, d2));
    }
    // InterAll input
    for (index, value) in &def.inter_all {
        let s1 = site(index[0], index[1]);
        let s2 = site(index[2], index[3]);
        let s3 = site(index[4], index[5]);
        let s4 = site(index[6], index[7]);
        let tmp = *value;
        let flag = !(s1 == s2 && s3 == s4 && s1 == s3);
        if flag {
            // Diagonal Fock term
            ham[s1 * n2 + s2] += cm(tmp, at(s3, s4));
            ham[s3 * n2 + s4] += cm(tmp, at(s1, s2));
            // Off-Diagonal Fock term
            let m_tmp = -tmp;
            ham[s1 * n2 + s4] += cm(m_tmp, at(s3, s2));
            ham[s3 * n2 + s2] += cm(m_tmp, at(s1, s4));
            if s2 == s3 {
                ham[s1 * n2 + s4] += tmp;
            }
        } else {
            ham[s1 * n2 + s1] += tmp;
        }
    }
}

/// `diag()`: diagonalise `Ham` (upper triangle) with `zheev` and store the
/// occupied eigenvectors as `R_SLT = U^*` and `L_SLT = U^T`.
pub fn diag(def: &UhfDefinition, state: &mut HfState) -> Result<(), UhfError> {
    let n2 = state.n2;
    let (values, vectors) = linalg::zheev_all(n2, &state.ham)?;
    state.eigen_values.copy_from_slice(&values);
    for k in 0..def.nsize {
        for l in 0..n2 {
            let v = vectors[k * n2 + l];
            state.r_slt[l * def.nsize + k] = v.conj();
            state.l_slt[k * n2 + l] = v;
        }
    }
    Ok(())
}

/// `green()`: `G = R_SLT * L_SLT` through `zgemm`.
pub fn green(def: &UhfDefinition, state: &mut HfState) {
    state.g_old.copy_from_slice(&state.g);
    let product = linalg::zgemm_nn(state.n2, def.nsize, &state.r_slt, &state.l_slt);
    state.g.copy_from_slice(&product);
}

/// `cal_energy()`: energy, particle number, residual and linear mixing.
pub fn cal_energy(def: &UhfDefinition, state: &mut HfState) {
    let ns = def.nsite;
    let n2 = state.n2;
    let site = |s: i32, spin: i32| s as usize + spin as usize * ns;
    let g = |state: &HfState, a: usize, b: usize| state.at(a, b);

    let mut e_band = 0.0_f64;
    for k in 0..def.nsize {
        e_band += state.eigen_values[k];
    }
    // Intra U energy
    let mut e_coulomb_intra = 0.0_f64;
    for &(s, tmp) in &def.coulomb_intra {
        let (us, ds) = (site(s, 0), site(s, 1));
        e_coulomb_intra += cm(rc(-1.0 * tmp, g(state, us, us)), g(state, ds, ds)).re;
        // Off-Diagonal Fock term
        e_coulomb_intra += cm(rc(1.0 * tmp, g(state, us, ds)), g(state, ds, us)).re;
    }
    // Inter U energy
    let mut e_coulomb_inter = 0.0_f64;
    for &(s1, s2, tmp) in &def.coulomb_inter {
        let (u1, d1) = (site(s1, 0), site(s1, 1));
        let (u2, d2) = (site(s2, 0), site(s2, 1));
        let charge_1 = g(state, u1, u1).re + g(state, d1, d1).re;
        let charge_2 = g(state, u2, u2).re + g(state, d2, d2).re;
        e_coulomb_inter += -1.0 * tmp * charge_1 * charge_2;
        // Diagonal Fock term
        for (a, b, c, d) in [
            (u1, u2, u2, u1),
            (d1, d2, d2, d1),
            // Off-Diagonal Fock term
            (u1, d2, d2, u1),
            (d1, u2, u2, d1),
        ] {
            e_coulomb_inter += cm(rc(1.0 * tmp, g(state, a, b)), g(state, c, d)).re;
        }
    }
    // Hund energy
    let mut e_hund = 0.0_f64;
    for &(s1, s2, j) in &def.hund {
        let tmp = -j;
        let (u1, d1) = (site(s1, 0), site(s1, 1));
        let (u2, d2) = (site(s2, 0), site(s2, 1));
        e_hund += cm(rc(-1.0 * tmp, g(state, u1, u1)), g(state, u2, u2)).re;
        e_hund += cm(rc(-1.0 * tmp, g(state, d1, d1)), g(state, d2, d2)).re;
        // Diagonal Fock term
        e_hund += cm(rc(1.0 * tmp, g(state, u1, u2)), g(state, u2, u1)).re;
        e_hund += cm(rc(1.0 * tmp, g(state, d1, d2)), g(state, d2, d1)).re;
    }
    // Exchange energy
    let mut e_exchange = 0.0_f64;
    for &(s1, s2, tmp) in &def.exchange {
        let (u1, d1) = (site(s1, 0), site(s1, 1));
        let (u2, d2) = (site(s2, 0), site(s2, 1));
        // Diagonal Fock term
        e_exchange += cm(rc(-1.0 * tmp, g(state, u1, u2)), g(state, d2, d1)).re;
        e_exchange += cm(rc(-1.0 * tmp, g(state, d1, d2)), g(state, u2, u1)).re;
        // Off-Diagonal Fock term
        e_exchange += cm(rc(1.0 * tmp, g(state, u1, d1)), g(state, d2, u2)).re;
        e_exchange += cm(rc(1.0 * tmp, g(state, d1, u1)), g(state, u2, d2)).re;
    }
    // PairHopping energy
    let mut e_pair_hopping = 0.0_f64;
    for &(s1, s2, tmp) in &def.pair_hopping {
        let (u1, d1) = (site(s1, 0), site(s1, 1));
        let (u2, d2) = (site(s2, 0), site(s2, 1));
        // Diagonal Fock term
        e_pair_hopping += cm(rc(-1.0 * tmp, g(state, u1, u2)), g(state, d1, d2)).re;
        e_pair_hopping += cm(rc(1.0 * tmp, g(state, u1, d2)), g(state, d1, u2)).re;
    }
    // InterAll energy
    let mut e_inter_all = 0.0_f64;
    for (index, ctmp) in &def.inter_all {
        let s1 = site(index[0], index[1]);
        let s2 = site(index[2], index[3]);
        let s3 = site(index[4], index[5]);
        let s4 = site(index[6], index[7]);
        e_inter_all -= cm(cm(*ctmp, g(state, s1, s2)), g(state, s3, s4)).re;
        e_inter_all += cm(cm(*ctmp, g(state, s1, s4)), g(state, s3, s2)).re;
    }
    // Calculating Total Energy
    state.energy = e_band + e_coulomb_intra + e_coulomb_inter + e_hund;
    state.energy += e_exchange + e_pair_hopping + e_inter_all;

    let mix = def.mix;
    state.rest = 0.0;
    let mut num = 0.0_f64;
    for i in 0..n2 {
        num += state.g[i * n2 + i].re;
        for j in 0..n2 {
            let tmp = (state.g_old[i * n2 + j] - state.g[i * n2 + j]).norm();
            state.rest += tmp * tmp;
            state.g[i * n2 + j] =
                rc(1.0 - mix, state.g_old[i * n2 + j]) + rc(mix, state.g[i * n2 + j]);
        }
    }
    state.num = num;
    state.rest = state.rest.sqrt() / (2.0 * def.nsite as f64 * def.nsite as f64);
}

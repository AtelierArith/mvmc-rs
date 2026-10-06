//! C's `|g| <= 1e-12` branch of the Lanczos operators (`calHCA2`, `calHCACA2`; issue #481).
//!
//! `calHCA`/`calHCACA` (`extern/mVMC-1.3.0/src/mVMC/lslocgrn{,_real}.c`) evaluate
//! `<psi|H C_is A_js|x>/<psi|x>` as `<psi|x'>/<psi|x> * <psi|H|x'>/<psi|x'>` (`calHCA1`,
//! `calHCACA1`) only while the hopped Pfaffian ratio `g = checkGF1/checkGF2` is above `1e-12` in
//! magnitude. For a nodal hop (`|g| <= 1e-12`, e.g. the hopped configuration has weight zero)
//! the ratio form is useless and C expands the Hamiltonian instead,
//!
//! ```text
//! <psi|H CA|x>/<psi|x> = g' H0(x') + sum_terms  coef * <psi| (term) CA |x>/<psi|x>
//! ```
//!
//! with the one-, two- and n-body Green functions of the unhopped configuration `x`
//! (`GreenFunc1/2/N`). This module ports `checkGF1/2`, `GreenFuncN` (with its operator
//! reductions and the `calculateNewPfMN` Pfaffian) and `calHCA2/calHCACA2`, in C's operation
//! order for one thread (the OpenMP `reduction(+:v)` of the term loops is the serial running sum
//! `myValue`, then `val += v`).
//!
//! Real wavefunctions use the real kernels (no RBM, real quotient); complex ones use C's complex
//! conventions (`GreenFunc*` return `conj(z/ip)`). C's complex `GreenFuncN` hands `&n` instead of
//! the matrix order `2n` and an uninitialised `lda` to `M_ZSKPFA` (its source comment says
//! "ignore GreenFuncN: to be added"), which is undefined behaviour; the complex port here uses
//! the mathematically intended `2n` Pfaffian.

use num_complex::Complex64;

use super::*;

/// C's threshold between the ratio form and the Hamiltonian expansion.
pub(super) const NODAL_THRESHOLD: f64 = 1.0e-12;

/// The unhopped configuration and tables the Green functions of one sample read.
pub(super) struct NodalContext<'a> {
    pub data: &'a ExpertModeData,
    pub state: &'a VmcOptimizationState,
    pub ele_idx: &'a [i64],
    pub ele_cfg: &'a [i64],
    pub ele_num: &'a [i64],
    pub ele_proj_cnt: &'a [i64],
    pub ip: Complex64,
}

impl NodalContext<'_> {
    fn n_site(&self) -> usize {
        self.data.modpara.nsite.max(0) as usize
    }

    fn n_elec(&self) -> usize {
        self.data.modpara.nelec.max(0) as usize
    }

    fn is_real(&self) -> bool {
        !self.state.slater_matrix.pf_m_real.is_empty()
    }

    pub(super) fn green1(&self, ri: usize, rj: usize, spin: u8) -> Complex64 {
        green_func1_impl::<false, false, true>(
            ri,
            rj,
            spin,
            spin,
            self.ip,
            self.data,
            self.state,
            self.ele_idx,
            self.ele_cfg,
            self.ele_num,
            self.ele_proj_cnt,
            &mut GreenScratch::default(),
            &mut CTimer::<false>::new(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn green2(&self, ri: usize, rj: usize, rk: usize, rl: usize, s: u8, t: u8) -> Complex64 {
        green_func2_impl::<true>(
            ri,
            rj,
            rk,
            rl,
            s,
            t,
            self.ip,
            self.data,
            self.state,
            self.ele_idx,
            self.ele_cfg,
            self.ele_num,
            self.ele_proj_cnt,
        )
    }

    /// `|z/ip|` of the Pfaffian of the configuration with the listed `(electron slot, site)`
    /// hops (no projection factor), via the rank-one / rank-two update formulas.
    fn pfaffian_ratio(&self, hops: &[(usize, usize)]) -> f64 {
        let n_site = self.n_site();
        let n_elec = self.n_elec();
        let n_qp = self.state.slater_matrix.pf_m.len();
        let n_size = 2 * n_elec;
        let stride = n_size * n_size + 1;
        let mut new_idx = self.ele_idx.to_vec();
        for &(slot, site) in hops {
            new_idx[slot] = site as i64;
        }
        let slater = &self.state.slater_matrix;
        let split = |slot: usize| (slot % n_elec, (slot / n_elec) as u8);
        if self.is_real() {
            let mut pf_new = vec![0.0_f64; n_qp];
            match hops {
                [(slot, _)] => {
                    let (ma, spin) = split(*slot);
                    calculate_new_pf_m2_real_flat(
                        ma,
                        spin,
                        &mut pf_new,
                        &new_idx,
                        &slater.slater_elm_real,
                        slater.inv_m_real.as_slice(),
                        stride,
                        &slater.pf_m_real,
                        0,
                        n_qp,
                        n_site,
                        n_elec,
                    );
                }
                [(first, _), (second, _)] => {
                    let (ma, sa) = split(*first);
                    let (mb, sb) = split(*second);
                    calculate_new_pf_m_two2_real_flat::<true>(
                        ma,
                        sa,
                        mb,
                        sb,
                        &mut pf_new,
                        &new_idx,
                        &slater.slater_elm_real,
                        slater.inv_m_real.as_slice(),
                        stride,
                        &slater.pf_m_real,
                        0,
                        n_qp,
                        n_site,
                        n_elec,
                    );
                }
                _ => unreachable!("checkGF takes one or two hops"),
            }
            let z = calculate_ip_real(&pf_new, 0, n_qp, self.data);
            (z / self.ip.re).abs()
        } else {
            let mut pf_new = vec![Complex64::new(0.0, 0.0); n_qp];
            match hops {
                [(slot, _)] => {
                    let (ma, spin) = split(*slot);
                    calculate_new_pf_m2_complex_flat(
                        ma,
                        spin,
                        &mut pf_new,
                        &new_idx,
                        &slater.slater_elm,
                        slater.inv_m.as_slice(),
                        stride,
                        &slater.pf_m,
                        0,
                        n_qp,
                        n_site,
                        n_elec,
                    );
                }
                [(first, _), (second, _)] => {
                    let (ma, sa) = split(*first);
                    let (mb, sb) = split(*second);
                    calculate_new_pf_m_two2_complex_flat(
                        ma,
                        sa,
                        mb,
                        sb,
                        &mut pf_new,
                        &new_idx,
                        &slater.slater_elm,
                        slater.inv_m.as_slice(),
                        stride,
                        &slater.pf_m,
                        0,
                        n_qp,
                        n_site,
                        n_elec,
                    );
                }
                _ => unreachable!("checkGF takes one or two hops"),
            }
            let z = calculate_ip_complex(&pf_new, 0, n_qp, self.data);
            crate::c_complex::divide(z, self.ip).norm()
        }
    }

    /// C `checkGF1`: `|<psi|x'>/<psi|x>| ` of the single hop `rj -> ri` (no projection).
    fn check_gf1(&self, ri: usize, rj: usize, s: u8) -> f64 {
        let n_site = self.n_site();
        let mj = self.ele_cfg[rj + s as usize * n_site] as usize;
        let slot = mj + s as usize * self.n_elec();
        self.pfaffian_ratio(&[(slot, ri)])
    }

    /// C `checkGF2`: the same for the hops `rl -> rk` (spin `t`) and `rj -> ri` (spin `s`).
    #[allow(clippy::too_many_arguments)]
    fn check_gf2(&self, ri: usize, rj: usize, rk: usize, rl: usize, s: u8, t: u8) -> f64 {
        let n_site = self.n_site();
        let n_elec = self.n_elec();
        let mj = self.ele_cfg[rj + s as usize * n_site] as usize;
        let ml = self.ele_cfg[rl + t as usize * n_site] as usize;
        self.pfaffian_ratio(&[
            (ml + t as usize * n_elec, rk),
            (mj + s as usize * n_elec, ri),
        ])
    }

    /// C `GreenFuncN`: `<phi| c1 a1 c2 a2 ... cn an |x>/<phi|x>` for creation/annihilation
    /// spin-orbital indices `rsi[k]`/`rsj[k]`.
    fn green_n(&self, rsi: &mut Vec<usize>, rsj: &mut Vec<usize>) -> Complex64 {
        let n_site = self.n_site();
        let zero = Complex64::new(0.0, 0.0);
        let n = rsi.len();
        for k in 0..n {
            if rsi[k] / n_site != rsj[k] / n_site {
                return zero;
            }
        }
        match n {
            0 => return zero,
            1 => return self.green1(rsi[0] % n_site, rsj[0] % n_site, (rsi[0] / n_site) as u8),
            2 => {
                return self.green2(
                    rsi[0] % n_site,
                    rsj[0] % n_site,
                    rsi[1] % n_site,
                    rsj[1] % n_site,
                    (rsi[0] / n_site) as u8,
                    (rsi[1] / n_site) as u8,
                )
            }
            _ => {}
        }
        // Reduction.
        for k in (0..n).rev() {
            let mut rsk = rsj[k];
            for l in (k + 1)..n {
                if rsk == rsi[l] {
                    rsj[k] = rsj[l];
                    rsi.remove(l);
                    rsj.remove(l);
                    return self.green_n(rsi, rsj);
                }
                if rsk == rsj[l] {
                    return zero;
                }
            }
            if self.ele_num[rsk] == 0 {
                return zero;
            }
            rsk = rsi[k];
            if rsk == rsj[k] {
                rsi.remove(k);
                rsj.remove(k);
                return self.green_n(rsi, rsj);
            }
            for l in (k + 1)..n {
                if rsk == rsi[l] {
                    return zero;
                }
                if rsk == rsj[l] {
                    rsi[k] = rsi[l];
                    rsi.remove(l);
                    rsj.remove(l);
                    return -self.green_n(rsi, rsj);
                }
            }
            if self.ele_num[rsk] == 1 {
                return zero;
            }
        }
        self.green_n_hopped(rsi, rsj)
    }

    /// The tail of `GreenFuncN` after the reductions: all `n` hops at once and the
    /// `calculateNewPfMN` Pfaffian of every QP.
    fn green_n_hopped(&self, rsi: &[usize], rsj: &[usize]) -> Complex64 {
        let n_site = self.n_site();
        let n_elec = self.n_elec();
        let n = rsi.len();
        let n_qp = self.state.slater_matrix.pf_m.len();
        let mut new_idx = self.ele_idx.to_vec();
        let mut new_num = self.ele_num.to_vec();
        let mut msj = Vec::with_capacity(n);
        for k in 0..n {
            let ri = rsi[k] % n_site;
            let sj = rsj[k] / n_site;
            let mj = self.ele_cfg[rsj[k]];
            if mj < 0 {
                return Complex64::new(0.0, 0.0);
            }
            let slot = mj as usize + sj * n_elec;
            msj.push(slot);
            new_idx[slot] = ri as i64;
            new_num[rsj[k]] = 0;
            new_num[rsi[k]] = 1;
        }
        let mut proj_new = vec![0_i64; self.ele_proj_cnt.len()];
        crate::sampling::projection::make_proj_cnt(&mut proj_new, &new_num, self.data);
        let log_ratio =
            crate::sampling::projection::log_proj_ratio(&proj_new, self.ele_proj_cnt, self.data);
        let real = self.is_real();
        let x = if real {
            Complex64::new(log_ratio.exp(), 0.0)
        } else {
            with_rbm_ratio(log_ratio.exp(), &new_num, self.ele_num, self.data)
        };
        let slater = &self.state.slater_matrix;
        let n_size = 2 * n_elec;
        let stride = n_size * n_size + 1;
        let mut pf_new = vec![Complex64::new(0.0, 0.0); n_qp];
        for qp in 0..n_qp {
            pf_new[qp] = if real {
                Complex64::new(
                    new_pf_child_real(
                        qp,
                        &msj,
                        &new_idx,
                        n_site,
                        n_elec,
                        &slater.slater_elm_real,
                        &slater.inv_m_real.as_slice()[qp * stride..qp * stride + n_size * n_size],
                        slater.pf_m_real[qp],
                    ),
                    0.0,
                )
            } else {
                new_pf_child_complex(
                    qp,
                    &msj,
                    &new_idx,
                    n_site,
                    n_elec,
                    &slater.slater_elm,
                    &slater.inv_m.as_slice()[qp * stride..qp * stride + n_size * n_size],
                    slater.pf_m[qp],
                )
            };
        }
        if real {
            let pf_real: Vec<f64> = pf_new.iter().map(|v| v.re).collect();
            let z = calculate_ip_real(&pf_real, 0, n_qp, self.data);
            Complex64::new(x.re * z / self.ip.re, 0.0)
        } else {
            let z = calculate_ip_complex(&pf_new, 0, n_qp, self.data);
            crate::c_complex::divide(x * z, self.ip).conj()
        }
    }

    /// C `CalculateHamiltonian0` of the configuration after the listed occupation changes.
    fn h0_after(&self, creations: &[usize], annihilations: &[usize]) -> Complex64 {
        let mut num = self.ele_num.to_vec();
        for &r in creations {
            num[r] = 1;
        }
        for &r in annihilations {
            num[r] = 0;
        }
        calculate_hamiltonian_diagonal(&num, self.data)
    }

    /// `myValue` of the OpenMP loops: the serial running sum over Transfer, PairHop, Exchange
    /// and InterAll with the operators `extra_i/extra_j` appended to each term's Green function.
    fn expand_terms(&self, extra_i: &[usize], extra_j: &[usize]) -> Complex64 {
        let n_site = self.n_site();
        let real = self.is_real();
        let coef = |value: Complex64| {
            if real {
                Complex64::new(value.re, 0.0)
            } else {
                value
            }
        };
        let mut value = Complex64::new(0.0, 0.0);
        let green_with = |first_i: &[usize], first_j: &[usize]| -> Complex64 {
            let mut rsi: Vec<usize> = first_i.to_vec();
            let mut rsj: Vec<usize> = first_j.to_vec();
            rsi.extend_from_slice(extra_i);
            rsj.extend_from_slice(extra_j);
            self.green_n(&mut rsi, &mut rsj)
        };
        // Transfer. With one extra operator pair C calls `GreenFunc2` directly with the term
        // first (`rk, rl, ri, rj`); `green_n` reaches the same function for n == 2.
        for term in &self.data.transfer_terms {
            let (Ok(rk), Ok(rl)) = (usize::try_from(term.site1), usize::try_from(term.site2))
            else {
                continue;
            };
            let first_i = [rk + term.spin1.as_code() as usize * n_site];
            let first_j = [rl + term.spin2.as_code() as usize * n_site];
            value -= coef(term.value) * green_with(&first_i, &first_j);
        }
        for term in &self.data.pair_hop_terms {
            let (Ok(r0), Ok(r1)) = (usize::try_from(term.site1), usize::try_from(term.site2))
            else {
                continue;
            };
            let first_i = [r0, r0 + n_site];
            let first_j = [r1, r1 + n_site];
            value += Complex64::new(term.value, 0.0) * green_with(&first_i, &first_j);
        }
        for term in &self.data.exchange_terms {
            let (Ok(r0), Ok(r1)) = (usize::try_from(term.site1), usize::try_from(term.site2))
            else {
                continue;
            };
            let first_i = [r0, r1 + n_site];
            let first_j = [r1, r0 + n_site];
            value += Complex64::new(term.value, 0.0) * green_with(&first_i, &first_j);
            let first_i = [r0 + n_site, r1];
            let first_j = [r1 + n_site, r0];
            value += Complex64::new(term.value, 0.0) * green_with(&first_i, &first_j);
        }
        for term in &self.data.inter_all_terms {
            let (Ok(a), Ok(b), Ok(c), Ok(d)) = (
                usize::try_from(term.site0),
                usize::try_from(term.site1),
                usize::try_from(term.site2),
                usize::try_from(term.site3),
            ) else {
                continue;
            };
            let s1 = term.spin1 as usize;
            let s3 = term.spin3 as usize;
            let first_i = [a + s1 * n_site, c + s3 * n_site];
            let first_j = [b + s1 * n_site, d + s3 * n_site];
            value += coef(term.value) * green_with(&first_i, &first_j);
        }
        value
    }

    /// C `calHCA2` for the hop `rj -> ri` of spin `s` (`ri != rj`, `ri` empty, `rj` occupied).
    pub(super) fn cal_hca2(&self, ri: usize, rj: usize, s: u8) -> Complex64 {
        let n_site = self.n_site();
        let rsi = ri + s as usize * n_site;
        let rsj = rj + s as usize * n_site;
        let g = self.green1(ri, rj, s);
        let mut val = g * self.h0_after(&[rsi], &[rsj]);
        let v = self.expand_terms(&[rsi], &[rsj]);
        val += v;
        val
    }

    /// C `calHCACA2` for `c†_{ri,si} c_{rj,si} c†_{rk,sk} c_{rl,sk}` with four distinct
    /// spin orbitals (`ri`, `rk` empty, `rj`, `rl` occupied).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn cal_hcaca2(
        &self,
        ri: usize,
        rj: usize,
        rk: usize,
        rl: usize,
        si: u8,
        sk: u8,
    ) -> Complex64 {
        let n_site = self.n_site();
        let rsi = ri + si as usize * n_site;
        let rsj = rj + si as usize * n_site;
        let rsk = rk + sk as usize * n_site;
        let rsl = rl + sk as usize * n_site;
        let g = self.green2(ri, rj, rk, rl, si, sk);
        let mut val = g * self.h0_after(&[rsi, rsk], &[rsj, rsl]);
        let v = self.expand_terms(&[rsi, rsk], &[rsj, rsl]);
        val += v;
        val
    }

    /// C `calHCA`'s dispatch: `Some(calHCA2(..))` exactly when C takes the `calHCA2` branch
    /// (the hop is allowed and `|checkGF1| <= 1e-12`), otherwise `None` (the ratio form
    /// `calHCA1`, or one of the trivial early returns, applies).
    pub(super) fn nodal_hca(&self, ri: usize, rj: usize, s: u8) -> Option<Complex64> {
        let n_site = self.n_site();
        if ri >= n_site || rj >= n_site || s > 1 {
            return None;
        }
        let rsi = ri + s as usize * n_site;
        let rsj = rj + s as usize * n_site;
        if rsi == rsj || self.ele_num[rsj] == 0 || self.ele_num[rsi] == 1 {
            return None;
        }
        let g = self.check_gf1(ri, rj, s);
        (g <= NODAL_THRESHOLD).then(|| self.cal_hca2(ri, rj, s))
    }

    /// C `calHCACA`'s dispatch with the same convention as [`Self::nodal_hca`].
    #[allow(clippy::too_many_arguments)]
    pub(super) fn nodal_hcaca(
        &self,
        ri: usize,
        rj: usize,
        rk: usize,
        rl: usize,
        si: u8,
        sk: u8,
    ) -> Option<Complex64> {
        let n_site = self.n_site();
        if [ri, rj, rk, rl].iter().any(|&r| r >= n_site) || si > 1 || sk > 1 {
            return None;
        }
        let rsi = ri + si as usize * n_site;
        let rsj = rj + si as usize * n_site;
        let rsk = rk + sk as usize * n_site;
        let rsl = rl + sk as usize * n_site;
        let occupied = |r: usize| self.ele_num[r] == 1;
        if rsk == rsl {
            return if occupied(rsk) {
                self.nodal_hca(ri, rj, si)
            } else {
                None
            };
        } else if rsj == rsk {
            return if occupied(rsj) {
                None
            } else {
                self.nodal_hca(ri, rl, si)
            };
        } else if rsj == rsl {
            return None;
        } else if rsi == rsj {
            return if occupied(rsi) {
                self.nodal_hca(rk, rl, sk)
            } else {
                None
            };
        } else if rsi == rsk {
            return None;
        } else if rsi == rsl {
            return if occupied(rsi) {
                self.nodal_hca(rk, rj, sk).map(|v| -v)
            } else {
                None
            };
        }
        if !occupied(rsl) || occupied(rsk) || !occupied(rsj) || occupied(rsi) {
            return None;
        }
        (self.check_gf2(ri, rj, rk, rl, si, sk) <= NODAL_THRESHOLD)
            .then(|| self.cal_hcaca2(ri, rj, rk, rl, si, sk))
    }
}

/// What C's `calHCA`/`calHCACA` dispatch reduces an operator to before the numerical branch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum Reduced {
    /// The operator annihilates the configuration.
    Zero,
    /// `sign * <psi|H|x>/<psi|x>` (the local energy `h1`).
    H1(f64),
    /// `sign * calHCA(ri, rj, s)` with `ri != rj`, `ri` empty and `rj` occupied.
    Hop {
        sign: f64,
        ri: usize,
        rj: usize,
        s: u8,
    },
    /// Four distinct spin orbitals with the required occupations (`calHCACA1/2`).
    General,
}

impl Reduced {
    fn negated(self) -> Self {
        match self {
            Reduced::H1(sign) => Reduced::H1(-sign),
            Reduced::Hop { sign, ri, rj, s } => Reduced::Hop {
                sign: -sign,
                ri,
                rj,
                s,
            },
            other => other,
        }
    }
}

impl NodalContext<'_> {
    /// The early returns of C `calHCA`.
    pub(super) fn plan_hca(&self, ri: usize, rj: usize, s: u8) -> Reduced {
        let n_site = self.n_site();
        if ri >= n_site || rj >= n_site || s > 1 {
            return Reduced::Zero;
        }
        let rsi = ri + s as usize * n_site;
        let rsj = rj + s as usize * n_site;
        if rsi == rsj {
            return if self.ele_num[rsi] == 1 {
                Reduced::H1(1.0)
            } else {
                Reduced::Zero
            };
        }
        if self.ele_num[rsj] == 0 || self.ele_num[rsi] == 1 {
            return Reduced::Zero;
        }
        Reduced::Hop {
            sign: 1.0,
            ri,
            rj,
            s,
        }
    }

    /// The coincident-index reductions and early returns of C `calHCACA`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn plan_hcaca(
        &self,
        ri: usize,
        rj: usize,
        rk: usize,
        rl: usize,
        si: u8,
        sk: u8,
    ) -> Reduced {
        let n_site = self.n_site();
        if [ri, rj, rk, rl].iter().any(|&r| r >= n_site) || si > 1 || sk > 1 {
            return Reduced::Zero;
        }
        let rsi = ri + si as usize * n_site;
        let rsj = rj + si as usize * n_site;
        let rsk = rk + sk as usize * n_site;
        let rsl = rl + sk as usize * n_site;
        let occupied = |r: usize| self.ele_num[r] == 1;
        if rsk == rsl {
            return if occupied(rsk) {
                self.plan_hca(ri, rj, si)
            } else {
                Reduced::Zero
            };
        } else if rsj == rsk {
            return if occupied(rsj) {
                Reduced::Zero
            } else {
                self.plan_hca(ri, rl, si)
            };
        } else if rsj == rsl {
            return Reduced::Zero;
        } else if rsi == rsj {
            return if occupied(rsi) {
                self.plan_hca(rk, rl, sk)
            } else {
                Reduced::Zero
            };
        } else if rsi == rsk {
            return Reduced::Zero;
        } else if rsi == rsl {
            return if occupied(rsi) {
                self.plan_hca(rk, rj, sk).negated()
            } else {
                Reduced::Zero
            };
        }
        if !occupied(rsl) || occupied(rsk) || !occupied(rsj) || occupied(rsi) {
            return Reduced::Zero;
        }
        Reduced::General
    }
}

/// C `calculateNewPfMN_real_child` followed by `sgn * pfaff * PfM`: the Pfaffian of the
/// configuration in which the electrons `msa` have hopped (`ele_idx` already holds their new
/// sites), from the unhopped inverse `inv` (`[nsize][nsize]`, row-major, no pad slot) and
/// Pfaffian `pf_m`.
///
/// C fills a row-major matrix and hands it to the column-major `DSKPFA`, which therefore sees the
/// transpose. The same layout is kept here, so the sign convention of C is preserved.
fn new_pf_child_real(
    qp: usize,
    msa: &[usize],
    ele_idx: &[i64],
    n_site: usize,
    n_elec: usize,
    slater: &crate::state::SlaterElmFlat<f64>,
    inv: &[f64],
    pf_m: f64,
) -> f64 {
    let n = msa.len();
    let n2 = 2 * n;
    let nsize = 2 * n_elec;
    let rs_of = |slot: usize| ele_idx[slot] as usize + (slot / n_elec) * n_site;
    let vec: Vec<Vec<f64>> = (0..n)
        .map(|k| {
            let rsk = rs_of(msa[k]);
            (0..nsize)
                .map(|msi| slater.get(qp, rsk, rs_of(msi)))
                .collect()
        })
        .collect();
    let mut mat = vec![0.0_f64; n2 * n2];
    for k in 0..n {
        for l in (k + 1)..n {
            let mut val = 0.0;
            for msi in 0..nsize {
                let mut tmp = 0.0;
                for msj in 0..nsize {
                    tmp += inv[msi * nsize + msj] * vec[l][msj];
                }
                val += tmp * vec[k][msi];
            }
            mat[n2 * k + l] = val + vec[k][msa[l]];
        }
    }
    for k in 0..n {
        for l in 0..n {
            let mut val = 0.0;
            for msi in 0..nsize {
                val += vec[k][msi] * inv[msa[l] * nsize + msi];
            }
            mat[n2 * k + n + l] = val;
        }
    }
    for k in 0..n {
        for l in (k + 1)..n {
            mat[n2 * (k + n) + n + l] = inv[msa[k] * nsize + msa[l]];
        }
    }
    for k in 0..n2 {
        for l in 0..k {
            mat[n2 * k + l] = -mat[n2 * l + k];
        }
        mat[n2 * k + k] = 0.0;
    }
    let pfaff = pfapack::pfaffian_ltl_real(&mut pfapack::SqMat::new(&mut mat, n2));
    let sgn = if (n * (n - 1) / 2).is_multiple_of(2) {
        1.0
    } else {
        -1.0
    };
    sgn * pfaff * pf_m
}

/// Complex counterpart of [`new_pf_child_real`] with the intended `2n` matrix order.
fn new_pf_child_complex(
    qp: usize,
    msa: &[usize],
    ele_idx: &[i64],
    n_site: usize,
    n_elec: usize,
    slater: &crate::state::SlaterElmFlat<Complex64>,
    inv: &[Complex64],
    pf_m: Complex64,
) -> Complex64 {
    let zero = Complex64::new(0.0, 0.0);
    let n = msa.len();
    let n2 = 2 * n;
    let nsize = 2 * n_elec;
    let rs_of = |slot: usize| ele_idx[slot] as usize + (slot / n_elec) * n_site;
    let vec: Vec<Vec<Complex64>> = (0..n)
        .map(|k| {
            let rsk = rs_of(msa[k]);
            (0..nsize)
                .map(|msi| slater.get(qp, rsk, rs_of(msi)))
                .collect()
        })
        .collect();
    let mut mat = vec![zero; n2 * n2];
    for k in 0..n {
        for l in (k + 1)..n {
            let mut val = zero;
            for msi in 0..nsize {
                let mut tmp = zero;
                for msj in 0..nsize {
                    tmp += inv[msi * nsize + msj] * vec[l][msj];
                }
                val += tmp * vec[k][msi];
            }
            mat[n2 * k + l] = val + vec[k][msa[l]];
        }
    }
    for k in 0..n {
        for l in 0..n {
            let mut val = zero;
            for msi in 0..nsize {
                val += vec[k][msi] * inv[msa[l] * nsize + msi];
            }
            mat[n2 * k + n + l] = val;
        }
    }
    for k in 0..n {
        for l in (k + 1)..n {
            mat[n2 * (k + n) + n + l] = inv[msa[k] * nsize + msa[l]];
        }
    }
    for k in 0..n2 {
        for l in 0..k {
            mat[n2 * k + l] = -mat[n2 * l + k];
        }
        mat[n2 * k + k] = zero;
    }
    let pfaff = pfapack::pfaffian_ltl_complex(&mut pfapack::SqMat::new(&mut mat, n2));
    let sgn = if (n * (n - 1) / 2).is_multiple_of(2) {
        1.0
    } else {
        -1.0
    };
    pfaff * pf_m * sgn
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `calculateNewPfMN` for n = 1 to 4 moved electrons equals the Pfaffian of the moved
    /// configuration recomputed from scratch (issue #481). C hands its row-major matrix to the
    /// column-major `DSKPFA`; the port keeps that layout and the result is still the true
    /// Pfaffian, with the same sign.
    #[test]
    fn new_pf_child_matches_full_pfaffian() {
        let (n_site, n_elec, n_qp) = (6usize, 3usize, 2usize);
        let n2 = 2 * n_site;
        let n_size = 2 * n_elec;
        let mut state = VmcOptimizationState::zeros(n_site, n_elec, 0, 0, n_qp, 0, false, false);
        let mut seed = 987_u64;
        let mut rand = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        for qp in 0..n_qp {
            for i in 0..n2 {
                for j in (i + 1)..n2 {
                    let v = rand();
                    state.slater_matrix.slater_elm_real.set(qp, i, j, v);
                    state.slater_matrix.slater_elm_real.set(qp, j, i, -v);
                }
            }
        }
        let pool = crate::state::ThreadedPfaPackWorkspace::new(n_size, 1);
        let base_idx: Vec<i64> = vec![0, 2, 4, 1, 3, 5];
        let recompute = |tables: &mut crate::state::SlaterMatrixData, idx: &[i64]| {
            crate::pfaffian::calc_m_all_real(
                idx,
                &tables.slater_elm_real,
                &mut tables.inv_m_real,
                &mut tables.pf_m_real,
                0,
                n_qp,
                n_site,
                n_elec,
                &pool,
            )
            .unwrap();
        };
        recompute(&mut state.slater_matrix, &base_idx);
        let base = state.slater_matrix.clone();
        let stride = n_size * n_size + 1;
        // (electron slots that move, moved configuration)
        let cases: Vec<(Vec<usize>, Vec<i64>)> = vec![
            (vec![1], vec![0, 3, 4, 1, 3, 5]),
            (vec![0, 3], vec![1, 2, 4, 0, 3, 5]),
            (vec![0, 1, 3], vec![1, 3, 4, 0, 3, 5]),
            (vec![0, 1, 3, 5], vec![1, 3, 4, 0, 3, 2]),
        ];
        for (msa, moved) in cases {
            let mut full = base.clone();
            recompute(&mut full, &moved);
            let n = msa.len();
            for qp in 0..n_qp {
                let child = new_pf_child_real(
                    qp,
                    &msa,
                    &moved,
                    n_site,
                    n_elec,
                    &base.slater_elm_real,
                    &base.inv_m_real.as_slice()[qp * stride..qp * stride + n_size * n_size],
                    base.pf_m_real[qp],
                );
                let expected = full.pf_m_real[qp];
                assert!(
                    (child - expected).abs() <= 1e-9 * expected.abs().max(1e-3),
                    "n={n} qp={qp}: child {child} full {expected}"
                );
            }
        }
    }
}

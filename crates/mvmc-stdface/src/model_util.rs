//! Port of `StdFace_ModelUtil.c` (mVMC branch): value printing/validation helpers, the
//! super-cell setup, transfer/interaction accumulation and the geometry, Jastrow, orbital
//! and projection writers shared by every lattice.
#![allow(non_snake_case)]

use crate::ccomplex::{cpow, C64};
use crate::cexpr;
use crate::cfmt;
use crate::out::{exit, Out, Res};
use crate::outf;
use crate::vals::{StdIntList, NAN_I};

const DEFAULT_TAG: &str = "  ######  DEFAULT VALUE IS USED  ######";

/// `StdFace_trans`: add `trans0 c^dagger_{i s} c_{j s'}` (skipped when `|trans0| < 1e-12`).
pub fn trans(s: &mut StdIntList, trans0: C64, isite: i32, ispin: i32, jsite: i32, jspin: i32) {
    if trans0.abs() < 1.0e-12 {
        return;
    }
    s.trans.push(trans0);
    s.transindx.push([isite, ispin, jsite, jspin]);
}

/// `StdFace_Hopping` (mVMC): both spin channels, both directions. `dR` is only used by HPhi.
pub fn hopping(s: &mut StdIntList, trans0: C64, isite: i32, jsite: i32, _dR: &[f64; 3]) {
    for ispin in 0..2 {
        trans(s, trans0, jsite, ispin, isite, ispin);
        trans(s, trans0.conj(), isite, ispin, jsite, ispin);
    }
}

/// `StdFace_HubbardLocal`: chemical potential, fields and the on-site Coulomb term.
pub fn hubbard_local(
    s: &mut StdIntList,
    mu0: f64,
    h0: f64,
    gamma0: f64,
    gamma0_y: f64,
    u0: f64,
    isite: i32,
) {
    trans(s, C64::real(mu0 - 0.5 * h0), isite, 0, isite, 0);
    trans(s, C64::real(mu0 + 0.5 * h0), isite, 1, isite, 1);
    trans(s, C64::real(-0.5 * gamma0), isite, 1, isite, 0);
    trans(s, C64::real(-0.5 * gamma0), isite, 0, isite, 1);
    trans(s, cexpr::hubbard_local_im_neg(gamma0_y), isite, 1, isite, 0);
    trans(s, cexpr::hubbard_local_im_pos(gamma0_y), isite, 0, isite, 1);
    s.Cintra.push(u0);
    s.CintraIndx.push([isite]);
}

/// `StdFace_MagField`: Bogoliubov representation of a longitudinal and transverse field.
pub fn mag_field(s: &mut StdIntList, s2: i32, h: f64, gamma: f64, gamma_y: f64, isite: i32) {
    let spin = s2 as f64 * 0.5;
    for ispin in 0..=s2 {
        let sz = spin - ispin as f64;
        trans(s, C64::real(-h * sz), isite, ispin, isite, ispin);
        if ispin > 0 {
            trans(
                s,
                cexpr::mag_field_minus(gamma, gamma_y, spin, sz),
                isite,
                ispin,
                isite,
                ispin - 1,
            );
            trans(
                s,
                cexpr::mag_field_plus(gamma, gamma_y, spin, sz),
                isite,
                ispin - 1,
                isite,
                ispin,
            );
        }
    }
}

/// `StdFace_intr`: add a general two-body term (skipped when `|intr0| < 1e-12`).
#[allow(clippy::too_many_arguments)]
pub fn intr(
    s: &mut StdIntList,
    intr0: C64,
    site1: i32,
    spin1: i32,
    site2: i32,
    spin2: i32,
    site3: i32,
    spin3: i32,
    site4: i32,
    spin4: i32,
) {
    if intr0.abs() < 1.0e-12 {
        return;
    }
    s.intr.push(intr0);
    s.intrindx
        .push([site1, spin1, site2, spin2, site3, spin3, site4, spin4]);
}

/// `StdFace_GeneralJ`: a 3x3 spin coupling between spins `Si2/2` and `Sj2/2`.
pub fn general_j(
    s: &mut StdIntList,
    j: &[[f64; 3]; 3],
    si2: i32,
    sj2: i32,
    isite: i32,
    jsite: i32,
) {
    let mut z_general = true;
    let mut ex_general = true;
    if si2 == 1 || sj2 == 1 {
        z_general = false;
        s.Hund.push(-0.5 * j[2][2]);
        s.HundIndx.push([isite, jsite]);
        s.Cinter.push(-0.25 * j[2][2]);
        s.CinterIndx.push([isite, jsite]);
        if j[0][1].abs() < 0.000001
            && j[1][0].abs() < 0.000001
            && (j[0][0] - j[1][1]).abs() < 0.000001
        {
            ex_general = false;
            s.Ex.push(-0.25 * (j[0][0] + j[1][1]));
            s.ExIndx.push([isite, jsite]);
            s.PairLift.push(0.25 * (j[0][0] - j[1][1]));
            s.PLIndx.push([isite, jsite]);
        }
    }
    let si = 0.5 * si2 as f64;
    let sj = 0.5 * sj2 as f64;
    for ispin in 0..=si2 {
        let siz = si - ispin as f64;
        for jspin in 0..=sj2 {
            let sjz = sj - jspin as f64;
            if z_general {
                let intr0 = cexpr::general_j_zz(j[2][2], siz, sjz);
                intr(
                    s, intr0, isite, ispin, isite, ispin, jsite, jspin, jsite, jspin,
                );
            }
            if ispin > 0 && jspin > 0 && ex_general {
                let intr0 = cexpr::general_j_pm(j, si, siz, sj, sjz);
                intr(
                    s,
                    intr0,
                    isite,
                    ispin - 1,
                    isite,
                    ispin,
                    jsite,
                    jspin,
                    jsite,
                    jspin - 1,
                );
                intr(
                    s,
                    intr0.conj(),
                    isite,
                    ispin,
                    isite,
                    ispin - 1,
                    jsite,
                    jspin - 1,
                    jsite,
                    jspin,
                );
            }
            if ispin > 0 && jspin > 0 && ex_general {
                let intr0 = cexpr::general_j_pp(j, si, siz, sj, sjz);
                intr(
                    s,
                    intr0,
                    isite,
                    ispin - 1,
                    isite,
                    ispin,
                    jsite,
                    jspin - 1,
                    jsite,
                    jspin,
                );
                intr(
                    s,
                    intr0.conj(),
                    isite,
                    ispin,
                    isite,
                    ispin - 1,
                    jsite,
                    jspin,
                    jsite,
                    jspin - 1,
                );
            }
            if ispin > 0 {
                let intr0 = cexpr::general_j_pz(j, si, siz, sjz);
                intr(
                    s,
                    intr0,
                    isite,
                    ispin - 1,
                    isite,
                    ispin,
                    jsite,
                    jspin,
                    jsite,
                    jspin,
                );
                intr(
                    s,
                    intr0.conj(),
                    jsite,
                    jspin,
                    jsite,
                    jspin,
                    isite,
                    ispin,
                    isite,
                    ispin - 1,
                );
            }
            if jspin > 0 {
                let intr0 = cexpr::general_j_zp(j, sj, siz, sjz);
                intr(
                    s,
                    intr0,
                    isite,
                    ispin,
                    isite,
                    ispin,
                    jsite,
                    jspin - 1,
                    jsite,
                    jspin,
                );
                intr(
                    s,
                    intr0.conj(),
                    jsite,
                    jspin,
                    jsite,
                    jspin - 1,
                    isite,
                    ispin,
                    isite,
                    ispin,
                );
            }
        }
    }
}

/// `StdFace_Coulomb`: add `V n_i n_j`.
pub fn coulomb(s: &mut StdIntList, v: f64, isite: i32, jsite: i32) {
    s.Cinter.push(v);
    s.CinterIndx.push([isite, jsite]);
}

/// `StdFace_PrintVal_d`.
pub fn print_val_d(o: &mut Out, valname: &str, val: &mut f64, val0: f64) {
    if val.is_nan() {
        *val = val0;
        outf!(
            o,
            "  {} = {}{DEFAULT_TAG}\n",
            cfmt::s(valname, 15),
            cfmt::f_left(*val, 10, 5)
        );
    } else {
        outf!(
            o,
            "  {} = {}\n",
            cfmt::s(valname, 15),
            cfmt::f_left(*val, 10, 5)
        );
    }
}

/// `StdFace_PrintVal_dd`.
pub fn print_val_dd(o: &mut Out, valname: &str, val: &mut f64, val0: f64, val1: f64) {
    if val.is_nan() {
        *val = if val0.is_nan() { val1 } else { val0 };
        outf!(
            o,
            "  {} = {}{DEFAULT_TAG}\n",
            cfmt::s(valname, 15),
            cfmt::f_left(*val, 10, 5)
        );
    } else {
        outf!(
            o,
            "  {} = {}\n",
            cfmt::s(valname, 15),
            cfmt::f_left(*val, 10, 5)
        );
    }
}

/// `StdFace_PrintVal_c`.
pub fn print_val_c(o: &mut Out, valname: &str, val: &mut C64, val0: C64) {
    if val.re.is_nan() {
        *val = val0;
        outf!(
            o,
            "  {} = {} {}{DEFAULT_TAG}\n",
            cfmt::s(valname, 15),
            cfmt::f_left(val.re, 10, 5),
            cfmt::f_left(val.im, 10, 5)
        );
    } else {
        outf!(
            o,
            "  {} = {} {}\n",
            cfmt::s(valname, 15),
            cfmt::f_left(val.re, 10, 5),
            cfmt::f_left(val.im, 10, 5)
        );
    }
}

/// `StdFace_PrintVal_i`.
pub fn print_val_i(o: &mut Out, valname: &str, val: &mut i32, val0: i32) {
    if *val == NAN_I {
        *val = val0;
        outf!(
            o,
            "  {} = {}{DEFAULT_TAG}\n",
            cfmt::s(valname, 15),
            cfmt::d_left(*val, 10)
        );
    } else {
        outf!(
            o,
            "  {} = {}\n",
            cfmt::s(valname, 15),
            cfmt::d_left(*val, 10)
        );
    }
}

fn not_used_message(o: &mut Out, valname: &str) {
    outf!(
        o,
        "\n Check !  {valname} is SPECIFIED but will NOT be USED. \n"
    );
    o.print("            Please COMMENT-OUT this line \n");
    o.print("            or check this input is REALLY APPROPRIATE for your purpose ! \n\n");
}

/// `StdFace_NotUsed_d`.
pub fn not_used_d(o: &mut Out, valname: &str, val: f64) -> Res<()> {
    if !val.is_nan() {
        not_used_message(o, valname);
        return exit(-1);
    }
    Ok(())
}

/// `StdFace_NotUsed_c`.
pub fn not_used_c(o: &mut Out, valname: &str, val: C64) -> Res<()> {
    if !val.re.is_nan() {
        not_used_message(o, valname);
        return exit(-1);
    }
    Ok(())
}

/// `StdFace_NotUsed_J`.
pub fn not_used_j(o: &mut Out, valname: &str, j_all: f64, j: &[[f64; 3]; 3]) -> Res<()> {
    let suffix = [["x", "xy", "xz"], ["yx", "y", "yz"], ["zx", "zy", "z"]];
    not_used_d(o, valname, j_all)?;
    for i1 in 0..3 {
        for i2 in 0..3 {
            not_used_d(o, &format!("{valname}{}", suffix[i1][i2]), j[i1][i2])?;
        }
    }
    Ok(())
}

/// `StdFace_NotUsed_i`.
pub fn not_used_i(o: &mut Out, valname: &str, val: i32) -> Res<()> {
    if val != NAN_I {
        not_used_message(o, valname);
        return exit(-1);
    }
    Ok(())
}

/// `StdFace_RequiredVal_i`.
pub fn required_val_i(o: &mut Out, valname: &str, val: i32) -> Res<()> {
    if val == NAN_I {
        outf!(o, "ERROR ! {valname} is NOT specified !\n");
        return exit(-1);
    }
    outf!(o, "  {} = {}\n", cfmt::s(valname, 15), cfmt::d_left(val, 3));
    Ok(())
}

/// `StdFace_FoldSite`: returns `(nBox, iCellV_fold)`.
pub fn fold_site(s: &StdIntList, icell_v: [i32; 3]) -> ([i32; 3], [i32; 3]) {
    let mut frac = [0i32; 3];
    for ii in 0..3 {
        for jj in 0..3 {
            frac[ii] += s.rbox[ii][jj] * icell_v[jj];
        }
    }
    let mut n_box = [0i32; 3];
    for ii in 0..3 {
        n_box[ii] = (frac[ii] + s.NCell * 1000) / s.NCell - 1000;
    }
    for ii in 0..3 {
        frac[ii] -= s.NCell * n_box[ii];
    }
    let mut fold = [0i32; 3];
    for ii in 0..3 {
        for jj in 0..3 {
            fold[ii] += s.box_[jj][ii] * frac[jj];
        }
        fold[ii] = (fold[ii] + s.NCell * 1000) / s.NCell - 1000;
    }
    (n_box, fold)
}

/// `StdFace_InitSite`: super-cell setup. `gp` is the optional `lattice.gp` buffer (`fp`).
pub fn init_site(o: &mut Out, s: &mut StdIntList, gp: &mut Option<String>, dim: i32) -> Res<()> {
    o.print("\n  @ Super-Lattice setting\n\n");
    let box_set = s.box_.iter().flatten().any(|&b| b != NAN_I);
    let lwh_set = s.L != NAN_I || s.W != NAN_I || s.Height != NAN_I;
    if lwh_set && box_set {
        o.print("\nERROR ! (L, W, Height) and (a0W, ..., a2H) conflict !\n\n");
        return exit(-1);
    } else if lwh_set {
        print_val_i(o, "L", &mut s.L, 1);
        print_val_i(o, "W", &mut s.W, 1);
        print_val_i(o, "Height", &mut s.Height, 1);
        s.box_ = [[0; 3]; 3];
        s.box_[0][0] = s.W;
        s.box_[1][1] = s.L;
        s.box_[2][2] = s.Height;
    } else {
        print_val_i(o, "a0W", &mut s.box_[0][0], 1);
        print_val_i(o, "a0L", &mut s.box_[0][1], 0);
        print_val_i(o, "a0H", &mut s.box_[0][2], 0);
        print_val_i(o, "a1W", &mut s.box_[1][0], 0);
        print_val_i(o, "a1L", &mut s.box_[1][1], 1);
        print_val_i(o, "a1H", &mut s.box_[1][2], 0);
        print_val_i(o, "a2W", &mut s.box_[2][0], 0);
        print_val_i(o, "a2L", &mut s.box_[2][1], 0);
        print_val_i(o, "a2H", &mut s.box_[2][2], 1);
    }
    if dim == 2 {
        s.direct[0][2] = 0.0;
        s.direct[1][2] = 0.0;
        s.direct[2][0] = 0.0;
        s.direct[2][1] = 0.0;
        s.direct[2][2] = 1.0;
    }
    if dim == 2 {
        s.phase[2] = 0.0;
    }
    for ii in 0..3 {
        let angle = s.pi180 * s.phase[ii];
        s.ExpPhase[ii] = C64::new(angle.cos(), 0.0) + C64::I * C64::real(angle.sin());
        s.AntiPeriod[ii] = if (s.ExpPhase[ii] + C64::real(1.0)).abs() < 0.000001 {
            1
        } else {
            0
        };
    }
    s.tau = vec![[0.0; 3]; s.NsiteUC.max(0) as usize];
    s.NCell = 0;
    for ii in 0..3 {
        s.NCell += s.box_[0][ii] * s.box_[1][(ii + 1) % 3] * s.box_[2][(ii + 2) % 3]
            - s.box_[0][ii] * s.box_[1][(ii + 2) % 3] * s.box_[2][(ii + 1) % 3];
    }
    outf!(o, "   Number of Cell = {}\n", s.NCell.abs());
    if s.NCell == 0 {
        return exit(-1);
    }
    for ii in 0..3 {
        for jj in 0..3 {
            s.rbox[ii][jj] = s.box_[(ii + 1) % 3][(jj + 1) % 3]
                * s.box_[(ii + 2) % 3][(jj + 2) % 3]
                - s.box_[(ii + 1) % 3][(jj + 2) % 3] * s.box_[(ii + 2) % 3][(jj + 1) % 3];
        }
    }
    if s.NCell < 0 {
        for row in s.rbox.iter_mut() {
            for v in row.iter_mut() {
                *v = -*v;
            }
        }
        s.NCell = -s.NCell;
    }
    let mut bound = [[0i32; 2]; 3];
    for ii in 0..3 {
        for n2 in 0..2 {
            for n1 in 0..2 {
                for n0 in 0..2 {
                    let n_box = [n0, n1, n2];
                    let mut edge = 0;
                    for jj in 0..3 {
                        edge += n_box[jj] * s.box_[jj][ii];
                    }
                    if edge < bound[ii][0] {
                        bound[ii][0] = edge;
                    }
                    if edge > bound[ii][1] {
                        bound[ii][1] = edge;
                    }
                }
            }
        }
    }
    s.Cell = Vec::with_capacity(s.NCell as usize);
    for c2 in bound[2][0]..=bound[2][1] {
        for c1 in bound[1][0]..=bound[1][1] {
            for c0 in bound[0][0]..=bound[0][1] {
                let cell_v = [c0, c1, c2];
                let (n_box, _) = fold_site(s, cell_v);
                if n_box == [0, 0, 0] {
                    s.Cell.push(cell_v);
                }
            }
        }
    }
    s.Cell.resize(s.NCell as usize, [0; 3]);
    if dim == 2 {
        let d = s.direct;
        let bx = s.box_;
        let mut pos = [[0.0f64; 2]; 4];
        pos[1][0] = d[0][0] * bx[0][0] as f64 + d[1][0] * bx[0][1] as f64;
        pos[1][1] = d[0][1] * bx[0][0] as f64 + d[1][1] * bx[0][1] as f64;
        pos[2][0] = d[0][0] * bx[1][0] as f64 + d[1][0] * bx[1][1] as f64;
        pos[2][1] = d[0][1] * bx[1][0] as f64 + d[1][1] * bx[1][1] as f64;
        pos[3][0] = pos[1][0] + pos[2][0];
        pos[3][1] = pos[1][1] + pos[2][1];
        let mut xmin = 0.0f64;
        let mut xmax = 0.0f64;
        for p in &pos {
            if p[0] < xmin {
                xmin = p[0];
            }
            if p[0] > xmax {
                xmax = p[0];
            }
            if p[1] < xmin {
                xmin = p[1];
            }
            if p[1] > xmax {
                xmax = p[1];
            }
        }
        xmin -= 2.0;
        xmax += 2.0;
        if let Some(fp) = gp.as_mut() {
            fp.push_str("#set terminal pdf color enhanced \\\n");
            fp.push_str("#dashed dl 1.0 size 20.0cm, 20.0cm \n");
            fp.push_str("#set output \"lattice.pdf\"\n");
            fp.push_str(&format!(
                "set xrange [{}: {}]\n",
                cfmt::f_raw(xmin, 6),
                cfmt::f_raw(xmax, 6)
            ));
            fp.push_str(&format!(
                "set yrange [{}: {}]\n",
                cfmt::f_raw(xmin, 6),
                cfmt::f_raw(xmax, 6)
            ));
            fp.push_str("set size square\n");
            fp.push_str("unset key\n");
            fp.push_str("unset tics\n");
            fp.push_str("unset border\n");
            fp.push_str("set style line 1 lc 1 lt 1\n");
            fp.push_str("set style line 2 lc 5 lt 1\n");
            fp.push_str("set style line 3 lc 0 lt 1\n");
            for (a, b) in [(0usize, 1usize), (1, 3), (3, 2), (2, 0)] {
                fp.push_str(&format!(
                    "set arrow from {}, {} to {}, {} nohead front ls 3\n",
                    cfmt::f_raw(pos[a][0], 6),
                    cfmt::f_raw(pos[a][1], 6),
                    cfmt::f_raw(pos[b][0], 6),
                    cfmt::f_raw(pos[b][1], 6)
                ));
            }
        }
    }
    Ok(())
}

/// Result of [`find_site`] / [`set_label`].
#[derive(Clone, Copy, Debug)]
pub struct SitePair {
    /// Initial site index.
    pub isite: i32,
    /// Final site index.
    pub jsite: i32,
    /// Boundary phase when the bond crosses the super-cell boundary.
    pub cphase: C64,
    /// `R_i - R_j` in fractional coordinates.
    pub dR: [f64; 3],
}

/// `StdFace_FindSite`.
#[allow(clippy::too_many_arguments)]
pub fn find_site(
    s: &StdIntList,
    iw: i32,
    il: i32,
    ih: i32,
    diw: i32,
    dil: i32,
    dih: i32,
    isite_uc: i32,
    jsite_uc: i32,
) -> SitePair {
    let (iu, ju) = (isite_uc as usize, jsite_uc as usize);
    let dR = [
        -(diw as f64) + s.tau[iu][0] - s.tau[ju][0],
        -(dil as f64) + s.tau[iu][1] - s.tau[ju][1],
        -(dih as f64) + s.tau[iu][2] - s.tau[ju][2],
    ];
    let (n_box, jcell_v) = fold_site(s, [iw + diw, il + dil, ih + dih]);
    let mut cphase = C64::real(1.0);
    for ii in 0..3 {
        cphase = cphase * cpow(s.ExpPhase[ii], C64::real(n_box[ii] as f64));
    }
    let (mut icell, mut jcell) = (0i32, 0i32);
    for (kcell, cell) in s.Cell.iter().enumerate() {
        if jcell_v == *cell {
            jcell = kcell as i32;
        }
        if [iw, il, ih] == *cell {
            icell = kcell as i32;
        }
    }
    let mut isite = icell * s.NsiteUC + isite_uc;
    let mut jsite = jcell * s.NsiteUC + jsite_uc;
    if s.model == "kondo" {
        isite += s.NCell * s.NsiteUC;
        jsite += s.NCell * s.NsiteUC;
    }
    SitePair {
        isite,
        jsite,
        cphase,
        dR,
    }
}

fn gp_label(fp: &mut String, site: i32, x: f64, y: f64) {
    let width = if site < 10 { 1 } else { 2 };
    fp.push_str(&format!(
        "set label \"{}\" at {}, {} center front\n",
        cfmt::d(site, width),
        cfmt::f_raw(x, 6),
        cfmt::f_raw(y, 6)
    ));
}

/// `StdFace_SetLabel`: gnuplot labels (2D) plus the site pair of a bond.
#[allow(clippy::too_many_arguments)]
pub fn set_label(
    s: &StdIntList,
    gp: &mut Option<String>,
    iw: i32,
    il: i32,
    diw: i32,
    dil: i32,
    isite_uc: i32,
    jsite_uc: i32,
    connect: i32,
) -> SitePair {
    let (iu, ju) = (isite_uc as usize, jsite_uc as usize);
    let d = s.direct;
    // First print the reversed one.
    let pair = find_site(s, iw, il, 0, -diw, -dil, 0, jsite_uc, isite_uc);
    let xi = d[0][0] * (iw as f64 + s.tau[ju][0]) + d[1][0] * (il as f64 + s.tau[ju][1]);
    let yi = d[0][1] * (iw as f64 + s.tau[ju][0]) + d[1][1] * (il as f64 + s.tau[ju][1]);
    let xj =
        d[0][0] * ((iw - diw) as f64 + s.tau[iu][0]) + d[1][0] * ((il - dil) as f64 + s.tau[iu][1]);
    let yj =
        d[0][1] * ((iw - diw) as f64 + s.tau[iu][0]) + d[1][1] * ((il - dil) as f64 + s.tau[iu][1]);
    if let Some(fp) = gp.as_mut() {
        gp_label(fp, pair.isite, xi, yi);
        gp_label(fp, pair.jsite, xj, yj);
        if connect < 3 {
            fp.push_str(&format!(
                "set arrow from {}, {} to {}, {} nohead ls {}\n",
                cfmt::f_raw(xi, 6),
                cfmt::f_raw(yi, 6),
                cfmt::f_raw(xj, 6),
                cfmt::f_raw(yj, 6),
                connect
            ));
        }
    }
    // Then the normal one; these differ when the bond crosses the boundary.
    let pair = find_site(s, iw, il, 0, diw, dil, 0, isite_uc, jsite_uc);
    let xi = d[1][0] * (il as f64 + s.tau[iu][1]) + d[0][0] * (iw as f64 + s.tau[iu][0]);
    let yi = d[1][1] * (il as f64 + s.tau[iu][1]) + d[0][1] * (iw as f64 + s.tau[iu][0]);
    let xj =
        d[0][0] * ((iw + diw) as f64 + s.tau[ju][0]) + d[1][0] * ((il + dil) as f64 + s.tau[ju][1]);
    let yj =
        d[0][1] * ((iw + diw) as f64 + s.tau[ju][0]) + d[1][1] * ((il + dil) as f64 + s.tau[ju][1]);
    if let Some(fp) = gp.as_mut() {
        gp_label(fp, pair.isite, xi, yi);
        gp_label(fp, pair.jsite, xj, yj);
        if connect < 3 {
            fp.push_str(&format!(
                "set arrow from {}, {} to {}, {} nohead ls {}\n",
                cfmt::f_raw(xi, 6),
                cfmt::f_raw(yi, 6),
                cfmt::f_raw(xj, 6),
                cfmt::f_raw(yj, 6),
                connect
            ));
        }
    }
    pair
}

/// `StdFace_PrintXSF`: `lattice.xsf` (XCrysDen).
pub fn print_xsf(o: &mut Out, s: &StdIntList) -> Res<()> {
    let do_convvec = matches!(
        s.lattice.as_str(),
        "orthorhombic" | "face-centeredorthorhombic" | "fcorthorhombic" | "fco" | "pyrochlore"
    );
    let mut t = String::new();
    t.push_str("CRYSTAL\nPRIMVEC\n");
    for ii in 0..3 {
        let mut vec = [0.0f64; 3];
        for jj in 0..3 {
            for kk in 0..3 {
                vec[jj] += s.box_[ii][kk] as f64 * s.direct[kk][jj];
            }
        }
        t.push_str(&format!(
            "{} {} {}\n",
            cfmt::f(vec[0], 15, 5),
            cfmt::f(vec[1], 15, 5),
            cfmt::f(vec[2], 15, 5)
        ));
    }
    if do_convvec {
        t.push_str("CONVVEC\n");
        for ii in 0..3 {
            for jj in 0..3 {
                let v = if ii == jj { s.length[ii] } else { 0.0 };
                t.push_str(&format!("{} ", cfmt::f(v, 15, 5)));
            }
            t.push('\n');
        }
    }
    t.push_str("PRIMCOORD\n");
    t.push_str(&format!("{} 1\n", s.NCell * s.NsiteUC));
    for icell in 0..s.NCell as usize {
        for isite in 0..s.NsiteUC as usize {
            let mut vec = [0.0f64; 3];
            for jj in 0..3 {
                for kk in 0..3 {
                    vec[jj] += (s.Cell[icell][kk] as f64 + s.tau[isite][kk]) * s.direct[kk][jj];
                }
            }
            t.push_str(&format!(
                "H {} {} {}\n",
                cfmt::f(vec[0], 15, 5),
                cfmt::f(vec[1], 15, 5),
                cfmt::f(vec[2], 15, 5)
            ));
        }
    }
    o.write_file("lattice.xsf", &t)
}

const J_SUFFIX: [[&str; 3]; 3] = [["x", "xy", "xz"], ["yx", "y", "yz"], ["zx", "zy", "z"]];

/// `StdFace_InputSpinNN`: nearest-neighbour spin coupling input and defaults.
pub fn input_spin_nn(
    o: &mut Out,
    j: &[[f64; 3]; 3],
    j_all: f64,
    j0: &mut [[f64; 3]; 3],
    j0_all: f64,
    j0name: &str,
) -> Res<()> {
    if !j_all.is_nan() && !j0_all.is_nan() {
        outf!(o, "\n ERROR! {j0name} conflict !\n\n");
        return exit(-1);
    }
    for i1 in 0..3 {
        for i2 in 0..3 {
            let jn = J_SUFFIX[i1][i2];
            if !j_all.is_nan() && !j[i1][i2].is_nan() {
                outf!(o, "\n ERROR! J{jn} conflict !\n\n");
                return exit(-1);
            } else if !j0_all.is_nan() && !j[i1][i2].is_nan() {
                outf!(o, "\n ERROR! {j0name} and J{jn} conflict !\n\n");
                return exit(-1);
            } else if !j0_all.is_nan() && !j0[i1][i2].is_nan() {
                outf!(o, "\n ERROR! {j0name} and {j0name}{jn} conflict !\n\n");
                return exit(-1);
            } else if !j0[i1][i2].is_nan() && !j_all.is_nan() {
                outf!(o, "\n ERROR! {j0name}{jn} conflict !\n\n");
                return exit(-1);
            }
        }
    }
    for i1 in 0..3 {
        for i2 in 0..3 {
            for i3 in 0..3 {
                for i4 in 0..3 {
                    if !j0[i1][i2].is_nan() && !j[i3][i4].is_nan() {
                        outf!(
                            o,
                            "\n ERROR! {j0name}{} and J{} conflict !\n\n",
                            J_SUFFIX[i1][i2],
                            J_SUFFIX[i3][i4]
                        );
                        return exit(-1);
                    }
                }
            }
        }
    }
    for i1 in 0..3 {
        for i2 in 0..3 {
            let jn = J_SUFFIX[i1][i2];
            if !j0[i1][i2].is_nan() {
                outf!(
                    o,
                    "  {}{jn} = {}\n",
                    cfmt::s(j0name, 14),
                    cfmt::f_left(j0[i1][i2], 10, 5)
                );
            } else if !j[i1][i2].is_nan() {
                j0[i1][i2] = j[i1][i2];
                outf!(
                    o,
                    "  {}{jn} = {}\n",
                    cfmt::s(j0name, 14),
                    cfmt::f_left(j0[i1][i2], 10, 5)
                );
            } else if i1 == i2 && !j0_all.is_nan() {
                j0[i1][i2] = j0_all;
                outf!(
                    o,
                    "  {}{jn} = {}\n",
                    cfmt::s(j0name, 14),
                    cfmt::f_left(j0[i1][i2], 10, 5)
                );
            } else if i1 == i2 && !j_all.is_nan() {
                j0[i1][i2] = j_all;
                outf!(
                    o,
                    "  {}{jn} = {}\n",
                    cfmt::s(j0name, 14),
                    cfmt::f_left(j0[i1][i2], 10, 5)
                );
            } else {
                j0[i1][i2] = 0.0;
            }
        }
    }
    Ok(())
}

/// `StdFace_InputSpin`: spin coupling beyond nearest neighbour.
pub fn input_spin(o: &mut Out, jp: &mut [[f64; 3]; 3], jp_all: f64, jpname: &str) -> Res<()> {
    for i1 in 0..3 {
        for i2 in 0..3 {
            if !jp_all.is_nan() && !jp[i1][i2].is_nan() {
                outf!(
                    o,
                    "\n ERROR! {jpname} and {jpname}{} conflict !\n\n",
                    J_SUFFIX[i1][i2]
                );
                return exit(-1);
            }
        }
    }
    for i1 in 0..3 {
        for i2 in 0..3 {
            let jn = J_SUFFIX[i1][i2];
            if !jp[i1][i2].is_nan() {
                outf!(
                    o,
                    "  {}{jn} = {}\n",
                    cfmt::s(jpname, 14),
                    cfmt::f_left(jp[i1][i2], 10, 5)
                );
            } else if i1 == i2 && !jp_all.is_nan() {
                jp[i1][i2] = jp_all;
                outf!(
                    o,
                    "  {}{jn} = {}\n",
                    cfmt::s(jpname, 14),
                    cfmt::f_left(jp[i1][i2], 10, 5)
                );
            } else {
                jp[i1][i2] = 0.0;
            }
        }
    }
    Ok(())
}

/// `StdFace_InputCoulombV`: off-site Coulomb input and defaults.
pub fn input_coulomb_v(o: &mut Out, v: f64, v0: &mut f64, v0name: &str) -> Res<()> {
    if !v.is_nan() && !v0.is_nan() {
        outf!(o, "\n ERROR! {v0name} conflicts !\n\n");
        return exit(-1);
    } else if !v0.is_nan() {
        outf!(
            o,
            "  {} = {}\n",
            cfmt::s(v0name, 15),
            cfmt::f_left(*v0, 10, 5)
        );
    } else if !v.is_nan() {
        *v0 = v;
        outf!(
            o,
            "  {} = {}\n",
            cfmt::s(v0name, 15),
            cfmt::f_left(*v0, 10, 5)
        );
    } else {
        *v0 = 0.0;
    }
    Ok(())
}

/// `StdFace_InputHopp`: hopping integral input and defaults.
pub fn input_hopp(o: &mut Out, t: C64, t0: &mut C64, t0name: &str) -> Res<()> {
    if !t.re.is_nan() && !t0.re.is_nan() {
        outf!(o, "\n ERROR! {t0name} conflicts !\n\n");
        return exit(-1);
    } else if !t0.re.is_nan() {
        outf!(
            o,
            "  {} = {} {}\n",
            cfmt::s(t0name, 15),
            cfmt::f_left(t0.re, 10, 5),
            cfmt::f_left(t0.im, 10, 5)
        );
    } else if !t.re.is_nan() {
        *t0 = t;
        outf!(
            o,
            "  {} = {} {}\n",
            cfmt::s(t0name, 15),
            cfmt::f_left(t0.re, 10, 5),
            cfmt::f_left(t0.im, 10, 5)
        );
    } else {
        *t0 = C64::real(0.0);
    }
    Ok(())
}

/// `StdFace_PrintGeometry`: `geometry.dat`.
pub fn print_geometry(o: &mut Out, s: &StdIntList) -> Res<()> {
    let mut t = String::new();
    for ii in 0..3 {
        t.push_str(&format!(
            "{} {} {}\n",
            cfmt::e(s.direct[ii][0], 25, 15),
            cfmt::e(s.direct[ii][1], 25, 15),
            cfmt::e(s.direct[ii][2], 25, 15)
        ));
    }
    t.push_str(&format!(
        "{} {} {}\n",
        cfmt::e(s.phase[0], 25, 15),
        cfmt::e(s.phase[1], 25, 15),
        cfmt::e(s.phase[2], 25, 15)
    ));
    for ii in 0..3 {
        t.push_str(&format!(
            "{} {} {}\n",
            s.box_[ii][0], s.box_[ii][1], s.box_[ii][2]
        ));
    }
    let n_uc = s.NsiteUC;
    let n_blocks = if s.model == "kondo" { 2 } else { 1 };
    for block in 0..n_blocks {
        for cell in &s.Cell {
            for isite in 0..n_uc {
                t.push_str(&format!(
                    "{} {} {} {}\n",
                    cell[0] - s.Cell[0][0],
                    cell[1] - s.Cell[0][1],
                    cell[2] - s.Cell[0][2],
                    isite + block * n_uc
                ));
            }
        }
    }
    o.write_file("geometry.dat", &t)
}

/// `StdFace_MallocInteractions`: the C code only reserves arrays; here the lists restart empty.
pub fn malloc_interactions(s: &mut StdIntList, ntrans_max: i32, nintr_max: i32) {
    let (nt, ni) = (ntrans_max.max(0) as usize, nintr_max.max(0) as usize);
    s.transindx = Vec::with_capacity(nt);
    s.trans = Vec::with_capacity(nt);
    s.intrindx = Vec::with_capacity(ni);
    s.intr = Vec::with_capacity(ni);
    s.CintraIndx = Vec::with_capacity(ni);
    s.Cintra = Vec::with_capacity(ni);
    s.CinterIndx = Vec::with_capacity(ni);
    s.Cinter = Vec::with_capacity(ni);
    s.HundIndx = Vec::with_capacity(ni);
    s.Hund = Vec::with_capacity(ni);
    s.ExIndx = Vec::with_capacity(ni);
    s.Ex = Vec::with_capacity(ni);
    s.PLIndx = Vec::with_capacity(ni);
    s.PairLift = Vec::with_capacity(ni);
    s.PHIndx = Vec::with_capacity(ni);
    s.PairHopp = Vec::with_capacity(ni);
}

/// `StdFace_FoldSiteSub`: fold into the sub-lattice; returns `(nBox, iCellV_fold)`.
fn fold_site_sub(s: &StdIntList, icell_v: [i32; 3]) -> ([i32; 3], [i32; 3]) {
    let mut frac = [0i32; 3];
    for ii in 0..3 {
        for jj in 0..3 {
            frac[ii] += s.rboxsub[ii][jj] * icell_v[jj];
        }
    }
    let mut n_box = [0i32; 3];
    for ii in 0..3 {
        n_box[ii] = (frac[ii] + s.NCellsub * 1000) / s.NCellsub - 1000;
    }
    for ii in 0..3 {
        frac[ii] -= s.NCellsub * n_box[ii];
    }
    let mut fold = [0i32; 3];
    for ii in 0..3 {
        for jj in 0..3 {
            fold[ii] += s.boxsub[jj][ii] * frac[jj];
        }
        fold[ii] = (fold[ii] + s.NCellsub * 1000) / s.NCellsub - 1000;
    }
    (n_box, fold)
}

/// `StdFace_Proj`: quantum-number projection, `qptransidx.def`.
pub fn proj(o: &mut Out, s: &mut StdIntList) -> Res<()> {
    let nsite = s.nsite as usize;
    let mut sym = vec![vec![0i32; nsite]; nsite];
    let mut anti = vec![vec![0i32; nsite]; nsite];
    let n_uc = s.NsiteUC;
    s.NSym = 0;
    for icell in 0..s.NCell as usize {
        let (_, icell_v) = fold_site_sub(s, s.Cell[icell]);
        let (_, icell_v) = fold_site(s, icell_v);
        if icell_v == s.Cell[icell] {
            let nsym = s.NSym as usize;
            for jcell in 0..s.NCell as usize {
                let jcell_v0 = [
                    s.Cell[jcell][0] + icell_v[0],
                    s.Cell[jcell][1] + icell_v[1],
                    s.Cell[jcell][2] + icell_v[2],
                ];
                let (n_box, jcell_v) = fold_site(s, jcell_v0);
                for kcell in 0..s.NCell as usize {
                    if jcell_v == s.Cell[kcell] {
                        for jsite in 0..n_uc {
                            let (jc, kc) = (jcell as i32, kcell as i32);
                            let phase = s.AntiPeriod[0] * n_box[0]
                                + s.AntiPeriod[1] * n_box[1]
                                + s.AntiPeriod[2] * n_box[2];
                            sym[nsym][(jc * n_uc + jsite) as usize] = kc * n_uc + jsite;
                            anti[nsym][(jc * n_uc + jsite) as usize] = phase;
                            if s.model == "kondo" {
                                let half = s.nsite / 2;
                                sym[nsym][(half + jc * n_uc + jsite) as usize] =
                                    half + kc * n_uc + jsite;
                                anti[nsym][(half + jc * n_uc + jsite) as usize] = phase;
                            }
                        }
                    }
                }
            }
            s.NSym += 1;
        }
    }
    let mut t = String::new();
    t.push_str("=============================================\n");
    t.push_str(&format!("NQPTrans {}\n", cfmt::d(s.NSym, 10)));
    t.push_str("=============================================\n");
    t.push_str("======== TrIdx_TrWeight_and_TrIdx_i_xi ======\n");
    t.push_str("=============================================\n");
    for isym in 0..s.NSym {
        t.push_str(&format!("{} {}\n", isym, cfmt::f(1.0, 10, 5)));
    }
    for isym in 0..s.NSym as usize {
        for jsite in 0..nsite {
            anti[isym][jsite] = if anti[isym][jsite] % 2 == 0 { 1 } else { -1 };
            t.push_str(&format!(
                "{}  {}  {}  {}\n",
                cfmt::d(isym as i32, 5),
                cfmt::d(jsite as i32, 5),
                cfmt::d(sym[isym][jsite], 5),
                cfmt::d(anti[isym][jsite], 5)
            ));
        }
    }
    o.write_file("qptransidx.def", &t)?;
    o.print("    qptransidx.def is written.\n");
    Ok(())
}

/// `StdFace_InitSiteSub`: sub-lattice shape and commensurability check.
fn init_site_sub(o: &mut Out, s: &mut StdIntList) -> Res<()> {
    let boxsub_set = s.boxsub.iter().flatten().any(|&b| b != NAN_I);
    let lwh_set = s.Lsub != NAN_I || s.Wsub != NAN_I || s.Hsub != NAN_I;
    if lwh_set && boxsub_set {
        o.print("\nERROR ! (Lsub, Wsub, Hsub) and (a0Wsub, ..., a2Hsub) conflict !\n\n");
        return exit(-1);
    } else if lwh_set {
        print_val_i(o, "Lsub", &mut s.Lsub, 1);
        print_val_i(o, "Wsub", &mut s.Wsub, 1);
        print_val_i(o, "Hsub", &mut s.Hsub, 1);
        s.boxsub = [[0; 3]; 3];
        s.boxsub[0][0] = s.Wsub;
        s.boxsub[1][1] = s.Lsub;
        s.boxsub[2][2] = s.Hsub;
    } else {
        let names = [
            ["a0Wsub", "a0Lsub", "a0Hsub"],
            ["a1Wsub", "a1Lsub", "a1Hsub"],
            ["a2Wsub", "a2Lsub", "a2Hsub"],
        ];
        for ii in 0..3 {
            for jj in 0..3 {
                print_val_i(o, names[ii][jj], &mut s.boxsub[ii][jj], s.box_[ii][jj]);
            }
        }
    }
    s.NCellsub = 0;
    for ii in 0..3 {
        s.NCellsub += s.boxsub[0][ii] * s.boxsub[1][(ii + 1) % 3] * s.boxsub[2][(ii + 2) % 3]
            - s.boxsub[0][ii] * s.boxsub[1][(ii + 2) % 3] * s.boxsub[2][(ii + 1) % 3];
    }
    outf!(
        o,
        "         Number of Cell in the sublattice: {}\n",
        s.NCellsub.abs()
    );
    if s.NCellsub == 0 {
        return exit(-1);
    }
    for ii in 0..3 {
        for jj in 0..3 {
            s.rboxsub[ii][jj] = s.boxsub[(ii + 1) % 3][(jj + 1) % 3]
                * s.boxsub[(ii + 2) % 3][(jj + 2) % 3]
                - s.boxsub[(ii + 1) % 3][(jj + 2) % 3] * s.boxsub[(ii + 2) % 3][(jj + 1) % 3];
        }
    }
    if s.NCellsub < 0 {
        for row in s.rboxsub.iter_mut() {
            for v in row.iter_mut() {
                *v = -*v;
            }
        }
        s.NCellsub = -s.NCellsub;
    }
    for ii in 0..3 {
        for jj in 0..3 {
            let mut prod = 0i32;
            for kk in 0..3 {
                prod += s.rboxsub[ii][kk] * s.box_[jj][kk];
            }
            if prod % s.NCellsub != 0 {
                o.print("\n ERROR ! Sublattice is INCOMMENSURATE !\n\n");
                return exit(-1);
            }
        }
    }
    Ok(())
}

/// `StdFace_generate_orb`: orbital index and anti-periodic switch.
pub fn generate_orb(o: &mut Out, s: &mut StdIntList) -> Res<()> {
    init_site_sub(o, s)?;
    let nsite = s.nsite as usize;
    s.Orb = vec![vec![0; nsite]; nsite];
    s.AntiOrb = vec![vec![0; nsite]; nsite];
    let ncell = s.NCell as usize;
    let mut cell_done = vec![vec![false; ncell]; ncell];
    let n_uc = s.NsiteUC;
    let half = s.nsite / 2;
    let kondo = s.model == "kondo";
    let ix = |a: i32| a as usize;
    let mut iorb = 0i32;
    for icell in 0..ncell {
        let (_, icell_v) = fold_site_sub(s, s.Cell[icell]);
        let (_, icell_v) = fold_site(s, icell_v);
        let mut icell2 = 0usize;
        for (kcell, cell) in s.Cell.iter().enumerate() {
            if icell_v == *cell {
                icell2 = kcell;
            }
        }
        for jcell in 0..ncell {
            let jcell_v0 = [
                s.Cell[jcell][0] + icell_v[0] - s.Cell[icell][0],
                s.Cell[jcell][1] + icell_v[1] - s.Cell[icell][1],
                s.Cell[jcell][2] + icell_v[2] - s.Cell[icell][2],
            ];
            let (_, jcell_v) = fold_site(s, jcell_v0);
            let mut jcell2 = 0usize;
            for (kcell, cell) in s.Cell.iter().enumerate() {
                if jcell_v == *cell {
                    jcell2 = kcell;
                }
            }
            let dcell_v = [
                s.Cell[jcell][0] - s.Cell[icell][0],
                s.Cell[jcell][1] - s.Cell[icell][1],
                s.Cell[jcell][2] - s.Cell[icell][2],
            ];
            let (n_box, _) = fold_site(s, dcell_v);
            let mut anti = 0;
            for ii in 0..3 {
                anti += s.AntiPeriod[ii] * n_box[ii];
            }
            let anti = if anti % 2 == 0 { 1 } else { -1 };
            let (ic, jc) = (icell as i32, jcell as i32);
            let (ic2, jc2) = (icell2 as i32, jcell2 as i32);
            for isite in 0..n_uc {
                for jsite in 0..n_uc {
                    let (a2, b2) = (ix(ic2 * n_uc + isite), ix(jc2 * n_uc + jsite));
                    let (a, b) = (ix(ic * n_uc + isite), ix(jc * n_uc + jsite));
                    if !cell_done[icell2][jcell2] {
                        s.Orb[a2][b2] = iorb;
                        s.AntiOrb[a2][b2] = anti;
                        iorb += 1;
                    }
                    s.Orb[a][b] = s.Orb[a2][b2];
                    s.AntiOrb[a][b] = anti;
                    if kondo {
                        let (ah2, ah) =
                            (ix(half + ic2 * n_uc + isite), ix(half + ic * n_uc + isite));
                        let (bh2, bh) =
                            (ix(half + jc2 * n_uc + jsite), ix(half + jc * n_uc + jsite));
                        if !cell_done[icell2][jcell2] {
                            s.Orb[ah2][b2] = iorb;
                            s.AntiOrb[ah2][b2] = anti;
                            iorb += 1;
                            s.Orb[a2][bh2] = iorb;
                            s.AntiOrb[a2][bh2] = anti;
                            iorb += 1;
                            s.Orb[ah2][bh2] = iorb;
                            s.AntiOrb[ah2][bh2] = anti;
                            iorb += 1;
                        }
                        s.Orb[ah][b] = s.Orb[ah2][b2];
                        s.AntiOrb[ah][b] = anti;
                        s.Orb[a][bh] = s.Orb[a2][bh2];
                        s.AntiOrb[a][bh] = anti;
                        s.Orb[ah][bh] = s.Orb[ah2][bh2];
                        s.AntiOrb[ah][bh] = anti;
                    }
                }
            }
            cell_done[icell2][jcell2] = true;
        }
    }
    s.NOrb = iorb;
    Ok(())
}

/// `PrintJastrow`: `jastrowidx.def`.
pub fn print_jastrow(o: &mut Out, s: &mut StdIntList) -> Res<()> {
    let nsite = s.nsite as usize;
    let mut jastrow = vec![vec![0i32; nsite]; nsite];
    let n_jastrow: i32;
    if s.NMPTrans.abs() == 1 || s.NMPTrans == NAN_I {
        for isite in 0..nsite {
            for jsite in 0..nsite {
                jastrow[isite][jsite] = s.Orb[isite][jsite];
            }
        }
        for iorb in 0..s.NOrb {
            for isite in 0..nsite {
                for jsite in 0..nsite {
                    if jastrow[isite][jsite] == iorb {
                        jastrow[jsite][isite] = jastrow[isite][jsite];
                    }
                }
            }
        }
        let mut nj = if s.model == "hubbard" { 0 } else { -1 };
        for isite in 0..nsite {
            if s.locspinflag[isite] != 0 {
                for jsite in 0..nsite {
                    jastrow[isite][jsite] = -1;
                    jastrow[jsite][isite] = -1;
                }
                continue;
            }
            for jsite in 0..isite {
                if jastrow[isite][jsite] >= 0 {
                    let ijastrow = jastrow[isite][jsite];
                    nj -= 1;
                    for row in jastrow.iter_mut() {
                        for v in row.iter_mut() {
                            if *v == ijastrow {
                                *v = nj;
                            }
                        }
                    }
                }
            }
        }
        n_jastrow = -nj;
        for row in jastrow.iter_mut() {
            for v in row.iter_mut() {
                *v = -1 - *v;
            }
        }
    } else {
        let mut nj;
        if s.model == "spin" {
            nj = 1;
            for row in jastrow.iter_mut() {
                for v in row.iter_mut() {
                    *v = 0;
                }
            }
        } else {
            nj = 0;
            if s.model == "kondo" {
                for isite in 0..nsite {
                    for jsite in 0..nsite / 2 {
                        jastrow[isite][jsite] = 0;
                        jastrow[jsite][isite] = 0;
                    }
                }
                nj += 1;
            }
            for dcell in 0..s.NCell as usize {
                let dc = s.Cell[dcell];
                let pair = find_site(s, 0, 0, 0, -dc[0], -dc[1], -dc[2], 0, 0);
                let mut jsite = pair.jsite;
                if s.model == "kondo" {
                    jsite += -s.NCell * s.NsiteUC;
                }
                let icell = jsite / s.NsiteUC;
                if icell < dcell as i32 {
                    // If -R has been already done, skip.
                    continue;
                }
                // Reversal symmetry [Fold(-R) = R]: J(R,i,j) = J(R,j,i).
                let revarsal = icell == dcell as i32;
                for isite_uc in 0..s.NsiteUC {
                    for jsite_uc in 0..s.NsiteUC {
                        if revarsal && jsite_uc > isite_uc {
                            continue;
                        }
                        if isite_uc == jsite_uc && dc == [0, 0, 0] {
                            continue;
                        }
                        for icell in 0..s.NCell as usize {
                            let ic = s.Cell[icell];
                            let pair = find_site(
                                s, ic[0], ic[1], ic[2], dc[0], dc[1], dc[2], isite_uc, jsite_uc,
                            );
                            jastrow[pair.isite as usize][pair.jsite as usize] = nj;
                            jastrow[pair.jsite as usize][pair.isite as usize] = nj;
                        }
                        nj += 1;
                    }
                }
            }
        }
        n_jastrow = nj;
    }
    let mut t = String::new();
    t.push_str("=============================================\n");
    t.push_str(&format!("NJastrowIdx {}\n", cfmt::d(n_jastrow, 10)));
    t.push_str(&format!("ComplexType {}\n", cfmt::d(0, 10)));
    t.push_str("=============================================\n");
    t.push_str("=============================================\n");
    for isite in 0..nsite {
        for jsite in 0..nsite {
            if isite == jsite {
                continue;
            }
            t.push_str(&format!(
                "{}  {}  {}\n",
                cfmt::d(isite as i32, 5),
                cfmt::d(jsite as i32, 5),
                cfmt::d(jastrow[isite][jsite], 5)
            ));
        }
    }
    for ijastrow in 0..n_jastrow {
        let flag = if s.model == "hubbard" || ijastrow > 0 {
            1
        } else {
            0
        };
        t.push_str(&format!("{}  {}\n", cfmt::d(ijastrow, 5), cfmt::d(flag, 5)));
    }
    o.write_file("jastrowidx.def", &t)?;
    o.print("    jastrowidx.def is written.\n");
    Ok(())
}

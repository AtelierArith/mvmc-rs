//! Port of `Wannier90.c` (`StdFace_Wannier90`, mVMC branch): a lattice defined by Wannier90
//! (`*_geom.dat`) with hopping, Coulomb and Hund integrals read from `*_hr.dat`, `*_ur.dat`,
//! `*_jr.dat` (and the density matrix `*_dr.dat` for double-counting corrections).
//!
//! `export_wannier90.c` is HWAVE-only and not ported. The data files are read from
//! [`Out::data_dir`] (C: the current directory). Corrections of C undefined behaviour are marked
//! with `C defect` comments and listed in `tests/fixtures/stdface/README.md`.
#![allow(non_snake_case)]

use crate::ccomplex::C64;
use crate::cfmt;
use crate::model_util as mu;
use crate::out::{exit, Out, Res};
use crate::outf;
use crate::scanner::{Scanner, EOF};
use crate::vals::{StdIntList, NAN_I};

#[derive(Clone, Copy, PartialEq, Eq)]
enum DcMode {
    NotCorrect,
    Hartree,
    HartreeU,
    Full,
}

/// `_calc_inverse_matrix`.
fn calc_inverse_matrix(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut det = m[0][0] * m[1][1] * m[2][2];
    det += m[1][0] * m[2][1] * m[0][2];
    det += m[2][0] * m[0][1] * m[1][2];
    det -= m[2][0] * m[1][1] * m[0][2];
    det -= m[1][0] * m[0][1] * m[2][2];
    det -= m[0][0] * m[2][1] * m[1][2];
    let mut inv = [[0.0f64; 3]; 3];
    inv[0][0] = m[1][1] * m[2][2] - m[1][2] * m[2][1];
    inv[0][1] = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]);
    inv[0][2] = m[0][1] * m[1][2] - m[0][2] * m[1][1];
    inv[1][0] = -(m[1][0] * m[2][2] - m[2][0] * m[1][2]);
    inv[1][1] = m[0][0] * m[2][2] - m[0][2] * m[2][0];
    inv[1][2] = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]);
    inv[2][0] = m[1][0] * m[2][1] - m[2][0] * m[1][1];
    inv[2][1] = -(m[0][0] * m[2][1] - m[2][0] * m[0][1]);
    inv[2][2] = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    for row in inv.iter_mut() {
        for v in row.iter_mut() {
            *v /= det;
        }
    }
    inv
}

/// `_check_in_box`.
fn check_in_box(rvec: &[i32; 3], inv: &[[f64; 3]; 3]) -> bool {
    let mut judge = [0.0f64; 3];
    for i in 0..3 {
        for j in 0..3 {
            judge[i] += rvec[j] as f64 * inv[j][i];
        }
    }
    judge[0].abs() <= 1.0 && judge[1].abs() <= 1.0 && judge[2].abs() <= 1.0
}

fn tau_at(s: &StdIntList, wan: usize, k: usize) -> f64 {
    // C defect: indexes `tau` with the band index of the file, which is out of bounds when the
    // file has more bands than Wannier centres; such terms read 0 here.
    s.tau.get(wan).map_or(0.0, |t| t[k])
}

/// `geometry_W90`: `<CDataFileHead>_geom.dat`.
fn geometry_w90(o: &mut Out, s: &mut StdIntList) -> Res<()> {
    let filename = format!("{}_geom.dat", s.CDataFileHead);
    outf!(o, "    Wannier90 Geometry file = {filename}\n");
    let Ok(data) = std::fs::read(o.data_dir.join(&filename)) else {
        o.eprint(&format!(
            "\n  Error: Fail to open the file {filename}. \n\n"
        ));
        return exit(-1);
    };
    let mut sc = Scanner::new(&data);
    let mut ierr = 0;
    for ii in 0..3 {
        let mut row = [s.direct[ii][0], s.direct[ii][1], s.direct[ii][2]];
        ierr = sc.scan_doubles(&mut row);
        s.direct[ii] = row;
    }
    if ierr == EOF {
        o.print(&format!("{ierr}\n"));
    }
    let mut nsite_uc = s.NsiteUC;
    sc.scan_int(&mut nsite_uc);
    s.NsiteUC = nsite_uc;
    outf!(o, "    Number of Correlated Sites = {}\n", s.NsiteUC);
    s.tau = vec![[0.0; 3]; s.NsiteUC.max(0) as usize];
    for isite in 0..s.tau.len() {
        let mut row = [0.0; 3];
        sc.scan_doubles(&mut row);
        s.tau[isite] = row;
    }
    o.print("    Direct lattice vectors:\n");
    for ii in 0..3 {
        outf!(
            o,
            "      {} {} {}\n",
            cfmt::f(s.direct[ii][0], 10, 5),
            cfmt::f(s.direct[ii][1], 10, 5),
            cfmt::f(s.direct[ii][2], 10, 5)
        );
    }
    o.print("    Wannier centres:\n");
    for t in &s.tau {
        outf!(
            o,
            "      {} {} {}\n",
            cfmt::f(t[0], 10, 5),
            cfmt::f(t[1], 10, 5),
            cfmt::f(t[2], 10, 5)
        );
    }
    Ok(())
}

/// Matrix elements `(R0, R1, R2, band_i, band_f)` and their values.
type Terms = (Vec<[i32; 5]>, Vec<C64>);

/// `read_W90`: read a `*_hr`-style file and keep the terms above `cutoff`.
#[allow(clippy::too_many_arguments)]
fn read_w90(
    o: &mut Out,
    s: &StdIntList,
    filename: &str,
    cutoff: f64,
    cutoff_r: &[i32; 3],
    cutoff_rvec: &[[f64; 3]; 3],
    cutoff_length: f64,
    lambda: f64,
) -> Terms {
    let Ok(data) = std::fs::read(o.data_dir.join(filename)) else {
        outf!(o, "\n  Skip to read the file {filename}. \n\n");
        return (Vec::new(), Vec::new());
    };
    let nsite_uc = s.NsiteUC.max(0) as usize;
    let flg_vec = cutoff_rvec[0][0] != NAN_I as f64;
    let mut sc = Scanner::new(&data);
    sc.fgets();
    let (mut n_wan, mut n_wsc) = (0i32, 0i32);
    sc.scan_int(&mut n_wan);
    sc.scan_int(&mut n_wsc);
    let (n_wan, n_wsc) = (n_wan.max(0) as usize, n_wsc.max(0) as usize);
    for _ in 0..n_wsc {
        let mut ii = 0;
        sc.scan_int(&mut ii);
    }
    let mut mat_tot =
        vec![vec![vec![C64::real(0.0); n_wan.max(nsite_uc)]; n_wan.max(nsite_uc)]; n_wsc];
    let mut indx_tot = vec![[0i32; 3]; n_wsc];
    let inverse_rvec = if flg_vec {
        calc_inverse_matrix(cutoff_rvec)
    } else {
        [[0.0; 3]; 3]
    };

    let (mut iwan0, mut jwan0) = (0i32, 0i32);
    let mut dtmp = [0.0f64; 2];
    for iwsc in 0..n_wsc {
        for iwan in 0..n_wan {
            for jwan in 0..n_wan {
                let mut ints = [
                    indx_tot[iwsc][0],
                    indx_tot[iwsc][1],
                    indx_tot[iwsc][2],
                    iwan0,
                    jwan0,
                ];
                sc.scan_row(&mut ints, &mut dtmp);
                indx_tot[iwsc] = [ints[0], ints[1], ints[2]];
                iwan0 = ints[3];
                jwan0 = ints[4];
                // Euclid length
                let mut d_r = [0.0f64; 3];
                for ii in 0..3 {
                    for jj in 0..3 {
                        d_r[ii] += s.direct[jj][ii]
                            * (tau_at(s, jwan, jj) - tau_at(s, iwan, jj)
                                + indx_tot[iwsc][jj] as f64);
                    }
                }
                let length = (d_r[0] * d_r[0] + d_r[1] * d_r[1] + d_r[2] * d_r[2]).sqrt();
                if length > cutoff_length && cutoff_length > 0.0 {
                    dtmp = [0.0, 0.0];
                }
                if flg_vec {
                    if !check_in_box(&indx_tot[iwsc], &inverse_rvec) {
                        dtmp = [0.0, 0.0];
                    }
                } else if indx_tot[iwsc][0].abs() > cutoff_r[0]
                    || indx_tot[iwsc][1].abs() > cutoff_r[1]
                    || indx_tot[iwsc][2].abs() > cutoff_r[2]
                {
                    dtmp = [0.0, 0.0];
                }
                if iwan0 <= s.NsiteUC && jwan0 <= s.NsiteUC && iwan0 >= 1 && jwan0 >= 1 {
                    // lambda * (dtmp[0] + I * dtmp[1])
                    mat_tot[iwsc][iwan0 as usize - 1][jwan0 as usize - 1] =
                        C64::real_plus(dtmp[0], C64::I.scale(dtmp[1])).scale(lambda);
                }
            }
        }
        // (1) Apply inversion symmetry and delete duplication
        for jwsc in 0..iwsc {
            if indx_tot[iwsc][0] == -indx_tot[jwsc][0]
                && indx_tot[iwsc][1] == -indx_tot[jwsc][1]
                && indx_tot[iwsc][2] == -indx_tot[jwsc][2]
            {
                for iwan in 0..nsite_uc {
                    for jwan in 0..nsite_uc {
                        mat_tot[iwsc][iwan][jwan] = C64::real(0.0);
                    }
                }
            }
        }
        if indx_tot[iwsc] == [0, 0, 0] {
            for iwan in 0..nsite_uc {
                for jwan in 0..iwan {
                    mat_tot[iwsc][iwan][jwan] = C64::real(0.0);
                }
            }
        }
    }

    // (2) Apply weight
    let mut band_lattice = [0i32; 3];
    for idx in &indx_tot {
        for ii in 0..3 {
            if idx[ii].abs() > band_lattice[ii] {
                band_lattice[ii] = idx[ii].abs();
            }
        }
    }
    let mut weight_tot = vec![1.0f64; n_wsc];
    if s.W != NAN_I && s.L != NAN_I && s.Height != NAN_I {
        let model_lattice = [
            if s.W % 2 == 0 { s.W / 2 } else { 0 },
            if s.L % 2 == 0 { s.L / 2 } else { 0 },
            if s.Height % 2 == 0 { s.Height / 2 } else { 0 },
        ];
        for ii in 0..3 {
            if model_lattice[ii] < band_lattice[ii] && model_lattice[ii] != 0 {
                for iwsc in 0..n_wsc {
                    if indx_tot[iwsc][ii].abs() == model_lattice[ii] {
                        weight_tot[iwsc] *= 0.5;
                    }
                }
            }
        }
    }

    // (3-1) Terms larger than the cut-off
    o.print("\n      EFFECTIVE terms:\n");
    o.print("           R0   R1   R2 band_i band_f Hamiltonian\n");
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for iwsc in 0..n_wsc {
        for iwan in 0..nsite_uc {
            for jwan in 0..nsite_uc {
                mat_tot[iwsc][iwan][jwan] = mat_tot[iwsc][iwan][jwan].scale(weight_tot[iwsc]);
                if cutoff < mat_tot[iwsc][iwan][jwan].abs() {
                    outf!(
                        o,
                        "        {}{}{}{}{}{}{}\n",
                        cfmt::d(indx_tot[iwsc][0], 5),
                        cfmt::d(indx_tot[iwsc][1], 5),
                        cfmt::d(indx_tot[iwsc][2], 5),
                        cfmt::d(iwan as i32, 5),
                        cfmt::d(jwan as i32, 5),
                        cfmt::f(mat_tot[iwsc][iwan][jwan].re, 12, 6),
                        cfmt::f(mat_tot[iwsc][iwan][jwan].im, 12, 6)
                    );
                    indices.push([
                        indx_tot[iwsc][0],
                        indx_tot[iwsc][1],
                        indx_tot[iwsc][2],
                        iwan as i32,
                        jwan as i32,
                    ]);
                    values.push(mat_tot[iwsc][iwan][jwan]);
                }
            }
        }
    }
    outf!(
        o,
        "      Total number of EFFECTIVE term = {}\n",
        values.len()
    );
    (indices, values)
}

/// Density matrix `D[R][i][j]` on the box `Rmin..=Rmax` (`read_density_matrix`).
struct DenMat {
    rmin: [i32; 3],
    rmax: [i32; 3],
    n: usize,
    data: Vec<C64>,
}

impl DenMat {
    fn slot(&self, r: [i32; 3], i: usize, j: usize) -> Option<usize> {
        for ii in 0..3 {
            if r[ii] < self.rmin[ii] || r[ii] > self.rmax[ii] {
                return None;
            }
        }
        if i >= self.n || j >= self.n {
            return None;
        }
        let nr = |k: usize| (self.rmax[k] - self.rmin[k] + 1) as usize;
        let ix = |k: usize| (r[k] - self.rmin[k]) as usize;
        Some((((ix(0) * nr(1) + ix(1)) * nr(2) + ix(2)) * self.n + i) * self.n + j)
    }

    /// `D[R][i][j]`; an index outside the stored range is out of bounds in C, here 0.
    fn get(&self, r: [i32; 3], i: i32, j: i32) -> C64 {
        match self.slot(r, i as usize, j as usize) {
            Some(k) => self.data[k],
            None => C64::real(0.0),
        }
    }
}

fn read_density_matrix(o: &mut Out, s: &StdIntList, filename: &str) -> Res<DenMat> {
    let Ok(data) = std::fs::read(o.data_dir.join(filename)) else {
        o.eprint(&format!(
            "\n  Error: Fail to open the file {filename}. \n\n"
        ));
        return exit(-1);
    };
    let nsite_uc = s.NsiteUC.max(0) as usize;
    let mut sc = Scanner::new(&data);
    sc.fgets();
    let (mut n_wan, mut n_wsc) = (0i32, 0i32);
    let ierr = sc.scan_int(&mut n_wan);
    if ierr == EOF {
        o.print(&format!("{ierr} \n"));
    }
    sc.scan_int(&mut n_wsc);
    let (n_wan, n_wsc) = (n_wan.max(0) as usize, n_wsc.max(0) as usize);
    for _ in 0..n_wsc {
        let mut ii = 0;
        sc.scan_int(&mut ii);
    }
    let dim = n_wan.max(nsite_uc);
    let mut mat_tot = vec![vec![vec![C64::real(0.0); dim]; dim]; n_wsc];
    let mut indx_tot = vec![[0i32; 3]; n_wsc];
    let (mut rmin, mut rmax) = ([0i32; 3], [0i32; 3]);
    let (mut iwan0, mut jwan0) = (0i32, 0i32);
    let mut dtmp = [0.0f64; 2];
    for iwsc in 0..n_wsc {
        for _iwan in 0..n_wan {
            for _jwan in 0..n_wan {
                let mut ints = [
                    indx_tot[iwsc][0],
                    indx_tot[iwsc][1],
                    indx_tot[iwsc][2],
                    iwan0,
                    jwan0,
                ];
                sc.scan_row(&mut ints, &mut dtmp);
                indx_tot[iwsc] = [ints[0], ints[1], ints[2]];
                iwan0 = ints[3];
                jwan0 = ints[4];
                if iwan0 <= s.NsiteUC && jwan0 <= s.NsiteUC && iwan0 >= 1 && jwan0 >= 1 {
                    mat_tot[iwsc][iwan0 as usize - 1][jwan0 as usize - 1] =
                        C64::real_plus(dtmp[0], C64::I.scale(dtmp[1]));
                }
                for ii in 0..3 {
                    if indx_tot[iwsc][ii] < rmin[ii] {
                        rmin[ii] = indx_tot[iwsc][ii];
                    }
                    if indx_tot[iwsc][ii] > rmax[ii] {
                        rmax[ii] = indx_tot[iwsc][ii];
                    }
                }
            }
        }
    }
    let nr = [
        rmax[0] - rmin[0] + 1,
        rmax[1] - rmin[1] + 1,
        rmax[2] - rmin[2] + 1,
    ];
    outf!(o, "      Minimum R : {} {} {}\n", rmin[0], rmin[1], rmin[2]);
    outf!(o, "      Maximum R : {} {} {}\n", rmax[0], rmax[1], rmax[2]);
    outf!(o, "      Numver of R : {} {} {}\n", nr[0], nr[1], nr[2]);
    let mut den = DenMat {
        rmin,
        rmax,
        n: nsite_uc,
        data: vec![C64::real(0.0); (nr[0] * nr[1] * nr[2]).max(0) as usize * nsite_uc * nsite_uc],
    };
    for iwsc in 0..n_wsc {
        // C defect: copies `nWan x nWan` into `NsiteUC x NsiteUC` arrays (heap overflow when the
        // file has more bands than Wannier centres); only the first NsiteUC bands are kept.
        for iwan in 0..nsite_uc.min(n_wan) {
            for jwan in 0..nsite_uc.min(n_wan) {
                if let Some(k) = den.slot(indx_tot[iwsc], iwan, jwan) {
                    den.data[k] = mat_tot[iwsc][iwan][jwan];
                }
            }
        }
    }
    Ok(den)
}

/// `PrintUHFinitial`: `initial.def`.
fn print_uhf_initial(
    o: &mut Out,
    s: &StdIntList,
    tuj_indx: &[Vec<[i32; 5]>; 3],
    den: &DenMat,
) -> Res<()> {
    let nsite = s.nsite as usize;
    // C defect: `IniGuess` is allocated with malloc and only partly assigned before it is
    // read; zero-initialised here.
    let mut ini = vec![vec![C64::real(0.0); nsite]; nsite];
    for kcell in 0..s.NCell as usize {
        let [iw, il, ih] = s.Cell[kcell];
        for isite in 0..s.NsiteUC {
            let jsite = (isite + s.NsiteUC * kcell as i32) as usize;
            ini[jsite][jsite] = den.get([0, 0, 0], isite, isite);
        }
        // Coulomb integral (U), then exchange integral (J)
        for family in [1usize, 2] {
            for t in &tuj_indx[family] {
                let pair = mu::find_site(s, iw, il, ih, t[0], t[1], t[2], t[3], t[4]);
                let d = den.get([t[0], t[1], t[2]], t[3], t[4]);
                ini[pair.isite as usize][pair.jsite as usize] = d;
                ini[pair.jsite as usize][pair.isite as usize] = d.conj();
            }
        }
    }
    let n_ini = ini.iter().flatten().filter(|c| c.abs() > 1.0e-6).count();
    let mut t = String::new();
    t.push_str("======================== \n");
    t.push_str(&format!(
        "NInitialGuess {}  \n",
        cfmt::d(n_ini as i32 * 2, 7)
    ));
    t.push_str("======================== \n");
    t.push_str("========i_j_s_tijs====== \n");
    t.push_str("======================== \n");
    for isite in 0..nsite {
        for jsite in 0..nsite {
            if ini[isite][jsite].abs() > 1.0e-6 {
                for ispin in 0..2 {
                    t.push_str(&format!(
                        "{} {} {} {} {} {}\n",
                        cfmt::d(jsite as i32, 5),
                        cfmt::d(ispin, 5),
                        cfmt::d(isite as i32, 5),
                        cfmt::d(ispin, 5),
                        cfmt::f(0.5 * ini[isite][jsite].re, 25, 15),
                        cfmt::f(0.5 * ini[isite][jsite].im, 25, 15)
                    ));
                }
            }
        }
    }
    o.write_file("initial.def", &t)?;
    o.print("      initial.def is written.\n");
    Ok(())
}

fn push_diag_trans(s: &mut StdIntList, value: C64, site: i32, ispin: i32) {
    s.trans.push(value);
    s.transindx.push([site, ispin, site, ispin]);
}

/// Setup a Hamiltonian for the Wannier90 `*_hr.dat` model.
pub fn std_face_wannier90(o: &mut Out, s: &mut StdIntList) -> Res<()> {
    // fp = fopen("lattice.xsf", "w"): created empty here, rewritten by StdFace_PrintXSF.
    o.write_file("lattice.xsf", "")?;
    mu::print_val_d(o, "phase0", &mut s.phase[0], 0.0);
    mu::print_val_d(o, "phase1", &mut s.phase[1], 0.0);
    mu::print_val_d(o, "phase2", &mut s.phase[2], 0.0);
    s.NsiteUC = 1;
    mu::init_site(o, s, &mut None, 3)?;
    o.print("\n  @ Wannier90 Geometry \n\n");
    geometry_w90(o, s)?;

    // Parameters that tune the strength of the interactions
    if s.lambda.is_nan() {
        mu::print_val_d(o, "lambda_U", &mut s.lambda_U, 1.0);
        mu::print_val_d(o, "lambda_J", &mut s.lambda_J, 1.0);
    } else {
        mu::print_val_d(o, "lambda_U", &mut s.lambda_U, s.lambda);
        mu::print_val_d(o, "lambda_J", &mut s.lambda_J, s.lambda);
    }
    if s.lambda_U < 0.0 || s.lambda_J < 0.0 {
        o.eprint(
            "\n  Error: the value of lambda_U / lambda_J must be greater than or equal to 0. \n\n",
        );
        return exit(-1);
    }
    let idcmode = match s.double_counting_mode.as_str() {
        "none" | "****" => DcMode::NotCorrect,
        "hartree" => DcMode::Hartree,
        "hartree_u" => DcMode::HartreeU,
        "full" => DcMode::Full,
        _ => {
            o.eprint(
                "\n  Error: the word of doublecounting is not correct (select from none, hartree, hartree_u, full). \n\n",
            );
            return exit(-1);
        }
    };
    mu::print_val_d(o, "alpha", &mut s.alpha, 0.5);
    if s.alpha > 1.0 || s.alpha < 0.0 {
        o.eprint("\n  Error: the value of alpha must be in the range 0<= alpha <= 1. \n\n");
        return exit(-1);
    }

    // Read Hopping
    o.print("\n  @ Wannier90 hopping \n\n");
    mu::print_val_d(o, "cutoff_t", &mut s.cutoff_t, 1.0e-8);
    mu::print_val_d(o, "cutoff_length_t", &mut s.cutoff_length_t, -1.0);
    if s.W != NAN_I {
        mu::print_val_i(o, "cutoff_tR[0]", &mut s.cutoff_tR[0], (s.W - 1) / 2);
    }
    if s.L != NAN_I {
        mu::print_val_i(o, "cutoff_tR[1]", &mut s.cutoff_tR[1], (s.L - 1) / 2);
    }
    if s.Height != NAN_I {
        mu::print_val_i(o, "cutoff_tR[2]", &mut s.cutoff_tR[2], (s.Height - 1) / 2);
    }
    for i in 0..3 {
        for j in 0..3 {
            let name = format!("cutoff_tVec[{i}][{j}]");
            mu::print_val_d(
                o,
                &name,
                &mut s.cutoff_tVec[i][j],
                s.box_[i][j] as f64 * 0.5,
            );
        }
    }
    let filename = format!("{}_hr.dat", s.CDataFileHead);
    let (t_indx, t_val) = read_w90(
        o,
        s,
        &filename,
        s.cutoff_t,
        &s.cutoff_tR,
        &s.cutoff_tVec,
        s.cutoff_length_t,
        1.0,
    );

    // Read Coulomb
    o.print("\n  @ Wannier90 Coulomb \n\n");
    mu::print_val_d(o, "cutoff_u", &mut s.cutoff_u, 1.0e-8);
    mu::print_val_d(o, "cutoff_length_U", &mut s.cutoff_length_U, 0.3);
    mu::print_val_i(o, "cutoff_UR[0]", &mut s.cutoff_UR[0], 0);
    mu::print_val_i(o, "cutoff_UR[1]", &mut s.cutoff_UR[1], 0);
    mu::print_val_i(o, "cutoff_UR[2]", &mut s.cutoff_UR[2], 0);
    for i in 0..3 {
        for j in 0..3 {
            let name = format!("cutoff_UVec[{i}][{j}]");
            mu::print_val_d(
                o,
                &name,
                &mut s.cutoff_UVec[i][j],
                s.box_[i][j] as f64 * 0.5,
            );
        }
    }
    let filename = format!("{}_ur.dat", s.CDataFileHead);
    let (u_indx, u_val) = read_w90(
        o,
        s,
        &filename,
        s.cutoff_u,
        &s.cutoff_UR,
        &s.cutoff_UVec,
        s.cutoff_length_U,
        s.lambda_U,
    );

    // Read Hund
    o.print("\n  @ Wannier90 Hund \n\n");
    mu::print_val_d(o, "cutoff_j", &mut s.cutoff_j, 1.0e-8);
    mu::print_val_d(o, "cutoff_length_J", &mut s.cutoff_length_J, 0.3);
    mu::print_val_i(o, "cutoff_JR[0]", &mut s.cutoff_JR[0], 0);
    mu::print_val_i(o, "cutoff_JR[1]", &mut s.cutoff_JR[1], 0);
    mu::print_val_i(o, "cutoff_JR[2]", &mut s.cutoff_JR[2], 0);
    for i in 0..3 {
        for j in 0..3 {
            let name = format!("cutoff_JVec[{i}][{j}]");
            mu::print_val_d(
                o,
                &name,
                &mut s.cutoff_JVec[i][j],
                s.box_[i][j] as f64 * 0.5,
            );
        }
    }
    let filename = format!("{}_jr.dat", s.CDataFileHead);
    let (j_indx, j_val) = read_w90(
        o,
        s,
        &filename,
        s.cutoff_j,
        &s.cutoff_JR,
        &s.cutoff_JVec,
        s.cutoff_length_J,
        s.lambda_J,
    );
    let tuj_indx = [t_indx, u_indx, j_indx];

    // Read Density matrix
    let mut den_mat = None;
    if idcmode != DcMode::NotCorrect {
        o.print("\n  @ Wannier90 Density-matrix \n\n");
        let filename = format!("{}_dr.dat", s.CDataFileHead);
        den_mat = Some(read_density_matrix(o, s, &filename)?);
    }

    // (2) check & store parameters of Hamiltonian
    o.print("\n  @ Hamiltonian \n\n");
    mu::not_used_d(o, "K", s.K)?;
    mu::print_val_d(o, "h", &mut s.h, 0.0);
    mu::print_val_d(o, "Gamma", &mut s.Gamma, 0.0);
    mu::print_val_d(o, "Gamma_y", &mut s.Gamma_y, 0.0);
    mu::not_used_d(o, "U", s.U)?;
    if s.model == "spin" {
        mu::print_val_i(o, "2S", &mut s.S2, 1);
    } else if s.model == "hubbard" {
        mu::print_val_d(o, "mu", &mut s.mu, 0.0);
    } else {
        o.print("wannier + Kondo is not available !\n");
        return exit(-1);
    }
    o.print("\n  @ Numerical conditions\n\n");

    // (3) Set local spin flag and the number of sites
    s.nsite = s.NsiteUC * s.NCell;
    s.locspinflag = vec![if s.model == "spin" { s.S2 } else { 0 }; s.nsite as usize];
    mu::malloc_interactions(s, 0, 0);

    // (4.5) For a spin system, compute the super exchange interaction.
    let spin = s.model == "spin";
    let mut uspin: Vec<Option<f64>> = vec![None; s.NsiteUC.max(0) as usize];
    if spin {
        for (it, t) in tuj_indx[1].iter().enumerate() {
            if t[0] == 0 && t[1] == 0 && t[2] == 0 && t[3] == t[4] {
                uspin[t[3] as usize] = Some(u_val[it].re);
            }
        }
    }
    let zero_den = DenMat {
        rmin: [0; 3],
        rmax: [0; 3],
        n: 0,
        data: Vec::new(),
    };
    let den = den_mat.as_ref().unwrap_or(&zero_den);
    let (t_val, u_val, j_val) = (&t_val, &u_val, &j_val);
    let mut jtmp = [[0.0f64; 3]; 3];

    // (5) Set Transfer & Interaction
    for kcell in 0..s.NCell {
        let [iw, il, ih] = s.Cell[kcell as usize];
        // Local term 1
        for isite in s.NsiteUC * kcell..s.NsiteUC * (kcell + 1) {
            if spin {
                mu::mag_field(s, s.S2, -s.h, -s.Gamma, -s.Gamma_y, isite);
            } else {
                mu::hubbard_local(s, s.mu, -s.h, -s.Gamma, -s.Gamma_y, 0.0, isite);
            }
        }
        // Hopping
        for (it, t) in tuj_indx[0].iter().enumerate() {
            if t[0] == 0 && t[1] == 0 && t[2] == 0 && t[3] == t[4] {
                // Local term
                if s.model == "hubbard" {
                    let isite = s.NsiteUC * kcell + t[3];
                    for ispin in 0..2 {
                        push_diag_trans(s, -t_val[it], isite, ispin);
                    }
                }
            } else {
                // Non-local term
                let pair = mu::find_site(s, iw, il, ih, t[0], t[1], t[2], t[3], t[4]);
                if spin {
                    let (ua, ub) = (uspin[t[3] as usize], uspin[t[4] as usize]);
                    let (Some(ua), Some(ub)) = (ua, ub) else {
                        // C defect: reads uninitialised memory when an orbital has no
                        // on-site U in the `_ur` file.
                        o.eprint(
                            "\n  Error: the on-site Coulomb U of a Wannier orbital is not found, but the spin model needs it for the super exchange. \n\n",
                        );
                        return exit(-1);
                    };
                    for ii in 0..3 {
                        // 2.0 * t * conj(t) * (1.0 / Uspin[i] + 1.0 / Uspin[j])
                        jtmp[ii][ii] = (t_val[it].scale(2.0) * t_val[it].conj())
                            .scale(1.0 / ua + 1.0 / ub)
                            .re;
                    }
                    let j = jtmp;
                    mu::general_j(s, &j, s.S2, s.S2, pair.isite, pair.jsite);
                } else {
                    mu::hopping(
                        s,
                        (-pair.cphase) * t_val[it],
                        pair.jsite,
                        pair.isite,
                        &pair.dR,
                    );
                }
            }
        }
        // Coulomb integral (U)
        for (it, t) in tuj_indx[1].iter().enumerate() {
            let u_re = u_val[it].re;
            if t[0] == 0 && t[1] == 0 && t[2] == 0 && t[3] == t[4] {
                // Local term
                s.Cintra.push(u_re);
                s.CintraIndx.push([s.NsiteUC * kcell + t[3]]);
                if idcmode != DcMode::NotCorrect {
                    let isite = s.NsiteUC * kcell + t[3];
                    for ispin in 0..2 {
                        let den0 = den.get([0, 0, 0], t[3], t[3]);
                        push_diag_trans(s, den0.scale(s.alpha * u_re), isite, ispin);
                    }
                }
            } else {
                // Non-local term
                let pair = mu::find_site(s, iw, il, ih, t[0], t[1], t[2], t[3], t[4]);
                mu::coulomb(s, u_re, pair.isite, pair.jsite);
                if idcmode != DcMode::NotCorrect {
                    for ispin in 0..2 {
                        let den0 = den.get([0, 0, 0], t[4], t[4]);
                        push_diag_trans(s, den0.scale(u_re), pair.isite, ispin);
                        let den0 = den.get([0, 0, 0], t[3], t[3]);
                        push_diag_trans(s, den0.scale(u_re), pair.jsite, ispin);
                    }
                    if idcmode == DcMode::Full {
                        let den0 = den.get([t[0], t[1], t[2]], t[3], t[4]);
                        mu::hopping(
                            s,
                            pair.cphase.scale(-0.5).scale(u_re) * den0,
                            pair.jsite,
                            pair.isite,
                            &pair.dR,
                        );
                    }
                }
            }
        }
        // Hund coupling (J)
        for (it, t) in tuj_indx[2].iter().enumerate() {
            let j_re = j_val[it].re;
            // The local term is not computed.
            if t[0] != 0 || t[1] != 0 || t[2] != 0 || t[3] != t[4] {
                let pair = mu::find_site(s, iw, il, ih, t[0], t[1], t[2], t[3], t[4]);
                s.Hund.push(j_re);
                s.HundIndx.push([pair.isite, pair.jsite]);
                if s.model == "hubbard" {
                    s.Ex.push(j_re);
                    s.ExIndx.push([pair.isite, pair.jsite]);
                    s.PairHopp.push(j_re);
                    s.PHIndx.push([pair.isite, pair.jsite]);
                    if idcmode != DcMode::NotCorrect && idcmode != DcMode::HartreeU {
                        for ispin in 0..2 {
                            let den0 = den.get([0, 0, 0], t[4], t[4]);
                            push_diag_trans(
                                s,
                                den0.scale(-(1.0 - s.alpha) * j_re),
                                pair.isite,
                                ispin,
                            );
                            let den0 = den.get([0, 0, 0], t[3], t[3]);
                            push_diag_trans(
                                s,
                                den0.scale(-(1.0 - s.alpha) * j_re),
                                pair.jsite,
                                ispin,
                            );
                        }
                        if idcmode == DcMode::Full {
                            let den0 = den.get([t[0], t[1], t[2]], t[3], t[4]);
                            mu::hopping(
                                s,
                                pair.cphase.scale(0.5).scale(j_re) * den0.plus_real(2.0 * den0.re),
                                pair.jsite,
                                pair.isite,
                                &pair.dR,
                            );
                        }
                    }
                } else {
                    s.Ex.push(j_re);
                    s.ExIndx.push([pair.isite, pair.jsite]);
                }
            }
        }
    }

    if idcmode != DcMode::NotCorrect {
        print_uhf_initial(o, s, &tuj_indx, den)?;
    }
    mu::print_xsf(o, s)?;
    mu::print_geometry(o, s)?;

    let mut t = String::new();
    t.push_str("======================== \n");
    t.push_str(&format!(
        "Total site number {}  \n",
        cfmt::d(s.NCell * s.NsiteUC, 7)
    ));
    t.push_str("======================== \n");
    t.push_str("========site nx ny nz norb====== \n");
    t.push_str("======================== \n");
    for kcell in 0..s.NCell {
        let [nx, ny, nz] = s.Cell[kcell as usize];
        for it in 0..s.NsiteUC {
            let isite = s.NsiteUC * kcell + it;
            t.push_str(&format!(
                "{}{}{}{}{}\n",
                cfmt::d(isite, 5),
                cfmt::d(nx, 5),
                cfmt::d(ny, 5),
                cfmt::d(nz, 5),
                cfmt::d(it, 5)
            ));
        }
    }
    o.write_file("wan2site.dat", &t)?;
    Ok(())
}

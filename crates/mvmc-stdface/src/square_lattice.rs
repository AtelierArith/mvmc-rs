//! Port of `SquareLattice.c` (`StdFace_Tetragonal`, mVMC branch): the tetragonal (square) lattice.
//!
//! Generated from the C source with the corrections of issue #404
//! (`c_toolbox/stdface/lattice_defects.patch`) by a mechanical statement-by-statement translation and then
//! reviewed; the output is checked byte for byte against the C program (see
//! `tests/stdface_c_fixtures.rs`). The HPhi-only boost routines are not ported.
#![allow(non_snake_case)]

use crate::ccomplex::C64;
use crate::model_util as mu;
use crate::out::{Out, Res};
use crate::vals::StdIntList;

/// Setup a Hamiltonian for the Hubbard / spin / Kondo model on the tetragonal (square) lattice.
pub fn std_face_tetragonal(o: &mut Out, s: &mut StdIntList) -> Res<()> {
    // fp = fopen("lattice.gp", "w"); exit() flushes what was written so far.
    let mut gp = Some(String::new());
    let result = body(o, s, &mut gp);
    let gp_text = gp.unwrap_or_default();
    o.write_file("lattice.gp", &gp_text)?;
    result?;
    mu::print_geometry(o, s)
}

fn body(o: &mut Out, s: &mut StdIntList, gp: &mut Option<String>) -> Res<()> {
    // (1) Compute the shape of the super-cell and sites in the super-cell
    s.NsiteUC = 1;
    o.print("  @ Lattice Size & Shape\n\n");

    mu::print_val_d(o, "a", &mut s.a, 1.0);
    mu::print_val_d(o, "Wlength", &mut s.length[0], s.a);
    mu::print_val_d(o, "Llength", &mut s.length[1], s.a);
    mu::print_val_d(o, "Wx", &mut s.direct[0][0], s.length[0]);
    mu::print_val_d(o, "Wy", &mut s.direct[0][1], 0.0);
    mu::print_val_d(o, "Lx", &mut s.direct[1][0], 0.0);
    mu::print_val_d(o, "Ly", &mut s.direct[1][1], s.length[1]);

    mu::print_val_d(o, "phase0", &mut s.phase[0], 0.0);
    mu::print_val_d(o, "phase1", &mut s.phase[1], 0.0);
    mu::init_site(o, s, gp, 2)?;
    s.tau[0] = [0.0, 0.0, 0.0];

    // (2) check & store parameters of Hamiltonian
    o.print("\n  @ Hamiltonian \n\n");
    mu::not_used_j(o, "J2", s.J2All, &s.J2)?;
    mu::not_used_j(o, "J2'", s.J2pAll, &s.J2p)?;
    mu::not_used_c(o, "t2", s.t2)?;
    mu::not_used_d(o, "t2'", s.t2p.re)?;
    mu::not_used_d(o, "V2", s.V2)?;
    mu::not_used_d(o, "V2'", s.V2p)?;
    mu::not_used_d(o, "K", s.K)?;
    mu::print_val_d(o, "h", &mut s.h, 0.0);
    mu::print_val_d(o, "Gamma", &mut s.Gamma, 0.0);
    mu::print_val_d(o, "Gamma_y", &mut s.Gamma_y, 0.0);
    if s.model == "spin" {
        mu::print_val_i(o, "2S", &mut s.S2, 1);
        mu::print_val_d(o, "D", &mut s.D[2][2], 0.0);
        mu::input_spin_nn(o, &s.J, s.JAll, &mut s.J0, s.J0All, "J0")?;
        mu::input_spin_nn(o, &s.J, s.JAll, &mut s.J1, s.J1All, "J1")?;
        mu::input_spin_nn(o, &s.Jp, s.JpAll, &mut s.J0p, s.J0pAll, "J0'")?;
        mu::input_spin_nn(o, &s.Jp, s.JpAll, &mut s.J1p, s.J1pAll, "J1'")?;
        mu::input_spin_nn(o, &s.Jpp, s.JppAll, &mut s.J0pp, s.J0ppAll, "J0''")?;
        mu::input_spin_nn(o, &s.Jpp, s.JppAll, &mut s.J1pp, s.J1ppAll, "J1''")?;
        mu::not_used_d(o, "mu", s.mu)?;
        mu::not_used_d(o, "U", s.U)?;
        mu::not_used_c(o, "t", s.t)?;
        mu::not_used_c(o, "t0", s.t0)?;
        mu::not_used_c(o, "t1", s.t1)?;
        mu::not_used_c(o, "t'", s.tp)?;
        mu::not_used_c(o, "t0'", s.t0p)?;
        mu::not_used_c(o, "t1'", s.t1p)?;
        mu::not_used_c(o, "t''", s.tpp)?;
        mu::not_used_c(o, "t0''", s.t0pp)?;
        mu::not_used_c(o, "t1''", s.t1pp)?;
        mu::not_used_d(o, "V", s.V)?;
        mu::not_used_d(o, "V0", s.V0)?;
        mu::not_used_d(o, "V1", s.V1)?;
        mu::not_used_d(o, "V'", s.Vp)?;
        mu::not_used_d(o, "V0'", s.V0p)?;
        mu::not_used_d(o, "V1'", s.V1p)?;
        mu::not_used_d(o, "V''", s.Vpp)?;
        mu::not_used_d(o, "V0''", s.V0pp)?;
        mu::not_used_d(o, "V1''", s.V1pp)?;
    } else {
        mu::print_val_d(o, "mu", &mut s.mu, 0.0);
        mu::print_val_d(o, "U", &mut s.U, 0.0);
        mu::input_hopp(o, s.t, &mut s.t0, "t0")?;
        mu::input_hopp(o, s.t, &mut s.t1, "t1")?;
        mu::input_hopp(o, s.tp, &mut s.t0p, "t0'")?;
        mu::input_hopp(o, s.tp, &mut s.t1p, "t1'")?;
        mu::input_hopp(o, s.tpp, &mut s.t0pp, "t0''")?;
        mu::input_hopp(o, s.tpp, &mut s.t1pp, "t1''")?;
        mu::input_coulomb_v(o, s.V, &mut s.V0, "V0")?;
        mu::input_coulomb_v(o, s.V, &mut s.V1, "V1")?;
        mu::input_coulomb_v(o, s.Vp, &mut s.V0p, "V0'")?;
        mu::input_coulomb_v(o, s.Vp, &mut s.V1p, "V1'")?;
        mu::input_coulomb_v(o, s.Vpp, &mut s.V0pp, "V0''")?;
        mu::input_coulomb_v(o, s.Vpp, &mut s.V1pp, "V1''")?;
        mu::not_used_j(o, "J0", s.J0All, &s.J0)?;
        mu::not_used_j(o, "J1", s.J1All, &s.J1)?;
        mu::not_used_j(o, "J'", s.JpAll, &s.Jp)?;
        mu::not_used_d(o, "D", s.D[2][2])?;
        if s.model == "hubbard" {
            mu::not_used_i(o, "2S", s.S2)?;
            mu::not_used_j(o, "J", s.JAll, &s.J)?;
        } else {
            mu::print_val_i(o, "2S", &mut s.S2, 1);
            mu::input_spin(o, &mut s.J, s.JAll, "J")?;
        }
    }
    o.print("\n  @ Numerical conditions\n\n");

    // (3) Set local spin flag and the number of sites
    s.nsite = s.NsiteUC * s.NCell;
    if s.model == "kondo" {
        s.nsite *= 2;
    }
    s.locspinflag = vec![0; s.nsite as usize];
    if s.model == "spin" {
        s.locspinflag.iter_mut().for_each(|f| *f = s.S2);
    } else if s.model != "hubbard" {
        let half = (s.nsite / 2) as usize;
        for isite in 0..half {
            s.locspinflag[isite] = s.S2;
            s.locspinflag[isite + half] = 0;
        }
    }

    // (4) C sizes its arrays here; the Rust lists grow on demand.
    mu::malloc_interactions(s, 0, 0);

    // (5) Set Transfer & Interaction
    let mut isite: i32;
    let mut jsite: i32;
    let mut cphase: C64;
    let mut dr: [f64; 3];
    for kcell in 0..s.NCell {
        let iw = s.Cell[kcell as usize][0];
        let il = s.Cell[kcell as usize][1];
        isite = kcell;
        if s.model == "kondo" {
            isite += s.NCell;
        }
        if s.model == "spin" {
            mu::mag_field(s, s.S2, -s.h, -s.Gamma, -s.Gamma_y, isite);
            {
                let j = s.D;
                mu::general_j(s, &j, s.S2, s.S2, isite, isite);
            }
        } else {
            mu::hubbard_local(s, s.mu, -s.h, -s.Gamma, -s.Gamma_y, s.U, isite);
            if s.model == "kondo" {
                jsite = kcell;
                {
                    let j = s.J;
                    mu::general_j(s, &j, 1, s.S2, isite, jsite);
                }
                mu::mag_field(s, s.S2, -s.h, -s.Gamma, -s.Gamma_y, jsite);
            }
        }
        let pair = mu::set_label(s, gp, iw, il, 1, 0, 0, 0, 1);
        (isite, jsite, cphase, dr) = (pair.isite, pair.jsite, pair.cphase, pair.dR);
        if s.model == "spin" {
            {
                let j = s.J0;
                mu::general_j(s, &j, s.S2, s.S2, isite, jsite);
            }
        } else {
            mu::hopping(s, cphase * s.t0, isite, jsite, &dr);
            mu::coulomb(s, s.V0, isite, jsite);
        }
        let pair = mu::set_label(s, gp, iw, il, 0, 1, 0, 0, 1);
        (isite, jsite, cphase, dr) = (pair.isite, pair.jsite, pair.cphase, pair.dR);
        if s.model == "spin" {
            {
                let j = s.J1;
                mu::general_j(s, &j, s.S2, s.S2, isite, jsite);
            }
        } else {
            mu::hopping(s, cphase * s.t1, isite, jsite, &dr);
            mu::coulomb(s, s.V1, isite, jsite);
        }
        let pair = mu::set_label(s, gp, iw, il, 1, 1, 0, 0, 2);
        (isite, jsite, cphase, dr) = (pair.isite, pair.jsite, pair.cphase, pair.dR);
        if s.model == "spin" {
            {
                let j = s.J0p;
                mu::general_j(s, &j, s.S2, s.S2, isite, jsite);
            }
        } else {
            mu::hopping(s, cphase * s.t0p, isite, jsite, &dr);
            mu::coulomb(s, s.V0p, isite, jsite);
        }
        let pair = mu::set_label(s, gp, iw, il, 1, -1, 0, 0, 2);
        (isite, jsite, cphase, dr) = (pair.isite, pair.jsite, pair.cphase, pair.dR);
        if s.model == "spin" {
            {
                let j = s.J1p;
                mu::general_j(s, &j, s.S2, s.S2, isite, jsite);
            }
        } else {
            mu::hopping(s, cphase * s.t1p, isite, jsite, &dr);
            mu::coulomb(s, s.V1p, isite, jsite);
        }
        let pair = mu::set_label(s, gp, iw, il, 2, 0, 0, 0, 3);
        (isite, jsite, cphase, dr) = (pair.isite, pair.jsite, pair.cphase, pair.dR);
        if s.model == "spin" {
            {
                let j = s.J0pp;
                mu::general_j(s, &j, s.S2, s.S2, isite, jsite);
            }
        } else {
            mu::hopping(s, cphase * s.t0pp, isite, jsite, &dr);
            mu::coulomb(s, s.V0pp, isite, jsite);
        }
        let pair = mu::set_label(s, gp, iw, il, 0, 2, 0, 0, 3);
        (isite, jsite, cphase, dr) = (pair.isite, pair.jsite, pair.cphase, pair.dR);
        if s.model == "spin" {
            {
                let j = s.J1pp;
                mu::general_j(s, &j, s.S2, s.S2, isite, jsite);
            }
        } else {
            mu::hopping(s, cphase * s.t1pp, isite, jsite, &dr);
            mu::coulomb(s, s.V1pp, isite, jsite);
        }
    }
    if let Some(fp) = gp.as_mut() {
        fp.push_str("plot '-' w d lc 7\n0.0 0.0\nend\npause -1\n");
    }
    Ok(())
}

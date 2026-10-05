//! Port of `Ladder.c` (`StdFace_Ladder`, mVMC branch): the W-leg ladder lattice.
//!
//! The HPhi-only `StdFace_Ladder_Boost` is not ported.
#![allow(non_snake_case)]

use crate::model_util as mu;
use crate::out::{Out, Res};
use crate::vals::StdIntList;

/// Setup a Hamiltonian for the Hubbard / spin / Kondo model on a ladder lattice.
pub fn std_face_ladder(o: &mut Out, s: &mut StdIntList) -> Res<()> {
    // fp = fopen("lattice.gp", "w"); exit() flushes what was written so far.
    let mut gp = Some(String::new());
    let result = ladder_body(o, s, &mut gp);
    let gp_text = gp.unwrap_or_default();
    o.write_file("lattice.gp", &gp_text)?;
    result?;
    mu::print_geometry(o, s)
}

fn ladder_body(o: &mut Out, s: &mut StdIntList, gp: &mut Option<String>) -> Res<()> {
    // (1) Compute the shape of the super-cell and sites in the super-cell
    o.print("  @ Lattice Size & Shape\n\n");

    mu::print_val_d(o, "a", &mut s.a, 1.0);
    mu::print_val_d(o, "Wlength", &mut s.length[0], s.a);
    mu::print_val_d(o, "Llength", &mut s.length[1], s.a);
    mu::print_val_d(o, "Wx", &mut s.direct[0][0], s.length[0]);
    mu::print_val_d(o, "Wy", &mut s.direct[0][1], 0.0);
    mu::print_val_d(o, "Lx", &mut s.direct[1][0], 0.0);
    mu::print_val_d(o, "Ly", &mut s.direct[1][1], s.length[1]);

    mu::required_val_i(o, "L", s.L)?;
    mu::required_val_i(o, "W", s.W)?;
    mu::not_used_i(o, "a0W", s.box_[0][0])?;
    mu::not_used_i(o, "a0L", s.box_[0][1])?;
    mu::not_used_i(o, "a1W", s.box_[1][0])?;
    mu::not_used_i(o, "a1L", s.box_[1][1])?;

    mu::print_val_d(o, "phase0", &mut s.phase[0], 0.0);
    mu::not_used_d(o, "phase1", s.phase[1])?;
    s.phase[1] = s.phase[0];
    s.phase[0] = 0.0;

    s.NsiteUC = s.W;
    s.W = 1;
    // C overwrites Wx after printing it (the printed value is not the one used).
    s.direct[0][0] = s.NsiteUC as f64;
    mu::init_site(o, s, gp, 2)?;
    for isite in 0..s.NsiteUC as usize {
        s.tau[isite] = [isite as f64 / s.NsiteUC as f64, 0.0, 0.0];
    }

    // (2) check & store parameters of Hamiltonian
    o.print("\n  @ Hamiltonian \n\n");
    mu::not_used_j(o, "J", s.JAll, &s.J)?;
    mu::not_used_j(o, "J'", s.JpAll, &s.Jp)?;
    mu::not_used_c(o, "t", s.t)?;
    mu::not_used_c(o, "t'", s.tp)?;
    mu::not_used_d(o, "V", s.V)?;
    mu::not_used_d(o, "V'", s.Vp)?;
    mu::not_used_d(o, "K", s.K)?;
    mu::print_val_d(o, "h", &mut s.h, 0.0);
    mu::print_val_d(o, "Gamma", &mut s.Gamma, 0.0);
    mu::print_val_d(o, "Gamma_y", &mut s.Gamma_y, 0.0);

    if s.model == "spin" {
        mu::print_val_i(o, "2S", &mut s.S2, 1);
        mu::print_val_d(o, "D", &mut s.D[2][2], 0.0);
        mu::input_spin(o, &mut s.J0, s.J0All, "J0")?;
        mu::input_spin(o, &mut s.J1, s.J1All, "J1")?;
        mu::input_spin(o, &mut s.J2, s.J2All, "J2")?;
        mu::input_spin(o, &mut s.J1p, s.J1pAll, "J1'")?;
        mu::input_spin(o, &mut s.J2p, s.J2pAll, "J2'")?;

        mu::not_used_d(o, "mu", s.mu)?;
        mu::not_used_d(o, "U", s.U)?;
        mu::not_used_c(o, "t0", s.t0)?;
        mu::not_used_c(o, "t1", s.t1)?;
        mu::not_used_c(o, "t2", s.t2)?;
        mu::not_used_c(o, "t1'", s.t1p)?;
        mu::not_used_c(o, "t2'", s.t2p)?;
        mu::not_used_d(o, "V0", s.V0)?;
        mu::not_used_d(o, "V1", s.V1)?;
        mu::not_used_d(o, "V2", s.V2)?;
        mu::not_used_d(o, "V1'", s.V1p)?;
        mu::not_used_d(o, "V2'", s.V2p)?;
    } else {
        mu::print_val_d(o, "mu", &mut s.mu, 0.0);
        mu::print_val_d(o, "U", &mut s.U, 0.0);
        mu::input_hopp(o, s.t, &mut s.t0, "t0")?;
        mu::input_hopp(o, s.t, &mut s.t1, "t1")?;
        mu::input_hopp(o, s.t, &mut s.t2, "t2")?;
        mu::input_hopp(o, s.t, &mut s.t1p, "t1'")?;
        mu::input_hopp(o, s.t, &mut s.t2p, "t2'")?;
        mu::input_coulomb_v(o, s.V, &mut s.V0, "V0")?;
        mu::input_coulomb_v(o, s.V, &mut s.V1, "V1")?;
        mu::input_coulomb_v(o, s.V, &mut s.V2, "V2")?;
        mu::input_coulomb_v(o, s.V, &mut s.V1p, "V1'")?;
        mu::input_coulomb_v(o, s.V, &mut s.V2p, "V2'")?;

        mu::not_used_j(o, "J0", s.J0All, &s.J0)?;
        mu::not_used_j(o, "J1", s.J1All, &s.J1)?;
        mu::not_used_j(o, "J2", s.J2All, &s.J2)?;
        // C spells these "J1p"/"J2p" (not "J1'"/"J2'"): the component names differ in messages.
        mu::not_used_j(o, "J1p", s.J1pAll, &s.J1p)?;
        mu::not_used_j(o, "J2p", s.J2pAll, &s.J2p)?;
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
    s.nsite = s.L * s.NsiteUC;
    if s.model == "kondo" {
        s.nsite *= 2;
    }
    s.locspinflag = vec![0; s.nsite as usize];
    if s.model == "spin" {
        s.locspinflag.iter_mut().for_each(|f| *f = s.S2);
    } else if s.model == "kondo" {
        let half = (s.nsite / 2) as usize;
        for isite in 0..half {
            s.locspinflag[isite] = s.S2;
            s.locspinflag[isite + half] = 0;
        }
    }

    // (4) Upper limit of the number of Transfer & Interaction
    let (mut ntrans_max, mut nintr_max);
    if s.model == "spin" {
        ntrans_max = s.L * s.NsiteUC * (s.S2 + 1 + 2 * s.S2);
        nintr_max = s.L * s.NsiteUC * (1 + 1 + 1) * (3 * s.S2 + 1) * (3 * s.S2 + 1)
            + s.L * (s.NsiteUC - 1) * (1 + 1 + 1) * (3 * s.S2 + 1) * (3 * s.S2 + 1);
    } else {
        ntrans_max = s.L * s.NsiteUC * 2 * (2 + 2 + 2) + s.L * (s.NsiteUC - 1) * 2 * (2 + 2 + 2);
        nintr_max = s.L * s.NsiteUC
            + s.L * s.NsiteUC * 4 * (1 + 1)
            + s.L * (s.NsiteUC - 1) * 4 * (1 + 1 + 1);
        if s.model == "kondo" {
            ntrans_max += s.L * s.NsiteUC * (s.S2 + 1 + 2 * s.S2);
            nintr_max += s.nsite / 2 * (3 + 1) * (3 * s.S2 + 1);
        }
    }
    mu::malloc_interactions(s, ntrans_max, nintr_max);

    // (5) Set Transfer & Interaction
    let spin = s.model == "spin";
    for il in 0..s.L {
        for isite_uc in 0..s.NsiteUC {
            let mut isite = isite_uc + il * s.NsiteUC;
            if s.model == "kondo" {
                isite += s.L * s.NsiteUC;
            }
            // Local term
            if spin {
                mu::mag_field(s, s.S2, -s.h, -s.Gamma, -s.Gamma_y, isite);
                let d = s.D;
                mu::general_j(s, &d, s.S2, s.S2, isite, isite);
            } else {
                mu::hubbard_local(s, s.mu, -s.h, -s.Gamma, -s.Gamma_y, s.U, isite);
                if s.model == "kondo" {
                    let jsite = isite_uc + il * s.NsiteUC;
                    let j = s.J;
                    mu::general_j(s, &j, 1, s.S2, isite, jsite);
                    mu::mag_field(s, s.S2, -s.h, -s.Gamma, -s.Gamma_y, jsite);
                }
            }
            // Nearest neighbor along the ladder
            let pair = mu::set_label(s, gp, 0, il, 0, 1, isite_uc, isite_uc, 1);
            if spin {
                let j = s.J1;
                mu::general_j(s, &j, s.S2, s.S2, pair.isite, pair.jsite);
            } else {
                mu::hopping(s, pair.cphase * s.t1, pair.isite, pair.jsite, &pair.dR);
                mu::coulomb(s, s.V1, pair.isite, pair.jsite);
            }
            // Second nearest neighbor along the ladder
            let pair = mu::set_label(s, gp, 0, il, 0, 2, isite_uc, isite_uc, 2);
            if spin {
                let j = s.J1p;
                mu::general_j(s, &j, s.S2, s.S2, pair.isite, pair.jsite);
            } else {
                mu::hopping(s, pair.cphase * s.t1p, pair.isite, pair.jsite, &pair.dR);
                mu::coulomb(s, s.V1p, pair.isite, pair.jsite);
            }
            // Across rung
            if isite_uc < s.NsiteUC - 1 {
                // Vertical
                let pair = mu::set_label(s, gp, 0, il, 0, 0, isite_uc, isite_uc + 1, 1);
                if spin {
                    let j = s.J0;
                    mu::general_j(s, &j, s.S2, s.S2, pair.isite, pair.jsite);
                } else {
                    mu::hopping(s, pair.cphase * s.t0, pair.isite, pair.jsite, &pair.dR);
                    mu::coulomb(s, s.V0, pair.isite, pair.jsite);
                }
                // Diagonal 1
                let pair = mu::set_label(s, gp, 0, il, 0, 1, isite_uc, isite_uc + 1, 1);
                if spin {
                    let j = s.J2;
                    mu::general_j(s, &j, s.S2, s.S2, pair.isite, pair.jsite);
                } else {
                    mu::hopping(s, pair.cphase * s.t2, pair.isite, pair.jsite, &pair.dR);
                    mu::coulomb(s, s.V2, pair.isite, pair.jsite);
                }
                // Diagonal 2
                let pair = mu::set_label(s, gp, 0, il, 0, -1, isite_uc, isite_uc + 1, 1);
                if spin {
                    let j = s.J2p;
                    mu::general_j(s, &j, s.S2, s.S2, pair.isite, pair.jsite);
                } else {
                    mu::hopping(s, pair.cphase * s.t2p, pair.isite, pair.jsite, &pair.dR);
                    mu::coulomb(s, s.V2p, pair.isite, pair.jsite);
                }
            }
        }
    }
    if let Some(fp) = gp.as_mut() {
        fp.push_str("plot '-' w d lc 7\n0.0 0.0\nend\npause -1\n");
    }
    Ok(())
}

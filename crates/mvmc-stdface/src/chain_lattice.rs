//! Port of `ChainLattice.c` (`StdFace_Chain`, mVMC branch): the 1D chain lattice.
//!
//! The HPhi-only `StdFace_Chain_Boost` is not ported.
#![allow(non_snake_case)]

use crate::model_util as mu;
use crate::out::{Out, Res};
use crate::vals::StdIntList;

/// Setup a Hamiltonian for the Hubbard / spin / Kondo model on a chain lattice.
pub fn std_face_chain(o: &mut Out, s: &mut StdIntList) -> Res<()> {
    // fp = fopen("lattice.gp", "w"): the file exists even if a later check exits, and `exit()`
    // flushes whatever was written so far.
    let mut gp = Some(String::new());
    let result = chain_body(o, s, &mut gp);
    let gp_text = gp.unwrap_or_default();
    o.write_file("lattice.gp", &gp_text)?;
    result?;
    mu::print_geometry(o, s)
}

fn chain_body(o: &mut Out, s: &mut StdIntList, gp: &mut Option<String>) -> Res<()> {
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
    mu::not_used_d(o, "phase1", s.phase[1])?;
    s.phase[1] = s.phase[0];
    s.phase[0] = 0.0;

    mu::required_val_i(o, "L", s.L)?;
    mu::not_used_i(o, "W", s.W)?;
    s.W = 1;
    mu::init_site(o, s, gp, 2)?;
    s.tau[0] = [0.0, 0.0, 0.0];

    // (2) check & store parameters of Hamiltonian
    o.print("\n  @ Hamiltonian \n\n");
    mu::not_used_j(o, "J1", s.J1All, &s.J1)?;
    mu::not_used_j(o, "J2", s.J2All, &s.J2)?;
    mu::not_used_j(o, "J1'", s.J1pAll, &s.J1p)?;
    mu::not_used_j(o, "J2'", s.J2pAll, &s.J2p)?;
    mu::not_used_c(o, "t1", s.t1)?;
    mu::not_used_c(o, "t2", s.t2)?;
    mu::not_used_d(o, "t1'", s.t1p.re)?;
    mu::not_used_d(o, "t2'", s.t2p.re)?;
    mu::not_used_d(o, "V1", s.V1)?;
    mu::not_used_d(o, "V2", s.V2)?;
    mu::not_used_d(o, "V1'", s.V1p)?;
    mu::not_used_d(o, "V2'", s.V2p)?;
    mu::not_used_d(o, "K", s.K)?;
    mu::print_val_d(o, "h", &mut s.h, 0.0);
    mu::print_val_d(o, "Gamma", &mut s.Gamma, 0.0);
    mu::print_val_d(o, "Gamma_y", &mut s.Gamma_y, 0.0);

    if s.model == "spin" {
        mu::print_val_i(o, "2S", &mut s.S2, 1);
        mu::print_val_d(o, "D", &mut s.D[2][2], 0.0);
        mu::input_spin_nn(o, &s.J, s.JAll, &mut s.J0, s.J0All, "J0")?;
        mu::input_spin_nn(o, &s.Jp, s.JpAll, &mut s.J0p, s.J0pAll, "J0'")?;
        // C labels the third neighbour "J0'" (copy-paste slip); corrected to "J0''" (#404).
        mu::input_spin_nn(o, &s.Jpp, s.JppAll, &mut s.J0pp, s.J0ppAll, "J0''")?;
        mu::not_used_d(o, "mu", s.mu)?;
        mu::not_used_d(o, "U", s.U)?;
        mu::not_used_c(o, "t", s.t)?;
        mu::not_used_c(o, "t0", s.t0)?;
        mu::not_used_c(o, "t'", s.tp)?;
        mu::not_used_d(o, "V", s.V)?;
        mu::not_used_d(o, "V0", s.V0)?;
        mu::not_used_d(o, "V'", s.Vp)?;
    } else {
        mu::print_val_d(o, "mu", &mut s.mu, 0.0);
        mu::print_val_d(o, "U", &mut s.U, 0.0);
        mu::input_hopp(o, s.t, &mut s.t0, "t0")?;
        mu::input_hopp(o, s.tp, &mut s.t0p, "t0'")?;
        mu::input_hopp(o, s.tpp, &mut s.t0pp, "t0''")?;
        mu::input_coulomb_v(o, s.V, &mut s.V0, "V0")?;
        mu::input_coulomb_v(o, s.Vp, &mut s.V0p, "V0'")?;
        mu::input_coulomb_v(o, s.Vpp, &mut s.V0pp, "V0''")?;

        mu::not_used_j(o, "J0", s.J0All, &s.J0)?;
        mu::not_used_j(o, "J0'", s.J0pAll, &s.J0p)?;
        mu::not_used_j(o, "J0''", s.J0ppAll, &s.J0pp)?;
        mu::not_used_d(o, "D", s.D[2][2])?;

        if s.model == "hubbard" {
            mu::not_used_i(o, "2S", s.S2)?;
            mu::not_used_j(o, "J", s.JAll, &s.J)?;
        } else if s.model == "kondo" {
            mu::print_val_i(o, "2S", &mut s.S2, 1);
            mu::input_spin(o, &mut s.J, s.JAll, "J")?;
        }
    }
    o.print("\n  @ Numerical conditions\n\n");

    // (3) Set local spin flag and the number of sites
    s.nsite = s.L;
    if s.model == "kondo" {
        s.nsite *= 2;
    }
    s.locspinflag = vec![0; s.nsite as usize];
    if s.model == "spin" {
        s.locspinflag.iter_mut().for_each(|f| *f = s.S2);
    } else if s.model == "hubbard" {
        s.locspinflag.iter_mut().for_each(|f| *f = 0);
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
        ntrans_max = s.L * (s.S2 + 1 + 2 * s.S2);
        nintr_max = s.L * (s.NsiteUC + 1 + 1 + 1) * (3 * s.S2 + 1) * (3 * s.S2 + 1);
    } else {
        ntrans_max = s.L * 2 * (2 * s.NsiteUC + 2 + 2 + 2);
        nintr_max = s.L * (s.NsiteUC + 4 * (1 + 1 + 1));
        if s.model == "kondo" {
            ntrans_max += s.L * (s.S2 + 1 + 2 * s.S2);
            nintr_max += s.nsite / 2 * (3 + 1) * (3 * s.S2 + 1);
        }
    }
    mu::malloc_interactions(s, ntrans_max, nintr_max);

    // (5) Set Transfer & Interaction
    for il in 0..s.L {
        let mut isite = il;
        if s.model == "kondo" {
            isite += s.L;
        }
        // Local term
        if s.model == "spin" {
            mu::mag_field(s, s.S2, -s.h, -s.Gamma, -s.Gamma_y, isite);
            let d = s.D;
            mu::general_j(s, &d, s.S2, s.S2, isite, isite);
        } else {
            mu::hubbard_local(s, s.mu, -s.h, -s.Gamma, -s.Gamma_y, s.U, isite);
            if s.model == "kondo" {
                let jsite = il;
                let j = s.J;
                mu::general_j(s, &j, 1, s.S2, isite, jsite);
                mu::mag_field(s, s.S2, -s.h, -s.Gamma, -s.Gamma_y, jsite);
            }
        }
        // Nearest neighbor
        let pair = mu::set_label(s, gp, 0, il, 0, 1, 0, 0, 1);
        if s.model == "spin" {
            let j = s.J0;
            mu::general_j(s, &j, s.S2, s.S2, pair.isite, pair.jsite);
        } else {
            mu::hopping(s, pair.cphase * s.t0, pair.isite, pair.jsite, &pair.dR);
            mu::coulomb(s, s.V0, pair.isite, pair.jsite);
        }
        // Second nearest neighbor
        let pair = mu::set_label(s, gp, 0, il, 0, 2, 0, 0, 2);
        if s.model == "spin" {
            let j = s.J0p;
            mu::general_j(s, &j, s.S2, s.S2, pair.isite, pair.jsite);
        } else {
            mu::hopping(s, pair.cphase * s.t0p, pair.isite, pair.jsite, &pair.dR);
            mu::coulomb(s, s.V0p, pair.isite, pair.jsite);
        }
        // Third nearest neighbor
        let pair = mu::set_label(s, gp, 0, il, 0, 3, 0, 0, 3);
        if s.model == "spin" {
            let j = s.J0pp;
            mu::general_j(s, &j, s.S2, s.S2, pair.isite, pair.jsite);
        } else {
            mu::hopping(s, pair.cphase * s.t0pp, pair.isite, pair.jsite, &pair.dR);
            mu::coulomb(s, s.V0pp, pair.isite, pair.jsite);
        }
    }
    if let Some(fp) = gp.as_mut() {
        fp.push_str("plot '-' w d lc 7\n0.0 0.0\nend\npause -1\n");
    }
    Ok(())
}

//! Port of `Pyrochlore.c` (`StdFace_Pyrochlore`, mVMC branch): the pyrochlore lattice
//! (four sites per fcc unit cell).
//!
//! Defects of the C code that are NOT reproduced (see `tests/fixtures/stdface/README_3d_defects.md`):
//! the Kondo coupling `J` was applied as `GeneralJ(.., isite + 3, jsite + isiteUC)`, i.e. every
//! localized spin of a cell coupled to conduction site 3 and conduction sites 0-2 had no Kondo
//! coupling (now `isite + isiteUC`); and the unused parameters `t'`, `t''`, `V'`, `V''`, `V2`,
//! `V0'`, `V1'`, `V2'`, `J'`, `J''` were accepted without a check.
#![allow(non_snake_case)]

use crate::lattice3d::{bond, finish, set_sites_and_flags};
use crate::model_util as mu;
use crate::out::{Out, Res};
use crate::vals::StdIntList;

/// Setup a Hamiltonian for the Hubbard / spin / Kondo model on the pyrochlore lattice.
pub fn std_face_pyrochlore(o: &mut Out, s: &mut StdIntList) -> Res<()> {
    finish(o, s, body)
}

fn body(o: &mut Out, s: &mut StdIntList) -> Res<()> {
    // (1) Compute the shape of the super-cell and sites in the super-cell
    s.NsiteUC = 4;
    o.print("  @ Lattice Size & Shape\n\n");

    mu::print_val_d(o, "a", &mut s.a, 1.0);
    mu::print_val_d(o, "Wlength", &mut s.length[0], s.a);
    mu::print_val_d(o, "Llength", &mut s.length[1], s.a);
    mu::print_val_d(o, "Hlength", &mut s.length[2], s.a);
    mu::print_val_d(o, "Wx", &mut s.direct[0][0], 0.0);
    mu::print_val_d(o, "Wy", &mut s.direct[0][1], 0.5 * s.length[1]);
    mu::print_val_d(o, "Wz", &mut s.direct[0][2], 0.5 * s.length[2]);
    mu::print_val_d(o, "Lx", &mut s.direct[1][0], 0.5 * s.length[0]);
    mu::print_val_d(o, "Ly", &mut s.direct[1][1], 0.0);
    mu::print_val_d(o, "Lz", &mut s.direct[1][2], 0.5 * s.length[2]);
    mu::print_val_d(o, "Hx", &mut s.direct[2][0], 0.5 * s.length[0]);
    mu::print_val_d(o, "Hy", &mut s.direct[2][1], 0.5 * s.length[1]);
    mu::print_val_d(o, "Hz", &mut s.direct[2][2], 0.0);

    mu::print_val_d(o, "phase0", &mut s.phase[0], 0.0);
    mu::print_val_d(o, "phase1", &mut s.phase[1], 0.0);
    mu::print_val_d(o, "phase2", &mut s.phase[2], 0.0);

    mu::init_site(o, s, &mut None, 3)?;
    s.tau[0] = [0.0, 0.0, 0.0];
    s.tau[1] = [0.5, 0.0, 0.0];
    s.tau[2] = [0.0, 0.5, 0.0];
    s.tau[3] = [0.0, 0.0, 0.5];

    // (2) check & store parameters of Hamiltonian
    o.print("\n  @ Hamiltonian \n\n");
    mu::not_used_d(o, "K", s.K)?;
    mu::print_val_d(o, "h", &mut s.h, 0.0);
    mu::print_val_d(o, "Gamma", &mut s.Gamma, 0.0);
    mu::print_val_d(o, "Gamma_y", &mut s.Gamma_y, 0.0);

    if s.model == "spin" {
        mu::print_val_i(o, "2S", &mut s.S2, 1);
        mu::print_val_d(o, "D", &mut s.D[2][2], 0.0);
        // C passes J (not J') as the default of the primed couplings of this lattice.
        mu::input_spin_nn(o, &s.J, s.JAll, &mut s.J0, s.J0All, "J0")?;
        mu::input_spin_nn(o, &s.J, s.JAll, &mut s.J1, s.J1All, "J1")?;
        mu::input_spin_nn(o, &s.J, s.JAll, &mut s.J2, s.J2All, "J2")?;
        mu::input_spin_nn(o, &s.J, s.JAll, &mut s.J0p, s.J0pAll, "J0'")?;
        mu::input_spin_nn(o, &s.J, s.JAll, &mut s.J1p, s.J1pAll, "J1'")?;
        mu::input_spin_nn(o, &s.J, s.JAll, &mut s.J2p, s.J2pAll, "J2'")?;

        mu::not_used_d(o, "mu", s.mu)?;
        mu::not_used_d(o, "U", s.U)?;
        mu::not_used_c(o, "t", s.t)?;
        mu::not_used_c(o, "t0", s.t0)?;
        mu::not_used_c(o, "t1", s.t1)?;
        mu::not_used_c(o, "t2", s.t2)?;
        mu::not_used_c(o, "t'", s.tp)?;
        mu::not_used_c(o, "t0'", s.t0p)?;
        mu::not_used_c(o, "t1'", s.t1p)?;
        mu::not_used_c(o, "t2'", s.t2p)?;
        mu::not_used_c(o, "t''", s.tpp)?;
        mu::not_used_d(o, "V", s.V)?;
        mu::not_used_d(o, "V0", s.V0)?;
        mu::not_used_d(o, "V1", s.V1)?;
        mu::not_used_d(o, "V'", s.Vp)?;
        // Not reproduced from C: the remaining unused parameters.
        mu::not_used_d(o, "V2", s.V2)?;
        mu::not_used_d(o, "V0'", s.V0p)?;
        mu::not_used_d(o, "V1'", s.V1p)?;
        mu::not_used_d(o, "V2'", s.V2p)?;
        mu::not_used_d(o, "V''", s.Vpp)?;
        mu::not_used_j(o, "J'", s.JpAll, &s.Jp)?;
        mu::not_used_j(o, "J''", s.JppAll, &s.Jpp)?;
    } else {
        mu::print_val_d(o, "mu", &mut s.mu, 0.0);
        mu::print_val_d(o, "U", &mut s.U, 0.0);
        mu::input_hopp(o, s.t, &mut s.t0, "t0")?;
        mu::input_hopp(o, s.t, &mut s.t1, "t1")?;
        mu::input_hopp(o, s.t, &mut s.t2, "t2")?;
        mu::input_hopp(o, s.t, &mut s.t0p, "t0'")?;
        mu::input_hopp(o, s.t, &mut s.t1p, "t1'")?;
        mu::input_hopp(o, s.t, &mut s.t2p, "t2'")?;
        mu::input_coulomb_v(o, s.V, &mut s.V0, "V0")?;
        mu::input_coulomb_v(o, s.V, &mut s.V1, "V1")?;
        mu::input_coulomb_v(o, s.V, &mut s.V2, "V2")?;
        mu::input_coulomb_v(o, s.V, &mut s.V0p, "V0'")?;
        mu::input_coulomb_v(o, s.V, &mut s.V1p, "V1'")?;
        mu::input_coulomb_v(o, s.V, &mut s.V2p, "V2'")?;
        // Not reproduced from C: t', t'', V', V'' do not exist on this lattice.
        mu::not_used_c(o, "t'", s.tp)?;
        mu::not_used_c(o, "t''", s.tpp)?;
        mu::not_used_d(o, "V'", s.Vp)?;
        mu::not_used_d(o, "V''", s.Vpp)?;

        mu::not_used_j(o, "J0", s.J0All, &s.J0)?;
        mu::not_used_j(o, "J1", s.J1All, &s.J1)?;
        mu::not_used_j(o, "J2", s.J2All, &s.J2)?;
        mu::not_used_j(o, "J0'", s.J0pAll, &s.J0p)?;
        mu::not_used_j(o, "J1'", s.J1pAll, &s.J1p)?;
        mu::not_used_j(o, "J2'", s.J2pAll, &s.J2p)?;
        mu::not_used_j(o, "J'", s.JpAll, &s.Jp)?;
        mu::not_used_j(o, "J''", s.JppAll, &s.Jpp)?;
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
    set_sites_and_flags(s);

    // (4) Upper limit of the number of Transfer & Interaction (C counts 2 local terms per site
    // and omits the Gamma_y ones, overflowing its arrays; 3 here as in the corrected C build).
    let (mut ntrans_max, mut nintr_max);
    if s.model == "spin" {
        ntrans_max = s.nsite * (s.S2 + 1 + 2 * s.S2);
        nintr_max = s.NCell * (s.NsiteUC + 12) * (3 * s.S2 + 1) * (3 * s.S2 + 1);
    } else {
        ntrans_max = s.NCell * 2 * (3 * s.NsiteUC + 24);
        nintr_max = s.NCell * (s.NsiteUC + 4 * 12);
        if s.model == "kondo" {
            ntrans_max += s.nsite / 2 * (s.S2 + 1 + 2 * s.S2);
            nintr_max += s.nsite / 2 * (3 * s.S2 + 1) * (3 * s.S2 + 1);
        }
    }
    mu::malloc_interactions(s, ntrans_max, nintr_max);

    // (5) Set Transfer & Interaction
    let spin = s.model == "spin";
    for kcell in 0..s.NCell {
        let cell = s.Cell[kcell as usize];
        // (1) Local term
        let mut isite = s.NsiteUC * kcell;
        if s.model == "kondo" {
            isite += s.nsite / 2;
        }
        if spin {
            for isite_uc in 0..s.NsiteUC {
                mu::mag_field(s, s.S2, -s.h, -s.Gamma, -s.Gamma_y, isite + isite_uc);
                let d = s.D;
                mu::general_j(s, &d, s.S2, s.S2, isite + isite_uc, isite + isite_uc);
            }
        } else {
            for isite_uc in 0..s.NsiteUC {
                mu::hubbard_local(s, s.mu, -s.h, -s.Gamma, -s.Gamma_y, s.U, isite + isite_uc);
            }
            if s.model == "kondo" {
                let jsite = s.NsiteUC * kcell;
                for isite_uc in 0..s.NsiteUC {
                    // C used `isite + 3` here (every local spin coupled to conduction site 3).
                    let j = s.J;
                    mu::general_j(s, &j, 1, s.S2, isite + isite_uc, jsite + isite_uc);
                    mu::mag_field(s, s.S2, -s.h, -s.Gamma, -s.Gamma_y, jsite + isite_uc);
                }
            }
        }
        // (2)-(4) Intra-cell along W, L, H
        bond(s, cell, [0, 0, 0], [0, 1], s.J0, s.t0, s.V0);
        bond(s, cell, [0, 0, 0], [0, 2], s.J1, s.t1, s.V1);
        bond(s, cell, [0, 0, 0], [0, 3], s.J2, s.t2, s.V2);
        // (5)-(7) Intra-cell along L-H, H-W, W-L
        bond(s, cell, [0, 0, 0], [2, 3], s.J0p, s.t0p, s.V0p);
        bond(s, cell, [0, 0, 0], [3, 1], s.J1p, s.t1p, s.V1p);
        bond(s, cell, [0, 0, 0], [1, 2], s.J2p, s.t2p, s.V2p);
        // (8)-(10) Inter-cell along W, L, H
        bond(s, cell, [1, 0, 0], [1, 0], s.J0, s.t0, s.V0);
        bond(s, cell, [0, 1, 0], [2, 0], s.J1, s.t1, s.V1);
        bond(s, cell, [0, 0, 1], [3, 0], s.J2, s.t2, s.V2);
        // (11)-(13) Inter-cell along L-H, H-W, W-L
        bond(s, cell, [0, -1, 1], [3, 2], s.J0p, s.t0p, s.V0p);
        bond(s, cell, [1, 0, -1], [1, 3], s.J1p, s.t1p, s.V1p);
        bond(s, cell, [-1, 1, 0], [2, 1], s.J2p, s.t2p, s.V2p);
    }
    Ok(())
}

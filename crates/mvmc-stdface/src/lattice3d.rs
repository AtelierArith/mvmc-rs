//! Shared pieces of the three-dimensional lattices (`Orthorhombic.c`, `FCOrtho.c`,
//! `Pyrochlore.c`): the bond helper and the `lattice.xsf` / geometry epilogue.
//!
//! The C files repeat the same `if (spin) GeneralJ else { Hopping; Coulomb }` block for every
//! bond; [`bond`] is that block.
#![allow(non_snake_case)]

use crate::ccomplex::C64;
use crate::model_util as mu;
use crate::out::{Out, Res};
use crate::vals::StdIntList;

/// One bond: `StdFace_FindSite(.., iW, iL, iH, dW, dL, dH, isiteUC, jsiteUC, ..)` followed by
/// `GeneralJ(J, S2, S2)` for the spin model or `Hopping(Cphase * t)` and `Coulomb(V)` otherwise.
#[allow(clippy::too_many_arguments)]
pub(crate) fn bond(
    s: &mut StdIntList,
    cell: [i32; 3],
    d: [i32; 3],
    uc: [i32; 2],
    j: [[f64; 3]; 3],
    t: C64,
    v: f64,
) {
    let pair = mu::find_site(s, cell[0], cell[1], cell[2], d[0], d[1], d[2], uc[0], uc[1]);
    if s.model == "spin" {
        mu::general_j(s, &j, s.S2, s.S2, pair.isite, pair.jsite);
    } else {
        mu::hopping(s, pair.cphase * t, pair.isite, pair.jsite, &pair.dR);
        mu::coulomb(s, v, pair.isite, pair.jsite);
    }
}

/// Run a lattice body between the `fopen("lattice.xsf", "w")` of the C code and
/// `fclose; StdFace_PrintXSF; StdFace_PrintGeometry`.
///
/// If a check exits first, C has already created `lattice.xsf` (empty).
pub(crate) fn finish(
    o: &mut Out,
    s: &mut StdIntList,
    body: fn(&mut Out, &mut StdIntList) -> Res<()>,
) -> Res<()> {
    let result = body(o, s);
    if result.is_err() {
        o.write_file("lattice.xsf", "")?;
    }
    result?;
    mu::print_xsf(o, s)?;
    mu::print_geometry(o, s)
}

/// Section (3) common to the 3D lattices: number of sites and the local spin flags.
pub(crate) fn set_sites_and_flags(s: &mut StdIntList) {
    s.nsite = s.NsiteUC * s.NCell;
    if s.model == "kondo" {
        s.nsite *= 2;
    }
    s.locspinflag = vec![0; s.nsite as usize];
    if s.model == "spin" {
        s.locspinflag.iter_mut().for_each(|f| *f = s.S2);
    } else if s.model == "hubbard" {
        s.locspinflag.iter_mut().for_each(|f| *f = 0);
    } else {
        let half = (s.nsite / 2) as usize;
        for il in 0..half {
            s.locspinflag[il] = s.S2;
            s.locspinflag[il + half] = 0;
        }
    }
}

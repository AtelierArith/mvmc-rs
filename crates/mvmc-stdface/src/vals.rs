//! Port of `StdFace_vals.h` (`struct StdIntList`, mVMC branch) and `StdFace_ResetVals`.
//!
//! Conventions copied from C: an unspecified real input is NaN, an unspecified integer input is
//! [`NAN_I`] (`2147483647`) and an unspecified string is `"****"`. Field names keep the C
//! spelling (`J0All`, `Gamma_y`, ...) so the transcription can be audited line by line.
#![allow(non_snake_case)]

use crate::ccomplex::C64;

/// `NaN_i`: the "not specified" marker of integer inputs.
pub const NAN_I: i32 = 2147483647;
/// The "not specified" marker of string inputs.
pub const UNSET_STR: &str = "****";

/// `struct StdIntList` restricted to the mVMC (`_mVMC`) solver.
#[derive(Clone, Debug)]
pub struct StdIntList {
    /// `NaN_i` (always [`NAN_I`]).
    pub NaN_i: i32,
    /// `pi`.
    pub pi: f64,
    // --- LATTICE ---
    /// Lattice name (lower-cased).
    pub lattice: String,
    /// Lattice constant `a`.
    pub a: f64,
    /// `wlength`, `llength`, `hlength`.
    pub length: [f64; 3],
    /// Sites along the first axis.
    pub W: i32,
    /// Sites along the second axis.
    pub L: i32,
    /// Sites along the third axis.
    pub Height: i32,
    /// Unit direct lattice vectors.
    pub direct: [[f64; 3]; 3],
    /// Super-cell shape (`box` in C).
    pub box_: [[i32; 3]; 3],
    /// Inverse of `box` (times `NCell`).
    pub rbox: [[i32; 3]; 3],
    /// Number of unit cells in the super-cell.
    pub NCell: i32,
    /// Cell positions in fractional coordinates.
    pub Cell: Vec<[i32; 3]>,
    /// Sites per unit cell.
    pub NsiteUC: i32,
    /// Intra-cell site positions.
    pub tau: Vec<[f64; 3]>,
    // --- MODEL ---
    /// Model name.
    pub model: String,
    /// Chemical potential.
    pub mu: f64,
    /// Hopping `t`.
    pub t: C64,
    /// Hopping `t'`.
    pub tp: C64,
    /// Hopping `t0`.
    pub t0: C64,
    /// Hopping `t0'`.
    pub t0p: C64,
    /// Hopping `t0''`.
    pub t0pp: C64,
    /// Hopping `t1`.
    pub t1: C64,
    /// Hopping `t1'`.
    pub t1p: C64,
    /// Hopping `t1''`.
    pub t1pp: C64,
    /// Hopping `t2`.
    pub t2: C64,
    /// Hopping `t2'`.
    pub t2p: C64,
    /// Hopping `t2''`.
    pub t2pp: C64,
    /// Hopping `t''`.
    pub tpp: C64,
    /// On-site Coulomb `U`.
    pub U: f64,
    /// `V`.
    pub V: f64,
    /// `V'`.
    pub Vp: f64,
    /// `V0`.
    pub V0: f64,
    /// `V0'`.
    pub V0p: f64,
    /// `V0''`.
    pub V0pp: f64,
    /// `V1`.
    pub V1: f64,
    /// `V1'`.
    pub V1p: f64,
    /// `V1''`.
    pub V1pp: f64,
    /// `V2`.
    pub V2: f64,
    /// `V2'`.
    pub V2p: f64,
    /// `V2''`.
    pub V2pp: f64,
    /// `V''`.
    pub Vpp: f64,
    /// Isotropic `J`.
    pub JAll: f64,
    /// Isotropic `J'`.
    pub JpAll: f64,
    /// `J0`.
    pub J0All: f64,
    /// `J0'`.
    pub J0pAll: f64,
    /// `J0''`.
    pub J0ppAll: f64,
    /// `J1`.
    pub J1All: f64,
    /// `J1'`.
    pub J1pAll: f64,
    /// `J1''`.
    pub J1ppAll: f64,
    /// `J2`.
    pub J2All: f64,
    /// `J2'`.
    pub J2pAll: f64,
    /// `J2''`.
    pub J2ppAll: f64,
    /// `J''`.
    pub JppAll: f64,
    /// `Jx`, `Jxy`, ...
    pub J: [[f64; 3]; 3],
    /// `J'x`, ...
    pub Jp: [[f64; 3]; 3],
    /// `J0x`, ...
    pub J0: [[f64; 3]; 3],
    /// `J0'x`, ...
    pub J0p: [[f64; 3]; 3],
    /// `J0''x`, ...
    pub J0pp: [[f64; 3]; 3],
    /// `J1x`, ...
    pub J1: [[f64; 3]; 3],
    /// `J1'x`, ...
    pub J1p: [[f64; 3]; 3],
    /// `J1''x`, ...
    pub J1pp: [[f64; 3]; 3],
    /// `J2x`, ...
    pub J2: [[f64; 3]; 3],
    /// `J2'x`, ...
    pub J2p: [[f64; 3]; 3],
    /// `J2''x`, ...
    pub J2pp: [[f64; 3]; 3],
    /// `J''x`, ...
    pub Jpp: [[f64; 3]; 3],
    /// Single-ion anisotropy; only `D[2][2]` is used.
    pub D: [[f64; 3]; 3],
    /// Longitudinal field.
    pub h: f64,
    /// Transverse field (x).
    pub Gamma: f64,
    /// Transverse field (y).
    pub Gamma_y: f64,
    /// 4-spin term (unused).
    pub K: f64,
    // --- boundary phase ---
    /// `pi / 180`.
    pub pi180: f64,
    /// Boundary phases in degrees.
    pub phase: [f64; 3],
    /// `exp(i pi phase / 180)`.
    pub ExpPhase: [C64; 3],
    /// 1 where the boundary phase is 180 degrees.
    pub AntiPeriod: [i32; 3],
    // --- transfer / interaction lists ---
    /// Number of sites.
    pub nsite: i32,
    /// Local-spin flag per site.
    pub locspinflag: Vec<i32>,
    /// Transfer indices.
    pub transindx: Vec<[i32; 4]>,
    /// Transfer coefficients.
    pub trans: Vec<C64>,
    /// Whether `interall.def` is printed.
    pub Lintr: i32,
    /// InterAll indices.
    pub intrindx: Vec<[i32; 8]>,
    /// InterAll coefficients.
    pub intr: Vec<C64>,
    /// Whether `coulombintra.def` is printed.
    pub LCintra: i32,
    /// CoulombIntra indices.
    pub CintraIndx: Vec<[i32; 1]>,
    /// CoulombIntra coefficients.
    pub Cintra: Vec<f64>,
    /// Whether `coulombinter.def` is printed.
    pub LCinter: i32,
    /// CoulombInter indices.
    pub CinterIndx: Vec<[i32; 2]>,
    /// CoulombInter coefficients.
    pub Cinter: Vec<f64>,
    /// Whether `hund.def` is printed.
    pub LHund: i32,
    /// Hund indices.
    pub HundIndx: Vec<[i32; 2]>,
    /// Hund coefficients.
    pub Hund: Vec<f64>,
    /// Whether `exchange.def` is printed.
    pub LEx: i32,
    /// Exchange indices.
    pub ExIndx: Vec<[i32; 2]>,
    /// Exchange coefficients.
    pub Ex: Vec<f64>,
    /// Whether `pairlift.def` is printed.
    pub LPairLift: i32,
    /// PairLift indices.
    pub PLIndx: Vec<[i32; 2]>,
    /// PairLift coefficients.
    pub PairLift: Vec<f64>,
    /// Whether `pairhopp.def` is printed.
    pub LPairHopp: i32,
    /// PairHopp indices.
    pub PHIndx: Vec<[i32; 2]>,
    /// PairHopp coefficients.
    pub PairHopp: Vec<f64>,
    /// Boost flag (HPhi only; always 0 here).
    pub lBoost: i32,
    // --- calculation conditions ---
    /// Number of electrons.
    pub ncond: i32,
    /// 1 for grand-canonical models.
    pub lGC: i32,
    /// Local spin `2S`.
    pub S2: i32,
    /// Output mode name.
    pub outputmode: String,
    /// Output file header (`CDataFileHead`).
    pub CDataFileHead: String,
    /// Total `2Sz`.
    pub Sz2: i32,
    /// Numeric output mode.
    pub ioutputmode: i32,
    // --- Wannier90 keywords (accepted by the reader) ---
    /// `cutoff_t`.
    pub cutoff_t: f64,
    /// `cutoff_u`.
    pub cutoff_u: f64,
    /// `cutoff_j`.
    pub cutoff_j: f64,
    /// `cutoff_length_t`.
    pub cutoff_length_t: f64,
    /// `cutoff_length_U`.
    pub cutoff_length_U: f64,
    /// `cutoff_length_J`.
    pub cutoff_length_J: f64,
    /// `cutoff_tR`.
    pub cutoff_tR: [i32; 3],
    /// `cutoff_UR`.
    pub cutoff_UR: [i32; 3],
    /// `cutoff_JR`.
    pub cutoff_JR: [i32; 3],
    /// `cutoff_tVec`.
    pub cutoff_tVec: [[f64; 3]; 3],
    /// `cutoff_UVec`.
    pub cutoff_UVec: [[f64; 3]; 3],
    /// `cutoff_JVec`.
    pub cutoff_JVec: [[f64; 3]; 3],
    /// `lambda`.
    pub lambda: f64,
    /// `lambda_U`.
    pub lambda_U: f64,
    /// `lambda_J`.
    pub lambda_J: f64,
    /// `doublecounting`.
    pub double_counting_mode: String,
    /// `alpha`.
    pub alpha: f64,
    // --- mVMC modpara ---
    /// `CParaFileHead`.
    pub CParaFileHead: String,
    /// `NVMCCalMode`.
    pub NVMCCalMode: i32,
    /// `NLanczosMode`.
    pub NLanczosMode: i32,
    /// `NDataIdxStart`.
    pub NDataIdxStart: i32,
    /// `NDataQtySmp`.
    pub NDataQtySmp: i32,
    /// `NSPGaussLeg`.
    pub NSPGaussLeg: i32,
    /// `NMPTrans`.
    pub NMPTrans: i32,
    /// `NSROptItrStep`.
    pub NSROptItrStep: i32,
    /// `NSROptItrSmp`.
    pub NSROptItrSmp: i32,
    /// `DSROptRedCut`.
    pub DSROptRedCut: f64,
    /// `DSROptStaDel`.
    pub DSROptStaDel: f64,
    /// `DSROptStepDt`.
    pub DSROptStepDt: f64,
    /// `NVMCWarmUp`.
    pub NVMCWarmUp: i32,
    /// `NVMCInterval`.
    pub NVMCInterval: i32,
    /// `NVMCSample`.
    pub NVMCSample: i32,
    /// `NExUpdatePath`.
    pub NExUpdatePath: i32,
    /// `RndSeed`.
    pub RndSeed: i32,
    /// `NSplitSize`.
    pub NSplitSize: i32,
    /// `NSPStot`.
    pub NSPStot: i32,
    /// `NStore`.
    pub NStore: i32,
    /// `NSRCG`.
    pub NSRCG: i32,
    /// `ComplexType`.
    pub ComplexType: i32,
    // --- sub-lattice ---
    /// `Lsub`.
    pub Lsub: i32,
    /// `Wsub`.
    pub Wsub: i32,
    /// `Hsub`.
    pub Hsub: i32,
    /// Cells in the sub-lattice.
    pub NCellsub: i32,
    /// Sub-lattice shape.
    pub boxsub: [[i32; 3]; 3],
    /// Inverse of `boxsub` (times `NCellsub`).
    pub rboxsub: [[i32; 3]; 3],
    // --- 2-body part of the trial wavefunction ---
    /// Orbital index `[nsite][nsite]`.
    pub Orb: Vec<Vec<i32>>,
    /// Anti-periodic switch `[nsite][nsite]`.
    pub AntiOrb: Vec<Vec<i32>>,
    /// Number of independent orbital indices.
    pub NOrb: i32,
    /// Number of translation symmetries.
    pub NSym: i32,
}

impl StdIntList {
    /// `StdFace_ResetVals`: every input is "not specified".
    pub fn reset_vals() -> Self {
        let nan = f64::NAN;
        let nan_c = C64::real(nan);
        let nan_i = NAN_I;
        let nan33 = [[nan; 3]; 3];
        let mut d = [[0.0; 3]; 3];
        d[2][2] = nan;
        let pi = (-1.0f64).acos();
        Self {
            NaN_i: nan_i,
            pi,
            lattice: UNSET_STR.to_string(),
            a: nan,
            length: [nan; 3],
            W: nan_i,
            L: nan_i,
            Height: nan_i,
            direct: nan33,
            box_: [[nan_i; 3]; 3],
            rbox: [[0; 3]; 3],
            NCell: 0,
            Cell: Vec::new(),
            NsiteUC: 0,
            tau: Vec::new(),
            model: UNSET_STR.to_string(),
            mu: nan,
            t: nan_c,
            tp: nan_c,
            t0: nan_c,
            t0p: nan_c,
            t0pp: nan_c,
            t1: nan_c,
            t1p: nan_c,
            t1pp: nan_c,
            t2: nan_c,
            t2p: nan_c,
            t2pp: nan_c,
            tpp: nan_c,
            U: nan,
            V: nan,
            Vp: nan,
            V0: nan,
            V0p: nan,
            V0pp: nan,
            V1: nan,
            V1p: nan,
            V1pp: nan,
            V2: nan,
            V2p: nan,
            V2pp: nan,
            Vpp: nan,
            JAll: nan,
            JpAll: nan,
            J0All: nan,
            J0pAll: nan,
            J0ppAll: nan,
            J1All: nan,
            J1pAll: nan,
            J1ppAll: nan,
            J2All: nan,
            J2pAll: nan,
            J2ppAll: nan,
            JppAll: nan,
            J: nan33,
            Jp: nan33,
            J0: nan33,
            J0p: nan33,
            J0pp: nan33,
            J1: nan33,
            J1p: nan33,
            J1pp: nan33,
            J2: nan33,
            J2p: nan33,
            J2pp: nan33,
            Jpp: nan33,
            D: d,
            h: nan,
            Gamma: nan,
            Gamma_y: nan,
            K: nan,
            pi180: pi / 180.0,
            phase: [nan; 3],
            ExpPhase: [C64::real(0.0); 3],
            AntiPeriod: [0; 3],
            nsite: 0,
            locspinflag: Vec::new(),
            transindx: Vec::new(),
            trans: Vec::new(),
            Lintr: 0,
            intrindx: Vec::new(),
            intr: Vec::new(),
            LCintra: 0,
            CintraIndx: Vec::new(),
            Cintra: Vec::new(),
            LCinter: 0,
            CinterIndx: Vec::new(),
            Cinter: Vec::new(),
            LHund: 0,
            HundIndx: Vec::new(),
            Hund: Vec::new(),
            LEx: 0,
            ExIndx: Vec::new(),
            Ex: Vec::new(),
            LPairLift: 0,
            PLIndx: Vec::new(),
            PairLift: Vec::new(),
            LPairHopp: 0,
            PHIndx: Vec::new(),
            PairHopp: Vec::new(),
            lBoost: 0,
            ncond: nan_i,
            lGC: 0,
            S2: nan_i,
            outputmode: UNSET_STR.to_string(),
            CDataFileHead: UNSET_STR.to_string(),
            Sz2: nan_i,
            ioutputmode: 0,
            cutoff_t: nan,
            cutoff_u: nan,
            cutoff_j: nan,
            cutoff_length_t: nan,
            cutoff_length_U: nan,
            cutoff_length_J: nan,
            cutoff_tR: [nan_i; 3],
            cutoff_UR: [nan_i; 3],
            cutoff_JR: [nan_i; 3],
            cutoff_tVec: nan33,
            cutoff_UVec: nan33,
            cutoff_JVec: nan33,
            lambda: nan,
            lambda_U: nan,
            lambda_J: nan,
            double_counting_mode: UNSET_STR.to_string(),
            alpha: nan,
            CParaFileHead: UNSET_STR.to_string(),
            NVMCCalMode: nan_i,
            NLanczosMode: nan_i,
            NDataIdxStart: nan_i,
            NDataQtySmp: nan_i,
            NSPGaussLeg: nan_i,
            NSPStot: nan_i,
            NMPTrans: nan_i,
            NSROptItrStep: nan_i,
            NSROptItrSmp: nan_i,
            DSROptRedCut: nan,
            DSROptStaDel: nan,
            DSROptStepDt: nan,
            NVMCWarmUp: nan_i,
            NVMCInterval: nan_i,
            NVMCSample: nan_i,
            NExUpdatePath: nan_i,
            RndSeed: nan_i,
            NSplitSize: nan_i,
            NStore: nan_i,
            NSRCG: nan_i,
            ComplexType: nan_i,
            Lsub: nan_i,
            Wsub: nan_i,
            Hsub: nan_i,
            NCellsub: 0,
            boxsub: [[nan_i; 3]; 3],
            rboxsub: [[0; 3]; 3],
            Orb: Vec::new(),
            AntiOrb: Vec::new(),
            NOrb: 0,
            NSym: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_vals_marks_everything_unspecified() {
        let s = StdIntList::reset_vals();
        assert!(s.a.is_nan() && s.t.re.is_nan());
        assert_eq!(s.L, NAN_I);
        assert_eq!(s.model, "****");
        assert!(s.D[2][2].is_nan());
        assert_eq!(s.D[0][0], 0.0);
        assert_eq!(s.pi, std::f64::consts::PI);
    }
}

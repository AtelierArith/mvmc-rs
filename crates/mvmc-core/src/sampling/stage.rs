//! Pfaffian stages of the real normal-mode sampler as a swappable backend (issue #434).
//!
//! The Metropolis sampler `vmc_make_sample_real` needs five Pfaffian/inverse operations:
//! the rank-one and rank-two ratio of a proposal, the rank-one and rank-two inverse update of an
//! accepted proposal, and the periodic recomputation (`CalculateMAll`). Everything else
//! (candidate generation, projection counters, the acceptance draw, SFMT) is host code in a
//! fixed order. [`RealPfStage`] is the seam between the two: [`CpuStage`] runs the existing
//! flat CPU kernels in place (the production path, unchanged arithmetic), and a device-resident
//! stage keeps the inverses on the accelerator and only exchanges moves and `NQP`-vectors with the
//! host (`crate::device_sampler`).
//!
//! The seam changes no RNG draw: the draws happen in the host code around the stage calls.
//! [`RealPfStage::decide`] is an observational hook with a default that returns the computed
//! decision; a teacher-forced replay overrides it to follow a reference run.

use crate::pfaffian::calc_m_all_real;
use crate::sampling::metropolis::MetropolisDecision;
use crate::sampling::updates::{
    calculate_new_pf_m2_real_flat, calculate_new_pf_m_two2_real_flat, update_m_all_real_flat,
    update_m_all_two_real_flat,
};
use crate::state::{InvMColMajor, SlaterElmFlat, ThreadedPfaPackWorkspace};

/// Sizes shared by every stage call of one walker.
#[derive(Debug, Clone, Copy)]
pub struct StageGeom {
    /// `Nsite`.
    pub n_site: usize,
    /// Electrons per spin.
    pub n_elec: usize,
    /// First QP of this rank.
    pub qp_start: usize,
    /// One past the last QP of this rank.
    pub qp_end: usize,
    /// Stride between QP planes of the host inverse table (`n_size^2 + 1`).
    pub inv_stride: usize,
}

/// Host tables a stage may read or update in place. A device-resident stage ignores `inv_m`
/// (it holds the inverses itself) and writes `pf_m` only after a recomputation.
pub struct StageTables<'a> {
    /// Slater orbital table `[NQP][2 Nsite][2 Nsite]`.
    pub slater_elm: &'a SlaterElmFlat<f64>,
    /// Host inverse table, mVMC convention (`invM = -X^-1`).
    pub inv_m: &'a mut InvMColMajor<f64>,
    /// Pfaffians per QP.
    pub pf_m: &'a mut Vec<f64>,
    /// PfaPack workspace of the CPU path.
    pub pool: &'a ThreadedPfaPackWorkspace,
}

/// Error of a device stage (the CPU stage never fails).
pub type StageResult<T> = Result<T, String>;

/// Backend of the Pfaffian operations of the real normal-mode sampler.
pub trait RealPfStage {
    /// Called once after the initial configuration and tables exist. `ele_idx` is the working
    /// configuration, `t.pf_m` the initial Pfaffians. A device stage builds its resident
    /// inverses here.
    fn begin(
        &mut self,
        _g: &StageGeom,
        _t: &mut StageTables<'_>,
        _ele_idx: &[i64],
    ) -> StageResult<()> {
        Ok(())
    }

    /// Ratios `pf_new[qp]` of the one-electron move already applied to `ele_idx` (candidate
    /// configuration): electron slot `ma` of `spin`.
    fn propose_hop(
        &mut self,
        g: &StageGeom,
        t: &mut StageTables<'_>,
        ele_idx: &[i64],
        ma: usize,
        spin: u8,
        pf_new: &mut [f64],
    ) -> StageResult<()>;

    /// Apply the accepted one-electron move (the candidate is still in `ele_idx`); afterwards
    /// the current Pfaffians equal `pf_new`.
    fn accept_hop(
        &mut self,
        g: &StageGeom,
        t: &mut StageTables<'_>,
        ele_idx: &[i64],
        ma: usize,
        spin: u8,
        pf_new: &[f64],
    ) -> StageResult<()>;

    /// Ratios of the two-electron exchange move already applied to `ele_idx`.
    fn propose_exchange(
        &mut self,
        g: &StageGeom,
        t: &mut StageTables<'_>,
        ele_idx: &[i64],
        slots: [(usize, u8); 2],
        pf_new: &mut [f64],
    ) -> StageResult<()>;

    /// Apply the accepted exchange move; `ra_old`, `rb_old` are the sites before the move.
    fn accept_exchange(
        &mut self,
        g: &StageGeom,
        t: &mut StageTables<'_>,
        ele_idx: &[i64],
        slots: [(usize, u8); 2],
        old_sites: [usize; 2],
    ) -> StageResult<()>;

    /// Recompute the Pfaffians and inverses of `ele_idx`. `Ok(true)` reports a numeric failure
    /// (the CPU `calc_m_all_real(..).is_err()`); the tables are then unspecified as on the CPU.
    /// On success `t.pf_m` holds the new Pfaffians.
    fn recompute(
        &mut self,
        g: &StageGeom,
        t: &mut StageTables<'_>,
        ele_idx: &[i64],
    ) -> StageResult<bool>;

    /// Final accept/reject of a computed decision. The default follows the computed decision;
    /// a teacher-forced stage returns the reference decision and records the flip.
    fn decide(&mut self, decision: &MetropolisDecision) -> bool {
        decision.accepted
    }
}

/// The in-place CPU kernels (the production path).
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuStage;

impl RealPfStage for CpuStage {
    fn propose_hop(
        &mut self,
        g: &StageGeom,
        t: &mut StageTables<'_>,
        ele_idx: &[i64],
        ma: usize,
        spin: u8,
        pf_new: &mut [f64],
    ) -> StageResult<()> {
        calculate_new_pf_m2_real_flat(
            ma,
            spin,
            pf_new,
            ele_idx,
            t.slater_elm,
            t.inv_m.as_slice(),
            g.inv_stride,
            t.pf_m,
            g.qp_start,
            g.qp_end,
            g.n_site,
            g.n_elec,
        );
        Ok(())
    }

    fn accept_hop(
        &mut self,
        g: &StageGeom,
        t: &mut StageTables<'_>,
        ele_idx: &[i64],
        ma: usize,
        spin: u8,
        pf_new: &[f64],
    ) -> StageResult<()> {
        update_m_all_real_flat(
            ma,
            spin,
            ele_idx,
            t.slater_elm,
            t.inv_m.as_mut_slice(),
            g.inv_stride,
            t.pf_m,
            g.qp_start,
            g.qp_end,
            g.n_site,
            g.n_elec,
        );
        t.pf_m.copy_from_slice(pf_new);
        Ok(())
    }

    fn propose_exchange(
        &mut self,
        g: &StageGeom,
        t: &mut StageTables<'_>,
        ele_idx: &[i64],
        slots: [(usize, u8); 2],
        pf_new: &mut [f64],
    ) -> StageResult<()> {
        calculate_new_pf_m_two2_real_flat::<false>(
            slots[0].0,
            slots[0].1,
            slots[1].0,
            slots[1].1,
            pf_new,
            ele_idx,
            t.slater_elm,
            t.inv_m.as_slice(),
            g.inv_stride,
            t.pf_m,
            g.qp_start,
            g.qp_end,
            g.n_site,
            g.n_elec,
        );
        Ok(())
    }

    fn accept_exchange(
        &mut self,
        g: &StageGeom,
        t: &mut StageTables<'_>,
        ele_idx: &[i64],
        slots: [(usize, u8); 2],
        old_sites: [usize; 2],
    ) -> StageResult<()> {
        update_m_all_two_real_flat(
            slots[0].0,
            slots[0].1,
            slots[1].0,
            slots[1].1,
            old_sites[0],
            old_sites[1],
            ele_idx,
            t.slater_elm,
            t.inv_m.as_mut_slice(),
            g.inv_stride,
            t.pf_m,
            g.qp_start,
            g.qp_end,
            g.n_site,
            g.n_elec,
        );
        Ok(())
    }

    fn recompute(
        &mut self,
        g: &StageGeom,
        t: &mut StageTables<'_>,
        ele_idx: &[i64],
    ) -> StageResult<bool> {
        Ok(calc_m_all_real(
            ele_idx,
            t.slater_elm,
            t.inv_m,
            t.pf_m,
            g.qp_start,
            g.qp_end,
            g.n_site,
            g.n_elec,
            t.pool,
        )
        .is_err())
    }
}

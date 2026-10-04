//! Phase 4.3.7 — initial-sample generator.
//!
//! Port target: `make_initial_sample!` in
//! `MVMCOptimizers.jl/src/vmc_sampling.jl` (~line 1199). The RNG draw
//! order matches upstream: for every local-spin site we draw
//! `(gen_rand_mod(n_elec), spin_coin)`; for itinerant electrons we
//! draw `gen_rand_mod(n_site)` until a free non-local site is found.

#![allow(
    clippy::result_unit_err,
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::items_after_test_module
)]

use crate::reducer::{Reducer, SingleProcessReducer};
use mvmc_expert_parsers::ExpertModeData;
use sfmt19937::Sfmt19937Rng;

use crate::pfaffian::{calc_m_all_fsz_complex, calc_m_all_fsz_real, CalcMAllError};
use crate::sampling::projection::{init_loc_spn, make_proj_cnt};
use crate::state::{ThreadedPfaPackWorkspace, VmcOptimizationState};

/// A local kernel failure or a comm1 peer failure, without inventing a local
/// numerical defect on ranks whose owned QP slice succeeded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SamplingInitializationError {
    /// This rank's owned QP kernel failed.
    Local(CalcMAllError),
    /// This rank succeeded but another comm1 rank failed.
    PeerFailure,
}

impl From<CalcMAllError> for SamplingInitializationError {
    fn from(error: CalcMAllError) -> Self {
        Self::Local(error)
    }
}

impl std::fmt::Display for SamplingInitializationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Local(error) => error.fmt(f),
            Self::PeerFailure => write!(f, "sampling initialization failed on another comm1 rank"),
        }
    }
}

impl std::error::Error for SamplingInitializationError {}

pub(super) fn coordinate_initialization_result<R: Reducer + ?Sized>(
    result: Result<(), CalcMAllError>,
    reducer: &R,
) -> Result<(), SamplingInitializationError> {
    if reducer.sampling_any_failure(result.is_err()) {
        Err(result
            .err()
            .map(SamplingInitializationError::Local)
            .unwrap_or(SamplingInitializationError::PeerFailure))
    } else {
        Ok(())
    }
}

fn serial_error(error: SamplingInitializationError) -> CalcMAllError {
    match error {
        SamplingInitializationError::Local(error) => error,
        SamplingInitializationError::PeerFailure => unreachable!("serial reducer has no peers"),
    }
}

/// Draw one normal placement and its projection counters.
///
/// This helper does not factorize or retry. Shared C-style validation belongs
/// to [`super::normal_initial::make_initial_sample_normal_with_info`].
pub fn make_initial_sample(
    ele_idx: &mut [i64],
    ele_cfg: &mut [i64],
    ele_num: &mut [i64],
    ele_proj_cnt: &mut [i64],
    data: &ExpertModeData,
    loc_spn: &[i64],
    rng: &mut Sfmt19937Rng,
) -> Result<(), ()> {
    let n_site = data.modpara.nsite.max(0) as usize;
    let n_elec = data.modpara.nelec.max(0) as usize;
    let n_site2 = 2 * n_site;

    for slot in ele_idx.iter_mut() {
        *slot = -1;
    }
    for slot in ele_cfg.iter_mut() {
        *slot = -1;
    }

    // Local spin sites.
    for ri in 0..n_site {
        if loc_spn.get(ri).copied().unwrap_or(0) != 1 {
            continue;
        }
        loop {
            let mi = crate::sampling::driver::trace::draw_mod(rng, n_elec as u32) as usize;
            let r = crate::sampling::driver::trace::draw_real2(rng);
            let si = if r < 0.5 { 0 } else { 1 };
            if ele_idx[mi + si * n_elec] == -1 {
                ele_cfg[ri + si * n_site] = mi as i64;
                ele_idx[mi + si * n_elec] = ri as i64;
                break;
            }
        }
    }

    // Itinerant electrons.
    for si in 0..2 {
        for mi in 0..n_elec {
            if ele_idx[mi + si * n_elec] != -1 {
                continue;
            }
            loop {
                let ri = crate::sampling::driver::trace::draw_mod(rng, n_site as u32) as usize;
                if ele_cfg[ri + si * n_site] == -1 && loc_spn.get(ri).copied().unwrap_or(0) != 1 {
                    ele_cfg[ri + si * n_site] = mi as i64;
                    ele_idx[mi + si * n_elec] = ri as i64;
                    break;
                }
            }
        }
    }

    // Electron-number array.
    for rsi in 0..n_site2 {
        ele_num[rsi] = if ele_cfg[rsi] < 0 { 0 } else { 1 };
    }
    // Projection counts.
    make_proj_cnt(ele_proj_cnt, ele_num, data);
    // Placement only; this is not a native factorization INFO or retry result.
    Ok(())
}

/// Convenience wrapper that initialises `loc_spn` from `data.locspin_terms`
/// before delegating to [`make_initial_sample`].
pub fn make_initial_sample_init_loc_spn(
    ele_idx: &mut [i64],
    ele_cfg: &mut [i64],
    ele_num: &mut [i64],
    ele_proj_cnt: &mut [i64],
    loc_spn: &mut [i64],
    data: &ExpertModeData,
    rng: &mut Sfmt19937Rng,
) -> Result<(), ()> {
    init_loc_spn(loc_spn, data);
    make_initial_sample(ele_idx, ele_cfg, ele_num, ele_proj_cnt, data, loc_spn, rng)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn make_initial_sample_respects_n_elec_and_loc_spn() {
        let mut data = ExpertModeData::new();
        data.modpara.nsite = 4;
        data.modpara.nelec = 2;
        let loc_spn = vec![1, 0, 1, 0];
        let mut ele_idx = vec![0_i64; 4];
        let mut ele_cfg = vec![0_i64; 8];
        let mut ele_num = vec![0_i64; 8];
        let mut proj = vec![0_i64; 0];
        let mut rng = Sfmt19937Rng::new(11272);
        make_initial_sample(
            &mut ele_idx,
            &mut ele_cfg,
            &mut ele_num,
            &mut proj,
            &data,
            &loc_spn,
            &mut rng,
        )
        .expect("layout succeeds");
        // Local spin sites must hold an electron in some spin sector.
        for (ri, flag) in loc_spn.iter().enumerate() {
            if *flag == 1 {
                assert!(
                    ele_num[ri] + ele_num[ri + 4] == 1,
                    "local spin site {} must host exactly one electron",
                    ri
                );
            }
        }
        // Every electron index must point to a non-negative site.
        for v in &ele_idx {
            assert!(*v >= 0);
        }
    }
}

/// The integer configuration-generation stage shared by Julia's FSZ initializers.
pub fn generate_initial_fsz_configuration(
    ele_idx: &mut [i64],
    ele_cfg: &mut [i64],
    ele_num: &mut [i64],
    ele_proj_cnt: &mut [i64],
    ele_spn: &mut [i64],
    data: &ExpertModeData,
    loc_spn: &[i64],
    rng: &mut Sfmt19937Rng,
) -> Result<(), ()> {
    let n_site = data.modpara.nsite.max(0) as usize;
    let n_elec = data.modpara.nelec.max(0) as usize;
    let n_size = 2 * n_elec;
    let n_site2 = 2 * n_site;
    let tmp_two_sz = if data.modpara.two_sz == -1 {
        0
    } else {
        data.modpara.two_sz / 2
    };

    for slot in ele_idx.iter_mut() {
        *slot = -1;
    }
    for slot in ele_spn.iter_mut() {
        *slot = -1;
    }
    for slot in ele_cfg.iter_mut() {
        *slot = -1;
    }

    for mi in 0..n_size {
        ele_spn[mi] = if (mi as i64) < data.modpara.nelec + tmp_two_sz {
            0
        } else {
            1
        };
    }

    for ri in 0..n_site {
        if loc_spn.get(ri).copied().unwrap_or(0) == 1 {
            loop {
                let mi = crate::sampling::driver::trace::draw_mod(rng, n_size as u32) as usize;
                if ele_idx[mi] == -1 {
                    let si = ele_spn[mi] as usize;
                    ele_cfg[ri + si * n_site] = mi as i64;
                    ele_idx[mi] = ri as i64;
                    break;
                }
            }
        }
    }

    for mi in 0..n_size {
        if ele_idx[mi] == -1 {
            let si = ele_spn[mi] as usize;
            loop {
                let ri = crate::sampling::driver::trace::draw_mod(rng, n_site as u32) as usize;
                if ele_cfg[ri + si * n_site] == -1 && loc_spn.get(ri).copied().unwrap_or(0) == 0 {
                    ele_cfg[ri + si * n_site] = mi as i64;
                    ele_idx[mi] = ri as i64;
                    break;
                }
            }
        }
    }

    for rsi in 0..n_site2 {
        ele_num[rsi] = if ele_cfg[rsi] == -1 { 0 } else { 1 };
    }
    make_proj_cnt(ele_proj_cnt, ele_num, data);
    Ok(())
}

/// Julia's complex FSZ initializer, including all 101 Pfaffian-validation attempts.
/// Failed attempts consume their original SFMT draws and retain the last state.
pub fn make_initial_sample_fsz(
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    rng: &mut Sfmt19937Rng,
    qp_start: usize,
    qp_end: usize,
    pool: &ThreadedPfaPackWorkspace,
) -> Result<(), CalcMAllError> {
    make_initial_sample_fsz_with_reducer(
        data,
        state,
        rng,
        qp_start,
        qp_end,
        pool,
        &SingleProcessReducer,
    )
    .map_err(serial_error)
}

/// FSZ initializer with comm1-coordinated retry status and owned QP range.
pub fn make_initial_sample_fsz_with_reducer<R: Reducer + ?Sized>(
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    rng: &mut Sfmt19937Rng,
    qp_start: usize,
    qp_end: usize,
    pool: &ThreadedPfaPackWorkspace,
    reducer: &R,
) -> Result<(), SamplingInitializationError> {
    let n_site = data.modpara.nsite as usize;
    let n_elec = data.modpara.nelec as usize;
    init_loc_spn(&mut state.workspace.loc_spn, data);
    for attempt in 0..=100 {
        let c = &mut state.electron_config;
        generate_initial_fsz_configuration(
            &mut c.tmp_ele_idx,
            &mut c.tmp_ele_cfg,
            &mut c.tmp_ele_num,
            &mut c.tmp_ele_proj_cnt,
            &mut c.tmp_ele_spn,
            data,
            &state.workspace.loc_spn,
            rng,
        )
        .expect("FSZ configuration generation does not fail");
        let mat = &mut state.slater_matrix;
        let result = calc_m_all_fsz_complex(
            &c.tmp_ele_idx,
            &c.tmp_ele_spn,
            &mat.slater_elm,
            &mut mat.inv_m,
            &mut mat.pf_m,
            qp_start,
            qp_end,
            n_site,
            n_elec,
            pool,
        );
        if !reducer.sampling_any_failure(result.is_err()) {
            return Ok(());
        }
        if attempt == 100 {
            // A successful local slice can still have a failed peer. Report
            // a distinct coordination error without inventing a local QP defect.
            return Err(result
                .err()
                .map(SamplingInitializationError::Local)
                .unwrap_or(SamplingInitializationError::PeerFailure));
        }
    }
    unreachable!("the last failed attempt returns its error")
}

/// Julia's real FSZ initialization, including Pfaffian validation and up to
/// 101 attempts. Every failed attempt consumes its original SFMT draws;
/// the last attempted configuration is retained when the limit is reached.
pub fn make_initial_sample_fsz_real(
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    rng: &mut Sfmt19937Rng,
    qp_start: usize,
    qp_end: usize,
    pool: &ThreadedPfaPackWorkspace,
) -> Result<(), CalcMAllError> {
    make_initial_sample_fsz_real_with_reducer(
        data,
        state,
        rng,
        qp_start,
        qp_end,
        pool,
        &SingleProcessReducer,
    )
    .map_err(serial_error)
}

/// Real FSZ initializer with comm1-coordinated retry status.
pub fn make_initial_sample_fsz_real_with_reducer<R: Reducer + ?Sized>(
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    rng: &mut Sfmt19937Rng,
    qp_start: usize,
    qp_end: usize,
    pool: &ThreadedPfaPackWorkspace,
    reducer: &R,
) -> Result<(), SamplingInitializationError> {
    let n_site = data.modpara.nsite as usize;
    let n_elec = data.modpara.nelec as usize;
    init_loc_spn(&mut state.workspace.loc_spn, data);
    for attempt in 0..=100 {
        let c = &mut state.electron_config;
        generate_initial_fsz_configuration(
            &mut c.tmp_ele_idx,
            &mut c.tmp_ele_cfg,
            &mut c.tmp_ele_num,
            &mut c.tmp_ele_proj_cnt,
            &mut c.tmp_ele_spn,
            data,
            &state.workspace.loc_spn,
            rng,
        )
        .expect("FSZ configuration generation does not fail");
        let result = calc_m_all_fsz_real(
            &c.tmp_ele_idx,
            &c.tmp_ele_spn,
            &mut state.slater_matrix,
            qp_start,
            qp_end,
            n_site,
            n_elec,
            pool,
        );
        if !reducer.sampling_any_failure(result.is_err()) {
            return Ok(());
        }
        if attempt == 100 {
            return Err(result
                .err()
                .map(SamplingInitializationError::Local)
                .unwrap_or(SamplingInitializationError::PeerFailure));
        }
    }
    unreachable!("the last failed attempt returns its error")
}

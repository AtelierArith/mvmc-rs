//! C shared normal initializer: placement, complex validation, integer comm1 INFO.
//!
//! Unlike FSZ initialization, even the real sampler validates with the complex
//! master table before its separate real setup. C checks exhaustion after the
//! 101st completed call and before testing INFO > 0.

use std::ops::Range;

use mvmc_expert_parsers::ExpertModeData;
use sfmt19937::Sfmt19937Rng;

use crate::pfaffian::{calc_m_all_complex_native_info, CalcMAllError};
use crate::state::{SlaterMatrixData, ThreadedPfaPackWorkspace, VmcOptimizationState};

/// Additional supported-storage preflight for normal callers, including burn.
/// This coordinated unsupported-error stage is NOT native factor INFO/MAX.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct NormalSamplerShape {
    pub n_site: usize,
    pub n_elec: usize,
    pub n_size: usize,
    pub n_qp_full: usize,
    pub qp_range: Range<usize>,
    pub n_vmc_sample: usize,
    pub n_in_step: usize,
    pub n_out_warm: usize,
    pub n_out_burn: usize,
}

/// Return checked dimensions before caller arithmetic, cloning or indexing.
pub(crate) fn preflight_normal_sampler(
    data: &ExpertModeData,
    state: &VmcOptimizationState,
    loc_spn: &[i64],
    owned_range: impl FnOnce(usize) -> Range<usize>,
    require_real: bool,
    mut max_info: impl FnMut(i32) -> Result<i32, String>,
) -> Result<NormalSamplerShape, NormalInitializationError> {
    let local = (|| {
        let n_site = usize::try_from(data.modpara.nsite).map_err(|_| "negative Nsite")?;
        let ne = usize::try_from(data.modpara.nelec).map_err(|_| "negative Ne")?;
        let size = ne.checked_mul(2).ok_or("Nsize overflow")?;
        let site2 = n_site.checked_mul(2).ok_or("Nsite2 overflow")?;
        let n_vmc_sample = usize::try_from(data.modpara.nvmc_sample.max(0))
            .map_err(|_| "normal sample count domain")?;
        let warmup = usize::try_from(data.modpara.nvmc_warmup.max(0))
            .map_err(|_| "normal warmup count domain")?;
        let interval = usize::try_from(data.modpara.nvmc_interval.max(0))
            .map_err(|_| "normal interval count domain")?;
        let n_out_warm = warmup
            .checked_add(n_vmc_sample)
            .ok_or("normal warmup/sample overflow")?;
        let n_out_burn = n_vmc_sample
            .checked_add(1)
            .ok_or("normal burn/sample overflow")?;
        let n_in_step = interval
            .checked_mul(n_site.max(1))
            .ok_or("normal inner step overflow")?;
        let matrix = &state.slater_matrix;
        let config = &state.electron_config;
        let qp = matrix.slater_elm.n_qp_full();
        if ne > n_site || size >= i32::MAX as usize || qp >= i32::MAX as usize {
            return Err("normal dimensions/native INFO domain");
        }
        if config.tmp_ele_idx.len() != size
            || config.tmp_ele_cfg.len() != site2
            || config.tmp_ele_num.len() != site2
            || config.tmp_ele_proj_cnt.len() != data.projection_layout().n_proj
            || !config.tmp_ele_spn.is_empty()
            || loc_spn.len() != n_site
            || loc_spn.iter().any(|&value| value != 0 && value != 1)
            || loc_spn.iter().filter(|&&value| value == 1).count() > size
            || state.workspace.n_size != size
            || matrix.slater_elm.n_site2() != site2
            || matrix.inv_m.n_size() != size
            || matrix.inv_m.n_qp_full() != qp
            || matrix.pf_m.len() != qp
        {
            return Err("normal scratch/master/real storage shape");
        }
        // all_complex=true deliberately allocates empty real fields. Do not
        // reject that supported constructor for a complex-only caller.
        if require_real
            && (matrix.slater_elm_real.n_site2() != site2
                || matrix.slater_elm_real.n_qp_full() != qp
                || matrix.inv_m_real.n_size() != size
                || matrix.inv_m_real.n_qp_full() != qp
                || matrix.pf_m_real.len() != qp)
        {
            return Err("normal real storage shape");
        }
        let burn_size = [
            config.tmp_ele_idx.len(),
            config.tmp_ele_cfg.len(),
            config.tmp_ele_num.len(),
            config.tmp_ele_proj_cnt.len(),
            config.tmp_ele_spn.len(),
        ]
        .into_iter()
        .try_fold(0usize, |sum, len| sum.checked_add(len))
        .ok_or("burn storage size overflow")?;
        if config.counter[9] != 0 && config.burn_ele_idx.len() < burn_size {
            return Err("combined burn storage shape");
        }
        let qp_range = owned_range(qp);
        if qp_range.start > qp_range.end || qp_range.end > qp {
            return Err("normal owned QP range");
        }
        Ok(NormalSamplerShape {
            n_site,
            n_elec: ne,
            n_size: size,
            n_qp_full: qp,
            qp_range,
            n_vmc_sample,
            n_in_step,
            n_out_warm,
            n_out_burn,
        })
    })()
    .map_err(|reason| NormalInitializationError::Precondition(reason.into()));
    coordinate_supported_result(local, &mut max_info)
}

fn coordinate_supported_result<T>(
    local: Result<T, NormalInitializationError>,
    mut max_info: impl FnMut(i32) -> Result<i32, String>,
) -> Result<T, NormalInitializationError> {
    // Numeric INFO is not reduced here; only unsupported typed-error presence.
    let status = max_info(if local.is_ok() { 0 } else { i32::MAX })
        .map_err(NormalInitializationError::Coordination)?;
    if status == i32::MAX {
        return Err(local
            .err()
            .unwrap_or(NormalInitializationError::PeerPrecondition));
    }
    if status != 0 || local.is_err() {
        return Err(NormalInitializationError::Coordination(
            "unsupported-error MAX returned an inconsistent status".into(),
        ));
    }
    local
}

/// Coordinate typed setup errors BEFORE overlap reduction on every rank.
/// A signed native numeric INFO remains unchanged and is ignored by C callers.
pub(crate) fn coordinate_normal_setup(
    local: Result<i32, CalcMAllError>,
    max_info: impl FnMut(i32) -> Result<i32, String>,
) -> Result<i32, NormalInitializationError> {
    coordinate_supported_result(local.map_err(NormalInitializationError::Kernel), max_info)
}

/// Last actual local and comm1 INFO, including negative native codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NormalInitializationInfo {
    /// Number of completed placement/complex-validation/collective attempts.
    pub attempts: usize,
    /// Actual status from this rank's owned QP slice.
    pub local_info: i32,
    /// MPI_MAX status for the chain, not a boolean all-rank failure flag.
    pub comm1_info: i32,
}

/// Typed propagation instead of C's process-wide abort or an empty success.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NormalInitializationError {
    /// Placement cannot satisfy a validated input precondition.
    Placement,
    /// Checked input/storage shape is invalid before any RNG draw.
    Precondition(String),
    /// A peer has an unsupported input/kernel error, not a native INFO.
    PeerPrecondition,
    /// Invalid Rust/kernel precondition that is not a native numeric INFO.
    Kernel(CalcMAllError),
    /// Integer comm1 status coordination failed.
    Coordination(String),
    /// All 101 calls completed; C aborts even if the last INFO is nonpositive.
    RetryLimit(NormalInitializationInfo),
}

impl std::fmt::Display for NormalInitializationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Placement => write!(f, "normal sampling initialization placement failed"),
            Self::Precondition(error) => write!(f, "normal initialization precondition: {error}"),
            Self::PeerPrecondition => write!(
                f,
                "normal initialization precondition failed on a comm1 peer"
            ),
            Self::Kernel(error) => error.fmt(f),
            Self::Coordination(error) => write!(f, "normal initialization comm1 INFO: {error}"),
            Self::RetryLimit(info) => write!(
                f,
                "normal initialization exhausted {} calls (local INFO {}, comm1 INFO {})",
                info.attempts, info.local_info, info.comm1_info
            ),
        }
    }
}

impl std::error::Error for NormalInitializationError {}

/// Execute C's loop/control order with one actual validation and collective.
///
/// The callbacks are used to bind the real kernel and comm1 implementation;
/// they are not alternate RNG or numerical algorithms.
fn initialize_loop(
    mut attempt: impl FnMut() -> Result<i32, NormalInitializationError>,
    mut max_info: impl FnMut(i32) -> Result<i32, String>,
) -> Result<NormalInitializationInfo, NormalInitializationError> {
    for attempts in 1..=101 {
        let local = attempt();
        // Reserved only for unsupported Rust/precondition errors. Supported
        // model dimensions/owned-QP counts are checked below i32::MAX before
        // entry, so a defined native INFO cannot collide with this sentinel.
        // Every rank still enters the same collective, including a failing one.
        let local_info = local.as_ref().copied().unwrap_or(i32::MAX);
        let comm1_info = max_info(local_info).map_err(NormalInitializationError::Coordination)?;
        if comm1_info == i32::MAX {
            return Err(local
                .err()
                .unwrap_or(NormalInitializationError::PeerPrecondition));
        }
        if let Err(error) = local {
            return Err(NormalInitializationError::Coordination(format!(
                "integer MAX omitted local unsupported error: {error}"
            )));
        }
        let info = NormalInitializationInfo {
            attempts,
            local_info,
            comm1_info,
        };
        if attempts > 100 {
            return Err(NormalInitializationError::RetryLimit(info));
        }
        if comm1_info <= 0 {
            return Ok(info);
        }
    }
    unreachable!("the 101st completed call returns a typed exhaustion error")
}

/// Draw and validate a shared normal configuration using the complex master.
///
/// The caller supplies its owned QP range and actual integer comm1 MAX callback.
/// No boolean reducer is substituted, and no collective executes inside a worker.
/// Real setup and log-IP recovery remain distinct sampler caller operations.
#[allow(clippy::too_many_arguments)]
pub fn make_initial_sample_normal_with_info(
    ele_idx: &mut [i64],
    ele_cfg: &mut [i64],
    ele_num: &mut [i64],
    ele_proj_cnt: &mut [i64],
    data: &ExpertModeData,
    loc_spn: &[i64],
    rng: &mut Sfmt19937Rng,
    matrix: &mut SlaterMatrixData,
    qp_range: Range<usize>,
    pool: &ThreadedPfaPackWorkspace,
    mut max_info: impl FnMut(i32) -> Result<i32, String>,
) -> Result<NormalInitializationInfo, NormalInitializationError> {
    // A coordinated preflight precedes placement/RNG. It is not reported as
    // native factor INFO. All ranks use the same MAX stage even if one shape
    // is invalid; do not return early and strand its peers.
    let shape = (|| {
        let n_site = usize::try_from(data.modpara.nsite).map_err(|_| "negative Nsite")?;
        let n_elec = usize::try_from(data.modpara.nelec).map_err(|_| "negative Ne")?;
        if n_elec > n_site {
            return Err("unsupported normal electron/site dimensions");
        }
        let n_size = n_elec.checked_mul(2).ok_or("Nsize overflow")?;
        let n_site2 = n_site.checked_mul(2).ok_or("Nsite2 overflow")?;
        let n_qp = matrix.slater_elm.n_qp_full();
        if n_size >= i32::MAX as usize || n_qp >= i32::MAX as usize {
            return Err("native INFO dimension domain");
        }
        if ele_idx.len() != n_size || ele_cfg.len() != n_site2
            || ele_num.len() != n_site2 || loc_spn.len() != n_site
            || ele_proj_cnt.len() != data.projection_layout().n_proj
            || matrix.slater_elm.n_site2() != n_site2
            || matrix.inv_m.n_size() != n_size || matrix.inv_m.n_qp_full() != n_qp
            || matrix.pf_m.len() != n_qp
            || pool.n_size() != n_size
            || qp_range.start > qp_range.end || qp_range.end > n_qp
            // This is the caller's projected LocSpn array, not raw LocSpin
            // definitions: init_loc_spn maps exactly raw==1 to1, others to0.
            || loc_spn.iter().any(|&flag| flag != 0 && flag != 1)
            || loc_spn.iter().filter(|&&flag| flag == 1).count() > n_size
        {
            return Err("inconsistent normal initialization buffers/range");
        }
        Ok((n_site, n_elec))
    })();
    let (n_site, n_elec) = coordinate_supported_result(
        shape.map_err(|error| NormalInitializationError::Precondition(error.into())),
        &mut max_info,
    )?;
    initialize_loop(
        || {
            super::initial::make_initial_sample(
                ele_idx,
                ele_cfg,
                ele_num,
                ele_proj_cnt,
                data,
                loc_spn,
                rng,
            )
            .map_err(|()| NormalInitializationError::Placement)?;
            calc_m_all_complex_native_info(
                ele_idx,
                &matrix.slater_elm,
                &mut matrix.inv_m,
                &mut matrix.pf_m,
                qp_range.start,
                qp_range.end,
                n_site,
                n_elec,
                pool,
            )
            .map_err(NormalInitializationError::Kernel)
        },
        max_info,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_preflight_rejects_nonzero_typed_status_before_placement() {
        for reduced in [0, -3, 1, i32::MAX] {
            let mut data = ExpertModeData::new();
            data.modpara.nsite = 2;
            data.modpara.nelec = 1;
            let mut state = VmcOptimizationState::zeros(
                2,
                1,
                data.projection_layout().n_proj,
                0,
                0,
                1,
                true,
                false,
            );
            let pool = ThreadedPfaPackWorkspace::new(2, 1);
            let configuration = state.electron_config.clone();
            let matrix = state.slater_matrix.clone();
            let mut rng = Sfmt19937Rng::new(12395);
            let raw = rng.state_snapshot();
            let count = rng.words_consumed();
            let mut future = [0; 624];
            rng.dump_rand32(&mut future);
            let mut calls = Vec::new();
            let config = &mut state.electron_config;
            let result = make_initial_sample_normal_with_info(
                &mut config.tmp_ele_idx,
                &mut config.tmp_ele_cfg,
                &mut config.tmp_ele_num,
                &mut config.tmp_ele_proj_cnt,
                &data,
                &[0, 0],
                &mut rng,
                &mut state.slater_matrix,
                0..0,
                &pool,
                |status| {
                    calls.push(status);
                    Ok(reduced)
                },
            );
            if reduced == 0 {
                assert_eq!(result.unwrap().attempts, 1);
                assert_eq!(calls, [0, 0]);
                continue;
            }
            if reduced == i32::MAX {
                assert_eq!(result, Err(NormalInitializationError::PeerPrecondition));
            } else {
                assert!(matches!(
                    result,
                    Err(NormalInitializationError::Coordination(_))
                ));
            }
            assert_eq!(calls, [0]);
            assert_eq!(rng.state_snapshot(), raw);
            assert_eq!(rng.words_consumed(), count);
            let mut after = [0; 624];
            rng.dump_rand32(&mut after);
            assert_eq!(after, future);
            assert_eq!(state.electron_config, configuration);
            assert_eq!(state.slater_matrix, matrix);
        }
    }

    #[test]
    fn malformed_shared_pool_is_rejected_before_rng_or_configuration_mutation() {
        let mut data = ExpertModeData::new();
        data.modpara.nsite = 2;
        data.modpara.nelec = 1;
        let mut state = VmcOptimizationState::zeros(
            2,
            1,
            data.projection_layout().n_proj,
            0,
            2,
            1,
            true,
            false,
        );
        let wrong_pool = ThreadedPfaPackWorkspace::new(4, 1);
        let config_before = state.electron_config.clone();
        let matrix_before = state.slater_matrix.clone();
        let mut rng = Sfmt19937Rng::new(12395);
        let before = rng.state_snapshot();
        let count = rng.words_consumed();
        let mut future = [0; 624];
        rng.dump_rand32(&mut future);
        let mut statuses = Vec::new();
        let config = &mut state.electron_config;
        let result = make_initial_sample_normal_with_info(
            &mut config.tmp_ele_idx,
            &mut config.tmp_ele_cfg,
            &mut config.tmp_ele_num,
            &mut config.tmp_ele_proj_cnt,
            &data,
            &[0, 0],
            &mut rng,
            &mut state.slater_matrix,
            0..2,
            &wrong_pool,
            |status| {
                statuses.push(status);
                Ok(status)
            },
        );
        assert!(matches!(
            result,
            Err(NormalInitializationError::Precondition(_))
        ));
        assert_eq!(statuses, [i32::MAX]);
        assert_eq!(rng.state_snapshot(), before);
        assert_eq!(rng.words_consumed(), count);
        let mut after = [0; 624];
        rng.dump_rand32(&mut after);
        assert_eq!(after, future);
        assert_eq!(state.electron_config, config_before);
        assert_eq!(state.slater_matrix, matrix_before);
    }

    #[test]
    fn complex_constructor_empty_real_buffers_are_supported() {
        let mut data = ExpertModeData::new();
        data.modpara.nsite = 2;
        data.modpara.nelec = 1;
        let state = VmcOptimizationState::zeros(
            2,
            1,
            data.projection_layout().n_proj,
            0,
            2,
            1,
            true,
            false,
        );
        assert!(state.slater_matrix.pf_m_real.is_empty());
        assert_eq!(
            preflight_normal_sampler(&data, &state, &[0, 0], |_| 0..2, false, Ok).map(|_| ()),
            Ok(())
        );
        assert!(matches!(
            preflight_normal_sampler(&data, &state, &[0, 0], |_| 0..2, true, Ok),
            Err(NormalInitializationError::Precondition(_))
        ));
        // The counter is a fixed [i64;10], not a potentially short Vec.
        assert_eq!(state.electron_config.counter.len(), 10);
    }

    #[test]
    fn burn_combined_storage_is_checked_before_restore() {
        let mut data = ExpertModeData::new();
        data.modpara.nsite = 2;
        data.modpara.nelec = 1;
        let mut state = VmcOptimizationState::zeros(
            2,
            1,
            data.projection_layout().n_proj,
            0,
            2,
            1,
            false,
            false,
        );
        state.electron_config.counter[9] = 1;
        state.electron_config.burn_ele_idx.clear();
        let before = state.electron_config.clone();
        let mut statuses = Vec::new();
        assert!(matches!(
            preflight_normal_sampler(
                &data,
                &state,
                &[0, 0],
                |_| 0..2,
                true,
                |status| {
                    statuses.push(status);
                    Ok(status)
                }
            ),
            Err(NormalInitializationError::Precondition(_))
        ));
        assert_eq!(statuses, [i32::MAX]);
        assert_eq!(state.electron_config, before);
    }

    #[test]
    fn both_normal_callers_reject_overflow_and_burn_shape_without_rng_draws() {
        for complex in [false, true] {
            for invalid in [
                "ne-domain",
                "site-domain",
                "negative-ne",
                "burn-shape",
                "scratch-shape",
            ] {
                let mut data = ExpertModeData::new();
                data.modpara.nsite = 2;
                data.modpara.nelec = 1;
                let mut state = VmcOptimizationState::zeros(
                    2,
                    1,
                    data.projection_layout().n_proj,
                    0,
                    2,
                    1,
                    complex,
                    false,
                );
                match invalid {
                    "ne-domain" => data.modpara.nelec = i64::MAX,
                    "site-domain" => data.modpara.nsite = i64::MAX,
                    "negative-ne" => data.modpara.nelec = -1,
                    "burn-shape" => {
                        state.electron_config.counter[9] = 1;
                        state.electron_config.burn_ele_idx.clear();
                    }
                    "scratch-shape" => {
                        state.electron_config.tmp_ele_cfg.pop();
                    }
                    _ => unreachable!(),
                }
                let before_config = state.electron_config.clone();
                let mut rng = Sfmt19937Rng::new(12395);
                let before = rng.state_snapshot();
                let count = rng.words_consumed();
                let mut future = [0; 624];
                rng.dump_rand32(&mut future);
                let result = if complex {
                    crate::sampling::driver::vmc_make_sample(&data, &mut state, &mut rng)
                } else {
                    crate::sampling::driver::vmc_make_sample_real(&data, &mut state, &mut rng)
                };
                assert!(
                    matches!(result, Err(NormalInitializationError::Precondition(_))),
                    "{complex}/{invalid}: {result:?}"
                );
                assert_eq!(rng.state_snapshot(), before);
                assert_eq!(rng.words_consumed(), count);
                let mut after = [0; 624];
                rng.dump_rand32(&mut after);
                assert_eq!(after, future);
                assert_eq!(state.electron_config, before_config);
            }
        }
    }

    #[test]
    fn separate_setup_ignores_signed_numeric_info_without_retry() {
        for native_info in [-5, -3, 0, 2] {
            let mut coordinated = Vec::new();
            assert_eq!(
                coordinate_normal_setup(Ok(native_info), |status| {
                    coordinated.push(status);
                    Ok(status)
                }),
                Ok(native_info)
            );
            // This stage coordinates typed errors, not C's numeric INFO.
            assert_eq!(coordinated, [0]);
        }
    }

    #[test]
    fn separate_setup_typed_error_coordinates_before_return() {
        let error = CalcMAllError::InputShape {
            reason: "owned QP range",
        };
        let mut coordinated = Vec::new();
        assert_eq!(
            coordinate_normal_setup(Err(error.clone()), |status| {
                coordinated.push(status);
                Ok(status)
            }),
            Err(NormalInitializationError::Kernel(error))
        );
        assert_eq!(coordinated, [i32::MAX]);
        assert_eq!(
            coordinate_normal_setup(Ok(-3), |_| Ok(i32::MAX)),
            Err(NormalInitializationError::PeerPrecondition)
        );
    }

    #[test]
    fn shared_loop_matches_independent_c_caller_controls() {
        const ROOT: &str = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/issue274_caller_control"
        );
        for name in [
            "success",
            "retry",
            "negative",
            "peer-retry",
            "call101-success",
            "exhaustion",
        ] {
            let text = std::fs::read_to_string(format!("{ROOT}/{name}.stdout")).unwrap();
            let mut local = Vec::new();
            let mut maxima = Vec::new();
            let mut expected = None;
            for line in text.lines() {
                let fields: Vec<_> = line.split_whitespace().collect();
                match fields[0] {
                    "complex" => {
                        let call: usize = fields[1].parse().unwrap();
                        assert_eq!(call, local.len() + 1);
                        assert_eq!(fields[3].parse::<usize>().unwrap(), call * 2);
                        local.push(fields[2].parse::<i32>().unwrap());
                    }
                    "integer-max" => {
                        assert_eq!(fields.len(), 5);
                        let ordinal = fields[1].parse::<usize>().unwrap();
                        let rank_local = fields[2].parse::<i32>().unwrap();
                        let peer = fields[3].parse::<i32>().unwrap();
                        let maximum = fields[4].parse::<i32>().unwrap();
                        assert_eq!(ordinal, maxima.len() + 1);
                        assert_eq!(ordinal, local.len());
                        assert_eq!(rank_local, local[ordinal - 1]);
                        assert_eq!(maximum, rank_local.max(peer));
                        maxima.push(maximum);
                    }
                    "returned" => expected = Some((false, fields[2].parse::<usize>().unwrap())),
                    "exhausted" => expected = Some((true, fields[1].parse::<usize>().unwrap())),
                    "real-setup" => assert_eq!(fields[1], "-7"),
                    "continued-after-real-setup" => {}
                    other => panic!("unexpected C fixture record: {other}"),
                }
            }
            if maxima.is_empty() {
                maxima.clone_from(&local);
            }
            assert_eq!(maxima.len(), local.len());
            let mut calls = 0;
            let mut reductions = 0;
            let result = initialize_loop(
                || {
                    let info = local[calls];
                    calls += 1;
                    Ok(info)
                },
                |info| {
                    assert_eq!(info, local[reductions]);
                    let max = maxima[reductions];
                    reductions += 1;
                    Ok(max)
                },
            );
            let (exhausted, expected_calls) = expected.unwrap();
            assert_eq!(calls, expected_calls, "{name}");
            // Rust reducer callback invocations, not native MPI collective
            // counts: C size-one cases skip MPI_Allreduce altogether.
            assert_eq!(reductions, expected_calls, "{name}");
            let info = NormalInitializationInfo {
                attempts: expected_calls,
                local_info: *local.last().unwrap(),
                comm1_info: *maxima.last().unwrap(),
            };
            if exhausted {
                assert_eq!(
                    result,
                    Err(NormalInitializationError::RetryLimit(info)),
                    "{name}"
                );
            } else {
                assert_eq!(result, Ok(info), "{name}");
            }
        }
    }

    #[test]
    fn negative_info_is_not_a_boolean_retry() {
        let info = initialize_loop(|| Ok(-3), Ok).unwrap();
        assert_eq!(
            info,
            NormalInitializationInfo {
                attempts: 1,
                local_info: -3,
                comm1_info: -3
            }
        );
    }

    #[test]
    fn maximum_status_controls_retry_not_local_status() {
        let mut calls = 0;
        let info = initialize_loop(
            || Ok(0),
            |_| {
                calls += 1;
                Ok(if calls == 1 { 4 } else { 0 })
            },
        )
        .unwrap();
        assert_eq!(calls, 2);
        assert_eq!(info.attempts, 2);
    }

    #[test]
    fn call101_exhausts_even_if_its_status_succeeds() {
        let mut calls = 0;
        let error = initialize_loop(
            || {
                calls += 1;
                Ok(if calls == 101 { 0 } else { 1 })
            },
            Ok,
        )
        .unwrap_err();
        assert_eq!(calls, 101);
        assert_eq!(
            error,
            NormalInitializationError::RetryLimit(NormalInitializationInfo {
                attempts: 101,
                local_info: 0,
                comm1_info: 0
            })
        );
    }

    #[test]
    fn local_precondition_error_still_enters_collective() {
        let mut received = Vec::new();
        let error = initialize_loop(
            || Err(NormalInitializationError::Placement),
            |info| {
                received.push(info);
                Ok(info)
            },
        )
        .unwrap_err();
        assert_eq!(received, [i32::MAX]);
        assert_eq!(error, NormalInitializationError::Placement);
    }

    #[test]
    fn successful_rank_gets_typed_peer_precondition() {
        let error = initialize_loop(|| Ok(0), |_| Ok(i32::MAX)).unwrap_err();
        assert_eq!(error, NormalInitializationError::PeerPrecondition);
    }
}

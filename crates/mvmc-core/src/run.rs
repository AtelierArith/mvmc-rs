//! Top-level entry points.
//!
//! Port targets:
//! * `vmc_para_opt.jl` -> `vmc_para_opt`
//! * `vmc_phys_cal.jl` -> `vmc_phys_cal`
//! * `run_para_opt_from_namelist.jl` -> `run_para_opt_from_namelist`
//! * `initial_params.jl` -> `read_initial_def`
//!
//! BIT-PARITY CRITICAL: preserve the upstream phase order documented in
//! `run_para_opt_from_namelist.jl:65-100`:
//!     init_gen_rand -> InitParameter -> ReadInitParameter
//!         -> ReadInputParameters -> SyncModifiedParameter -> InitQPWeight.

use std::fs;
use std::io;
use std::path::Path;

use mvmc_expert_parsers::parse_expert_mode_files;
use mvmc_expert_parsers::utils::parameter_init::{
    init_parameter, n_slater, sync_modified_parameter,
};
use mvmc_expert_parsers::utils::qp_weight::init_qp_weight;
use mvmc_expert_parsers::utils::read_input_parameters::read_input_parameters;
use mvmc_expert_parsers::ExpertModeData;
use num_complex::Complex64;
sfmt19937::Sfmt19937Rng;

use crate::average::{weight_average_sr_opt, weight_average_sr_opt_real, weight_average_we};
use crate::counter::reduce_counter;
use crate::io::{output_data, output_opt_data};
use crate::observables::{calculate_local_energy, clear_phys_quantity};
use crate::reducer::{Reducer, SingleProcessReducer};
use crate::sampling::driver::{vmc_make_sample, vmc_make_sample_fsz, vmc_make_sample_real};
use crate::slater_update::{update_slater_elm, update_slater_elm_fsz};
use crate::sr::{stochastic_opt_complex, stochastic_opt_real};
use crate::state::VmcOptimizationState;
use crate::sync::sync_modified_parameter as sync_modified;

/// C-compatible fallback seed used when `modpara.rnd_seed <= 0`. Mirrors
/// `run_para_opt_from_namelist.jl` and the C reference.
pub const FALLBACK_SEED: i64 = 11272;

/// Run `nsteps` SR steps starting from `data` and the seeded `rng`.
///
/// Mirrors `vmc_para_opt!` for the Phase-4 cut: real-mode driver +
/// real-mode SR step. Complex models go through the complex driver and
/// the complex SR solver. RBM / BackFlow / FSZ paths return
/// `Err("unsupported")` until the corresponding kernels land.
pub fn vmc_para_opt<R: Reducer + ?Sized>(
    data: &mut ExpertModeData,
    state: &mut VmcOptimizationState,
    rng: &mut Sfmt19937Rng,
    output_dir: Option<&Path>,
    reducer: &R,
) -> Result<(), String> {
    let n_steps = data.modpara.nsr_opt_itr_step.max(0) as usize;
    let n_proj = data.gutzwiller_terms.len() + data.jastrow_terms.len();
    let n_orb = n_slater(data);
    let n_para = n_proj + n_orb;
    data.ensure_optimization_flags(n_para);
    let all_complex = data.modpara.complex_flag != 0
        || data.gutzwiller_terms.iter().any(|t| t.is_complex)
        || data.jastrow_terms.iter().any(|t| t.is_complex)
        || data.orbital_terms.iter().any(|t| t.is_complex);
    let i_flg_general = data.i_flg_orbital_general;
    let use_fsz = i_flg_general != 0;

    for step in 0..n_steps {
        // 1. Slater table refresh.
        if use_fsz {
            update_slater_elm_fsz(data, state);
        } else {
            update_slater_elm(data, state);
        }
        // 2. Sampler.
        let stats = if use_fsz {
            vmc_make_sample_fsz(data, state, rng)
        } else if !all_complex {
            vmc_make_sample_real(data, state, rng)
        } else {
            vmc_make_sample(data, state, rng)
        };
        if stats.saved == 0 {
            return Err(format!("vmc_para_opt: no samples saved at step {step}"));
        }

        // 3. Main accumulator.
        clear_phys_quantity(state);
        accumulate_observables(data, state, all_complex, use_fsz);

        // 4. Weighted averages + counter reduction.
        weight_average_we(state);
        if all_complex {
            weight_average_sr_opt(state);
        } else {
            weight_average_sr_opt_real(state);
        }
        reduce_counter(state, reducer);

        // 5. Output.
        output_data(data, state, step, output_dir).map_err(|e| e.to_string())?;

        // 6. SR update.
        let info = if all_complex {
            stochastic_opt_complex(data, state)
        } else {
            stochastic_opt_real(data, state)
        };
        if info != 0 {
            tracing::warn!(
                step = step,
                "vmc_para_opt: SR factorisation failed; continuing without parameter update"
            );
        }

        // 7. Sync modified parameters.
        sync_modified(data, reducer);
    }

    output_opt_data(data, output_dir).map_err(|e| e.to_string())?;
    Ok(())
}

/// Convenience entry point mirroring `run_para_opt_from_namelist`.
pub fn run_para_opt_from_namelist<P: AsRef<Path>>(
    namelist_path: P,
    nsteps: usize,
    seed: Option<i64>,
    output_dir: Option<&Path>,
) -> Result<RunSummary, String> {
    let mut data = parse_expert_mode_files(&namelist_path).map_err(|e| e.to_string())?;

    // 1. Seed the RNG (the parser leaves `optimization_flags` empty so
    //    `init_parameter` defaults every Slater slot to optimised).
    let actual_seed = match seed {
        Some(s) => s,
        None => {
            if data.modpara.rnd_seed > 0 {
                data.modpara.rnd_seed
            } else {
                FALLBACK_SEED
            }
        }
    };
    let mut rng = Sfmt19937Rng::new(actual_seed as u32);

    // 2. Random Slater seeding.
    init_parameter(&mut data, &mut rng);

    // 3. Optional initial.def overlay (C ReadInitParameter phase).
    let initial = namelist_path
        .as_ref()
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("initial.def");
    if initial.is_file() {
        read_initial_def(&mut data, &initial).map_err(|e| e.to_string())?;
    }

    // 4. In*.def overlay (no-op in Phase-4 stub).
    read_input_parameters(&mut data, &namelist_path);

    // 5. Sync (Slater rescale + Gutzwiller/Jastrow shift).
    sync_modified_parameter(&mut data);

    // 6. Quantum projection weights.
    init_qp_weight(&mut data);

    // 7. Cap optimisation length so tests stay fast.
    data.modpara.nsr_opt_itr_step = nsteps as i64;

    // 8. Build state.
    let n_site = data.modpara.nsite.max(0) as usize;
    let n_elec = data.modpara.nelec.max(0) as usize;
    let n_proj = data.gutzwiller_terms.len() + data.jastrow_terms.len();
    let n_orb = n_slater(&data);
    let n_para = n_proj + n_orb;
    let n_sp = data.modpara.nsp_gauss_leg.max(1) as usize;
    let n_mp = data.modpara.nmp_trans.unsigned_abs() as usize;
    let n_opt = data.n_qp_opt_trans.max(1) as usize;
    let n_qp_full = n_sp * n_mp * n_opt;
    let n_vmc_sample = data.modpara.nvmc_sample.max(0) as usize;
    let all_complex =
        data.modpara.complex_flag != 0 || data.orbital_terms.iter().any(|t| t.is_complex);
    let mut state = VmcOptimizationState::zeros(
        n_site,
        n_elec,
        n_proj,
        n_para,
        n_qp_full,
        n_vmc_sample,
        all_complex,
        data.i_flg_orbital_general != 0,
    );

    let reducer = SingleProcessReducer;
    vmc_para_opt(&mut data, &mut state, &mut rng, output_dir, &reducer)?;

    // Read back the final energy per site from zvo_out.dat, mirroring
    // Julia's `run_para_opt_from_namelist` return value.
    let final_energy_per_site = output_dir.and_then(|dir| {
        let content = std::fs::read_to_string(dir.join("zvo_out.dat")).ok()?;
        let last = content.lines().rfind(|l| !l.trim().is_empty())?;
        let e: f64 = last.split_whitespace().next()?.parse().ok()?;
        if n_site > 0 {
            Some(e / n_site as f64)
        } else {
            None
        }
    });

    Ok(RunSummary {
        nsteps,
        nsite: n_site,
        output_dir: output_dir.map(|p| p.to_path_buf()),
        final_energy_per_site,
    })
}

/// Summary returned by [`run_para_opt_from_namelist`].
///
/// Mirrors the `NamedTuple` returned by Julia's `run_para_opt_from_namelist`.
#[derive(Debug, Clone)]
pub struct RunSummary {
    /// Number of SR steps executed.
    pub nsteps: usize,
    /// Number of lattice sites (`Nsite` in `modpara.def`).
    pub nsite: usize,
    /// Output directory the run wrote to (when `Some`).
    pub output_dir: Option<std::path::PathBuf>,
    /// Total energy of the last SR step divided by `nsite`.
    /// `None` when no output directory was provided or when
    /// `zvo_out.dat` could not be read.
    pub final_energy_per_site: Option<f64>,
}

/// `read_initial_def!(data, path)` mirror. Phase-4 cut: covers the
/// non-RBM, non-OptTrans case used by the upstream `examples/inputs/*`
/// fixtures.
pub fn read_initial_def<P: AsRef<Path>>(data: &mut ExpertModeData, path: P) -> io::Result<()> {
    let content = fs::read_to_string(path)?;
    let floats: Vec<f64> = content
        .split_whitespace()
        .filter_map(|tok| tok.parse::<f64>().ok())
        .collect();
    let n_gutz = data.gutzwiller_terms.len();
    let n_jast = data.jastrow_terms.len();
    let n_proj = n_gutz + n_jast;
    let n_orb = data.modpara.n_orbital_idx.max(0) as usize;
    let expected = 6 + 3 * (n_proj + n_orb);
    if floats.len() < expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "initial.def too short: got {}, expected {}",
                floats.len(),
                expected
            ),
        ));
    }
    let mut idx = 6_usize;
    for term in data.gutzwiller_terms.iter_mut() {
        term.value = Complex64::new(floats[idx], floats[idx + 1]);
        idx += 3;
    }
    for term in data.jastrow_terms.iter_mut() {
        term.value = Complex64::new(floats[idx], floats[idx + 1]);
        idx += 3;
    }
    let mut slater = vec![Complex64::new(0.0, 0.0); n_orb];
    for v in slater.iter_mut().take(n_orb) {
        *v = Complex64::new(floats[idx], floats[idx + 1]);
        idx += 3;
    }
    for term in data.orbital_terms.iter_mut() {
        if term.idx >= 0 && (term.idx as usize) < n_orb {
            term.value = slater[term.idx as usize];
        }
    }
    Ok(())
}

fn accumulate_observables(
    data: &ExpertModeData,
    state: &mut VmcOptimizationState,
    all_complex: bool,
    use_fsz: bool,
) {
    let n_site = data.modpara.nsite.max(0) as usize;
    let n_elec = data.modpara.nelec.max(0) as usize;
    let n_size = 2 * n_elec;
    let n_qp_full = state.slater_matrix.pf_m.len();
    let n_vmc_sample = data.modpara.nvmc_sample.max(0) as usize;
    let n_proj = data.gutzwiller_terms.len() + data.jastrow_terms.len();
    let sr_opt_size = state.sr_opt.sr_opt_size;
    let n_orb_total = sr_opt_size.saturating_sub(1 + n_proj);
    let pool = crate::state::ThreadedPfaPackWorkspace::new(n_size, 1);
    let mut slater_derivative_scratch = crate::slater_derivative::SlaterDerivativeScratch::new();

    for sample in 0..n_vmc_sample {
        let ele_idx = state.electron_config.ele_idx_slice(sample).to_vec();
        if ele_idx.iter().all(|&v| v == 0) || ele_idx.iter().all(|&v| v < 0) {
            continue;
        }
        let ele_cfg = state.electron_config.ele_cfg_slice(sample).to_vec();
        let ele_num = state.electron_config.ele_num_slice(sample).to_vec();
        let ele_spn = if use_fsz {
            state.electron_config.ele_spn_slice(sample).to_vec()
        } else {
            Vec::new()
        };
        let ele_proj_cnt = if n_proj > 0 {
            state.electron_config.ele_proj_cnt_slice(sample).to_vec()
        } else {
            Vec::new()
        };

        // Refresh Pfaffian for the saved walker.
        let info = if use_fsz {
            crate::pfaffian::calc_m_all_fsz_complex(
                &ele_idx,
                &ele_spn,
                &state.slater_matrix.slater_elm,
                &mut state.slater_matrix.inv_m,
                &mut state.slater_matrix.pf_m,
                0,
                n_qp_full,
                n_site,
                n_elec,
                &pool,
            )
            .err()
        } else if all_complex {
            crate::pfaffian::calc_m_all_complex(
                &ele_idx,
                &state.slater_matrix.slater_elm,
                &mut state.slater_matrix.inv_m,
                &mut state.slater_matrix.pf_m,
                0,
                n_qp_full,
                n_site,
                n_elec,
                &pool,
            )
            .err()
        } else {
            crate::pfaffian::calc_m_all_real(
                &ele_idx,
                &state.slater_matrix.slater_elm_real,
                &mut state.slater_matrix.inv_m_real,
                &mut state.slater_matrix.pf_m_real,
                0,
                n_qp_full,
                n_site,
                n_elec,
                &pool,
            )
            .err()
        };
        if info.is_some() {
            continue;
        }
        if !all_complex {
            for qp in 0..n_qp_full {
                let real_plane = state.slater_matrix.inv_m_real.qp_matrix_slice(qp);
                for col in 0..n_size {
                    for row in 0..n_size {
                        let value = real_plane[row + col * n_size];
                        state
                            .slater_matrix
                            .inv_m
                            .set(qp, row, col, Complex64::new(value, 0.0));
                    }
                }
                state.slater_matrix.pf_m[qp] =
                    Complex64::new(state.slater_matrix.pf_m_real[qp], 0.0);
            }
        }
        let ip = if all_complex {
            crate::observables::calculate_ip_complex(&state.slater_matrix.pf_m, 0, n_qp_full, data)
        } else {
            Complex64::new(
                crate::observables::calculate_ip_real(
                    &state.slater_matrix.pf_m_real,
                    0,
                    n_qp_full,
                    data,
                ),
                0.0,
            )
        };
        if ip.norm() < 1.0e-100 {
            continue;
        }
        let w = 1.0;
        let e = if use_fsz {
            crate::observables::calculate_local_energy_fsz(
                ip,
                data,
                state,
                &ele_idx,
                &ele_cfg,
                &ele_num,
                &ele_proj_cnt,
                &ele_spn,
            )
        } else {
            calculate_local_energy(ip, data, state, &ele_idx, &ele_cfg, &ele_num, &ele_proj_cnt)
        };
        let sz = crate::observables::calculate_sz(&ele_num, n_site);

        state.energy.wc += Complex64::new(w, 0.0);
        state.energy.etot += Complex64::new(w, 0.0) * e;
        state.energy.etot2 += Complex64::new(w, 0.0) * e.conj() * e;
        state.energy.sztot += Complex64::new(w * sz, 0.0);
        state.energy.sztot2 += Complex64::new(w * sz * sz, 0.0);

        // SR `O` vector — projection diff fills the leading block.
        for slot in state.sr_opt.sr_opt_o.iter_mut() {
            *slot = Complex64::new(0.0, 0.0);
        }
        crate::observables::set_projection_diff(&mut state.sr_opt.sr_opt_o, &ele_proj_cnt, n_proj);
        let slater_offset = 2 * (1 + n_proj);
        if n_orb_total > 0 && slater_offset < state.sr_opt.sr_opt_o.len() {
            let n_copy = (2 * n_orb_total).min(state.sr_opt.sr_opt_o.len() - slater_offset);
            let slater_o = &mut state.sr_opt.sr_opt_o[slater_offset..slater_offset + n_copy];
            if use_fsz {
                crate::slater_derivative::slater_elm_diff_fsz_with_scratch(
                    slater_o,
                    ip,
                    &ele_idx,
                    &ele_spn,
                    data,
                    &state.slater_matrix,
                    &mut slater_derivative_scratch,
                );
            } else {
                crate::slater_derivative::slater_elm_diff_with_scratch(
                    slater_o,
                    ip,
                    &ele_idx,
                    data,
                    &state.slater_matrix,
                    &mut slater_derivative_scratch,
                );
            }
        }

        if all_complex {
            crate::observables::calculate_oo(
                &mut state.sr_opt.sr_opt_oo,
                &mut state.sr_opt.sr_opt_ho,
                &state.sr_opt.sr_opt_o,
                w,
                e,
                sr_opt_size,
            );
        } else {
            for i in 0..sr_opt_size {
                state.sr_opt.sr_opt_o_real[i] = state.sr_opt.sr_opt_o[2 * i].re;
            }
            crate::observables::calculate_oo_real(
                &mut state.sr_opt.sr_opt_oo_real,
                &mut state.sr_opt.sr_opt_ho_real,
                &state.sr_opt.sr_opt_o_real,
                w,
                e.re,
                sr_opt_size,
            );
        }
    }
}

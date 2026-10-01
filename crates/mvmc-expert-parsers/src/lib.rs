//! Expert Mode `.def` parsers + [`ExpertModeData`] container.
//!
//! Port target: `extern/Julia-mVMC/MVMCExpertModeParsers.jl/src/`.
//!
//! Phase 3 status: the parsers, types and orchestration needed to
//! round-trip the four upstream `examples/inputs/*` namelists are
//! implemented. The remaining upstream modules (RBM, doublon-holon,
//! backflow) stay as skeleton stubs and
//! will land alongside Phase 4 once `mvmc-core` actually consumes them.
//!
//! License: GPL-3.0-or-later (inherits from upstream).

#![warn(missing_docs)]

pub mod constants;
pub mod parsers;
pub mod types;
pub mod utils;

pub use types::{
    CoulombInterTerm, CoulombIntraTerm, ExchangeTerm, ExpertModeData, GreenOneTerm, GreenTwoTerm,
    GutzwillerTerm, HundTerm, InterAllTerm, JastrowTerm, LocSpinTerm, ModParaParameters,
    OrbitalTerm, ProjectionLayout, QPTransEntry, QuantumProjectionWeights, Spin, TransferTerm,
    ValidationResult,
};

pub use utils::validation::{
    validate_coulomb_inter_terms, validate_coulomb_intra_terms, validate_expert_mode_data,
    validate_gutzwiller_terms, validate_jastrow_terms, validate_modpara_params,
    validate_orbital_terms, validate_transfer_terms,
};

pub use utils::opt_flag::{
    ensure_optimization_flags_size, get_slater_opt_flag_index, is_gutzwiller_optimized,
    is_jastrow_optimized, is_slater_optimized, set_orbital_opt_flags, set_projection_opt_flags,
};

use std::collections::BTreeMap;
use std::io;
use std::path::Path;

use crate::parsers::{
    coulomb, exchange, green, gutzwiller, hund, interall, jastrow, locspin, modpara, orbital,
    qptrans, trans,
};
use crate::utils::file::{parse_namelist_content, read_def_file};

/// Errors surfaced by [`parse_expert_mode_files`].
#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    /// Orbital blocks cannot be interpreted in the supplied namelist order.
    #[error("invalid Expert input: {message}")]
    InvalidInput {
        /// Explanation of the invalid ordering.
        message: String,
    },
    /// Failed to read the top-level `namelist.def`.
    #[error("failed to read namelist.def at {path}: {source}")]
    Io {
        /// Path that failed to open.
        path: String,
        /// Underlying IO error.
        #[source]
        source: io::Error,
    },
}

/// Parse all Expert Mode files referenced by `namelist_path`. Mirrors
/// `parse_expert_mode_files(namelist_path)` in upstream Julia. Missing
/// child files are logged via [`tracing::warn`] and skipped, matching
/// the C / Julia "continue on error" policy. Unknown keywords in the
/// namelist are silently ignored (the round-trip set covers everything
/// the four `examples/inputs/*` cases use).
pub fn parse_expert_mode_files<P: AsRef<Path>>(
    namelist_path: P,
) -> Result<ExpertModeData, ParseError> {
    let namelist_path = namelist_path.as_ref();
    let base_dir = namelist_path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::path::PathBuf::from("."));

    let namelist_content = read_def_file(namelist_path).map_err(|e| ParseError::Io {
        path: namelist_path.display().to_string(),
        source: e,
    })?;
    let file_list = parse_namelist_content(&namelist_content);
    let parallel_pos = file_list
        .iter()
        .position(|(kind, _)| kind == "OrbitalParallel");
    let anti_pos = file_list
        .iter()
        .position(|(kind, _)| kind == "Orbital" || kind == "OrbitalAntiParallel");
    if matches!((parallel_pos, anti_pos), (Some(p), Some(a)) if p < a) {
        return Err(ParseError::InvalidInput {
            message: "OrbitalParallel must be listed after Orbital/OrbitalAntiParallel in namelist.def: the anti-parallel block defines the NArrayAP offset".into(),
        });
    }

    let mut data = ExpertModeData::new();
    data.namelist = file_list.clone();
    let mut orbital_flags = BTreeMap::new();

    for (file_type, file_name) in &file_list {
        let full_path = base_dir.join(file_name);
        if !full_path.is_file() {
            // Julia's parameter overlays are optional, including referenced
            // files that are absent. Their reader handles them after seeding.
            // InterAll is a Hamiltonian definition, not a parameter overlay.
            if file_type.starts_with("In") && file_type != "InterAll" {
                continue;
            }
            tracing::warn!("File not found: {}", full_path.display());
            data.input_errors.push(format!(
                "{file_type} file not found: {}",
                full_path.display()
            ));
            continue;
        }
        if let Err(e) = parse_file_by_type(&mut data, file_type, &full_path, &mut orbital_flags) {
            data.input_errors.push(format!(
                "error parsing {file_type} file {}: {e}",
                full_path.display()
            ));
            tracing::warn!(
                "Error parsing {} file {}: {}",
                file_type,
                full_path.display(),
                e
            );
        }
    }

    // Slater flags need all projection counts and AP/P declared widths.
    set_orbital_opt_flags(&mut data, &orbital_flags);

    // Mirror the post-parse pass from upstream:
    //   - If only `OrbitalAntiParallel` is parsed, the orbital mode stays
    //     at 0; if both AP and P appear, switch to general mode.
    if data.i_flg_orbital_general == 0
        && data.i_flg_orbital_anti_parallel == 1
        && data.i_flg_orbital_parallel == 1
    {
        data.i_flg_orbital_general = 1;
    }

    // Julia builds the lookup matrices after judging the orbital mode,
    // before initialization and runtime projection-count normalization.
    if !data.orbital_terms.is_empty() {
        data.ensure_orbital_idx_matrix();
    }

    // `ncond` propagation (matches `readdef.c:593`): Ne = (Nlocspin + Ncond) / 2.
    if data.modpara.ncond != -1 {
        if data.modpara.ncond % 2 != 0 {
            tracing::warn!("NCond must be even, got {}", data.modpara.ncond);
        } else if data.modpara.nelec == 0 {
            data.modpara.nelec = (data.modpara.nlocspin + data.modpara.ncond) / 2;
        }
    }

    Ok(data)
}

fn parse_file_by_type(
    data: &mut ExpertModeData,
    file_type: &str,
    path: &Path,
    orbital_flags: &mut BTreeMap<i64, i64>,
) -> io::Result<()> {
    match file_type {
        "ModPara" => {
            data.modpara = modpara::parse_modpara_def(path)?;
        }
        "LocSpin" => {
            // Mirror upstream: read NlocalSpin from the header before parsing.
            let content = read_def_file(path)?;
            if let Some(n) = locspin::read_nlocspin(&content) {
                data.modpara.nlocspin = n;
            }
            data.locspin_terms = locspin::parse_locspin_content(&content);
        }
        "Trans" => {
            data.transfer_terms = trans::parse_trans_def(path)?;
        }
        "CoulombIntra" => {
            data.coulomb_intra_terms = coulomb::parse_coulomb_intra_def(path)?;
        }
        "CoulombInter" => {
            data.coulomb_inter_terms = coulomb::parse_coulomb_inter_def(path)?;
        }
        "Hund" => {
            data.hund_terms = hund::parse_hund_def(path)?;
        }
        "Exchange" => {
            data.exchange_terms = exchange::parse_exchange_def(path)?;
        }
        "InterAll" => {
            data.inter_all_terms = interall::parse_interall_def(path)?;
        }
        "Gutzwiller" => {
            let content = read_def_file(path)?;
            let section = gutzwiller::parse_gutzwiller_content(&content);
            data.gutzwiller_terms = section.terms;
            data.n_gutzwiller_idx = section.n_gutzwiller_idx;
            set_projection_opt_flags(
                data,
                &section.opt_flags,
                &BTreeMap::new(),
                section.is_complex,
                false,
            );
            if !section.site_idx_map.is_empty() {
                let max_site = *section
                    .site_idx_map
                    .keys()
                    .max()
                    .expect("site_idx_map is non-empty");
                let mut idx_vec = vec![0i64; (max_site as usize) + 1];
                for (site, idx) in section.site_idx_map.iter() {
                    idx_vec[*site as usize] = *idx;
                }
                data.gutzwiller_idx = idx_vec;
            }
        }
        "Jastrow" => {
            let content = read_def_file(path)?;
            let section = jastrow::parse_jastrow_content(&content);
            data.jastrow_terms = section.terms;
            data.n_jastrow_idx = section.n_jastrow_idx;
            set_projection_opt_flags(
                data,
                &BTreeMap::new(),
                &section.opt_flags,
                false,
                section.is_complex,
            );
            let nsite = data.modpara.nsite;
            if nsite > 0 {
                let (matrix, n_idx) = jastrow::build_jastrow_idx_matrix(&content, nsite as usize);
                data.jastrow_idx = matrix;
                data.n_jastrow_idx = n_idx;
            }
        }
        "Orbital" | "OrbitalAntiParallel" => {
            let section = orbital::parse_orbital_def(path)?;
            data.orbital_terms = section.terms.clone();
            data.i_flg_orbital_anti_parallel = 1;
            data.modpara.n_orbital_idx = orbital_count(&section);
            data.n_orbital_anti_parallel = data.modpara.n_orbital_idx;
            orbital_flags.extend(section.opt_flags);
        }
        "OrbitalParallel" => {
            let section = orbital::parse_orbital_def(path)?;
            // Interleave with the existing (anti-parallel) orbital list.
            let n_orbital_ap = data.n_orbital_anti_parallel;
            for term in &section.terms {
                let up = OrbitalTerm {
                    idx: n_orbital_ap + 2 * term.idx,
                    ..*term
                };
                let down = OrbitalTerm {
                    idx: n_orbital_ap + 2 * term.idx + 1,
                    ..*term
                };
                data.orbital_terms.push(up);
                data.orbital_terms.push(down);
            }
            data.i_flg_orbital_parallel = 1;
            data.modpara.n_orbital_idx = n_orbital_ap + 2 * orbital_count(&section);
            for (idx, flag) in section.opt_flags {
                orbital_flags.insert(n_orbital_ap + 2 * idx, flag);
                orbital_flags.insert(n_orbital_ap + 2 * idx + 1, flag);
            }
        }
        "OrbitalGeneral" => {
            let section = orbital::parse_orbital_def(path)?;
            data.modpara.n_orbital_idx = orbital_count(&section);
            data.orbital_terms = section.terms;
            data.i_flg_orbital_general = 1;
            orbital_flags.extend(section.opt_flags);
        }
        "OneBodyG" => {
            data.green_one_terms = green::parse_green_one_def(path)?;
        }
        "TwoBodyG" => {
            data.green_two_terms = green::parse_green_two_def(path)?;
        }
        "TransSym" | "QPTrans" => {
            let section = qptrans::parse_qptrans_def(path, data.modpara.nsite)?;
            data.n_qp_trans = section.n_qp_trans;
            data.qp_trans_entries = section.entries;
            data.para_qp_trans = data.qp_trans_entries.iter().map(|e| e.weight).collect();
        }
        _ => {
            // Unknown / not-yet-supported keyword (e.g. RBM blocks).
            // Silently skip to match the upstream `@warn`-only policy.
            tracing::debug!("unknown namelist keyword: {}", file_type);
        }
    }
    Ok(())
}

fn orbital_count(section: &orbital::OrbitalSection) -> i64 {
    section.n_orbital_idx
}

/// C-parity defaults that occur in more than one upstream Julia file.
///
/// Centralised here so a typo can't desynchronise the parser, the
/// optimizer, and the quadrature module independently.
pub mod c_const {
    /// `1.0 / 2^32`. Used to convert a `u32` SFMT draw to `[0, 1)`.
    pub const RNG_REAL2_INV: f64 = 1.0 / 4_294_967_296.0;

    /// Maximum Slater-parameter amplitude (`D_AmpMax` in `parameter.c`).
    pub const D_AMP_MAX: f64 = 4.0;

    /// Gauss–Legendre Newton–Raphson convergence threshold.
    pub const GAUSSLEG_EPS: f64 = 5.0e-14;

    /// Real RBM initial-value scale (`RBM[i] = 0.01 * (genrand_real2() - 0.5) / Nneuron`).
    pub const RBM_INIT_REAL_SCALE: f64 = 0.01;

    /// Complex RBM initial-value scale (`RBM[i] = 1e-2 * genrand_real2() * cexp(...)`).
    pub const RBM_INIT_COMPLEX_SCALE: f64 = 1.0e-2;
}

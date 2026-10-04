//! Expert Mode `.def` parsers + [`ExpertModeData`] container.
//!
//! Port target: `extern/Julia-mVMC/MVMCExpertModeParsers.jl/src/`.
//!
//! Phase 3 status: the parsers, types and orchestration needed to
//! round-trip the four upstream `examples/inputs/*` namelists are
//! implemented, including strict DH2/DH4 definitions and their parameter layout.
//! Nine RBM index sections and initialization follow the canonical layout.
//! OptTrans input contracts are implemented; its production integration and
//! backflow remain pending.
//!
//! License: GPL-3.0-or-later (inherits from upstream).

#![warn(missing_docs)]

pub mod constants;
pub mod parsers;
pub mod types;
pub mod utils;

pub use types::{
    ChargeRBMHiddenLayerTerm, ChargeRBMPhysHiddenTerm, ChargeRBMPhysLayerTerm, CoulombInterTerm,
    CoulombIntraTerm, DoublonHolon2SiteDefinition, DoublonHolon2SiteIndex,
    DoublonHolon4SiteDefinition, DoublonHolon4SiteIndex, ExchangeTerm, ExpertModeData,
    GeneralRBMHiddenLayerTerm, GeneralRBMPhysHiddenTerm, GeneralRBMPhysLayerTerm, GreenOneTerm,
    GreenTwoExTerm, GreenTwoTerm, GutzwillerTerm, HundTerm, InterAllTerm, JastrowTerm, LocSpinTerm,
    ModParaParameters, OrbitalTerm, PairHopTerm, ProjectionLayout, QPTransEntry,
    QPTransInverseError, QuantumProjectionWeights, RbmParameter, Spin, SpinRBMHiddenLayerTerm,
    SpinRBMPhysHiddenTerm, SpinRBMPhysLayerTerm, TransferTerm, ValidationResult,
};

pub use utils::validation::{
    validate_coulomb_inter_terms, validate_coulomb_intra_terms,
    validate_doublon_holon_2site_indices, validate_doublon_holon_4site_indices,
    validate_expert_mode_data, validate_gutzwiller_terms, validate_jastrow_terms,
    validate_modpara_params, validate_orbital_terms, validate_rbm_parameters,
    validate_transfer_terms,
};

pub use utils::opt_flag::{
    ensure_optimization_flags_size, get_slater_opt_flag_index, is_gutzwiller_optimized,
    is_jastrow_optimized, is_slater_optimized, set_dh_opt_flags, set_opt_trans_c_opt_flags,
    set_opt_trans_opt_flags, set_orbital_opt_flags, set_projection_opt_flags, set_rbm_opt_flags,
};

use std::collections::BTreeMap;
use std::io;
use std::path::Path;

use crate::parsers::{
    coulomb, doublon_holon, exchange, green, gutzwiller, hund, interall, jastrow, locspin, modpara,
    opttrans, orbital, pairhop, qptrans, rbm, trans,
};
use crate::utils::file::{parse_namelist_content, read_def_file};

/// Errors surfaced by [`parse_expert_mode_files`].
#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    /// A referenced definition is invalid or missing.
    #[error("invalid Expert input: {message}")]
    InvalidInput {
        /// Explanation of the invalid input.
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

/// Parse all Expert Mode files referenced by `namelist_path` in C's fixed
/// keyword order, retaining the original namelist as metadata. Missing child
/// files currently populate `input_errors`, except optional parameter overlays.
/// DH2/DH4 and Green-function read/format/bounds failures return an error immediately.
pub fn parse_expert_mode_files<P: AsRef<Path>>(
    namelist_path: P,
) -> Result<ExpertModeData, ParseError> {
    parse_expert_mode_files_with_opt_trans(namelist_path, true)
}

/// Parse Expert Mode files while explicitly selecting C's OptTrans activation.
///
/// C only reads and activates an `OptTrans` definition when `FlagOptTrans` is
/// enabled by the caller. The default parser preserves Julia's definition-file
/// behavior; C-facing runners should pass `false` unless their `-o` mode is on.
pub fn parse_expert_mode_files_with_opt_trans<P: AsRef<Path>>(
    namelist_path: P,
    enable_opt_trans: bool,
) -> Result<ExpertModeData, ParseError> {
    parse_expert_mode_files_mode(namelist_path, enable_opt_trans, false)
}

/// Parse Expert Mode files using C's OptTrans activation and flag layout.
pub fn parse_expert_mode_files_with_c_opt_trans<P: AsRef<Path>>(
    namelist_path: P,
    enable_opt_trans: bool,
) -> Result<ExpertModeData, ParseError> {
    parse_expert_mode_files_mode(namelist_path, enable_opt_trans, true)
}

fn parse_expert_mode_files_mode<P: AsRef<Path>>(
    namelist_path: P,
    enable_opt_trans: bool,
    c_opt_trans_flags: bool,
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

    // C GetFileName checks keyword slots before opening any child definition,
    // including inactive OptTrans entries. Preserve raw metadata separately.
    let mut seen = std::collections::HashSet::new();
    let mut canonical_files = Vec::with_capacity(file_list.len());
    for (kind, filename) in &file_list {
        let canonical = canonical_namelist_keyword(kind);
        if let Some(keyword) = canonical {
            if !seen.insert(keyword) {
                return Err(ParseError::InvalidInput {
                    message: format!("duplicate keyword {keyword} in {}", namelist_path.display()),
                });
            }
        }
        canonical_files.push((canonical.unwrap_or(kind).to_owned(), filename.clone()));
    }

    let mut data = ExpertModeData::new();
    data.namelist = file_list.clone();
    let mut orbital_flags = BTreeMap::new();
    let mut rbm_flags = BTreeMap::new();

    let mut definitions: Vec<_> = canonical_files.iter().collect();
    definitions.sort_by_key(|(kind, _)| definition_order(kind));
    for (file_type, file_name) in definitions {
        if !enable_opt_trans && matches!(file_type.as_str(), "OptTrans" | "InOptTrans") {
            continue;
        }
        let full_path = base_dir.join(file_name);
        if !full_path.is_file() {
            if matches!(
                file_type.as_str(),
                "DH2"
                    | "CoulombIntra"
                    | "CoulombInter"
                    | "Trans"
                    | "DoublonHolon2Site"
                    | "DH4"
                    | "DoublonHolon4Site"
                    | "OneBodyG"
                    | "TwoBodyG"
                    | "TwoBodyGEx"
            ) {
                return Err(ParseError::InvalidInput {
                    message: format!(
                        "Required {file_type} file not found: {}",
                        full_path.display()
                    ),
                });
            }
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
        if let Err(e) = parse_file_by_type(
            &mut data,
            file_type,
            &full_path,
            &mut orbital_flags,
            &mut rbm_flags,
        ) {
            if matches!(
                file_type.as_str(),
                "DH2"
                    | "CoulombIntra"
                    | "CoulombInter"
                    | "Trans"
                    | "DoublonHolon2Site"
                    | "DH4"
                    | "DoublonHolon4Site"
                    | "OneBodyG"
                    | "TwoBodyG"
                    | "TwoBodyGEx"
            ) {
                return Err(ParseError::InvalidInput {
                    message: format!(
                        "Error parsing required {file_type} file {}: {e}",
                        full_path.display()
                    ),
                });
            }
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

    // DH and Slater flags need the final projection layout.
    set_dh_opt_flags(&mut data);
    let sizes = data.rbm_section_sizes();
    data.rbm_params
        .resize(sizes.iter().sum(), num_complex::Complex64::new(0.0, 0.0));
    let mut offset = data.projection_layout().n_proj;
    for (name, size) in rbm::SECTION_NAMES.iter().zip(sizes) {
        if size > 0 {
            if let Some((flags, complex)) = rbm_flags.get(*name) {
                utils::opt_flag::set_rbm_opt_flags(&mut data, flags, offset, *complex);
            }
        }
        offset += size;
    }
    set_orbital_opt_flags(&mut data, &orbital_flags);
    if c_opt_trans_flags {
        utils::opt_flag::set_opt_trans_c_opt_flags(&mut data);
    } else {
        set_opt_trans_opt_flags(&mut data);
    }
    data.slater_params.resize(
        data.modpara.n_orbital_idx.max(0) as usize,
        num_complex::Complex64::new(0.0, 0.0),
    );

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

    data.canonicalize_green_two_ex();

    Ok(data)
}

/// C's filename-keyword slots in native enum order (readdef.h).
const C_NAMELIST_KEYWORDS: &[&str] = &[
    "ModPara",
    "LocSpin",
    "Trans",
    "CoulombIntra",
    "CoulombInter",
    "Hund",
    "PairHop",
    "Exchange",
    "Gutzwiller",
    "Jastrow",
    "DH2",
    "DH4",
    "ChargeRBM_HiddenLayer",
    "ChargeRBM_PhysLayer",
    "ChargeRBM_PhysHidden",
    "SpinRBM_HiddenLayer",
    "SpinRBM_PhysLayer",
    "SpinRBM_PhysHidden",
    "GeneralRBM_HiddenLayer",
    "GeneralRBM_PhysLayer",
    "GeneralRBM_PhysHidden",
    "Orbital",
    "OrbitalAntiParallel",
    "OrbitalParallel",
    "OrbitalGeneral",
    "TransSym",
    "InGutzwiller",
    "InJastrow",
    "InDH2",
    "InDH4",
    "InChargeRBM_HiddenLayer",
    "InChargeRBM_PhysLayer",
    "InChargeRBM_PhysHidden",
    "InSpinRBM_HiddenLayer",
    "InSpinRBM_PhysLayer",
    "InSpinRBM_PhysHidden",
    "InGeneralRBM_HiddenLayer",
    "InGeneralRBM_PhysLayer",
    "InGeneralRBM_PhysHidden",
    "InOrbital",
    "InOrbitalAntiParallel",
    "InOrbitalParallel",
    "InOrbitalGeneral",
    "OneBodyG",
    "TwoBodyG",
    "TwoBodyGEx",
    "InterAll",
    "OptTrans",
    "InOptTrans",
    "BF",
    "BFRange",
];

/// ModPara precedes dimensions; AP precedes P regardless of namelist order.
fn definition_order(kind: &str) -> usize {
    let canonical = canonical_namelist_keyword(kind).unwrap_or(kind);
    C_NAMELIST_KEYWORDS
        .iter()
        .position(|&keyword| keyword == canonical)
        .unwrap_or(usize::MAX)
}

// Native CheckWords is ASCII case-insensitive. Legacy aliases are architecture
// extensions, not spellings accepted by the native C filename table.
pub(crate) fn canonical_namelist_keyword(kind: &str) -> Option<&'static str> {
    for (alias, canonical) in [
        ("DoublonHolon2Site", "DH2"),
        ("DoublonHolon4Site", "DH4"),
        ("QPTrans", "TransSym"),
    ] {
        if kind.eq_ignore_ascii_case(alias) {
            return Some(canonical);
        }
    }
    C_NAMELIST_KEYWORDS
        .iter()
        .copied()
        .find(|keyword| kind.eq_ignore_ascii_case(keyword))
}

// C readdef.c GetInfoOneBodyG/GetInfoTwoBodyG/GetInfoTwoBodyGEx check every
// site against Nsite and require the declared number of records. Perform the
// supported-input checks before publishing any parsed Green section. Typed
// spins and malformed-integer rejection are Rust safety rules, not claims
// about C's unchecked sscanf or out-of-bounds indirect lookup behavior.
fn validate_green_definition(
    path: &Path,
    family: &str,
    nsite: i64,
    fields: usize,
) -> io::Result<usize> {
    let content = read_def_file(path)?;
    let lines: Vec<_> = content.lines().collect();
    let invalid = |message: String| io::Error::new(io::ErrorKind::InvalidData, message);
    let count = lines
        .get(1)
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|value| value.parse::<usize>().ok())
        .ok_or_else(|| invalid(format!("{family}: missing or invalid declared row count")))?;
    // Original C readers return immediately for NArray==0, ignoring body rows.
    if count == 0 {
        return Ok(0);
    }
    let mut actual = 0;
    for (index, line) in lines.iter().enumerate().skip(5) {
        let cleaned = utils::file::clean_line(line);
        if cleaned.is_empty() {
            continue;
        }
        let tokens = utils::file::split_def_line(cleaned);
        if tokens.len() < fields {
            return Err(invalid(format!(
                "{family} line {}: expected at least {fields} integer fields",
                index + 1
            )));
        }
        for (column, token) in tokens.iter().take(fields).enumerate() {
            let value = token.parse::<i64>().map_err(|_| {
                invalid(format!(
                    "{family} line {}: invalid integer field {}",
                    index + 1,
                    column + 1
                ))
            })?;
            if column % 2 == 0 && (value < 0 || value >= nsite) {
                return Err(invalid(format!(
                    "{family} line {}: site{} ({value}) out of range [0, {})",
                    index + 1,
                    column / 2 + 1,
                    nsite
                )));
            }
            if column % 2 == 1 && !(0..=1).contains(&value) {
                return Err(invalid(format!(
                    "{family} line {}: Rust typed spin must be 0 or 1",
                    index + 1
                )));
            }
        }
        actual += 1;
    }
    if actual != count {
        return Err(invalid(format!(
            "{family}: declared row count {count}, got {actual}"
        )));
    }
    Ok(count)
}

type RbmFlags = BTreeMap<String, (BTreeMap<i64, i64>, bool)>;

fn parse_file_by_type(
    data: &mut ExpertModeData,
    file_type: &str,
    path: &Path,
    orbital_flags: &mut BTreeMap<i64, i64>,
    rbm_flags: &mut RbmFlags,
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
            data.transfer_terms = trans::parse_trans_definition(path, data.modpara.nsite)?;
        }
        "CoulombIntra" => {
            data.coulomb_intra_terms =
                coulomb::parse_coulomb_intra_definition(path, data.modpara.nsite)?;
        }
        "CoulombInter" => {
            data.coulomb_inter_terms =
                coulomb::parse_coulomb_inter_definition(path, data.modpara.nsite)?;
        }
        "Hund" => {
            data.hund_terms = hund::parse_hund_def(path)?;
        }
        "Exchange" => {
            data.exchange_terms = exchange::parse_exchange_def(path)?;
        }
        "PairHop" => {
            let section = pairhop::parse_pairhop_def(path)?;
            if !section.is_success() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    section.errors.join("; "),
                ));
            }
            data.pair_hop_terms = section.terms;
        }
        "InterAll" => {
            data.inter_all_terms =
                interall::parse_interall_def(path, data.modpara.nsite, data.modpara.two_sz)?;
        }
        "DH2" | "DoublonHolon2Site" => {
            let section = doublon_holon::parse_doublon_holon_2site_def(path, data.modpara.nsite)?;
            let definition = section.data.ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "Failed to parse DH2 file '{}': {}",
                        path.display(),
                        section.error_message
                    ),
                )
            })?;
            data.doublon_holon_2site_indices = definition.indices;
            data.doublon_holon_2site_complex = definition.is_complex;
            data.doublon_holon_2site_params =
                vec![num_complex::Complex64::new(0.0, 0.0); definition.opt_flags.len()];
            data.doublon_holon_2site_opt_flags = definition.opt_flags;
        }
        "DH4" | "DoublonHolon4Site" => {
            let section = doublon_holon::parse_doublon_holon_4site_def(path, data.modpara.nsite)?;
            let definition = section.data.ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "Failed to parse DH4 file '{}': {}",
                        path.display(),
                        section.error_message
                    ),
                )
            })?;
            data.doublon_holon_4site_indices = definition.indices;
            data.doublon_holon_4site_complex = definition.is_complex;
            data.doublon_holon_4site_params =
                vec![num_complex::Complex64::new(0.0, 0.0); definition.opt_flags.len()];
            data.doublon_holon_4site_opt_flags = definition.opt_flags;
        }
        "Gutzwiller" => {
            let content = read_def_file(path)?;
            let section = gutzwiller::parse_gutzwiller_content(&content, data.modpara.nsite)?;
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
            let section = jastrow::parse_jastrow_content(&content, data.modpara.nsite)?;
            data.jastrow_terms = section.terms;
            data.n_jastrow_idx = section.n_jastrow_idx;
            set_projection_opt_flags(
                data,
                &BTreeMap::new(),
                &section.opt_flags,
                false,
                section.is_complex,
            );
            data.jastrow_idx = section.idx_matrix;
        }
        "Orbital" | "OrbitalAntiParallel" => {
            let section = orbital::parse_orbital_def(
                path,
                data.modpara.nsite,
                orbital::OrbitalKind::AntiParallel,
            )?;
            data.orbital_terms = section.terms.clone();
            data.i_flg_orbital_anti_parallel = 1;
            data.modpara.n_orbital_idx = section.n_orbital_idx;
            data.n_orbital_anti_parallel = data.modpara.n_orbital_idx;
            orbital_flags.extend(section.opt_flags);
        }
        "OrbitalParallel" => {
            let section = orbital::parse_orbital_def(
                path,
                data.modpara.nsite,
                orbital::OrbitalKind::Parallel,
            )?;
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
            data.modpara.n_orbital_idx = n_orbital_ap + 2 * section.n_orbital_idx;
            for (idx, flag) in section.opt_flags {
                orbital_flags.insert(n_orbital_ap + 2 * idx, flag);
                orbital_flags.insert(n_orbital_ap + 2 * idx + 1, flag);
            }
        }
        "OrbitalGeneral" => {
            let section = orbital::parse_orbital_def(
                path,
                data.modpara.nsite,
                orbital::OrbitalKind::General,
            )?;
            data.modpara.n_orbital_idx = section.n_orbital_idx;
            data.orbital_terms = section.terms;
            data.i_flg_orbital_general = 1;
            orbital_flags.extend(section.opt_flags);
        }
        kind if rbm::SECTION_NAMES.contains(&kind) => {
            let index = rbm::SECTION_NAMES
                .iter()
                .position(|&name| name == kind)
                .unwrap();
            let hidden = [
                data.modpara.nneuron_charge,
                data.modpara.nneuron_spin,
                data.modpara.nneuron_general,
            ][index % 3];
            let section =
                rbm::parse_rbm_content(&read_def_file(path)?, index, data.modpara.nsite, hidden)?;
            data.rbm_section_widths[index] = section.width;
            let complex = section.is_complex;
            let zero = num_complex::Complex64::new(0.0, 0.0);
            match index {
                0 => {
                    data.charge_rbm_phys_layer_terms = section
                        .mappings
                        .iter()
                        .map(|v| types::ChargeRBMPhysLayerTerm {
                            site: v[0],
                            idx: v[1],
                            value: zero,
                            is_complex: complex,
                        })
                        .collect()
                }
                1 => {
                    data.spin_rbm_phys_layer_terms = section
                        .mappings
                        .iter()
                        .map(|v| types::SpinRBMPhysLayerTerm {
                            site: v[0],
                            idx: v[1],
                            value: zero,
                            is_complex: complex,
                        })
                        .collect()
                }
                2 => {
                    data.general_rbm_phys_layer_terms = section
                        .mappings
                        .iter()
                        .map(|v| types::GeneralRBMPhysLayerTerm {
                            site: v[0],
                            spin: v[1],
                            idx: v[2],
                            value: zero,
                            is_complex: complex,
                        })
                        .collect()
                }
                3 => {
                    data.charge_rbm_hidden_layer_terms = section
                        .mappings
                        .iter()
                        .map(|v| types::ChargeRBMHiddenLayerTerm {
                            site: v[0],
                            idx: v[1],
                            value: zero,
                            is_complex: complex,
                        })
                        .collect()
                }
                4 => {
                    data.spin_rbm_hidden_layer_terms = section
                        .mappings
                        .iter()
                        .map(|v| types::SpinRBMHiddenLayerTerm {
                            site: v[0],
                            idx: v[1],
                            value: zero,
                            is_complex: complex,
                        })
                        .collect()
                }
                5 => {
                    data.general_rbm_hidden_layer_terms = section
                        .mappings
                        .iter()
                        .map(|v| types::GeneralRBMHiddenLayerTerm {
                            site: v[0],
                            idx: v[1],
                            value: zero,
                            is_complex: complex,
                        })
                        .collect()
                }
                6 => {
                    data.charge_rbm_phys_hidden_terms = section
                        .mappings
                        .iter()
                        .map(|v| types::ChargeRBMPhysHiddenTerm {
                            site1: v[0],
                            site2: v[1],
                            idx: v[2],
                            value: zero,
                            is_complex: complex,
                        })
                        .collect()
                }
                7 => {
                    data.spin_rbm_phys_hidden_terms = section
                        .mappings
                        .iter()
                        .map(|v| types::SpinRBMPhysHiddenTerm {
                            site1: v[0],
                            site2: v[1],
                            idx: v[2],
                            value: zero,
                            is_complex: complex,
                        })
                        .collect()
                }
                8 => {
                    data.general_rbm_phys_hidden_terms = section
                        .mappings
                        .iter()
                        .map(|v| types::GeneralRBMPhysHiddenTerm {
                            site1: v[0],
                            spin: v[1],
                            site2: v[2],
                            idx: v[3],
                            value: zero,
                            is_complex: complex,
                        })
                        .collect()
                }
                _ => unreachable!(),
            }
            rbm_flags.insert(file_type.to_owned(), (section.opt_flags, complex));
        }
        "OneBodyG" => {
            let count = validate_green_definition(path, file_type, data.modpara.nsite, 4)?;
            data.green_one_terms = if count == 0 {
                Vec::new()
            } else {
                green::parse_green_one_def(path)?
            };
        }
        "TwoBodyG" => {
            let count = validate_green_definition(path, file_type, data.modpara.nsite, 8)?;
            data.green_two_terms = if count == 0 {
                Vec::new()
            } else {
                green::parse_green_two_def(path)?
            };
        }
        "TwoBodyGEx" => {
            let count = validate_green_definition(path, file_type, data.modpara.nsite, 8)?;
            data.green_two_ex_terms = if count == 0 {
                Vec::new()
            } else {
                green::parse_green_two_ex_def(path)?
            };
        }
        "TransSym" | "QPTrans" => {
            let section = qptrans::parse_qptrans_def(path, data.modpara.nsite)?;
            data.n_qp_trans = section.n_qp_trans;
            data.qp_trans_entries = section.entries;
            data.para_qp_trans = data.qp_trans_entries.iter().map(|e| e.weight).collect();
            if data.n_qp_trans > 0
                && data.modpara.nsite > 0
                && (data.qp_opt_trans.is_empty() || data.qp_opt_trans_sgn.is_empty())
            {
                let count = data.n_qp_opt_trans.max(1) as usize;
                data.qp_opt_trans = vec![(0..data.modpara.nsite).collect(); count];
                data.qp_opt_trans_sgn = vec![vec![1; data.modpara.nsite as usize]; count];
            }
        }
        "OptTrans" => opttrans::parse_opttrans_def(data, path)?,
        _ => {
            // Unknown / not-yet-supported keyword.
            // Silently skip to match the upstream `@warn`-only policy.
            tracing::debug!("unknown namelist keyword: {}", file_type);
        }
    }
    Ok(())
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

#[cfg(test)]
#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;

#[cfg(test)]
mod namelist_keyword_tests {
    use super::{canonical_namelist_keyword, C_NAMELIST_KEYWORDS};

    #[test]
    fn native_slots_and_legacy_aliases_have_exact_case_insensitive_identity() {
        assert_eq!(C_NAMELIST_KEYWORDS.len(), 51);
        for keyword in C_NAMELIST_KEYWORDS {
            assert_eq!(canonical_namelist_keyword(keyword), Some(*keyword));
            assert_eq!(
                canonical_namelist_keyword(&keyword.to_ascii_lowercase()),
                Some(*keyword)
            );
            assert_eq!(
                canonical_namelist_keyword(&keyword.to_ascii_uppercase()),
                Some(*keyword)
            );
        }
        for (alias, native) in [
            ("DoublonHolon2Site", "DH2"),
            ("DoublonHolon4Site", "DH4"),
            ("QPTrans", "TransSym"),
        ] {
            assert_eq!(canonical_namelist_keyword(alias), Some(native));
            assert_eq!(
                canonical_namelist_keyword(&alias.to_ascii_lowercase()),
                Some(native)
            );
            assert_eq!(
                canonical_namelist_keyword(&alias.to_ascii_uppercase()),
                Some(native)
            );
        }
        for unknown in ["", "ModParaExtra", "NotACKeyword", "ＭodPara"] {
            assert_eq!(canonical_namelist_keyword(unknown), None);
        }
        assert_ne!(
            canonical_namelist_keyword("Orbital"),
            canonical_namelist_keyword("OrbitalAntiParallel")
        );
    }
}

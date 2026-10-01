//! Expert-mode data types.
//!
//! Port target: `MVMCExpertModeParsers.jl/src/types/expert_types.jl`.
//!
//! Phase 3 status: the round-trip subset is implemented (`ModPara`, the
//! 8 simple term structs, plus `ExpertModeData`). The RBM / doublon-
//! holon / backflow tree from the upstream Julia file is still pending
//! and will land alongside Phase 4. Anything not used by the four
//! upstream `examples/inputs/*/namelist.def` test cases is omitted on
//! purpose.

use num_complex::Complex64;

/// Structured result of Julia-compatible consistency validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationResult {
    /// True when there are no errors, regardless of warnings.
    pub is_valid: bool,
    /// Validation errors in upstream check/family order.
    pub errors: Vec<String>,
    /// Nonfatal diagnostics in upstream check/family order.
    pub warnings: Vec<String>,
}

/// Declared C-compatible projection widths and offsets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProjectionLayout {
    /// Gutzwiller parameter count, preferring a positive declared width.
    pub n_gutzwiller: usize,
    /// Jastrow parameter count, preferring a positive declared width.
    pub n_jastrow: usize,
    /// Start of Gutzwiller parameters (zero).
    pub gutzwiller_offset: usize,
    /// Start of Jastrow parameters after the declared Gutzwiller block.
    pub jastrow_offset: usize,
    /// Total projection parameter count.
    pub n_proj: usize,
}

impl ValidationResult {
    /// Construct a result whose validity is derived from the errors.
    pub fn new(errors: Vec<String>, warnings: Vec<String>) -> Self {
        Self {
            is_valid: errors.is_empty(),
            errors,
            warnings,
        }
    }
}

/// Single-particle spin tag. Stored as a `u8` (0 = up, 1 = down) so the
/// round-trip back to a `.def` file is byte-stable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Spin {
    /// 0 in the upstream `.def` files.
    Up,
    /// 1 in the upstream `.def` files.
    Down,
}

impl Spin {
    /// Decode a 0/1 spin code from a `.def` file. Returns `None` on
    /// other values so the parser can record a per-line error.
    pub fn from_code(c: i64) -> Option<Self> {
        match c {
            0 => Some(Self::Up),
            1 => Some(Self::Down),
            _ => None,
        }
    }

    /// Encode back to the 0/1 representation used in upstream `.def` files.
    pub fn as_code(self) -> u8 {
        match self {
            Self::Up => 0,
            Self::Down => 1,
        }
    }
}

/// `modpara.def` parameters (port of `ModParaParameters`).
///
/// Fields are listed in the same order as upstream, but unused fields
/// (RBM, Lanczos, exchange-update, orbital-idx) are stored as raw `i64`
/// / `f64` without computed defaults so we can ferry them back out at
/// round-trip time.
#[derive(Debug, Clone, PartialEq)]
pub struct ModParaParameters {
    /// `NSite`.
    pub nsite: i64,
    /// `NElec`.
    pub nelec: i64,
    /// `NLocSpin`.
    pub nlocspin: i64,
    /// `NCond` (-1 sentinel).
    pub ncond: i64,

    /// `NVMCCalMode`.
    pub vmc_calc_mode: i64,
    /// `NLanczosMode`.
    pub lanczos_mode: i64,

    /// `NSROptItrStep`.
    pub nsr_opt_itr_step: i64,
    /// `NSROptItrSmp`.
    pub nsr_opt_itr_smp: i64,
    /// `NSROptFixSmp`.
    pub nsr_opt_fix_smp: i64,
    /// `NVMCWarmUp`.
    pub nvmc_warmup: i64,
    /// `NVMCInterval`.
    pub nvmc_interval: i64,
    /// `NVMCSample`.
    pub nvmc_sample: i64,

    /// `DSROptRedCut`.
    pub dsr_opt_red_cut: f64,
    /// `DSROptStaDel`.
    pub dsr_opt_sta_del: f64,
    /// `DSROptStepDt`.
    pub dsr_opt_step_dt: f64,
    /// `DSROptCGTol`.
    pub dsr_opt_cg_tol: f64,
    /// `NSROptCGMaxIter`.
    pub nsr_opt_cg_max_iter: i64,

    /// `NSRCG` (0 = direct, non-zero = CG).
    pub nsrcg: i64,
    /// `useDiagScale` (unsupported preconditioned CG mode).
    pub use_diag_scale: i64,
    /// `RescaleSmat` (unsupported S-matrix rescaling).
    pub rescale_smat: i64,
    /// `NStore`.
    pub nstore_o: i64,

    /// `RndSeed`.
    pub rnd_seed: i64,
    /// `NSplitSize`.
    pub nsplit_size: i64,

    /// `NSPGaussLeg`.
    pub nsp_gauss_leg: i64,
    /// `NSPStot`.
    pub nsp_stot: i64,
    /// `NMPTrans`.
    pub nmp_trans: i64,
    /// `2Sz` (-1 = FSZ).
    pub two_sz: i64,

    /// `NDataIdxStart`.
    pub n_data_idx_start: i64,
    /// `NDataQtySmp`.
    pub n_data_qty_smp: i64,
    /// `CDataFileHead`.
    pub c_data_file_head: String,
    /// `CParaFileHead`.
    pub c_para_file_head: String,

    /// `NFileFlushInterval`.
    pub n_file_flush_interval: i64,
    /// `ComplexType`.
    pub complex_flag: i64,

    /// `Nneuron`.
    pub nneuron: i64,
    /// `NneuronGeneral`.
    pub nneuron_general: i64,
    /// `NneuronCharge`.
    pub nneuron_charge: i64,
    /// `NneuronSpin`.
    pub nneuron_spin: i64,
    /// `NBlockSize_RBMRatio`.
    pub nblock_size_rbm_ratio: i64,

    /// `NOneBodyG`.
    pub n_one_body_g: i64,
    /// `NTwoBodyG`.
    pub n_two_body_g: i64,
    /// `NTwoBodyGEx`.
    pub n_two_body_g_ex: i64,

    /// `NExUpdatePath`.
    pub nex_update_path: i64,

    /// Number of unique orbital parameters (`NOrbitalIdx`).
    pub n_orbital_idx: i64,
}

impl Default for ModParaParameters {
    fn default() -> Self {
        Self {
            nsite: 0,
            nelec: 0,
            nlocspin: 0,
            ncond: -1,
            vmc_calc_mode: 0,
            lanczos_mode: 0,
            nsr_opt_itr_step: 1000,
            nsr_opt_itr_smp: 1000,
            nsr_opt_fix_smp: 0,
            nvmc_warmup: 1000,
            nvmc_interval: 1,
            nvmc_sample: 10000,
            dsr_opt_red_cut: 1e-6,
            dsr_opt_sta_del: 0.0,
            dsr_opt_step_dt: 0.01,
            dsr_opt_cg_tol: 1e-10,
            nsr_opt_cg_max_iter: 0,
            nsrcg: 0,
            use_diag_scale: 0,
            rescale_smat: 0,
            nstore_o: 1,
            rnd_seed: 11272,
            nsplit_size: 1,
            nsp_gauss_leg: 1,
            nsp_stot: 0,
            nmp_trans: 0,
            two_sz: -1,
            n_data_idx_start: 0,
            n_data_qty_smp: 1,
            c_data_file_head: String::from("zvo"),
            c_para_file_head: String::from("zqp"),
            n_file_flush_interval: 1,
            complex_flag: 0,
            nneuron: 0,
            nneuron_general: 0,
            nneuron_charge: 0,
            nneuron_spin: 0,
            nblock_size_rbm_ratio: 200,
            n_one_body_g: 0,
            n_two_body_g: 0,
            n_two_body_g_ex: 0,
            nex_update_path: 1,
            n_orbital_idx: 0,
        }
    }
}

/// Single transfer (hopping) term from `trans.def`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransferTerm {
    /// `i` (creation site).
    pub site1: i64,
    /// `s` (creation spin code 0/1).
    pub spin1: Spin,
    /// `j` (annihilation site).
    pub site2: i64,
    /// Annihilation spin code 0/1.
    pub spin2: Spin,
    /// `t_ij` value as a complex (real part second-to-last column,
    /// imag part last column).
    pub value: Complex64,
}

/// `CoulombIntra` term: same-site Coulomb `U n_up n_down`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoulombIntraTerm {
    /// Site index.
    pub site: i64,
    /// `U` value.
    pub value: f64,
}

/// `CoulombInter` term: between-site density-density Coulomb.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoulombInterTerm {
    /// First site.
    pub site1: i64,
    /// Second site.
    pub site2: i64,
    /// Value.
    pub value: f64,
}

/// `Hund` coupling term.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HundTerm {
    /// First site.
    pub site1: i64,
    /// Second site.
    pub site2: i64,
    /// Value.
    pub value: f64,
}

/// `Exchange` coupling term.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExchangeTerm {
    /// First site.
    pub site1: i64,
    /// Second site.
    pub site2: i64,
    /// Value.
    pub value: f64,
}

/// `LocSpin` term (locspn.def).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocSpinTerm {
    /// Site index.
    pub site: i64,
    /// 0 = itinerant, 1 = localised spin (matches the C convention).
    pub spin_value: i64,
}

/// Gutzwiller-index entry from `gutzwilleridx.def`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GutzwillerTerm {
    /// Site index.
    pub site: i64,
    /// Initial Gutzwiller parameter value (typically 0 + 0i in `.def`
    /// files; populated later from `InGutzwiller.def`).
    pub value: Complex64,
    /// Whether this parameter is treated as complex (from `ComplexType`).
    pub is_complex: bool,
}

/// Jastrow-index entry from `jastrowidx.def`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JastrowTerm {
    /// First site.
    pub site1: i64,
    /// Second site.
    pub site2: i64,
    /// Initial Jastrow parameter value (typically 0 + 0i in `.def`).
    pub value: Complex64,
    /// Whether this parameter is complex.
    pub is_complex: bool,
}

/// Orbital-parameter entry from `orbitalidx.def`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrbitalTerm {
    /// First site.
    pub site1: i64,
    /// Second site.
    pub site2: i64,
    /// Orbital parameter index (0-based; capped at `n_orbital_idx - 1`).
    pub idx: i64,
    /// Initial parameter value (typically 0 + 0i in `.def`).
    pub value: Complex64,
    /// Complex-parameter flag.
    pub is_complex: bool,
    /// Optional sign (+1 / -1) from the optional fourth column.
    pub sign: i64,
}

/// One-body Green-function term (`greenone.def`): ri si rj sj.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GreenOneTerm {
    /// `ri`.
    pub site1: i64,
    /// `si`.
    pub spin1: Spin,
    /// `rj`.
    pub site2: i64,
    /// `sj`.
    pub spin2: Spin,
}

/// Two-body Green-function term (`greentwo.def`): ri si rj sj rk sk rl sl.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GreenTwoTerm {
    /// `ri`.
    pub site1: i64,
    /// `si`.
    pub spin1: Spin,
    /// `rj`.
    pub site2: i64,
    /// `sj`.
    pub spin2: Spin,
    /// `rk`.
    pub site3: i64,
    /// `sk`.
    pub spin3: Spin,
    /// `rl`.
    pub site4: i64,
    /// `sl`.
    pub spin4: Spin,
}

/// One `qptransidx.def` entry: the per-translation `ParaQPTrans` weight
/// plus its (`origin -> translated_site`, `sign`) lookup table.
#[derive(Debug, Clone, PartialEq)]
pub struct QPTransEntry {
    /// `ParaQPTrans[mpidx]`.
    pub weight: Complex64,
    /// `QPTrans[mpidx][ori] = trj`.
    pub site_map: Vec<i64>,
    /// `QPTransSgn[mpidx][ori] = +/-1`.
    pub site_sign: Vec<i64>,
}

impl QPTransEntry {
    /// Translation signs are active only for antiperiodic boundaries.
    pub fn boundary_sign(&self, site: usize, antiperiodic: bool) -> i64 {
        if antiperiodic {
            self.site_sign.get(site).copied().unwrap_or(1)
        } else {
            1
        }
    }
}

/// Quantum-projection weights (port of `QuantumProjectionWeights` in
/// `utils/qp_weight.jl`). Populated by
/// [`crate::utils::qp_weight::init_qp_weight`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct QuantumProjectionWeights {
    /// `QPFullWeight[NQPFull]`.
    pub qp_full_weight: Vec<Complex64>,
    /// `QPFixWeight[NQPFix]`.
    pub qp_fix_weight: Vec<Complex64>,
    /// `cos(beta/2)`.
    pub spgl_cos: Vec<Complex64>,
    /// `sin(beta/2)`.
    pub spgl_sin: Vec<Complex64>,
    /// `cos(beta/2) * sin(beta/2)`.
    pub spgl_cos_sin: Vec<Complex64>,
    /// `cos(beta/2)^2`.
    pub spgl_cos_cos: Vec<Complex64>,
    /// `sin(beta/2)^2`.
    pub spgl_sin_sin: Vec<Complex64>,
}

impl QuantumProjectionWeights {
    /// Construct an empty `QuantumProjectionWeights`.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Owned container for all parsed Expert-mode `.def` data.
///
/// Phase 3 covers the data the four upstream `examples/inputs/*` cases
/// need; RBM / doublon-holon / backflow / OptTrans payloads land later.
#[derive(Debug, Clone, Default)]
pub struct ExpertModeData {
    /// `modpara.def` payload.
    pub modpara: ModParaParameters,
    /// Resolved (file_type, path) pairs from `namelist.def`, in file order.
    pub namelist: Vec<(String, String)>,
    /// Child-file read failures retained for runtime validation.
    /// Parsing remains a format-reading operation; runners must not silently
    /// execute an incomplete Hamiltonian after a recoverable parser warning.
    pub input_errors: Vec<String>,

    /// Transfer (hopping) terms.
    pub transfer_terms: Vec<TransferTerm>,
    /// On-site Coulomb terms.
    pub coulomb_intra_terms: Vec<CoulombIntraTerm>,
    /// Inter-site Coulomb terms.
    pub coulomb_inter_terms: Vec<CoulombInterTerm>,
    /// Hund coupling terms.
    pub hund_terms: Vec<HundTerm>,
    /// Exchange coupling terms.
    pub exchange_terms: Vec<ExchangeTerm>,

    /// Local-spin specifiers.
    pub locspin_terms: Vec<LocSpinTerm>,

    /// Gutzwiller (site -> idx) entries.
    pub gutzwiller_terms: Vec<GutzwillerTerm>,
    /// `NGutzwillerIdx`.
    pub n_gutzwiller_idx: i64,
    /// Site -> Gutzwiller-index lookup (1 entry per site, 0-based).
    pub gutzwiller_idx: Vec<i64>,

    /// Jastrow (site1, site2, idx) entries.
    pub jastrow_terms: Vec<JastrowTerm>,
    /// `NJastrowIdx`.
    pub n_jastrow_idx: i64,
    /// `JastrowIdx[ri+1, rj+1]` (row-major; -1 for unset diagonals).
    pub jastrow_idx: Vec<Vec<i64>>,

    /// Orbital (site1, site2, idx, sign) entries.
    pub orbital_terms: Vec<OrbitalTerm>,
    /// Set to 1 once an `Orbital` or `OrbitalAntiParallel` file is parsed.
    pub i_flg_orbital_anti_parallel: i64,
    /// Set to 1 once an `OrbitalParallel` file is parsed.
    pub i_flg_orbital_parallel: i64,
    /// Set to 1 once `OrbitalGeneral` is parsed (or AP+P together).
    pub i_flg_orbital_general: i64,
    /// Declared anti-parallel parameter count (`NArrayAP`), the exact
    /// starting offset of interleaved parallel parameters.
    pub n_orbital_anti_parallel: i64,

    /// Green-function measurements (one body).
    pub green_one_terms: Vec<GreenOneTerm>,
    /// Green-function measurements (two body).
    pub green_two_terms: Vec<GreenTwoTerm>,

    /// `qptransidx.def` payload.
    pub qp_trans_entries: Vec<QPTransEntry>,
    /// `NQPTrans`.
    pub n_qp_trans: i64,

    /// `ParaQPTrans` values (real / complex weights from `qptransidx.def`).
    /// Convenience mirror of `qp_trans_entries[i].weight`, populated by
    /// `parse_expert_mode_files` so downstream Phase-4 code does not have
    /// to walk `qp_trans_entries`.
    pub para_qp_trans: Vec<num_complex::Complex64>,
    /// `NQPOptTrans` (defaults to 1; tracked for the future OptTrans port).
    pub n_qp_opt_trans: i64,

    /// `OrbitalIdx[ri+1, rj+1] = idx` lookup matrix (row-major). General
    /// orbitals use `2*nsite` rows and columns, including spin offsets;
    /// normal orbitals use `nsite`. Built after parsing or on demand by
    /// [`ExpertModeData::ensure_orbital_idx_matrix`].
    pub orbital_idx_matrix: Option<Vec<Vec<i64>>>,
    /// `OrbitalSgn[ri+1, rj+1]`, same shape as `orbital_idx_matrix`.
    pub orbital_sgn_matrix: Option<Vec<Vec<i64>>>,

    /// `OptFlag[2*i + spin]` flags. `true` -> optimised, `false` -> fixed.
    /// Mirrors `optimization_flags::Vector{Bool}` in upstream. Empty
    /// until `vmc_para_opt` populates it on the first step.
    pub optimization_flags: Vec<bool>,

    /// Optional authoritative runtime ComplexType flags. An empty vector
    /// selects inference from factor declarations and current values.
    pub complex_flags: Vec<i64>,

    /// Quantum-projection weights (`init_qp_weight!`).
    /// `None` until [`crate::utils::qp_weight::init_qp_weight`] runs.
    pub qp_weights: Option<QuantumProjectionWeights>,
}

impl ExpertModeData {
    /// Construct an empty `ExpertModeData`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Julia's projection layout: reserve declared widths even with sparse terms.
    pub fn projection_layout(&self) -> ProjectionLayout {
        let n_gutzwiller = if self.n_gutzwiller_idx > 0 {
            self.n_gutzwiller_idx as usize
        } else {
            self.gutzwiller_terms.len()
        };
        let n_jastrow = if self.n_jastrow_idx > 0 {
            self.n_jastrow_idx as usize
        } else {
            self.jastrow_terms.len()
        };
        ProjectionLayout {
            n_gutzwiller,
            n_jastrow,
            gutzwiller_offset: 0,
            jastrow_offset: n_gutzwiller,
            n_proj: n_gutzwiller + n_jastrow,
        }
    }

    /// Pack projection values at their declared offsets, filling reserved slots with zero.
    pub fn projection_parameters(&self) -> Vec<Complex64> {
        let layout = self.projection_layout();
        let mut values = vec![Complex64::new(0.0, 0.0); layout.n_proj];
        for (i, term) in self
            .gutzwiller_terms
            .iter()
            .take(layout.n_gutzwiller)
            .enumerate()
        {
            values[layout.gutzwiller_offset + i] = term.value;
        }
        for (i, term) in self.jastrow_terms.iter().take(layout.n_jastrow).enumerate() {
            values[layout.jastrow_offset + i] = term.value;
        }
        values
    }

    /// Normalize a zero translation count to one before projection setup.
    /// Keep negative counts as the original antiperiodic boundary marker;
    /// allocation and kernels use their absolute value.
    pub fn normalize_projection_count(&mut self) {
        if self.modpara.nmp_trans == 0 {
            self.modpara.nmp_trans = 1;
        }
    }

    /// Populate `optimization_flags` with `true` entries for every
    /// (real, imag) slot of the `n_para` variational parameters. Mirrors
    /// the `if isempty(data.optimization_flags) ... fill!(true, 2*n_para)`
    /// guard at the top of upstream `vmc_para_opt!` and `stochastic_opt!`.
    pub fn ensure_optimization_flags(&mut self, n_para: usize) {
        if self.optimization_flags.is_empty() {
            self.optimization_flags = vec![true; 2 * n_para];
        }
    }

    /// Build (and cache) the dense `OrbitalIdx[ri+1, rj+1]` lookup matrix
    /// for the current `orbital_terms`. Mirrors
    /// `MVMCExpertModeParsers.jl/src/utils/orbital_qptrans_utils.jl ::
    /// build_orbital_sgn_matrix!`.
    ///
    /// Cells without an entry have index zero. Signs follow the original
    /// boundary condition, including Julia's periodic sign override.
    pub fn ensure_orbital_idx_matrix(&mut self) {
        if self.orbital_idx_matrix.is_some() && self.orbital_sgn_matrix.is_some() {
            return;
        }
        let (idx, sgn) = self.build_orbital_matrices();
        self.orbital_idx_matrix = Some(idx);
        self.orbital_sgn_matrix = Some(sgn);
    }

    /// Construct Julia's normal or spin-resolved orbital index/sign matrices.
    ///
    /// Normal definitions fill only explicitly listed cells. General
    /// definitions use spin-site indices and fill their antisymmetric
    /// counterpart; AP/P definitions use the recorded AP block boundary.
    pub fn build_orbital_matrices(&self) -> (Vec<Vec<i64>>, Vec<Vec<i64>>) {
        let nsite = self.modpara.nsite.max(0) as usize;
        let general = self.i_flg_orbital_general != 0;
        let pure_general =
            general && self.i_flg_orbital_anti_parallel == 0 && self.i_flg_orbital_parallel == 0;
        let size = if general { 2 * nsite } else { nsite };
        let mut idx = vec![vec![0_i64; size]; size];
        let mut sgn = vec![vec![0_i64; size]; size];
        let ap_count = if self.i_flg_orbital_anti_parallel == 1 && self.i_flg_orbital_parallel == 1
        {
            self.n_orbital_anti_parallel
        } else {
            0
        };
        for term in &self.orbital_terms {
            if term.site1 < 0 || term.site2 < 0 {
                continue;
            }
            let mut ri = term.site1 as usize;
            let mut rj = term.site2 as usize;
            if ri >= size || rj >= size {
                continue;
            }
            if general && !pure_general {
                if ri >= nsite || rj >= nsite {
                    continue;
                }
                if term.idx < ap_count {
                    rj += nsite;
                } else if (term.idx - ap_count) % 2 == 1 {
                    ri += nsite;
                    rj += nsite;
                }
            }
            idx[ri][rj] = term.idx;
            sgn[ri][rj] = term.sign;
            if general && ri != rj {
                idx[rj][ri] = term.idx;
                sgn[rj][ri] = -term.sign;
            }
        }
        if self.modpara.nmp_trans >= 0 {
            if general {
                for (i, row) in sgn.iter_mut().enumerate() {
                    row[..i].fill(-1);
                    row[i + 1..].fill(1);
                }
            } else {
                for row in &mut sgn {
                    row.fill(1);
                }
            }
        }
        (idx, sgn)
    }
}

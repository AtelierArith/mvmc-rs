//! Expert-mode data types.
//!
//! Port target: `MVMCExpertModeParsers.jl/src/types/expert_types.jl`.
//!
//! Phase 3 status: the round-trip subset is implemented (`ModPara`, the
//! simple term structs, DH2/DH4 definitions, nine RBM mappings and
//! `ExpertModeData`). The upstream backflow tree remains pending.

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
    /// SpinJastrow count (zero; unsupported by the canonical runtime).
    pub n_spinjastrow: usize,
    /// Number of DH2 neighbor-definition tables, with six parameters each.
    pub n_dh2: usize,
    /// Number of DH4 neighbor-definition tables, with ten parameters each.
    pub n_dh4: usize,
    /// Start of Gutzwiller parameters (zero).
    pub gutzwiller_offset: usize,
    /// Start of Jastrow parameters after the declared Gutzwiller block.
    pub jastrow_offset: usize,
    /// Start of the empty SpinJastrow block.
    pub spinjastrow_offset: usize,
    /// Start of DH2 parameters after Gutzwiller/Jastrow.
    pub dh2_offset: usize,
    /// Start of DH4 parameters after the six-component DH2 blocks.
    pub dh4_offset: usize,
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

/// Directed pair hopping `value * c†(site1,up) c(site2,up)
/// c†(site1,down) c(site2,down)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairHopTerm {
    /// Pair destination (0-based).
    pub site1: i64,
    /// Pair source (0-based).
    pub site2: i64,
    /// Real coupling; each input row supplies both directed terms.
    pub value: f64,
}

/// General interaction `value * c†(site0,spin0) c(site1,spin1)
/// c†(site2,spin2) c(site3,spin3)`, in the input operator order.
///
/// Production C parsing validates site/spin bounds and fixed-TwoSz pairs.
/// Coefficients do not select the wavefunction mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InterAllTerm {
    /// First creation operator's site (0-based).
    pub site0: i64,
    /// First creation operator's spin (0 = up, 1 = down).
    pub spin0: i64,
    /// First annihilation operator's site (0-based).
    pub site1: i64,
    /// First annihilation operator's spin.
    pub spin1: i64,
    /// Second creation operator's site (0-based).
    pub site2: i64,
    /// Second creation operator's spin.
    pub spin2: i64,
    /// Second annihilation operator's site (0-based).
    pub site3: i64,
    /// Second annihilation operator's spin.
    pub spin3: i64,
    /// Complex coupling, including signed zeros.
    pub value: Complex64,
    /// Julia's coefficient classification: `abs(imag(value)) > 1e-14`.
    pub is_complex: bool,
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

/// Factored two-body Green-function term (`greentwoex.def`). The second
/// one-body factor is stored in C's canonical order `(x6, x7, x4, x5)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GreenTwoExTerm {
    /// First factor creation site.
    pub site1: i64,
    /// First factor creation spin.
    pub spin1: Spin,
    /// First factor annihilation site.
    pub site2: i64,
    /// First factor annihilation spin.
    pub spin2: Spin,
    /// Second factor creation site.
    pub site3: i64,
    /// Second factor creation spin.
    pub spin3: Spin,
    /// Second factor annihilation site.
    pub site4: i64,
    /// Second factor annihilation spin.
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

/// A failure to derive an inverse from the current forward translation map.
/// This query contract does not add rejection rules to the C-compatible reader.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum QPTransInverseError {
    /// The forward map contains no sites.
    #[error("cannot invert an empty translation map")]
    EmptyMapping,
    /// An original site maps to a negative translated-site index.
    #[error("translation origin {origin} has negative target {target}")]
    NegativeTarget {
        /// Zero-based original site.
        origin: usize,
        /// Signed translated-site index from the current forward map.
        target: i64,
    },
    /// A translated-site index cannot index the current forward map.
    #[error("translation origin {origin} has target {target} outside 0..{nsite}")]
    OutOfRangeTarget {
        /// Zero-based original site.
        origin: usize,
        /// Signed translated-site index from the current forward map.
        target: i64,
        /// Number of sites in the current forward map.
        nsite: usize,
    },
    /// Two original sites map to the same translated site.
    #[error("translation target {target} is shared by origins {first_origin} and {origin}")]
    DuplicateTarget {
        /// Zero-based translated site shared by both origins.
        target: usize,
        /// Zero-based origin first encountered for this target.
        first_origin: usize,
        /// Zero-based origin that repeats this target.
        origin: usize,
    },
}

impl QPTransEntry {
    /// Derive `inverse[site_map[origin]] = origin` from the current forward map.
    ///
    /// The map must be a nonempty permutation of `0..site_map.len()`. Empty,
    /// negative, out-of-range and repeated targets return a specific error;
    /// no identity fallback or partial inverse is returned. With `n` entries,
    /// unique in-range targets also guarantee that no target is missing.
    ///
    /// This allocates in O(n) time and space on every query, so mutations of the
    /// public forward map cannot leave a cached inverse stale. Weights and signs
    /// are neither read nor changed. This does not validate the input reader or
    /// implement BackFlow; it exposes the inverse relation used by C/Julia.
    pub fn inverse_site_map(&self) -> Result<Vec<usize>, QPTransInverseError> {
        let nsite = self.site_map.len();
        if nsite == 0 {
            return Err(QPTransInverseError::EmptyMapping);
        }
        let mut inverse = vec![usize::MAX; nsite];
        for (origin, &target) in self.site_map.iter().enumerate() {
            if target < 0 {
                return Err(QPTransInverseError::NegativeTarget { origin, target });
            }
            let translated = usize::try_from(target)
                .ok()
                .filter(|&site| site < nsite)
                .ok_or(QPTransInverseError::OutOfRangeTarget {
                    origin,
                    target,
                    nsite,
                })?;
            let first_origin = inverse[translated];
            if first_origin != usize::MAX {
                return Err(QPTransInverseError::DuplicateTarget {
                    target: translated,
                    first_origin,
                    origin,
                });
            }
            inverse[translated] = origin;
        }
        Ok(inverse)
    }

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

/// C-compatible DH2 neighbor table for one definition index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DoublonHolon2SiteIndex {
    /// One fixed-width row per center site, containing two neighbor site IDs.
    pub neighbors: Vec<[i64; 2]>,
}

/// Complete strict DH2 definition; optimization indices are intentionally ignored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DoublonHolon2SiteDefinition {
    /// Raw signed C header; local imaginary optimization writes require >0.
    pub complex_type: i32,
    /// Tables indexed by the final column of each neighbor row.
    pub indices: Vec<DoublonHolon2SiteIndex>,
    /// Six flags per table, in input row order.
    pub opt_flags: Vec<i64>,
    /// Whether the header's ComplexType integer is nonzero.
    pub is_complex: bool,
}

/// C-compatible DH4 neighbor table for one definition index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DoublonHolon4SiteIndex {
    /// One fixed-width row per center site, containing four neighbor site IDs.
    pub neighbors: Vec<[i64; 4]>,
}

/// Complete strict DH4 definition; optimization indices are intentionally ignored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DoublonHolon4SiteDefinition {
    /// Raw signed C header; local imaginary optimization writes require >0.
    pub complex_type: i32,
    /// Tables indexed by the final column of each neighbor row.
    pub indices: Vec<DoublonHolon4SiteIndex>,
    /// Ten flags per table, in input row order.
    pub opt_flags: Vec<i64>,
    /// Whether the header's ComplexType integer is nonzero.
    pub is_complex: bool,
}

/// Common indexed parameter access for the nine RBM term shapes.
pub trait RbmParameter {
    /// Section-local parameter index.
    fn idx(&self) -> i64;
    /// Current variational value.
    fn value(&self) -> Complex64;
    /// Assign a shared indexed variational value.
    fn set_value(&mut self, value: Complex64);
}

/// Charge RBM PhysLayer mapping, with raw zero-based coordinates as in Julia.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChargeRBMPhysLayerTerm {
    /// Raw `site` mapping coordinate.
    pub site: i64,
    /// Section-local parameter index.
    pub idx: i64,
    /// Variational value shared by entries with the same index.
    pub value: Complex64,
    /// Header ComplexType declaration.
    pub is_complex: bool,
}
impl RbmParameter for ChargeRBMPhysLayerTerm {
    fn idx(&self) -> i64 {
        self.idx
    }
    fn value(&self) -> Complex64 {
        self.value
    }
    fn set_value(&mut self, value: Complex64) {
        self.value = value;
    }
}

/// Spin RBM PhysLayer mapping, with raw zero-based coordinates as in Julia.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpinRBMPhysLayerTerm {
    /// Raw `site` mapping coordinate.
    pub site: i64,
    /// Section-local parameter index.
    pub idx: i64,
    /// Variational value shared by entries with the same index.
    pub value: Complex64,
    /// Header ComplexType declaration.
    pub is_complex: bool,
}
impl RbmParameter for SpinRBMPhysLayerTerm {
    fn idx(&self) -> i64 {
        self.idx
    }
    fn value(&self) -> Complex64 {
        self.value
    }
    fn set_value(&mut self, value: Complex64) {
        self.value = value;
    }
}

/// General RBM PhysLayer mapping, with raw zero-based coordinates as in Julia.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeneralRBMPhysLayerTerm {
    /// Raw `site` mapping coordinate.
    pub site: i64,
    /// Raw `spin` mapping coordinate.
    pub spin: i64,
    /// Section-local parameter index.
    pub idx: i64,
    /// Variational value shared by entries with the same index.
    pub value: Complex64,
    /// Header ComplexType declaration.
    pub is_complex: bool,
}
impl RbmParameter for GeneralRBMPhysLayerTerm {
    fn idx(&self) -> i64 {
        self.idx
    }
    fn value(&self) -> Complex64 {
        self.value
    }
    fn set_value(&mut self, value: Complex64) {
        self.value = value;
    }
}

/// Charge RBM HiddenLayer mapping, with raw zero-based coordinates as in Julia.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChargeRBMHiddenLayerTerm {
    /// Raw `site` mapping coordinate.
    pub site: i64,
    /// Section-local parameter index.
    pub idx: i64,
    /// Variational value shared by entries with the same index.
    pub value: Complex64,
    /// Header ComplexType declaration.
    pub is_complex: bool,
}
impl RbmParameter for ChargeRBMHiddenLayerTerm {
    fn idx(&self) -> i64 {
        self.idx
    }
    fn value(&self) -> Complex64 {
        self.value
    }
    fn set_value(&mut self, value: Complex64) {
        self.value = value;
    }
}

/// Spin RBM HiddenLayer mapping, with raw zero-based coordinates as in Julia.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpinRBMHiddenLayerTerm {
    /// Raw `site` mapping coordinate.
    pub site: i64,
    /// Section-local parameter index.
    pub idx: i64,
    /// Variational value shared by entries with the same index.
    pub value: Complex64,
    /// Header ComplexType declaration.
    pub is_complex: bool,
}
impl RbmParameter for SpinRBMHiddenLayerTerm {
    fn idx(&self) -> i64 {
        self.idx
    }
    fn value(&self) -> Complex64 {
        self.value
    }
    fn set_value(&mut self, value: Complex64) {
        self.value = value;
    }
}

/// General RBM HiddenLayer mapping, with raw zero-based coordinates as in Julia.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeneralRBMHiddenLayerTerm {
    /// Raw `site` mapping coordinate.
    pub site: i64,
    /// Section-local parameter index.
    pub idx: i64,
    /// Variational value shared by entries with the same index.
    pub value: Complex64,
    /// Header ComplexType declaration.
    pub is_complex: bool,
}
impl RbmParameter for GeneralRBMHiddenLayerTerm {
    fn idx(&self) -> i64 {
        self.idx
    }
    fn value(&self) -> Complex64 {
        self.value
    }
    fn set_value(&mut self, value: Complex64) {
        self.value = value;
    }
}

/// Charge RBM PhysHidden mapping, with raw zero-based coordinates as in Julia.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChargeRBMPhysHiddenTerm {
    /// Raw `site1` mapping coordinate.
    pub site1: i64,
    /// Raw `site2` mapping coordinate.
    pub site2: i64,
    /// Section-local parameter index.
    pub idx: i64,
    /// Variational value shared by entries with the same index.
    pub value: Complex64,
    /// Header ComplexType declaration.
    pub is_complex: bool,
}
impl RbmParameter for ChargeRBMPhysHiddenTerm {
    fn idx(&self) -> i64 {
        self.idx
    }
    fn value(&self) -> Complex64 {
        self.value
    }
    fn set_value(&mut self, value: Complex64) {
        self.value = value;
    }
}

/// Spin RBM PhysHidden mapping, with raw zero-based coordinates as in Julia.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpinRBMPhysHiddenTerm {
    /// Raw `site1` mapping coordinate.
    pub site1: i64,
    /// Raw `site2` mapping coordinate.
    pub site2: i64,
    /// Section-local parameter index.
    pub idx: i64,
    /// Variational value shared by entries with the same index.
    pub value: Complex64,
    /// Header ComplexType declaration.
    pub is_complex: bool,
}
impl RbmParameter for SpinRBMPhysHiddenTerm {
    fn idx(&self) -> i64 {
        self.idx
    }
    fn value(&self) -> Complex64 {
        self.value
    }
    fn set_value(&mut self, value: Complex64) {
        self.value = value;
    }
}

/// General RBM PhysHidden mapping, with raw zero-based coordinates as in Julia.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeneralRBMPhysHiddenTerm {
    /// Raw `site1` mapping coordinate.
    pub site1: i64,
    /// Raw `spin` mapping coordinate.
    pub spin: i64,
    /// Raw `site2` mapping coordinate.
    pub site2: i64,
    /// Section-local parameter index.
    pub idx: i64,
    /// Variational value shared by entries with the same index.
    pub value: Complex64,
    /// Header ComplexType declaration.
    pub is_complex: bool,
}
impl RbmParameter for GeneralRBMPhysHiddenTerm {
    fn idx(&self) -> i64 {
        self.idx
    }
    fn value(&self) -> Complex64 {
        self.value
    }
    fn set_value(&mut self, value: Complex64) {
        self.value = value;
    }
}

/// Owned container for all parsed Expert-mode `.def` data.
#[derive(Debug, Clone)]
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
    /// Directed pair hopping, expanded forward/reverse in input order.
    pub pair_hop_terms: Vec<PairHopTerm>,
    /// General four-fermion interactions, in file order without deduplication.
    pub inter_all_terms: Vec<InterAllTerm>,

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

    /// DH2 tables in definition-index order; neighbor site IDs are zero-based.
    pub doublon_holon_2site_indices: Vec<DoublonHolon2SiteIndex>,
    /// Six complex parameters per DH2 table, in C projection order.
    pub doublon_holon_2site_params: Vec<Complex64>,
    /// Local real optimization flags in input row order.
    pub doublon_holon_2site_opt_flags: Vec<i64>,
    /// DH2 ComplexType declaration, including empty definitions.
    pub doublon_holon_2site_complex: bool,

    /// DH4 tables in definition-index order; neighbor site IDs are zero-based.
    pub doublon_holon_4site_indices: Vec<DoublonHolon4SiteIndex>,
    /// Ten complex parameters per DH4 table, in C projection order.
    pub doublon_holon_4site_params: Vec<Complex64>,
    /// Local real optimization flags in input row order.
    pub doublon_holon_4site_opt_flags: Vec<i64>,
    /// DH4 ComplexType declaration, including empty definitions.
    pub doublon_holon_4site_complex: bool,

    /// C declared widths in physical, hidden, physical-hidden family order.
    pub rbm_section_widths: [usize; 9],
    /// Complete C RBM coefficient array, including every unmapped slot.
    pub rbm_params: Vec<Complex64>,
    /// Charge RBM PhysLayer indexed mappings.
    pub charge_rbm_phys_layer_terms: Vec<ChargeRBMPhysLayerTerm>,
    /// Spin RBM PhysLayer indexed mappings.
    pub spin_rbm_phys_layer_terms: Vec<SpinRBMPhysLayerTerm>,
    /// General RBM PhysLayer indexed mappings.
    pub general_rbm_phys_layer_terms: Vec<GeneralRBMPhysLayerTerm>,
    /// Charge RBM HiddenLayer indexed mappings.
    pub charge_rbm_hidden_layer_terms: Vec<ChargeRBMHiddenLayerTerm>,
    /// Spin RBM HiddenLayer indexed mappings.
    pub spin_rbm_hidden_layer_terms: Vec<SpinRBMHiddenLayerTerm>,
    /// General RBM HiddenLayer indexed mappings.
    pub general_rbm_hidden_layer_terms: Vec<GeneralRBMHiddenLayerTerm>,
    /// Charge RBM PhysHidden indexed mappings.
    pub charge_rbm_phys_hidden_terms: Vec<ChargeRBMPhysHiddenTerm>,
    /// Spin RBM PhysHidden indexed mappings.
    pub spin_rbm_phys_hidden_terms: Vec<SpinRBMPhysHiddenTerm>,
    /// General RBM PhysHidden indexed mappings.
    pub general_rbm_phys_hidden_terms: Vec<GeneralRBMPhysHiddenTerm>,

    /// Orbital (site1, site2, idx, sign) entries.
    pub orbital_terms: Vec<OrbitalTerm>,
    /// C Slater parameter array in declared index order, including slots
    /// without any spatial mapping. Orbital terms contain only mappings.
    pub slater_params: Vec<Complex64>,
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
    /// Factored two-body Green measurements, in canonical C pair order.
    pub green_two_ex_terms: Vec<GreenTwoExTerm>,
    /// Indices of the two one-body factors for each TwoBodyGEx term.
    pub green_two_ex_indices: Vec<(usize, usize)>,

    /// `qptransidx.def` payload.
    pub qp_trans_entries: Vec<QPTransEntry>,
    /// `NQPTrans`.
    pub n_qp_trans: i64,

    /// `ParaQPTrans` values (real / complex weights from `qptransidx.def`).
    /// Convenience mirror of `qp_trans_entries[i].weight`, populated by
    /// `parse_expert_mode_files` so downstream Phase-4 code does not have
    /// to walk `qp_trans_entries`.
    pub para_qp_trans: Vec<num_complex::Complex64>,
    /// Number of sectors declared by an active `OptTrans` definition.
    pub n_qp_opt_trans: i64,
    /// Initial real optimized-translation weights from the definition.
    pub para_qp_opt_trans: Vec<Complex64>,
    /// Active optimized-translation parameters, after Slater in global order.
    pub opt_trans: Vec<Complex64>,
    /// Zero-based optimized-translation site maps, indexed by sector and site.
    pub qp_opt_trans: Vec<Vec<i64>>,
    /// Optimized-translation per-site signs, indexed by sector and site.
    pub qp_opt_trans_sgn: Vec<Vec<i64>>,

    /// `OrbitalIdx[ri+1, rj+1] = idx` lookup matrix (row-major). General
    /// orbitals use `2*nsite` rows and columns, including spin offsets;
    /// normal orbitals use `nsite`. Built after parsing or on demand by
    /// [`ExpertModeData::ensure_orbital_idx_matrix`].
    pub orbital_idx_matrix: Option<Vec<Vec<i64>>>,
    /// `OrbitalSgn[ri+1, rj+1]`, same shape as `orbital_idx_matrix`.
    pub orbital_sgn_matrix: Option<Vec<Vec<i64>>>,

    /// C `OptFlag[2*i + component]` integer flags for real/imaginary parts.
    /// Initialization tests the real flag against >0; SR and gauge selection
    /// require exactly 1. Definition readers populate this array before use.
    pub optimization_flags: Vec<i64>,

    /// Whether `optimization_flags` uses C's consecutive OptTrans writes.
    /// This is set only by the C-facing OptTrans parser.
    pub c_opt_trans_flags: bool,

    /// Optional authoritative runtime ComplexType flags. An empty vector
    /// selects inference from factor declarations and current values.
    pub complex_flags: Vec<i64>,

    /// Raw C header contributions, before orbital header normalization.
    /// Populated by public definition loading; absent for programmatic data.
    pub native_complex_headers: std::collections::BTreeMap<String, i32>,

    /// Loaded declaration snapshot, used to detect stale raw-header metadata.
    /// Coefficient values may change freely. To replace a declaration with
    /// programmatic data, clear both native metadata maps first.
    pub native_complex_declarations: std::collections::BTreeMap<String, bool>,

    /// Quantum-projection weights (`init_qp_weight!`).
    /// `None` until [`crate::utils::qp_weight::init_qp_weight`] runs.
    pub qp_weights: Option<QuantumProjectionWeights>,
}

impl Default for ExpertModeData {
    fn default() -> Self {
        Self {
            modpara: Default::default(),
            namelist: Default::default(),
            input_errors: Default::default(),
            transfer_terms: Default::default(),
            coulomb_intra_terms: Default::default(),
            coulomb_inter_terms: Default::default(),
            hund_terms: Default::default(),
            exchange_terms: Default::default(),
            pair_hop_terms: Default::default(),
            inter_all_terms: Default::default(),
            locspin_terms: Default::default(),
            gutzwiller_terms: Default::default(),
            n_gutzwiller_idx: Default::default(),
            gutzwiller_idx: Default::default(),
            jastrow_terms: Default::default(),
            n_jastrow_idx: Default::default(),
            jastrow_idx: Default::default(),
            doublon_holon_2site_indices: Default::default(),
            doublon_holon_2site_params: Default::default(),
            doublon_holon_2site_opt_flags: Default::default(),
            doublon_holon_2site_complex: Default::default(),
            doublon_holon_4site_indices: Default::default(),
            doublon_holon_4site_params: Default::default(),
            doublon_holon_4site_opt_flags: Default::default(),
            doublon_holon_4site_complex: Default::default(),
            rbm_section_widths: [0; 9],
            rbm_params: Default::default(),
            charge_rbm_phys_layer_terms: Default::default(),
            spin_rbm_phys_layer_terms: Default::default(),
            general_rbm_phys_layer_terms: Default::default(),
            charge_rbm_hidden_layer_terms: Default::default(),
            spin_rbm_hidden_layer_terms: Default::default(),
            general_rbm_hidden_layer_terms: Default::default(),
            charge_rbm_phys_hidden_terms: Default::default(),
            spin_rbm_phys_hidden_terms: Default::default(),
            general_rbm_phys_hidden_terms: Default::default(),
            orbital_terms: Default::default(),
            slater_params: Default::default(),
            i_flg_orbital_anti_parallel: Default::default(),
            i_flg_orbital_parallel: Default::default(),
            i_flg_orbital_general: Default::default(),
            n_orbital_anti_parallel: Default::default(),
            green_one_terms: Default::default(),
            green_two_terms: Default::default(),
            green_two_ex_terms: Default::default(),
            green_two_ex_indices: Default::default(),
            qp_trans_entries: Default::default(),
            n_qp_trans: Default::default(),
            para_qp_trans: Default::default(),
            n_qp_opt_trans: 1,
            para_qp_opt_trans: Default::default(),
            opt_trans: Default::default(),
            qp_opt_trans: Default::default(),
            qp_opt_trans_sgn: Default::default(),
            orbital_idx_matrix: Default::default(),
            orbital_sgn_matrix: Default::default(),
            optimization_flags: Default::default(),
            c_opt_trans_flags: false,
            complex_flags: Default::default(),
            native_complex_headers: Default::default(),
            native_complex_declarations: Default::default(),
            qp_weights: Default::default(),
        }
    }
}

impl ExpertModeData {
    /// Stable-deduplicate OneBodyG when TwoBodyGEx enables C's indirect reader,
    /// append missing constituents in TwoBodyGEx order, and retain
    /// references into the resulting canonical one-body list.
    pub fn canonicalize_green_two_ex(&mut self) {
        self.green_two_ex_indices.clear();
        if !self.green_two_ex_terms.is_empty() {
            let mut canonical = Vec::with_capacity(self.green_one_terms.len());
            for term in &self.green_one_terms {
                if !canonical.contains(term) {
                    canonical.push(*term);
                }
            }
            self.green_one_terms = canonical;
        }
        for term in self.green_two_ex_terms.clone() {
            let first = GreenOneTerm {
                site1: term.site1,
                spin1: term.spin1,
                site2: term.site2,
                spin2: term.spin2,
            };
            let second = GreenOneTerm {
                site1: term.site3,
                spin1: term.spin3,
                site2: term.site4,
                spin2: term.spin4,
            };
            let first_index = self
                .green_one_terms
                .iter()
                .position(|candidate| *candidate == first)
                .unwrap_or_else(|| {
                    self.green_one_terms.push(first);
                    self.green_one_terms.len() - 1
                });
            let second_index = self
                .green_one_terms
                .iter()
                .position(|candidate| *candidate == second)
                .unwrap_or_else(|| {
                    self.green_one_terms.push(second);
                    self.green_one_terms.len() - 1
                });
            self.green_two_ex_indices.push((first_index, second_index));
        }
    }

    /// Active optimized-translation parameter count, independent of mappings.
    pub fn count_opt_trans_parameters(&self) -> usize {
        self.opt_trans.len()
    }

    /// Total active width in projection, RBM, Slater and OptTrans order.
    pub fn count_variational_parameters(&self) -> usize {
        self.projection_layout().n_proj
            + self.count_rbm_parameters()
            + crate::utils::parameter_init::n_slater(self)
            + self.count_opt_trans_parameters()
    }

    /// Construct an empty `ExpertModeData`.
    pub fn new() -> Self {
        Self::default()
    }

    /// C declared coefficient widths, independent of spatial mappings.
    pub fn rbm_section_sizes(&self) -> [usize; 9] {
        self.rbm_section_widths
    }

    /// Complete coefficient storage in C block order, including unmapped slots.
    pub fn rbm_parameters(&self) -> Vec<Complex64> {
        self.rbm_params.clone()
    }

    /// Replace the complete RBM array and update all mapped kernel values.
    pub fn set_rbm_parameters(&mut self, values: Vec<Complex64>) {
        assert_eq!(values.len(), self.count_rbm_parameters());
        let sizes = self.rbm_section_sizes();
        let mut offsets = [0; 9];
        for index in 1..9 {
            offsets[index] = offsets[index - 1] + sizes[index - 1];
        }
        self.visit_rbm_terms_mut(|section, term| {
            let idx = term.idx();
            if idx >= 0 && (idx as usize) < sizes[section] {
                term.set_value(values[offsets[section] + idx as usize]);
            }
        });
        self.rbm_params = values;
    }

    /// Update one declared coefficient and every spatial mapping that uses it.
    pub fn set_rbm_parameter(&mut self, index: usize, value: Complex64) {
        self.rbm_params[index] = value;
        let sizes = self.rbm_section_sizes();
        let mut offsets = [0; 9];
        for i in 1..9 {
            offsets[i] = offsets[i - 1] + sizes[i - 1];
        }
        self.visit_rbm_terms_mut(|section, term| {
            if term.idx() >= 0 && offsets[section] + term.idx() as usize == index {
                term.set_value(value);
            }
        });
    }

    /// Whether any RBM mapping exists, independent of declared or inferred widths.
    pub fn has_rbm_terms(&self) -> bool {
        !self.charge_rbm_phys_layer_terms.is_empty()
            || !self.spin_rbm_phys_layer_terms.is_empty()
            || !self.general_rbm_phys_layer_terms.is_empty()
            || !self.charge_rbm_hidden_layer_terms.is_empty()
            || !self.spin_rbm_hidden_layer_terms.is_empty()
            || !self.general_rbm_hidden_layer_terms.is_empty()
            || !self.charge_rbm_phys_hidden_terms.is_empty()
            || !self.spin_rbm_phys_hidden_terms.is_empty()
            || !self.general_rbm_phys_hidden_terms.is_empty()
    }

    /// Total number of indexed RBM coefficients, including gaps in mappings.
    pub fn count_rbm_parameters(&self) -> usize {
        self.rbm_section_sizes().iter().sum()
    }

    /// Visit each mapping in canonical section and input-row order.
    pub fn visit_rbm_terms_mut(&mut self, mut visit: impl FnMut(usize, &mut dyn RbmParameter)) {
        for term in &mut self.charge_rbm_phys_layer_terms {
            visit(0, term);
        }
        for term in &mut self.spin_rbm_phys_layer_terms {
            visit(1, term);
        }
        for term in &mut self.general_rbm_phys_layer_terms {
            visit(2, term);
        }
        for term in &mut self.charge_rbm_hidden_layer_terms {
            visit(3, term);
        }
        for term in &mut self.spin_rbm_hidden_layer_terms {
            visit(4, term);
        }
        for term in &mut self.general_rbm_hidden_layer_terms {
            visit(5, term);
        }
        for term in &mut self.charge_rbm_phys_hidden_terms {
            visit(6, term);
        }
        for term in &mut self.spin_rbm_phys_hidden_terms {
            visit(7, term);
        }
        for term in &mut self.general_rbm_phys_hidden_terms {
            visit(8, term);
        }
    }

    /// Visit declared RBM coefficients in C's canonical nine-section order.
    pub fn visit_rbm_terms(&self, mut visit: impl FnMut(usize, Complex64)) {
        let mut offset = 0;
        for (section, width) in self.rbm_section_sizes().into_iter().enumerate() {
            for value in &self.rbm_params[offset..offset + width] {
                visit(section, *value);
            }
            offset += width;
        }
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
        let dh2_offset = n_gutzwiller + n_jastrow;
        let n_dh2 = self.doublon_holon_2site_indices.len();
        let dh4_offset = dh2_offset + 6 * n_dh2;
        let n_dh4 = self.doublon_holon_4site_indices.len();
        ProjectionLayout {
            n_gutzwiller,
            n_jastrow,
            n_spinjastrow: 0,
            n_dh2,
            n_dh4,
            gutzwiller_offset: 0,
            jastrow_offset: n_gutzwiller,
            spinjastrow_offset: dh2_offset,
            dh2_offset,
            dh4_offset,
            n_proj: dh4_offset + 10 * n_dh4,
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
        for (i, &value) in self
            .doublon_holon_2site_params
            .iter()
            .take(6 * layout.n_dh2)
            .enumerate()
        {
            values[layout.dh2_offset + i] = value;
        }
        for (i, &value) in self
            .doublon_holon_4site_params
            .iter()
            .take(10 * layout.n_dh4)
            .enumerate()
        {
            values[layout.dh4_offset + i] = value;
        }
        values
    }

    /// Populate `optimization_flags` with `1` entries for every
    /// (real, imag) slot of the `n_para` variational parameters. Mirrors
    /// the `if isempty(data.optimization_flags) ... fill!(true, 2*n_para)`
    /// guard at the top of upstream `vmc_para_opt!` and `stochastic_opt!`.
    pub fn ensure_optimization_flags(&mut self, n_para: usize) {
        if self.optimization_flags.is_empty() {
            self.optimization_flags = vec![1; 2 * n_para];
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

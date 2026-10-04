//! Julia-compatible structured consistency checks.
//!
//! These APIs are explicit and read-only, as in upstream `validation.jl`.
//! Neither parsing nor the VMC runner invokes the combined validator
//! automatically. Runtime support/rejection checks belong to `mvmc-core`.
//! NaNs, repeated Hamiltonian terms, and orbital parameter indices are not
//! rejected by these upstream validators. Strict parameter loading is a
//! separate operation. Rust's `Spin` type guarantees valid 0/1 spin codes.
//!

use crate::types::{
    ChargeRBMHiddenLayerTerm, ChargeRBMPhysHiddenTerm, ChargeRBMPhysLayerTerm, CoulombInterTerm,
    CoulombIntraTerm, DoublonHolon2SiteIndex, DoublonHolon4SiteIndex, ExpertModeData,
    GeneralRBMHiddenLayerTerm, GeneralRBMPhysHiddenTerm, GeneralRBMPhysLayerTerm, GutzwillerTerm,
    JastrowTerm, ModParaParameters, OrbitalTerm, SpinRBMHiddenLayerTerm, SpinRBMPhysHiddenTerm,
    SpinRBMPhysLayerTerm, TransferTerm, ValidationResult,
};

/// Validate ModPara dimensions, electron/spin consistency, and VMC/SR settings.
pub fn validate_modpara_params(p: &ModParaParameters) -> ValidationResult {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();
    for (bad, message) in [
        (p.nsite <= 0, "NSite must be positive"),
        (p.nelec < 0, "NElec must be non-negative"),
        (p.nlocspin < 0, "NLocSpin must be non-negative"),
        (
            p.ncond != -1 && p.ncond < 0,
            "NCond must be non-negative or -1",
        ),
    ] {
        if bad {
            errors.push(message.to_owned());
        }
    }
    if p.ncond != -1 {
        if p.ncond % 2 != 0 {
            errors.push("NCond must be even".into());
        }
        let expected = p.nlocspin.wrapping_add(p.ncond) / 2;
        if p.nelec != expected {
            warnings.push(format!(
                "NElec ({}) differs from expected value ({expected}) based on NCond",
                p.nelec
            ));
        }
    }
    if p.two_sz != 0 && p.two_sz % 2 != 0 && p.two_sz != -1 {
        errors.push("2Sz must be even or -1".into());
    }
    let twice_nelec = p.nelec.wrapping_mul(2);
    if p.nlocspin > 0 {
        if p.nlocspin == twice_nelec && p.nex_update_path != 2 {
            errors.push("NExUpdatePath must be 2 when 2*Ne = NLocalSpin (spin system)".into());
        } else if p.nex_update_path == 0 {
            errors.push("NExUpdatePath must be 1".into());
        } else if p.nlocspin > twice_nelec {
            errors.push("2*Ne must satisfy 2*Ne >= NLocalSpin".into());
        }
    }
    for (bad, message) in [
        (p.nvmc_sample <= 0, "NVMCSample must be positive"),
        (p.nvmc_interval <= 0, "NVMCInterval must be positive"),
        (p.nvmc_warmup < 0, "NVMCWarmUp must be non-negative"),
        (p.nsr_opt_itr_step <= 0, "NSROptItrStep must be positive"),
        (p.nsr_opt_itr_smp <= 0, "NSROptItrSmp must be positive"),
        (p.dsr_opt_red_cut < 0.0, "DSROptRedCut must be non-negative"),
        (p.dsr_opt_step_dt <= 0.0, "DSROptStepDt must be positive"),
    ] {
        if bad {
            errors.push(message.to_owned());
        }
    }
    if p.nblock_size_rbm_ratio > 0 && p.nblock_size_rbm_ratio % 8 != 0 {
        warnings.push("NBlockSize_RBMRatio should be multiple of 8".into());
    }
    ValidationResult::new(errors, warnings)
}

fn check_site(
    errors: &mut Vec<String>,
    family: &str,
    term: usize,
    field: &str,
    site: i64,
    nsite: i64,
) {
    if site < 0 || site >= nsite {
        errors.push(format!(
            "{family} term {term}: {field} ({site}) out of range [0, {}]",
            nsite.wrapping_sub(1)
        ));
    }
}

fn float_string(value: f64) -> String {
    // Julia includes .0 for integral Float64 values and capitalizes infinities.
    match value {
        f64::INFINITY => "Inf".into(),
        f64::NEG_INFINITY => "-Inf".into(),
        _ => format!("{value:?}"),
    }
}

/// Validate Transfer site bounds and warn about diagonal/spin-changing terms.
pub fn validate_transfer_terms(terms: &[TransferTerm], nsite: i64) -> ValidationResult {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();
    for (i, term) in terms.iter().enumerate() {
        let i = i + 1;
        check_site(&mut errors, "Transfer", i, "site1", term.site1, nsite);
        check_site(&mut errors, "Transfer", i, "site2", term.site2, nsite);
        if term.site1 == term.site2 {
            warnings.push(format!("Transfer term {i}: site1 == site2 (diagonal term)"));
        }
        if term.spin1 != term.spin2 {
            warnings.push(format!(
                "Transfer term {i}: spin indices differ ({} != {})",
                term.spin1.as_code(),
                term.spin2.as_code()
            ));
        }
    }
    ValidationResult::new(errors, warnings)
}

/// Validate CoulombIntra bounds; negative coefficients are warnings.
pub fn validate_coulomb_intra_terms(terms: &[CoulombIntraTerm], nsite: i64) -> ValidationResult {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();
    for (i, term) in terms.iter().enumerate() {
        let i = i + 1;
        check_site(&mut errors, "CoulombIntra", i, "site", term.site, nsite);
        if term.value < 0.0 {
            warnings.push(format!(
                "CoulombIntra term {i}: negative value {}",
                float_string(term.value)
            ));
        }
    }
    ValidationResult::new(errors, warnings)
}

/// Validate CoulombInter bounds; on-site and negative terms are warnings.
pub fn validate_coulomb_inter_terms(terms: &[CoulombInterTerm], nsite: i64) -> ValidationResult {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();
    for (i, term) in terms.iter().enumerate() {
        let i = i + 1;
        check_site(&mut errors, "CoulombInter", i, "site1", term.site1, nsite);
        check_site(&mut errors, "CoulombInter", i, "site2", term.site2, nsite);
        if term.site1 == term.site2 {
            warnings.push(format!(
                "CoulombInter term {i}: site1 == site2 (on-site term)"
            ));
        }
        if term.value < 0.0 {
            warnings.push(format!(
                "CoulombInter term {i}: negative value {}",
                float_string(term.value)
            ));
        }
    }
    ValidationResult::new(errors, warnings)
}

/// Validate Gutzwiller bounds; negative real parts are warnings.
pub fn validate_gutzwiller_terms(terms: &[GutzwillerTerm], nsite: i64) -> ValidationResult {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();
    for (i, term) in terms.iter().enumerate() {
        let i = i + 1;
        check_site(&mut errors, "Gutzwiller", i, "site", term.site, nsite);
        if term.value.re < 0.0 {
            warnings.push(format!(
                "Gutzwiller term {i}: negative real part {}",
                float_string(term.value.re)
            ));
        }
    }
    ValidationResult::new(errors, warnings)
}

/// Validate Jastrow bounds and warn about diagonal terms.
pub fn validate_jastrow_terms(terms: &[JastrowTerm], nsite: i64) -> ValidationResult {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();
    for (i, term) in terms.iter().enumerate() {
        let i = i + 1;
        check_site(&mut errors, "Jastrow", i, "site1", term.site1, nsite);
        check_site(&mut errors, "Jastrow", i, "site2", term.site2, nsite);
        if term.site1 == term.site2 {
            warnings.push(format!("Jastrow term {i}: site1 == site2 (diagonal term)"));
        }
    }
    ValidationResult::new(errors, warnings)
}

/// Validate orbital site coordinates against the supplied bound.
/// Julia's combined validator passes Nsite even for General definitions;
/// callers wanting spin-site bounds can explicitly pass 2*Nsite here.
pub fn validate_orbital_terms(terms: &[OrbitalTerm], nsite: i64) -> ValidationResult {
    let mut errors = Vec::new();
    for (i, term) in terms.iter().enumerate() {
        check_site(&mut errors, "Orbital", i + 1, "site1", term.site1, nsite);
        check_site(&mut errors, "Orbital", i + 1, "site2", term.site2, nsite);
    }
    ValidationResult::new(errors, Vec::new())
}

/// Validate strict DH2 table shape and neighbors in definition/site/column order.
pub fn validate_doublon_holon_2site_indices(
    indices: &[DoublonHolon2SiteIndex],
    nsite: i64,
) -> ValidationResult {
    let mut errors = Vec::new();
    for (index, table) in indices.iter().enumerate() {
        if table.neighbors.len() as i64 != nsite {
            errors.push(format!("DH2 index {index}: neighbors must be {nsite} x 2"));
            continue;
        }
        for (site, row) in table.neighbors.iter().enumerate() {
            for (column, &neighbor) in row.iter().enumerate() {
                if !(0..nsite).contains(&neighbor) {
                    errors.push(format!("DH2 index {index} site {site} neighbor {column}={neighbor} out of range [0, {}]",nsite-1));
                }
            }
        }
    }
    ValidationResult::new(errors, Vec::new())
}

/// Validate strict DH4 table shape and neighbors in definition/site/column order.
pub fn validate_doublon_holon_4site_indices(
    indices: &[DoublonHolon4SiteIndex],
    nsite: i64,
) -> ValidationResult {
    let mut errors = Vec::new();
    for (index, table) in indices.iter().enumerate() {
        if table.neighbors.len() as i64 != nsite {
            errors.push(format!("DH4 index {index}: neighbors must be {nsite} x 4"));
            continue;
        }
        for (site, row) in table.neighbors.iter().enumerate() {
            for (column, &neighbor) in row.iter().enumerate() {
                if !(0..nsite).contains(&neighbor) {
                    errors.push(format!("DH4 index {index} site {site} neighbor {column}={neighbor} out of range [0, {}]",nsite-1));
                }
            }
        }
    }
    ValidationResult::new(errors, Vec::new())
}

fn large_rbm_shadow(value: num_complex::Complex64) -> bool {
    // Julia hypot checks infinity before NaN. Preserve this discrete decision
    // independently of a platform's mixed Inf/NaN hypot behavior.
    if value.re.is_infinite() || value.im.is_infinite() {
        true
    } else if value.re.is_nan() || value.im.is_nan() {
        false
    } else {
        value.norm() > 1e10
    }
}

fn validate_rbm_shadows<const N: usize>(
    family: &str,
    fields: [&str; N],
    terms: impl Iterator<Item = ([i64; N], num_complex::Complex64)>,
    nsite: i64,
) -> ValidationResult {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();
    for (row, (sites, value)) in terms.enumerate() {
        for (field, site) in fields.iter().zip(sites) {
            check_site(&mut errors, family, row + 1, field, site, nsite);
        }
        if large_rbm_shadow(value) {
            warnings.push(format!(
                "{family} term {}: very large value {value:?}",
                row + 1
            ));
        }
    }
    ValidationResult::new(errors, warnings)
}

/// Manually check Charge PhysLayer sites and warn about large shadow values.
/// Does not validate dense storage, indices, finiteness or shadow consistency.
pub fn validate_charge_rbm_phys_layer_terms(
    terms: &[ChargeRBMPhysLayerTerm],
    nsite: i64,
) -> ValidationResult {
    validate_rbm_shadows(
        "ChargeRBM_PhysLayer",
        ["site"],
        terms.iter().map(|t| ([t.site], t.value)),
        nsite,
    )
}

/// Manually check Spin PhysLayer sites and warn about large shadow values.
/// Does not validate dense storage, indices, finiteness or shadow consistency.
pub fn validate_spin_rbm_phys_layer_terms(
    terms: &[SpinRBMPhysLayerTerm],
    nsite: i64,
) -> ValidationResult {
    validate_rbm_shadows(
        "SpinRBM_PhysLayer",
        ["site"],
        terms.iter().map(|t| ([t.site], t.value)),
        nsite,
    )
}

/// Manually check General PhysLayer sites and warn about large shadow values.
/// Spin and parameter index are outside this manual diagnostic contract.
pub fn validate_general_rbm_phys_layer_terms(
    terms: &[GeneralRBMPhysLayerTerm],
    nsite: i64,
) -> ValidationResult {
    validate_rbm_shadows(
        "GeneralRBM_PhysLayer",
        ["site"],
        terms.iter().map(|t| ([t.site], t.value)),
        nsite,
    )
}

/// Manually check Charge HiddenLayer sites against diagnostic `nsite`.
/// This Julia architecture utility is not C hidden-neuron input validation.
pub fn validate_charge_rbm_hidden_layer_terms(
    terms: &[ChargeRBMHiddenLayerTerm],
    nsite: i64,
) -> ValidationResult {
    validate_rbm_shadows(
        "ChargeRBM_HiddenLayer",
        ["site"],
        terms.iter().map(|t| ([t.site], t.value)),
        nsite,
    )
}

/// Manually check Spin HiddenLayer sites against diagnostic `nsite`.
/// This Julia architecture utility is not C hidden-neuron input validation.
pub fn validate_spin_rbm_hidden_layer_terms(
    terms: &[SpinRBMHiddenLayerTerm],
    nsite: i64,
) -> ValidationResult {
    validate_rbm_shadows(
        "SpinRBM_HiddenLayer",
        ["site"],
        terms.iter().map(|t| ([t.site], t.value)),
        nsite,
    )
}

/// Manually check General HiddenLayer sites against diagnostic `nsite`.
/// This Julia architecture utility is not C hidden-neuron input validation.
pub fn validate_general_rbm_hidden_layer_terms(
    terms: &[GeneralRBMHiddenLayerTerm],
    nsite: i64,
) -> ValidationResult {
    validate_rbm_shadows(
        "GeneralRBM_HiddenLayer",
        ["site"],
        terms.iter().map(|t| ([t.site], t.value)),
        nsite,
    )
}

/// Manually check Charge PhysHidden sites and warn about large shadow values.
/// Both sites use diagnostic `nsite`, not C hidden-neuron dimensions.
pub fn validate_charge_rbm_phys_hidden_terms(
    terms: &[ChargeRBMPhysHiddenTerm],
    nsite: i64,
) -> ValidationResult {
    validate_rbm_shadows(
        "ChargeRBM_PhysHidden",
        ["site1", "site2"],
        terms.iter().map(|t| ([t.site1, t.site2], t.value)),
        nsite,
    )
}

/// Manually check Spin PhysHidden sites and warn about large shadow values.
/// Both sites use diagnostic `nsite`, not C hidden-neuron dimensions.
pub fn validate_spin_rbm_phys_hidden_terms(
    terms: &[SpinRBMPhysHiddenTerm],
    nsite: i64,
) -> ValidationResult {
    validate_rbm_shadows(
        "SpinRBM_PhysHidden",
        ["site1", "site2"],
        terms.iter().map(|t| ([t.site1, t.site2], t.value)),
        nsite,
    )
}

/// Diagnose General RBM PhysHidden term values without changing input acceptance.
///
/// This manual, read-only utility observes caller-owned term values, including
/// shadows, not canonical dense RBM storage. It does not check shadow consistency,
/// spin, parameter index or finiteness, and does not repair any values.
/// Both sites use the supplied diagnostic `nsite`, following the Julia API.
/// The C loader's separate hidden-neuron dimension remains a different contract;
/// this utility is not invoked by parsing, packing or the runner.
/// Magnitudes above `1e10` are warnings, not errors. Infinity dominates NaN as
/// in Julia's complex `abs`; NaN without infinity does not warn.
pub fn validate_general_rbm_phys_hidden_terms(
    terms: &[GeneralRBMPhysHiddenTerm],
    nsite: i64,
) -> ValidationResult {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();
    for (row, term) in terms.iter().enumerate() {
        check_site(
            &mut errors,
            "GeneralRBM_PhysHidden",
            row + 1,
            "site1",
            term.site1,
            nsite,
        );
        check_site(
            &mut errors,
            "GeneralRBM_PhysHidden",
            row + 1,
            "site2",
            term.site2,
            nsite,
        );
        if large_rbm_shadow(term.value) {
            warnings.push(format!(
                "GeneralRBM_PhysHidden term {}: very large value {:?}",
                row + 1,
                term.value
            ));
        }
    }
    ValidationResult::new(errors, warnings)
}

/// Validate the complete C-declared RBM storage and all mapped slots.
pub fn validate_rbm_parameters(data: &ExpertModeData) -> ValidationResult {
    let mut errors = Vec::new();
    let expected = data.rbm_section_sizes().iter().sum::<usize>();
    if data.rbm_params.len() != expected {
        errors.push(format!(
            "RBM parameter storage has {} values; declared sections require {expected}",
            data.rbm_params.len()
        ));
    }
    for (index, value) in data.rbm_params.iter().enumerate() {
        if !value.re.is_finite() || !value.im.is_finite() {
            errors.push(format!("RBM parameter {index} must be finite"));
        }
    }
    let check_idx = |errors: &mut Vec<String>, family: &str, row: usize, idx: i64, width: usize| {
        if idx < 0 || idx as usize >= width {
            errors.push(format!(
                "{family} term {}: idx ({idx}) out of range [0, {}]",
                row + 1,
                width.saturating_sub(1)
            ));
        }
    };
    let check_site =
        |errors: &mut Vec<String>, family: &str, row: usize, field: &str, site: i64| {
            if site < 0 || site >= data.modpara.nsite {
                errors.push(format!(
                    "{family} term {}: {field} ({site}) out of range [0, {}]",
                    row + 1,
                    data.modpara.nsite.saturating_sub(1)
                ));
            }
        };
    let widths = data.rbm_section_sizes();
    for (row, term) in data.charge_rbm_phys_layer_terms.iter().enumerate() {
        check_idx(&mut errors, "ChargeRBM_PhysLayer", row, term.idx, widths[0]);
        check_site(&mut errors, "ChargeRBM_PhysLayer", row, "site", term.site);
    }
    for (row, term) in data.spin_rbm_phys_layer_terms.iter().enumerate() {
        check_idx(&mut errors, "SpinRBM_PhysLayer", row, term.idx, widths[1]);
        check_site(&mut errors, "SpinRBM_PhysLayer", row, "site", term.site);
    }
    for (row, term) in data.general_rbm_phys_layer_terms.iter().enumerate() {
        check_idx(
            &mut errors,
            "GeneralRBM_PhysLayer",
            row,
            term.idx,
            widths[2],
        );
        check_site(&mut errors, "GeneralRBM_PhysLayer", row, "site", term.site);
        if !(0..=1).contains(&term.spin) {
            errors.push(format!(
                "GeneralRBM_PhysLayer term {}: spin must be 0 or 1",
                row + 1
            ));
        }
    }
    for (row, term) in data.charge_rbm_hidden_layer_terms.iter().enumerate() {
        check_idx(
            &mut errors,
            "ChargeRBM_HiddenLayer",
            row,
            term.idx,
            widths[3],
        );
        check_site(&mut errors, "ChargeRBM_HiddenLayer", row, "site", term.site);
    }
    for (row, term) in data.spin_rbm_hidden_layer_terms.iter().enumerate() {
        check_idx(&mut errors, "SpinRBM_HiddenLayer", row, term.idx, widths[4]);
        check_site(&mut errors, "SpinRBM_HiddenLayer", row, "site", term.site);
    }
    for (row, term) in data.general_rbm_hidden_layer_terms.iter().enumerate() {
        check_idx(
            &mut errors,
            "GeneralRBM_HiddenLayer",
            row,
            term.idx,
            widths[5],
        );
        check_site(
            &mut errors,
            "GeneralRBM_HiddenLayer",
            row,
            "site",
            term.site,
        );
    }
    for (row, term) in data.charge_rbm_phys_hidden_terms.iter().enumerate() {
        check_idx(
            &mut errors,
            "ChargeRBM_PhysHidden",
            row,
            term.idx,
            widths[6],
        );
        check_site(
            &mut errors,
            "ChargeRBM_PhysHidden",
            row,
            "site1",
            term.site1,
        );
        check_site(
            &mut errors,
            "ChargeRBM_PhysHidden",
            row,
            "site2",
            term.site2,
        );
    }
    for (row, term) in data.spin_rbm_phys_hidden_terms.iter().enumerate() {
        check_idx(&mut errors, "SpinRBM_PhysHidden", row, term.idx, widths[7]);
        check_site(&mut errors, "SpinRBM_PhysHidden", row, "site1", term.site1);
        check_site(&mut errors, "SpinRBM_PhysHidden", row, "site2", term.site2);
    }
    for (row, term) in data.general_rbm_phys_hidden_terms.iter().enumerate() {
        check_idx(
            &mut errors,
            "GeneralRBM_PhysHidden",
            row,
            term.idx,
            widths[8],
        );
        check_site(
            &mut errors,
            "GeneralRBM_PhysHidden",
            row,
            "site1",
            term.site1,
        );
        check_site(
            &mut errors,
            "GeneralRBM_PhysHidden",
            row,
            "site2",
            term.site2,
        );
        if !(0..=1).contains(&term.spin) {
            errors.push(format!(
                "GeneralRBM_PhysHidden term {}: spin must be 0 or 1",
                row + 1
            ));
        }
    }
    ValidationResult::new(errors, Vec::new())
}

/// Check C-defined Green-function site bounds, including constructed typed data.
/// Spin codes are guaranteed by Rust's type; no C spin-validation claim follows.
pub fn validate_green_terms(data: &ExpertModeData) -> ValidationResult {
    let mut errors = Vec::new();
    let nsite = data.modpara.nsite;
    for (index, term) in data.green_one_terms.iter().enumerate() {
        for (field, site) in [("site1", term.site1), ("site2", term.site2)] {
            check_site(&mut errors, "OneBodyG", index + 1, field, site, nsite);
        }
    }
    for (index, term) in data.green_two_terms.iter().enumerate() {
        for (field, site) in [
            ("site1", term.site1),
            ("site2", term.site2),
            ("site3", term.site3),
            ("site4", term.site4),
        ] {
            check_site(&mut errors, "TwoBodyG", index + 1, field, site, nsite);
        }
    }
    for (index, term) in data.green_two_ex_terms.iter().enumerate() {
        for (field, site) in [
            ("site1", term.site1),
            ("site2", term.site2),
            ("site3", term.site3),
            ("site4", term.site4),
        ] {
            check_site(&mut errors, "TwoBodyGEx", index + 1, field, site, nsite);
        }
    }
    ValidationResult::new(errors, Vec::new())
}

/// Combine ModPara and supported term-family validators, preserving Julia order
/// for existing families and adding C Green-function site bounds afterward.
pub fn validate_expert_mode_data(data: &ExpertModeData) -> ValidationResult {
    let mut result = validate_modpara_params(&data.modpara);
    let nsite = data.modpara.nsite;
    for family in [
        validate_transfer_terms(&data.transfer_terms, nsite),
        validate_coulomb_intra_terms(&data.coulomb_intra_terms, nsite),
        validate_coulomb_inter_terms(&data.coulomb_inter_terms, nsite),
        validate_gutzwiller_terms(&data.gutzwiller_terms, nsite),
        validate_jastrow_terms(&data.jastrow_terms, nsite),
        validate_orbital_terms(&data.orbital_terms, nsite),
        validate_doublon_holon_2site_indices(&data.doublon_holon_2site_indices, nsite),
        validate_doublon_holon_4site_indices(&data.doublon_holon_4site_indices, nsite),
        validate_rbm_parameters(data),
        validate_green_terms(data),
    ] {
        result.errors.extend(family.errors);
        result.warnings.extend(family.warnings);
    }
    result.is_valid = result.errors.is_empty();
    result
}

/// Manually aggregate original Julia term diagnostics in source family order.
///
/// This opt-in, read-only API observes RBM term shadows, not canonical dense
/// storage. It does not check RBM indices, spin, finiteness or consistency and
/// does not repair values. Hidden coordinates use diagnostic `NSite`, separate
/// from C hidden-neuron dimensions. It is not invoked by loaders or runners.
/// Errors and warnings keep family/row/field order in separate vectors;
/// warnings alone remain valid. Green and dense storage checks belong to the
/// distinct existing [`validate_expert_mode_data`] API, not this aggregate.
pub fn validate_expert_mode_term_diagnostics(data: &ExpertModeData) -> ValidationResult {
    let mut result = validate_modpara_params(&data.modpara);
    let nsite = data.modpara.nsite;
    for family in [
        validate_transfer_terms(&data.transfer_terms, nsite),
        validate_coulomb_intra_terms(&data.coulomb_intra_terms, nsite),
        validate_coulomb_inter_terms(&data.coulomb_inter_terms, nsite),
        validate_gutzwiller_terms(&data.gutzwiller_terms, nsite),
        validate_jastrow_terms(&data.jastrow_terms, nsite),
        validate_orbital_terms(&data.orbital_terms, nsite),
        validate_charge_rbm_phys_layer_terms(&data.charge_rbm_phys_layer_terms, nsite),
        validate_spin_rbm_phys_layer_terms(&data.spin_rbm_phys_layer_terms, nsite),
        validate_general_rbm_phys_layer_terms(&data.general_rbm_phys_layer_terms, nsite),
        validate_charge_rbm_hidden_layer_terms(&data.charge_rbm_hidden_layer_terms, nsite),
        validate_spin_rbm_hidden_layer_terms(&data.spin_rbm_hidden_layer_terms, nsite),
        validate_general_rbm_hidden_layer_terms(&data.general_rbm_hidden_layer_terms, nsite),
        validate_charge_rbm_phys_hidden_terms(&data.charge_rbm_phys_hidden_terms, nsite),
        validate_spin_rbm_phys_hidden_terms(&data.spin_rbm_phys_hidden_terms, nsite),
        validate_general_rbm_phys_hidden_terms(&data.general_rbm_phys_hidden_terms, nsite),
        validate_doublon_holon_2site_indices(&data.doublon_holon_2site_indices, nsite),
        validate_doublon_holon_4site_indices(&data.doublon_holon_4site_indices, nsite),
    ] {
        result.errors.extend(family.errors);
        result.warnings.extend(family.warnings);
    }
    result.is_valid = result.errors.is_empty();
    result
}

//! Julia-compatible structured consistency checks.
//!
//! These APIs are explicit and read-only, as in upstream `validation.jl`.
//! Neither parsing nor the VMC runner invokes the combined validator
//! automatically. Runtime support/rejection checks belong to `mvmc-core`.
//! NaNs, repeated Hamiltonian terms, and orbital parameter indices are not
//! rejected by these upstream validators. Strict parameter loading is a
//! separate operation. Rust's `Spin` type guarantees valid 0/1 spin codes.
//!
//! RBM and DH4 validators remain pending production integration.

use crate::types::{
    CoulombInterTerm, CoulombIntraTerm, DoublonHolon2SiteIndex, ExpertModeData, GutzwillerTerm,
    JastrowTerm, ModParaParameters, OrbitalTerm, TransferTerm, ValidationResult,
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

/// Combine ModPara and currently supported term-family validators in Julia order.
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
    ] {
        result.errors.extend(family.errors);
        result.warnings.extend(family.warnings);
    }
    result.is_valid = result.errors.is_empty();
    result
}

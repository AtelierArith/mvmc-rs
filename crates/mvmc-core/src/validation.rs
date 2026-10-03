//! Runtime compatibility checks, separate from Expert-mode format parsing.
//!
//! Permanent restrictions follow Julia's `unsupported_inputs.jl`. Temporary
//! restrictions name the porting issue and must be removed when its full
//! production path passes deterministic Julia parity checks.

use mvmc_expert_parsers::{ExpertModeData, ModParaParameters};

/// Validate globally unsupported ModPara settings without mutating data.
pub fn validate_supported_modpara(p: &ModParaParameters) -> Result<(), String> {
    if p.nsplit_size < 1 {
        return Err(format!("NSplitSize must be >= 1; got {}", p.nsplit_size));
    }
    if !(0..=2).contains(&p.lanczos_mode) {
        return Err(format!(
            "NLanczosMode must be 0, 1, or 2; got {}",
            p.lanczos_mode
        ));
    }
    if p.nsrcg >= 2 {
        return Err("NSRCG >= 2 is not supported by Julia-mVMC; use NSRCG = 0 or 1".into());
    }
    if p.use_diag_scale != 0 {
        return Err("useDiagScale != 0 is not supported by Julia-mVMC".into());
    }
    if p.rescale_smat != 0 {
        return Err("RescaleSmat != 0 is not supported by Julia-mVMC".into());
    }
    Ok(())
}

/// Validate parameter-optimization entry points before initialization or IO.
pub fn validate_para_opt(data: &ExpertModeData) -> Result<(), String> {
    let p = &data.modpara;
    validate_supported_modpara(p)?;
    if p.lanczos_mode > 0 {
        return Err(
            "NLanczosMode > 0 is not supported for parameter optimization; use PhysCal".into(),
        );
    }
    if p.nsplit_size > 1 && p.nsrcg != 0 {
        return Err("NSplitSize > 1 with SR-CG is not supported by Julia-mVMC".into());
    }
    if p.vmc_calc_mode != 0 {
        return Err(format!(
            "NVMCCalMode={} cannot run parameter optimization; PhysCal is not implemented yet (issue #29)",
            p.vmc_calc_mode
        ));
    }
    if p.nsplit_size > 1
        && (data.n_qp_opt_trans.max(1) > 1
            || data.opt_trans.len() > 1
            || data.qp_opt_trans.len() > 1)
    {
        return Err(format!(
                "NSplitSize > 1 with NQPOptTrans > 1 / OptTrans is not supported: grouped QP-split sampling currently supports standard-projection NQPFull only (NQPOptTrans = 1), got NSplitSize = {}, NQPOptTrans = {}. Use NSplitSize = 1 for OptTrans-derived QP sectors.",
                p.nsplit_size, data.n_qp_opt_trans
            ));
    }
    let has_interall = !data.inter_all_terms.is_empty()
        || data.namelist.iter().any(|(kind, _)| kind == "InterAll");
    if has_interall {
        if data.i_flg_orbital_general == 0 {
            return Err("InterAll requires the general-orbital fixed-Sz path; non-FSZ InterAll is unsupported".into());
        }
        for (index, term) in data.inter_all_terms.iter().enumerate() {
            // Julia skips out-of-range sites before using any spin indices.
            if [term.site0, term.site1, term.site2, term.site3]
                .iter()
                .any(|&site| site < 0 || site >= p.nsite)
            {
                continue;
            }
            for (name, spin) in [
                ("spin0", term.spin0),
                ("spin1", term.spin1),
                ("spin2", term.spin2),
                ("spin3", term.spin3),
            ] {
                if !(0..=1).contains(&spin) {
                    return Err(format!(
                        "InterAll term {index}: {name} must be 0 or 1; got {spin}"
                    ));
                }
            }
        }
    }
    for (kind, _) in &data.namelist {
        let issue = match kind.as_str() {
            "SpinJastrow" => {
                return Err("SpinJastrow inputs are not supported by Julia-mVMC; projection layout would be wrong".into());
            }
            "PairHop" => None,
            "InterAll" => None,
            "DH2" | "DoublonHolon2Site" | "InDH2" => None,
            "DH4" | "DoublonHolon4Site" | "InDH4" => None,
            "OptTrans" | "InOptTrans" => None,
            "TwoBodyGEx" => Some(30),
            k if k.starts_with("ChargeRBM_")
                || k.starts_with("SpinRBM_")
                || k.starts_with("GeneralRBM_") =>
            {
                None
            }
            "InGutzwiller"
            | "InJastrow"
            | "InOrbital"
            | "InOrbitalAntiParallel"
            | "InOrbitalParallel"
            | "InOrbitalGeneral" => None,
            k if k.starts_with("InChargeRBM_")
                || k.starts_with("InSpinRBM_")
                || k.starts_with("InGeneralRBM_") =>
            {
                None
            }
            k if k.starts_with("In") => Some(20),
            "ModPara"
            | "LocSpin"
            | "Trans"
            | "CoulombIntra"
            | "CoulombInter"
            | "Hund"
            | "Exchange"
            | "Gutzwiller"
            | "Jastrow"
            | "Orbital"
            | "OrbitalAntiParallel"
            | "OrbitalParallel"
            | "OrbitalGeneral"
            | "OneBodyG"
            | "TwoBodyG"
            | "TransSym"
            | "QPTrans" => None,
            k => return Err(format!("unsupported namelist section {k}")),
        };
        if let Some(issue) = issue {
            return Err(format!("{kind} is not implemented yet (issue #{issue})"));
        }
    }
    if !data.input_errors.is_empty() {
        return Err(format!(
            "incomplete Expert input: {}",
            data.input_errors.join("; ")
        ));
    }
    // Avoid silently clamping malformed dimensions in the current runner.
    for (name, value) in [
        ("Nsite", p.nsite),
        ("NElec", p.nelec),
        ("NVMCWarmUp", p.nvmc_warmup),
        ("NSROptItrStep", p.nsr_opt_itr_step),
    ] {
        if value < 0 {
            return Err(format!("{name} must be nonnegative; got {value}"));
        }
    }
    for (name, value) in [
        ("NVMCSample", p.nvmc_sample),
        ("NVMCInterval", p.nvmc_interval),
    ] {
        if value <= 0 {
            return Err(format!("{name} must be positive; got {value}"));
        }
    }
    // C does not turn zero into an identity sector. Its Expert manual requires
    // one for no projection; reject zero before initialization or output.
    if p.nmp_trans == 0 {
        return Err(
            "NMPTrans must be nonzero; use 1 for no translation projection (mVMC C contract)"
                .into(),
        );
    }
    Ok(())
}

/// Validate PhysCal input combinations that are independent of the reducer.
pub fn validate_phys_cal(data: &ExpertModeData) -> Result<(), String> {
    let p = &data.modpara;
    validate_supported_modpara(p)?;
    if p.lanczos_mode > 0 {
        if p.nsplit_size > 1 {
            return Err(
                "NSplitSize > 1 with NLanczosMode > 0 is unsupported for PhysCal (issue #31)"
                    .into(),
            );
        }
        if data.i_flg_orbital_general != 0 {
            return Err(
                "Lanczos PhysCal for FSZ/general orbitals is not implemented yet (issue #31)"
                    .into(),
            );
        }
        if data
            .transfer_terms
            .iter()
            .any(|term| term.spin1 != term.spin2)
        {
            return Err(
                "spin-changing Transfer is unsupported for Lanczos PhysCal (issue #31)".into(),
            );
        }
        if !data.inter_all_terms.is_empty() {
            return Err("InterAll Lanczos PhysCal is not implemented yet (issue #31)".into());
        }
        if p.lanczos_mode > 1 && data.green_two_ex_terms.is_empty() {
            let mut seen = std::collections::HashSet::new();
            for term in &data.green_one_terms {
                let key = (term.site1, term.spin1, term.site2, term.spin2);
                if !seen.insert(key) {
                    return Err(format!(
                        "NLanczosMode = 2 does not support duplicate OneBodyG entries without TwoBodyGEx; duplicate=({}, {}, {}, {}) (issue #32)",
                        term.site1,
                        term.spin1.as_code(),
                        term.site2,
                        term.spin2.as_code()
                    ));
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
fn mpi_requested(get: impl Fn(&str) -> Option<String>) -> bool {
    if get("JULIA_MVMC_MPI").is_some_and(|v| v == "1")
        || get("OMPI_COMM_WORLD_SIZE").is_some()
        || get("PMIX_RANK").is_some()
    {
        return true;
    }
    let pmi_size = get("PMI_SIZE").and_then(|v| v.parse::<i64>().ok());
    let slurm_tasks = ["SLURM_NTASKS", "SLURM_NPROCS"]
        .iter()
        .find_map(|key| get(key).and_then(|v| v.parse::<i64>().ok()));
    pmi_size.is_some_and(|n| n > 1 || slurm_tasks.is_some_and(|n| n > 1))
}

#[cfg(test)]
mod tests {
    use super::mpi_requested;

    #[test]
    fn parsed_and_programmatic_rbm_terms_pass_runtime_validation() {
        use mvmc_expert_parsers::{
            parse_expert_mode_files, types::GeneralRBMHiddenLayerTerm, ExpertModeData,
        };
        use num_complex::Complex64;
        let rows: Vec<_> = include_str!("../../../tests/fixtures/rbm/c_reader_contracts.txt")
            .lines()
            .filter(|line| !line.starts_with('#'))
            .collect();
        let directory =
            std::env::temp_dir().join(format!("mvmc-c-rbm-validation-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(
            directory.join("modpara.def"),
            "Nsite 3\nNElec 1\nNMPTrans 1\nNneuronCharge 2\nNneuronSpin 2\nNneuronGeneral 2\n",
        )
        .unwrap();
        for (section, name) in mvmc_expert_parsers::parsers::rbm::SECTION_NAMES
            .iter()
            .enumerate()
        {
            let record = rows
                .as_chunks::<4>()
                .0
                .iter()
                .find(|record| {
                    let h: Vec<_> = record[0].split_whitespace().collect();
                    h[1] == section.to_string()
                        && h[2] == "3"
                        && h[3] == "2"
                        && h[6] == "0"
                        && h[0].ends_with("_complete")
                })
                .unwrap();
            std::fs::write(directory.join("rbm.def"), record[1].replace('|', "\n")).unwrap();
            std::fs::write(
                directory.join("namelist.def"),
                format!("{name} rbm.def\nModPara modpara.def\n"),
            )
            .unwrap();
            let mut data = parse_expert_mode_files(directory.join("namelist.def")).unwrap();
            assert!(
                data.input_errors.is_empty(),
                "{name}: {:?}",
                data.input_errors
            );
            super::validate_para_opt(&data).unwrap();
            data.namelist.clear();
            super::validate_para_opt(&data).unwrap();
        }
        std::fs::remove_dir_all(directory).unwrap();
        // A programmatic mapping still signals term presence when no declared
        // array is reserved; this check does not establish C index safety.
        let mut data = ExpertModeData::new();
        data.modpara.nmp_trans = 1;
        data.general_rbm_hidden_layer_terms
            .push(GeneralRBMHiddenLayerTerm {
                site: 0,
                idx: i64::MAX,
                value: Complex64::new(0.0, 0.0),
                is_complex: false,
            });
        assert_eq!(data.count_rbm_parameters(), 0);
        super::validate_para_opt(&data).unwrap();
    }

    #[test]
    fn mpi_detection_matches_julia_launch_policy() {
        for (env, expected) in [
            (vec![], false),
            (vec![("PMI_RANK", "0"), ("PMI_SIZE", "1")], false),
            (vec![("SLURM_NTASKS", "4")], false),
            (vec![("PMI_SIZE", "1"), ("SLURM_NTASKS", "4")], true),
            (vec![("PMI_SIZE", "2")], true),
            (vec![("OMPI_COMM_WORLD_SIZE", "1")], true),
            (vec![("PMIX_RANK", "0")], true),
            (vec![("JULIA_MVMC_MPI", "1")], true),
        ] {
            assert_eq!(
                mpi_requested(|key| env
                    .iter()
                    .find(|(k, _)| *k == key)
                    .map(|(_, v)| (*v).into())),
                expected,
                "{env:?}"
            );
        }
    }
}

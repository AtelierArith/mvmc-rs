//! Runtime compatibility checks, separate from Expert-mode format parsing.
//!
//! Permanent restrictions follow Julia's `unsupported_inputs.jl`. Temporary
//! restrictions name the porting issue and must be removed when its full
//! production path passes deterministic Julia parity checks.

use mvmc_expert_parsers::utils::parameter_init::all_complex_flag;
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
    if mpi_requested(|key| std::env::var(key).ok()) {
        return Err("MPI execution is not implemented yet (issue #35)".into());
    }
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
    if p.nsplit_size > 1 {
        if data.n_qp_opt_trans.max(1) > 1 || data.opt_trans.len() > 1 || data.qp_opt_trans.len() > 1
        {
            return Err(format!(
                "NSplitSize > 1 with NQPOptTrans > 1 / OptTrans is not supported: grouped QP-split sampling currently supports standard-projection NQPFull only (NQPOptTrans = 1), got NSplitSize = {}, NQPOptTrans = {}. Use NSplitSize = 1 for OptTrans-derived QP sectors.",
                p.nsplit_size, data.n_qp_opt_trans
            ));
        }
        return Err("NSplitSize > 1 is not implemented yet (issue #36)".into());
    }
    let has_interall = !data.inter_all_terms.is_empty()
        || data.namelist.iter().any(|(kind, _)| kind == "InterAll");
    if has_interall {
        if data.i_flg_orbital_general == 0 {
            return Err("InterAll in fixed-Sz mode is not implemented yet (issue #23): the Julia reference accumulator accesses a nonexistent term.sites field".into());
        }
        if !crate::run::get_all_complex_flag(data) {
            return Err("real FSZ InterAll is not implemented yet (issue #43)".into());
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
    if data.i_flg_orbital_general != 0
        && !crate::run::get_all_complex_flag(data)
        && !all_complex_flag(data)
    {
        return Err("real FSZ is not implemented yet (issue #43)".into());
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

fn mpi_requested(get: impl Fn(&str) -> Option<String>) -> bool {
    if get("JULIA_MVMC_MPI").is_some_and(|v| v == "1")
        || get("OMPI_COMM_WORLD_SIZE").is_some()
        || get("PMIX_RANK").is_some()
    {
        return true;
    }
    let pmi_size = get("PMI_SIZE").and_then(|v| v.parse::<i64>().ok());
    // Match Julia's first parseable SLURM task-count key, not PMI_RANK alone.
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
        let root =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/rbm");
        for name in [
            "ChargeRBM_PhysLayer",
            "SpinRBM_PhysLayer",
            "GeneralRBM_PhysLayer",
            "ChargeRBM_HiddenLayer",
            "SpinRBM_HiddenLayer",
            "GeneralRBM_HiddenLayer",
            "ChargeRBM_PhysHidden",
            "SpinRBM_PhysHidden",
            "GeneralRBM_PhysHidden",
        ] {
            let mut data =
                parse_expert_mode_files(root.join(format!("namelist_{name}.def"))).unwrap();
            // These parser-only namelists omit ModPara; supply a valid C sector count.
            data.modpara.nmp_trans = 1;
            super::validate_para_opt(&data).unwrap();
            data.namelist.clear();
            super::validate_para_opt(&data).unwrap();
        }
        // A nonempty mapping whose maximum index wraps to a zero width is still RBM.
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

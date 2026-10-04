//! `modpara.def` parser.
//!
//! Port of `MVMCExpertModeParsers.jl/src/parsers/modpara_parser.jl`.

use std::io;
use std::path::Path;

use crate::types::ModParaParameters;
use crate::utils::file::{
    clean_line, read_def_file, safe_parse_float, safe_parse_int, split_def_line,
};

/// Parse a `modpara.def` file from disk.
pub fn parse_modpara_def<P: AsRef<Path>>(path: P) -> io::Result<ModParaParameters> {
    let content = read_def_file(path)?;
    Ok(parse_modpara_content(&content))
}

/// Parse a `modpara.def` payload from memory.
pub fn parse_modpara_content(content: &str) -> ModParaParameters {
    let mut params = ModParaParameters::default();

    for line in content.lines() {
        let tokens = split_def_line(line);
        if tokens.len() < 2 {
            continue;
        }
        // Support "param = value" too (upstream accepts both).
        let (name, value) = if tokens.len() >= 3 && tokens[1] == "=" {
            (tokens[0], tokens[2])
        } else {
            (tokens[0], tokens[1])
        };
        // Skip lines that look like section banners ("VMC_Cal_Parameters"
        // has no value, but our 2-token guard already filtered those).
        // Lines like `Nsite          6   ` end up with 2 tokens, which is
        // what we want.
        let _ = clean_line; // keep import in scope; clean_line is invoked via split_def_line.
        apply_param(&mut params, name, value);
    }

    params
}

fn apply_param(p: &mut ModParaParameters, name: &str, value: &str) {
    match name {
        // Basic system parameters
        "NSite" | "Nsite" => p.nsite = safe_parse_int(value, 0),
        "NElec" | "Nelec" => p.nelec = safe_parse_int(value, 0),
        "NLocSpin" | "NlocalSpin" => p.nlocspin = safe_parse_int(value, 0),
        "NCond" | "Ncond" => p.ncond = safe_parse_int(value, -1),
        // Calculation modes
        "VMCCalMode" | "NVMCCalMode" => p.vmc_calc_mode = safe_parse_int(value, 0),
        "LanczosMode" | "NLanczosMode" => p.lanczos_mode = safe_parse_int(value, 0),
        // VMC parameters
        "NSROptItrStep" => p.nsr_opt_itr_step = safe_parse_int(value, 1000),
        "NSROptItrSmp" => p.nsr_opt_itr_smp = safe_parse_int(value, 1000),
        "NSROptFixSmp" => p.nsr_opt_fix_smp = safe_parse_int(value, 0),
        "NVMCWarmUp" => p.nvmc_warmup = safe_parse_int(value, 1000),
        "NVMCInterval" => p.nvmc_interval = safe_parse_int(value, 1),
        "NVMCSample" => p.nvmc_sample = safe_parse_int(value, 10000),
        // SR parameters
        "DSROptRedCut" => p.dsr_opt_red_cut = safe_parse_float(value, 1e-6),
        "DSROptStaDel" => p.dsr_opt_sta_del = safe_parse_float(value, 0.0),
        "DSROptStepDt" => p.dsr_opt_step_dt = safe_parse_float(value, 0.01),
        "DSROptCGTol" => p.dsr_opt_cg_tol = safe_parse_float(value, 1e-10),
        "NSROptCGMaxIter" => p.nsr_opt_cg_max_iter = safe_parse_int(value, 0),
        // SR solver selection
        "NSRCG" => p.nsrcg = safe_parse_int(value, 0),
        "useDiagScale" => p.use_diag_scale = safe_parse_int(value, 0),
        "RescaleSmat" => p.rescale_smat = safe_parse_int(value, 0),
        "NStore" => p.nstore_o = safe_parse_int(value, 1),
        // RNG
        "RndSeed" => p.rnd_seed = safe_parse_int(value, 11272),
        "NSplitSize" => p.nsplit_size = safe_parse_int(value, 1),
        // Quantum projection
        "NSPGaussLeg" => p.nsp_gauss_leg = safe_parse_int(value, 1),
        "NSPStot" => p.nsp_stot = safe_parse_int(value, 0),
        "NMPTrans" => p.nmp_trans = safe_parse_int(value, 0),
        "2Sz" => p.two_sz = safe_parse_int(value, -1),
        // Data output
        "NDataIdxStart" => p.n_data_idx_start = safe_parse_int(value, 0),
        "NDataQtySmp" => p.n_data_qty_smp = safe_parse_int(value, 1),
        "CDataFileHead" => p.c_data_file_head = value.to_string(),
        "CParaFileHead" => p.c_para_file_head = value.to_string(),
        // File control / flags
        "NFileFlushInterval" => p.n_file_flush_interval = safe_parse_int(value, 1),
        "ComplexType" => p.complex_flag = safe_parse_int(value, 0),
        // RBM
        "Nneuron" => p.nneuron = safe_parse_int(value, 0),
        "NneuronGeneral" => p.nneuron_general = safe_parse_int(value, 0),
        "NneuronCharge" => p.nneuron_charge = safe_parse_int(value, 0),
        "NneuronSpin" => p.nneuron_spin = safe_parse_int(value, 0),
        "NBlockSize_RBMRatio" => p.nblock_size_rbm_ratio = safe_parse_int(value, 200),
        // Lanczos
        "NOneBodyG" => p.n_one_body_g = safe_parse_int(value, 0),
        "NTwoBodyG" => p.n_two_body_g = safe_parse_int(value, 0),
        "NTwoBodyGEx" => p.n_two_body_g_ex = safe_parse_int(value, 0),
        // Exchange update
        "NExUpdatePath" => p.nex_update_path = safe_parse_int(value, 1),
        // Section / version banners we silently ignore. The Julia parser
        // emits a warning for unknown keys; we drop it to keep tests
        // chatter-free.
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_heisenberg_chain_real_modpara() {
        let content = "\
--------------------\n\
Model_Parameters   0\n\
--------------------\n\
VMC_Cal_Parameters\n\
--------------------\n\
CDataFileHead  zvo\n\
CParaFileHead  zqp\n\
--------------------\n\
NVMCCalMode    0\n\
NLanczosMode   0\n\
--------------------\n\
NDataIdxStart  1\n\
NDataQtySmp    1\n\
--------------------\n\
Nsite          6\n\
Ncond          0\n\
2Sz            0\n\
NSPGaussLeg    8\n\
NSPStot        0\n\
NMPTrans       -1\n\
NSROptItrStep  1000\n\
NSROptItrSmp   100\n\
DSROptRedCut   0.0000000001\n\
DSROptStaDel   0.0000100000\n\
DSROptStepDt   0.0100000000\n";
        let p = parse_modpara_content(content);
        assert_eq!(p.nsite, 6);
        assert_eq!(p.ncond, 0);
        assert_eq!(p.two_sz, 0);
        assert_eq!(p.nsp_gauss_leg, 8);
        assert_eq!(p.nmp_trans, -1);
        assert_eq!(p.nsr_opt_itr_step, 1000);
        assert_eq!(p.nsr_opt_itr_smp, 100);
        assert!((p.dsr_opt_red_cut - 1e-10).abs() < 1e-20);
        assert!((p.dsr_opt_sta_del - 1e-5).abs() < 1e-15);
        assert!((p.dsr_opt_step_dt - 0.01).abs() < 1e-15);
        assert_eq!(p.c_data_file_head, "zvo");
        assert_eq!(p.c_para_file_head, "zqp");
        assert_eq!(p.vmc_calc_mode, 0);
    }
}

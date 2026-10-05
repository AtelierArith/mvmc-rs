//! M0209–224: shared C solver-control defaults. Since #344 the parser follows
//! the C reader exactly: no Julia-only `useDiagScale`/`RescaleSmat`, no
//! `name = value` syntax and no aliases (see `issue344_c_modpara_reader.rs`).
use mvmc_expert_parsers::parsers::modpara::parse_modpara_content;

const HEADER: &str =
    "-\nModel_Parameters 0\n-\nVMC_Cal_Parameters\n-\nCDataFileHead zvo\nCParaFileHead zqp\n-\n";

#[test]
fn empty_public_modpara_parser_retains_c_solver_control_defaults() {
    // C readdef.c defaults: IdxSROptCGMaxIter=0 (1793),
    // IdxSROptCGTol=1e-10, NStoreO=1, NSRCG=0 (1820–1822).
    let defaults = parse_modpara_content(HEADER).unwrap();
    assert_eq!(defaults.dsr_opt_cg_tol, 1e-10);
    assert_eq!(defaults.nsr_opt_cg_max_iter, 0);
    assert_eq!(defaults.nsrcg, 0);
    assert_eq!(defaults.nstore_o, 1);
}

#[test]
fn julia_equals_syntax_and_extension_keys_are_not_c_input() {
    // Exact original test_parsers.jl lines 28–38 used '= ' syntax and the
    // Julia extension keys; C rejects the first unknown keyword (NSite is the
    // C key `Nsite`, but `=` leaves a stale %lf value, so NElec fails first).
    let content = format!("{HEADER}NSite = 4\nNElec = 2\n");
    let error = parse_modpara_content(&content).unwrap_err().to_string();
    assert!(error.contains("NElec"), "{error}");
}

#[test]
fn c_defined_solver_control_tokens_are_read() {
    // Valid scalar tokens from C readdef.c 1916–1919/1932–1935.
    let parsed = parse_modpara_content(&format!(
        "{HEADER}DSROptCGTol 2.5e-9\nNSROptCGMaxIter 41\nNStore 0\nNSRCG 1\n"
    ))
    .unwrap();
    assert_eq!(parsed.dsr_opt_cg_tol, 2.5e-9);
    assert_eq!(parsed.nsr_opt_cg_max_iter, 41);
    assert_eq!(parsed.nstore_o, 0);
    assert_eq!(parsed.nsrcg, 1);
}

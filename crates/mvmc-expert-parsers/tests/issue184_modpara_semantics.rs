//! M0209–224: shared C solver-control defaults and original Julia parser input.
//! Parsing Julia-only controls is not runtime support or native-C file parity.
use mvmc_expert_parsers::parsers::modpara::parse_modpara_content;

#[test]
fn empty_public_modpara_parser_retains_c_solver_control_defaults() {
    // C readdef.c defaults: IdxSROptCGMaxIter=0 (1793),
    // IdxSROptCGTol=1e-10, NStoreO=1, NSRCG=0 (1820–1822).
    let defaults = parse_modpara_content("");
    assert_eq!(defaults.dsr_opt_cg_tol, 1e-10);
    assert_eq!(defaults.nsr_opt_cg_max_iter, 0);
    assert_eq!(defaults.nsrcg, 0);
    assert_eq!(defaults.nstore_o, 1);
    // M0213–214: Julia architecture fields, not native C keys.
    assert_eq!(defaults.use_diag_scale, 0);
    assert_eq!(defaults.rescale_smat, 0);
}

#[test]
fn original_modpara_equals_payload_preserves_counts_and_declared_controls() {
    // Exact original test_parsers.jl lines 28–38 (leading whitespace retained).
    // '= ' syntax and useDiagScale/RescaleSmat are Julia parser extensions.
    // Native C uses sscanf("%s %lf") and has neither extension key; this
    // assertion must not be presented as supported native-C execution.
    let content = "
    NSite = 4
    NElec = 2
    NLocSpin = 0
    VMCCalMode = 0
    LanczosMode = 0
    NStore = 0
    NSRCG = 1
    useDiagScale = 1
    RescaleSmat = 1
    ";
    let parsed = parse_modpara_content(content);
    // Rust returns the owned value directly instead of Julia ParseResult.
    assert_eq!(parsed.nsite, 4);
    assert_eq!(parsed.nelec, 2);
    assert_eq!(parsed.nlocspin, 0);
    assert_eq!(parsed.dsr_opt_cg_tol, 1e-10);
    assert_eq!(parsed.nsr_opt_cg_max_iter, 0);
    assert_eq!(parsed.nstore_o, 0);
    assert_eq!(parsed.nsrcg, 1);
    assert_eq!(parsed.use_diag_scale, 1);
    assert_eq!(parsed.rescale_smat, 1);
    assert_eq!(parsed.vmc_calc_mode, 0);
    assert_eq!(parsed.lanczos_mode, 0);
}

#[test]
fn c_defined_solver_control_tokens_are_read_independently_of_julia_extensions() {
    // Valid scalar tokens from C readdef.c 1916–1919/1932–1935.
    // This exercises scalar parsing, not C's fixed eight-line file preamble.
    let parsed =
        parse_modpara_content("DSROptCGTol 2.5e-9\nNSROptCGMaxIter 41\nNStore 0\nNSRCG 1\n");
    assert_eq!(parsed.dsr_opt_cg_tol, 2.5e-9);
    assert_eq!(parsed.nsr_opt_cg_max_iter, 41);
    assert_eq!(parsed.nstore_o, 0);
    assert_eq!(parsed.nsrcg, 1);
    assert_eq!(parsed.use_diag_scale, 0);
    assert_eq!(parsed.rescale_smat, 0);
}

//! Original Julia ModPara constructor assertions M0206–0208 only.
//! `test_parsers.jl:14–17`, SHA256
//! 8336b21bca7627749e27c588e537e30c20ea4a6383733bae3e93a44461980185.
//! Rust public struct/default construction replaces Julia keyword construction.
//! This is not file parsing, supported-model validation, or C runtime evidence.
use mvmc_expert_parsers::ModParaParameters;

#[test]
fn original_modpara_constructor_preserves_three_explicit_fields() {
    let params = ModParaParameters {
        nsite: 4,
        nelec: 2,
        nlocspin: 0,
        ..ModParaParameters::default()
    };
    assert_eq!(params.nsite, 4);
    assert_eq!(params.nelec, 2);
    assert_eq!(params.nlocspin, 0);
}

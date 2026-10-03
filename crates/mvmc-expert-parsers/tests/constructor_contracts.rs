//! Original Julia constructor field assertions M0206–0208, M0225–0227,
//! M0236–0237 and M0242–0244, with explicit Transfer spin API adaptation.
//! `test_parsers.jl:14–17,56–59,83–84,101–103`, SHA256
//! 8336b21bca7627749e27c588e537e30c20ea4a6383733bae3e93a44461980185.
//! Rust public struct/default construction replaces Julia keyword construction.
//! This is not file parsing, supported-model validation, or C runtime evidence.
use mvmc_expert_parsers::{CoulombIntraTerm, ModParaParameters, PairHopTerm, Spin, TransferTerm};
use num_complex::Complex64;

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

#[test]
fn original_transfer_constructor_fields_use_explicit_up_spin_coordinates() {
    // Julia TransferTerm(0, 1, 1+0im, :up) sets both operator spin codes to 0.
    // M0228's derived `spin::Symbol` property is intentionally absent in Rust.
    let term = TransferTerm {
        site1: 0,
        spin1: Spin::Up,
        site2: 1,
        spin2: Spin::Up,
        value: Complex64::new(1.0, 0.0),
    };
    assert_eq!(term.site1, 0);
    assert_eq!(term.site2, 1);
    assert_eq!(term.value, Complex64::new(1.0, 0.0));
    assert_eq!(term.spin1.as_code(), 0);
    assert_eq!(term.spin2.as_code(), 0);
}

#[test]
fn original_coulomb_constructor_preserves_site_and_coupling() {
    let term = CoulombIntraTerm {
        site: 0,
        value: 4.0,
    };
    assert_eq!(term.site, 0);
    assert_eq!(term.value, 4.0);
}

#[test]
fn original_pairhop_constructor_preserves_one_directed_term() {
    // Constructing one term does not invoke the C reader's two-direction expansion.
    let term = PairHopTerm {
        site1: 0,
        site2: 1,
        value: 0.25,
    };
    assert_eq!(term.site1, 0);
    assert_eq!(term.site2, 1);
    assert_eq!(term.value, 0.25);
}

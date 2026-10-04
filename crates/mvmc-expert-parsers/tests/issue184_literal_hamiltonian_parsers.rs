//! Original test_parsers.jl M0229–235/M0238–241/M0245–253 payloads.
//! C numerical/layout authority: readdef.c GetTransferInfo (1977–2004),
//! GetInfoCoulombIntra (2016–2035), ReadPairHopValue (2038–2060).
//! Bare content is a Julia-style parser API; native C consumes declared-count
//! payloads after its header. Neither is a native-C executable validation here.
use mvmc_expert_parsers::parsers::coulomb::parse_coulomb_intra_content;
use mvmc_expert_parsers::parsers::pairhop::parse_pairhop_content;
use mvmc_expert_parsers::parsers::trans::parse_trans_content;
use mvmc_expert_parsers::Spin;
use num_complex::Complex64;

#[test]
fn original_transfer_payload_preserves_all_sites_spin_codes_and_values() {
    let terms = parse_trans_content(
        "    0 0 1 0  1.0  0.0\n    1 1 2 1  2.0  0.0\n    2 0 3 1  3.0  0.0\n",
    );
    assert_eq!(terms.len(), 3);
    let actual: Vec<_> = terms
        .iter()
        .map(|term| {
            (
                term.site1,
                term.spin1.as_code(),
                term.site2,
                term.spin2.as_code(),
                term.value,
            )
        })
        .collect();
    assert_eq!(
        actual,
        [
            (0, 0, 1, 0, Complex64::new(1.0, 0.0)),
            (1, 1, 2, 1, Complex64::new(2.0, 0.0)),
            (2, 0, 3, 1, Complex64::new(3.0, 0.0)),
        ]
    );
    assert_eq!(terms[0].spin1, Spin::Up);
    // M0235's Julia derived `spin` property has no duplicate Rust field.
    // Both explicit operator spin codes are checked, not a fabricated property.
}

#[test]
fn original_coulomb_payload_preserves_both_sites_and_couplings() {
    let mut context = mvmc_expert_parsers::ParsingContext::new("original-payload");
    let terms = parse_coulomb_intra_content("    0 4.0\n    1 4.0\n", &mut context);
    assert_eq!(terms.len(), 2);
    assert_eq!(
        terms
            .iter()
            .map(|term| (term.site, term.value))
            .collect::<Vec<_>>(),
        [(0, 4.0), (1, 4.0)]
    );
}

#[test]
fn original_pairhop_payloads_expand_every_directed_row_in_c_order() {
    let cases = [
        ("0 1 0.25\n2 3 -0.5\n".to_string(), vec![
            (0, 1, 0.25), (1, 0, 0.25), (2, 3, -0.5), (3, 2, -0.5),
        ]),
        ((0..6).map(|i| format!("{i} {} 0.1", i + 1)).collect::<Vec<_>>().join("\n"),
            (0..6).flat_map(|i| [(i, i + 1, 0.1), (i + 1, i, 0.1)]).collect()),
        ("=============================================\nNPairHopp          1\n=============================================\n====== Pair-Hopping term ============\n=============================================\n    4     5         1.250000000000000\n".to_string(),
            vec![(4, 5, 1.25), (5, 4, 1.25)]),
        ("2 2 0.75".to_string(), vec![(2, 2, 0.75), (2, 2, 0.75)]),
    ];
    for (payload, expected) in cases {
        let parsed = parse_pairhop_content(&payload);
        assert!(parsed.is_success(), "{payload}: {:?}", parsed.errors);
        assert_eq!(parsed.terms.len(), expected.len());
        let actual: Vec<_> = parsed
            .terms
            .iter()
            .map(|term| (term.site1, term.site2, term.value))
            .collect();
        assert_eq!(actual, expected);
    }
    // The same-site row is C-reader-defined (CheckPairSite checks bounds,
    // not inequality), not a claim about a nonzero physical operator value.
}

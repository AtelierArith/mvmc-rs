//! Original inverse relations M0121–23/M0136–37/M0141, through an actual query.
//! C GetInfoTransSym readdef.c2232–2265 stores inverse[forward[origin]]=origin.
//! No C/Julia runtime or BackFlow acceptance is exercised by these tests.
use mvmc_expert_parsers::parsers::qptrans::parse_qptrans_content;
use mvmc_expert_parsers::{QPTransEntry, QPTransInverseError};
use num_complex::Complex64;

#[test]
fn original_four_site_two_rows_have_explicit_identity_and_cyclic_inverses() {
    // Exact original Julia test weight/mapping payload, complete C row counts.
    let definition = "===\nNQPTrans 2\n===\n===\n===\n0 1.00000\n1 1.00000\n0 0 0 1\n0 1 1 1\n0 2 2 1\n0 3 3 1\n1 0 1 1\n1 1 2 1\n1 2 3 1\n1 3 0 1\n";
    let section = parse_qptrans_content(definition, 4);
    assert_eq!(section.n_qp_trans, 2);
    let inverses: Vec<_> = section
        .entries
        .iter()
        .map(|e| e.inverse_site_map().unwrap())
        .collect();
    assert_eq!(inverses.len(), 2);
    assert_eq!(inverses[0], [0, 1, 2, 3]);
    assert_eq!(inverses[1], [3, 0, 1, 2]);
    assert_eq!(inverses[1][1], 0);
}

#[test]
fn original_sixteen_site_four_rows_satisfy_all_sixty_four_inverse_identities() {
    // Byte-identical original sample, provenance in this fixture's README.
    let definition =
        include_str!("../../../tests/fixtures/issue184_heisenberg_parser/qptransidx.def");
    let section = parse_qptrans_content(definition, 16);
    assert_eq!(section.n_qp_trans, 4);
    assert_eq!(section.entries.len(), 4);
    let mut identities = 0;
    for (mpidx, entry) in section.entries.iter().enumerate() {
        assert_eq!(entry.site_map.len(), 16);
        let inverse = entry.inverse_site_map().unwrap();
        assert_eq!(inverse.len(), 16);
        for (origin, &translated) in entry.site_map.iter().enumerate() {
            assert_eq!(
                inverse[usize::try_from(translated).unwrap()],
                origin,
                "translation {mpidx}, original site {origin}"
            );
            identities += 1;
        }
    }
    assert_eq!(identities, 64);
}

fn entry(map: Vec<i64>) -> QPTransEntry {
    QPTransEntry {
        weight: Complex64::new(0.5, -0.25),
        site_sign: vec![-1; map.len()],
        site_map: map,
    }
}

#[test]
fn inverse_query_is_fresh_after_public_forward_mutation_and_preserves_other_fields() {
    let mut translation = entry(vec![1, 2, 3, 0]);
    let weight = translation.weight;
    let signs = translation.site_sign.clone();
    assert_eq!(translation.inverse_site_map().unwrap(), [3, 0, 1, 2]);
    translation.site_map = vec![2, 0, 3, 1];
    assert_eq!(translation.inverse_site_map().unwrap(), [1, 3, 0, 2]);
    assert_eq!(translation.site_map, [2, 0, 3, 1]);
    assert_eq!(translation.weight, weight);
    assert_eq!(translation.site_sign, signs);
    translation.site_map[0] = 0;
    assert_eq!(
        translation.inverse_site_map(),
        Err(QPTransInverseError::DuplicateTarget {
            target: 0,
            first_origin: 0,
            origin: 1,
        })
    );
}

#[test]
fn inverse_query_reports_empty_negative_out_of_range_and_duplicate_maps() {
    assert_eq!(
        entry(vec![]).inverse_site_map(),
        Err(QPTransInverseError::EmptyMapping)
    );
    assert_eq!(entry(vec![0]).inverse_site_map().unwrap(), [0]);
    assert_eq!(
        entry(vec![0, -1]).inverse_site_map(),
        Err(QPTransInverseError::NegativeTarget {
            origin: 1,
            target: -1
        })
    );
    for target in [2, i64::MAX] {
        assert_eq!(
            entry(vec![0, target]).inverse_site_map(),
            Err(QPTransInverseError::OutOfRangeTarget {
                origin: 1,
                target,
                nsite: 2
            })
        );
    }
    assert_eq!(
        entry(vec![1, 1]).inverse_site_map(),
        Err(QPTransInverseError::DuplicateTarget {
            target: 1,
            first_origin: 0,
            origin: 1
        })
    );
}

//! Bounded A007 public update architecture; C reader, not Julia payload admission.
use mvmc_expert_parsers::{
    load_hamiltonian_definition, CoulombInterTerm, CoulombIntraTerm, ExchangeTerm, ExpertModeData,
    GutzwillerTerm, HamiltonianDefinitionKind as Kind, HundTerm, ModParaParameters, PairHopTerm,
    Spin, TransferTerm,
};
use num_complex::Complex64;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Input(std::path::PathBuf);
impl Input {
    fn new(count: usize, body: &str) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "mvmc-issue332-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        let input = Self(path);
        write!(file, "===\nNDefinition {count}\n===\n===\n===\n{body}").unwrap();
        input
    }
}
impl Drop for Input {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn data() -> ExpertModeData {
    ExpertModeData {
        modpara: ModParaParameters {
            nsite: 4,
            ..ModParaParameters::default()
        },
        namelist: vec![("caller-owned".into(), "keep.def".into())],
        input_errors: vec!["existing error".into()],
        transfer_terms: vec![TransferTerm {
            site1: 0,
            spin1: Spin::Up,
            site2: 0,
            spin2: Spin::Down,
            value: Complex64::new(11.0, 22.0),
        }],
        coulomb_intra_terms: vec![CoulombIntraTerm {
            site: 0,
            value: 11.0,
        }],
        coulomb_inter_terms: vec![CoulombInterTerm {
            site1: 0,
            site2: 0,
            value: 11.0,
        }],
        hund_terms: vec![HundTerm {
            site1: 0,
            site2: 0,
            value: 11.0,
        }],
        exchange_terms: vec![ExchangeTerm {
            site1: 0,
            site2: 0,
            value: 11.0,
        }],
        pair_hop_terms: vec![PairHopTerm {
            site1: 0,
            site2: 0,
            value: 11.0,
        }],
        optimization_flags: vec![1, -1, 2],
        gutzwiller_terms: vec![GutzwillerTerm {
            site: 2,
            value: Complex64::new(0.25, -0.5),
            is_complex: true,
        }],
        rbm_params: vec![Complex64::new(0.25, -0.5)],
        slater_params: vec![Complex64::new(-2.0, 0.125)],
        para_qp_trans: vec![Complex64::new(0.5, -0.25)],
        ..ExpertModeData::new()
    }
}

fn kinds() -> [Kind; 6] {
    [
        Kind::Transfer,
        Kind::CoulombIntra,
        Kind::CoulombInter,
        Kind::Hund,
        Kind::Exchange,
        Kind::PairHop,
    ]
}

// Clear only the selected payload in both snapshots. The rest must match,
// including flags, buffers, declarations, prior diagnostics and namelist.
fn unrelated(data: &ExpertModeData, kind: Kind) -> String {
    let mut copy = data.clone();
    match kind {
        Kind::Transfer => copy.transfer_terms.clear(),
        Kind::CoulombIntra => copy.coulomb_intra_terms.clear(),
        Kind::CoulombInter => copy.coulomb_inter_terms.clear(),
        Kind::Hund => copy.hund_terms.clear(),
        Kind::Exchange => copy.exchange_terms.clear(),
        Kind::PairHop => copy.pair_hop_terms.clear(),
    }
    format!("{copy:?}")
}

#[test]
fn typed_transfer_preserves_literal_spins_complex_values_and_only_its_family() {
    let input = Input::new(2, "2 1 0 0 0.5 -1\n0 0 3 1 -2 0.25\n");
    let mut data = data();
    let before = unrelated(&data, Kind::Transfer);
    load_hamiltonian_definition(&mut data, Kind::Transfer, &input.0).unwrap();
    assert_eq!(
        data.transfer_terms,
        [
            TransferTerm {
                site1: 2,
                spin1: Spin::Down,
                site2: 0,
                spin2: Spin::Up,
                value: Complex64::new(0.5, -1.0)
            },
            TransferTerm {
                site1: 0,
                spin1: Spin::Up,
                site2: 3,
                spin2: Spin::Down,
                value: Complex64::new(-2.0, 0.25)
            },
        ]
    );
    assert_eq!(unrelated(&data, Kind::Transfer), before);
}

#[test]
fn typed_coulomb_intra_preserves_order_and_replaces_old_payload_only() {
    let input = Input::new(2, "3 4\n0 -0.5\n");
    let mut data = data();
    let before = unrelated(&data, Kind::CoulombIntra);
    load_hamiltonian_definition(&mut data, Kind::CoulombIntra, &input.0).unwrap();
    assert_eq!(
        data.coulomb_intra_terms,
        [
            CoulombIntraTerm {
                site: 3,
                value: 4.0
            },
            CoulombIntraTerm {
                site: 0,
                value: -0.5
            }
        ]
    );
    assert_eq!(unrelated(&data, Kind::CoulombIntra), before);
}

#[test]
fn typed_pair_families_preserve_signed_diagonal_and_reordered_literal_pairs() {
    let input = Input::new(2, "3 3 -2\n1 0 0.5\n");
    for kind in [Kind::CoulombInter, Kind::Hund, Kind::Exchange] {
        let mut data = data();
        let before = unrelated(&data, kind);
        load_hamiltonian_definition(&mut data, kind, &input.0).unwrap();
        let actual: Vec<_> = match kind {
            Kind::CoulombInter => data
                .coulomb_inter_terms
                .iter()
                .map(|t| (t.site1, t.site2, t.value))
                .collect(),
            Kind::Hund => data
                .hund_terms
                .iter()
                .map(|t| (t.site1, t.site2, t.value))
                .collect(),
            Kind::Exchange => data
                .exchange_terms
                .iter()
                .map(|t| (t.site1, t.site2, t.value))
                .collect(),
            _ => unreachable!(),
        };
        assert_eq!(actual, [(3, 3, -2.0), (1, 0, 0.5)]);
        assert_eq!(unrelated(&data, kind), before);
    }
}

#[test]
fn typed_pairhop_retains_forward_reverse_duplicate_rows_without_sign_change() {
    let input = Input::new(2, "2 0 -0.5\n2 0 -0.5\n");
    let mut data = data();
    let before = unrelated(&data, Kind::PairHop);
    load_hamiltonian_definition(&mut data, Kind::PairHop, &input.0).unwrap();
    assert_eq!(
        data.pair_hop_terms
            .iter()
            .map(|t| (t.site1, t.site2, t.value))
            .collect::<Vec<_>>(),
        [(2, 0, -0.5), (0, 2, -0.5), (2, 0, -0.5), (0, 2, -0.5)]
    );
    assert_eq!(unrelated(&data, Kind::PairHop), before);
}

#[test]
fn every_family_failed_io_and_consumed_record_preserves_all_caller_data() {
    for kind in kinds() {
        let missing = Input::new(0, "");
        std::fs::remove_file(&missing.0).unwrap();
        let invalid = Input::new(
            2,
            match kind {
                Kind::Transfer => "0 0 1 0 4 0\n4 0 1 0 2 0\n",
                Kind::CoulombIntra => "0 4\n4 2\n",
                _ => "0 1 4\n4 1 2\n",
            },
        );
        let mut data = data();
        let before = format!("{data:?}");
        assert_eq!(
            load_hamiltonian_definition(&mut data, kind, &missing.0)
                .unwrap_err()
                .kind(),
            std::io::ErrorKind::NotFound
        );
        assert_eq!(format!("{data:?}"), before);
        assert_eq!(
            load_hamiltonian_definition(&mut data, kind, &invalid.0)
                .unwrap_err()
                .kind(),
            std::io::ErrorKind::InvalidData
        );
        assert_eq!(format!("{data:?}"), before);
    }
}

#[test]
fn zero_count_clears_only_selected_family_and_ignores_original_body() {
    let input = Input::new(0, "ignored malformed body\n");
    for kind in kinds() {
        let mut data = data();
        let before = unrelated(&data, kind);
        load_hamiltonian_definition(&mut data, kind, &input.0).unwrap();
        let empty = match kind {
            Kind::Transfer => data.transfer_terms.is_empty(),
            Kind::CoulombIntra => data.coulomb_intra_terms.is_empty(),
            Kind::CoulombInter => data.coulomb_inter_terms.is_empty(),
            Kind::Hund => data.hund_terms.is_empty(),
            Kind::Exchange => data.exchange_terms.is_empty(),
            Kind::PairHop => data.pair_hop_terms.is_empty(),
        };
        assert!(empty);
        assert_eq!(unrelated(&data, kind), before);
    }
}

#[test]
fn typed_transfer_omitted_imaginary_field_carries_initialized_then_previous_value() {
    let input = Input::new(3, "0 0 1 1 0.5\n2 1 0 0 -2 0.25\n1 0 3 1 4\n");
    let mut data = data();
    load_hamiltonian_definition(&mut data, Kind::Transfer, &input.0).unwrap();
    assert_eq!(
        data.transfer_terms
            .iter()
            .map(|term| term.value)
            .collect::<Vec<_>>(),
        [
            Complex64::new(0.5, 0.0),
            Complex64::new(-2.0, 0.25),
            Complex64::new(4.0, 0.25)
        ]
    );
}

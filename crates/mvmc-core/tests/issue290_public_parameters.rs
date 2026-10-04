use mvmc_core::{
    get_parameter_value, pack_parameters, set_parameter_value, unpack_parameters, ExpertModeData,
    ParameterAccessError,
};
use mvmc_expert_parsers::QuantumProjectionWeights;
use num_complex::Complex64 as C;
use std::path::Path;

#[test]
fn complete_declared_block_order_matches_independent_literal_positions() {
    use mvmc_expert_parsers::{
        DoublonHolon2SiteIndex, DoublonHolon4SiteIndex, GutzwillerTerm, JastrowTerm,
    };
    let mut data = ExpertModeData {
        n_gutzwiller_idx: 1,
        gutzwiller_terms: vec![GutzwillerTerm {
            site: 0,
            value: C::new(11.0, -11.0),
            is_complex: false,
        }],
        n_jastrow_idx: 1,
        jastrow_terms: vec![JastrowTerm {
            site1: 0,
            site2: 1,
            value: C::new(22.0, -22.0),
            is_complex: false,
        }],
        doublon_holon_2site_indices: vec![DoublonHolon2SiteIndex {
            neighbors: vec![[0, 1]],
        }],
        doublon_holon_2site_params: (31..=36).map(|v| C::new(v as f64, 0.0)).collect(),
        doublon_holon_4site_indices: vec![DoublonHolon4SiteIndex {
            neighbors: vec![[0, 1, 2, 3]],
        }],
        doublon_holon_4site_params: (41..=50).map(|v| C::new(v as f64, 0.0)).collect(),
        rbm_section_widths: [1; 9],
        rbm_params: (61..=69).map(|v| C::new(v as f64, 0.0)).collect(),
        modpara: mvmc_expert_parsers::ModParaParameters {
            n_orbital_idx: 2,
            ..Default::default()
        },
        slater_params: vec![C::new(71.0, 0.0), C::new(72.0, 0.0)],
        opt_trans: vec![C::new(81.0, 0.0), C::new(82.0, 0.0)],
        ..Default::default()
    };
    let expected = [
        11., 22., 31., 32., 33., 34., 35., 36., 41., 42., 43., 44., 45., 46., 47., 48., 49., 50.,
        61., 62., 63., 64., 65., 66., 67., 68., 69., 71., 72., 81., 82.,
    ];
    let packed = pack_parameters(&data).unwrap();
    assert_eq!(packed.len(), 31);
    for (index, real) in expected.into_iter().enumerate() {
        let imaginary = match index {
            0 => -11.0,
            1 => -22.0,
            _ => 0.0,
        };
        assert_eq!(packed[index], C::new(real, imaginary), "slot {index}");
        assert_eq!(
            get_parameter_value(&data, index).unwrap(),
            C::new(real, imaginary)
        );
    }
    // Reverse-direction assignment checks each block, without a pack-generated oracle.
    let assigned: Vec<_> = (0..31).map(|i| C::new(101.0 + i as f64, -0.5)).collect();
    unpack_parameters(&mut data, &assigned).unwrap();
    assert_eq!(data.gutzwiller_terms[0].value, C::new(101.0, -0.5));
    assert_eq!(data.jastrow_terms[0].value, C::new(102.0, -0.5));
    assert_eq!(
        data.doublon_holon_2site_params,
        (103..=108)
            .map(|v| C::new(v as f64, -0.5))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        data.doublon_holon_4site_params,
        (109..=118)
            .map(|v| C::new(v as f64, -0.5))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        data.rbm_params,
        (119..=127)
            .map(|v| C::new(v as f64, -0.5))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        data.slater_params,
        vec![C::new(128.0, -0.5), C::new(129.0, -0.5)]
    );
    assert_eq!(
        data.opt_trans,
        vec![C::new(130.0, -0.5), C::new(131.0, -0.5)]
    );
}

fn assert_rejected_without_mutation(mut data: ExpertModeData) {
    let before = format!("{data:?}");
    assert!(pack_parameters(&data).is_err());
    assert!(get_parameter_value(&data, 0).is_err());
    assert!(set_parameter_value(&mut data, 0, C::new(11.0, 22.0)).is_err());
    assert!(unpack_parameters(&mut data, &[C::new(11.0, 22.0)]).is_err());
    assert_eq!(format!("{data:?}"), before);
}

#[test]
fn malformed_mapping_and_declared_overflow_fail_before_mutation() {
    use mvmc_expert_parsers::{ChargeRBMPhysLayerTerm, OrbitalTerm};
    for index in [-1, 1] {
        let mut data = ExpertModeData::default();
        data.modpara.n_orbital_idx = 1;
        data.slater_params = vec![C::new(0.5, -0.25)];
        data.orbital_terms = vec![OrbitalTerm {
            site1: 0,
            site2: 1,
            idx: index,
            is_complex: false,
            sign: 1,
        }];
        assert_rejected_without_mutation(data);
        let mut data = ExpertModeData::default();
        data.rbm_section_widths[0] = 1;
        data.rbm_params = vec![C::new(0.5, -0.25)];
        data.charge_rbm_phys_layer_terms = vec![ChargeRBMPhysLayerTerm {
            site: 0,
            idx: index,
            value: C::new(0.5, -0.25),
            is_complex: false,
        }];
        assert_rejected_without_mutation(data);
    }
    let data = ExpertModeData {
        rbm_section_widths: [usize::MAX, 1, 0, 0, 0, 0, 0, 0, 0],
        ..Default::default()
    };
    assert!(matches!(
        pack_parameters(&data),
        Err(ParameterAccessError::InvalidStorage("RBM width overflow"))
    ));
    assert_rejected_without_mutation(data);
    let mut data = ExpertModeData::default();
    data.rbm_section_widths[0] = usize::MAX;
    data.opt_trans = vec![C::new(0.5, -0.25)];
    assert!(matches!(
        pack_parameters(&data),
        Err(ParameterAccessError::InvalidStorage("total width overflow"))
    ));
    assert_rejected_without_mutation(data);
}

#[test]
fn stale_loaded_projection_declaration_is_nonmutating() {
    let mut data = original_heisenberg();
    assert!(data.native_complex_headers.contains_key("Gutzwiller"));
    for term in &mut data.gutzwiller_terms {
        term.is_complex = !term.is_complex;
    }
    assert!(matches!(
        pack_parameters(&data),
        Err(ParameterAccessError::InvalidDeclaration(_))
    ));
    assert_rejected_without_mutation(data);
}

#[test]
fn unchanged_opttrans_assignment_repairs_existing_weights_without_creating_absent_weights() {
    let mut data = ExpertModeData {
        opt_trans: vec![C::new(2.0, -1.0)],
        qp_weights: Some(QuantumProjectionWeights {
            qp_fix_weight: vec![C::new(0.5, 0.0)],
            qp_full_weight: vec![C::new(99.0, 99.0)],
            ..Default::default()
        }),
        ..Default::default()
    };
    unpack_parameters(&mut data, &[C::new(2.0, -1.0)]).unwrap();
    assert_eq!(
        data.qp_weights.as_ref().unwrap().qp_full_weight,
        vec![C::new(1.0, -0.5)]
    );
    data.qp_weights = None;
    unpack_parameters(&mut data, &[C::new(2.0, -1.0)]).unwrap();
    assert!(data.qp_weights.is_none());
}

fn original_heisenberg() -> ExpertModeData {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/original_heisenberg_parser_184/namelist.def");
    let data = mvmc_expert_parsers::parse_expert_mode_files(path).unwrap();
    assert!(data.input_errors.is_empty(), "{:?}", data.input_errors);
    data
}

#[test]
fn original_s120_roundtrip_and_first_last_updates() {
    let mut data = original_heisenberg();
    let n = data.count_variational_parameters();
    assert!(n > 1);
    assert_eq!(pack_parameters(&data).unwrap().len(), n);
    // Independent dyadic assignments, not a Rust-produced reference vector.
    set_parameter_value(&mut data, 0, C::new(0.5, -0.25)).unwrap();
    set_parameter_value(&mut data, n - 1, C::new(-0.75, 0.125)).unwrap();
    for (index, expected) in [(0, C::new(0.75, -0.75)), (n - 1, C::new(-0.5, -0.375))] {
        let before = get_parameter_value(&data, index).unwrap();
        set_parameter_value(&mut data, index, before + C::new(0.25, -0.5)).unwrap();
        assert_eq!(get_parameter_value(&data, index).unwrap(), expected);
        assert_eq!(pack_parameters(&data).unwrap()[index], expected);
    }
    // Original S120 algebraic roundtrip: no independent numerical oracle claim.
    let original = pack_parameters(&data).unwrap();
    let perturbed: Vec<_> = original.iter().map(|v| *v + C::new(0.01, 0.02)).collect();
    unpack_parameters(&mut data, &perturbed).unwrap();
    assert_eq!(pack_parameters(&data).unwrap(), perturbed);
    unpack_parameters(&mut data, &original).unwrap();
    assert_eq!(pack_parameters(&data).unwrap(), original);
}

#[test]
fn length_and_range_errors_leave_all_model_storage_unchanged() {
    let mut data = original_heisenberg();
    let before = format!("{data:?}");
    let values = pack_parameters(&data).unwrap();
    for wrong in [
        values[..values.len() - 1].to_vec(),
        vec![C::new(0.0, 0.0); values.len() + 1],
    ] {
        assert!(matches!(
            unpack_parameters(&mut data, &wrong),
            Err(ParameterAccessError::LengthMismatch { .. })
        ));
        assert_eq!(format!("{data:?}"), before);
    }
    for index in [values.len(), usize::MAX] {
        assert!(matches!(
            get_parameter_value(&data, index),
            Err(ParameterAccessError::IndexOutOfRange { .. })
        ));
        assert!(matches!(
            set_parameter_value(&mut data, index, C::new(11.0, 22.0)),
            Err(ParameterAccessError::IndexOutOfRange { .. })
        ));
        assert_eq!(format!("{data:?}"), before);
    }
}

#[test]
fn declared_slater_slots_and_opttrans_refresh_use_independent_values() {
    let mut data = ExpertModeData::default();
    // Three declared slots, including unmapped slots: never count mappings.
    data.modpara.n_orbital_idx = 3;
    data.slater_params = vec![C::new(1.0, 0.0), C::new(2.0, 0.0), C::new(3.0, 0.0)];
    data.opt_trans = vec![C::new(4.0, 0.0), C::new(5.0, 0.0)];
    data.qp_weights = Some(QuantumProjectionWeights {
        qp_fix_weight: vec![C::new(0.5, 0.0), C::new(-0.25, 0.0)],
        ..Default::default()
    });
    assert_eq!(
        pack_parameters(&data).unwrap(),
        vec![
            C::new(1.0, 0.0),
            C::new(2.0, 0.0),
            C::new(3.0, 0.0),
            C::new(4.0, 0.0),
            C::new(5.0, 0.0)
        ]
    );
    set_parameter_value(&mut data, 4, C::new(8.0, -4.0)).unwrap();
    assert_eq!(
        data.qp_weights.as_ref().unwrap().qp_full_weight,
        vec![
            C::new(2.0, 0.0),
            C::new(-1.0, 0.0),
            C::new(4.0, -2.0),
            C::new(-2.0, 1.0)
        ]
    );
    unpack_parameters(
        &mut data,
        &[
            C::new(-0.0, 0.0),
            C::new(11.0, 22.0),
            C::new(0.0, -0.0),
            C::new(2.0, 0.0),
            C::new(4.0, 0.0),
        ],
    )
    .unwrap();
    assert_eq!(data.slater_params[0].re.to_bits(), (-0.0f64).to_bits());
    assert_eq!(data.slater_params[2].im.to_bits(), (-0.0f64).to_bits());
    assert_eq!(data.slater_params[1], C::new(11.0, 22.0));
    assert_eq!(
        data.qp_weights.as_ref().unwrap().qp_full_weight,
        vec![
            C::new(1.0, 0.0),
            C::new(-0.5, 0.0),
            C::new(2.0, 0.0),
            C::new(-1.0, 0.0)
        ]
    );
}

#[test]
fn invalid_declared_storage_is_rejected_before_writes() {
    let mut data = ExpertModeData::default();
    data.modpara.n_orbital_idx = 2;
    data.slater_params = vec![C::new(0.5, -0.25)];
    let before = format!("{data:?}");
    assert!(matches!(
        pack_parameters(&data),
        Err(ParameterAccessError::InvalidStorage(_))
    ));
    assert!(matches!(
        unpack_parameters(&mut data, &[C::new(11.0, 22.0); 2]),
        Err(ParameterAccessError::InvalidStorage(_))
    ));
    assert!(matches!(
        set_parameter_value(&mut data, 0, C::new(11.0, 22.0)),
        Err(ParameterAccessError::InvalidStorage(_))
    ));
    assert_eq!(format!("{data:?}"), before);
}

#[test]
fn declared_rbm_holes_and_duplicate_mappings_are_preserved_and_repaired() {
    use mvmc_expert_parsers::ChargeRBMPhysLayerTerm;
    let mut data = ExpertModeData::default();
    data.rbm_section_widths[0] = 3;
    data.rbm_params = vec![C::new(1.0, 0.0), C::new(2.0, 0.0), C::new(3.0, 0.0)];
    data.charge_rbm_phys_layer_terms = vec![
        ChargeRBMPhysLayerTerm {
            site: 0,
            idx: 0,
            value: C::new(99.0, 0.0),
            is_complex: false,
        },
        ChargeRBMPhysLayerTerm {
            site: 1,
            idx: 0,
            value: C::new(-99.0, 0.0),
            is_complex: false,
        },
    ];
    assert_eq!(
        pack_parameters(&data).unwrap(),
        vec![C::new(1.0, 0.0), C::new(2.0, 0.0), C::new(3.0, 0.0)]
    );
    set_parameter_value(&mut data, 0, C::new(11.0, 22.0)).unwrap();
    assert!(data
        .charge_rbm_phys_layer_terms
        .iter()
        .all(|t| t.value == C::new(11.0, 22.0)));
    assert_eq!(get_parameter_value(&data, 2).unwrap(), C::new(3.0, 0.0));
    unpack_parameters(
        &mut data,
        &[C::new(0.25, -0.5), C::new(7.0, 8.0), C::new(9.0, 10.0)],
    )
    .unwrap();
    assert!(data
        .charge_rbm_phys_layer_terms
        .iter()
        .all(|t| t.value == C::new(0.25, -0.5)));
    assert_eq!(get_parameter_value(&data, 1).unwrap(), C::new(7.0, 8.0));
    assert_eq!(get_parameter_value(&data, 2).unwrap(), C::new(9.0, 10.0));
}

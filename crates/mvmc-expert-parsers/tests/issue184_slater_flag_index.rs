//! A143 direct public offset contract; no initialization or reference runtime.
use mvmc_expert_parsers::{
    get_slater_opt_flag_index, ChargeRBMPhysLayerTerm, DoublonHolon2SiteIndex,
    DoublonHolon4SiteIndex, ExpertModeData, GutzwillerTerm, JastrowTerm,
};
use num_complex::Complex64;

#[test]
fn declared_projection_and_nine_rbm_widths_set_direct_slater_component_offset() {
    let mut data = ExpertModeData::new();
    data.n_gutzwiller_idx = 3;
    data.n_jastrow_idx = 4;
    data.doublon_holon_2site_indices = vec![DoublonHolon2SiteIndex {
        neighbors: vec![[1, 2], [2, 0], [0, 1]],
    }];
    data.doublon_holon_4site_indices = vec![DoublonHolon4SiteIndex {
        neighbors: vec![[1, 2, 1, 2], [2, 0, 2, 0], [0, 1, 0, 1]],
    }];
    data.rbm_section_widths = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    data.rbm_params = vec![Complex64::new(0.0, 0.0); 45];
    data.modpara.n_orbital_idx = 3;
    data.optimization_flags = vec![7; 142];
    // Independent C block arithmetic: 3+4+6+10=23; 1+...+9=45.
    // Rust parameters0/2 address components136/140; Julia1/3 use137/141.
    for sparse_mapping in [false, true] {
        if sparse_mapping {
            data.gutzwiller_terms.push(GutzwillerTerm {
                site: 0,
                value: Complex64::new(0.0, 0.0),
                is_complex: false,
            });
            data.jastrow_terms.push(JastrowTerm {
                site1: 0,
                site2: 1,
                value: Complex64::new(0.0, 0.0),
                is_complex: false,
            });
            data.charge_rbm_phys_layer_terms
                .push(ChargeRBMPhysLayerTerm {
                    site: 0,
                    idx: 0,
                    value: Complex64::new(0.0, 0.0),
                    is_complex: false,
                });
        }
        let before = data.clone();
        assert_eq!(get_slater_opt_flag_index(&data, 0), 136);
        assert_eq!(get_slater_opt_flag_index(&data, 2), 140);
        assert_eq!(data.optimization_flags, before.optimization_flags);
        assert_eq!(data.rbm_params, before.rbm_params);
        assert_eq!(data.rbm_section_widths, before.rbm_section_widths);
        assert_eq!(data.gutzwiller_terms, before.gutzwiller_terms);
        assert_eq!(data.jastrow_terms, before.jastrow_terms);
        assert_eq!(
            data.charge_rbm_phys_layer_terms,
            before.charge_rbm_phys_layer_terms
        );
    }
}

#[test]
fn zero_projection_and_no_rbm_prefix_use_zero_based_real_components() {
    let mut data = ExpertModeData::new();
    data.modpara.n_orbital_idx = 3;
    assert_eq!(get_slater_opt_flag_index(&data, 0), 0);
    assert_eq!(get_slater_opt_flag_index(&data, 2), 4);
    assert!(data.optimization_flags.is_empty());
}

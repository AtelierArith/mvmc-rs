//! Julia test_orbital_qptrans_utils.jl: periodic signs and sparse defaults.
use mvmc_expert_parsers::{ExpertModeData, OrbitalTerm};

#[test]
fn normal_orbital_boundary_signs_and_unmapped_cells_match_julia() {
    for count in [4, -4] {
        let mut data = ExpertModeData::new();
        data.modpara.nsite = 4;
        data.modpara.nmp_trans = count;
        data.orbital_terms = vec![OrbitalTerm {
            site1: 0,
            site2: 1,
            idx: 2,

            is_complex: false,
            sign: -1,
        }];
        data.ensure_orbital_idx_matrix();
        let indices = data.orbital_idx_matrix.unwrap();
        let signs = data.orbital_sgn_matrix.unwrap();
        assert_eq!(indices[0][1], 2);
        assert_eq!(indices[1][0], 0, "Julia leaves unmapped indices at zero");
        if count > 0 {
            assert!(signs.iter().flatten().all(|&sign| sign == 1));
        } else {
            assert_eq!(signs[0][1], -1);
            assert_eq!(signs[1][0], 0, "normal orbitals are not mirrored");
            assert_eq!(signs[3][3], 0);
        }
    }
}

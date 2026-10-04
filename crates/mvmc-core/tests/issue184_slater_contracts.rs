//! Focused Slater47 semantic checks, not a claim of 47 runtime assertions.
use mvmc_core::{slater_update::update_slater_elm, state::VmcOptimizationState};
use mvmc_expert_parsers::{ExpertModeData, OrbitalTerm, QPTransEntry, QuantumProjectionWeights};
use num_complex::Complex64;

#[test]
fn declared_slater_slots_and_duplicate_mapping_preserve_coefficients() {
    // Original S163's three mappings. C owns coefficients separately from
    // mappings: a duplicate mapping cannot overwrite a declared coefficient.
    // readdef.c GetInfoOrbitalAntiParallel overrides signs for periodic input.
    for nmp in [-1, 1] {
        let mut data = ExpertModeData::new();
        data.modpara.nsite = 3;
        data.modpara.nmp_trans = nmp;
        data.modpara.n_orbital_idx = 3;
        data.slater_params = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(2.5, 0.0),
        ];
        for (site1, site2, idx, sign) in [(0, 1, 0, 1), (1, 2, 2, -1), (2, 0, 2, 1)] {
            data.orbital_terms.push(OrbitalTerm {
                site1,
                site2,
                idx,
                sign,
                is_complex: true,
            });
        }
        let before = data.slater_params.clone();
        data.ensure_orbital_idx_matrix();
        let idx = data.orbital_idx_matrix.as_ref().unwrap();
        let signs = data.orbital_sgn_matrix.as_ref().unwrap();
        assert_eq!(idx.len(), 3);
        assert!(idx.iter().all(|row| row.len() == 3));
        assert_eq!((idx[0][1], idx[1][2], idx[2][0]), (0, 2, 2));
        assert_eq!(
            (signs[0][1], signs[1][2], signs[2][0]),
            (1, if nmp < 0 { -1 } else { 1 }, 1)
        );
        assert_eq!(data.slater_params, before);
    }
}

#[test]
fn supplied_orbital_cache_is_preserved_and_missing_weights_leave_state_unchanged() {
    // S164's identity assertion has no Rust borrowed-object counterpart;
    // preserve contents instead. S103's missing-weight early return concerns
    // state, not data-cache atomicity (the cache is prepared before the return).
    let mut data = ExpertModeData::new();
    data.modpara.nsite = 2;
    data.orbital_idx_matrix = Some(vec![vec![0, 0], vec![0, 0]]);
    data.orbital_sgn_matrix = Some(vec![vec![1, -1], vec![-1, 1]]);
    let indices = data.orbital_idx_matrix.clone();
    let signs = data.orbital_sgn_matrix.clone();
    assert!(data.qp_weights.is_none());
    let mut state = VmcOptimizationState::zeros(2, 1, 0, 1, 1, 1, false, false);
    let before = format!("{state:?}");
    state.validate_declared_mode(&data).unwrap();
    update_slater_elm(&mut data, &mut state);
    assert_eq!(format!("{state:?}"), before);
    assert_eq!(data.orbital_idx_matrix, indices);
    assert_eq!(data.orbital_sgn_matrix, signs);
}

#[test]
fn public_slater_update_composes_opttrans_before_fixed_translation_in_each_plane() {
    // C slater.c UpdateSlaterElm_fcmp: ori=QPOptTrans[ri], tri=QPTrans[ori].
    // Unlike two cyclic shifts, these permutations do not commute. Expected
    // maps below are independently enumerated, not obtained from Rust helpers.
    // SPGL=(cs,cc,ss)=(0,1,0) is the literal beta=0 kernel input; this tests
    // wiring rather than claiming this is the one-node quadrature rule.
    let mut data = ExpertModeData::new();
    data.modpara.nsite = 4;
    data.modpara.nmp_trans = -2;
    data.modpara.nsp_gauss_leg = 1;
    data.modpara.n_orbital_idx = 16;
    data.n_qp_opt_trans = 2;
    data.qp_opt_trans = vec![vec![0, 1, 2, 3], vec![1, 0, 2, 3]];
    data.qp_opt_trans_sgn = vec![vec![1; 4], vec![-1, 1, 1, 1]];
    for (site_map, site_sign) in [
        (vec![0, 1, 2, 3], vec![1; 4]),
        (vec![1, 2, 3, 0], vec![1, -1, 1, 1]),
    ] {
        data.qp_trans_entries.push(QPTransEntry {
            weight: Complex64::new(1.0, 0.0),
            site_map,
            site_sign,
        });
    }
    for i in 0..4 {
        for j in 0..4 {
            data.orbital_terms.push(OrbitalTerm {
                site1: i,
                site2: j,
                idx: 4 * i + j,
                sign: 1,
                is_complex: false,
            });
            data.slater_params
                .push(Complex64::new((4 * i + j + 1) as f64, 0.0));
        }
    }
    data.qp_weights = Some(QuantumProjectionWeights {
        qp_full_weight: vec![Complex64::new(1.0, 0.0); 4],
        qp_fix_weight: vec![Complex64::new(1.0, 0.0); 2],
        spgl_cos: vec![Complex64::new(1.0, 0.0)],
        spgl_sin: vec![Complex64::new(0.0, 0.0)],
        spgl_cos_sin: vec![Complex64::new(0.0, 0.0)],
        spgl_cos_cos: vec![Complex64::new(1.0, 0.0)],
        spgl_sin_sin: vec![Complex64::new(0.0, 0.0)],
    });
    let mut state = VmcOptimizationState::zeros(4, 2, 0, 16, 4, 1, false, false);
    state.validate_declared_mode(&data).unwrap();
    update_slater_elm(&mut data, &mut state);
    let maps = [[0, 1, 2, 3], [1, 2, 3, 0], [1, 0, 2, 3], [2, 1, 3, 0]];
    let signs = [[1, 1, 1, 1], [1, -1, 1, 1], [-1, 1, 1, 1], [1, 1, 1, 1]];
    assert_eq!(state.slater_matrix.slater_elm.as_slice().len(), 4 * 8 * 8);
    for q in 0..4 {
        for i in 0..4 {
            for j in 0..4 {
                let sign = signs[q][i] * signs[q][j];
                let forward = ((4 * maps[q][i] + maps[q][j] + 1) * sign) as f64;
                let reverse = ((4 * maps[q][j] + maps[q][i] + 1) * sign) as f64;
                let elm = &state.slater_matrix.slater_elm;
                assert_eq!(elm.get(q, i, j), Complex64::new(0.0, 0.0));
                assert_eq!(elm.get(q, i + 4, j + 4), Complex64::new(0.0, 0.0));
                assert_eq!(elm.get(q, i, j + 4), Complex64::new(forward, 0.0));
                assert_eq!(elm.get(q, i + 4, j), Complex64::new(-reverse, 0.0));
            }
        }
    }
}

#[test]
fn parsed_heisenberg_initialization_wires_all_slater_planes_repeatably() {
    // S104's public parse -> seeded initialization -> QP weights -> update
    // lifecycle, using the offline six-site fixture, NOT the original
    // sixteen-site Julia sample and NOT a full sampling-trajectory claim.
    check_parsed_heisenberg_wiring("heisenberg_chain_real", 6, 1);
}

#[test]
fn original_s104_sixteen_site_seeded_input_wires_all_slater_planes() {
    // Exact original sample inputs, copied byte-for-byte with provenance.
    // Like the original Julia test this stops after Slater construction; it
    // does not run the 300 optimization steps or 1000 sampling records.
    check_parsed_heisenberg_wiring("slater-s104", 16, 123456789);
}

fn check_parsed_heisenberg_wiring(fixture: &str, nsite: usize, seed: i64) {
    let input = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/physcal_181")
        .join(fixture)
        .join("inputs/namelist.def");
    let mut repeats = Vec::new();
    for _ in 0..2 {
        let mut data = mvmc_expert_parsers::parse_expert_mode_files(&input).unwrap();
        assert_eq!(data.modpara.nsite, nsite as i64);
        assert_eq!(data.modpara.rnd_seed, seed);
        assert_eq!(data.modpara.nsp_gauss_leg, 8);
        assert_eq!(data.modpara.nmp_trans, -1);
        assert!(!data.qp_trans_entries.is_empty());
        let mut rng = sfmt19937::Sfmt19937Rng::new(data.modpara.rnd_seed as u32);
        mvmc_expert_parsers::utils::parameter_init::initialize_parameters(&mut data, &mut rng)
            .unwrap();
        mvmc_core::qp::init_qp_weight(&mut data);
        let nqp = data.modpara.nsp_gauss_leg as usize
            * data.modpara.nmp_trans.unsigned_abs() as usize
            * data.n_qp_opt_trans.max(1) as usize;
        assert_eq!(nqp, 8);
        assert_eq!(data.qp_weights.as_ref().unwrap().qp_full_weight.len(), nqp);
        let mut state = VmcOptimizationState::zeros(
            nsite,
            data.modpara.nelec as usize,
            0,
            data.slater_params.len(),
            nqp,
            1,
            false,
            false,
        );
        state.validate_declared_mode(&data).unwrap();
        update_slater_elm(&mut data, &mut state);
        let table = state.slater_matrix.slater_elm.as_slice();
        let nsite2 = 2 * nsite;
        assert_eq!(table.len(), 8 * nsite2 * nsite2);
        assert!(table.iter().all(|x| x.re.is_finite() && x.im == 0.0));
        assert!(table.iter().any(|x| x.re != 0.0));
        // C's four spin blocks are antisymmetric by construction. This is an
        // algebraic invariant, not a Rust-produced expected numerical table.
        for q in 0..nqp {
            for i in 0..nsite2 {
                for j in 0..nsite2 {
                    assert_eq!(
                        state.slater_matrix.slater_elm.get(q, i, j),
                        -state.slater_matrix.slater_elm.get(q, j, i)
                    );
                }
            }
        }
        repeats.push((data.slater_params, table.to_vec(), format!("{rng:?}")));
    }
    assert_eq!(repeats[0], repeats[1]);
}

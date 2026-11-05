#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;
use mvmc_core::{qp, read_opt_para_file, ExpertModeData};
use num_complex::Complex64;

#[test]
fn full_record_loads_opttrans_after_all_other_factors() {
    let mut data = ExpertModeData::new();
    data.opt_trans = vec![Complex64::new(1.0, 0.0); 2];
    let path = std::env::temp_dir().join(format!("mvmc-opttrans-red-{}.def", std::process::id()));
    std::fs::write(&path, "1 2 3 4 5 6 0.7 -0.8 9.9 0.9 0.2 9.9").unwrap();
    let result = read_opt_para_file(&mut data, &path);
    std::fs::remove_file(path).unwrap();
    assert_eq!(result.unwrap(), 2);
    assert_eq!(
        data.opt_trans,
        vec![Complex64::new(0.7, -0.8), Complex64::new(0.9, 0.2)]
    );
}

#[test]
fn data_weight_initialization_and_refresh_include_opttrans_sectors() {
    let mut data = ExpertModeData::new();
    data.modpara.nsp_gauss_leg = 1;
    data.modpara.nmp_trans = 2;
    data.para_qp_trans = vec![Complex64::new(1.0, 0.0), Complex64::new(0.5, 0.0)];
    data.opt_trans = vec![Complex64::new(0.25, 0.0), Complex64::new(0.75, 0.0)];
    qp::init_qp_weight(&mut data);
    assert_eq!(data.qp_weights.as_ref().unwrap().qp_full_weight.len(), 4);
    data.opt_trans[1] = Complex64::new(0.5, -0.25);
    qp::update_qp_weight_for(&mut data);
    let weight = data.qp_weights.unwrap().qp_full_weight[2];
    numerical_comparison::assert_values_close(
        [weight.re, weight.im],
        [data.opt_trans[1].re, data.opt_trans[1].im],
        4.0 * f64::EPSILON,
        4.0 * f64::EPSILON,
        "sector weight product",
    );
}

fn projection_model(mode: &str, leg: i64, boundary: i64, mapping: &str) -> ExpertModeData {
    use mvmc_expert_parsers::{OrbitalTerm, QPTransEntry};
    let mut data = ExpertModeData::new();
    data.modpara.nsite = 4;
    data.modpara.nelec = 1;
    data.modpara.nmp_trans = 2 * boundary;
    data.modpara.nsp_gauss_leg = leg;
    data.i_flg_orbital_general = i64::from(mode == "fsz");
    data.n_qp_trans = 2;
    data.n_qp_opt_trans = 3;
    let dim = if mode == "fsz" { 8 } else { 4 };
    data.modpara.n_orbital_idx = dim * dim;
    data.slater_params = (0..dim * dim)
        .map(|idx| {
            Complex64::new(
                (idx + 1) as f64 / 16.0,
                if mode == "real" {
                    0.0
                } else {
                    (-idx) as f64 / 32.0
                },
            )
        })
        .collect();
    for i in 0..dim {
        for j in 0..dim {
            let idx = i * dim + j;
            data.orbital_terms.push(OrbitalTerm {
                site1: i,
                site2: j,
                idx,
                is_complex: mode != "real",
                sign: if (i + j) % 2 == 1 { -1 } else { 1 },
            });
        }
    }
    data.ensure_orbital_idx_matrix();
    data.qp_trans_entries = vec![
        QPTransEntry {
            weight: Complex64::new(1.0, 0.25),
            site_map: vec![0, 1, 2, 3],
            site_sign: if boundary == -1 {
                vec![1, -1, -1, 1]
            } else {
                vec![1; 4]
            },
        },
        QPTransEntry {
            weight: Complex64::new(-0.5, -0.125),
            site_map: vec![3, 2, 1, 0],
            site_sign: if boundary == -1 {
                vec![-1, 1, -1, 1]
            } else {
                vec![1; 4]
            },
        },
    ];
    data.para_qp_trans = data.qp_trans_entries.iter().map(|t| t.weight).collect();
    data.qp_opt_trans = vec![vec![0, 1, 2, 3], vec![1, 2, 3, 0], vec![3, 2, 1, 0]];
    data.qp_opt_trans_sgn = if boundary == -1 {
        vec![vec![1; 4], vec![-1, 1, -1, 1], vec![1, -1, -1, 1]]
    } else {
        vec![vec![1; 4]; 3]
    };
    data.opt_trans = vec![
        Complex64::new(0.25, -0.125),
        Complex64::new(-0.75, 0.5),
        Complex64::new(0.5, 0.25),
    ];
    match mapping {
        "absent" => {
            data.qp_opt_trans.clear();
            data.qp_opt_trans_sgn.clear();
        }
        "short_maps" => {
            data.qp_opt_trans.truncate(1);
            data.qp_opt_trans[0] = vec![2, 3, 0, 1];
        }
        "short_signs" => {
            data.qp_opt_trans_sgn.truncate(1);
            data.qp_opt_trans_sgn[0] = vec![-1, 1, 1, -1];
        }
        "missing_maps" => data.qp_opt_trans.clear(),
        "missing_signs" => data.qp_opt_trans_sgn.clear(),
        "complete" => {}
        _ => panic!("unknown mapping {mapping}"),
    }
    qp::init_qp_weight(&mut data);
    data
}

#[test]
fn c_normal_slater_preserves_subthreshold_coefficients() {
    let mut data = projection_model("real", 1, 1, "complete");
    let value = f64::from_bits(1e-14_f64.to_bits() - 1);
    data.slater_params.fill(Complex64::new(value, 0.0));
    let mut state = mvmc_core::VmcOptimizationState::zeros(4, 1, 0, 4, 6, 1, false, false);
    state.validate_declared_mode(&data).unwrap();
    mvmc_core::slater_update::update_slater_elm(&mut data, &mut state);
    assert!(state
        .slater_matrix
        .slater_elm
        .as_slice()
        .iter()
        .any(|entry| entry.re != 0.0));
}

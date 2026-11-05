//! Source-level real FSZ sampler parity; see the fixture README for the
//! reference's unused save-log argument adaptation.

#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;
use mvmc_core::sampling::vmc_make_sample_fsz_real;
use mvmc_core::{ExpertModeData, VmcOptimizationState};
use mvmc_expert_parsers::{GutzwillerTerm, JastrowTerm, LocSpinTerm, QuantumProjectionWeights};
use num_complex::Complex64;
use sfmt19937::Sfmt19937Rng;

fn data(case: &str) -> ExpertModeData {
    let mut d = ExpertModeData::default();
    d.modpara.nsite = 4;
    d.modpara.nelec = 2;
    d.modpara.two_sz = if case == "fixed" { 0 } else { -1 };
    d.modpara.nex_update_path = match case {
        "localspin" => 2,
        "mixed" => 3,
        _ => 1,
    };
    d.modpara.nvmc_warmup = 2;
    d.modpara.nvmc_sample = 3;
    d.modpara.nvmc_interval = 2;
    d.i_flg_orbital_general = 1;
    d.qp_weights = Some(QuantumProjectionWeights {
        qp_full_weight: vec![Complex64::new(0.625, 0.0), Complex64::new(0.375, 0.0)],
        ..Default::default()
    });
    d.n_gutzwiller_idx = 1;
    d.n_jastrow_idx = 1;
    d.gutzwiller_idx = vec![0; 4];
    d.jastrow_idx = vec![vec![0; 4]; 4];
    d.gutzwiller_terms = (0..4)
        .map(|site| GutzwillerTerm {
            site,
            value: Complex64::new(0.13, 0.0),
            is_complex: false,
        })
        .collect();
    d.jastrow_terms = (0..4)
        .flat_map(|site1| {
            (site1 + 1..4).map(move |site2| JastrowTerm {
                site1,
                site2,
                value: Complex64::new(-0.17, 0.0),
                is_complex: false,
            })
        })
        .collect();
    d.locspin_terms = match case {
        "localspin" => (0..4)
            .map(|site| LocSpinTerm {
                site,
                spin_value: 1,
            })
            .collect(),
        "mixed" => vec![LocSpinTerm {
            site: 0,
            spin_value: 1,
        }],
        _ => vec![],
    };
    d
}
fn state() -> VmcOptimizationState {
    let mut s = VmcOptimizationState::zeros(4, 2, 2, 0, 2, 3, false, true);
    for qp in 0..2 {
        for i in 0..8 {
            for j in i + 1..8 {
                let z = Complex64::new(
                    (((17 * i + 13 * j + 7 * qp) % 31) as i64 - 15) as f64 / 7.0 + 0.125,
                    0.0,
                );
                s.slater_matrix.slater_elm.set(qp, i, j, z);
                s.slater_matrix.slater_elm.set(qp, j, i, -z);
            }
        }
    }
    s
}

#[test]
fn failed_real_fsz_initialization_preserves_saved_samples_and_counters() {
    let d = data("conduction");
    let mut s = state();
    s.slater_matrix
        .slater_elm
        .as_mut_slice()
        .fill(Complex64::new(f64::NAN, 0.0));
    s.electron_config.ele_idx.fill(7);
    s.electron_config.ele_cfg.fill(7);
    s.electron_config.ele_num.fill(7);
    s.electron_config.ele_proj_cnt.fill(7);
    s.electron_config.ele_spn.fill(7);
    let saved = s.electron_config.clone();
    let error = vmc_make_sample_fsz_real(&d, &mut s, &mut Sfmt19937Rng::new(1)).unwrap_err();
    assert_eq!(error, mvmc_core::CalcMAllError::NonFinitePfaffian { qp: 0 });
    assert_eq!(s.electron_config.ele_idx, saved.ele_idx);
    assert_eq!(s.electron_config.ele_cfg, saved.ele_cfg);
    assert_eq!(s.electron_config.ele_num, saved.ele_num);
    assert_eq!(s.electron_config.ele_proj_cnt, saved.ele_proj_cnt);
    assert_eq!(s.electron_config.ele_spn, saved.ele_spn);
    assert_eq!(s.electron_config.counter, saved.counter);
}

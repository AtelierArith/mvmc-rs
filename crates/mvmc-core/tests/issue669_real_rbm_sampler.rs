//! Issue #669: the real normal-mode sampler applies the RBM factor.
//!
//! C's real path has no RBM code (`vmcmake_real.c`, `calham_real.c`; a C defect,
//! tmisawa/Julia-mVMC#59). Rust keeps the correct mathematics: with real RBM coefficients and
//! real orbitals, the real sampler must sample the same `|Pf x projection x RBM|^2` as the
//! complex normal sampler (C `vmcmake.c`), which adds `LogRBMRatio` to every proposal.
//!
//! * `real_and_complex_samplers_agree_with_real_rbm` runs both samplers on the same model,
//!   seed and orbitals and requires identical saved configurations, counters and final RNG
//!   state. Every quantity of the complex path is real here; the two paths differ only by
//!   floating-point rounding of the Pfaffian updates (~1e-13 relative), so the comparison is
//!   exact unless a Metropolis decision lies within that rounding of its draw. The test
//!   checks the smallest decision margin (observed 8.1e-4) against a 1e-9 floor.
//! * `zero_rbm_coefficients_leave_both_samplers_unchanged` checks that RBM sections with all
//!   coefficients zero reproduce the RBM-free trajectory exactly in both paths (same draws,
//!   same configurations, same RNG state): the factor adds an exact zero to the weight.
//! * `rbm_factor_changes_the_real_trajectory` guards that the RBM factor matters in this model.
use mvmc_core::sampling::driver::trace;
use mvmc_core::{ExpertModeData, VmcOptimizationState};
use mvmc_expert_parsers::{
    ChargeRBMHiddenLayerTerm, ChargeRBMPhysHiddenTerm, ChargeRBMPhysLayerTerm,
    GeneralRBMHiddenLayerTerm, GeneralRBMPhysHiddenTerm, GeneralRBMPhysLayerTerm, GutzwillerTerm,
    JastrowTerm, LocSpinTerm,
};
use num_complex::Complex64;
use sfmt19937::Sfmt19937Rng;

const N_SITE: usize = 6;
const N_ELEC: usize = 2;
const N_QP: usize = 2;
const N_SAMPLE: usize = 60;
const SEED: u32 = 6690;

#[derive(Clone, Copy, PartialEq)]
enum Rbm {
    Absent,
    Zero,
    Real,
}

fn model(rbm: Rbm) -> ExpertModeData {
    let mut data = ExpertModeData::new();
    data.modpara.nsite = N_SITE as i64;
    data.modpara.nelec = N_ELEC as i64;
    data.modpara.nmp_trans = N_QP as i64;
    data.modpara.nvmc_warmup = 10;
    data.modpara.nvmc_sample = N_SAMPLE as i64;
    data.modpara.nvmc_interval = 2;
    data.modpara.two_sz = 0;
    // Hopping and exchange proposals (an accepted exchange is two RBM hops).
    data.modpara.nex_update_path = 1;
    data.locspin_terms = (0..N_SITE as i64)
        .map(|site| LocSpinTerm {
            site,
            spin_value: 0,
        })
        .collect();
    data.n_qp_trans = N_QP as i64;
    data.para_qp_trans = vec![Complex64::new(1.0, 0.0), Complex64::new(-0.375, 0.0)];
    data.n_gutzwiller_idx = 1;
    data.gutzwiller_idx = vec![0; N_SITE];
    data.gutzwiller_terms = vec![GutzwillerTerm {
        site: 0,
        value: Complex64::new(0.125, 0.0),
        is_complex: false,
    }];
    data.n_jastrow_idx = 1;
    data.jastrow_idx = (0..N_SITE)
        .map(|i| (0..N_SITE).map(|j| if i == j { -1 } else { 0 }).collect())
        .collect();
    data.jastrow_terms = vec![JastrowTerm {
        site1: 0,
        site2: 1,
        value: Complex64::new(-0.2, 0.0),
        is_complex: false,
    }];
    if rbm != Rbm::Absent {
        let scale = if rbm == Rbm::Zero { 0.0 } else { 1.0 };
        let value = |re: f64| Complex64::new(scale * re, 0.0);
        let n = N_SITE as i64;
        data.modpara.nneuron_charge = 2;
        data.modpara.nneuron_general = 2;
        data.charge_rbm_phys_layer_terms = (0..n)
            .map(|site| ChargeRBMPhysLayerTerm {
                site,
                idx: 0,
                value: value(0.0625),
                is_complex: false,
            })
            .collect();
        data.charge_rbm_hidden_layer_terms = (0..2)
            .map(|h| ChargeRBMHiddenLayerTerm {
                site: h,
                idx: h,
                value: value(0.125 - 0.25 * h as f64),
                is_complex: false,
            })
            .collect();
        data.charge_rbm_phys_hidden_terms = (0..2)
            .flat_map(|h| {
                (0..n).map(move |site| ChargeRBMPhysHiddenTerm {
                    site1: site,
                    site2: h,
                    idx: h * n + site,
                    value: Complex64::new(
                        scale * 0.375 * ((((h * n + site) % 5) as f64) - 2.0),
                        0.0,
                    ),
                    is_complex: false,
                })
            })
            .collect();
        data.general_rbm_phys_layer_terms = (0..2 * n)
            .map(|k| GeneralRBMPhysLayerTerm {
                site: k % n,
                spin: k / n,
                idx: k,
                value: value(0.15 * ((k % 3) as f64 - 1.0)),
                is_complex: false,
            })
            .collect();
        data.general_rbm_hidden_layer_terms = (0..2)
            .map(|h| GeneralRBMHiddenLayerTerm {
                site: h,
                idx: h,
                value: value(0.1 + 0.05 * h as f64),
                is_complex: false,
            })
            .collect();
        let mut coupling = Vec::new();
        for h in 0..2i64 {
            for spin in 0..2i64 {
                for site in 0..n {
                    let k = (h * 2 + spin) * n + site;
                    coupling.push(GeneralRBMPhysHiddenTerm {
                        site1: site,
                        spin,
                        site2: h,
                        idx: k,
                        value: value(0.3 * ((k % 4) as f64 - 1.5)),
                        is_complex: false,
                    });
                }
            }
        }
        data.general_rbm_phys_hidden_terms = coupling;
        // C section order: charge/spin/general phys, charge/spin/general hidden, couplings.
        let n = N_SITE;
        data.rbm_section_widths = [1, 0, 2 * n, 2, 0, 2, 2 * n, 0, 2 * 2 * n];
    }
    mvmc_expert_parsers::utils::qp_weight::init_qp_weight(&mut data);
    data
}

fn state(complex: bool) -> VmcOptimizationState {
    let mut state =
        VmcOptimizationState::zeros(N_SITE, N_ELEC, 2, 0, N_QP, N_SAMPLE, complex, false);
    // Real antisymmetric orbitals; the complex sampler sees them with zero imaginary part.
    for qp in 0..N_QP {
        for i in 0..2 * N_SITE {
            for j in i + 1..2 * N_SITE {
                let z = Complex64::new(
                    (((17 * i + 13 * j + 7 * qp) % 31) as i64 - 15) as f64 / 7.0 + 0.125,
                    0.0,
                );
                state.slater_matrix.slater_elm.set(qp, i, j, z);
                state.slater_matrix.slater_elm.set(qp, j, i, -z);
            }
        }
    }
    state
}

struct Run {
    ele_idx: Vec<i64>,
    ele_num: Vec<i64>,
    counter: Vec<i64>,
    accepted: usize,
    rng_words: u128,
    next_words: Vec<u32>,
    min_margin: f64,
}

fn run(rbm: Rbm, complex: bool) -> Run {
    let data = model(rbm);
    let mut state = state(complex);
    let mut rng = Sfmt19937Rng::new(SEED);
    let capture = trace::nested_decisions();
    let stats = if complex {
        mvmc_core::vmc_make_sample(&data, &mut state, &mut rng)
    } else {
        mvmc_core::vmc_make_sample_real(&data, &mut state, &mut rng)
    }
    .unwrap();
    let decisions = capture.finish();
    assert_eq!(stats.saved, N_SAMPLE);
    let min_margin = decisions
        .iter()
        .filter(|(weight, _)| *weight >= 0.0)
        .map(|(weight, draw)| (weight - draw).abs())
        .fold(f64::INFINITY, f64::min);
    let rng_words = rng.words_consumed();
    let mut next_words = vec![0; 624];
    rng.dump_rand32(&mut next_words);
    Run {
        ele_idx: state.electron_config.ele_idx.clone(),
        ele_num: state.electron_config.ele_num.clone(),
        counter: state.electron_config.counter[..4].to_vec(),
        accepted: stats.accepted,
        rng_words,
        next_words,
        min_margin,
    }
}

fn assert_same_trajectory(a: &Run, b: &Run, label: &str) {
    assert_eq!(a.ele_idx, b.ele_idx, "{label}: saved ele_idx");
    assert_eq!(a.ele_num, b.ele_num, "{label}: saved ele_num");
    assert_eq!(
        a.counter, b.counter,
        "{label}: proposal/acceptance counters"
    );
    assert_eq!(a.accepted, b.accepted, "{label}: accepted moves");
    assert_eq!(a.rng_words, b.rng_words, "{label}: RNG words consumed");
    assert_eq!(a.next_words, b.next_words, "{label}: next RNG words");
}

#[test]
fn real_and_complex_samplers_agree_with_real_rbm() {
    let real = run(Rbm::Real, false);
    let complex = run(Rbm::Real, true);
    // Real vs complex Pfaffian updates round differently (~1e-13 relative); decisions closer
    // than this floor to their draw could legitimately flip.
    assert!(real.min_margin > 1e-9, "margin {}", real.min_margin);
    assert!(complex.min_margin > 1e-9, "margin {}", complex.min_margin);
    assert!(
        real.counter[2] > 0 && real.counter[3] > 0,
        "{:?}",
        real.counter
    );
    assert_same_trajectory(&real, &complex, "real vs complex with RBM");
}

#[test]
fn zero_rbm_coefficients_leave_both_samplers_unchanged() {
    for complex in [false, true] {
        let absent = run(Rbm::Absent, complex);
        let zero = run(Rbm::Zero, complex);
        assert_same_trajectory(
            &absent,
            &zero,
            &format!("complex={complex}: zero vs absent"),
        );
    }
    // Without RBM, the real and complex samplers also agree.
    assert_same_trajectory(
        &run(Rbm::Absent, false),
        &run(Rbm::Absent, true),
        "real vs complex without RBM",
    );
}

#[test]
fn rbm_factor_changes_the_real_trajectory() {
    let with = run(Rbm::Real, false);
    let without = run(Rbm::Absent, false);
    assert_ne!(with.ele_idx, without.ele_idx);
}

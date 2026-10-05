//! Issue #403: the complex FSZ sampler applies the RBM factor.
//!
//! C's FSZ code has no RBM factor (a C defect, tmisawa/Julia-mVMC#59); Rust follows the
//! correct math consistently with its FSZ Hamiltonian and Green-function kernels.
//!
//! * `per_move_acceptance_matches_full_recomputation` replays every proposal recorded by
//!   the sampling trace against an independent full evaluation of
//!   `Psi = Pf x projection x RBM` (fresh Pfaffians from `calc_m_all_fsz_complex`, projection
//!   counters from `make_proj_cnt`, direct `log_rbm_val`), and requires the recorded
//!   Metropolis decision to equal `w > draw` for every move.
//! * `incremental_hopping_update_matches_full_recomputation` cross-checks, on
//!   spin-conserving hops where the non-FSZ complex path is valid, that its incremental
//!   `update_rbm_cnt_hopping` ratio equals the full-recomputation ratio used by the FSZ path.
use mvmc_core::sampling::driver::{trace, vmc_make_sample_fsz};
use mvmc_core::sampling::projection::make_proj_cnt;
use mvmc_core::sampling::rbm::{
    log_rbm_ratio, log_rbm_val, make_rbm_cnt, update_rbm_cnt_hopping, RbmConfig,
};
use mvmc_core::{ExpertModeData, VmcOptimizationState};
use mvmc_expert_parsers::{
    GeneralRBMHiddenLayerTerm, GeneralRBMPhysHiddenTerm, GeneralRBMPhysLayerTerm, GutzwillerTerm,
    JastrowTerm, LocSpinTerm,
};
use num_complex::Complex64;
use sfmt19937::Sfmt19937Rng;

const N_SITE: usize = 4;
const N_ELEC: usize = 2;
const N_SAMPLE: usize = 40;

fn model(with_rbm: bool) -> (ExpertModeData, VmcOptimizationState) {
    let mut data = ExpertModeData::new();
    data.modpara.nsite = N_SITE as i64;
    data.modpara.nelec = N_ELEC as i64;
    data.modpara.nmp_trans = 2;
    data.modpara.nvmc_warmup = 10;
    data.modpara.nvmc_sample = 40;
    data.modpara.nvmc_interval = 1;
    data.modpara.two_sz = -1;
    data.modpara.nex_update_path = 0;
    data.i_flg_orbital_general = 1;
    data.complex_flags = vec![1];
    data.locspin_terms = (0..N_SITE as i64)
        .map(|site| LocSpinTerm {
            site,
            spin_value: 0,
        })
        .collect();
    data.n_qp_trans = 2;
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
    if with_rbm {
        data.modpara.nneuron_general = 2;
        data.general_rbm_phys_layer_terms = (0..2 * N_SITE as i64)
            .map(|k| GeneralRBMPhysLayerTerm {
                site: k % N_SITE as i64,
                spin: k / N_SITE as i64,
                idx: k,
                value: Complex64::new(0.15 * ((k % 3) as f64 - 1.0), 0.08 * ((k % 2) as f64 - 0.5)),
                is_complex: true,
            })
            .collect();
        data.general_rbm_hidden_layer_terms = (0..2)
            .map(|h| GeneralRBMHiddenLayerTerm {
                site: h,
                idx: h,
                value: Complex64::new(0.1 + 0.05 * h as f64, -0.07),
                is_complex: true,
            })
            .collect();
        let mut coupling = Vec::new();
        for h in 0..2i64 {
            for spin in 0..2i64 {
                for site in 0..N_SITE as i64 {
                    let k = (h * 2 + spin) * N_SITE as i64 + site;
                    coupling.push(GeneralRBMPhysHiddenTerm {
                        site1: site,
                        spin,
                        site2: h,
                        idx: k,
                        value: Complex64::new(
                            0.3 * ((k % 4) as f64 - 1.5),
                            0.12 * ((k % 2) as f64 - 0.5),
                        ),
                        is_complex: true,
                    });
                }
            }
        }
        data.general_rbm_phys_hidden_terms = coupling;
        // C section order: charge/spin/general phys, charge/spin/general hidden, then couplings.
        data.rbm_section_widths = [0, 0, 2 * N_SITE, 0, 0, 2, 0, 0, 2 * 2 * N_SITE];
    }
    mvmc_expert_parsers::utils::qp_weight::init_qp_weight(&mut data);
    let mut state = VmcOptimizationState::zeros(N_SITE, N_ELEC, 2, 0, 2, N_SAMPLE, true, true);
    for qp in 0..2 {
        for i in 0..2 * N_SITE {
            for j in i + 1..2 * N_SITE {
                let z = Complex64::new(
                    (((17 * i + 13 * j + 7 * qp) % 31) as i64 - 15) as f64 / 7.0 + 0.125,
                    (((11 * i + 3 * j + qp) % 19) as i64 - 9) as f64 / 13.0,
                );
                state.slater_matrix.slater_elm.set(qp, i, j, z);
                state.slater_matrix.slater_elm.set(qp, j, i, -z);
            }
        }
    }
    (data, state)
}

/// Independent full evaluation of ln Psi up to configuration-independent constants.
struct Wavefunction<'a> {
    data: &'a ExpertModeData,
    slater: &'a VmcOptimizationState,
    pool: mvmc_core::state::ThreadedPfaPackWorkspace,
}

impl Wavefunction<'_> {
    fn log_psi(&self, idx: &[i64], spn: &[i64]) -> (Complex64, Complex64, f64) {
        let n_size = 2 * N_ELEC;
        let mut num = vec![0_i64; 2 * N_SITE];
        for (&site, &spin) in idx.iter().zip(spn) {
            num[site as usize + spin as usize * N_SITE] = 1;
        }
        let mut inv = mvmc_core::state::InvMColMajor::zeros(2, N_ELEC);
        let mut pf = vec![Complex64::default(); 2];
        mvmc_core::pfaffian::calc_m_all_fsz_complex(
            idx,
            spn,
            &self.slater.slater_matrix.slater_elm,
            &mut inv,
            &mut pf,
            0,
            2,
            N_SITE,
            N_ELEC,
            &self.pool,
        )
        .unwrap();
        assert_eq!(idx.len(), n_size);
        let ip = mvmc_core::observables::calculate_ip_complex(&pf, 0, 2, self.data);
        let mut proj =
            vec![0_i64; self.data.n_gutzwiller_idx as usize + self.data.n_jastrow_idx as usize];
        make_proj_cnt(&mut proj, &num, self.data);
        // log of the projection weight from scratch: ratio against the empty counters
        let zero = vec![0_i64; proj.len()];
        let log_proj = mvmc_core::sampling::projection::log_proj_ratio(&proj, &zero, self.data);
        let cfg = RbmConfig::from(self.data);
        let log_rbm = log_rbm_val(&num, &cfg);
        (ip.ln(), log_rbm, log_proj)
    }
}

fn run_replay(with_rbm: bool) -> (usize, usize, usize) {
    let (data, mut state) = model(with_rbm);
    // Starting configuration: the sampler's own initial placement, reproduced from the
    // same seed on a probe state (initial placement is an input here, not the object tested).
    let mut probe = VmcOptimizationState::zeros(N_SITE, N_ELEC, 2, 0, 2, N_SAMPLE, true, true);
    probe.slater_matrix.slater_elm = state.slater_matrix.slater_elm.clone();
    let mut probe_rng = Sfmt19937Rng::new(4403);
    mvmc_core::sampling::initial::make_initial_sample_fsz(
        &data,
        &mut probe,
        &mut probe_rng,
        0,
        2,
        &mvmc_core::state::ThreadedPfaPackWorkspace::new(4, 1),
    )
    .unwrap();
    let mut idx = probe.electron_config.tmp_ele_idx.clone();
    let mut spn = probe.electron_config.tmp_ele_spn.clone();

    let mut rng = Sfmt19937Rng::new(4403);
    trace::start();
    vmc_make_sample_fsz(&data, &mut state, &mut rng);
    let events = trace::finish();

    let wave = Wavefunction {
        data: &data,
        slater: &state,
        pool: mvmc_core::state::ThreadedPfaPackWorkspace::new(4, 1),
    };
    let (mut moves, mut accepted, mut rbm_sensitive) = (0, 0, 0);
    let mut position = 0;
    while position < events.len() {
        let event = &events[position];
        position += 1;
        let kind = event[0];
        if !(3..=6).contains(&kind) {
            continue;
        }
        let reject = *event.last().unwrap() != 0;
        if reject {
            continue;
        }
        let mi = event[1] as usize;
        let mut new_idx = idx.clone();
        let mut new_spn = spn.clone();
        match kind {
            3..=5 => {
                new_idx[mi] = event[3];
                new_spn[mi] = event[5];
            }
            6 => {
                let (ri, rj, s) = (event[2], event[3], event[4]);
                let mj = (0..idx.len())
                    .find(|&m| idx[m] == rj && spn[m] == 1 - s)
                    .expect("exchange partner");
                new_idx[mi] = rj;
                new_idx[mj] = ri;
            }
            _ => unreachable!(),
        }
        let decision = events[position..]
            .iter()
            .find(|e| e[0] == 7)
            .expect("Metropolis event follows an unrejected candidate");
        let (accepted_flag, draw) = (decision[1] != 0, decision[2] as f64 / 4294967296.0);
        let (ip_old, rbm_old, proj_old) = wave.log_psi(&idx, &spn);
        let (ip_new, rbm_new, proj_new) = wave.log_psi(&new_idx, &new_spn);
        let full = (proj_new - proj_old) + (rbm_new - rbm_old).re + (ip_new - ip_old).re;
        let weight = (2.0 * full).exp();
        let without_rbm = (2.0 * ((proj_new - proj_old) + (ip_new - ip_old).re)).exp();
        if (weight - draw).abs() > 1e-9 {
            assert_eq!(
                weight > draw,
                accepted_flag,
                "move {moves}: w={weight}, draw={draw}, kind={kind}"
            );
            if (without_rbm > draw) != (weight > draw) {
                rbm_sensitive += 1;
            }
        }
        moves += 1;
        if accepted_flag {
            accepted += 1;
            idx = new_idx;
            spn = new_spn;
        }
        position = events[position..]
            .iter()
            .position(|e| e[0] == 7)
            .map(|offset| position + offset + 1)
            .unwrap();
    }
    (moves, accepted, rbm_sensitive)
}

#[test]
fn per_move_acceptance_matches_full_recomputation() {
    let (moves, accepted, rbm_sensitive) = run_replay(true);
    assert!(moves >= 100, "only {moves} moves replayed");
    assert!(
        accepted > 10 && accepted < moves,
        "accepted {accepted} of {moves}"
    );
    // The RBM factor must matter: some decisions differ from the RBM-free weight.
    assert!(
        rbm_sensitive >= 3,
        "RBM changed only {rbm_sensitive} decisions"
    );
}

#[test]
fn replay_harness_agrees_without_rbm_terms() {
    let (moves, _, rbm_sensitive) = run_replay(false);
    assert!(moves >= 100);
    assert_eq!(rbm_sensitive, 0);
}

#[test]
fn incremental_hopping_update_matches_full_recomputation() {
    let (data, _) = model(true);
    let cfg = RbmConfig::from(&data);
    let n = 2 * N_SITE;
    let mut rng = Sfmt19937Rng::new(77);
    for _ in 0..200 {
        // random spin-conserving configuration with an empty target site
        let mut num = vec![0_i64; n];
        for slot in num.iter_mut() {
            *slot = i64::from(rng.gen_rand32().is_multiple_of(2));
        }
        let spin = (rng.gen_rand32() % 2) as usize;
        let occupied: Vec<usize> = (0..N_SITE)
            .filter(|&r| num[r + spin * N_SITE] == 1)
            .collect();
        let empty: Vec<usize> = (0..N_SITE)
            .filter(|&r| num[r + spin * N_SITE] == 0)
            .collect();
        if occupied.is_empty() || empty.is_empty() {
            continue;
        }
        let ri = occupied[rng.gen_rand32() as usize % occupied.len()];
        let rj = empty[rng.gen_rand32() as usize % empty.len()];
        let old = make_rbm_cnt(&num, &cfg);
        let mut incremental = old.clone();
        update_rbm_cnt_hopping(
            &mut incremental,
            &old,
            ri as i64,
            rj as i64,
            spin as u8,
            &cfg,
        );
        let inc_ratio = log_rbm_ratio(&incremental, &old, &cfg);
        num[ri + spin * N_SITE] = 0;
        num[rj + spin * N_SITE] = 1;
        let new = make_rbm_cnt(&num, &cfg);
        let full_ratio = log_rbm_ratio(&new, &old, &cfg);
        // Same value, up to the principal-branch bookkeeping of the logarithms: compare the weights.
        let (a, b) = (inc_ratio.exp(), full_ratio.exp());
        assert!((a - b).norm() <= 1e-10 * (1.0 + a.norm()), "{a} vs {b}");
        // and the direct values agree with the counter ratios
        let direct = (log_rbm_val(&num, &cfg)
            - log_rbm_val(
                &{
                    let mut before = num.clone();
                    before[ri + spin * N_SITE] = 1;
                    before[rj + spin * N_SITE] = 0;
                    before
                },
                &cfg,
            ))
        .exp();
        assert!(
            (direct - b).norm() <= 1e-10 * (1.0 + b.norm()),
            "{direct} vs {b}"
        );
    }
}

//! Within-one-sample observable producers. No aggregate or RNG ownership.
use crate::state::VmcOptimizationState;
use mvmc_expert_parsers::ExpertModeData;
use num_complex::Complex64;

/// Serial structural checks precede item dispatch; no physical plane is written.
/// Runners already validate supported model definitions. This boundary checks
/// indexed sample/storage ownership, not a new numerical rejection policy.
/// `projection_layout()` assumes runner-validated projection definitions; these
/// dimension checks do not make arbitrary malformed projection widths safe.
#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_green_sample(
    data: &ExpertModeData,
    state: &VmcOptimizationState,
    idx: &[i64],
    cfg: &[i64],
    num: &[i64],
    cnt: &[i64],
    spins: Option<&[i64]>,
) {
    let ns = usize::try_from(data.modpara.nsite).expect("Green nsite must be nonnegative");
    let ne = usize::try_from(data.modpara.nelec).expect("Green nelec must be nonnegative");
    let sites = ns.checked_mul(2).expect("Green site width overflow");
    let n = ne.checked_mul(2).expect("Green electron width overflow");
    assert!(
        ns > 0 && ne > 0,
        "Green sample requires positive dimensions"
    );
    assert_eq!(idx.len(), n, "Green electron index width");
    assert_eq!(cfg.len(), sites, "Green configuration width");
    assert_eq!(num.len(), sites, "Green occupation width");
    assert_eq!(
        cnt.len(),
        data.projection_layout().n_proj,
        "Green projection width"
    );
    if let Some(spins) = spins {
        assert_eq!(spins.len(), n, "Green spin width");
        assert!(spins.iter().all(|&s| s == 0 || s == 1), "Green spin domain");
    }
    assert!(
        num.iter().all(|&v| v == 0 || v == 1),
        "Green occupation domain"
    );
    for (electron, &site) in idx.iter().enumerate() {
        let site = usize::try_from(site).expect("Green negative electron site");
        assert!(site < ns, "Green electron site bounds");
        let spin = spins.map_or(electron / ne, |s| s[electron] as usize);
        let label = if spins.is_some() {
            electron
        } else {
            electron % ne
        };
        assert_eq!(num[site + spin * ns], 1, "Green occupied electron mapping");
        assert_eq!(
            cfg[site + spin * ns],
            label as i64,
            "Green electron label mapping"
        );
    }
    for (slot, (&label, &occupation)) in cfg.iter().zip(num).enumerate() {
        if occupation == 0 {
            assert_eq!(label, -1, "Green empty slot label");
        } else {
            let label = usize::try_from(label).expect("Green occupied label negative");
            let limit = if spins.is_some() { n } else { ne };
            assert!(label < limit, "Green occupied label bounds");
            if let Some(spins) = spins {
                assert_eq!(
                    spins[label],
                    (slot / ns) as i64,
                    "Green reverse spin mapping"
                );
            }
            assert_eq!(
                idx[if spins.is_some() {
                    label
                } else {
                    label + (slot / ns) * ne
                }],
                (slot % ns) as i64,
                "Green reverse electron mapping"
            );
        }
    }
    for term in &data.green_one_terms {
        assert!(
            [term.site1, term.site2]
                .iter()
                .all(|&site| site >= 0 && (site as usize) < ns),
            "Green one-body site bounds"
        );
    }
    for term in &data.green_two_terms {
        assert!(
            [term.site1, term.site2, term.site3, term.site4]
                .iter()
                .all(|&site| site >= 0 && (site as usize) < ns),
            "Green direct site bounds"
        );
    }
    // Ordinary publication skips cached map slots beyond its destination.
    // In particular, removed TwoBodyGEx terms leave an empty destination even
    // when the caller retains the old, unused canonical map. FSZ below keeps
    // its original full-map width contract.
    let factored_width = state
        .phys_quantities
        .as_ref()
        .expect("Green physical storage must exist")
        .phys_cis_ajs_ckt_alt
        .len();
    for &(first, second) in data.green_two_ex_indices.iter().take(factored_width) {
        assert!(
            first < data.green_one_terms.len() && second < data.green_one_terms.len(),
            "Green factored index bounds"
        );
    }
    for table in &data.doublon_holon_2site_indices {
        assert_eq!(table.neighbors.len(), ns, "Green DH2 site width");
        assert!(
            table
                .neighbors
                .iter()
                .flatten()
                .all(|&site| site >= 0 && (site as usize) < ns),
            "Green DH2 neighbor bounds"
        );
    }
    for table in &data.doublon_holon_4site_indices {
        assert_eq!(table.neighbors.len(), ns, "Green DH4 site width");
        assert!(
            table
                .neighbors
                .iter()
                .flatten()
                .all(|&site| site >= 0 && (site as usize) < ns),
            "Green DH4 neighbor bounds"
        );
    }
    let matrix = &state.slater_matrix;
    let nq = matrix.pf_m.len();
    assert_eq!(
        matrix.slater_elm.n_site2(),
        sites,
        "Green complex Slater side"
    );
    assert_eq!(
        matrix.slater_elm.n_qp_full(),
        nq,
        "Green complex Slater QP width"
    );
    assert_eq!(matrix.inv_m.n_size(), n, "Green complex inverse side");
    assert_eq!(
        matrix.inv_m.n_qp_full(),
        nq,
        "Green complex inverse QP width"
    );
    if spins.is_none() && !matrix.pf_m_real.is_empty() {
        assert_eq!(matrix.pf_m_real.len(), nq, "Green real PF width");
        assert_eq!(
            matrix.slater_elm_real.n_site2(),
            sites,
            "Green real Slater side"
        );
        assert_eq!(
            matrix.slater_elm_real.n_qp_full(),
            nq,
            "Green real Slater QP width"
        );
        assert_eq!(matrix.inv_m_real.n_size(), n, "Green real inverse side");
        assert_eq!(
            matrix.inv_m_real.n_qp_full(),
            nq,
            "Green real inverse QP width"
        );
    }
    let phys = state
        .phys_quantities
        .as_ref()
        .expect("Green physical buffers required");
    assert_eq!(
        phys.local_cis_ajs.len(),
        data.green_one_terms.len(),
        "Green local one width"
    );
    assert_eq!(
        phys.phys_cis_ajs.len(),
        data.green_one_terms.len(),
        "Green aggregate one width"
    );
    assert_eq!(
        phys.local_cis_ajs_ckt_alt_dc.len(),
        data.green_two_terms.len(),
        "Green local direct width"
    );
    assert_eq!(
        phys.phys_cis_ajs_ckt_alt_dc.len(),
        data.green_two_terms.len(),
        "Green aggregate direct width"
    );
    if spins.is_some() {
        assert_eq!(
            phys.phys_cis_ajs_ckt_alt.len(),
            data.green_two_ex_indices.len(),
            "Green factored width"
        );
    }
}

// One output cell has exactly one producer; kernel reductions inside it stay
// serial. No worker subtotal is merged into an aggregate. RBM stays serial.
pub(crate) fn collect_green_values(
    len: usize,
    cost_ns: usize,
    allow_parallel: bool,
    evaluate: impl Fn(usize) -> Complex64 + Send + Sync,
) -> Vec<Complex64> {
    let mut values = vec![Complex64::new(0.0, 0.0); len];
    if allow_parallel {
        crate::threading::for_each_mut(&mut values, cost_ns, |i, value| *value = evaluate(i));
    } else {
        let observed =
            crate::threading::observe_kernel(crate::threading::ObservedWork::Entry, false);
        for (i, value) in values.iter_mut().enumerate() {
            let _item = observed.enter_item();
            *value = evaluate(i);
        }
    }
    values
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ordinary_green_values(
    data: &ExpertModeData,
    state: &VmcOptimizationState,
    ip: Complex64,
    idx: &[i64],
    cfg: &[i64],
    num: &[i64],
    cnt: &[i64],
) -> (Vec<Complex64>, Vec<Complex64>) {
    let parallel = !data.has_rbm_terms();
    if parallel {
        validate_green_sample(data, state, idx, cfg, num, cnt, None);
    }
    let one = collect_green_values(
        data.green_one_terms.len(),
        crate::threading::green_cost_ns(idx.len(), 1),
        parallel,
        |i| {
            let term = &data.green_one_terms[i];
            super::green_func1(
                term.site1 as usize,
                term.site2 as usize,
                super::spin_code(term.spin1),
                super::spin_code(term.spin2),
                ip,
                data,
                state,
                idx,
                cfg,
                num,
                cnt,
            )
        },
    );
    let direct = collect_green_values(
        data.green_two_terms.len(),
        crate::threading::green_cost_ns(idx.len(), 2),
        parallel,
        |i| {
            let term = &data.green_two_terms[i];
            super::green_func2(
                term.site1 as usize,
                term.site2 as usize,
                term.site3 as usize,
                term.site4 as usize,
                super::spin_code(term.spin1),
                super::spin_code(term.spin3),
                ip,
                data,
                state,
                idx,
                cfg,
                num,
                cnt,
            )
        },
    );
    (one, direct)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::PhysicalQuantities;
    use mvmc_expert_parsers::{GreenOneTerm, GreenTwoTerm, Spin};
    use std::process::Command;

    fn diagonal_sample(m: usize, complex: bool) -> (ExpertModeData, VmcOptimizationState) {
        let mut data = ExpertModeData::new();
        data.modpara.nsite = 2;
        data.modpara.nelec = 1;
        for i in 0..m {
            data.green_one_terms.push(GreenOneTerm {
                site1: (i % 2) as i64,
                spin1: Spin::Up,
                site2: (i % 2) as i64,
                spin2: Spin::Up,
            });
            data.green_two_terms.push(GreenTwoTerm {
                site1: 0,
                spin1: Spin::Up,
                site2: 0,
                spin2: Spin::Up,
                site3: 1,
                spin3: Spin::Down,
                site4: 1,
                spin4: Spin::Down,
            });
        }
        if m > 0 {
            data.green_two_ex_indices.push((0, 0));
        }
        let mut state = VmcOptimizationState::zeros(2, 1, 0, 0, 1, 1, complex, true);
        state.phys_quantities = Some(PhysicalQuantities::zeros(m, usize::from(m > 0), m));
        (data, state)
    }

    fn check_observation(snapshot: crate::threading::ExecutionSnapshot, m: usize, parallel: bool) {
        println!("GREEN_INDEX_ITEMS size={m} parallel={parallel} actual={snapshot:?}");
        assert_eq!(
            snapshot.parallel_entry_items,
            if parallel { 2 * m } else { 0 }
        );
        assert_eq!(
            snapshot.serial_entry_items,
            if parallel { 0 } else { 2 * m }
        );
        if parallel && m > 0 {
            assert_eq!(snapshot.worker_entries, 2 * m);
            assert!(snapshot.distinct_workers > 0);
            assert!(snapshot
                .worker_ids
                .iter()
                .all(|&id| id < crate::threading::inner_thread_config().threads));
        } else {
            assert_eq!(snapshot.worker_entries, 0);
        }
    }

    #[test]
    #[ignore = "child process selected by green_index_workers_and_threshold_matrix"]
    fn green_index_child() {
        assert_eq!(std::env::var("MVMC_GREEN_INDEX_CHILD").as_deref(), Ok("1"));
        let workers: usize = std::env::var("MVMC_RS_INNER_THREADS")
            .unwrap()
            .parse()
            .unwrap();
        let configured: usize = std::env::var("MVMC_RS_INNER_THRESHOLD")
            .unwrap()
            .parse()
            .unwrap();
        assert_eq!(crate::threading::inner_thread_config().threads, workers);
        assert_eq!(
            crate::threading::inner_thread_config().threshold,
            if configured == 0 { 32 } else { configured }
        );
        for complex in [false, true] {
            for m in [0, 1, 31, 32, 33] {
                let (data, mut state) = diagonal_sample(m, complex);
                let before = state.slater_matrix.clone();
                let idx = [0, 1];
                let num = [1, 0, 0, 1];
                let cfg = [0, -1, -1, 0];
                let observer = crate::threading::start_observation();
                let (one, direct) = ordinary_green_values(
                    &data,
                    &state,
                    Complex64::new(1.0, 0.0),
                    &idx,
                    &cfg,
                    &num,
                    &[],
                );
                let snapshot = observer.finish();
                // Threshold 0 is invalid and selects the automatic work gate, under which
                // these two-electron cells are far too cheap to pool (#361).
                let parallel = crate::threading::inner_parallel_work(
                    m,
                    crate::threading::green_cost_ns(idx.len(), 1),
                );
                check_observation(snapshot, m, parallel);
                // Independent occupation identity <n_0up>=1, <n_1up>=0;
                // <n_0up*n_1down>=1. Not Rust-generated expected data.
                for (i, actual) in one.iter().enumerate() {
                    assert_eq!(
                        *actual,
                        Complex64::new(if i.is_multiple_of(2) { 1.0 } else { 0.0 }, 0.0)
                    );
                }
                assert!(direct.iter().all(|&z| z == Complex64::new(1.0, 0.0)));
                assert_eq!(state.slater_matrix, before);
                let phys = state.phys_quantities.as_mut().unwrap();
                phys.phys_cis_ajs.fill(Complex64::new(7.0, 0.0));
                phys.phys_cis_ajs_ckt_alt_dc.fill(Complex64::new(9.0, 0.0));
                phys.phys_cis_ajs_ckt_alt.fill(Complex64::new(11.0, 0.0));
                phys.phys_lanczos_qqqq.fill(Complex64::new(13.0, 2.0));
                // Native FSZ labels address full electron indices.
                let cfg = [0, -1, -1, 1];
                for weight in [0.5, 1.0] {
                    let observer = crate::threading::start_observation();
                    super::super::calculate_green_func_fsz(
                        &data,
                        &mut state,
                        weight,
                        Complex64::new(1.0, 0.0),
                        &idx,
                        &cfg,
                        &num,
                        &[],
                        &[0, 1],
                    );
                    check_observation(observer.finish(), m, parallel);
                }
                let phys = state.phys_quantities.as_ref().unwrap();
                for (i, actual) in phys.phys_cis_ajs.iter().enumerate() {
                    assert_eq!(
                        *actual,
                        Complex64::new(if i.is_multiple_of(2) { 8.5 } else { 7.0 }, 0.0)
                    );
                }
                assert!(phys
                    .phys_cis_ajs_ckt_alt_dc
                    .iter()
                    .all(|&z| z == Complex64::new(10.5, 0.0)));
                assert!(phys
                    .phys_cis_ajs_ckt_alt
                    .iter()
                    .all(|&z| z == Complex64::new(12.5, 0.0)));
                assert!(phys
                    .phys_lanczos_qqqq
                    .iter()
                    .all(|&z| z == Complex64::new(13.0, 2.0)));
                assert_eq!(state.slater_matrix, before);
                assert_eq!(idx, [0, 1]);
                assert_eq!(num, [1, 0, 0, 1]);
            }
        }
        // Explicit producer serial eligibility (RBM route policy), independent
        // of configured workers; does not claim a native RBM model comparison.
        let observer = crate::threading::start_observation();
        assert_eq!(
            collect_green_values(33, 1, false, |i| Complex64::new(i as f64, 0.0)).len(),
            33
        );
        let snapshot = observer.finish();
        assert_eq!(snapshot.serial_entry_items, 33);
        assert_eq!(snapshot.parallel_entry_items, 0);
        println!("GREEN_INDEX_CHILD_PASS workers={workers} threshold={configured}");
    }

    #[test]
    fn green_index_workers_and_threshold_matrix() {
        for workers in [1, 2, 4] {
            for threshold in [0, 1, 31, 32, 33] {
                let output = Command::new(std::env::current_exe().unwrap())
                    .args([
                        "--ignored",
                        "--exact",
                        "observables::green_measurements::tests::green_index_child",
                        "--nocapture",
                        "--test-threads=1",
                    ])
                    .env("MVMC_GREEN_INDEX_CHILD", "1")
                    .env("MVMC_RS_INNER_THREADS", workers.to_string())
                    .env("MVMC_RS_INNER_THRESHOLD", threshold.to_string())
                    .env("OPENBLAS_NUM_THREADS", "1")
                    .env("OMP_NUM_THREADS", "1")
                    .env("MKL_NUM_THREADS", "1")
                    .env("BLIS_NUM_THREADS", "1")
                    .output()
                    .unwrap();
                assert!(
                    output.status.success(),
                    "workers={workers} threshold={threshold}:\n{}\n{}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
                assert!(String::from_utf8_lossy(&output.stdout).contains("GREEN_INDEX_CHILD_PASS"));
                assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed; 0 failed"));
                println!("{}", String::from_utf8_lossy(&output.stdout));
            }
        }
    }

    #[test]
    fn malformed_sample_fails_serially_without_new_publication_or_worker_items() {
        for failure in 0..7 {
            let (mut data, mut state) = diagonal_sample(33, true);
            let mut cfg = [0, -1, -1, 1];
            let mut spins = [0, 1];
            let mut num = [1, 0, 0, 1];
            match failure {
                0 => data.green_two_terms[32].site4 = 2,
                1 => cfg[3] = 0,
                2 => spins[1] = 2,
                3 => state
                    .phys_quantities
                    .as_mut()
                    .unwrap()
                    .phys_cis_ajs
                    .pop()
                    .map(|_| ())
                    .unwrap(),
                4 => data.green_two_ex_indices[0] = (0, 33),
                5 => state.slater_matrix.inv_m = crate::state::InvMColMajor::zeros(1, 2),
                _ => {
                    num[2] = 1;
                    cfg[2] = 0;
                }
            }
            // Model existing early QQQQ publication; invalid later Green must
            // neither roll it back nor partially publish another Green plane.
            state.phys_quantities.as_mut().unwrap().phys_lanczos_qqqq[0] = Complex64::new(8.0, 1.0);
            let before = state.phys_quantities.clone();
            let observer = crate::threading::start_observation();
            let error = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                super::super::calculate_green_func_fsz(
                    &data,
                    &mut state,
                    1.0,
                    Complex64::new(1.0, 0.0),
                    &[0, 1],
                    &cfg,
                    &num,
                    &[],
                    &spins,
                );
            }));
            assert!(error.is_err());
            let payload = error.unwrap_err();
            let message = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .expect("string validation panic");
            let expected = [
                "Green direct site bounds",
                "Green electron label mapping",
                "Green spin domain",
                "left == right",
                "Green factored index bounds",
                "Green complex inverse side",
                "Green reverse spin mapping",
            ];
            assert!(
                message.contains(expected[failure]),
                "unexpected validation cause: {message}"
            );
            assert_eq!(state.phys_quantities, before);
            let snapshot = observer.finish();
            assert_eq!(
                snapshot.parallel_entry_items + snapshot.serial_entry_items,
                0
            );
        }
    }

    #[test]
    fn ordinary_factored_validation_ignores_only_unused_map_slots() {
        for width in [0, 1] {
            let (mut data, mut state) = diagonal_sample(33, false);
            // Match state_from_data's empty destination after terms are removed;
            // also cover one active slot with an unused out-of-range tail.
            assert!(data.green_two_ex_terms.is_empty());
            data.green_two_ex_indices = if width == 0 {
                vec![(usize::MAX, usize::MAX)]
            } else {
                vec![(0, 0), (usize::MAX, usize::MAX)]
            };
            state.phys_quantities = Some(PhysicalQuantities::zeros(33, width, 33));
            state
                .phys_quantities
                .as_mut()
                .unwrap()
                .phys_cis_ajs_ckt_alt
                .fill(Complex64::new(11.0, 0.0));
            let map_before = data.green_two_ex_indices.clone();
            let matrix_before = state.slater_matrix.clone();
            let phys_before = state.phys_quantities.clone();
            let observer = crate::threading::start_observation();
            let (one, direct) = ordinary_green_values(
                &data,
                &state,
                Complex64::new(1.0, 0.0),
                &[0, 1],
                &[0, -1, -1, 0],
                &[1, 0, 0, 1],
                &[],
            );
            check_observation(
                observer.finish(),
                33,
                crate::threading::inner_parallel_work(33, crate::threading::green_cost_ns(2, 1)),
            );
            for (i, value) in one.iter().enumerate() {
                assert_eq!(
                    *value,
                    Complex64::new(if i.is_multiple_of(2) { 1.0 } else { 0.0 }, 0.0)
                );
            }
            assert!(direct.iter().all(|&z| z == Complex64::new(1.0, 0.0)));
            assert_eq!(state.slater_matrix, matrix_before);
            assert_eq!(state.phys_quantities, phys_before);
            assert_eq!(data.green_two_ex_indices, map_before);
            let destination = &mut state.phys_quantities.as_mut().unwrap().phys_cis_ajs_ckt_alt;
            super::super::accumulate_two_body_gex_sample(
                destination,
                &one,
                &data.green_two_ex_indices,
                Complex64::new(0.5, -0.25),
            );
            assert_eq!(destination.len(), width);
            if width == 1 {
                assert_eq!(destination[0], Complex64::new(11.5, -0.25));
            }
        }
        // Active indices remain rejected before producers; this is not a
        // relaxation for a malformed slot that would actually be consumed.
        let (mut data, state) = diagonal_sample(1, false);
        data.green_two_ex_indices[0] = (usize::MAX, 0);
        let observer = crate::threading::start_observation();
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ordinary_green_values(
                &data,
                &state,
                Complex64::new(1.0, 0.0),
                &[0, 1],
                &[0, -1, -1, 0],
                &[1, 0, 0, 1],
                &[],
            );
        }))
        .is_err());
        let actual = observer.finish();
        assert_eq!(actual.parallel_entry_items + actual.serial_entry_items, 0);
    }

    #[test]
    fn producer_panic_keeps_output_unpublished_and_restores_observer_scope() {
        let (data, state) = diagonal_sample(33, true);
        let before = state.phys_quantities.clone();
        let observer = crate::threading::start_observation();
        let failure = std::panic::catch_unwind(|| {
            collect_green_values(33, 1, true, |i| {
                if i == 1 {
                    panic!("synthetic producer failure");
                }
                Complex64::new(i as f64, 0.0)
            })
        });
        assert!(failure.is_err());
        assert_eq!(state.phys_quantities, before);
        let attempted = observer.finish();
        assert!(attempted.parallel_entry_items + attempted.serial_entry_items > 0);
        let after = crate::threading::start_observation();
        let _ = ordinary_green_values(
            &data,
            &state,
            Complex64::new(1.0, 0.0),
            &[0, 1],
            &[0, -1, -1, 0],
            &[1, 0, 0, 1],
            &[],
        );
        let observed = after.finish();
        assert_eq!(
            observed.parallel_entry_items + observed.serial_entry_items,
            66
        );
    }
}

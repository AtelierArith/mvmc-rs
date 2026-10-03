//! S196/M0998 accepted singleton OptTrans payload through actual grouped PhysCal.
#![cfg(feature = "mpi")]

use mvmc_core::{mpi::MpiContext, Reducer};
use num_complex::Complex64;
use sfmt19937::Sfmt19937Rng;
use std::{fs, path::PathBuf};

fn next624(rng: &Sfmt19937Rng) -> Vec<u32> {
    let mut peek = rng.clone();
    (0..624).map(|_| peek.gen_rand32()).collect()
}

fn close(a: f64, b: f64) -> bool {
    a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-12 + 1e-12 * a.abs().max(b.abs())
}

fn output_agrees(first: &std::path::Path, second: &std::path::Path, indices: usize) -> bool {
    let (Ok(a), Ok(b)) = (fs::read_to_string(first), fs::read_to_string(second)) else {
        return false;
    };
    text_agrees(&a, &b, indices)
}

fn text_agrees(a: &str, b: &str, indices: usize) -> bool {
    let a: Vec<_> = a.lines().collect();
    let b: Vec<_> = b.lines().collect();
    a.iter().any(|line| !line.trim().is_empty())
        && a.len() == b.len()
        && a.iter().zip(b).all(|(a, b)| {
            let a: Vec<_> = a.split_whitespace().collect();
            let b: Vec<_> = b.split_whitespace().collect();
            a.len() == b.len()
                && (a.is_empty()
                    || (a.len() > indices
                        && a[..indices] == b[..indices]
                        && a[indices..].iter().zip(&b[indices..]).all(|(a, b)| {
                            match (a.parse::<f64>(), b.parse::<f64>()) {
                                (Ok(a), Ok(b)) => close(a, b),
                                _ => false,
                            }
                        })))
        })
}

#[test]
fn singleton_output_checker_preserves_layout_and_rejects_bad_values() {
    assert!(text_agrees("0 0 1.0 0.0\n\n", "0 0 1.0 0.0\n\n", 2));
    assert!(!text_agrees("0 0 1.0 0.0\n\n", "0 0 1.0 0.0\n", 2));
    assert!(!text_agrees("\n", "\n", 0));
    assert!(!text_agrees("0 0 1.0 0.0\n", "0 1 1.0 0.0\n", 2));
    assert!(!text_agrees("1.0\n", "1.000001\n", 0));
    assert!(!text_agrees("NaN\n", "NaN\n", 0));
}

#[test]
#[ignore = "requires isolated mpiexec -n 2 or -n 4 and MPI179_SINGLETON_OUTPUT"]
fn issue179_grouped_physcal_singleton_is_accepted_and_repeatable() {
    let world = MpiContext::initialize().unwrap();
    assert!(matches!(world.world_size(), 2 | 4));
    let group = world.split_groups(2).unwrap();
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/physcal_181/heisenberg_chain_real");
    let output = PathBuf::from(std::env::var("MPI179_SINGLETON_OUTPUT").unwrap());
    let mut baseline = None;
    let mut numerical_baseline: Option<Vec<f64>> = None;
    for repeat in 1..=2 {
        let mut preparation = mvmc_core::prepare_phys_cal_from_namelist_with_reducer(
            fixture.join("inputs/namelist.def"),
            fixture.join("zqp_opt.dat"),
            "real",
            Some(1),
            &group,
        )
        .unwrap();
        // Julia's validator has synthetic [[0]] for one-site metadata. This
        // six-site runtime fixture completes the same single identity sector.
        let data = &mut preparation.data;
        data.modpara.nsplit_size = 2;
        data.modpara.lanczos_mode = 0;
        data.i_flg_orbital_general = 0;
        data.n_qp_opt_trans = 1;
        data.opt_trans = vec![Complex64::new(1.0, 0.0)];
        data.qp_opt_trans = vec![(0..data.modpara.nsite).collect()];
        data.modpara.nsp_gauss_leg = 1;
        data.modpara.nmp_trans = 1;
        data.modpara.nvmc_sample = 3;
        data.modpara.nvmc_warmup = 1;
        data.modpara.n_data_qty_smp = 1;
        assert!(data.inter_all_terms.is_empty());
        let seeded = Sfmt19937Rng::new((1 + group.seed_offset()) as u32);
        assert_eq!(preparation.rng.words_consumed(), 0);
        assert_eq!(next624(&preparation.rng), next624(&seeded));
        let dir = output.join(format!("repeat{repeat}"));
        let observation = mvmc_core::threading::start_observation();
        let result = mvmc_core::vmc_phys_cal_with_reducer(preparation, Some(&dir), &group).unwrap();
        let execution = observation.finish();
        let requested_workers = std::env::var("MVMC_RS_INNER_THREADS")
            .unwrap()
            .parse::<usize>()
            .unwrap();
        assert!(matches!(requested_workers, 1 | 2 | 4));
        if execution.parallel_qp_items + execution.parallel_term_items > 0 {
            assert!(requested_workers > 1 && execution.parallel_calls > 0);
            assert!(execution.worker_entries > 0 && execution.distinct_workers > 0);
        }
        assert!(execution
            .worker_ids
            .iter()
            .all(|&id| id < requested_workers));
        println!(
            "S196_WORKERS rank={} repeat={repeat} actual={execution:?}",
            world.rank()
        );
        assert_eq!(result.iterations, 1);
        assert_eq!(result.data.n_qp_opt_trans, 1);
        assert_eq!(result.data.opt_trans, [Complex64::new(1.0, 0.0)]);
        assert_eq!(
            result.state.energy.wc,
            Complex64::new((world.world_size() / 2 * 3) as f64, 0.0)
        );
        assert!(result.state.energy.etot.re.is_finite());
        assert!(result.state.energy.etot2.re.is_finite());
        let phys = result.state.phys_quantities.as_ref().unwrap();
        assert_eq!(phys.phys_cis_ajs.len(), 2);
        assert!(phys
            .phys_cis_ajs
            .iter()
            .all(|z| z.re.is_finite() && z.im.is_finite()));
        // Independent local-spin occupancy identity: n(0,up)+n(0,down)=1.
        let density = phys.phys_cis_ajs[0] + phys.phys_cis_ajs[1];
        assert!((density.re - 1.0).abs() <= 1e-12 && density.im.abs() <= 1e-12);
        let actual = (
            result.state.electron_config.ele_idx.clone(),
            result.state.electron_config.ele_cfg.clone(),
            result.state.electron_config.counter.clone(),
            result.final_rng.words_consumed(),
            next624(&result.final_rng),
        );
        if let Some(previous) = &baseline {
            assert_eq!(&actual, previous);
        }
        baseline = Some(actual);
        let energy = &result.state.energy;
        let numerical: Vec<_> = [
            energy.wc,
            energy.etot,
            energy.etot2,
            energy.sztot,
            energy.sztot2,
        ]
        .into_iter()
        .chain(phys.phys_cis_ajs.iter().copied())
        .chain(phys.phys_cis_ajs_ckt_alt.iter().copied())
        .chain(phys.phys_cis_ajs_ckt_alt_dc.iter().copied())
        .flat_map(|z| [z.re, z.im])
        .collect();
        if let Some(previous) = &numerical_baseline {
            assert_eq!(numerical.len(), previous.len());
            assert!(numerical.iter().zip(previous).all(|(&a, &b)| close(a, b)));
        }
        numerical_baseline = Some(numerical);
        world.barrier();
        let mut output_failed = false;
        if world.rank() == 0 {
            for (name, indices) in [
                ("zvo_out_001.dat", 0),
                ("zvo_var_001.dat", 0),
                ("zvo_cisajs_001.dat", 4),
                ("zvo_cisajscktalt_001.dat", 8),
                ("zvo_cisajscktaltex_001.dat", 0),
            ] {
                let file = dir.join(name);
                let previous = if repeat == 2 {
                    output.join("repeat1").join(name)
                } else {
                    file.clone()
                };
                output_failed |= !output_agrees(&file, &previous, indices);
            }
        }
        assert!(
            !world.any_failure(output_failed),
            "indexed PhysCal output check failed"
        );
        println!("S196_SINGLETON rank={} world={} width=2 repeat={repeat} seed={} workers={} density={density:?} energy={:?} words={}", world.rank(), world.world_size(), 1 + group.seed_offset(), std::env::var("MVMC_RS_INNER_THREADS").unwrap_or_else(|_| "default".into()), result.state.energy.etot, result.final_rng.words_consumed());
    }
}

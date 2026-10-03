//! Optional actual-MPI callback boundary checks, independent of reference runtimes.
//! Launch this binary with 2 or 4 ranks, --ignored --test-threads=1, and a new
//! shared directory in MPI_PHYSCAL_CALLBACK_OUTPUT. These are same-implementation
//! observation/error-boundary contracts, not independent C numerical parity.
#![cfg(feature = "mpi")]

use std::fs;
use std::path::{Path, PathBuf};

use mvmc_core::{ExpertModeData, PhysCalPreparation, PhysCalResult, Reducer, VmcOptimizationState};
use num_complex::Complex64;
use sfmt19937::Sfmt19937Rng;

fn fixed_bits(data: &ExpertModeData) -> Vec<(u64, u64)> {
    data.projection_parameters()
        .into_iter()
        .chain(data.rbm_parameters())
        .chain(data.slater_params.iter().copied())
        .chain(data.opt_trans.iter().copied())
        // Loaded coefficient copying preserves signed zero; computed values below
        // use numerical tolerances, including same-implementation MPI reductions.
        .map(|z| (z.re.to_bits(), z.im.to_bits()))
        .collect()
}

fn qp_buffers(data: &ExpertModeData) -> Option<[&[Complex64]; 7]> {
    data.qp_weights.as_ref().map(|q| {
        [
            q.qp_full_weight.as_slice(),
            q.qp_fix_weight.as_slice(),
            q.spgl_cos.as_slice(),
            q.spgl_sin.as_slice(),
            q.spgl_cos_sin.as_slice(),
            q.spgl_cos_cos.as_slice(),
            q.spgl_sin_sin.as_slice(),
        ]
    })
}

type QpBits = Vec<Vec<(u64, u64)>>;

fn qp_bits(data: &ExpertModeData) -> Option<QpBits> {
    qp_buffers(data).map(|buffers| {
        buffers
            .into_iter()
            .map(|buffer| {
                buffer
                    .iter()
                    .map(|z| (z.re.to_bits(), z.im.to_bits()))
                    .collect()
            })
            .collect()
    })
}

fn qp_agrees(a: &ExpertModeData, b: &ExpertModeData) -> bool {
    // Different preparations independently compute these seven buffers. Compare
    // their lengths exactly and values using the existing numerical repeat budget.
    match (qp_buffers(a), qp_buffers(b)) {
        (Some(a), Some(b)) => a.into_iter().zip(b).all(|(a, b)| values_close(a, b)),
        _ => false,
    }
}

fn observe_qp_copy(errors: &mut Vec<String>, data: &ExpertModeData, first: &mut Option<QpBits>) {
    // Only a snapshot from this execution establishes the immutable-copy
    // contract. Preserve all seven buffer boundaries and signed-zero bits.
    let current = qp_bits(data);
    check(
        errors,
        current
            .as_ref()
            .is_some_and(|buffers| buffers.iter().all(|buffer| !buffer.is_empty())),
        "callback initialized QP buffers",
    );
    if first.is_some() {
        check(errors, current == *first, "same-run callback QP copies");
    } else {
        *first = current;
    }
}

fn close(a: f64, b: f64) -> bool {
    // Same inputs/control path and repeated elementary/BLAS/MPI arithmetic:
    // the existing scoped MPI PhysCal repeat budget, not a C oracle tolerance.
    a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-12 + 1e-12 * a.abs().max(b.abs())
}

fn complex_close(a: Complex64, b: Complex64) -> bool {
    close(a.re, b.re) && close(a.im, b.im)
}

fn values_close(a: &[Complex64], b: &[Complex64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(&a, &b)| complex_close(a, b))
}

fn next624(rng: &Sfmt19937Rng) -> Vec<u32> {
    let mut copy = rng.clone();
    (0..624).map(|_| copy.gen_rand32()).collect()
}

fn numerical(state: &VmcOptimizationState) -> Vec<Complex64> {
    let e = state.energy;
    let mut values = vec![e.wc, e.etot, e.etot2, e.sztot, e.sztot2];
    if let Some(p) = &state.phys_quantities {
        for buffer in [
            &p.phys_cis_ajs,
            &p.phys_cis_ajs_ckt_alt,
            &p.phys_cis_ajs_ckt_alt_dc,
            &p.phys_lanczos_qqqq,
            &p.phys_lanczos_qcisajsq,
            &p.phys_lanczos_qcisajscktaltq,
            &p.phys_lanczos_qcisajscktaltq_dc,
        ] {
            values.extend(buffer);
        }
    }
    values
}

fn check(errors: &mut Vec<String>, ok: bool, message: &str) {
    if !ok {
        errors.push(message.into());
    }
}

fn agree(reducer: &dyn Reducer, errors: &[String], phase: &str) {
    // Never assert a rank-local observation while peers may enter another
    // production collective. All ranks finish this agreement before panicking.
    let failed = reducer.any_failure(!errors.is_empty());
    assert!(!failed, "{phase}: rank {}: {errors:?}", reducer.rank());
}

fn agreed<T>(reducer: &dyn Reducer, result: Result<T, String>, phase: &str) -> T {
    let errors: Vec<_> = result.as_ref().err().cloned().into_iter().collect();
    agree(reducer, &errors, phase);
    result.unwrap()
}

fn prepared(root: &Path, reducer: &dyn Reducer, width: i64, samples: i64) -> PhysCalPreparation {
    let mut prepared = agreed(
        reducer,
        mvmc_core::prepare_phys_cal_from_namelist_with_reducer(
            root.join("inputs/namelist.def"),
            root.join("zqp_opt.dat"),
            "real",
            Some(1),
            reducer,
        ),
        "preparation",
    );
    prepared.data.modpara.nsplit_size = width;
    prepared.data.modpara.nvmc_sample = 3;
    prepared.data.modpara.nvmc_warmup = 1;
    prepared.data.modpara.n_data_qty_smp = samples;
    let mut errors = Vec::new();
    let expected = Sfmt19937Rng::new((1 + reducer.seed_offset()) as u32);
    check(
        &mut errors,
        prepared.rng.words_consumed() == 0,
        "initial draw count",
    );
    check(
        &mut errors,
        prepared.rng.state_snapshot() == expected.state_snapshot(),
        "initial raw RNG",
    );
    check(
        &mut errors,
        next624(&prepared.rng) == next624(&expected),
        "initial next624",
    );
    check(
        &mut errors,
        prepared.data.i_flg_orbital_general == 0 && prepared.data.modpara.lanczos_mode == 0,
        "normal PhysCal input",
    );
    check(
        &mut errors,
        prepared.data.modpara.nsp_gauss_leg == 8 && prepared.data.modpara.nmp_trans == -1,
        "nontrivial fixture projection",
    );
    agree(reducer, &errors, "prepared contracts");
    prepared
}

fn inventory(dir: &Path, samples: usize) -> Result<Vec<(String, String)>, String> {
    let mut actual = fs::read_dir(dir)
        .map_err(|e| e.to_string())?
        .map(|entry| {
            let entry = entry.map_err(|e| e.to_string())?;
            let name = entry
                .file_name()
                .into_string()
                .map_err(|_| "non-UTF8 filename")?;
            let body = fs::read_to_string(entry.path()).map_err(|e| e.to_string())?;
            if body.trim().is_empty() {
                return Err(format!("empty {name}"));
            }
            Ok((name, body))
        })
        .collect::<Result<Vec<_>, String>>()?;
    actual.sort_by(|a, b| a.0.cmp(&b.0));
    let mut expected = Vec::new();
    for sample in 1..=samples {
        for stem in ["out", "var", "cisajs", "cisajscktalt", "cisajscktaltex"] {
            expected.push(format!("zvo_{stem}_{sample:03}.dat"));
        }
    }
    expected.sort();
    if actual.iter().map(|row| &row.0).ne(expected.iter()) {
        return Err(format!(
            "unexpected inventory: {:?}",
            actual.iter().map(|r| &r.0).collect::<Vec<_>>()
        ));
    }
    Ok(actual)
}

fn output_agrees(a: &Path, b: &Path, samples: usize) -> bool {
    let (Ok(a), Ok(b)) = (inventory(a, samples), inventory(b, samples)) else {
        return false;
    };
    a.iter()
        .zip(&b)
        .all(|((name, a), (other_name, b))| name == other_name && output_body_agrees(name, a, b))
}

fn output_body_agrees(name: &str, a: &str, b: &str) -> bool {
    let indices = if name.starts_with("zvo_cisajscktalt_") {
        8
    } else if name.starts_with("zvo_cisajs_") {
        4
    } else {
        0
    };
    !a.trim().is_empty() && a.lines().count() == b.lines().count()
            && a.lines().zip(b.lines()).all(|(a, b)| {
                let a: Vec<_> = a.split_whitespace().collect();
                let b: Vec<_> = b.split_whitespace().collect();
                a.len() == b.len() && (a.is_empty() || (a.len() > indices
                    && a[..indices] == b[..indices]
                    && a[indices..].iter().zip(&b[indices..]).all(|(a, b)| {
                        matches!((a.parse::<f64>(), b.parse::<f64>()), (Ok(a), Ok(b)) if close(a, b))
                    })))
            })
}

fn retained_agrees(
    errors: &mut Vec<String>,
    baseline: &PhysCalResult,
    data: &ExpertModeData,
    state: &VmcOptimizationState,
    rng: &Sfmt19937Rng,
) {
    check(
        errors,
        fixed_bits(data) == fixed_bits(&baseline.data),
        "retained fixed coefficient bits",
    );
    check(
        errors,
        data.optimization_flags == baseline.data.optimization_flags
            && data.complex_flags == baseline.data.complex_flags,
        "retained flags",
    );
    check(
        errors,
        qp_agrees(data, &baseline.data),
        "retained QP numerical repeat",
    );
    // Includes saved, scratch, burn-in configurations and all ten counters.
    check(
        errors,
        state.electron_config == baseline.state.electron_config,
        "retained full configurations/counters",
    );
    check(
        errors,
        rng.words_consumed() == baseline.final_rng.words_consumed(),
        "retained primitive draw count",
    );
    check(
        errors,
        rng.state_snapshot() == baseline.final_rng.state_snapshot(),
        "retained raw 624 words/index",
    );
    check(
        errors,
        next624(rng) == next624(&baseline.final_rng),
        "retained next624",
    );
    check(
        errors,
        values_close(&numerical(state), &numerical(&baseline.state)),
        "retained averaged energy/Green buffers",
    );
}

#[test]
#[ignore = "actual MPI2/4 contract; requires MPI_PHYSCAL_CALLBACK_OUTPUT=new-shared-directory"]
fn mpi_physcal_callback_preserves_two_samples_and_collective_error_boundaries() {
    let world = mvmc_core::mpi::MpiContext::initialize().expect("MPI initialization");
    let mut errors = Vec::new();
    check(
        &mut errors,
        matches!(world.world_size(), 2 | 4),
        "actual MPI2/4 world required",
    );
    agree(&world, &errors, "world");
    let output = agreed(
        &world,
        std::env::var("MPI_PHYSCAL_CALLBACK_OUTPUT")
            .map(PathBuf::from)
            .map_err(|e| e.to_string()),
        "exclusive output environment",
    );
    let setup = if world.is_root() {
        fs::create_dir(&output).map_err(|e| e.to_string())
    } else {
        Ok(())
    };
    agreed(&world, setup, "new exclusive output root");
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/physcal_181/heisenberg_chain_real");
    let group = world.split_groups(2).expect("group split");
    for (width, reducer) in [(1, &world as &dyn Reducer), (2, &group as &dyn Reducer)] {
        // Nonroot paths must remain absent, proving that only the output root writes.
        let directory = |case: &str| {
            output.join(format!(
                "width{width}-{case}-writer{}",
                if reducer.is_output_root() {
                    0
                } else {
                    world.rank()
                }
            ))
        };
        let mut baselines = Vec::new();
        for samples in 1..=2 {
            let dir = directory(&format!("off-{samples}"));
            let result = agreed(
                reducer,
                mvmc_core::vmc_phys_cal_with_reducer(
                    prepared(&root, reducer, width, samples),
                    Some(&dir),
                    reducer,
                ),
                "callback off",
            );
            let mut errors = Vec::new();
            check(
                &mut errors,
                result.iterations == samples as usize,
                "completed sample count",
            );
            if reducer.is_output_root() {
                check(
                    &mut errors,
                    inventory(&dir, samples as usize).is_ok(),
                    "baseline root inventory",
                );
            } else {
                check(&mut errors, !dir.exists(), "baseline peer wrote output");
            }
            agree(reducer, &errors, "baseline output");
            baselines.push(result);
        }
        let baseline = &baselines[1];
        let mut preparation = prepared(&root, reducer, width, 2);
        let fixed = fixed_bits(&preparation.data);
        let flags = (
            preparation.data.optimization_flags.clone(),
            preparation.data.complex_flags.clone(),
        );
        let dir = directory("on");
        let mut calls = Vec::new();
        let mut callback_errors = Vec::new();
        let mut first_qp = None;
        let mut callback =
            |sample: usize, data: &ExpertModeData, energy: Complex64, status: i32| {
                calls.push(sample);
                check(&mut callback_errors, status == 0, "callback status");
                check(
                    &mut callback_errors,
                    fixed_bits(data) == fixed,
                    "callback fixed signed bits",
                );
                check(
                    &mut callback_errors,
                    data.optimization_flags == flags.0 && data.complex_flags == flags.1,
                    "callback flags",
                );
                check(
                    &mut callback_errors,
                    qp_agrees(data, &baseline.data),
                    "callback QP numerical repeat",
                );
                observe_qp_copy(&mut callback_errors, data, &mut first_qp);
                check(
                    &mut callback_errors,
                    baselines
                        .get(sample)
                        .is_some_and(|b| complex_close(energy, b.state.energy.etot)),
                    "callback post-average energy",
                );
                if reducer.is_output_root() {
                    check(
                        &mut callback_errors,
                        inventory(&dir, sample + 1).is_ok(),
                        "callback after complete output/no later sample",
                    );
                } else {
                    check(
                        &mut callback_errors,
                        !dir.exists(),
                        "callback peer output absent",
                    );
                }
                Ok(())
            };
        let observed = agreed(
            reducer,
            mvmc_core::vmc_phys_cal_with_reducer_and_callback(
                preparation,
                Some(&dir),
                reducer,
                Some(&mut callback),
            ),
            "callback on",
        );
        check(
            &mut callback_errors,
            calls == [0, 1],
            "one callback per sample on every rank",
        );
        retained_agrees(
            &mut callback_errors,
            baseline,
            &observed.data,
            &observed.state,
            &observed.final_rng,
        );
        check(
            &mut callback_errors,
            first_qp.is_some() && qp_bits(&observed.data) == first_qp,
            "same-run final QP copies",
        );
        check(
            &mut callback_errors,
            observed.iterations == 2,
            "callback iterations",
        );
        if reducer.is_output_root() {
            check(
                &mut callback_errors,
                output_agrees(&dir, &directory("off-2"), 2),
                "callback output repeat",
            );
        }
        agree(reducer, &callback_errors, "callback observations");

        for failed_sample in 0..=1 {
            for absent_peer in [false, true] {
                preparation = prepared(&root, reducer, width, 3);
                let dir = directory(&format!("error-{failed_sample}-absent-{absent_peer}"));
                let mut state = VmcOptimizationState::zeros(0, 0, 0, 0, 0, 0, false, false);
                let last_peer = world.rank() + 1 == world.world_size();
                let absent = absent_peer && world.is_root();
                let mut calls = Vec::new();
                let mut callback_errors = Vec::new();
                let mut first_qp = None;
                let mut callback =
                    |sample: usize, data: &ExpertModeData, energy: Complex64, status: i32| {
                        calls.push(sample);
                        check(&mut callback_errors, status == 0, "error callback status");
                        check(
                            &mut callback_errors,
                            fixed_bits(data) == fixed,
                            "error callback fixed bits",
                        );
                        check(
                            &mut callback_errors,
                            qp_agrees(data, &baseline.data),
                            "error callback QP numerical repeat",
                        );
                        observe_qp_copy(&mut callback_errors, data, &mut first_qp);
                        check(
                            &mut callback_errors,
                            baselines
                                .get(sample)
                                .is_some_and(|b| complex_close(energy, b.state.energy.etot)),
                            "error callback averaged energy",
                        );
                        if reducer.is_output_root() {
                            check(
                                &mut callback_errors,
                                inventory(&dir, sample + 1).is_ok(),
                                "error callback after output/no later sample",
                            );
                        }
                        if last_peer && sample == failed_sample {
                            Err(format!("last-peer callback sample {sample}"))
                        } else {
                            Ok(())
                        }
                    };
                let result = mvmc_core::vmc_phys_cal_in_place(
                    &mut preparation.data,
                    &mut state,
                    &mut preparation.rng,
                    Some(&dir),
                    reducer,
                    if absent { None } else { Some(&mut callback) },
                );
                let expected_error = if last_peer {
                    format!("last-peer callback sample {failed_sample}")
                } else {
                    "PhysCal callback failed on another rank".into()
                };
                check(
                    &mut callback_errors,
                    result.as_ref().err() == Some(&expected_error),
                    "collective callback error on every rank",
                );
                let expected_calls: Vec<_> = if absent {
                    Vec::new()
                } else {
                    (0..=failed_sample).collect()
                };
                check(
                    &mut callback_errors,
                    calls == expected_calls,
                    "callback count before collective stop/absent peer",
                );
                retained_agrees(
                    &mut callback_errors,
                    &baselines[failed_sample],
                    &preparation.data,
                    &state,
                    &preparation.rng,
                );
                if !absent {
                    check(
                        &mut callback_errors,
                        first_qp.is_some() && qp_bits(&preparation.data) == first_qp,
                        "same-run error-retained QP copies",
                    );
                }
                if reducer.is_output_root() {
                    check(
                        &mut callback_errors,
                        output_agrees(
                            &dir,
                            &directory(&format!("off-{}", failed_sample + 1)),
                            failed_sample + 1,
                        ),
                        "failure boundary output inventory/content/no later sample",
                    );
                } else {
                    check(
                        &mut callback_errors,
                        !dir.exists(),
                        "error peer output absent",
                    );
                }
                agree(
                    reducer,
                    &callback_errors,
                    "retained callback failure boundary",
                );
                println!("MPI_PHYSCAL_CALLBACK rank={} world={} width={width} error_sample={failed_sample} absent_peer={absent_peer} words={}", world.rank(), world.world_size(), preparation.rng.words_consumed());
            }
        }

        // A genuine filesystem failure on the sole output rank must stop every
        // peer before callbacks or another sample, retaining the completed first
        // measurement. The blocking directory is test input, not oracle output.
        preparation = prepared(&root, reducer, width, 3);
        let dir = directory("output-error");
        let blocker = dir.join("zvo_cisajs_001.dat");
        let setup = if reducer.is_output_root() {
            fs::create_dir_all(&blocker).map_err(|e| e.to_string())
        } else {
            Ok(())
        };
        agreed(reducer, setup, "root write-failure setup");
        let local_error = if reducer.is_output_root() {
            fs::File::create(&blocker).err().map(|e| e.to_string())
        } else {
            Some("output sample 0 failed on another MPI rank".into())
        };
        let mut state = VmcOptimizationState::zeros(0, 0, 0, 0, 0, 0, false, false);
        let mut calls = Vec::new();
        let mut callback = |sample, _: &ExpertModeData, _: Complex64, _| {
            calls.push(sample);
            Ok(())
        };
        let result = mvmc_core::vmc_phys_cal_in_place(
            &mut preparation.data,
            &mut state,
            &mut preparation.rng,
            Some(&dir),
            reducer,
            Some(&mut callback),
        );
        let mut errors = Vec::new();
        check(
            &mut errors,
            result.is_err() && result.as_ref().err() == local_error.as_ref(),
            "root filesystem error propagated to all ranks",
        );
        check(
            &mut errors,
            calls.is_empty(),
            "output failure prevents every callback",
        );
        retained_agrees(
            &mut errors,
            &baselines[0],
            &preparation.data,
            &state,
            &preparation.rng,
        );
        if reducer.is_output_root() {
            let names = fs::read_dir(&dir).and_then(|entries| {
                entries
                    .map(|entry| entry.map(|entry| entry.file_name()))
                    .collect::<Result<Vec<_>, _>>()
            });
            let correct_inventory = names.is_ok_and(|mut names| {
                names.sort();
                names
                    == ["zvo_cisajs_001.dat", "zvo_out_001.dat", "zvo_var_001.dat"]
                        .map(std::ffi::OsString::from)
            });
            check(
                &mut errors,
                correct_inventory && blocker.is_dir(),
                "partial root output only/no later sample",
            );
            for name in ["zvo_out_001.dat", "zvo_var_001.dat"] {
                let actual = fs::read_to_string(dir.join(name));
                let expected = fs::read_to_string(directory("off-1").join(name));
                check(
                    &mut errors,
                    matches!((actual, expected), (Ok(a), Ok(b)) if output_body_agrees(name, &a, &b)),
                    "completed root energy/parameter output before write failure",
                );
            }
        } else {
            check(
                &mut errors,
                !dir.exists(),
                "output failure peer writes absent",
            );
        }
        agree(reducer, &errors, "retained root output failure boundary");
        println!(
            "MPI_PHYSCAL_CALLBACK rank={} world={} width={width} output_error=true words={}",
            world.rank(),
            world.world_size(),
            preparation.rng.words_consumed()
        );
    }
}

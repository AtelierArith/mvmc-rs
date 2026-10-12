//! Device RNG and transfer-free interval scheduling (#560). Opt E2E is separate.

use mvmc_core::pfaffian::calc_m_all_real;
use mvmc_core::state::{InvMColMajor, SlaterElmFlat, ThreadedPfaPackWorkspace};
use mvmc_gpu_cuda::interval::{IntervalGeometry, IntervalHost, IntervalSampler, ResidentInterval};
use tenferro_ext::raw_exec::{raw_exec_counters, with_raw_exec, DeviceRng};
use tenferro_gpu::cuda::{cuda_devices, CudaBackend};
use tenferro_tensor::BackendSessionHost;

fn backend() -> CudaBackend {
    let devices = cuda_devices().expect("CUDA discovery");
    CudaBackend::new(devices[0].id()).expect("CUDA backend")
}

#[test]
#[ignore = "requires CUDA"]
fn device_rng_matches_julia_calls_reseed_and_independent_owners() {
    let mut backend = backend();
    let (mut rng, mut buffer) = backend
        .with_backend_session(|s| {
            with_raw_exec(s, "rng.fixture.prepare", |exec| {
                Ok((DeviceRng::new(exec, 42)?, exec.alloc_bytes(4097 * 4)?))
            })
        })
        .unwrap();
    let fixture = include_str!("fixtures/curand_julia.txt");
    let mut tokens = fixture
        .lines()
        .filter(|l| !l.starts_with('#'))
        .flat_map(str::split_whitespace);
    while let Some(key) = tokens.next() {
        let value: u64 = tokens.next().unwrap().parse().unwrap();
        match key {
            "seed" => backend
                .with_backend_session(|s| {
                    with_raw_exec(s, "rng.fixture.reseed", |exec| rng.reseed(exec, value))
                })
                .unwrap(),
            "words" => {
                let words = value as usize;
                let expected: Vec<u32> = (0..words)
                    .map(|_| tokens.next().unwrap().parse().unwrap())
                    .collect();
                // A second generator must not reseed or consume the first one's stream.
                let before = raw_exec_counters();
                backend
                    .with_backend_session(|s| {
                        with_raw_exec(s, "rng.fixture.generate", |exec| {
                            let mut unrelated = DeviceRng::new(exec, 999)?;
                            unrelated.fill_u32(exec, &mut buffer, words)?;
                            rng.fill_u32(exec, &mut buffer, words)?;
                            exec.synchronize()?;
                            Ok(())
                        })
                    })
                    .unwrap();
                let traffic = raw_exec_counters() - before;
                assert_eq!((traffic.upload_calls, traffic.download_calls), (0, 0));
                let mut actual = vec![0u32; words];
                backend
                    .with_backend_session(|s| {
                        with_raw_exec(s, "rng.fixture.read", |exec| unsafe {
                            exec.download(&mut actual, exec.ptr(&buffer)?)
                        })
                    })
                    .unwrap();
                assert_eq!(
                    actual, expected,
                    "exact Julia cuRAND UInt32 sequence, length {words}"
                );
            }
            _ => panic!("unexpected fixture token {key}"),
        }
    }
    // Wrong capacity is rejected before launching; no RNG state is consumed.
    backend
        .with_backend_session(|s| {
            with_raw_exec(s, "rng.bounds", |exec| {
                assert!(rng.fill_u32(exec, &mut buffer, 4098).is_err());
                Ok(())
            })
        })
        .unwrap();
    // Owners retain the primary context and may be destroyed on another host thread.
    std::thread::spawn(move || drop((rng, buffer)))
        .join()
        .unwrap();
}

struct State {
    geom: IntervalGeometry,
    ele: Vec<i32>,
    inv: Vec<f64>,
    pf: Vec<f64>,
    slater: SlaterElmFlat<f64>,
    weights: Vec<f64>,
    gutz: Vec<i32>,
    jast: Vec<i32>,
    params: Vec<f64>,
}

impl State {
    fn host(&self) -> IntervalHost<'_> {
        IntervalHost {
            ele: &self.ele,
            inv: &self.inv,
            pf: &self.pf,
            slater: self.slater.as_slice(),
            weights: &self.weights,
            gutz: &self.gutz,
            jast: &self.jast,
            params: &self.params,
        }
    }

    fn new(walkers: usize, nsite: usize, ne: usize, nqp: usize, constant: bool) -> Self {
        let n = 2 * ne;
        let mut slater = SlaterElmFlat::<f64>::zeros(nqp, nsite);
        let mut rng = sfmt19937::Sfmt19937Rng::new(79);
        for q in 0..nqp {
            for i in 0..2 * nsite {
                for j in 0..i {
                    let v = if constant {
                        1.0
                    } else {
                        rng.gen_rand32() as f64 / 4294967296.0 - 0.5
                    };
                    slater.set(q, i, j, v);
                    slater.set(q, j, i, -v);
                }
            }
        }
        let one: Vec<i64> = (0..n).map(|i| (i % ne) as i64).collect();
        let mut inv = InvMColMajor::<f64>::zeros(nqp, ne);
        let mut pf = vec![0.; nqp];
        calc_m_all_real(
            &one,
            &slater,
            &mut inv,
            &mut pf,
            0,
            nqp,
            nsite,
            ne,
            &ThreadedPfaPackWorkspace::new(n, 1),
        )
        .unwrap();
        let inverse: Vec<_> = (0..nqp)
            .flat_map(|q| inv.qp_matrix_slice(q).iter().copied())
            .collect();
        Self {
            geom: IntervalGeometry {
                walkers,
                n,
                ne,
                nsite,
                nqp,
                ngutz: 0,
                nproj: 0,
            },
            ele: one
                .iter()
                .map(|&x| x as i32)
                .collect::<Vec<_>>()
                .repeat(walkers),
            inv: inverse.repeat(walkers),
            pf: pf.repeat(walkers),
            slater,
            weights: vec![1.; nqp],
            gutz: vec![-1; nsite],
            jast: vec![-1; nsite * nsite],
            params: vec![],
        }
    }
}

#[test]
#[ignore = "requires CUDA"]
fn resident_intervals_complete_rebuild_boundaries_without_transfers() {
    let mut backend = backend();
    // Two sites and one electron per spin: every proposal flips exactly one site.
    // Constant nonzero amplitudes make every move accepted, hitting the scheduling
    // bound (including rebuild on the very last hop), across 20 persistent calls.
    for hops in [0, 1, 2, 3, 4, 17] {
        let state = State::new(3, 2, 1, 1, true);
        let mut resident = backend
            .with_backend_session(|s| {
                ResidentInterval::prepare(s, state.geom, &state.host(), hops, 79)
            })
            .unwrap();
        let mut expected = state.ele.clone();
        for step in 0..20 {
            let mut tape = vec![0u32; 5 * 3 * hops];
            for hop in 0..hops {
                for w in 0..3 {
                    let spin = (step + hop + w) % 2;
                    tape[5 * (w + 3 * hop) + 1] = (spin as u32) << 31;
                    expected[w * 2 + spin] ^= 1;
                }
            }
            backend
                .with_backend_session(|s| resident.advance_with_tape(s, &tape))
                .unwrap();
            assert_eq!(
                backend
                    .with_backend_session(|s| resident.accepted(s))
                    .unwrap(),
                vec![hops as i32; 3]
            );
            let snapshot = backend
                .with_backend_session(|s| resident.snapshot(s))
                .unwrap();
            assert_eq!(snapshot.ele, expected);
            assert!(snapshot
                .aw
                .iter()
                .all(|&a| a == ((step + 1) * hops % 3) as i32));
            for p in snapshot.pf {
                assert!((p - state.pf[0]).abs() < 1e-13);
            }
        }
        let before = raw_exec_counters();
        for _ in 0..20 {
            backend
                .with_backend_session(|s| resident.advance(s))
                .unwrap();
        }
        let traffic = raw_exec_counters() - before;
        assert_eq!(
            (
                traffic.upload_calls,
                traffic.upload_bytes,
                traffic.download_calls,
                traffic.download_bytes
            ),
            (0, 0, 0, 0)
        );
        assert_eq!((traffic.scopes, traffic.synchronizations), (20, 20));
        backend
            .with_backend_session(|s| resident.check_status(s))
            .unwrap();
    }
}

#[test]
#[ignore = "requires CUDA"]
fn resident_schedule_matches_host_control_for_mixed_walkers_and_full_filling() {
    let mut backend = backend();
    for (nsite, ne, nqp, hops) in [(8, 4, 3, 73), (3, 3, 2, 17), (66, 32, 2, 140)] {
        let state = State::new(4, nsite, ne, nqp, false);
        let mut resident = backend
            .with_backend_session(|s| {
                ResidentInterval::prepare(s, state.geom, &state.host(), hops, 79)
            })
            .unwrap();
        let host_control = backend
            .with_backend_session(|s| {
                IntervalSampler::prepare(s, state.geom, &state.host(), hops, 0.0)
            })
            .unwrap();
        let mut rng = sfmt19937::Sfmt19937Rng::new(19);
        for _ in 0..20 {
            let tape: Vec<u32> = (0..4 * hops * 5).map(|_| rng.gen_rand32()).collect();
            backend
                .with_backend_session(|s| resident.advance_with_tape(s, &tape))
                .unwrap();
            backend
                .with_backend_session(|s| host_control.upload_tape(s, &tape))
                .unwrap();
            backend
                .with_backend_session(|s| host_control.reset_offsets(s))
                .unwrap();
            let mut accepted = vec![0i32; 4];
            loop {
                let progress = backend
                    .with_backend_session(|s| host_control.run(s))
                    .unwrap();
                let mut rebuild = vec![];
                for (w, &(_, flag, count)) in progress.iter().enumerate() {
                    accepted[w] += count;
                    if flag != 0 {
                        rebuild.push(w);
                    }
                }
                if rebuild.is_empty() {
                    break;
                }
                backend
                    .with_backend_session(|s| host_control.rebuild(s, &rebuild))
                    .unwrap();
            }
            let actual = backend
                .with_backend_session(|s| resident.snapshot(s))
                .unwrap();
            let expected = backend
                .with_backend_session(|s| host_control.download_all(s))
                .unwrap();
            assert_eq!(actual.ele, expected.ele);
            assert_eq!(actual.aw, expected.aw);
            assert_eq!(
                backend
                    .with_backend_session(|s| resident.accepted(s))
                    .unwrap(),
                accepted
            );
            for (a, b) in actual.pf.iter().zip(&expected.pf) {
                assert!(
                    (a - b).abs() <= 1e-12 + 1e-12 * b.abs(),
                    "Pfaffian {a} vs {b}"
                );
            }
            for (a, b) in actual.logip.iter().zip(&expected.logip) {
                assert!(
                    (a - b).abs() <= 1e-12 + 1e-12 * b.abs(),
                    "log overlap {a} vs {b}"
                );
            }
        }
    }
}

#[test]
#[ignore = "requires two CUDA devices"]
fn device_rng_rejects_a_different_device_context() {
    let devices = cuda_devices().unwrap();
    if devices.len() < 2 {
        return;
    }
    let mut first = CudaBackend::new(devices[0].id()).unwrap();
    let mut second = CudaBackend::new(devices[1].id()).unwrap();
    let (mut rng, mut buffer) = first
        .with_backend_session(|s| {
            with_raw_exec(s, "rng.device.first", |exec| {
                Ok((DeviceRng::new(exec, 79)?, exec.alloc_bytes(64)?))
            })
        })
        .unwrap();
    second
        .with_backend_session(|s| {
            with_raw_exec(s, "rng.device.second", |exec| {
                let mut foreign = exec.alloc_bytes(64)?;
                assert!(exec.ptr(&buffer).is_err());
                assert!(rng.fill_u32(exec, &mut buffer, 16).is_err());
                assert!(rng.fill_u32(exec, &mut foreign, 16).is_err());
                assert!(rng.reseed(exec, 79).is_err());
                Ok(())
            })
        })
        .unwrap();
    first
        .with_backend_session(|s| {
            with_raw_exec(s, "rng.device.valid", |exec| {
                rng.fill_u32(exec, &mut buffer, 16)
            })
        })
        .unwrap();
}

#[test]
#[ignore = "requires CUDA; isolated L128 sampling benchmark, not Opt E2E"]
fn resident_l128_sampling_transfer_benchmark() {
    use std::time::Instant;
    let mut backend = backend();
    for walkers in [64, 320] {
        let hops = 128;
        let steps = 20;
        let words = walkers * hops * 5;
        let state = State::new(walkers, 128, 64, 32, false);
        // Untimed fixture staging gives the host-controlled baseline the exact
        // same 20 library calls. Baseline time excludes generating these tapes.
        let tapes = backend
            .with_backend_session(|s| {
                with_raw_exec(s, "bench.tapes", |exec| {
                    let mut rng = DeviceRng::new(exec, 79)?;
                    let mut buffer = exec.alloc_bytes(words * 4)?;
                    let mut tapes = vec![];
                    for _ in 0..steps {
                        rng.fill_u32(exec, &mut buffer, words)?;
                        let mut tape = vec![0u32; words];
                        unsafe {
                            exec.download(&mut tape, exec.ptr(&buffer)?)?;
                        }
                        tapes.push(tape);
                    }
                    Ok(tapes)
                })
            })
            .unwrap();
        for rep in 0..4 {
            let legacy = backend
                .with_backend_session(|s| {
                    IntervalSampler::prepare(s, state.geom, &state.host(), hops, 0.0)
                })
                .unwrap();
            let before = raw_exec_counters();
            let start = Instant::now();
            for tape in &tapes {
                backend
                    .with_backend_session(|s| legacy.upload_tape(s, tape))
                    .unwrap();
                backend
                    .with_backend_session(|s| legacy.reset_offsets(s))
                    .unwrap();
                loop {
                    let result = backend.with_backend_session(|s| legacy.run(s)).unwrap();
                    let rebuild: Vec<_> = result
                        .iter()
                        .enumerate()
                        .filter_map(|(w, &(_, flag, _))| (flag != 0).then_some(w))
                        .collect();
                    if rebuild.is_empty() {
                        break;
                    }
                    backend
                        .with_backend_session(|s| legacy.rebuild(s, &rebuild))
                        .unwrap();
                }
            }
            let elapsed = start.elapsed().as_secs_f64();
            let traffic = raw_exec_counters() - before;
            let expected = backend
                .with_backend_session(|s| legacy.download_all(s))
                .unwrap();
            eprintln!("L128 sampling only W={walkers} QP=32 steps={steps} hops={hops} rep={rep} host_control seconds={elapsed:.9} {traffic:?}");
            drop(legacy);

            let mut resident = backend
                .with_backend_session(|s| {
                    ResidentInterval::prepare(s, state.geom, &state.host(), hops, 79)
                })
                .unwrap();
            let before = raw_exec_counters();
            let start = Instant::now();
            for _ in 0..steps {
                backend
                    .with_backend_session(|s| resident.advance(s))
                    .unwrap();
            }
            let elapsed = start.elapsed().as_secs_f64();
            let traffic = raw_exec_counters() - before;
            let actual = backend
                .with_backend_session(|s| resident.snapshot(s))
                .unwrap();
            assert_eq!(traffic.upload_bytes + traffic.download_bytes, 0);
            assert_eq!(
                actual.ele, expected.ele,
                "same seeded tapes/configurations W={walkers} rep={rep}"
            );
            assert_eq!(actual.aw, expected.aw);
            for (a, b) in actual.pf.iter().zip(&expected.pf) {
                assert!(
                    (a - b).abs() < 1e-11 + 1e-11 * b.abs(),
                    "Pfaffian {a} vs {b}"
                );
            }
            eprintln!("L128 sampling only W={walkers} QP=32 steps={steps} hops={hops} rep={rep} resident seconds={elapsed:.9} {traffic:?}");
        }
    }
}

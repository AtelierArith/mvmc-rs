//! Group coordinator gate: threaded `IntervalGroup::run_block` must reproduce a
//! single-threaded W-walker `IntervalSampler` run with the same tapes (issue #547).
//! Ignored by default; requires CUDA.

use std::sync::{Arc, Barrier};

use mvmc_core::multichain::{ThreadComm, ThreadReducer};
use mvmc_core::pfaffian::calc_m_all_real;
use mvmc_core::sampling::engine::SamplingEngine;
use mvmc_core::sampling::stage::{StageGeom, StageTables};
use mvmc_core::state::{InvMColMajor, SlaterElmFlat, ThreadedPfaPackWorkspace};
use mvmc_gpu_cuda::interval::{IntervalGeometry, IntervalHost, IntervalSampler};
use mvmc_gpu_cuda::interval_engine::{IntervalEngine, IntervalGroup};
use tenferro_gpu::cuda::{cuda_devices, gpu_available, CudaBackend};
use tenferro_tensor::BackendSessionHost;

const NSITE: usize = 8;
const NE: usize = 4;
const NQP: usize = 2;
const WALKERS: usize = 4;
const BLOCK: usize = 32;
const BLOCKS: usize = 3;

struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u32() as usize) % n
    }
}

fn random_config(rng: &mut Lcg) -> Vec<i32> {
    let mut ele = vec![0i32; 2 * NE];
    for spin in 0..2usize {
        let mut sites: Vec<usize> = (0..NSITE).collect();
        for i in 0..NE {
            let j = i + rng.below(NSITE - i);
            sites.swap(i, j);
            ele[i + spin * NE] = sites[i] as i32;
        }
    }
    ele
}

fn block_tape(walker: usize, block: usize) -> Vec<u32> {
    let mut rng = Lcg(0x1234_5678 ^ (walker as u64 * 131 + block as u64 * 977));
    (0..5 * BLOCK).map(|_| rng.next_u32()).collect()
}

fn geom() -> IntervalGeometry {
    IntervalGeometry {
        walkers: WALKERS,
        n: 2 * NE,
        ne: NE,
        nsite: NSITE,
        nqp: NQP,
        ngutz: 0,
        nproj: 0,
    }
}

#[test]
#[ignore = "requires CUDA"]
fn interval_group_threaded_matches_single_threaded() {
    if !gpu_available() {
        eprintln!("skipped: no CUDA device");
        return;
    }
    let n = 2 * NE;
    let n2 = 2 * NSITE;
    let pool = ThreadedPfaPackWorkspace::new(n, 1);
    let mut rng = Lcg(0xfeed_face_cafe_0001);

    let mut slater = SlaterElmFlat::<f64>::zeros(NQP, NSITE);
    for q in 0..NQP {
        for row in 0..n2 {
            for col in row + 1..n2 {
                let value = rng.next_u32() as f64 / u32::MAX as f64 - 0.5;
                slater.set(q, row, col, value);
                slater.set(q, col, row, -value);
            }
        }
    }
    let weights = vec![1.0f64; NQP];
    let gutz = vec![-1i32; NSITE];
    let jast = vec![-1i32; NSITE * NSITE];
    let params = vec![0.0f64];

    // Initial states per walker.
    let mut ele_all = Vec::new();
    let mut inv_all = Vec::new();
    let mut pf_all = Vec::new();
    for _ in 0..WALKERS {
        let ele = random_config(&mut rng);
        let ele_i64: Vec<i64> = ele.iter().map(|&v| v as i64).collect();
        let mut inv = InvMColMajor::<f64>::zeros(NQP, NE);
        let mut pf = vec![0.0f64; NQP];
        calc_m_all_real(
            &ele_i64, &slater, &mut inv, &mut pf, 0, NQP, NSITE, NE, &pool,
        )
        .expect("host Pfaffian");
        ele_all.extend_from_slice(&ele);
        for q in 0..NQP {
            inv_all.extend_from_slice(inv.qp_matrix_slice(q));
        }
        pf_all.extend_from_slice(&pf);
    }

    // --- Reference: single-threaded W-walker IntervalSampler. ---
    let devices = cuda_devices().expect("device discovery");
    let mut backend = CudaBackend::new(devices[0].id()).expect("cuda backend");
    let host = IntervalHost {
        ele: &ele_all,
        inv: &inv_all,
        pf: &pf_all,
        slater: slater.as_slice(),
        weights: &weights,
        gutz: &gutz,
        jast: &jast,
        params: &params,
    };
    let sampler = backend
        .with_backend_session(|s| IntervalSampler::prepare(s, geom(), &host, BLOCK, 0.0))
        .expect("prepare reference");
    let mut reference = Vec::new();
    for block in 0..BLOCKS {
        // Concatenated tape [word][walker][step].
        let mut tape = vec![0u32; WALKERS * BLOCK * 5];
        for w in 0..WALKERS {
            let t = block_tape(w, block);
            for step in 0..BLOCK {
                for word in 0..5 {
                    tape[5 * (w + WALKERS * step) + word] = t[5 * step + word];
                }
            }
        }
        backend
            .with_backend_session(|s| sampler.reset_offsets(s))
            .expect("reset");
        backend
            .with_backend_session(|s| sampler.upload_tape(s, &tape))
            .expect("upload tape");
        loop {
            let results = backend
                .with_backend_session(|s| sampler.run(s))
                .expect("run");
            let mut need = Vec::new();
            for (w, &(_, r, _)) in results.iter().enumerate() {
                if r != 0 {
                    need.push(w);
                }
            }
            if need.is_empty() {
                break;
            }
            backend
                .with_backend_session(|s| sampler.rebuild(s, &need))
                .expect("rebuild");
        }
        let mut states = Vec::new();
        for w in 0..WALKERS {
            states.push(
                backend
                    .with_backend_session(|s| sampler.download_walker(s, w))
                    .expect("download"),
            );
        }
        reference = states;
    }

    // --- Threaded group: the common tables are published once, by walker 0.
    // Exercise both arrival orders; a non-root rank may stage before or after the root.
    for (root_last, device_initialization) in [(false, false), (true, false), (false, true)] {
        let group = IntervalGroup::new(
            geom(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        );
        for arrival in 0..WALKERS {
            let w = if root_last {
                WALKERS - 1 - arrival
            } else {
                arrival
            };
            group.begin_series_walker(
                w,
                geom(),
                if w == 0 {
                    slater.as_slice().to_vec()
                } else {
                    Vec::new()
                },
                if w == 0 { weights.clone() } else { Vec::new() },
                if w == 0 { gutz.clone() } else { Vec::new() },
                if w == 0 { jast.clone() } else { Vec::new() },
                if w == 0 { params.clone() } else { Vec::new() },
                ele_all[w * n..(w + 1) * n].to_vec(),
                inv_all[w * NQP * n * n..(w + 1) * NQP * n * n].to_vec(),
                pf_all[w * NQP..(w + 1) * NQP].to_vec(),
            );
        }
        let barrier = Arc::new(Barrier::new(WALKERS));
        let comm = ThreadComm::new(WALKERS);
        let mut out = vec![(Vec::<i32>::new(), Vec::<f64>::new()); WALKERS];
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for w in 0..WALKERS {
                let group = group.clone();
                let barrier = barrier.clone();
                let reducer = ThreadReducer::new(comm.clone(), w, 0);
                let (slater, weights, gutz, jast, params, ele_all, pf_all) =
                    (&slater, &weights, &gutz, &jast, &params, &ele_all, &pf_all);
                handles.push(scope.spawn(move || {
                    if device_initialization {
                        let mut engine = IntervalEngine::new(
                            group.clone(),
                            w,
                            geom(),
                            weights.clone(),
                            gutz.clone(),
                            jast.clone(),
                            params.clone(),
                        );
                        let mut inv = InvMColMajor::<f64>::zeros(NQP, NE);
                        let mut pf = vec![f64::NAN; NQP];
                        let pool = ThreadedPfaPackWorkspace::new(n, 1);
                        let cfg: Vec<i64> = ele_all[w * n..(w + 1) * n]
                            .iter()
                            .map(|&v| i64::from(v))
                            .collect();
                        let stage_geom = StageGeom {
                            n_site: NSITE,
                            n_elec: NE,
                            qp_start: 0,
                            qp_end: NQP,
                            inv_stride: n * n + 1,
                        };
                        let mut tables = StageTables {
                            slater_elm: slater,
                            inv_m: &mut inv,
                            pf_m: &mut pf,
                            pool: &pool,
                        };
                        assert!(engine
                            .initialize_real(&stage_geom, &mut tables, &cfg, BLOCK, &reducer)
                            .expect("device initialization"));
                        // Same pivoted FP64 algorithm on valid skew input. No relaxed
                        // trajectory comparison: the following tapes/configurations are exact.
                        for (q, &actual) in tables.pf_m.iter().enumerate() {
                            let expected = pf_all[w * NQP + q];
                            assert!((actual - expected).abs() <= 1e-12 + 1e-12 * expected.abs());
                        }
                        assert!(tables.inv_m.is_stale());
                        engine
                            .begin(&stage_geom, &mut tables, &cfg)
                            .expect("resident begin");
                    }
                    let mut tapes = Vec::new();
                    for block in 0..BLOCKS {
                        tapes.push(block_tape(w, block));
                    }
                    for tape in &tapes {
                        group
                            .run_block(w, BLOCK, tape, &|| {
                                barrier.wait();
                            })
                            .expect("group run_block");
                    }
                    group.walker_state(w)
                }));
            }
            for (w, h) in handles.into_iter().enumerate() {
                let st = h.join().expect("thread");
                out[w] = (st.ele.clone(), st.pf.clone());
            }
        });

        for w in 0..WALKERS {
            let (ref_ele, ref_pf, _, _) = &reference[w];
            assert_eq!(&out[w].0, ref_ele, "walker {w} configuration");
            for (&a, &b) in out[w].1.iter().zip(ref_pf) {
                assert!(
                    (a - b).abs() <= 1e-12 + 1e-12 * b.abs(),
                    "walker {w} Pfaffian: {a} vs {b}"
                );
            }
        }
    }
}

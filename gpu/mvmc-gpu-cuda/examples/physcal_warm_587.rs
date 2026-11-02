//! Developer benchmark: repeated public PhysCal calls, including parsing and file output.
//! Run without correctness tracing or profiling for timings; repetition zero is warmup.
use mvmc_core::{run, stage_backend, threading};
use std::{io::Write, path::PathBuf, time::Instant};

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert!(
        args.len() == 4
            || (args.len() == 5 && matches!(args[4].as_str(), "--contract" | "--initial-contract")),
        "usage: physcal_warm_587 NAMELIST PARAMS FRESH_OUTPUT [--contract|--initial-contract]"
    );
    let output = PathBuf::from(&args[3]);
    std::fs::create_dir(&output).expect("fresh output directory");
    mvmc_gpu_cuda::install();
    threading::install(|| {
        for rep in 0..4 {
            let contract = args.len() == 5;
            if contract {
                mvmc_core::sampling::driver::trace::start_decisions();
                if args[4] == "--initial-contract" {
                    mvmc_core::sampling::driver::trace::start();
                }
            }
            let start = Instant::now();
            let preparation =
                run::prepare_phys_cal_from_namelist(&args[1], &args[2], "real", Some(79))
                    .expect("PhysCal preparation");
            let prepared = start.elapsed().as_secs_f64();
            let result = run::vmc_phys_cal_to_dir(preparation, output.join(format!("rep{rep}")))
                .expect("PhysCal run");
            let seconds = start.elapsed().as_secs_f64();
            assert_eq!(result.iterations, 1);
            let energy = result.state.energy.etot;
            assert!(energy.re.is_finite() && energy.im.is_finite());
            let (_, cursor) = result.final_rng.state_snapshot();
            println!("BENCH rep={rep} seconds={seconds:.9} preparation={prepared:.9} energy={energy:?} rng_words={} rng_cursor={cursor}", result.final_rng.words_consumed());
            if contract {
                let decisions = mvmc_core::sampling::driver::trace::finish_decisions();
                let mut record = std::io::BufWriter::new(
                    std::fs::File::create(output.join(format!("contract-{rep}.txt"))).unwrap(),
                );
                if args[4] == "--initial-contract" {
                    let events = mvmc_core::sampling::driver::trace::finish();
                    let first_words: Vec<_> = events
                        .iter()
                        .filter(|e| e[0] == 9)
                        .take(8)
                        .map(|e| e[1])
                        .collect();
                    writeln!(record, "first_initial_words={first_words:?}").unwrap();
                }
                let (words, index) = result.final_rng.state_snapshot();
                writeln!(record, "rng_words={:?}", words.as_slice()).unwrap();
                writeln!(record, "rng_index={index}").unwrap();
                writeln!(
                    record,
                    "rng_words_consumed={}",
                    result.final_rng.words_consumed()
                )
                .unwrap();
                let cfg = &result.state.electron_config;
                writeln!(record, "ele_idx={:?}", cfg.ele_idx).unwrap();
                writeln!(record, "ele_cfg={:?}", cfg.ele_cfg).unwrap();
                writeln!(record, "ele_num={:?}", cfg.ele_num).unwrap();
                writeln!(record, "ele_spn={:?}", cfg.ele_spn).unwrap();
                writeln!(record, "ele_proj_cnt={:?}", cfg.ele_proj_cnt).unwrap();
                writeln!(record, "counter={:?}", cfg.counter).unwrap();
                for (weight, draw) in decisions {
                    writeln!(
                        record,
                        "decision={},{}",
                        u8::from(weight > draw),
                        (draw * 4294967296.0) as u32
                    )
                    .unwrap();
                }
            }
            let mut handle = stage_backend::acquire();
            println!(
                "RESIDENT rep={rep} {:?}",
                handle.backend().measurement().resident_stats()
            );
        }
    });
}

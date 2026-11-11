//! Matched CPU benchmark worker; serialization is outside the timed public call.
use mvmc_core::{run, SingleProcessReducer};
use std::fs::File;
use std::io::{self, BufRead, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

fn floats(w: &mut impl Write, name: &str, values: impl IntoIterator<Item = f64>) {
    let values: Vec<_> = values.into_iter().collect();
    writeln!(w, "f {name} {}", values.len()).unwrap();
    for value in values {
        writeln!(w, "{value:.17e}").unwrap();
    }
}

fn integers(w: &mut impl Write, name: &str, values: impl IntoIterator<Item = i64>) {
    let values: Vec<_> = values.into_iter().collect();
    writeln!(w, "i {name} {}", values.len()).unwrap();
    for value in values {
        writeln!(w, "{value}").unwrap();
    }
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 6, "mode input output steps real|cmp");
    let output = PathBuf::from(&args[3]);
    std::fs::create_dir_all(&output).unwrap();
    let steps: i64 = args[4].parse().unwrap();
    println!("READY");
    for line in io::stdin().lock().lines() {
        let line = line.unwrap();
        if line == "quit" {
            break;
        }
        let rep: usize = line.parse().unwrap();
        let files = output.join(format!("files-{rep}"));
        assert!(!files.exists());
        let start = Instant::now();
        let (state, rng) = if args[1] == "opt" {
            let mut config = run::RunConfig::new(steps, &args[5]);
            config.nsmp = Some(steps);
            config.seed = Some(79);
            config.output_dir = Some(files.clone());
            let (summary, state, rng) =
                run::run_para_opt_from_namelist_observed(&args[2], config, &SingleProcessReducer)
                    .unwrap();
            assert_eq!(summary.effective_nsteps, usize::try_from(steps).unwrap());
            (state, rng)
        } else {
            assert_eq!(args[1], "phys");
            let prep = run::prepare_phys_cal_without_parameter_file_with_reducer_and_opt_trans(
                &args[2],
                &args[5],
                Some(79),
                &SingleProcessReducer,
                false,
            )
            .unwrap();
            let result = run::vmc_phys_cal_to_dir(prep, &files).unwrap();
            (result.state, result.final_rng)
        };
        let seconds = start.elapsed().as_secs_f64();
        let mut w = BufWriter::new(File::create(output.join(format!("rep-{rep}.txt"))).unwrap());
        let (words, index) = rng.state_snapshot();
        writeln!(w, "rng {} {index}", rng.words_consumed()).unwrap();
        integers(&mut w, "rng", words.into_iter().map(i64::from));
        let e = &state.electron_config;
        for (name, values) in [
            ("counter", e.counter.as_slice()),
            ("saved_idx", e.ele_idx.as_slice()),
            ("current_idx", e.tmp_ele_idx.as_slice()),
            ("current_cfg", e.tmp_ele_cfg.as_slice()),
            ("current_num", e.tmp_ele_num.as_slice()),
            ("current_proj", e.tmp_ele_proj_cnt.as_slice()),
        ] {
            integers(&mut w, name, values.iter().copied());
        }
        let e = &state.energy;
        floats(
            &mut w,
            "energy",
            [e.wc, e.etot, e.etot2, e.sztot, e.sztot2]
                .into_iter()
                .flat_map(|z| [z.re, z.im]),
        );
        floats(
            &mut w,
            "pf_real",
            state.slater_matrix.pf_m_real.iter().copied(),
        );
        floats(
            &mut w,
            "pf_complex",
            state.slater_matrix.pf_m.iter().flat_map(|z| [z.re, z.im]),
        );
        for (i, point) in state.opt_data.iter().enumerate() {
            floats(
                &mut w,
                &format!("step_{i}"),
                [point.energy, point.energy_squared]
                    .into_iter()
                    .chain(point.parameters.iter().copied())
                    .flat_map(|z| [z.re, z.im]),
            );
        }
        if let Some(p) = &state.phys_quantities {
            for (name, values) in [
                ("green1", &p.phys_cis_ajs),
                ("green2", &p.phys_cis_ajs_ckt_alt),
                ("green2_direct", &p.phys_cis_ajs_ckt_alt_dc),
                ("lanczos", &p.phys_lanczos_qqqq),
            ] {
                floats(&mut w, name, values.iter().flat_map(|z| [z.re, z.im]));
            }
        }
        w.flush().unwrap();
        // Only remove this worker's newly created API output; compact evidence remains above.
        std::fs::remove_dir_all(files).unwrap();
        println!("RESULT,{rep},{seconds:.9}");
        io::stdout().flush().unwrap();
    }
}

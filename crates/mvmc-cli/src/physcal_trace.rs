//! Explicit serial PhysCal diagnostics; no oracle execution or sampler replay.

use mvmc_core::run::{PhysCalGreenObserver, PhysCalGreenView};
use mvmc_core::{ExpertModeData, VmcOptimizationState};
use sfmt19937::Sfmt19937Rng;
use std::cell::RefCell;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

pub(crate) struct Trace {
    root: PathBuf,
    stages: RefCell<Vec<String>>,
    error: RefCell<Option<String>>,
}

impl Trace {
    pub(crate) fn create(
        root: &Path,
        seed: Option<i64>,
        mode: &str,
        opt_trans: bool,
    ) -> Result<Self, String> {
        fs::create_dir(root).map_err(|e| format!("PhysCal trace requires a NEW directory: {e}"))?;
        let trace = Self {
            root: root.to_owned(),
            stages: RefCell::new(Vec::new()),
            error: RefCell::new(None),
        };
        trace.write("schema.txt", b"schema=1\norder=Rust actual boundaries, not C chronological replay\nraw_state=not emitted; next624 are future outputs\n")?;
        trace.write(
            "request.txt",
            format!("requested_seed_override={seed:?}\nrequested_mode={mode}\nrequested_opt_trans={opt_trans}\nranks=1\n")
                .as_bytes(),
        )?;
        Ok(trace)
    }

    fn write(&self, name: &str, bytes: &[u8]) -> Result<(), String> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(self.root.join(name))
            .map_err(|e| e.to_string())?;
        file.write_all(bytes).map_err(|e| e.to_string())
    }

    fn record(
        &self,
        stage: &str,
        data: &ExpertModeData,
        rng: Option<&Sfmt19937Rng>,
        consumed: Option<usize>,
        state: Option<&VmcOptimizationState>,
    ) -> Result<(), String> {
        if self.stages.borrow().iter().any(|s| s == stage) {
            return Err(format!("duplicate PhysCal trace stage {stage}"));
        }
        let directory = self.root.join(stage);
        fs::create_dir(&directory).map_err(|e| e.to_string())?;
        let mut parameters = String::new();
        for value in data
            .projection_parameters()
            .into_iter()
            .chain(data.rbm_parameters())
            .chain(data.slater_params.iter().copied())
            .chain(data.opt_trans.iter().copied())
        {
            parameters.push_str(&format!("{:.17e} {:.17e}\n", value.re, value.im));
        }
        fs::write(directory.join("parameters.txt"), parameters).map_err(|e| e.to_string())?;
        fs::write(directory.join("resolved.txt"), format!(
            "actual_c_opt_trans_flags={}\nactual_opt_trans_slots={}\nactual_n_qp_opt_trans={}\nactual_all_complex={}\nactual_orbital_general={}\nactual_qp_total={:?}\n",
            data.c_opt_trans_flags, data.opt_trans.len(), data.n_qp_opt_trans,
            mvmc_core::get_all_complex_flag(data), data.i_flg_orbital_general,
            data.qp_weights.as_ref().map(|weights| weights.qp_full_weight.len())
        )).map_err(|e| e.to_string())?;
        fs::write(directory.join("settings.txt"), format!("nsite={}\nseed_declaration={}\nNSRCG={}\nNStore={}\nNSplitSize={}\nNDataQtySmp={}\nNVMCSample={}\nNVMCWarmUp={}\nNVMCSampleInterval={}\nNMPTrans={}\nconsumed={consumed:?}\nflags={:?}\n", data.modpara.nsite, data.modpara.rnd_seed, data.modpara.nsrcg, data.modpara.nstore_o, data.modpara.nsplit_size, data.modpara.n_data_qty_smp, data.modpara.nvmc_sample, data.modpara.nvmc_warmup, data.modpara.nvmc_interval, data.modpara.nmp_trans, data.optimization_flags)).map_err(|e| e.to_string())?;
        if let Some(rng) = rng {
            let mut peek = rng.clone();
            let next: String = (0..624)
                .map(|_| format!("{}\n", peek.gen_rand32()))
                .collect();
            fs::write(directory.join("next624.txt"), next).map_err(|e| e.to_string())?;
            fs::write(
                directory.join("draw-count.txt"),
                format!("{}\n", rng.words_consumed()),
            )
            .map_err(|e| e.to_string())?;
        }
        if let Some(state) = state {
            let c = &state.electron_config;
            for (name, values) in [
                ("ele_idx", &c.ele_idx),
                ("ele_cfg", &c.ele_cfg),
                ("ele_num", &c.ele_num),
                ("ele_proj_cnt", &c.ele_proj_cnt),
                ("ele_spn", &c.ele_spn),
            ] {
                let text: String = values.iter().map(|v| format!("{v}\n")).collect();
                fs::write(directory.join(format!("{name}.txt")), text)
                    .map_err(|e| e.to_string())?;
            }
            let text: String = c.counter.iter().map(|v| format!("{v}\n")).collect();
            fs::write(directory.join("counter.txt"), text).map_err(|e| e.to_string())?;
        }
        self.stages.borrow_mut().push(stage.to_owned());
        Ok(())
    }

    fn retain(&self, result: Result<(), String>) {
        if let Err(error) = result {
            self.error.borrow_mut().get_or_insert(error);
        }
    }

    pub(crate) fn finish(&self, iterations: usize) -> Result<(), String> {
        if let Some(error) = self.error.borrow().as_ref() {
            return Err(error.clone());
        }
        let mut expected: Vec<String> = [
            "fixed-loaded",
            "overlaid",
            "synchronized",
            "seeded",
            "initialized-clone",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        expected.extend((0..iterations).map(|sample| format!("sample-{sample}")));
        if *self.stages.borrow() != expected {
            return Err("incomplete or reordered PhysCal trace".into());
        }
        self.write(
            "stages.txt",
            format!("{}\n", expected.join("\n")).as_bytes(),
        )?;
        self.write("terminal.txt", b"status=0\n")
    }
}

impl PhysCalGreenObserver for Trace {
    fn lifecycle(
        &self,
        stage: &'static str,
        data: &ExpertModeData,
        rng: Option<&Sfmt19937Rng>,
        consumed: Option<usize>,
    ) {
        self.retain(self.record(stage, data, rng, consumed, None));
    }
    fn sample_completed(
        &self,
        data: &ExpertModeData,
        sample: usize,
        state: &VmcOptimizationState,
        rng: &Sfmt19937Rng,
    ) {
        self.retain(self.record(
            &format!("sample-{sample}"),
            data,
            Some(rng),
            None,
            Some(state),
        ));
    }
    fn accumulated(&self, _view: PhysCalGreenView<'_>) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Directory(PathBuf);
    impl Directory {
        fn new() -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            loop {
                let path = std::env::temp_dir().join(format!(
                    "mvmc-trace-unit-{}-{}",
                    std::process::id(),
                    NEXT.fetch_add(1, Ordering::Relaxed)
                ));
                match fs::create_dir(&path) {
                    Ok(()) => return Self(path),
                    Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                    Err(e) => panic!("{e}"),
                }
            }
        }
    }
    impl Drop for Directory {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }

    #[test]
    fn diagnostics_preserve_actual_rng_state_and_count() {
        let dir = Directory::new();
        let trace = Trace::create(&dir.0.join("trace"), Some(1), "real", true).unwrap();
        let mut rng = Sfmt19937Rng::new(1);
        for _ in 0..625 {
            rng.gen_rand32();
        }
        let before = rng.state_snapshot();
        let words = rng.words_consumed();
        trace
            .record("seeded", &ExpertModeData::new(), Some(&rng), None, None)
            .unwrap();
        assert_eq!(rng.state_snapshot(), before);
        assert_eq!(rng.words_consumed(), words);
        let mut expected = rng.clone();
        let values: Vec<u32> = fs::read_to_string(trace.root.join("seeded/next624.txt"))
            .unwrap()
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        assert_eq!(
            values,
            (0..624).map(|_| expected.gen_rand32()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn observer_on_off_preserves_runner_rng_and_configurations() {
        // Observation-equivalence check, NOT an independent numerical oracle.
        // Public CLI integration tests separately use C/Julia expectations.
        let dir = Directory::new();
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/physcal_181/hubbard_chain_dh_overlays");
        let run = |out: &Path| {
            let preparation = mvmc_core::prepare_phys_cal_from_namelist_with_reducer_and_opt_trans(
                root.join("inputs/namelist.def"),
                root.join("zqp_opt.dat"),
                "real",
                Some(1),
                &mvmc_core::SingleProcessReducer,
                true,
            )
            .unwrap();
            mvmc_core::vmc_phys_cal_to_dir(preparation, out).unwrap()
        };
        let off = run(&dir.0.join("off"));
        let trace =
            std::rc::Rc::new(Trace::create(&dir.0.join("trace"), Some(1), "real", true).unwrap());
        let guard = mvmc_core::run::install_physcal_green_observer(trace.clone()).unwrap();
        let on = run(&dir.0.join("on"));
        drop(guard);
        trace.finish(on.iterations).unwrap();
        assert_eq!(
            on.final_rng.state_snapshot(),
            off.final_rng.state_snapshot()
        );
        assert_eq!(
            on.final_rng.words_consumed(),
            off.final_rng.words_consumed()
        );
        let a = &on.state.electron_config;
        let b = &off.state.electron_config;
        for (a, b) in [
            (&a.ele_idx, &b.ele_idx),
            (&a.ele_cfg, &b.ele_cfg),
            (&a.ele_num, &b.ele_num),
            (&a.ele_proj_cnt, &b.ele_proj_cnt),
            (&a.ele_spn, &b.ele_spn),
            (&a.burn_ele_idx, &b.burn_ele_idx),
            (&a.burn_ele_cfg, &b.burn_ele_cfg),
            (&a.burn_ele_num, &b.burn_ele_num),
            (&a.burn_ele_proj_cnt, &b.burn_ele_proj_cnt),
            (&a.burn_ele_spn, &b.burn_ele_spn),
        ] {
            assert_eq!(a, b);
        }
        assert_eq!(a.counter, b.counter);
        assert_eq!(
            on.data.projection_parameters(),
            off.data.projection_parameters()
        );
        assert_eq!(on.data.slater_params, off.data.slater_params);
        assert_eq!(on.data.opt_trans, off.data.opt_trans);
    }

    #[test]
    fn incomplete_duplicate_and_io_failed_traces_cannot_publish_success() {
        let dir = Directory::new();
        let data = ExpertModeData::new();
        let trace = Trace::create(&dir.0.join("trace"), Some(1), "real", true).unwrap();
        assert!(trace.finish(1).is_err());
        trace.lifecycle("fixed-loaded", &data, None, Some(0));
        trace.lifecycle("fixed-loaded", &data, None, Some(0));
        assert!(trace.finish(1).unwrap_err().contains("duplicate"));
        assert!(!trace.root.join("terminal.txt").exists());
        let reordered = Trace::create(&dir.0.join("reordered"), Some(1), "real", true).unwrap();
        for stage in [
            "overlaid",
            "fixed-loaded",
            "synchronized",
            "seeded",
            "initialized-clone",
        ] {
            reordered.lifecycle(stage, &data, None, Some(0));
        }
        assert!(reordered.finish(0).unwrap_err().contains("reordered"));
        assert!(!reordered.root.join("terminal.txt").exists());
        let missing = Trace::create(&dir.0.join("missing"), Some(1), "real", true).unwrap();
        for stage in [
            "fixed-loaded",
            "overlaid",
            "synchronized",
            "initialized-clone",
        ] {
            missing.lifecycle(stage, &data, None, Some(0));
        }
        assert!(missing.finish(0).is_err());
        assert!(!missing.root.join("terminal.txt").exists());
        let failed = Trace::create(&dir.0.join("failed"), Some(1), "real", true).unwrap();
        fs::write(failed.root.join("fixed-loaded"), "blocked").unwrap();
        failed.lifecycle("fixed-loaded", &data, None, Some(0));
        assert!(failed.finish(0).is_err());
        assert!(!failed.root.join("terminal.txt").exists());
    }
}

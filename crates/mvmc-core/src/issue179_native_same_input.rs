//! Opt-in developer acquisition SOURCE; no C execution/read in Cargo.
//! Same input definitions must be staged and independently SHA-bound externally.
use crate as mvmc_core; // Crate-local #[cfg(test)] module, not integration crate.
use mvmc_core::run::{
    install_optimization_measurement_observer, OptimizationMeasurementObserver,
    OptimizationMeasurementView, RunConfig,
};
use std::{cell::RefCell, path::PathBuf, rc::Rc};

#[derive(Default)]
struct BorrowedSamples {
    samples: RefCell<Vec<Vec<u64>>>,
    rng: RefCell<Vec<(&'static str, [u32; 624], usize, u128, [u32; 624])>>,
}
impl OptimizationMeasurementObserver for BorrowedSamples {
    fn measured(&self, view: OptimizationMeasurementView<'_>) {
        assert_eq!(view.sample, self.samples.borrow().len());
        assert_eq!(view.state.sr_opt.sr_opt_size, 15);
        assert_eq!(view.state.sr_opt.sr_opt_o.len(), 30);
        assert_eq!(view.data.modpara.nvmc_sample, 100);
        assert_eq!(view.data.modpara.nstore_o, 1);
        assert_eq!(view.data.modpara.nsrcg, 0);
        assert_eq!(view.data.modpara.nelec, 3);
        let values = view
            .state
            .sr_opt
            .sr_opt_o
            .iter()
            .flat_map(|z| [z.re.to_bits(), z.im.to_bits()])
            .collect();
        self.samples.borrow_mut().push(values);
    }
    fn rng_boundary(&self, phase: &'static str, rng: &sfmt19937::Sfmt19937Rng) {
        let (raw, cursor) = rng.state_snapshot();
        let count = rng.words_consumed();
        let mut future = [0_u32; 624];
        rng.dump_rand32(&mut future);
        assert_eq!(
            rng.state_snapshot(),
            (raw, cursor),
            "passive future getter state"
        );
        assert_eq!(rng.words_consumed(), count, "passive future getter count");
        self.rng
            .borrow_mut()
            .push((phase, raw, cursor, count, future));
    }
}

#[test]
#[ignore = "explicit independent-input acquisition; not ordinary regression"]
fn issue179_native_same_input_public_runner() {
    let input =
        PathBuf::from(std::env::var_os("ISSUE179_INPUT_NAMELIST").expect("sealed 13 inputs"));
    let out = PathBuf::from(std::env::var_os("ISSUE179_RUST_OUTPUT").expect("exclusive output"));
    let mode = std::env::var("ISSUE179_OBSERVER_MODE").expect("explicit off/on");
    assert!(matches!(mode.as_str(), "off" | "on"));
    let enabled = mode == "on";
    assert!(input.is_absolute() && out.is_absolute());
    assert_eq!(input.file_name().unwrap(), "namelist.def");
    assert_eq!(
        std::fs::canonicalize(&input).unwrap(),
        input,
        "canonical input required"
    );
    assert_eq!(
        std::fs::canonicalize(out.parent().unwrap()).unwrap(),
        out.parent().unwrap()
    );
    assert!(!out.exists());
    assert!(std::fs::symlink_metadata(&out).is_err());
    let diagnostic = out.with_extension("observer");
    assert!(std::fs::symlink_metadata(&diagnostic).is_err());
    std::fs::create_dir(&diagnostic).unwrap();
    let parent = input.parent().unwrap();
    macro_rules! original {
        ($($name:literal),+ $(,)?) => {
            $(assert!(!std::fs::symlink_metadata(parent.join($name)).unwrap().file_type().is_symlink(),$name);
              assert_eq!(std::fs::read(parent.join($name)).unwrap(),
                include_bytes!(concat!("../tests/fixtures/issue179_native_prefix1/",$name)).as_slice(),$name);)+
        };
    }
    original!(
        "namelist.def",
        "modpara.def",
        "locspn.def",
        "trans.def",
        "coulombinter.def",
        "hund.def",
        "exchange.def",
        "greenone.def",
        "greentwo.def",
        "gutzwilleridx.def",
        "jastrowidx.def",
        "orbitalidx.def",
        "qptransidx.def"
    );
    assert!(!parent.join("initial.def").exists());
    // Existing raw schema2 captures actual candidate/draw/decision/checkpoint;
    // no fabricated native C draw counter and no RNG replay.
    if enabled {
        mvmc_core::sampling::driver::trace::start_with_raw_checkpoints();
    }
    let mut systems = enabled.then(|| mvmc_core::sr::observer::capture_with_normalized().unwrap());
    let observer = Rc::new(BorrowedSamples::default());
    let _guard =
        enabled.then(|| install_optimization_measurement_observer(observer.clone()).unwrap());
    let mut config = RunConfig::new(1, "real");
    config.nsmp = Some(1);
    config.output_dir = Some(out.clone());
    // Do not override input seed, sampling count, parameter initialization.
    config.enable_opt_trans = Some(false); // C input has no OptTrans definition.
    let result = mvmc_core::run_para_opt_from_namelist(input, config);
    let trace = if enabled {
        mvmc_core::sampling::driver::trace::finish()
    } else {
        Vec::new()
    };
    let normalized = systems
        .as_mut()
        .map(|s| s.take_normalized())
        .unwrap_or_default();
    let direct = systems.map(|s| s.finish()).unwrap_or_default();
    // Save partial actual trace BEFORE checking the native runner outcome.
    std::fs::write(
        diagnostic.join("outcome.txt"),
        format!("mode={mode}\nresult={result:?}\n"),
    )
    .unwrap();
    if enabled {
        std::fs::write(
            diagnostic.join("actual-sampling-schema2.txt"),
            format!("{trace:?}\n"),
        )
        .unwrap();
        std::fs::write(
            diagnostic.join("actual-normalized.txt"),
            format!("{normalized:?}\n"),
        )
        .unwrap();
        std::fs::write(
            diagnostic.join("actual-direct-systems.txt"),
            format!("{direct:?}\n"),
        )
        .unwrap();
        std::fs::write(
            diagnostic.join("actual-runner-rng-boundaries.txt"),
            format!("{:?}\n", observer.rng.borrow()),
        )
        .unwrap();
        std::fs::write(
            diagnostic.join("borrowed-O-complex-bits.txt"),
            format!("{:?}\n", observer.samples.borrow()),
        )
        .unwrap();
    }
    let summary = result.unwrap();
    assert_eq!(summary.status, 0);
    assert_eq!(summary.effective_nsteps, 1);
    if !enabled {
        assert!(observer.samples.borrow().is_empty());
        return;
    }
    assert_eq!(observer.samples.borrow().len(), 100);
    assert_eq!(
        observer
            .rng
            .borrow()
            .iter()
            .map(|r| r.0)
            .collect::<Vec<_>>(),
        vec!["initialized", "sampling-return"]
    );
    assert_eq!(direct.len(), 1);
    assert_eq!(normalized.len(), 1);
    assert_eq!(direct[0].capture_step, Some(0));
    assert_eq!(normalized[0].step, 0);
    assert_eq!(direct[0].triangle, 'U');
    assert_eq!(direct[0].dimension, 10);
    assert_eq!(direct[0].settings.input_seed, 1);
    assert_eq!(direct[0].settings.steps, 1);
    assert_eq!(direct[0].settings.window, 1);
    assert_eq!(direct[0].settings.nstore, 1);
    assert_eq!(direct[0].settings.nsrcg, 0);
    // No equality/tolerance assertion until independent discrete/system join.
}

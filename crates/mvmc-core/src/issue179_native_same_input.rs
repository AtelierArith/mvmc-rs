//! Opt-in developer acquisition SOURCE; no C execution/read in Cargo.
//! Same input definitions must be staged and independently SHA-bound externally.
use crate as mvmc_core; // Crate-local #[cfg(test)] module, not integration crate.
use mvmc_core::run::{
    install_optimization_measurement_observer, OptimizationMeasurementObserver,
    OptimizationMeasurementView, RunConfig,
};
use std::{cell::RefCell, path::PathBuf, rc::Rc};

type RngSnapshot = (&'static str, [u32; 624], usize, u128, [u32; 624]);
type LocalMoments = (usize, Vec<u64>, Vec<u64>);

#[derive(Debug, Default)]
struct OperandCapture {
    weighted: Vec<Vec<u64>>,
    local: Vec<LocalMoments>,
}
thread_local! {
    static OPERANDS: RefCell<Option<OperandCapture>> = const { RefCell::new(None) };
}
struct OperandGuard;
impl OperandGuard {
    fn start() -> Self {
        OPERANDS.with(|slot| {
            assert!(slot.borrow().is_none(), "nested native operand capture");
            *slot.borrow_mut() = Some(OperandCapture::default());
        });
        Self
    }
    fn finish(self) -> OperandCapture {
        OPERANDS.with(|slot| slot.borrow_mut().take().expect("operand capture active"))
    }
}
impl Drop for OperandGuard {
    fn drop(&mut self) {
        OPERANDS.with(|slot| *slot.borrow_mut() = None);
    }
}
pub(super) fn weighted_store(active: &[f64], n: usize, samples: usize) {
    OPERANDS.with(|slot| {
        if let Some(records) = slot.borrow_mut().as_mut() {
            assert!(
                records.weighted.is_empty(),
                "one weighted operand capture only"
            );
            assert_eq!((n, samples, active.len()), (15, 100, 1500));
            records
                .weighted
                .push(active.iter().map(|x| x.to_bits()).collect());
        }
    });
}
pub(super) fn local_moments(
    step: usize,
    state: &crate::state::VmcOptimizationState,
    all_complex: bool,
) {
    OPERANDS.with(|slot| {
        if let Some(records) = slot.borrow_mut().as_mut() {
            assert!(records.local.is_empty(), "one local moment capture only");
            assert!(!all_complex);
            assert_eq!(state.sr_opt.sr_opt_size, 15);
            assert_eq!(state.sr_opt.sr_opt_oo_real.len(), 255);
            assert_eq!(state.sr_opt.sr_opt_ho_real.len(), 15);
            // Preserve all extra native storage; semantic OO is first 225.
            records.local.push((
                step,
                state
                    .sr_opt
                    .sr_opt_oo_real
                    .iter()
                    .map(|x| x.to_bits())
                    .collect(),
                state
                    .sr_opt
                    .sr_opt_ho_real
                    .iter()
                    .map(|x| x.to_bits())
                    .collect(),
            ));
        }
    });
}

#[test]
fn operand_hook_rejects_geometry_and_second_copy_without_losing_receipt() {
    let capture = OperandGuard::start();
    let weighted = [1.0; 1500];
    assert!(std::panic::catch_unwind(|| weighted_store(&weighted, 14, 100)).is_err());
    OPERANDS.with(|slot| assert!(slot.borrow().as_ref().unwrap().weighted.is_empty()));
    weighted_store(&weighted, 15, 100);
    assert!(std::panic::catch_unwind(|| weighted_store(&weighted, 15, 100)).is_err());
    let receipt = capture.finish();
    assert_eq!(receipt.weighted.len(), 1);
    assert_eq!(receipt.weighted[0], vec![1.0_f64.to_bits(); 1500]);
    assert!(receipt.local.is_empty());
    OPERANDS.with(|slot| assert!(slot.borrow().is_none()));
}

#[derive(Default)]
struct BorrowedSamples {
    samples: RefCell<Vec<Vec<u64>>>,
    rng: RefCell<Vec<RngSnapshot>>,
}

#[test]
fn optimization_operand_hook_receipt_and_passivity() {
    let input = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/issue179_native_prefix1/namelist.def");
    let root = super::fresh_output_directory().unwrap();
    let mut snapshots = Vec::new();
    for enabled in [false, true] {
        let out = root.join(if enabled { "on" } else { "off" });
        // Common passive RNG/measurement tap in both control runs; this does not
        // replace the ignored acquisition's genuinely observer-free OFF process.
        let observer = Rc::new(BorrowedSamples::default());
        let tap = install_optimization_measurement_observer(observer.clone()).unwrap();
        let operands = enabled.then(OperandGuard::start);
        let mut config = RunConfig::new(1, "real");
        config.nsmp = Some(1);
        config.output_dir = Some(out);
        config.enable_opt_trans = Some(false);
        let result = mvmc_core::run_para_opt_from_namelist(&input, config);
        let captured = operands.map(OperandGuard::finish);
        drop(tap);
        assert_eq!(result.unwrap().status, 0);
        assert_eq!(observer.samples.borrow().len(), 100);
        snapshots.push(observer.rng.borrow().clone());
        if let Some(captured) = captured {
            assert_eq!(captured.weighted.len(), 1);
            assert_eq!(captured.weighted[0].len(), 1500);
            assert_eq!(captured.local.len(), 1);
            assert_eq!(captured.local[0].0, 0);
            assert_eq!(captured.local[0].1.len(), 255);
            assert_eq!(captured.local[0].2.len(), 15);
        } else {
            OPERANDS.with(|slot| assert!(slot.borrow().is_none()));
        }
    }
    assert_eq!(
        snapshots[0], snapshots[1],
        "raw/cursor/count/future passivity"
    );
    for name in [
        "zqp_opt.dat",
        "zvo_SRinfo.dat",
        "zvo_out_001.dat",
        "zvo_var_001.dat",
    ] {
        assert_eq!(
            std::fs::read(root.join("off").join(name)).unwrap(),
            std::fs::read(root.join("on").join(name)).unwrap(),
            "same-language operand hook passivity {name}"
        );
    }
    println!("optimization-local-hook: calls=1 step=0 OO=255 HO=15 weighted=1500; common-tap RNG/public passivity");
    std::fs::remove_dir_all(root).unwrap();
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
    let operands = enabled.then(OperandGuard::start);
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
    let operands = operands.map(OperandGuard::finish);
    // Save partial actual trace BEFORE checking the native runner outcome.
    std::fs::write(
        diagnostic.join("outcome.txt"),
        format!("mode={mode}\nresult={result:?}\n"),
    )
    .unwrap();
    if enabled {
        let operands = operands.as_ref().expect("enabled operand capture");
        std::fs::write(
            diagnostic.join("actual-weighted-store-bits.txt"),
            format!("{:?}\n", operands.weighted),
        )
        .unwrap();
        std::fs::write(
            diagnostic.join("actual-local-moments-bits.txt"),
            format!("{:?}\n", operands.local),
        )
        .unwrap();
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
    let operands = operands.unwrap();
    assert_eq!(operands.weighted.len(), 1);
    assert_eq!(operands.local.len(), 1);
    assert_eq!(operands.local[0].0, 0);
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

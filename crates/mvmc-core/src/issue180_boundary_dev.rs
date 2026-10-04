//! Private cfg(test) capture proposal. Not a public runner API or normal fixture.
use crate::state::VmcOptimizationState;
use mvmc_expert_parsers::ExpertModeData;
use num_complex::Complex64;
use sfmt19937::Sfmt19937Rng;
use std::cell::RefCell;

#[derive(Debug, PartialEq)]
pub(super) struct Boundary {
    pub label: String,
    pub raw: [u32; 624],
    pub cursor: usize,
    pub consumed: u128,
    pub future: [u32; 624],
    pub configuration: Vec<Vec<i64>>,
    pub counters: [i64; 10],
    pub parameters: Vec<Complex64>,
    pub energy: [Complex64; 2],
}

thread_local! {
    static RECORDS: RefCell<Option<Vec<Boundary>>> = const { RefCell::new(None) };
    static TERMINAL_ONLY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

// Private to the runner's test module. Dropping scope never leaves capture enabled.
pub(super) struct Scope(std::marker::PhantomData<std::rc::Rc<()>>);
pub(super) fn start() -> Result<Scope, &'static str> {
    RECORDS.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_some() {
            return Err("boundary capture already active");
        }
        *slot = Some(Vec::new());
        Ok(Scope(std::marker::PhantomData))
    })
}
fn start_terminal() -> Result<Scope, &'static str> {
    let scope = start()?;
    TERMINAL_ONLY.with(|flag| flag.set(true));
    Ok(scope)
}
impl Scope {
    pub(super) fn take(&mut self) -> Vec<Boundary> {
        RECORDS.with(|slot| std::mem::take(slot.borrow_mut().as_mut().expect("active scope")))
    }
}
impl Drop for Scope {
    fn drop(&mut self) {
        RECORDS.with(|slot| *slot.borrow_mut() = None);
        TERMINAL_ONLY.with(|flag| flag.set(false));
    }
}
pub(super) fn record_step(
    step: usize,
    data: &ExpertModeData,
    state: &VmcOptimizationState,
    rng: &Sfmt19937Rng,
) {
    if TERMINAL_ONLY.with(|flag| flag.get()) || RECORDS.with(|slot| slot.borrow().is_none()) {
        return;
    }
    record(&format!("post-sync-{step}"), data, state, rng);
}

pub(super) fn record(
    label: &str,
    data: &ExpertModeData,
    state: &VmcOptimizationState,
    rng: &Sfmt19937Rng,
) {
    if TERMINAL_ONLY.with(|flag| flag.get()) && label.starts_with("post-sync-") {
        return;
    }
    RECORDS.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(records) = slot.as_mut() else {
            return;
        };
        let mut boundary = discrete_snapshot(label, state, rng);
        boundary.parameters =
            crate::parameters::pack_parameters(data).expect("validated declared parameters");
        boundary.energy = [state.energy.etot, state.energy.etot2];
        records.push(boundary);
    });
}

fn discrete_snapshot(label: &str, state: &VmcOptimizationState, rng: &Sfmt19937Rng) -> Boundary {
    let ec = &state.electron_config;
    let fields = [
        &ec.ele_idx,
        &ec.ele_cfg,
        &ec.ele_num,
        &ec.ele_proj_cnt,
        &ec.ele_spn,
        &ec.tmp_ele_idx,
        &ec.tmp_ele_cfg,
        &ec.tmp_ele_num,
        &ec.tmp_ele_proj_cnt,
        &ec.tmp_ele_spn,
        &ec.burn_ele_idx,
        &ec.burn_ele_cfg,
        &ec.burn_ele_num,
        &ec.burn_ele_proj_cnt,
        &ec.burn_ele_spn,
    ];
    let (raw, cursor) = rng.state_snapshot();
    let consumed = rng.words_consumed();
    let mut future = [0; 624];
    rng.dump_rand32(&mut future);
    assert_eq!(
        (raw, cursor, consumed),
        {
            let (after, position) = rng.state_snapshot();
            (after, position, rng.words_consumed())
        },
        "passive dump advanced source RNG"
    );
    Boundary {
        label: label.to_owned(),
        raw,
        cursor,
        consumed,
        future,
        configuration: fields.iter().map(|field| (**field).clone()).collect(),
        counters: ec.counter,
        parameters: Vec::new(),
        energy: [Complex64::new(0.0, 0.0); 2],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::Path;

    const FIELDS: [&str; 15] = [
        "ele_idx",
        "ele_cfg",
        "ele_num",
        "ele_proj_cnt",
        "ele_spn",
        "tmp_ele_idx",
        "tmp_ele_cfg",
        "tmp_ele_num",
        "tmp_ele_proj_cnt",
        "tmp_ele_spn",
        "burn_ele_idx",
        "burn_ele_cfg",
        "burn_ele_num",
        "burn_ele_proj_cnt",
        "burn_ele_spn",
    ];

    fn normalized_and_system_inventory_valid(
        steps: &[usize],
        systems: &[Option<usize>],
        direct: bool,
    ) -> bool {
        steps.iter().copied().eq(0..20)
            && if direct {
                systems.iter().copied().eq((0..20).map(Some))
            } else {
                systems.is_empty()
            }
    }

    fn check_missing_duplicate_and_reorder_controls() {
        let valid: Vec<_> = (0..20).collect();
        let systems: Vec<_> = (0..20).map(Some).collect();
        assert!(normalized_and_system_inventory_valid(
            &valid, &systems, true
        ));
        assert!(normalized_and_system_inventory_valid(&valid, &[], false));
        assert!(!normalized_and_system_inventory_valid(
            &valid[..19],
            &systems,
            true
        ));
        let mut duplicate = valid.clone();
        duplicate[19] = 18;
        assert!(!normalized_and_system_inventory_valid(
            &duplicate, &systems, true
        ));
        let mut reordered = valid.clone();
        reordered.swap(0, 1);
        assert!(!normalized_and_system_inventory_valid(
            &reordered, &systems, true
        ));
        assert!(!normalized_and_system_inventory_valid(
            &valid,
            &systems[..19],
            true
        ));
        let mut duplicate_system = systems.clone();
        duplicate_system[19] = Some(18);
        assert!(!normalized_and_system_inventory_valid(
            &valid,
            &duplicate_system,
            true
        ));
        let mut reordered_system = systems.clone();
        reordered_system.swap(0, 1);
        assert!(!normalized_and_system_inventory_valid(
            &valid,
            &reordered_system,
            true
        ));
        duplicate_system[19] = None;
        assert!(!normalized_and_system_inventory_valid(
            &valid,
            &duplicate_system,
            true
        ));
        assert!(!normalized_and_system_inventory_valid(
            &valid, &systems, false
        ));
    }
    fn vector<T: std::fmt::Display>(path: &Path, values: &[T]) {
        let mut out = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .unwrap();
        for (index, value) in values.iter().enumerate() {
            if index > 0 {
                write!(out, " ").unwrap();
            }
            write!(out, "{value}").unwrap();
        }
        writeln!(out).unwrap();
    }
    fn numeric(path: &Path, values: &[Complex64]) {
        use crate::issue180_binary_diagnostics::{write_record, Kind};
        let name = path.file_name().unwrap().to_str().unwrap();
        let kind = if name.ends_with("-sr_oo.txt") {
            Kind::Oo
        } else if name.ends_with("-sr_ho.txt") {
            Kind::Ho
        } else if name.ends_with("-parameters.txt") {
            Kind::Parameters
        } else if name.ends_with("-energy.txt") {
            Kind::Energy
        } else {
            panic!("unregistered numeric record");
        };
        let binary_path = path.with_extension("f64bin");
        write_record(
            &binary_path,
            kind,
            values.len(),
            values.iter().flat_map(|value| [value.re, value.im]),
        )
        .unwrap();
    }
    fn write_boundary(root: &Path, label: &str, item: &Boundary) {
        let mut rng = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(root.join(format!("{label}-rng-state.txt")))
            .unwrap();
        for (index, word) in item.raw.iter().enumerate() {
            if index > 0 {
                write!(rng, " ").unwrap();
            }
            write!(rng, "{word}").unwrap();
        }
        writeln!(rng, "\n{}\n{}", item.cursor, item.consumed).unwrap();
        vector(&root.join(format!("{label}-future624.txt")), &item.future);
        let mut config = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(root.join(format!("{label}-config-all.txt")))
            .unwrap();
        for (name, values) in FIELDS.iter().zip(&item.configuration) {
            writeln!(config, "{name} {}", values.len()).unwrap();
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    write!(config, " ").unwrap();
                }
                write!(config, "{value}").unwrap();
            }
            writeln!(config).unwrap();
        }
        writeln!(config, "counter 10").unwrap();
        for (index, value) in item.counters.iter().enumerate() {
            if index > 0 {
                write!(config, " ").unwrap();
            }
            write!(config, "{value}").unwrap();
        }
        writeln!(config).unwrap();
        numeric(
            &root.join(format!("{label}-parameters.txt")),
            &item.parameters,
        );
        numeric(&root.join(format!("{label}-energy.txt")), &item.energy);
    }

    #[test]
    fn passive_snapshot_preserves_raw_cursor_count_and_future() {
        let state = VmcOptimizationState::zeros(4, 2, 0, 0, 1, 1, false, false);
        let mut rng = Sfmt19937Rng::new(12395);
        for _ in 0..625 {
            rng.gen_rand32();
        }
        let before = (rng.state_snapshot(), rng.words_consumed());
        let mut scope = start().unwrap();
        assert!(start().is_err());
        // Use the isolated snapshot helper without needing a model input/data.
        let snapshot = discrete_snapshot("control", &state, &rng);
        assert_eq!(before, (rng.state_snapshot(), rng.words_consumed()));
        let mut clone = rng.clone();
        let expected: [u32; 624] = std::array::from_fn(|_| clone.gen_rand32());
        assert_eq!(snapshot.raw, before.0 .0);
        assert_eq!(snapshot.cursor, before.0 .1);
        assert_eq!(snapshot.consumed, before.1);
        assert_eq!(snapshot.future, expected);
        assert_eq!(snapshot.counters, state.electron_config.counter);
        let ec = &state.electron_config;
        let expected_fields = [
            &ec.ele_idx,
            &ec.ele_cfg,
            &ec.ele_num,
            &ec.ele_proj_cnt,
            &ec.ele_spn,
            &ec.tmp_ele_idx,
            &ec.tmp_ele_cfg,
            &ec.tmp_ele_num,
            &ec.tmp_ele_proj_cnt,
            &ec.tmp_ele_spn,
            &ec.burn_ele_idx,
            &ec.burn_ele_cfg,
            &ec.burn_ele_num,
            &ec.burn_ele_proj_cnt,
            &ec.burn_ele_spn,
        ];
        assert_eq!(
            snapshot.configuration,
            expected_fields
                .iter()
                .map(|field| (**field).clone())
                .collect::<Vec<_>>()
        );
        assert_eq!(before, (rng.state_snapshot(), rng.words_consumed()));
        assert!(scope.take().is_empty());
        drop(scope);
        assert!(start().is_ok());
    }

    #[test]
    fn scope_panic_cleanup_and_thread_local_isolation() {
        let failed = std::panic::catch_unwind(|| {
            let _scope = start_terminal().unwrap();
            assert!(start().is_err());
            std::thread::spawn(|| {
                let _independent = start().unwrap();
            })
            .join()
            .unwrap();
            panic!("deliberate scope unwind");
        });
        assert!(failed.is_err());
        assert!(!TERMINAL_ONLY.with(|flag| flag.get()));
        let _replacement = start().unwrap();
    }

    #[test]
    fn disabled_hook_has_no_records_or_rng_mutation() {
        let state = VmcOptimizationState::zeros(4, 2, 0, 0, 1, 1, false, false);
        let data = ExpertModeData::new();
        let rng = Sfmt19937Rng::new(1);
        let before = (rng.state_snapshot(), rng.words_consumed());
        assert!(RECORDS.with(|slot| slot.borrow().is_none()));
        // Disabled capture returns before any parameter packing/clone.
        record("disabled", &data, &state, &rng);
        assert!(RECORDS.with(|slot| slot.borrow().is_none()));
        assert_eq!(before, (rng.state_snapshot(), rng.words_consumed()));
    }

    #[test]
    fn full_field_order_lengths_and_explicit_boundary_sequence() {
        check_missing_duplicate_and_reorder_controls();
        let mut state = VmcOptimizationState::zeros(4, 2, 0, 0, 1, 1, false, false);
        let ec = &mut state.electron_config;
        let fields = [
            &mut ec.ele_idx,
            &mut ec.ele_cfg,
            &mut ec.ele_num,
            &mut ec.ele_proj_cnt,
            &mut ec.ele_spn,
            &mut ec.tmp_ele_idx,
            &mut ec.tmp_ele_cfg,
            &mut ec.tmp_ele_num,
            &mut ec.tmp_ele_proj_cnt,
            &mut ec.tmp_ele_spn,
            &mut ec.burn_ele_idx,
            &mut ec.burn_ele_cfg,
            &mut ec.burn_ele_num,
            &mut ec.burn_ele_proj_cnt,
            &mut ec.burn_ele_spn,
        ];
        for (index, field) in fields.into_iter().enumerate() {
            *field = vec![(index as i64) - 7; index % 5];
        }
        ec.counter = std::array::from_fn(|i| i as i64 - 5);
        let rng = Sfmt19937Rng::new(1);
        let mut scope = start().unwrap();
        let mut labels = vec!["initialized".to_string()];
        labels.extend((0..20).map(|step| format!("post-sync-{step}")));
        labels.push("final".to_string());
        for label in &labels {
            let snapshot = discrete_snapshot(label, &state, &rng);
            RECORDS.with(|slot| slot.borrow_mut().as_mut().unwrap().push(snapshot));
        }
        let records = scope.take();
        assert_eq!(records.len(), 22);
        for (record, label) in records.iter().zip(labels) {
            assert_eq!(record.label, label);
            assert_eq!(record.configuration.len(), 15);
            for (index, field) in record.configuration.iter().enumerate() {
                assert_eq!(field, &vec![index as i64 - 7; index % 5]);
            }
            let expected_counters: [i64; 10] = std::array::from_fn(|i| i as i64 - 5);
            assert_eq!(record.counters, expected_counters);
        }
    }

    #[test]
    #[ignore = "developer-only explicit external inputs/receipt; never normal Cargo oracle"]
    fn issue180_capture20_public_runner() {
        let input = std::path::PathBuf::from(std::env::var("ISSUE180_NAMELIST").unwrap());
        let out = std::path::PathBuf::from(std::env::var("ISSUE180_CAPTURE_OUT").unwrap());
        let model = std::env::var("ISSUE180_MODEL").unwrap();
        let solver = std::env::var("ISSUE180_SOLVER").unwrap();
        let mode = std::env::var("ISSUE180_CAPTURE_MODE").unwrap();
        assert!(matches!(mode.as_str(), "off" | "terminal" | "observed"));
        assert!(input.is_absolute() && out.is_absolute());
        let declared =
            mvmc_expert_parsers::parse_expert_mode_files_with_c_opt_trans(&input, false).unwrap();
        assert_eq!(declared.modpara.vmc_calc_mode, 0);
        assert_eq!(declared.i_flg_orbital_general, 0);
        match (model.as_str(), solver.as_str()) {
            (
                "heisenberg_chain_real" | "heisenberg_chain_cmp" | "hubbard_chain_real",
                "canonical",
            ) => {
                assert_eq!(declared.modpara.rnd_seed, 1);
                assert_eq!(declared.modpara.nvmc_sample, 100);
                assert_eq!((declared.modpara.nsrcg, declared.modpara.nstore_o), (0, 1));
                assert_eq!(
                    super::super::get_all_complex_flag(&declared).unwrap(),
                    model == "heisenberg_chain_cmp",
                );
            }
            ("general_rbm_cmp", "direct" | "cg") => {
                assert_eq!(declared.modpara.rnd_seed, 12395);
                assert_eq!(declared.modpara.nvmc_sample, 100);
            }
            _ => panic!("reviewed ordinary first/next-tranche model/solver only"),
        }
        assert_eq!(declared.modpara.nvmc_warmup, 10);
        assert_eq!(declared.modpara.nvmc_interval, 1);
        if solver == "direct" {
            assert_eq!((declared.modpara.nsrcg, declared.modpara.nstore_o), (0, 1));
        }
        if solver == "cg" {
            assert_eq!((declared.modpara.nsrcg, declared.modpara.nstore_o), (1, 0));
        }
        std::fs::create_dir(&out).unwrap();
        std::fs::copy(
            input
                .parent()
                .unwrap()
                .join("developer-input-identity.json"),
            out.join("input-identity.json"),
        )
        .unwrap();
        std::fs::write(out.join("model-settings.txt"), format!(
            "model={model} solver={solver} mode={mode} steps=20 window=20 seed={} samples={} warmup={} interval={} NSRCG={} NStore={}\n",
            declared.modpara.rnd_seed, declared.modpara.nvmc_sample, declared.modpara.nvmc_warmup,
            declared.modpara.nvmc_interval, declared.modpara.nsrcg, declared.modpara.nstore_o)).unwrap();
        let mut scope = match mode.as_str() {
            "off" => None,
            "terminal" => Some(start_terminal().unwrap()),
            "observed" => Some(start().unwrap()),
            _ => unreachable!(),
        };
        let mut normalized = if mode == "observed" {
            Some(crate::sr::observer::capture_with_normalized().unwrap())
        } else {
            None
        };
        let mut config = super::super::RunConfig::new(
            20,
            if matches!(
                model.as_str(),
                "heisenberg_chain_real" | "hubbard_chain_real"
            ) {
                "real"
            } else {
                "cmp"
            },
        );
        config.nsmp = Some(20);
        config.output_dir = Some(out.join("public-output"));
        let result = super::super::run_para_opt_from_namelist(&input, config);
        // Preserve actual observed prefix even when runner failed; no invented final.
        let records = scope.as_mut().map_or_else(Vec::new, Scope::take);
        for item in &records {
            let label = if let Some(step) = item.label.strip_prefix("post-sync-") {
                format!("frame-{}", step.parse::<usize>().unwrap() + 1)
            } else {
                item.label.clone()
            };
            write_boundary(&out, &label, item);
        }
        let normalized_records = normalized
            .as_mut()
            .map_or_else(Vec::new, |guard| guard.take_normalized());
        let normalized_steps: Vec<usize> =
            normalized_records.iter().map(|item| item.step).collect();
        for item in normalized_records {
            let label = format!("frame-{}", item.step + 1);
            if item.mode == crate::sr::observer::DirectMode::Real {
                numeric(
                    &out.join(format!("{label}-sr_oo.txt")),
                    &item
                        .oo_real
                        .iter()
                        .map(|x| Complex64::new(*x, 0.0))
                        .collect::<Vec<_>>(),
                );
                numeric(
                    &out.join(format!("{label}-sr_ho.txt")),
                    &item
                        .ho_real
                        .iter()
                        .map(|x| Complex64::new(*x, 0.0))
                        .collect::<Vec<_>>(),
                );
            } else {
                numeric(&out.join(format!("{label}-sr_oo.txt")), &item.oo);
                numeric(&out.join(format!("{label}-sr_ho.txt")), &item.ho);
            }
        }
        // Serialize existing actual solve observations, including prefixes on Err.
        // This adds no solve, observer, reduction, or RNG call.
        let systems = normalized
            .take()
            .map_or_else(Vec::new, |guard| guard.finish());
        let system_steps: Vec<_> = systems.iter().map(|item| item.capture_step).collect();
        for (ordinal, item) in systems.iter().enumerate() {
            let label = format!("system-{}", ordinal + 1);
            let mut metadata = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(out.join(format!("{label}-metadata.txt")))
                .unwrap();
            writeln!(metadata, "ordinal={ordinal} mode={:?} dimension={} triangle={} not_solved={:?} status={:?} factor_info={:?} solve_info={:?}", item.mode, item.dimension, item.triangle, item.not_solved, item.status, item.factor_info, item.solve_info).unwrap();
            writeln!(metadata, "settings={:?}", item.settings).unwrap();
            writeln!(metadata, "capture_step={:?}", item.capture_step).unwrap();
            vector(
                &out.join(format!("{label}-active-components.txt")),
                &item.active_indices,
            );
            vector(&out.join(format!("{label}-flags.txt")), &item.flags);
            for (field, values) in [
                ("original-matrix-U", &item.matrix),
                ("original-rhs", &item.rhs),
                ("actual-increment", &item.increment),
            ] {
                use crate::issue180_binary_diagnostics::{write_record, Kind};
                let kind = match field {
                    "original-matrix-U" => Kind::Matrix,
                    "original-rhs" => Kind::Rhs,
                    "actual-increment" => Kind::Increment,
                    _ => unreachable!(),
                };
                write_record(
                    &out.join(format!("{label}-{field}.f64bin")),
                    kind,
                    values.len(),
                    values.iter().copied(),
                )
                .unwrap();
            }
        }
        let status = if result.is_ok() { "0\n" } else { "1\n" };
        std::fs::write(out.join("runner.status"), status).unwrap();
        if let Err(error) = &result {
            std::fs::write(out.join("runner.error"), error).unwrap();
        }
        let result = result.unwrap();
        assert_eq!(result.effective_nsteps, 20);
        if mode == "observed" {
            assert!(
                normalized_and_system_inventory_valid(
                    &normalized_steps,
                    &system_steps,
                    declared.modpara.nsrcg == 0
                ),
                "missing, duplicate, or reordered normalized/system inventory"
            );
        } else {
            assert!(normalized_steps.is_empty());
            assert!(system_steps.is_empty());
        }
        assert_eq!(
            records.len(),
            match mode.as_str() {
                "off" => 0,
                "terminal" => 2,
                _ => 22,
            }
        );
        if mode != "off" {
            assert_eq!(records.first().unwrap().label, "initialized");
            assert_eq!(records.last().unwrap().label, "final");
        }
        if mode == "observed" {
            for step in 0..20 {
                assert_eq!(records[step + 1].label, format!("post-sync-{step}"));
            }
        }
        // Written only after complete boundary/normalized/system inventory validation.
        // runner.status=0 by itself is never acquisition completeness evidence.
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(out.join("capture-complete.status"))
            .unwrap()
            .write_all(b"0\n")
            .unwrap();
    }
}

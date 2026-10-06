//! Native Linux C initialization fixtures only; no oracle execution or toolbox reads.
use mvmc_core::{ExpertModeData, SingleProcessReducer};
use num_complex::Complex64;
use serde_json::Value;
use sfmt19937::Sfmt19937Rng;
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf};

const MODELS: [(&str, usize, u32); 13] = [
    ("HeisenbergChain", 14, 1),
    ("HubbardChain", 19, 1),
    ("HubbardTetragonal", 52, 1),
    ("HubbardTetragonal_MomentumProjection", 38, 1),
    ("KondoChain", 76, 1),
    ("HeisenbergChain_cmp", 14, 1),
    ("HubbardChain_cmp", 19, 1),
    ("KondoChain_cmp", 76, 1),
    ("KondoChain_Stot1_cmp", 76, 123456789),
    ("GeneralRBM_cmp", 102, 12395),
    ("HeisenbergChain_fsz", 24, 1),
    ("HubbardChain_fsz", 87, 1),
    ("KondoChain_fsz", 132, 1),
];

fn word(value: &Value) -> u32 {
    let parsed = value.as_u64().expect("C raw unsigned JSON integer");
    u32::try_from(parsed).expect("C u32 domain")
}

fn check_rng(rng: &Sfmt19937Rng, expected: &Value, model: &str) {
    let before = rng.state_snapshot();
    let count = rng.words_consumed();
    let raw = expected["raw624"].as_array().unwrap();
    let future = expected["next624"].as_array().unwrap();
    assert_eq!(raw.len(), 624);
    assert_eq!(future.len(), 624);
    assert_eq!(
        before.0.as_slice(),
        raw.iter().map(word).collect::<Vec<_>>(),
        "{model}"
    );
    assert_eq!(
        before.1 as u64,
        expected["cursor"].as_u64().unwrap(),
        "{model}"
    );
    assert_eq!(
        count,
        expected["word_count"]
            .as_str()
            .unwrap()
            .parse::<u128>()
            .unwrap(),
        "{model}"
    );
    let mut next = [0; 624];
    rng.dump_rand32(&mut next);
    assert_eq!(
        next.as_slice(),
        future.iter().map(word).collect::<Vec<_>>(),
        "{model}"
    );
    assert_eq!(rng.state_snapshot(), before);
    assert_eq!(rng.words_consumed(), count);
}

fn parameters(data: &ExpertModeData) -> Vec<Complex64> {
    data.projection_parameters()
        .into_iter()
        .chain(data.rbm_params.iter().copied())
        .chain(data.slater_params.iter().copied())
        .chain(data.opt_trans.iter().copied())
        .collect()
}

fn check_stage(
    data: &ExpertModeData,
    rng: &Sfmt19937Rng,
    expected: &Value,
    model: &str,
    operation: usize,
) {
    check_rng(rng, expected, model);
    let actual = parameters(data);
    let reference = expected["parameters"].as_array().unwrap();
    assert_eq!(actual.len(), reference.len());
    let projection = data.projection_layout().n_proj;
    let rbm = data.count_rbm_parameters();
    for (index, (a, c)) in actual.iter().zip(reference).enumerate() {
        for (component, value) in [a.re, a.im].into_iter().enumerate() {
            let c: f64 = c[component].as_str().unwrap().parse().unwrap();
            assert!(value.is_finite() && c.is_finite());
            // Empirical, input-specific user-approved absolute policy. The first
            // divergence is C cexp imaginary vs Julia-style sine at identical
            // radius/phase, not an RNG/algorithm difference. No relative waiver.
            let bound: f64 = if model == "GeneralRBM_cmp"
                && operation == 2
                && (projection..projection + rbm).contains(&index)
            {
                1e-18
            } else {
                0.0
            };
            // The C reference values come from glibc (Linux). Linux reproduces them exactly
            // (bound above). The macOS libm `sin`/`cos`/`exp` differ from glibc by at most one
            // ulp (#457), so there the same contract holds to a few ulp of the value; RNG
            // state and draw counts (`check_rng`) stay exact on every platform.
            let bound = if cfg!(target_os = "macos") {
                bound.max(4.0 * f64::EPSILON * c.abs())
            } else {
                bound
            };
            let error = (value - c).abs();
            assert!(
                error <= bound,
                "{model} stage={operation} parameter={index} component={component}: \
                 Rust={value:.17e}, C={c:.17e}, abs={error:.17e}, bound={bound:.17e}"
            );
        }
    }
    let mask = expected["written_mask"].as_array().unwrap();
    assert_eq!(mask.len(), 2 * actual.len());
    let flags = expected["defined_flags"].as_array().unwrap();
    assert_eq!(
        flags.len(),
        mask.iter().filter(|v| v.as_u64() == Some(1)).count()
    );
    let mut seen = vec![false; mask.len()];
    for item in flags {
        let axis = item[0].as_u64().unwrap() as usize;
        assert!(!seen[axis]);
        seen[axis] = true;
        assert_eq!(mask[axis], 1);
        assert_eq!(data.optimization_flags[axis], item[1].as_i64().unwrap());
    }
    for (axis, written) in mask.iter().enumerate() {
        assert_eq!(seen[axis], written.as_u64() == Some(1));
    }
}

#[test]
fn public_initialization_matches_native13_defined_contracts() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/native13_initialization");
    for (model, npara, seed) in MODELS {
        let expected: Value =
            serde_json::from_slice(&fs::read(root.join(format!("{model}.json"))).unwrap()).unwrap();
        assert_eq!(expected["model"], model);
        let stages = expected["stages"].as_array().unwrap();
        assert_eq!(stages.len(), 6);
        for (stage, name) in stages.iter().zip([
            "seeded",
            "workspace-query",
            "random-initialized",
            "loaded-inputs",
            "synchronized",
            "pre-sampling",
        ]) {
            assert_eq!(stage["stage"], name);
        }
        let input = root.join("inputs").join(model).join("namelist.def");
        let directory = input.parent().unwrap();
        let mut expected_names = Vec::new();
        for definition in expected["definitions"].as_array().unwrap() {
            let name = definition["name"].as_str().unwrap();
            assert!(!name.contains('/'));
            expected_names.push(name.to_owned());
            let digest = format!(
                "{:x}",
                Sha256::digest(fs::read(directory.join(name)).unwrap())
            );
            assert_eq!(digest, definition["sha256"].as_str().unwrap());
        }
        let mut actual_names = fs::read_dir(directory)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .filter(|n| n.ends_with(".def"))
            .collect::<Vec<_>>();
        expected_names.sort();
        actual_names.sort();
        assert_eq!(expected_names, actual_names);
        let mut data =
            mvmc_expert_parsers::parse_expert_mode_files_with_c_opt_trans(&input, false).unwrap();
        assert_eq!(data.count_variational_parameters(), npara);
        let segments = stages[2]["segments"].as_array().unwrap();
        assert_eq!(segments.len(), 13);
        let names = [
            "Gutz",
            "Jast",
            "ChargeRBM_PhysLayer",
            "SpinRBM_PhysLayer",
            "GeneralRBM_PhysLayer",
            "ChargeRBM_HiddenLayer",
            "SpinRBM_HiddenLayer",
            "GeneralRBM_HiddenLayer",
            "ChargeRBM_PhysHidden",
            "SpinRBM_PhysHidden",
            "GeneralRBM_PhysHidden",
            "OrbitalAP",
            "OrbitalParallel",
        ];
        let layout = data.projection_layout();
        let rbm_sizes = data.rbm_section_sizes();
        let mut offset = 0usize;
        let mut mask = Vec::new();
        let mut orbital_count = 0;
        for (index, segment) in segments.iter().enumerate() {
            let fields = segment
                .as_str()
                .unwrap()
                .split_whitespace()
                .collect::<Vec<_>>();
            assert_eq!(fields.len(), 6);
            assert_eq!(fields[0], "segment");
            assert_eq!(fields[1], names[index]);
            let start = fields[2]
                .strip_prefix("start=")
                .unwrap()
                .parse::<usize>()
                .unwrap();
            let count = fields[3]
                .strip_prefix("count=")
                .unwrap()
                .parse::<usize>()
                .unwrap();
            let complex = fields[4]
                .strip_prefix("ComplexFlag=")
                .unwrap()
                .parse::<usize>()
                .unwrap();
            let reader = fields[5]
                .strip_prefix("reader=")
                .unwrap()
                .parse::<usize>()
                .unwrap();
            assert_eq!(start, offset);
            assert!(complex <= 1);
            assert_eq!(reader, if index == 12 { 2 } else { 1 });
            if index == 0 {
                assert_eq!(count, layout.n_gutzwiller);
            } else if index == 1 {
                assert_eq!(count, layout.n_jastrow);
            } else if index < 11 {
                assert_eq!(count, rbm_sizes[index - 2]);
            } else {
                orbital_count += count;
            }
            for _ in 0..count {
                mask.push(1u64);
                mask.push(u64::from(complex > 0 || reader == 2));
            }
            offset += count;
        }
        assert_eq!(offset, npara);
        assert_eq!(orbital_count, data.slater_params.len());
        assert_eq!(
            stages[2]["written_mask"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap())
                .collect::<Vec<_>>(),
            mask
        );
        assert_eq!(data.modpara.rnd_seed, i64::from(seed));
        assert_eq!(data.modpara.nsr_opt_itr_step, 20);
        assert_eq!(data.modpara.nsr_opt_itr_smp, 20);
        let mut rng = Sfmt19937Rng::new(seed);
        check_rng(&rng, &stages[0], model);
        // C LAPACK workspace query has no Rust initializer counterpart.
        // Check its recorded RNG identity without inventing a Rust numerical stage.
        check_rng(&rng, &stages[1], model);
        mvmc_expert_parsers::utils::parameter_init::init_parameter(&mut data, &mut rng).unwrap();
        check_stage(&data, &rng, &stages[2], model, 2);
        let initial = input.parent().unwrap().join("initial.def");
        if initial.exists() {
            assert!(mvmc_core::read_initial_def(&mut data, &initial).unwrap());
        } else {
            assert!(matches!(model, "HeisenbergChain" | "HeisenbergChain_cmp"));
        }
        mvmc_expert_parsers::utils::read_input_parameters::read_input_parameters(&mut data, &input)
            .unwrap();
        check_stage(&data, &rng, &stages[3], model, 3);
        mvmc_core::sync::sync_modified_parameter(&mut data, &SingleProcessReducer);
        check_stage(&data, &rng, &stages[4], model, 4);
        mvmc_expert_parsers::utils::qp_weight::init_qp_weight(&mut data);
        check_stage(&data, &rng, &stages[5], model, 5);
    }
}

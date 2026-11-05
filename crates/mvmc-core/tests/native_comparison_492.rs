//! Independent C derivative layout fixture; no C compiler or runtime in Rust tests.
use mvmc_expert_parsers::{ExpertModeData, QuantumProjectionWeights};
use num_complex::Complex64 as C;

#[test]
fn native_rbm_sr_operands_active_components_and_rng_match_c_capture() {
    use mvmc_core::run::{InitialDef, RunConfig};
    use std::path::PathBuf;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures");
    let source = root.join("physcal_181/hubbard_chain_dh_rbm_opttrans/inputs");
    let directory = std::env::temp_dir().join(format!("mvmc-native-492-{}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    for entry in std::fs::read_dir(source).unwrap() {
        let path = entry.unwrap().path();
        if path.is_file() {
            std::fs::copy(&path, directory.join(path.file_name().unwrap())).unwrap();
        }
    }
    let modpara = directory.join("modpara.def");
    let text = std::fs::read_to_string(&modpara).unwrap();
    let text = text
        .lines()
        .map(|line| match line.split_whitespace().next() {
            Some("NVMCCalMode") => "NVMCCalMode 0",
            Some("NVMCSample") => "NVMCSample 1000",
            Some("NLanczosMode") => "NLanczosMode 0",
            _ => line,
        })
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(modpara, text).unwrap();
    let mut config = RunConfig::new(1, "cmp");
    config.nsmp = Some(1);
    config.initial_def = InitialDef::None;
    config.enable_opt_trans = Some(true);
    config.output_dir = Some(directory.join("output"));
    let capture = mvmc_core::sr::observer::capture().unwrap();
    let (_, state, rng) = mvmc_core::run::run_para_opt_from_namelist_observed(
        directory.join("namelist.def"),
        config,
        &mvmc_core::SingleProcessReducer,
    )
    .unwrap();
    let systems = capture.finish();
    let actual = &systems[0];
    let expected: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/native_comparison_492/rbm_sr_system.json"
    ))
    .unwrap();
    let indices: Vec<_> = expected["direct_map"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as usize)
        .collect();
    // C 1.3.0 `GetInfoOptTrans` never flags the OptTrans weights (#676,
    // tmisawa/Julia-mVMC#94); on this input its stray writes land on
    // components that are already active. Rust additionally optimizes the
    // real weight components, which follow C's active set and are excluded
    // from this C comparison.
    let n_c = indices.len();
    assert!(actual.active_indices.len() >= n_c);
    let (common, opttrans) = actual.active_indices.split_at(n_c);
    assert_eq!(common, indices.as_slice());
    let tail = actual.flags.len() - 2 * opttrans.len();
    let expected_opttrans: Vec<_> = (0..opttrans.len()).map(|k| tail + 2 * k).collect();
    assert_eq!(opttrans, expected_opttrans.as_slice());
    assert_eq!(opttrans.len(), 3, "three OptTrans weights are optimized");
    let dimension = actual.dimension;
    assert_eq!(dimension, n_c + opttrans.len());
    // Leading C block of the column-major matrix and the matching RHS rows.
    let matrix: Vec<f64> = (0..n_c)
        .flat_map(|column| actual.matrix[column * dimension..][..n_c].iter().copied())
        .collect();
    let rhs = actual.rhs[..n_c].to_vec();
    for (key, values) in [("direct_S", &matrix), ("direct_g", &rhs)] {
        let expected = expected[key].as_array().unwrap();
        assert_eq!(values.len(), expected.len());
        for (i, (a, e)) in values.iter().zip(expected).enumerate() {
            let e = e.as_f64().unwrap();
            // Measured first divergence: complex moment accumulation <= 3.6e-15.
            // Allow portable BLAS/reduction rounding without masking layout defects.
            assert!(
                (a - e).abs() <= 1e-13 + 1e-13 * a.abs().max(e.abs()),
                "{key}[{i}]: {a} vs {e}"
            );
        }
    }
    let mut peek = rng.clone();
    for line in include_str!("../../../tests/fixtures/native_comparison_492/rbm_state.txt").lines()
    {
        let mut fields = line.split_whitespace();
        let key = fields.next().unwrap();
        let values: Vec<i64> = fields.map(|v| v.parse().unwrap()).collect();
        match key {
            "draws" => assert_eq!(rng.words_consumed(), values[0] as u128),
            "next624" => assert_eq!(
                (0..624)
                    .map(|_| peek.gen_rand32() as i64)
                    .collect::<Vec<_>>(),
                values
            ),
            "ele_idx" => assert_eq!(state.electron_config.ele_idx, values),
            "ele_cfg" => assert_eq!(state.electron_config.ele_cfg, values),
            "ele_num" => assert_eq!(state.electron_config.ele_num, values),
            "ele_spn" => assert_eq!(state.electron_config.ele_spn, values),
            "ele_proj_cnt" => assert_eq!(state.electron_config.ele_proj_cnt, values),
            "counter" => assert_eq!(&state.electron_config.counter[..values.len()], values),
            _ => (),
        }
    }
    std::fs::remove_dir_all(directory).unwrap();
}

// C `calculateOptTransDiff` writes sector `i` into consecutive slot `i`
// (`vmccal.c:639-655`); Rust keeps each sector's (value, i*value) component
// pair (#370, tmisawa/Julia-mVMC#55), the layout that the OptTrans flags
// address (#676, tmisawa/Julia-mVMC#94). The C slot layout is excluded from
// this comparison; the C sector values are compared at Rust's real slots.
#[test]
fn native_opttrans_derivative_values_match_c_in_component_pair_layout() {
    let mut data = ExpertModeData::new();
    data.opt_trans = vec![C::new(1.0, 0.0); 2];
    let mut weights = QuantumProjectionWeights::new();
    weights.qp_fix_weight = vec![C::new(0.5, 0.25), C::new(-0.5, 0.125)];
    data.qp_weights = Some(weights);
    let pf = vec![
        C::new(1.0, 2.0),
        C::new(3.0, -1.0),
        C::new(2.0, -1.0),
        C::new(-1.0, 4.0),
    ];
    let sentinel = C::new(7.0, -9.0);
    let mut actual = vec![sentinel; 6];
    mvmc_core::observables::opt_trans_diff(&mut actual, C::new(1.25, -0.75), &data, &pf);
    let expected: Vec<f64> =
        include_str!("../../../tests/fixtures/native_comparison_492/opttrans_derivative.txt")
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
    assert_eq!(expected.len(), 12);
    let c_sector = |sector: usize| C::new(expected[2 * sector], expected[2 * sector + 1]);
    // The C probe leaves its four trailing slots at the sentinel.
    assert!(expected[4..]
        .chunks(2)
        .all(|v| C::new(v[0], v[1]) == sentinel));
    for sector in 0..2 {
        let e = c_sector(sector);
        for (component, a, e) in [
            (2 * sector, actual[2 * sector], e),
            (2 * sector + 1, actual[2 * sector + 1], C::new(0.0, 1.0) * e),
        ] {
            // Two complex products, a sum and C-compatible complex division.
            for (a, e) in [(a.re, e.re), (a.im, e.im)] {
                assert!(
                    (a - e).abs() <= 1e-14 + 1e-14 * a.abs().max(e.abs()),
                    "slot {component}: {a} vs {e}"
                );
            }
        }
    }
    assert_eq!(&actual[4..], &[sentinel; 2]);
}

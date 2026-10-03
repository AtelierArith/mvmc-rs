mod common;
#[path = "../../../tests/support/historical_optimization_flags.rs"]
mod historical_optimization_flags;
#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;
use common::historical_kernel_model as parse_expert_mode_files;
use mvmc_expert_parsers::utils::parameter_init::all_complex_flag;
use std::path::PathBuf;

#[test]
fn all_nine_rbm_sections_keep_raw_flags_at_historical_mapped_widths() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/rbm");
    for name in [
        "ChargeRBM_PhysLayer",
        "SpinRBM_PhysLayer",
        "GeneralRBM_PhysLayer",
        "ChargeRBM_HiddenLayer",
        "SpinRBM_HiddenLayer",
        "GeneralRBM_HiddenLayer",
        "ChargeRBM_PhysHidden",
        "SpinRBM_PhysHidden",
        "GeneralRBM_PhysHidden",
    ] {
        let data = parse_expert_mode_files(root.join(format!("namelist_{name}.def"))).unwrap();
        let complex = name.starts_with("General");
        assert_eq!(
            data.optimization_flags,
            vec![1, i64::from(complex), 2, 2 * i64::from(complex), 0, 0],
            "{name}"
        );
    }
}

use mvmc_expert_parsers::utils::parameter_init::init_parameter;
use num_complex::Complex64;
use sfmt19937::Sfmt19937Rng;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/rbm")
}
#[test]
fn rbm_layout_and_values_match_julia_with_full_declared_slater_rng_from_c() {
    let fixture = std::fs::read_to_string(root().join("initial.txt")).unwrap();
    let mut lines = fixture.lines().filter(|l| !l.starts_with('#'));
    while let Some(header) = lines.next() {
        let words: Vec<_> = header.split_whitespace().collect();
        let (case, mode) = (words[0], words[1]);
        if matches!(mode, "zero_neuron" | "negative_neuron") || case == "cross_section_flags" {
            // Archived Julia used a divisor-one repair and printed labels
            // could overwrite other sections. C normalization and ordered,
            // bounded flag placement are covered by the native fixtures.
            for _ in 0..4 {
                lines.next().unwrap();
            }
            continue;
        }
        let mut data =
            parse_expert_mode_files(root().join(format!("namelist_{case}.def"))).unwrap();
        match mode {
            "complex" => {
                // C ignores ModPara.ComplexType for AllComplexFlag; make the
                // definition-level projection header complex instead.
                data.gutzwiller_terms
                    .iter_mut()
                    .for_each(|term| term.is_complex = true);
            }
            "missing_flags" => data.optimization_flags.clear(),
            "inactive" => data.optimization_flags.fill(0),
            "truncated" => data.optimization_flags.truncate(2),
            "zero_neuron" | "negative_neuron" => {
                data.modpara.nneuron = if mode == "zero_neuron" { 0 } else { -7 };
                data.modpara.nneuron_charge = 0;
                data.modpara.nneuron_spin = 0;
                data.modpara.nneuron_general = 0;
            }
            "parsed" => (),
            _ => unreachable!(),
        }
        if mode == "complex" && !all_complex_flag(&data) {
            // This historical case used ModPara.ComplexType as a Julia-only
            // mode switch; C leaves it real when all definition headers are
            // real. Consume its archived rows without treating them as C
            // numerical evidence.
            for _ in 0..4 {
                lines.next().unwrap();
            }
            continue;
        }
        let sizes: Vec<usize> = lines
            .next()
            .unwrap()
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        assert_eq!(data.rbm_section_sizes().as_slice(), sizes, "{header}");
        assert_eq!(
            data.count_rbm_parameters(),
            sizes.iter().sum::<usize>(),
            "{header}"
        );
        let flags: Vec<i64> = lines
            .next()
            .unwrap()
            .split_whitespace()
            .map(|s| s.parse::<i64>().unwrap())
            .collect();
        let flags = historical_optimization_flags::c_orbital_representation(&data, flags);
        // Archived Julia initialization records stored positive raw flags as
        // bool. Native component assembly tests assert the full integer values.
        assert_eq!(
            data.optimization_flags
                .iter()
                .map(|&v| i64::from(v > 0))
                .collect::<Vec<_>>(),
            flags,
            "{header}"
        );
        data.visit_rbm_terms_mut(|_, t| t.set_value(Complex64::new(7.0, -9.0)));
        let mut rng = Sfmt19937Rng::new(11272);
        init_parameter(&mut data, &mut rng);
        let mut values = data.projection_parameters();
        data.visit_rbm_terms_mut(|_, t| values.push(t.value()));
        values.extend(
            data.orbital_terms
                .iter()
                .map(|t| data.slater_params[t.idx as usize]),
        );
        let bits: Vec<u64> = lines
            .next()
            .unwrap()
            .split_whitespace()
            .map(|s| u64::from_str_radix(s, 16).unwrap())
            .collect();
        // Radius/phase/trig initialization uses a small sequence of operations;
        // exact SFMT block below independently verifies the draw trajectory.
        numerical_comparison::assert_values_close(
            values.iter().flat_map(|v| [v.re, v.im]),
            bits.into_iter().map(f64::from_bits),
            1e-14,
            1e-14,
            header,
        );
        let state: Vec<u32> = lines
            .next()
            .unwrap()
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        let expected_rng = if data.modpara.n_orbital_idx == 4 {
            common::declared_slater_rng(&data)
        } else {
            state
        };
        assert_eq!(
            (0..624).map(|_| rng.gen_rand32()).collect::<Vec<_>>(),
            expected_rng,
            "{header}"
        );
    }
}

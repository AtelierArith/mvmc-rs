mod common;
use mvmc_expert_parsers::parse_expert_mode_files;
use std::path::PathBuf;

#[test]
fn all_nine_rbm_sections_apply_source_flags_at_mapped_widths() {
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
            vec![true, complex, true, complex, false, false],
            "{name}"
        );
    }
}

use mvmc_expert_parsers::utils::parameter_init::init_parameter;
use mvmc_expert_parsers::{parsers::rbm::*, types::RbmParameter};
use num_complex::Complex64;
use sfmt19937::Sfmt19937Rng;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/rbm")
}
fn render<T: RbmParameter>(r: RbmParseResult<T>, coords: impl Fn(&T) -> Vec<i64>) -> String {
    use std::fmt::Write;
    let mut out = format!(
        "{} {} {} {}\n{}\n",
        u8::from(r.success),
        r.n_rbm_idx,
        u8::from(r.is_complex_flag),
        r.line_number,
        r.error_message
    );
    writeln!(
        &mut out,
        "{}",
        r.opt_flags
            .iter()
            .map(|(idx, flag)| format!("{idx}:{flag}"))
            .collect::<Vec<_>>()
            .join(" ")
    )
    .unwrap();
    if let Some(terms) = r.terms {
        writeln!(&mut out, "{}", terms.len()).unwrap();
        for term in terms {
            writeln!(
                &mut out,
                "{} {:016x} {:016x}",
                coords(&term)
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(" "),
                term.value().re.to_bits(),
                term.value().im.to_bits()
            )
            .unwrap();
        }
    } else {
        writeln!(&mut out, "-1").unwrap();
    }
    out
}

#[test]
fn nine_parsers_match_original_maps_headers_flags_errors_and_line_numbers() {
    let expected = std::fs::read_to_string(root().join("parser.txt")).unwrap();
    let mut actual = String::new();
    let names = [
        "ChargeRBM_PhysLayer",
        "SpinRBM_PhysLayer",
        "GeneralRBM_PhysLayer",
        "ChargeRBM_HiddenLayer",
        "SpinRBM_HiddenLayer",
        "GeneralRBM_HiddenLayer",
        "ChargeRBM_PhysHidden",
        "SpinRBM_PhysHidden",
        "GeneralRBM_PhysHidden",
    ];
    for (section, name) in names.iter().enumerate() {
        let cols = [2, 2, 3, 2, 2, 2, 3, 3, 4][section];
        let mut cases: Vec<_> = std::fs::read_dir(root())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with(&format!("{cols}_")) && n.ends_with(".def"))
            .collect();
        cases.sort();
        for case in cases {
            let content = std::fs::read_to_string(root().join(&case)).unwrap();
            let rendered = match section {
                0 => render(parse_charge_rbm_phys_layer_content(&content), |t| {
                    vec![t.site, t.idx, i64::from(t.is_complex)]
                }),
                1 => render(parse_spin_rbm_phys_layer_content(&content), |t| {
                    vec![t.site, t.idx, i64::from(t.is_complex)]
                }),
                2 => render(parse_general_rbm_phys_layer_content(&content), |t| {
                    vec![t.site, t.spin, t.idx, i64::from(t.is_complex)]
                }),
                3 => render(parse_charge_rbm_hidden_layer_content(&content), |t| {
                    vec![t.site, t.idx, i64::from(t.is_complex)]
                }),
                4 => render(parse_spin_rbm_hidden_layer_content(&content), |t| {
                    vec![t.site, t.idx, i64::from(t.is_complex)]
                }),
                5 => render(parse_general_rbm_hidden_layer_content(&content), |t| {
                    vec![t.site, t.idx, i64::from(t.is_complex)]
                }),
                6 => render(parse_charge_rbm_phys_hidden_content(&content), |t| {
                    vec![t.site1, t.site2, t.idx, i64::from(t.is_complex)]
                }),
                7 => render(parse_spin_rbm_phys_hidden_content(&content), |t| {
                    vec![t.site1, t.site2, t.idx, i64::from(t.is_complex)]
                }),
                8 => render(parse_general_rbm_phys_hidden_content(&content), |t| {
                    vec![t.site1, t.spin, t.site2, t.idx, i64::from(t.is_complex)]
                }),
                _ => unreachable!(),
            };
            actual.push_str(&format!("{name} {case} {rendered}"));
        }
    }
    assert_eq!(
        actual,
        expected
            .lines()
            .filter(|l| !l.starts_with('#'))
            .collect::<Vec<_>>()
            .join("\n")
            + "\n"
    );
}

#[test]
fn rbm_layout_and_values_match_julia_with_full_declared_slater_rng_from_c() {
    let fixture = std::fs::read_to_string(root().join("initial.txt")).unwrap();
    let mut lines = fixture.lines().filter(|l| !l.starts_with('#'));
    while let Some(header) = lines.next() {
        let words: Vec<_> = header.split_whitespace().collect();
        let (case, mode) = (words[0], words[1]);
        let mut data = mvmc_expert_parsers::parse_expert_mode_files(
            root().join(format!("namelist_{case}.def")),
        )
        .unwrap();
        match mode {
            "complex" => data.modpara.complex_flag = 1,
            "missing_flags" => data.optimization_flags.clear(),
            "inactive" => data.optimization_flags.fill(false),
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
        let flags: Vec<bool> = lines
            .next()
            .unwrap()
            .split_whitespace()
            .map(|s| s == "1")
            .collect();
        assert_eq!(data.optimization_flags, flags, "{header}");
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
        let actual: Vec<u64> = values
            .iter()
            .flat_map(|v| [v.re.to_bits(), v.im.to_bits()])
            .collect();
        assert_eq!(actual, bits, "{header}");
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

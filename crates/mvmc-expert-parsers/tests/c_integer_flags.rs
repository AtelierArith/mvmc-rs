//! Complete C header/reader/init expectations without toolbox dependencies.
use mvmc_expert_parsers::{parse_expert_mode_files, utils::parameter_init::init_parameter};
use sfmt19937::Sfmt19937Rng;
use std::fs;

fn integers(line: &str) -> Vec<i64> {
    line.split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect()
}

#[test]
fn projection_and_rbm_component_assembly_preserves_c_generic_raw_flags() {
    use mvmc_expert_parsers::utils::opt_flag::{set_projection_opt_flags, set_rbm_opt_flags};
    use mvmc_expert_parsers::{
        ChargeRBMPhysLayerTerm, ExpertModeData, GutzwillerTerm, JastrowTerm,
    };
    use num_complex::Complex64;
    use std::collections::BTreeMap;
    // DH calls the same actual GetInfoOpt used by G/J/RBM. These checks cover
    // assembly from ordered flag maps, not the remaining G/J/RBM file readers.
    let rows: Vec<_> = include_str!("../../../tests/fixtures/optimization_flags/c_dh_flags.txt")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .collect();
    for record in rows
        .chunks_exact(5)
        .filter(|record| record[0].split_whitespace().nth(1) == Some("2"))
    {
        let complex = record[0].split_whitespace().nth(2) == Some("1");
        let expected = integers(record[3]);
        let flags: BTreeMap<i64, i64> = (0..6).map(|i| (i as i64, expected[4 + 2 * i])).collect();
        for rbm in [false, true] {
            let mut data = ExpertModeData::new();
            data.gutzwiller_terms = (0..2)
                .map(|site| GutzwillerTerm {
                    site,
                    value: Complex64::new(0.0, 0.0),
                    is_complex: false,
                })
                .collect();
            if rbm {
                data.rbm_section_widths[0] = 6;
                data.rbm_params = vec![Complex64::new(0.0, 0.0); 6];
                data.charge_rbm_phys_layer_terms = (0..6)
                    .map(|idx| ChargeRBMPhysLayerTerm {
                        site: idx % 3,
                        idx,
                        value: Complex64::new(0.0, 0.0),
                        is_complex: complex,
                    })
                    .collect();
                data.optimization_flags = vec![0; 16];
                set_rbm_opt_flags(&mut data, &flags, 2, complex);
            } else {
                data.jastrow_terms = (0..6)
                    .map(|_| JastrowTerm {
                        site1: 0,
                        site2: 1,
                        value: Complex64::new(0.0, 0.0),
                        is_complex: complex,
                    })
                    .collect();
                data.optimization_flags = vec![0; 16];
                set_projection_opt_flags(&mut data, &BTreeMap::new(), &flags, false, complex);
            }
            assert_eq!(data.optimization_flags, expected, "{} rbm={rbm}", record[0]);
        }
    }
}

#[test]
fn dh_row_order_raw_flags_and_component_offsets_match_native_c() {
    use mvmc_expert_parsers::parsers::doublon_holon::{
        parse_doublon_holon_2site_content, parse_doublon_holon_4site_content,
    };
    use mvmc_expert_parsers::utils::opt_flag::set_dh_opt_flags;
    use mvmc_expert_parsers::{ExpertModeData, GutzwillerTerm};
    use num_complex::Complex64;
    let rows: Vec<_> = include_str!("../../../tests/fixtures/optimization_flags/c_dh_flags.txt")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .collect();
    assert_eq!(rows.len(), 24 * 5);
    for record in rows.chunks_exact(5) {
        let header: Vec<_> = record[0].split_whitespace().collect();
        let family = header[1];
        let complex = header[2] == "1";
        let payload = format!(
            "====\nCount 1\nComplexType {}\n====\n====\n{}",
            header[2],
            record[1].replace('|', "\n")
        );
        let mut data = ExpertModeData::new();
        data.modpara.nsite = 3;
        data.gutzwiller_terms = (0..2)
            .map(|site| GutzwillerTerm {
                site,
                value: Complex64::new(0.0, 0.0),
                is_complex: false,
            })
            .collect();
        let (flags, neighbors) = if family == "2" {
            let parsed = parse_doublon_holon_2site_content(&payload, 3);
            assert!(
                parsed.is_success(),
                "{}: {}",
                header[0],
                parsed.error_message
            );
            let definition = parsed.data.unwrap();
            data.doublon_holon_2site_indices = definition.indices.clone();
            data.doublon_holon_2site_params = vec![Complex64::new(0.0, 0.0); 6];
            data.doublon_holon_2site_opt_flags = definition.opt_flags.clone();
            data.doublon_holon_2site_complex = complex;
            (
                definition.opt_flags,
                definition.indices[0]
                    .neighbors
                    .iter()
                    .flatten()
                    .copied()
                    .collect::<Vec<_>>(),
            )
        } else {
            let parsed = parse_doublon_holon_4site_content(&payload, 3);
            assert!(
                parsed.is_success(),
                "{}: {}",
                header[0],
                parsed.error_message
            );
            let definition = parsed.data.unwrap();
            data.doublon_holon_4site_indices = definition.indices.clone();
            data.doublon_holon_4site_params = vec![Complex64::new(0.0, 0.0); 10];
            data.doublon_holon_4site_opt_flags = definition.opt_flags.clone();
            data.doublon_holon_4site_complex = complex;
            (
                definition.opt_flags,
                definition.indices[0]
                    .neighbors
                    .iter()
                    .flatten()
                    .copied()
                    .collect::<Vec<_>>(),
            )
        };
        assert_eq!(integers(record[2]), [0, flags.len() as i64]);
        data.optimization_flags = vec![0; 2 * data.projection_layout().n_proj];
        set_dh_opt_flags(&mut data);
        assert_eq!(
            data.optimization_flags,
            integers(record[3]),
            "{}",
            header[0]
        );
        assert_eq!(neighbors, integers(record[4]), "{}", header[0]);
    }
}

#[test]
fn correlation_gauge_eligibility_matches_c_set_flag_shift() {
    use mvmc_expert_parsers::utils::parameter_init::sync_modified_parameter;
    use mvmc_expert_parsers::{ExpertModeData, GutzwillerTerm, JastrowTerm};
    use num_complex::Complex64;
    let rows: Vec<_> = include_str!("../../../tests/fixtures/optimization_flags/c_shift_flags.txt")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .collect();
    assert_eq!(rows.len(), 86 * 3);
    for record in rows.chunks_exact(3) {
        let header: Vec<_> = record[0].split_whitespace().collect();
        let counts: Vec<usize> = header[1..].iter().map(|s| s.parse().unwrap()).collect();
        let gates = integers(record[2]);
        let mut data = ExpertModeData::new();
        data.modpara.nsite = 3;
        data.doublon_holon_2site_indices = vec![
            mvmc_expert_parsers::DoublonHolon2SiteIndex {
                neighbors: vec![[1, 2]; 3]
            };
            counts[2]
        ];
        data.doublon_holon_4site_indices = vec![
            mvmc_expert_parsers::DoublonHolon4SiteIndex {
                neighbors: vec![[1, 2, 1, 2]; 3]
            };
            counts[3]
        ];
        data.gutzwiller_terms = (0..counts[0])
            .map(|i| GutzwillerTerm {
                site: i as i64,
                value: Complex64::new((i + 1) as f64, 7.0),
                is_complex: true,
            })
            .collect();
        data.jastrow_terms = (0..counts[1])
            .map(|i| JastrowTerm {
                site1: 0,
                site2: i as i64 + 1,
                value: Complex64::new((i + 3) as f64, 7.0),
                is_complex: true,
            })
            .collect();
        data.doublon_holon_2site_params = (0..6 * counts[2])
            .map(|i| Complex64::new((i + 10) as f64, 7.0))
            .collect();
        data.doublon_holon_4site_params = (0..10 * counts[3])
            .map(|i| Complex64::new((i + 30) as f64, 7.0))
            .collect();
        data.optimization_flags = integers(record[1]);
        let before = data.clone();
        sync_modified_parameter(&mut data, true);
        for (changed, gate) in [
            (data.jastrow_terms != before.jastrow_terms, gates[0]),
            (
                data.doublon_holon_2site_params != before.doublon_holon_2site_params,
                gates[1],
            ),
            (
                data.doublon_holon_4site_params != before.doublon_holon_4site_params,
                gates[2],
            ),
        ] {
            assert_eq!(changed, gate == 1, "{}", header[0]);
        }
        assert_eq!(data.optimization_flags, before.optimization_flags);
        assert!(data
            .projection_parameters()
            .iter()
            .all(|value| value.im == 7.0));
        if gates == [0, 0, 0] {
            assert_eq!(data.projection_parameters(), before.projection_parameters());
        }
    }
}

#[test]
fn integer_component_flags_header_normalization_coefficients_and_rng_match_c() {
    let rows: Vec<_> =
        include_str!("../../../tests/fixtures/optimization_flags/c_integer_flags.txt")
            .lines()
            .filter(|line| !line.starts_with('#'))
            .collect();
    assert_eq!(rows.len(), 70 * 7);
    let dir = std::env::temp_dir().join(format!("mvmc-c-integer-flags-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let mut failures = Vec::new();
    for record in rows.chunks_exact(7) {
        let header: Vec<_> = record[0].split_whitespace().collect();
        let seed = header[1].parse().unwrap();
        fs::write(dir.join("modpara.def"), "Nsite 2\nNElec 1\nNMPTrans -1\n").unwrap();
        fs::write(dir.join("ap.def"), record[1].replace('|', "\n")).unwrap();
        let mut namelist = "ModPara modpara.def\nOrbitalAntiParallel ap.def\n".to_owned();
        if !record[2].is_empty() {
            fs::write(dir.join("p.def"), record[2].replace('|', "\n")).unwrap();
            namelist.push_str("OrbitalParallel p.def\n");
        }
        fs::write(dir.join("namelist.def"), namelist).unwrap();
        let mut data = parse_expert_mode_files(dir.join("namelist.def")).unwrap();
        assert!(data.input_errors.is_empty(), "{:?}", data.input_errors);
        let expected_flags: Vec<i64> = record[3]
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        let actual_flags: Vec<i64> = data.optimization_flags.clone();
        if actual_flags != expected_flags {
            failures.push(format!(
                "{} raw flags: Rust={actual_flags:?}, C={expected_flags:?}",
                header[0]
            ));
        }
        let mut rng = Sfmt19937Rng::new(seed);
        init_parameter(&mut data, &mut rng);
        let expected_bits: Vec<u64> = record[4]
            .split_whitespace()
            .map(|s| u64::from_str_radix(s, 16).unwrap())
            .collect();
        let actual_bits: Vec<u64> = data
            .slater_params
            .iter()
            .flat_map(|z| [z.re.to_bits(), z.im.to_bits()])
            .collect();
        if actual_bits != expected_bits {
            failures.push(format!("{} initial coefficient bits differ", header[0]));
        }
        let words: Vec<u32> = record[5]
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        assert_eq!(words.len(), 624);
        let actual_words: Vec<_> = (0..624).map(|_| rng.gen_rand32()).collect();
        if actual_words != words {
            failures.push(format!("{} subsequent SFMT block differs", header[0]));
        }
        let selected: Vec<_> = actual_flags
            .iter()
            .enumerate()
            .filter_map(|(i, &flag)| (flag == 1).then_some(i))
            .collect();
        let expected_selected: Vec<usize> = record[6]
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        if selected != expected_selected {
            failures.push(format!("{} SR selection differs", header[0]));
        }
    }
    fs::remove_dir_all(dir).unwrap();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

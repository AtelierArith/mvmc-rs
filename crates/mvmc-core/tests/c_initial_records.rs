//! Native C ReadInitParameter results, without invoking the optional C toolbox.
use mvmc_core::{read_initial_def, read_opt_para_file, ExpertModeData};
use mvmc_expert_parsers::*;
use num_complex::Complex64;
use sfmt19937::Sfmt19937Rng;

#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;

fn model(dims: &[usize]) -> ExpertModeData {
    let mut data = ExpertModeData::new();
    let sentinel = Complex64::new(99.0, 99.0);
    if dims[0] != 0 {
        assert_eq!(dims[0], 37);
        data.n_gutzwiller_idx = 2;
        data.n_jastrow_idx = 3;
        data.gutzwiller_terms = (0..2)
            .map(|site| GutzwillerTerm {
                site,
                value: sentinel,
                is_complex: true,
            })
            .collect();
        data.jastrow_terms = (0..3)
            .map(|site1| JastrowTerm {
                site1,
                site2: (site1 + 1) % 3,
                value: sentinel,
                is_complex: true,
            })
            .collect();
        data.doublon_holon_2site_indices = vec![
            DoublonHolon2SiteIndex {
                neighbors: vec![[1, 2]; 3]
            };
            2
        ];
        data.doublon_holon_4site_indices = vec![
            DoublonHolon4SiteIndex {
                neighbors: vec![[1, 2, 1, 2]; 3]
            };
            2
        ];
        data.doublon_holon_2site_params = vec![sentinel; 12];
        data.doublon_holon_4site_params = vec![sentinel; 20];
    }
    if dims[1] != 0 {
        assert_eq!(dims[1], 27);
        data.rbm_section_widths = [3; 9];
        data.rbm_params = vec![sentinel; 27];
        for idx in 0..3 {
            data.charge_rbm_phys_layer_terms
                .push(ChargeRBMPhysLayerTerm {
                    site: idx,
                    idx,
                    value: sentinel,
                    is_complex: true,
                });
            data.spin_rbm_phys_layer_terms.push(SpinRBMPhysLayerTerm {
                site: idx,
                idx,
                value: sentinel,
                is_complex: true,
            });
            data.general_rbm_phys_layer_terms
                .push(GeneralRBMPhysLayerTerm {
                    site: idx,
                    spin: 0,
                    idx,
                    value: sentinel,
                    is_complex: true,
                });
            data.charge_rbm_hidden_layer_terms
                .push(ChargeRBMHiddenLayerTerm {
                    site: idx,
                    idx,
                    value: sentinel,
                    is_complex: true,
                });
            data.spin_rbm_hidden_layer_terms
                .push(SpinRBMHiddenLayerTerm {
                    site: idx,
                    idx,
                    value: sentinel,
                    is_complex: true,
                });
            data.general_rbm_hidden_layer_terms
                .push(GeneralRBMHiddenLayerTerm {
                    site: idx,
                    idx,
                    value: sentinel,
                    is_complex: true,
                });
            data.charge_rbm_phys_hidden_terms
                .push(ChargeRBMPhysHiddenTerm {
                    site1: idx,
                    site2: idx,
                    idx,
                    value: sentinel,
                    is_complex: true,
                });
            data.spin_rbm_phys_hidden_terms.push(SpinRBMPhysHiddenTerm {
                site1: idx,
                site2: idx,
                idx,
                value: sentinel,
                is_complex: true,
            });
            data.general_rbm_phys_hidden_terms
                .push(GeneralRBMPhysHiddenTerm {
                    site1: idx,
                    site2: idx,
                    spin: 0,
                    idx,
                    value: sentinel,
                    is_complex: true,
                });
        }
    }
    data.modpara.n_orbital_idx = dims[2] as i64;
    data.slater_params = vec![sentinel; dims[2]];
    if dims[2] > 0 {
        data.orbital_terms = [0, 0, (dims[2] - 2) as i64, (dims[2] - 1) as i64]
            .into_iter()
            .map(|idx| OrbitalTerm {
                site1: 0,
                site2: 1,
                idx,
                sign: 1,
                is_complex: true,
            })
            .collect();
    }
    data.opt_trans = vec![sentinel; dims[3]];
    assert_eq!(data.projection_layout().n_proj, dims[0]);
    assert_eq!(data.count_rbm_parameters(), dims[1]);
    assert_eq!(data.count_opt_trans_parameters(), dims[3]);
    data
}

// Reader-only scalar conversion/scattering: no floating-point calculation.
// Exact storage also verifies unchanged reserved/sentinel slots.
fn parameter_bits(data: &ExpertModeData) -> Vec<u64> {
    data.projection_parameters()
        .into_iter()
        .chain(data.rbm_parameters())
        .chain(data.slater_params.iter().copied())
        .chain(data.opt_trans.iter().copied())
        .flat_map(|value| [value.re.to_bits(), value.im.to_bits()])
        .collect()
}

#[test]
fn complete_c_records_and_scalar_conversions_preserve_final_values_and_rng() {
    let rows: Vec<_> = include_str!("../../../tests/fixtures/initial_records/c_records.txt")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .collect();
    assert_eq!(rows.len(), 140);
    for record in rows.as_chunks::<4>().0.iter() {
        let fields: Vec<_> = record[0].split_whitespace().collect();
        let dims: Vec<usize> = fields[1..].iter().map(|s| s.parse().unwrap()).collect();
        let expected: Vec<u64> = record[2]
            .split_whitespace()
            .map(|s| u64::from_str_radix(s, 16).unwrap())
            .collect();
        let words: Vec<u32> = record[3]
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        assert_eq!(words.len(), 624);
        let path = std::env::temp_dir().join(format!(
            "mvmc-c-records-{}-{}.def",
            std::process::id(),
            fields[0]
        ));
        std::fs::write(&path, record[1]).unwrap();
        for optional in [false, true] {
            let mut data = model(&dims);
            let mut rng = Sfmt19937Rng::new(1);
            let mappings = data.orbital_terms.clone();
            if optional {
                assert!(
                    read_initial_def(&mut data, &path).unwrap(),
                    "{} optional",
                    record[0]
                );
            } else {
                let consumed = if record[1].is_empty() {
                    0
                } else {
                    dims.iter().sum()
                };
                assert_eq!(
                    read_opt_para_file(&mut data, &path).unwrap(),
                    consumed,
                    "{} strict",
                    record[0]
                );
            }
            let actual = parameter_bits(&data);
            assert_eq!(actual.len(), expected.len());
            for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    numerical_comparison::arithmetic_bits_match(actual, expected),
                    "{} optional={optional} component {index}: {actual:016x} != {expected:016x}",
                    record[0]
                );
            }
            assert_eq!(data.orbital_terms, mappings);
            assert_eq!(
                (0..624).map(|_| rng.gen_rand32()).collect::<Vec<_>>(),
                words
            );
        }
        std::fs::remove_file(path).unwrap();
    }
}

#[test]
fn successive_complete_records_leave_the_last_c_record_applied() {
    let mut data = model(&[0, 0, 2, 0]);
    let record = |base: f64| {
        let mut values = vec![0.0; 12];
        values[6] = base;
        values[9] = base + 1.0;
        values
            .into_iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    };
    let path = std::env::temp_dir().join(format!(
        "mvmc-c-successive-records-{}.def",
        std::process::id()
    ));
    std::fs::write(&path, format!("{}\n{}\n", record(1.0), record(7.0))).unwrap();
    assert_eq!(read_opt_para_file(&mut data, &path).unwrap(), 2);
    assert_eq!(data.slater_params[0], Complex64::new(7.0, 0.0));
    assert_eq!(data.slater_params[1], Complex64::new(8.0, 0.0));
    std::fs::remove_file(path).unwrap();
}

//! Canonical DH4 combinatorial kernels, with exhaustive original-source fixtures.
#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;
use mvmc_core::observables::set_projection_diff;
use mvmc_core::sampling::projection::{
    log_proj_ratio, log_proj_val, make_proj_cnt, update_proj_cnt,
};
use mvmc_core::{read_initial_def, read_opt_para_file, VmcOptimizationState};
use mvmc_expert_parsers::utils::parameter_init::sync_modified_parameter;
use mvmc_expert_parsers::utils::read_input_parameters::read_input_parameters;
use mvmc_expert_parsers::OrbitalTerm;
use mvmc_expert_parsers::{
    DoublonHolon2SiteIndex, DoublonHolon4SiteIndex, ExpertModeData, GutzwillerTerm, JastrowTerm,
};
use num_complex::Complex64;
use std::{fs, path::PathBuf};

fn model_with_dh2(combined: bool) -> ExpertModeData {
    let mut d = ExpertModeData::new();
    d.modpara.nsite = 4;
    d.n_gutzwiller_idx = 3;
    d.gutzwiller_idx = vec![0, 1, 0, 1];
    d.gutzwiller_terms = (0..2)
        .map(|i| GutzwillerTerm {
            site: i,
            value: Complex64::new((i + 1) as f64 / 8.0, -(i + 1) as f64 / 16.0),
            is_complex: true,
        })
        .collect();
    d.n_jastrow_idx = 4;
    d.jastrow_idx = vec![
        vec![-1, 0, 1, 2],
        vec![0, -1, 2, 1],
        vec![1, 2, -1, 0],
        vec![2, 1, 0, -1],
    ];
    d.jastrow_terms = (0..3)
        .map(|i| JastrowTerm {
            site1: 0,
            site2: i + 1,
            value: Complex64::new((i + 1) as f64 / 16.0, -(i + 1) as f64 / 32.0),
            is_complex: true,
        })
        .collect();
    d.doublon_holon_4site_indices = vec![
        DoublonHolon4SiteIndex {
            neighbors: vec![[1, 2, 3, 0], [0, 2, 3, 1], [0, 1, 3, 2], [0, 1, 2, 3]],
        },
        DoublonHolon4SiteIndex {
            neighbors: vec![[1, 1, 1, 1], [1, 1, 3, 3], [3, 3, 0, 0], [2, 2, 2, 2]],
        },
    ];
    d.doublon_holon_4site_params = (1..=20)
        .map(|i| Complex64::new(i as f64 / 10.0, -(i as f64) / 13.0))
        .collect();
    if combined {
        d.doublon_holon_2site_indices = vec![
            DoublonHolon2SiteIndex {
                neighbors: vec![[1, 2], [0, 2], [0, 1], [0, 1]],
            },
            DoublonHolon2SiteIndex {
                neighbors: vec![[0, 0], [0, 0], [3, 3], [2, 2]],
            },
        ];
        d.doublon_holon_2site_params = (1..=12)
            .map(|i| Complex64::new(i as f64 / 8.0, -(i as f64) / 16.0))
            .collect();
    }
    d.optimization_flags = vec![1; 2 * d.projection_layout().n_proj];
    d
}
fn source_fixture(name: &str, combined: bool) -> String {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/dh4");
    fs::read_to_string(
        root.join(if combined { "combined" } else { "kernels" })
            .join(name),
    )
    .unwrap()
}
fn occupancy(mask: usize) -> Vec<i64> {
    (0..8).map(|i| ((mask >> i) & 1) as i64).collect()
}
fn bits(text: &str) -> Vec<u64> {
    text.split_whitespace()
        .map(|s| u64::from_str_radix(s, 16).unwrap())
        .collect()
}
fn source_counts(combined: bool) -> (Vec<Vec<i64>>, Vec<u64>, Vec<Vec<u64>>) {
    let fixture = source_fixture("counts.txt", combined);
    let mut lines = fixture.lines().skip(1);
    let mut counts = Vec::new();
    let mut logs = Vec::new();
    let mut derivatives = Vec::new();
    for mask in 0..256 {
        let row: Vec<i64> = lines
            .next()
            .unwrap()
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        assert_eq!(row[0], mask);
        counts.push(row[1..].to_vec());
        logs.push(bits(lines.next().unwrap())[0]);
        derivatives.push(bits(lines.next().unwrap()));
    }
    assert!(lines.next().is_none());
    (counts, logs, derivatives)
}
fn check_every_occupation_matches_julia_dense_dh_counts_and_derivatives(combined: bool) {
    let d = model_with_dh2(combined);
    let (expected, logs, derivatives) = source_counts(combined);
    for mask in 0..256 {
        let mut actual = vec![99; d.projection_layout().n_proj];
        make_proj_cnt(&mut actual, &occupancy(mask), &d);
        assert_eq!(actual, expected[mask], "mask {mask}");
        // A projection log is a short parameter/count dot product.
        numerical_comparison::assert_close(
            log_proj_val(&actual, &d),
            f64::from_bits(logs[mask]),
            64.0 * f64::EPSILON,
            64.0 * f64::EPSILON,
            format!("mask {mask} log"),
        );
        let mut diff = vec![Complex64::new(99.0, 99.0); 2 * (actual.len() + 1)];
        set_projection_diff(&mut diff, &actual, actual.len());
        numerical_comparison::assert_values_close(
            diff.iter().flat_map(|z| [z.re, z.im]),
            derivatives[mask].iter().copied().map(f64::from_bits),
            4.0 * f64::EPSILON,
            4.0 * f64::EPSILON,
            format!("mask {mask} derivatives"),
        );
    }
}
fn check_every_allowed_normal_and_spin_changing_hop_matches_julia_fresh_counts(combined: bool) {
    let d = model_with_dh2(combined);
    let (counts, _, _) = source_counts(combined);
    for line in source_fixture("moves.txt", combined).lines().skip(1) {
        let row: Vec<_> = line.split_whitespace().collect();
        let n: Vec<usize> = row[..6].iter().map(|s| s.parse().unwrap()).collect();
        let (mask, ri, s, rj, next) = (n[0], n[1], n[2], n[3], n[5]);
        let mut actual = vec![99; counts[mask].len()];
        // Both source update APIs use site occupancies; spins affect the supplied configuration.
        update_proj_cnt(
            ri as i64,
            rj as i64,
            s as u8,
            &mut actual,
            &counts[mask],
            &occupancy(next),
            &d,
        );
        assert_eq!(actual, counts[next], "move {line}");
        numerical_comparison::assert_close(
            log_proj_ratio(&actual, &counts[mask], &d),
            f64::from_bits(u64::from_str_radix(row[6], 16).unwrap()),
            64.0 * f64::EPSILON,
            64.0 * f64::EPSILON,
            format!("move {line} log ratio"),
        );
    }
}
fn check_dh_gauge_matches_julia_declared_flags_compensation_and_shift_order(combined: bool) {
    let fixture = source_fixture("gauge.txt", combined);
    let mut lines = fixture.lines().skip(1);
    while let Some(header) = lines.next() {
        let name = header.split_whitespace().next().unwrap();
        let mut d = model_with_dh2(combined);
        let layout = d.projection_layout();
        match name {
            "all" | "disabled" => {}
            "fixed_dh2" => {
                if combined {
                    d.optimization_flags[2 * layout.dh2_offset] = 0;
                }
            }
            "fixed_gutz" => d.optimization_flags[4] = 0,
            "fixed_dh" => d.optimization_flags[2 * (layout.dh4_offset + 19)] = 0,
            "empty_flags" => d.optimization_flags.clear(),
            "short_flags" => d.optimization_flags.truncate(15),
            "partial_params" => d.doublon_holon_4site_params.truncate(18),
            "no_gutz" => {
                d.n_gutzwiller_idx = 0;
                d.gutzwiller_terms.clear();
            }
            "no_jast" => {
                d.n_jastrow_idx = 0;
                d.jastrow_terms.clear();
            }
            "no_dh" => {
                d.doublon_holon_4site_indices.clear();
                d.doublon_holon_4site_params.clear();
            }
            "cancellation" => {
                for (idx, value) in [0, 4, 8, 12, 16]
                    .into_iter()
                    .zip([1e16, 1.0, -1e16, 1.0, 1.0])
                {
                    d.doublon_holon_4site_params[idx] = Complex64::new(value, -value / 13.0);
                }
            }
            _ => panic!("unknown {name}"),
        }
        d.modpara.n_orbital_idx = 2;
        d.slater_params = vec![Complex64::new(2.0, 0.0), Complex64::new(0.0, 8.0)];
        d.orbital_terms = vec![
            OrbitalTerm {
                site1: 0,
                site2: 0,
                idx: 0,
                is_complex: true,
                sign: 1,
            },
            OrbitalTerm {
                site1: 0,
                site2: 1,
                idx: 1,
                is_complex: true,
                sign: 1,
            },
        ];
        sync_modified_parameter(&mut d, name != "disabled");
        let values = d
            .gutzwiller_terms
            .iter()
            .map(|t| t.value)
            .chain(d.jastrow_terms.iter().map(|t| t.value))
            .chain(d.doublon_holon_2site_params.iter().copied())
            .chain(d.doublon_holon_4site_params.iter().copied())
            .chain(
                d.orbital_terms
                    .iter()
                    .map(|t| d.slater_params[t.idx as usize]),
            );
        // Gauge shift/normalization includes short reductions, exp and sqrt.
        numerical_comparison::assert_values_close(
            values.flat_map(|z| [z.re, z.im]),
            bits(lines.next().unwrap()).into_iter().map(f64::from_bits),
            128.0 * f64::EPSILON,
            128.0 * f64::EPSILON,
            name,
        );
    }
}

fn directory(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mvmc-dh4-{}-{name}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    dir
}
fn check_initial_and_fixed_loaders_apply_dh_triples_between_reserved_jastrow_and_slater(
    combined: bool,
) {
    let mut d = model_with_dh2(combined);
    d.modpara.n_orbital_idx = 3;
    d.slater_params = vec![Complex64::new(99.0, 99.0); 3];
    d.orbital_terms = vec![OrbitalTerm {
        site1: 0,
        site2: 1,
        idx: 2,

        is_complex: true,
        sign: 1,
    }];
    let n = d.projection_layout().n_proj + 3;
    let text = format!(
        "0 0 0 0 0 0 {}",
        (0..n)
            .map(|i| format!("{} {} 99", i as f64 / 8.0, -(i as f64) / 16.0))
            .collect::<Vec<_>>()
            .join(" ")
    );
    let dir = directory(if combined {
        "combined-triples"
    } else {
        "triples"
    });
    let path = dir.join("initial.def");
    fs::write(&path, &text).unwrap();
    assert_eq!(read_opt_para_file(&mut d, &path).unwrap(), n);
    for (i, z) in d.doublon_holon_4site_params.iter().enumerate() {
        let index = d.projection_layout().dh4_offset + i;
        assert_eq!(
            *z,
            Complex64::new(index as f64 / 8.0, -(index as f64) / 16.0)
        );
    }
    assert_eq!(
        d.slater_params[d.orbital_terms[0].idx as usize],
        Complex64::new((n - 1) as f64 / 8.0, -((n - 1) as f64) / 16.0)
    );
    assert!(read_initial_def(&mut d, &path).unwrap());
    let before = d.projection_parameters();
    for bad in [
        format!("{text} 1 2 3"),
        text.replacen("99", "broken", 1),
        "0 0 0 0 0 0".into(),
    ] {
        fs::write(&path, bad).unwrap();
        assert!(read_opt_para_file(&mut d, &path).is_err());
        assert!(!read_initial_def(&mut d, &path).unwrap());
        assert_eq!(d.projection_parameters(), before);
    }
    fs::remove_dir_all(dir).unwrap();
}
fn check_strict_indh4_overlay_uses_definition_count_and_commits_each_record_atomically(
    combined: bool,
) {
    let mut d = model_with_dh2(combined);
    let dir = directory(if combined {
        "combined-overlay"
    } else {
        "overlay"
    });
    let nml = dir.join("namelist.def");
    let path = dir.join("indh4.def");
    fs::write(&nml, "InDH4 absent.def\nInDH4 indh4.def\n").unwrap();
    let before = d.projection_parameters();
    assert!(read_input_parameters(&mut d, &nml)
        .unwrap_err()
        .contains("duplicate keyword InDH4"));
    assert_eq!(d.projection_parameters(), before);
    fs::write(&nml, "InDH4 indh4.def\n").unwrap();
    let text = format!(
        "===\nNDoublonHolon4siteIdx 2\nComplexType 1\n===\n===\n{}\n",
        (0..20)
            .rev()
            .map(|i| format!("{i} {} {}", i as f64 / 8.0, -(i as f64) / 16.0))
            .collect::<Vec<_>>()
            .join("\n")
    );
    fs::write(&path, &text).unwrap();
    read_input_parameters(&mut d, &nml).unwrap();
    assert_eq!(
        d.doublon_holon_4site_params,
        (0..20)
            .map(|i| Complex64::new(i as f64 / 8.0, -(i as f64) / 16.0))
            .collect::<Vec<_>>()
    );
    let before = d.projection_parameters();
    for bad in [
        text.replace("Idx 2", "Idx 20"),
        text.replace("19 2.375 -1.1875", "18 2.375 -1.1875"),
        text.replace("19 2.375 -1.1875", "20 2.375 -1.1875"),
        text.replace("2.375", "NaN"),
        text.replace("19 2.375 -1.1875\n", ""),
    ] {
        fs::write(&path, bad).unwrap();
        assert!(read_input_parameters(&mut d, &nml).is_err());
        assert_eq!(d.projection_parameters(), before);
    }
    fs::write(&path, &text).unwrap();
    d.doublon_holon_4site_params.pop();
    assert!(read_input_parameters(&mut d, &nml)
        .unwrap_err()
        .contains("target parameter length mismatch"));
    fs::remove_dir_all(dir).unwrap();
}
fn check_direct_sr_updates_each_dh_component_without_writing_other_projection_slots(
    combined: bool,
) {
    let fixture = source_fixture("sr-writeback.txt", combined);
    let mut reference = fixture.lines().skip(1);
    for complex in [false, true] {
        for local in 0..20 {
            for imaginary in 0..=usize::from(complex) {
                let mut d = model_with_dh2(combined);
                let layout = d.projection_layout();
                let target = layout.dh4_offset + local;
                d.modpara.dsr_opt_red_cut = 0.0;
                d.modpara.dsr_opt_sta_del = 0.0;
                d.modpara.dsr_opt_step_dt = 0.25;
                d.optimization_flags = vec![0; 2 * layout.n_proj];
                d.optimization_flags[2 * target + imaginary] = 1;
                let before = d.projection_parameters();
                let mut state = VmcOptimizationState::zeros(
                    4,
                    1,
                    layout.n_proj,
                    layout.n_proj,
                    1,
                    1,
                    complex,
                    false,
                );
                let result = if complex {
                    let component = 2 * target + imaginary + 2;
                    let dim = 2 * state.sr_opt.sr_opt_size;
                    state.sr_opt.sr_opt_oo[component * dim + component] = Complex64::new(2.0, 0.0);
                    state.sr_opt.sr_opt_ho[component] = Complex64::new(3.0, 0.0);
                    mvmc_core::sr::stochastic_opt_complex(&mut d, &mut state)
                } else {
                    let component = target + 1;
                    let dim = state.sr_opt.sr_opt_size;
                    state.sr_opt.sr_opt_oo_real[component * dim + component] = 2.0;
                    state.sr_opt.sr_opt_ho_real[component] = 3.0;
                    mvmc_core::sr::stochastic_opt_real(&mut d, &mut state)
                };
                assert_eq!(result, 0);
                assert_eq!(
                    reference.next().unwrap(),
                    format!("{} {local} {imaginary}", usize::from(complex))
                );
                let expected = bits(reference.next().unwrap());
                let actual: Vec<f64> = d
                    .projection_parameters()
                    .iter()
                    .flat_map(|v| [v.re, v.im])
                    .collect();
                // This SR input is diagonal with one active component (condition 1).
                numerical_comparison::assert_values_close(
                    actual,
                    expected.into_iter().map(f64::from_bits),
                    16.0 * f64::EPSILON,
                    16.0 * f64::EPSILON,
                    format!("complex={complex} DH slot {local} component {imaginary}"),
                );
                for (i, (old, new)) in before.iter().zip(d.projection_parameters()).enumerate() {
                    if i != target {
                        assert_eq!(*old, new);
                    }
                }
            }
        }
    }
}

#[test]
fn every_occupation_matches_julia_dense_dh_counts_and_derivatives() {
    for combined in [false, true] {
        check_every_occupation_matches_julia_dense_dh_counts_and_derivatives(combined);
    }
}

#[test]
fn every_allowed_normal_and_spin_changing_hop_matches_julia_fresh_counts() {
    for combined in [false, true] {
        check_every_allowed_normal_and_spin_changing_hop_matches_julia_fresh_counts(combined);
    }
}

#[test]
fn dh_gauge_matches_julia_declared_flags_compensation_and_shift_order() {
    for combined in [false, true] {
        check_dh_gauge_matches_julia_declared_flags_compensation_and_shift_order(combined);
    }
}

#[test]
fn initial_and_fixed_loaders_apply_dh_triples_between_reserved_jastrow_and_slater() {
    for combined in [false, true] {
        check_initial_and_fixed_loaders_apply_dh_triples_between_reserved_jastrow_and_slater(
            combined,
        );
    }
}

#[test]
fn strict_indh4_overlay_uses_definition_count_and_commits_each_record_atomically() {
    for combined in [false, true] {
        check_strict_indh4_overlay_uses_definition_count_and_commits_each_record_atomically(
            combined,
        );
    }
}

#[test]
fn direct_sr_updates_each_dh_component_without_writing_other_projection_slots() {
    for combined in [false, true] {
        check_direct_sr_updates_each_dh_component_without_writing_other_projection_slots(combined);
    }
}

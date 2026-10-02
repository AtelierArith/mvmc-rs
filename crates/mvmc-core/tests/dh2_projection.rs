//! Canonical DH2 combinatorial kernels, with exhaustive original-source fixtures.
use mvmc_core::observables::set_projection_diff;
use mvmc_core::sampling::projection::{
    log_proj_ratio, log_proj_val, make_proj_cnt, update_proj_cnt,
};
use mvmc_core::{read_initial_def, read_opt_para_file, VmcOptimizationState};
use mvmc_expert_parsers::utils::parameter_init::sync_modified_parameter;
use mvmc_expert_parsers::utils::read_input_parameters::read_input_parameters;
use mvmc_expert_parsers::OrbitalTerm;
use mvmc_expert_parsers::{DoublonHolon2SiteIndex, ExpertModeData, GutzwillerTerm, JastrowTerm};
use num_complex::Complex64;
use std::{fs, path::PathBuf};

fn model() -> ExpertModeData {
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
    d.optimization_flags = vec![1; 2 * d.projection_layout().n_proj];
    d
}
fn occupancy(mask: usize) -> Vec<i64> {
    (0..8).map(|i| ((mask >> i) & 1) as i64).collect()
}
fn bits(text: &str) -> Vec<u64> {
    text.split_whitespace()
        .map(|s| u64::from_str_radix(s, 16).unwrap())
        .collect()
}
fn source_counts() -> (Vec<Vec<i64>>, Vec<u64>, Vec<Vec<u64>>) {
    let mut lines = include_str!("../../../tests/fixtures/dh2/counts.txt")
        .lines()
        .skip(1);
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
#[test]
fn every_occupation_matches_julia_dense_dh_counts_and_derivatives() {
    let d = model();
    let (expected, logs, derivatives) = source_counts();
    for mask in 0..256 {
        let mut actual = vec![99; d.projection_layout().n_proj];
        make_proj_cnt(&mut actual, &occupancy(mask), &d);
        assert_eq!(actual, expected[mask], "mask {mask}");
        assert_eq!(
            log_proj_val(&actual, &d).to_bits(),
            logs[mask],
            "mask {mask} log"
        );
        let mut diff = vec![Complex64::new(99.0, 99.0); 2 * (actual.len() + 1)];
        set_projection_diff(&mut diff, &actual, actual.len());
        assert_eq!(
            diff.iter()
                .flat_map(|v| [v.re.to_bits(), v.im.to_bits()])
                .collect::<Vec<_>>(),
            derivatives[mask]
        );
    }
}
#[test]
fn every_allowed_normal_and_spin_changing_hop_matches_julia_fresh_counts() {
    let d = model();
    let (counts, _, _) = source_counts();
    for line in include_str!("../../../tests/fixtures/dh2/moves.txt")
        .lines()
        .skip(1)
    {
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
        assert_eq!(
            log_proj_ratio(&actual, &counts[mask], &d).to_bits(),
            u64::from_str_radix(row[6], 16).unwrap()
        );
    }
}
#[test]
fn dh_gauge_matches_julia_declared_flags_compensation_and_shift_order() {
    let mut lines = include_str!("../../../tests/fixtures/dh2/gauge.txt")
        .lines()
        .skip(1);
    while let Some(header) = lines.next() {
        let name = header.split_whitespace().next().unwrap();
        let mut d = model();
        match name {
            "all" | "disabled" => {}
            "fixed_gutz" => d.optimization_flags[4] = 0,
            "fixed_dh" => d.optimization_flags[36] = 0,
            "empty_flags" => d.optimization_flags.clear(),
            "short_flags" => d.optimization_flags.truncate(15),
            "partial_params" => d.doublon_holon_2site_params.truncate(10),
            "no_gutz" => {
                d.n_gutzwiller_idx = 0;
                d.gutzwiller_terms.clear();
            }
            "no_jast" => {
                d.n_jastrow_idx = 0;
                d.jastrow_terms.clear();
            }
            "no_dh" => {
                d.doublon_holon_2site_indices.clear();
                d.doublon_holon_2site_params.clear();
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
            .chain(
                d.orbital_terms
                    .iter()
                    .map(|t| d.slater_params[t.idx as usize]),
            );
        assert_eq!(
            values
                .flat_map(|v| [v.re.to_bits(), v.im.to_bits()])
                .collect::<Vec<_>>(),
            bits(lines.next().unwrap()),
            "{name}"
        );
    }
}

fn directory(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mvmc-dh2-{}-{name}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    dir
}
#[test]
fn initial_and_fixed_loaders_apply_dh_triples_between_reserved_jastrow_and_slater() {
    let mut d = model();
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
    let dir = directory("triples");
    let path = dir.join("initial.def");
    fs::write(&path, &text).unwrap();
    assert_eq!(read_opt_para_file(&mut d, &path).unwrap(), n);
    for (i, z) in d.doublon_holon_2site_params.iter().enumerate() {
        let index = d.projection_layout().dh2_offset + i;
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
#[test]
fn strict_indh2_overlay_uses_definition_count_and_commits_each_record_atomically() {
    let mut d = model();
    let dir = directory("overlay");
    let nml = dir.join("namelist.def");
    let path = dir.join("indh2.def");
    fs::write(&nml, "InDH2 absent.def\nInDH2 indh2.def\n").unwrap();
    let text = format!(
        "===\nNDoublonHolon2siteIdx 2\nComplexType 1\n===\n===\n{}\n",
        (0..12)
            .rev()
            .map(|i| format!("{i} {} {}", i as f64 / 8.0, -(i as f64) / 16.0))
            .collect::<Vec<_>>()
            .join("\n")
    );
    fs::write(&path, &text).unwrap();
    read_input_parameters(&mut d, &nml).unwrap();
    assert_eq!(
        d.doublon_holon_2site_params,
        (0..12)
            .map(|i| Complex64::new(i as f64 / 8.0, -(i as f64) / 16.0))
            .collect::<Vec<_>>()
    );
    let before = d.projection_parameters();
    for bad in [
        text.replace("Idx 2", "Idx 12"),
        text.replace("11 1.375 -0.6875", "10 1.375 -0.6875"),
        text.replace("11 1.375 -0.6875", "12 1.375 -0.6875"),
        text.replace("1.375", "NaN"),
        text.replace("11 1.375 -0.6875\n", ""),
    ] {
        fs::write(&path, bad).unwrap();
        assert!(read_input_parameters(&mut d, &nml).is_err());
        assert_eq!(d.projection_parameters(), before);
    }
    fs::write(&path, &text).unwrap();
    d.doublon_holon_2site_params.pop();
    assert!(read_input_parameters(&mut d, &nml)
        .unwrap_err()
        .contains("target parameter length mismatch"));
    fs::remove_dir_all(dir).unwrap();
}
#[test]
fn direct_sr_updates_each_dh_component_without_writing_other_projection_slots() {
    let mut reference = include_str!("../../../tests/fixtures/dh2/sr-writeback.txt")
        .lines()
        .skip(1);
    for complex in [false, true] {
        for local in 0..12 {
            for imaginary in 0..=usize::from(complex) {
                let mut d = model();
                let layout = d.projection_layout();
                let target = layout.dh2_offset + local;
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
                let actual: Vec<u64> = d
                    .projection_parameters()
                    .iter()
                    .flat_map(|v| [v.re.to_bits(), v.im.to_bits()])
                    .collect();
                assert_eq!(
                    actual, expected,
                    "complex={complex} DH slot {local} component {imaginary}"
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

//! Canonical DH4 combinatorial kernels, with exhaustive original-source fixtures.
#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;
use mvmc_core::{read_initial_def, read_opt_para_file};
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
    // Original C ordered scatter keeps index19's initialized value when both
    // first and later records target18; the later original18 record wins.
    d.doublon_holon_4site_params[19] = Complex64::new(900.0, 901.0);
    fs::write(&path, text.replace("19 2.375 -1.1875", "18 2.375 -1.1875")).unwrap();
    read_input_parameters(&mut d, &nml).unwrap();
    assert_eq!(
        d.doublon_holon_4site_params[18],
        Complex64::new(2.25, -1.125)
    );
    assert_eq!(
        d.doublon_holon_4site_params[19],
        Complex64::new(900.0, 901.0)
    );
    fs::write(&path, &text).unwrap();
    read_input_parameters(&mut d, &nml).unwrap();
    let before = d.projection_parameters();
    for bad in [
        text.replace("Idx 2", "Idx 20"),
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

//! Canonical DH2 combinatorial kernels, with exhaustive original-source fixtures.
#[path = "../../../tests/support/numerical_comparison.rs"]
mod numerical_comparison;
use mvmc_core::{read_initial_def, read_opt_para_file};
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
    let before = d.projection_parameters();
    assert!(read_input_parameters(&mut d, &nml)
        .unwrap_err()
        .contains("duplicate keyword InDH2"));
    assert_eq!(d.projection_parameters(), before);
    fs::write(&nml, "InDH2 indh2.def\n").unwrap();
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
    // Native ReadInputParameters scatters sequentially: the duplicated first
    // row writes index10, then its later original row wins. Index11 is never
    // written and must retain its independent initialized sentinel.
    d.doublon_holon_2site_params[11] = Complex64::new(900.0, 901.0);
    fs::write(&path, text.replace("11 1.375 -0.6875", "10 1.375 -0.6875")).unwrap();
    read_input_parameters(&mut d, &nml).unwrap();
    assert_eq!(
        d.doublon_holon_2site_params[10],
        Complex64::new(1.25, -0.625)
    );
    assert_eq!(
        d.doublon_holon_2site_params[11],
        Complex64::new(900.0, 901.0)
    );
    fs::write(&path, &text).unwrap();
    read_input_parameters(&mut d, &nml).unwrap();
    let before = d.projection_parameters();
    for bad in [
        text.replace("Idx 2", "Idx 12"),
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

//! Definition flags must control initialization draws and gauge shifts.
use mvmc_expert_parsers::parse_expert_mode_files;
use mvmc_expert_parsers::utils::parameter_init::{init_parameter, sync_modified_parameter};
use num_complex::Complex64;
use sfmt19937::Sfmt19937Rng;
use std::fs;

fn definition(name: &str, width: usize, complex: usize, rows: &str) -> String {
    format!("===\n{name} {width}\nComplexType {complex}\n===\n===\n{rows}")
}

fn parsed(
    name: &str,
    complex_jastrow: usize,
    parallel: bool,
) -> mvmc_expert_parsers::ExpertModeData {
    let dir = std::env::temp_dir().join(format!("mvmc-flags-{name}-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("modpara.def"), "Nsite 3\nNElec 1\n").unwrap();
    fs::write(
        dir.join("g.def"),
        definition("NGutzwillerIdx", 2, 0, "0 0\n1 0\n2 1\n0 1\n1 0\n"),
    )
    .unwrap();
    fs::write(
        dir.join("j.def"),
        definition(
            "NJastrowIdx",
            2,
            complex_jastrow,
            "0 1 0\n1 2 1\n0 0\n1 1\n",
        ),
    )
    .unwrap();
    fs::write(
        dir.join("o.def"),
        definition(
            "NOrbitalIdx",
            if parallel { 7 } else { 2 },
            0,
            "0 1 0\n1 0 1\n0 0\n1 1\n",
        ),
    )
    .unwrap();
    fs::write(
        dir.join("p.def"),
        definition("NOrbitalParallel", 2, 0, "0 1 0\n1 2 1\n0 1\n1 0\n"),
    )
    .unwrap();
    let mut namelist =
        "ModPara modpara.def\nOrbitalAntiParallel o.def\nGutzwiller g.def\nJastrow j.def\n"
            .to_owned();
    if parallel {
        namelist.push_str("OrbitalParallel p.def\n");
    }
    fs::write(dir.join("namelist.def"), namelist).unwrap();
    let data = parse_expert_mode_files(dir.join("namelist.def")).unwrap();
    fs::remove_dir_all(dir).unwrap();
    data
}

#[test]
fn shared_indices_and_ap_parallel_offsets_populate_global_component_flags() {
    let data = parsed("complex", 1, true);
    let mut expected = vec![true, false, false, false, false, false, true, true];
    for active in [
        false, true, true, true, true, true, true, true, true, false, false,
    ] {
        expected.extend([active, active]);
    }
    assert_eq!(data.optimization_flags, expected);
    assert_eq!(data.gutzwiller_idx, [0, 0, 1]);
}

#[test]
fn real_projection_imaginary_flags_and_orbital_defaults_match_julia() {
    let data = parsed("real", 0, false);
    assert_eq!(
        data.optimization_flags,
        [true, false, false, false, false, false, true, false, false, true, true, true]
    );
}

#[test]
fn fixed_slater_slots_skip_rng_draws_including_shared_and_reserved_slots() {
    let mut data = parsed("rng", 1, true);
    let mut rng = Sfmt19937Rng::new(1);
    let mut probe = Sfmt19937Rng::new(1);
    init_parameter(&mut data, &mut rng);
    // Live Julia v0.5.0, including division by sqrt(2) rather than
    // multiplication by its rounded reciprocal.
    for (term, (re, im)) in data.orbital_terms.iter().zip([
        (0.0_f64, 0.0_f64),
        (-0.22854561532191836, 0.14249578128268756),
        (-0.299485566106154, -0.1347245207905717),
        (-0.28930135836557447, 0.17361291906410883),
        (0.0, 0.0),
        (0.0, 0.0),
    ]) {
        assert_eq!(term.value.re.to_bits(), re.to_bits());
        assert_eq!(term.value.im.to_bits(), im.to_bits());
    }
    for term in &data.orbital_terms {
        if [0, 9, 10].contains(&term.idx) {
            assert_eq!(term.value, Complex64::new(0.0, 0.0));
        }
    }
    for _ in 0..16 {
        probe.genrand_real2();
    }
    let mut hash = 0xcbf29ce484222325_u64;
    for _ in 0..624 {
        let word = rng.gen_rand32();
        assert_eq!(word, probe.gen_rand32());
        hash = (hash ^ u64::from(word)).wrapping_mul(0x100000001b3);
    }
    assert_eq!(hash, 16445735861883055124);
}

#[test]
fn fixed_correlation_blocks_disable_gauge_shift_but_not_slater_normalization() {
    let mut data = parsed("fixed-shift", 1, false);
    for (i, term) in data.gutzwiller_terms.iter_mut().enumerate() {
        term.value = Complex64::new(1.0 + i as f64, 0.2);
    }
    for (i, term) in data.jastrow_terms.iter_mut().enumerate() {
        term.value = Complex64::new(3.0 + i as f64, 0.5);
    }
    data.orbital_terms[0].value = Complex64::new(1.0, 0.0);
    data.orbital_terms[1].value = Complex64::new(2.0, 0.0);
    let before_g = data.gutzwiller_terms.clone();
    let before_j = data.jastrow_terms.clone();
    for _ in 0..3 {
        sync_modified_parameter(&mut data, true);
        assert_eq!(data.gutzwiller_terms, before_g);
        assert_eq!(data.jastrow_terms, before_j);
    }
    assert_eq!(
        data.orbital_terms[0].value.re, 2.0,
        "fixed Slater participates in Julia rescaling"
    );
    assert_eq!(data.orbital_terms[1].value.re, 4.0);
}

#[test]
fn declared_projection_widths_determine_slater_flag_and_rng_offsets() {
    use mvmc_expert_parsers::utils::opt_flag::set_orbital_opt_flags;
    use mvmc_expert_parsers::{ExpertModeData, GutzwillerTerm, JastrowTerm, OrbitalTerm};
    let mut data = ExpertModeData::new();
    data.n_gutzwiller_idx = 3;
    data.n_jastrow_idx = 4;
    data.gutzwiller_terms.push(GutzwillerTerm {
        site: 0,
        value: Complex64::new(1.0, 0.0),
        is_complex: false,
    });
    data.jastrow_terms.push(JastrowTerm {
        site1: 0,
        site2: 1,
        value: Complex64::new(2.0, 0.0),
        is_complex: false,
    });
    data.modpara.n_orbital_idx = 1;
    data.orbital_terms.push(OrbitalTerm {
        site1: 0,
        site2: 1,
        idx: 0,
        sign: 1,
        value: Complex64::new(0.0, 0.0),
        is_complex: false,
    });
    set_orbital_opt_flags(&mut data, &[(0, 0)].into());
    assert_eq!(data.optimization_flags.len(), 16);
    assert!(!data.optimization_flags[14]);
    assert_eq!(
        data.projection_parameters()
            .iter()
            .map(|v| v.re)
            .collect::<Vec<_>>(),
        [1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0]
    );
    mvmc_expert_parsers::utils::opt_flag::set_projection_opt_flags(
        &mut data,
        &Default::default(),
        &[(0, 0)].into(),
        false,
        false,
    );
    sync_modified_parameter(&mut data, true);
    assert_eq!(data.gutzwiller_terms[0].value.re, 1.0);
    assert_eq!(
        data.jastrow_terms[0].value.re, 2.0,
        "Jastrow flag uses the declared Gutzwiller offset"
    );
    let mut rng = Sfmt19937Rng::new(1);
    let mut probe = Sfmt19937Rng::new(1);
    init_parameter(&mut data, &mut rng);
    assert_eq!(data.orbital_terms[0].value, Complex64::new(0.0, 0.0));
    for _ in 0..624 {
        assert_eq!(rng.gen_rand32(), probe.gen_rand32());
    }
}

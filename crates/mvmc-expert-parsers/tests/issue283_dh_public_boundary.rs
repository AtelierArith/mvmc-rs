//! Native readdef.c header -> DH token streams -> final public component layout.
//! Independent literals; no historical loader adapter, oracle or model run.
use mvmc_expert_parsers::parsers::doublon_holon::{
    parse_doublon_holon_2site_content, parse_doublon_holon_4site_content,
};
use mvmc_expert_parsers::{
    parse_expert_mode_files,
    utils::parameter_init::{all_complex_flag, init_parameter},
};
use sfmt19937::Sfmt19937Rng;
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

struct Files(PathBuf);
impl Files {
    fn new() -> Self {
        static SERIAL: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir();
        for _ in 0..100 {
            let path = root.join(format!(
                "mvmc-283-{}-{}",
                std::process::id(),
                SERIAL.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => panic!("exclusive staging: {e}"),
            }
        }
        panic!("exclusive staging exhausted");
    }
    fn put(&self, name: &str, text: &str) {
        fs::write(self.0.join(name), text).unwrap();
    }
}
impl Drop for Files {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

fn definition(count: i32, complex: i32, body: &str) -> String {
    format!("===\nDeclared {count}\nComplexType {complex}\n===\n===\n{body}\n")
}
fn dh2_body() -> &'static str {
    // Complete reordered centers, then six printed labels that do not select slots.
    "1 0 1 0 0 1 0 0 99 -2 -7 -1 18 0 0 1 0 2 0 3"
}
fn dh4_body() -> &'static str {
    "1 0 1 0 1 0 0 1 0 1 0 0 99 0 98 1 97 0 96 1 95 0 94 1 93 0 92 1 91 0 90 1"
}

#[test]
fn complete_dh_bodies_use_tokens_not_physical_lines_and_keep_raw_flags() {
    for split in [false, true] {
        let body2 = if split {
            dh2_body().replace(' ', "\n\t")
        } else {
            dh2_body().into()
        };
        let body4 = if split {
            dh4_body().replace(' ', "\n\t")
        } else {
            dh4_body().into()
        };
        let a = parse_doublon_holon_2site_content(&definition(1, -2, &body2), 2)
            .data
            .unwrap();
        let b = parse_doublon_holon_4site_content(&definition(1, 3, &body4), 2)
            .data
            .unwrap();
        assert_eq!(a.indices[0].neighbors, [[1, 0], [0, 1]]);
        assert_eq!(b.indices[0].neighbors, [[1, 0, 1, 0], [0, 1, 0, 1]]);
        assert_eq!(a.opt_flags, [-2, -1, 0, 1, 2, 3]);
        assert_eq!(b.opt_flags, [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]);
        assert_eq!((a.complex_type, b.complex_type), (-2, 3));
    }
}

#[test]
fn zero_counts_and_incomplete_extra_or_unsafe_consumed_values_fail_atomically() {
    for (count, body) in [(0, ""), (-1, ""), (1, "0 1"), (1, "garbage")] {
        assert!(
            parse_doublon_holon_2site_content(&definition(count, 1, body), 2)
                .data
                .is_none()
        );
        assert!(
            parse_doublon_holon_4site_content(&definition(count, 1, body), 2)
                .data
                .is_none()
        );
    }
    for body in [
        format!("{} 0 1", dh2_body()),
        dh2_body().replacen("1 0 1 0", "2 0 1 0", 1),
        dh2_body().replacen("1 0 1 0", "0 0 1 0", 1),
        dh2_body().replace("99 -2", "99 2147483648"),
    ] {
        assert!(
            parse_doublon_holon_2site_content(&definition(1, 1, &body), 2)
                .data
                .is_none()
        );
    }
    for body in [
        format!("{} 0 1", dh4_body()),
        dh4_body().replacen("1 0 1 0 1 0", "2 0 1 0 1 0", 1),
        dh4_body().replacen("1 0 1 0 1 0", "0 0 1 0 1 0", 1),
        dh4_body().replace("98 1", "98 2147483648"),
    ] {
        assert!(
            parse_doublon_holon_4site_content(&definition(1, 1, &body), 2)
                .data
                .is_none()
        );
    }
}

fn combined(files: &Files, header2: i32, header4: i32, aliases: bool, reverse: bool) {
    files.put("mod.def", "Nsite 2\nNcond -1\nNe 1\n");
    files.put("g.def", &definition(1, 0, "0 0 1 0 0 1"));
    files.put("j.def", &definition(1, 0, "0 1 0 1 0 0 0 -2"));
    files.put("dh2.def", &definition(1, header2, dh2_body()));
    files.put("dh4.def", &definition(1, header4, dh4_body()));
    files.put("rbm.def", &definition(1, 1, "0 0 1 0 0 3"));
    files.put(
        "orb.def",
        &definition(2, 0, "0 0 0 1\n0 1 1 1\n1 0 1 1\n1 1 0 1\n0 0\n1 1"),
    );
    let (a, b) = if aliases {
        ("DoublonHolon2Site", "DoublonHolon4Site")
    } else {
        ("DH2", "DH4")
    };
    let mut list = [
        "ModPara mod.def".into(),
        "Gutzwiller g.def".into(),
        "Jastrow j.def".into(),
        format!("{a} dh2.def"),
        format!("{b} dh4.def"),
        "ChargeRBM_PhysLayer rbm.def".into(),
        "Orbital orb.def".into(),
    ];
    if reverse {
        list.reverse();
    }
    files.put("namelist.def", &list.join("\n"));
}

#[test]
fn actual_public_loader_assembles_combined_dh_rbm_orbital_layout_in_either_namelist_order() {
    for aliases in [false, true] {
        for reverse in [false, true] {
            let files = Files::new();
            combined(&files, 2, 0, aliases, reverse);
            let data = parse_expert_mode_files(files.0.join("namelist.def")).unwrap();
            assert!(data.input_errors.is_empty(), "{:?}", data.input_errors);
            let layout = data.projection_layout();
            assert_eq!(
                (
                    layout.n_gutzwiller,
                    layout.n_jastrow,
                    layout.n_dh2,
                    layout.n_dh4
                ),
                (1, 1, 1, 1)
            );
            assert_eq!(
                (layout.dh2_offset, layout.dh4_offset, layout.n_proj),
                (2, 8, 18)
            );
            assert_eq!(data.doublon_holon_2site_params.len(), 6);
            assert_eq!(data.doublon_holon_4site_params.len(), 10);
            assert_eq!(data.rbm_params.len(), 1);
            assert_eq!(data.slater_params.len(), 2);
            assert_eq!(
                data.optimization_flags,
                [
                    1, 0, -2, 0, -2, -2, -1, -1, 0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                    0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 3, 3, 0, 0, 1, 0
                ]
            );
            assert_eq!(data.native_complex_headers["DH2"], 2);
            assert_eq!(data.native_complex_headers["DH4"], 0);
            assert!(all_complex_flag(&data).unwrap());
        }
    }
}

#[test]
fn two_reordered_definitions_preserve_tables_offsets_and_74_literal_components() {
    for aliases in [false, true] {
        for reverse in [false, true] {
            let files = Files::new();
            combined(&files, 2, 0, aliases, reverse);
            files.put(
                "dh2.def",
                &definition(
                    2,
                    2,
                    "1 0 1 1 0 1 0 0 0 0 1 1 1 1 0 0 \
                 11 1 10 0 9 1 8 0 7 1 6 0 5 1 4 0 3 1 2 0 1 1 0 0",
                ),
            );
            files.put(
                "dh4.def",
                &definition(
                    2,
                    0,
                    "1 0 1 0 1 1 0 1 0 1 0 0 0 1 1 0 0 1 1 0 0 1 1 0 \
                 19 0 18 1 17 0 16 1 15 0 14 1 13 0 12 1 11 0 10 1 \
                 9 0 8 1 7 0 6 1 5 0 4 1 3 0 2 1 1 0 0 1",
                ),
            );
            let data = parse_expert_mode_files(files.0.join("namelist.def")).unwrap();
            assert!(data.input_errors.is_empty(), "{:?}", data.input_errors);
            let layout = data.projection_layout();
            assert_eq!(
                (
                    layout.n_dh2,
                    layout.n_dh4,
                    layout.dh2_offset,
                    layout.dh4_offset,
                    layout.n_proj
                ),
                (2, 2, 2, 14, 34)
            );
            assert_eq!(data.doublon_holon_2site_params.len(), 12);
            assert_eq!(data.doublon_holon_4site_params.len(), 20);
            assert_eq!(
                data.doublon_holon_2site_indices[0].neighbors,
                [[1, 0], [1, 0]]
            );
            assert_eq!(
                data.doublon_holon_2site_indices[1].neighbors,
                [[0, 1], [0, 1]]
            );
            assert_eq!(
                data.doublon_holon_4site_indices[0].neighbors,
                [[1, 0, 1, 0], [0, 0, 1, 1]]
            );
            assert_eq!(
                data.doublon_holon_4site_indices[1].neighbors,
                [[1, 1, 0, 0], [0, 1, 0, 1]]
            );
            assert_eq!(
                data.optimization_flags,
                [
                    1, 0, -2, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
                    0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0,
                    0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 3, 3, 0, 0, 1, 0,
                ]
            );
        }
    }
}

#[test]
fn stale_loaded_declarations_error_but_unloaded_programmatic_declarations_contribute() {
    let files = Files::new();
    combined(&files, 0, 0, false, false);
    let mut data = parse_expert_mode_files(files.0.join("namelist.def")).unwrap();
    data.doublon_holon_2site_params[0].im = 9.0;
    assert_eq!(all_complex_flag(&data), Ok(false));
    data.doublon_holon_2site_complex = true;
    assert!(all_complex_flag(&data).is_err());
    let before = data.clone();
    let mut rng = Sfmt19937Rng::new(1);
    let state_before = rng.state_snapshot();
    assert!(init_parameter(&mut data, &mut rng).is_err());
    assert_eq!(rng.state_snapshot(), state_before);
    assert_eq!(rng.words_consumed(), 0);
    assert_eq!(data.projection_parameters(), before.projection_parameters());
    assert_eq!(data.rbm_params, before.rbm_params);
    assert_eq!(data.slater_params, before.slater_params);
    data.native_complex_headers.remove("DH2");
    data.native_complex_declarations.remove("DH2");
    assert_eq!(all_complex_flag(&data), Ok(true));
}

#[test]
fn count2_duplicate_hole_out_of_range_and_truncated_flags_are_rejected() {
    let dh2 = "0 1 0 0 1 0 1 0 0 0 1 1 1 1 0 1 \
        0 1 1 0 2 1 3 0 4 1 5 0 6 1 7 0 8 1 9 0 10 1 11 0";
    let dh4 = "0 1 0 1 0 0 1 0 1 0 1 0 0 1 1 0 0 1 1 0 0 1 1 1 \
        0 0 1 1 2 0 3 1 4 0 5 1 6 0 7 1 8 0 9 1 \
        10 0 11 1 12 0 13 1 14 0 15 1 16 0 17 1 18 0 19 1";
    assert!(parse_doublon_holon_2site_content(&definition(2, 1, dh2), 2)
        .data
        .is_some());
    assert!(parse_doublon_holon_4site_content(&definition(2, 1, dh4), 2)
        .data
        .is_some());
    for body in [
        dh2.replacen("1 1 0 1", "0 1 0 1", 1),
        dh2.replacen("0 0 1 1", "0 0 1 2", 1),
        dh2.replacen("0 1 0 0", "0 2 0 0", 1),
        dh2.trim_end_matches("11 0").to_string(),
    ] {
        assert!(
            parse_doublon_holon_2site_content(&definition(2, 1, &body), 2)
                .data
                .is_none()
        );
    }
    for body in [
        dh4.replacen("1 0 0 1 1 1", "0 0 0 1 1 1", 1),
        dh4.replacen("0 1 1 0 0 1", "0 1 1 0 0 2", 1),
        dh4.replacen("0 1 0 1 0 0", "0 2 0 1 0 0", 1),
        dh4.trim_end_matches("19 1").to_string(),
    ] {
        assert!(
            parse_doublon_holon_4site_content(&definition(2, 1, &body), 2)
                .data
                .is_none()
        );
    }
}

#[test]
fn invalid_signed_headers_never_publish_a_fallback_or_native_binding() {
    use mvmc_expert_parsers::parsers::{gutzwiller, jastrow, orbital};
    for raw in ["garbage", "2147483648", "-2147483649", ""] {
        for (kind, filename, key) in [
            ("Gutzwiller", "g.def", "Gutzwiller"),
            ("Jastrow", "j.def", "Jastrow"),
            ("Orbital", "orb.def", "Orbital"),
            ("DH2", "dh2.def", "DH2"),
            ("DH4", "dh4.def", "DH4"),
        ] {
            let files = Files::new();
            combined(&files, 0, 0, false, false);
            let original = fs::read_to_string(files.0.join(filename)).unwrap();
            let invalid = original.replacen("ComplexType 0", &format!("ComplexType {raw}"), 1);
            files.put(filename, &invalid);
            match kind {
                "Gutzwiller" => assert!(gutzwiller::parse_gutzwiller_content(&invalid, 2).is_err()),
                "Jastrow" => assert!(jastrow::parse_jastrow_content(&invalid, 2).is_err()),
                "Orbital" => assert!(orbital::parse_orbital_content(
                    &invalid,
                    2,
                    orbital::OrbitalKind::AntiParallel
                )
                .is_err()),
                "DH2" => assert!(parse_doublon_holon_2site_content(&invalid, 2)
                    .data
                    .is_none()),
                "DH4" => assert!(parse_doublon_holon_4site_content(&invalid, 2)
                    .data
                    .is_none()),
                _ => unreachable!(),
            }
            if kind.starts_with("DH") {
                assert!(parse_expert_mode_files(files.0.join("namelist.def")).is_err());
            } else {
                let data = parse_expert_mode_files(files.0.join("namelist.def")).unwrap();
                assert!(!data.input_errors.is_empty());
                assert!(!data.native_complex_headers.contains_key(key));
            }
        }
    }
}

#[test]
fn orbital_ap_parallel_signed_sum_is_bound_before_local_flag_normalization() {
    for ap_key in ["Orbital", "OrbitalAntiParallel"] {
        for reverse in [false, true] {
            for (ap, parallel, sum, normalized) in
                [(1, -1, 0, 0), (-2, 1, -1, -1), (-1, 2, 1, 1), (2, 3, 5, 1)]
            {
                let files = Files::new();
                files.put("mod.def", "Nsite 2\nNcond 2\n");
                files.put(
                    "ap.def",
                    &definition(1, ap, "0 0 0 1\n0 1 0 1\n1 0 0 1\n1 1 0 1\n0 -2"),
                );
                files.put("p.def", &definition(1, parallel, "0 1 0 1\n0 3"));
                let mut list = [
                    "ModPara mod.def".to_string(),
                    format!("{ap_key} ap.def"),
                    "OrbitalParallel p.def".to_string(),
                ];
                if reverse {
                    list.reverse();
                }
                files.put("namelist.def", &list.join("\n"));
                let data = parse_expert_mode_files(files.0.join("namelist.def")).unwrap();
                assert!(data.input_errors.is_empty(), "{:?}", data.input_errors);
                assert_eq!(data.native_complex_headers[ap_key], ap);
                assert_eq!(data.native_complex_headers["OrbitalParallel"], parallel);
                assert!(!data
                    .native_complex_headers
                    .contains_key(if ap_key == "Orbital" {
                        "OrbitalAntiParallel"
                    } else {
                        "Orbital"
                    }));
                assert_eq!(all_complex_flag(&data), Ok(sum != 0));
                assert!(data
                    .orbital_terms
                    .iter()
                    .all(|term| term.is_complex == (sum != 0)));
                assert_eq!(
                    data.optimization_flags,
                    [
                        -2,
                        if normalized > 0 { -2 } else { 0 },
                        3,
                        normalized,
                        3,
                        normalized
                    ]
                );
            }
        }
    }
}

#[test]
fn raw_signed_headers_cancel_for_initialization_but_local_imaginary_flags_require_positive() {
    for (h2, h4, complex, expected_draws) in [(1, -1, false, 1), (-2, 0, true, 2), (2, -1, true, 2)]
    {
        let files = Files::new();
        combined(&files, h2, h4, false, false);
        let mut data = parse_expert_mode_files(files.0.join("namelist.def")).unwrap();
        // Remove active RBM coefficient from this initializer-only observation;
        // header sum excludes RBM in C. One active Slater slot remains.
        data.optimization_flags[36] = 0;
        assert_eq!(all_complex_flag(&data).unwrap(), complex);
        assert_eq!(data.optimization_flags[5], if h2 > 0 { -2 } else { 0 });
        assert_eq!(data.optimization_flags[19], if h4 > 0 { 1 } else { 0 });
        let mut rng = Sfmt19937Rng::new(1);
        init_parameter(&mut data, &mut rng).unwrap();
        assert_eq!(rng.words_consumed(), expected_draws);
        assert_eq!(data.slater_params[0].re, 0.0);
        assert_eq!(data.slater_params[0].im, 0.0);
    }
}

#[test]
fn public_loader_rejects_missing_zero_and_truncated_dh_for_both_aliases() {
    for (kind, filename) in [
        ("DH2", "dh2.def"),
        ("DoublonHolon2Site", "dh2.def"),
        ("DH4", "dh4.def"),
        ("DoublonHolon4Site", "dh4.def"),
    ] {
        let files = Files::new();
        files.put("mod.def", "Nsite 2\nNcond -1\nNe 1\n");
        files.put(
            "namelist.def",
            &format!("ModPara mod.def\n{kind} {filename}\n"),
        );
        assert!(parse_expert_mode_files(files.0.join("namelist.def")).is_err());
        for content in [
            definition(0, 1, ""),
            definition(1, 1, "0 1"),
            "===\nDeclared 1\n".into(),
        ] {
            files.put(filename, &content);
            assert!(parse_expert_mode_files(files.0.join("namelist.def")).is_err());
        }
    }
}

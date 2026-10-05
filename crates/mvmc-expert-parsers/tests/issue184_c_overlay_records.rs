//! SOURCE/TDD only: native C in-range repeated writes and header guard.
//! Expected literals follow original readdef.c cases, not Rust-generated goldens.
use mvmc_expert_parsers::utils::read_input_parameters::read_input_parameters;
use mvmc_expert_parsers::{parse_expert_mode_files, ExpertModeData};
use num_complex::Complex64 as C;
use std::{fs, path::PathBuf};

struct Bundle(PathBuf);
impl Drop for Bundle {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove exclusive overlay test directory");
    }
}
fn definition(count: usize, rows: &str) -> String {
    format!("===\nNGutzwillerIdx {count}\nComplexType 1\n===\n===\n{rows}")
}
fn model() -> (Bundle, ExpertModeData) {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let bundle = Bundle(
        std::env::temp_dir().join(format!("issue184-c-overlay-{}-{stamp}", std::process::id())),
    );
    fs::create_dir(&bundle.0).unwrap();
    fs::write(bundle.0.join("modpara.def"), concat!("--------------------\nModel_Parameters 0\n--------------------\nVMC_Cal_Parameters\n--------------------\nCDataFileHead zvo\nCParaFileHead zqp\n--------------------\n", "Nsite 2\nNe 1\n")).unwrap();
    fs::write(
        bundle.0.join("gutz.def"),
        definition(2, "0 0\n1 1\n0 1\n1 1\n"),
    )
    .unwrap();
    fs::write(
        bundle.0.join("namelist.def"),
        "ModPara modpara.def\nGutzwiller gutz.def\nInGutzwiller overlay.def\n",
    )
    .unwrap();
    let mut data = parse_expert_mode_files(bundle.0.join("namelist.def")).unwrap();
    assert_eq!(data.n_gutzwiller_idx, 2);
    assert_eq!(data.gutzwiller_terms.len(), 2);
    data.gutzwiller_terms[0].value = C::new(100.0, 200.0);
    data.gutzwiller_terms[1].value = C::new(101.0, 201.0);
    (bundle, data)
}
#[test]
fn valid_repeated_overlay_index_keeps_last_write_and_initialized_unwritten_slot() {
    let (bundle, mut data) = model();
    fs::write(
        bundle.0.join("overlay.def"),
        definition(2, "0 1 2\n0 3 4\n"),
    )
    .unwrap();
    read_input_parameters(&mut data, bundle.0.join("namelist.def")).unwrap();
    assert_eq!(data.gutzwiller_terms[0].value, C::new(3.0, 4.0));
    assert_eq!(data.gutzwiller_terms[1].value, C::new(101.0, 201.0));
}
#[test]
fn wrong_declared_overlay_count_fails_before_parameter_writes() {
    let (bundle, mut data) = model();
    let before: Vec<_> = data
        .gutzwiller_terms
        .iter()
        .map(|term| term.value)
        .collect();
    fs::write(
        bundle.0.join("overlay.def"),
        definition(3, "0 1 2\n0 3 4\n"),
    )
    .unwrap();
    // C sets info1 and its full caller aborts. Rust maps that to an error;
    // no assertion of identical C MPI exception/status architecture is made.
    assert!(read_input_parameters(&mut data, bundle.0.join("namelist.def")).is_err());
    assert_eq!(
        data.gutzwiller_terms
            .iter()
            .map(|term| term.value)
            .collect::<Vec<_>>(),
        before
    );
}

fn seeded_declared_rbm() -> ExpertModeData {
    let mut data = ExpertModeData::new();
    data.rbm_section_widths = [2; 9];
    data.rbm_params = (0..18)
        .map(|i| C::new(100.0 + i as f64, 200.0 + i as f64))
        .collect();
    data
}

#[test]
fn all_nine_declared_rbm_sections_scatter_duplicates_preserving_holes_and_offsets() {
    let (bundle, _) = model();
    for (section, name) in mvmc_expert_parsers::parsers::rbm::SECTION_NAMES
        .iter()
        .enumerate()
    {
        let mut data = seeded_declared_rbm();
        let before = data.rbm_parameters();
        fs::write(
            bundle.0.join("namelist.def"),
            format!("In{name} overlay.def\n"),
        )
        .unwrap();
        fs::write(
            bundle.0.join("overlay.def"),
            definition(2, "0 1 2\n0 3 4\n"),
        )
        .unwrap();
        read_input_parameters(&mut data, bundle.0.join("namelist.def")).unwrap();
        let mut expected = before.clone();
        expected[2 * section] = C::new(3.0, 4.0);
        assert_eq!(data.rbm_parameters(), expected, "{name}");
        for prefix in ["0 1 2\n0 3 4\n", "0 1 2 0 3 4 ", "\n0 1\n2 0\n3 4\n"] {
            for trailing in ["0 999 888\n", "invalid # ignored suffix\n"] {
                fs::write(
                    bundle.0.join("overlay.def"),
                    definition(2, &format!("{prefix}{trailing}")),
                )
                .unwrap();
                read_input_parameters(&mut data, bundle.0.join("namelist.def")).unwrap();
                assert_eq!(
                    data.rbm_parameters(),
                    expected,
                    "trailing {name}/{prefix}/{trailing}"
                );
            }
        }
        fs::write(
            bundle.0.join("overlay.def"),
            definition(3, "0 5 6\n0 7 8\n"),
        )
        .unwrap();
        assert!(read_input_parameters(&mut data, bundle.0.join("namelist.def")).is_err());
        assert_eq!(data.rbm_parameters(), expected, "header guard {name}");
    }
}

#[test]
fn malformed_or_out_of_range_rbm_section_rejects_without_current_section_mutation() {
    let (bundle, _) = model();
    fs::write(
        bundle.0.join("namelist.def"),
        "InChargeRBM_PhysLayer overlay.def\n",
    )
    .unwrap();
    for rows in ["0 1 2\n2 3 4\n", "0 1 2\n-1 3 4\n", "0 1 2\n0 bad 4\n"] {
        let mut data = seeded_declared_rbm();
        let before = data.rbm_parameters();
        fs::write(bundle.0.join("overlay.def"), definition(2, rows)).unwrap();
        assert!(read_input_parameters(&mut data, bundle.0.join("namelist.def")).is_err());
        assert_eq!(data.rbm_parameters(), before);
    }
}

#[test]
fn successful_prior_rbm_section_remains_applied_when_later_header_fails() {
    let (bundle, _) = model();
    let mut data = seeded_declared_rbm();
    let mut expected = data.rbm_parameters();
    // Native keyword order reads Charge Phys before Spin Phys even if reversed.
    fs::write(
        bundle.0.join("namelist.def"),
        "InSpinRBM_PhysLayer bad.def\nInChargeRBM_PhysLayer good.def\n",
    )
    .unwrap();
    fs::write(bundle.0.join("good.def"), definition(2, "0 1 2\n0 3 4\n")).unwrap();
    fs::write(bundle.0.join("bad.def"), definition(3, "0 5 6\n0 7 8\n")).unwrap();
    assert!(read_input_parameters(&mut data, bundle.0.join("namelist.def")).is_err());
    expected[0] = C::new(3.0, 4.0);
    assert_eq!(data.rbm_parameters(), expected);
}

fn seeded_projection() -> ExpertModeData {
    use mvmc_expert_parsers::{
        DoublonHolon2SiteIndex, DoublonHolon4SiteIndex, GutzwillerTerm, JastrowTerm,
    };
    let mut data = ExpertModeData::new();
    data.n_gutzwiller_idx = 2;
    data.n_jastrow_idx = 2;
    data.gutzwiller_terms = (0..2)
        .map(|i| GutzwillerTerm {
            site: i,
            value: C::new(100.0 + i as f64, 200.0 + i as f64),
            is_complex: true,
        })
        .collect();
    data.jastrow_terms = (0..2)
        .map(|i| JastrowTerm {
            site1: 0,
            site2: 1,
            value: C::new(102.0 + i as f64, 202.0 + i as f64),
            is_complex: true,
        })
        .collect();
    data.doublon_holon_2site_indices = vec![DoublonHolon2SiteIndex {
        neighbors: vec![[1, 0], [0, 1]],
    }];
    data.doublon_holon_4site_indices = vec![DoublonHolon4SiteIndex {
        neighbors: vec![[1, 0, 1, 0], [0, 1, 0, 1]],
    }];
    data.doublon_holon_2site_params = (4..10)
        .map(|i| C::new(100.0 + i as f64, 200.0 + i as f64))
        .collect();
    data.doublon_holon_4site_params = (10..20)
        .map(|i| C::new(100.0 + i as f64, 200.0 + i as f64))
        .collect();
    data
}

#[test]
fn projection_four_families_keep_duplicate_holes_and_c_block_offsets() {
    let (bundle, _) = model();
    for (name, header, width, offset) in [
        ("InGutzwiller", 2, 2, 0),
        ("InJastrow", 2, 2, 2),
        ("InDH2", 1, 6, 4),
        ("InDH4", 1, 10, 10),
    ] {
        let mut data = seeded_projection();
        let before = data.projection_parameters();
        fs::write(
            bundle.0.join("namelist.def"),
            format!("{name} overlay.def\n"),
        )
        .unwrap();
        let mut rows = String::from("0 1 2\n0 3 4\n");
        for index in 2..width {
            rows.push_str(&format!("{index} {} {}\n", 10 + index, 20 + index));
        }
        fs::write(bundle.0.join("overlay.def"), definition(header, &rows)).unwrap();
        read_input_parameters(&mut data, bundle.0.join("namelist.def")).unwrap();
        let mut expected = before;
        expected[offset] = C::new(3.0, 4.0);
        for index in 2..width {
            expected[offset + index] = C::new((10 + index) as f64, (20 + index) as f64);
        }
        assert_eq!(data.projection_parameters(), expected, "{name}");
        fs::write(bundle.0.join("overlay.def"), definition(header + 1, &rows)).unwrap();
        assert!(read_input_parameters(&mut data, bundle.0.join("namelist.def")).is_err());
        assert_eq!(data.projection_parameters(), expected, "header {name}");
    }
}

#[test]
fn later_dh_failure_keeps_successful_gutz_section_and_rejects_bad_index_atomically() {
    let (bundle, _) = model();
    for invalid in [
        "0 1 2\n6 3 4\n2 5 6\n3 7 8\n4 9 10\n5 11 12\n",
        "0 1 2\n0 bad 4\n2 5 6\n3 7 8\n4 9 10\n5 11 12\n",
    ] {
        let mut data = seeded_projection();
        let mut expected = data.projection_parameters();
        fs::write(
            bundle.0.join("namelist.def"),
            "InDH2 bad.def\nInGutzwiller good.def\n",
        )
        .unwrap();
        fs::write(bundle.0.join("good.def"), definition(2, "0 1 2\n0 3 4\n")).unwrap();
        fs::write(bundle.0.join("bad.def"), definition(1, invalid)).unwrap();
        assert!(read_input_parameters(&mut data, bundle.0.join("namelist.def")).is_err());
        expected[0] = C::new(3.0, 4.0);
        assert_eq!(data.projection_parameters(), expected);
    }
}

#[test]
fn declared_prefix_ignores_valid_and_invalid_trailing_input_for_all_projection_families() {
    let (bundle, _) = model();
    for (name, header, width, offset) in [
        ("InGutzwiller", 2, 2, 0),
        ("InJastrow", 2, 2, 2),
        ("InDH2", 1, 6, 4),
        ("InDH4", 1, 10, 10),
    ] {
        for trailing in ["0 999 888\n", "# not parsed\ninvalid -999 NaN\n"] {
            let mut data = seeded_projection();
            let mut expected = data.projection_parameters();
            fs::write(
                bundle.0.join("namelist.def"),
                format!("{name} overlay.def\n"),
            )
            .unwrap();
            // Prefix whitespace permits blank lines, same-line and split fields.
            for record in ["0 3 4 ", "0\n3\n4\n"] {
                let rows = format!("\n{}\n{trailing}", record.repeat(width));
                fs::write(bundle.0.join("overlay.def"), definition(header, &rows)).unwrap();
                read_input_parameters(&mut data, bundle.0.join("namelist.def")).unwrap();
                expected[offset] = C::new(3.0, 4.0);
                assert_eq!(
                    data.projection_parameters(),
                    expected,
                    "{name}/{record}/{trailing}"
                );
            }
        }
    }
}

#[test]
fn consumed_comments_or_missing_fields_reject_current_section_without_mutation() {
    let (bundle, _) = model();
    for rows in ["0 1 2\n# consumed comment\n0 3 4\n", "0 1 2\n0 3\n"] {
        let mut data = seeded_projection();
        let before = data.projection_parameters();
        fs::write(bundle.0.join("namelist.def"), "InGutzwiller overlay.def\n").unwrap();
        fs::write(bundle.0.join("overlay.def"), definition(2, rows)).unwrap();
        assert!(read_input_parameters(&mut data, bundle.0.join("namelist.def")).is_err());
        assert_eq!(data.projection_parameters(), before);
    }
}

#[test]
fn declared_gutz_and_jastrow_active_prefix_keeps_surplus_term_storage_untouched() {
    let (bundle, _) = model();
    let mut data = seeded_projection();
    let mut gutz_tail = data.gutzwiller_terms[0];
    gutz_tail.value = C::new(901.0, 902.0);
    data.gutzwiller_terms.push(gutz_tail);
    let mut jastrow_tail = data.jastrow_terms[0];
    jastrow_tail.value = C::new(903.0, 904.0);
    data.jastrow_terms.push(jastrow_tail);
    let mut expected = data.projection_parameters();
    fs::write(
        bundle.0.join("namelist.def"),
        "InGutzwiller gutz.def\nInJastrow jast.def\n",
    )
    .unwrap();
    for name in ["gutz.def", "jast.def"] {
        fs::write(bundle.0.join(name), definition(2, "0 1 2\n0 3 4\n")).unwrap();
    }
    read_input_parameters(&mut data, bundle.0.join("namelist.def")).unwrap();
    expected[0] = C::new(3.0, 4.0);
    expected[2] = C::new(3.0, 4.0);
    assert_eq!(data.projection_parameters(), expected);
    assert_eq!(data.gutzwiller_terms[2].value, gutz_tail.value);
    assert_eq!(data.jastrow_terms[2].value, jastrow_tail.value);
}

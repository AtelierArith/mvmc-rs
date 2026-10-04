use std::fs;
use std::path::Path;

mod support;
use support::{report_gate, GateStatus};

fn values(path: &Path) -> Vec<f64> {
    fs::read_to_string(path)
        .unwrap()
        .split_whitespace()
        .map(|value| value.parse::<f64>().unwrap())
        .collect()
}

fn require_empty_dc_contract(name: &str, parsed_gex_empty: bool, actual: &Path) {
    // C's successful mode2 zero-count LS writer emits exactly one LF.
    let regular_empty = fs::symlink_metadata(actual)
        .is_ok_and(|metadata| metadata.file_type().is_file() && metadata.len() == 1)
        && fs::read(actual).is_ok_and(|bytes| bytes == b"\n");
    if name != "zvo_ls_cisajscktaltex_001.dat" || !parsed_gex_empty || !regular_empty {
        support::missing_fixture(
            "lanczos-physcal",
            format!(
                "missing independent {name} reference; defined empty GEx contract not satisfied"
            ),
        );
    }
}

fn validate_selection(mode: &str, model: Option<&str>) {
    if !matches!(mode, "real" | "cmp")
        || model.is_some_and(|model| {
            !matches!(
                model,
                "hubbard_chain_real" | "hubbard_chain_lanczos" | "spin_chain_lanczos"
            )
        })
    {
        support::unsupported("lanczos-physcal", format!("mode={mode:?} model={model:?}"));
    }
}

#[test]
#[ignore = "optional Lanczos gate: MVMC_RS_LANCZOS_PHYSICAL required"]
fn serial_lanczos_matches_hubbard_and_exchange_references() {
    support::require_gate("lanczos-physcal", "MVMC_RS_LANCZOS_PHYSICAL");
    let root = support::julia_mvmc_root().unwrap_or_else(|| {
        support::missing_fixture("lanczos-physcal", "Julia-mVMC checkout not found")
    });
    let mode = std::env::var("MVMC_RS_LANCZOS_MODE").unwrap_or_else(|_| "real".into());
    let requested_model = std::env::var("MVMC_RS_LANCZOS_MODEL").ok();
    validate_selection(&mode, requested_model.as_deref());
    for model in [
        "hubbard_chain_real",
        "hubbard_chain_lanczos",
        "spin_chain_lanczos",
    ] {
        if requested_model
            .as_deref()
            .is_some_and(|requested| requested != model)
        {
            continue;
        }
        let fixture = root.join(format!("test/integration/reference/{model}/physcal_ref"));
        let namelist = fixture.join("inputs/namelist.def");
        let opt_para = fixture.join("zqp_opt.dat");
        for path in [
            &namelist,
            &opt_para,
            &fixture.join("expected/zvo_ls_qqqq_001.dat"),
            &fixture.join("expected/zvo_ls_out_001.dat"),
        ] {
            if !path.is_file() {
                support::missing_fixture("lanczos-physcal", path.display().to_string());
            }
        }

        let output =
            std::env::temp_dir().join(format!("mvmc-lanczos-{model}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&output);
        let preparation =
            mvmc_core::prepare_phys_cal_from_namelist(&namelist, &opt_para, &mode, Some(1))
                .unwrap();
        let parsed_gex_empty = preparation.data.green_two_ex_terms.is_empty()
            && preparation.data.green_two_ex_indices.is_empty();
        mvmc_core::vmc_phys_cal_to_dir(preparation, &output).unwrap();

        let actual_qqqq = values(&output.join("zvo_ls_qqqq_001.dat"));
        let expected_qqqq = values(&fixture.join("expected/zvo_ls_qqqq_001.dat"));
        assert_eq!(actual_qqqq.len(), 16, "{model}");
        assert_eq!(expected_qqqq.len(), 16, "{model}");
        for (actual, expected) in actual_qqqq.iter().zip(expected_qqqq.iter()) {
            assert!(
                (actual - expected).abs() <= 1.0e-8,
                "{model}: {actual} != {expected}"
            );
        }

        let actual_ls = values(&output.join("zvo_ls_out_001.dat"));
        let expected_ls = values(&fixture.join("expected/zvo_ls_out_001.dat"));
        assert_eq!(actual_ls.len(), 3, "{model}");
        assert_eq!(expected_ls.len(), 3, "{model}");
        for (actual, expected) in actual_ls.iter().zip(expected_ls.iter()) {
            assert!(
                (actual - expected).abs() <= 1.0e-8,
                "{model}: {actual} != {expected}"
            );
        }
        if model != "hubbard_chain_real" {
            for name in [
                "zvo_ls_cisajs_001.dat",
                "zvo_ls_cisajscktalt_001.dat",
                "zvo_ls_cisajscktaltex_001.dat",
            ] {
                let expected_path = fixture.join("expected").join(name);
                if !expected_path.is_file() {
                    require_empty_dc_contract(name, parsed_gex_empty, &output.join(name));
                    println!("OPTIONAL183_DC model={model} mode={mode} file={name} status=EMPTY_CONTRACT");
                    continue;
                }
                let actual = values(&output.join(name));
                let expected = values(&expected_path);
                assert_eq!(actual.len(), expected.len(), "{model}: {name}");
                for (actual, expected) in actual.iter().zip(expected.iter()) {
                    assert!(
                        (actual - expected).abs() <= 1.0e-8,
                        "{model} {name}: {actual} != {expected}"
                    );
                }
                println!("OPTIONAL183_DC model={model} mode={mode} file={name} status=REFERENCE_COMPARED");
            }
        }
        let _ = fs::remove_dir_all(output);
        report_gate("lanczos-physcal", GateStatus::Pass, model);
    }
}

#[test]
fn missing_dc_reference_requires_regular_empty_parsed_gex_contract() {
    let root = std::env::temp_dir().join(format!(
        "mvmc-dc-negative-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&root).unwrap();
    let actual = root.join("actual.dat");
    let name = "zvo_ls_cisajscktaltex_001.dat";
    assert!(std::panic::catch_unwind(|| require_empty_dc_contract(name, true, &actual)).is_err());
    for bytes in [b"".as_slice(), b" ", b"\r\n", b"\n\n", b"1\n"] {
        fs::write(&actual, bytes).unwrap();
        assert!(
            std::panic::catch_unwind(|| require_empty_dc_contract(name, true, &actual)).is_err()
        );
    }
    fs::write(&actual, b"\n").unwrap();
    require_empty_dc_contract(name, true, &actual);
    assert!(std::panic::catch_unwind(|| require_empty_dc_contract(name, false, &actual)).is_err());
    assert!(std::panic::catch_unwind(|| require_empty_dc_contract(
        "zvo_ls_cisajs_001.dat",
        true,
        &actual
    ))
    .is_err());
    assert!(std::panic::catch_unwind(|| require_empty_dc_contract(name, true, &root)).is_err());
    #[cfg(unix)]
    {
        let link = root.join("linked.dat");
        std::os::unix::fs::symlink(&actual, &link).unwrap();
        assert!(std::panic::catch_unwind(|| require_empty_dc_contract(name, true, &link)).is_err());
    }
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn unsupported_lanczos_selection_cannot_pass() {
    for (mode, model) in [
        ("", None),
        ("unknown", None),
        ("real", Some("interall")),
        ("cmp", Some("")),
    ] {
        assert!(std::panic::catch_unwind(|| validate_selection(mode, model)).is_err());
    }
    validate_selection("real", Some("spin_chain_lanczos"));
    validate_selection("cmp", Some("hubbard_chain_lanczos"));
}

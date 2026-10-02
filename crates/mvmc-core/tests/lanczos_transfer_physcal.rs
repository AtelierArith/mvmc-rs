use std::fs;
use std::path::{Path, PathBuf};

fn values(path: &Path) -> Vec<f64> {
    fs::read_to_string(path)
        .unwrap()
        .split_whitespace()
        .map(|value| value.parse::<f64>().unwrap())
        .collect()
}

#[test]
fn serial_lanczos_matches_hubbard_and_exchange_references() {
    if std::env::var_os("MVMC_RS_LANCZOS_PHYSICAL").is_none() {
        eprintln!("set MVMC_RS_LANCZOS_PHYSICAL=1 to run the Julia/C Lanczos PhysCal gate");
        return;
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../extern/Julia-mVMC");
    let mode = std::env::var("MVMC_RS_LANCZOS_MODE").unwrap_or_else(|_| "real".into());
    let requested_model = std::env::var("MVMC_RS_LANCZOS_MODEL").ok();
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
        if !namelist.is_file() || !opt_para.is_file() {
            eprintln!("Julia-mVMC reference submodule is unavailable; skipping {model}");
            continue;
        }

        let output =
            std::env::temp_dir().join(format!("mvmc-lanczos-{model}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&output);
        let preparation =
            mvmc_core::prepare_phys_cal_from_namelist(&namelist, &opt_para, &mode, Some(1))
                .unwrap();
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
            }
        }
        let _ = fs::remove_dir_all(output);
    }
}

//! M0331–340: original optional overlays and OptTrans input, bounded contracts.
//! Missing optional input is a Julia API policy, NOT native-C defined behavior:
//! readdef.c 1202–1206 calls fclose(NULL) in its failed-open branch.
use mvmc_expert_parsers::utils::read_input_parameters::read_input_parameters;
use mvmc_expert_parsers::{parse_expert_mode_files, parse_expert_mode_files_with_c_opt_trans};
use num_complex::Complex64 as C;
use std::fs;
use std::path::PathBuf;

struct Bundle(PathBuf);
impl Bundle {
    fn new(modpara: &str, namelist: &str, files: &[(&str, String)]) -> Self {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "issue184-opt-overlay-{}-{stamp}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        fs::write(path.join("modpara.def"), modpara).unwrap();
        fs::write(
            path.join("namelist.def"),
            format!("ModPara modpara.def\n{namelist}"),
        )
        .unwrap();
        for (name, text) in files {
            fs::write(path.join(name), text).unwrap();
        }
        Self(path)
    }
}
impl Drop for Bundle {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove exclusively owned optional-overlay bundle");
    }
}
fn record(header: &str, count: usize, rows: &str) -> String {
    format!("=============================================\n{header}          {count}\nComplexType         1\n=============================================\n=============================================\n{rows}")
}

#[test]
fn original_complex_gutzwiller_overlay_preserves_both_components() {
    let bundle = Bundle::new(
        "--------------------\nModel_Parameters 0\n--------------------\nVMC_Cal_Parameters\n--------------------\nCDataFileHead zvo\nCParaFileHead zqp\n--------------------\nNsite 2\nNe 1\n",
        "Gutzwiller g.def\nInGutzwiller in.def\n",
        &[
            ("g.def", record("NGutzwillerIdx", 1, "0 0\n1 0\n0 1\n")),
            ("in.def", record("NGutzwillerIdx", 1, "0 0.5 0.3\n")),
        ],
    );
    let mut data = parse_expert_mode_files(bundle.0.join("namelist.def")).unwrap();
    assert!(data.input_errors.is_empty());
    read_input_parameters(&mut data, bundle.0.join("namelist.def")).unwrap();
    assert_eq!(data.gutzwiller_terms[0].value, C::new(0.5, 0.3));
}

#[test]
fn julia_optional_missing_overlay_policy_preserves_existing_value() {
    // M0332 only; not a C acceptance/error contract or C parity claim.
    let bundle = Bundle::new(
        "--------------------\nModel_Parameters 0\n--------------------\nVMC_Cal_Parameters\n--------------------\nCDataFileHead zvo\nCParaFileHead zqp\n--------------------\nNsite 2\nNe 1\n",
        "Gutzwiller g.def\nInGutzwiller absent.def\n",
        &[("g.def", record("NGutzwillerIdx", 1, "0 0\n1 0\n0 1\n"))],
    );
    let mut data = parse_expert_mode_files(bundle.0.join("namelist.def")).unwrap();
    assert!(data.input_errors.is_empty());
    assert!(!bundle.0.join("absent.def").exists());
    let before = data.projection_parameters();
    read_input_parameters(&mut data, bundle.0.join("namelist.def")).unwrap();
    assert_eq!(data.gutzwiller_terms[0].value, C::new(0.0, 0.0));
    assert_eq!(data.projection_parameters(), before);
}

#[test]
fn original_opttrans_weights_swap_identity_and_periodic_signs_match_enabled_c_contract() {
    // Original test_read_input_parameters.jl 323–334 verbatim numeric payload;
    // unlike existing valid.def fixture, first weight is .25, not signed zero.
    let definition = "=============================================\nNQPOptTrans          2\n=============================================\n=============================================\n=============================================\n0 0.25\n1 0.75\n0 0 1 -1\n0 1 0 -1\n1 0 0 1\n1 1 1 -1\n";
    let bundle = Bundle::new(
        "--------------------\nModel_Parameters 0\n--------------------\nVMC_Cal_Parameters\n--------------------\nCDataFileHead zvo\nCParaFileHead zqp\n--------------------\nNsite 2\nNe 1\nNMPTrans 1\n",
        "OptTrans opt.def\n",
        &[("opt.def", definition.to_string())],
    );
    let data =
        parse_expert_mode_files_with_c_opt_trans(bundle.0.join("namelist.def"), true).unwrap();
    assert!(data.input_errors.is_empty());
    assert_eq!(data.n_qp_opt_trans, 2);
    assert_eq!(
        data.para_qp_opt_trans,
        [C::new(0.25, 0.0), C::new(0.75, 0.0)]
    );
    assert_eq!(data.opt_trans, data.para_qp_opt_trans);
    assert_eq!(data.qp_opt_trans, [vec![1, 0], vec![0, 1]]);
    assert_eq!(data.qp_opt_trans_sgn, [vec![1, 1], vec![1, 1]]);
}

#[test]
fn original_parallel_and_opttrans_overlays_keep_ap_offset_and_declared_unused_slots() {
    // Original Julia synthetic same-spin diagonal terms are not valid C input.
    // Supply complete AP rows and one valid P upper pair for Nsite2 instead.
    // Two declared P indices reserve four slots even if only index0 is mapped.
    let bundle = Bundle::new("--------------------\nModel_Parameters 0\n--------------------\nVMC_Cal_Parameters\n--------------------\nCDataFileHead zvo\nCParaFileHead zqp\n--------------------\nNsite 2\nNe 1\nNMPTrans 1\n", "OrbitalAntiParallel ap.def\nOrbitalParallel p.def\nOptTrans opt.def\nInOrbitalParallel inp.def\nInOptTrans inopt.def\n", &[
        ("ap.def",record("NOrbitalIdx",1,"0 0 0 1\n0 1 0 1\n1 0 0 1\n1 1 0 1\n0 1\n")),
        ("p.def",record("NOrbitalIdx",2,"0 1 0 1\n0 1\n1 1\n")),
        ("opt.def",record("NQPOptTrans",2,"0 1\n1 2\n0 0 0 1\n0 1 1 1\n1 0 1 1\n1 1 0 1\n")),
        ("inp.def",record("NOrbitalParallel",4,"0 10.0 0.1\n1 20.0 0.2\n2 30.0 0.3\n3 40.0 0.4\n")),
        ("inopt.def",record("NQPOptTrans",2,"1 0.25 -0.25\n0 0.50 -0.50\n")),
    ]);
    let mut data =
        parse_expert_mode_files_with_c_opt_trans(bundle.0.join("namelist.def"), true).unwrap();
    assert!(data.input_errors.is_empty());
    assert_eq!(data.n_orbital_anti_parallel, 1);
    assert_eq!(data.slater_params.len(), 5);
    read_input_parameters(&mut data, bundle.0.join("namelist.def")).unwrap();
    assert_eq!(
        data.slater_params,
        [
            C::new(0.0, 0.0),
            C::new(10.0, 0.1),
            C::new(20.0, 0.2),
            C::new(30.0, 0.3),
            C::new(40.0, 0.4)
        ]
    );
    assert_eq!(data.opt_trans, [C::new(0.50, -0.50), C::new(0.25, -0.25)]);
    // The two unmapped P slots are checked as declared parameters, not invented
    // spatial mappings or a full initial.def/sampling lifecycle proof.
}

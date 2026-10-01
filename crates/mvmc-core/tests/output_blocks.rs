//! Literal Julia data_io.jl output contracts, including its four-line headers.
use mvmc_core::{io::output_opt_data, ExpertModeData};
use mvmc_expert_parsers::{GutzwillerTerm, JastrowTerm, OrbitalTerm};
use num_complex::Complex64;
use std::fs;

#[test]
fn block_files_preserve_julia_headers_enumeration_and_flat_snapshot() {
    let dir = std::env::temp_dir().join(format!("mvmc-block-output-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let mut data = ExpertModeData::new();
    data.modpara.c_para_file_head = "custom".into();
    data.gutzwiller_terms.push(GutzwillerTerm {
        site: 9,
        value: Complex64::new(1.5, -2.0),
        is_complex: true,
    });
    data.jastrow_terms.push(JastrowTerm {
        site1: 3,
        site2: 7,
        value: Complex64::new(-2.0, 1.5),
        is_complex: true,
    });
    for idx in [3, 3] {
        data.orbital_terms.push(OrbitalTerm {
            site1: 0,
            site2: 1,
            idx,
            value: Complex64::new(1.5, -2.0),
            is_complex: true,
            sign: 1,
        });
    }
    output_opt_data(&data, Some(&dir)).unwrap();
    let row = " 1.500000000000000000e+00 -2.000000000000000000e+00 \n";
    let reverse = "-2.000000000000000000e+00  1.500000000000000000e+00 \n";
    let header = |name, count| {
        format!("===============================\n{name} {count}\n===============================\n===============================\n")
    };
    assert_eq!(
        fs::read_to_string(dir.join("custom_opt.dat")).unwrap(),
        format!("{row}{reverse}{row}{row}")
    );
    assert_eq!(
        fs::read_to_string(dir.join("custom_gutzwiller_opt.dat")).unwrap(),
        format!("{}0 {row}", header("NGutzwillerIdx", 1))
    );
    assert_eq!(
        fs::read_to_string(dir.join("custom_jastrow_opt.dat")).unwrap(),
        format!("{}0 {reverse}", header("NJastrowIdx", 1))
    );
    assert_eq!(
        fs::read_to_string(dir.join("custom_orbital_opt.dat")).unwrap(),
        format!("{}0 {row}1 {row}", header("NOrbitalIdx", 2))
    );
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn empty_blocks_are_skipped_and_no_directory_uses_the_given_prefix() {
    let dir = std::env::temp_dir().join(format!("mvmc-empty-output-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let mut data = ExpertModeData::new();
    data.modpara.c_para_file_head = dir.join("prefix").to_str().unwrap().to_owned();
    output_opt_data(&data, None).unwrap();
    assert_eq!(fs::read_to_string(dir.join("prefix_opt.dat")).unwrap(), "");
    assert!(!dir.join("prefix_gutzwiller_opt.dat").exists());
    assert!(!dir.join("prefix_jastrow_opt.dat").exists());
    assert!(!dir.join("prefix_orbital_opt.dat").exists());
    fs::remove_dir_all(dir).unwrap();
}

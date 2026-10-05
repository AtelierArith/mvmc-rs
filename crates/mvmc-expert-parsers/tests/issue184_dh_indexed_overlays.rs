//! M0341–342 original finite DH overlay values through C-complete definitions.
//! C ReadInputParameters 1250–1274 scatters six/ten rows by parameter index.
//! M0343–346's Julia atomic rejection expectations are NOT covered as C parity:
//! valid duplicate indices overwrite in C; fscanf accepts NaN. Those policies
//! require a separate native probe and parser-vs-runner validation audit.
use mvmc_expert_parsers::utils::read_input_parameters::read_input_parameters;
use mvmc_expert_parsers::{parse_expert_mode_files, ExpertModeData};
use num_complex::Complex64 as C;
use std::fs;
use std::path::PathBuf;

struct Bundle(PathBuf);
impl Drop for Bundle {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove exclusively owned DH overlay bundle");
    }
}
fn record(header: &str, rows: &str) -> String {
    format!("=============================================\n{header}          1\nComplexType         1\n=============================================\n=============================================\n{rows}")
}
fn load_original_finite_overlays() -> ExpertModeData {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "issue184-dh-overlay-{}-{stamp}",
        std::process::id()
    ));
    fs::create_dir(&path).unwrap();
    let bundle = Bundle(path);
    fs::write(bundle.0.join("modpara.def"), concat!("--------------------\nModel_Parameters 0\n--------------------\nVMC_Cal_Parameters\n--------------------\nCDataFileHead zvo\nCParaFileHead zqp\n--------------------\n", "Nsite 2\nNe 1\n")).unwrap();
    fs::write(
        bundle.0.join("namelist.def"),
        "ModPara modpara.def\nDH2 dh2.def\nDH4 dh4.def\nInDH2 in2.def\nInDH4 in4.def\n",
    )
    .unwrap();
    // Complete two-site tables reproduce the original Julia neighbor matrices;
    // unlike its synthetic constructor, dimensions/counts are publicly parsed.
    let flags2: String = (0..6).map(|i| format!("{i} 1\n")).collect();
    let flags4: String = (0..10).map(|i| format!("{i} 1\n")).collect();
    fs::write(
        bundle.0.join("dh2.def"),
        record(
            "NDoublonHolon2siteIdx",
            &format!("0 1 0 0\n1 0 1 0\n{flags2}"),
        ),
    )
    .unwrap();
    fs::write(
        bundle.0.join("dh4.def"),
        record(
            "NDoublonHolon4siteIdx",
            &format!("0 1 0 1 0 0\n1 0 1 0 1 0\n{flags4}"),
        ),
    )
    .unwrap();
    fs::write(
        bundle.0.join("in2.def"),
        record(
            "NDoublonHolon2siteIdx",
            "5 5.0 -5.0\n3 3.0 -3.0\n1 1.0 -1.0\n0 0.0 0.5\n2 2.0 -2.0\n4 4.0 -4.0\n",
        ),
    )
    .unwrap();
    // Original generator is i=1:10: TEN rows with values ELEVEN through TWENTY.
    let rows4: String = (1..=10)
        .map(|i| format!("{} {}.0 -{}.0\n", i - 1, 10 + i, 10 + i))
        .collect();
    fs::write(
        bundle.0.join("in4.def"),
        record("NDoublonHolon4siteIdx", &rows4),
    )
    .unwrap();
    let mut data = parse_expert_mode_files(bundle.0.join("namelist.def")).unwrap();
    assert!(data.input_errors.is_empty());
    assert_eq!(data.doublon_holon_2site_indices.len(), 1);
    assert_eq!(data.doublon_holon_4site_indices.len(), 1);
    assert_eq!(data.doublon_holon_2site_params, vec![C::new(0.0, 0.0); 6]);
    assert_eq!(data.doublon_holon_4site_params, vec![C::new(0.0, 0.0); 10]);
    read_input_parameters(&mut data, bundle.0.join("namelist.def")).unwrap();
    data
}

#[test]
fn original_six_scrambled_dh2_rows_scatter_into_declared_parameter_order() {
    let data = load_original_finite_overlays();
    assert_eq!(
        data.doublon_holon_2site_params,
        [
            C::new(0.0, 0.5),
            C::new(1.0, -1.0),
            C::new(2.0, -2.0),
            C::new(3.0, -3.0),
            C::new(4.0, -4.0),
            C::new(5.0, -5.0),
        ]
    );
}

#[test]
fn original_ten_dh4_rows_preserve_values_eleven_through_twenty() {
    let data = load_original_finite_overlays();
    assert_eq!(
        data.doublon_holon_4site_params,
        [
            C::new(11.0, -11.0),
            C::new(12.0, -12.0),
            C::new(13.0, -13.0),
            C::new(14.0, -14.0),
            C::new(15.0, -15.0),
            C::new(16.0, -16.0),
            C::new(17.0, -17.0),
            C::new(18.0, -18.0),
            C::new(19.0, -19.0),
            C::new(20.0, -20.0),
        ]
    );
}

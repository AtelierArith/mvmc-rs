//! M0320–330 original overlay bytes/values with public, C-complete definitions.
//! Julia's synthetic zero-site/sparse orbital objects are not native-C inputs.
//! Authority: readdef.c KWInGutzwiller 1221–1233, KWInOrbital 1408–1420,
//! KWInorbitalGeneral 1438–1448: scatter by declared parameter index, no signs.
use mvmc_expert_parsers::utils::read_input_parameters::read_input_parameters;
use mvmc_expert_parsers::{parse_expert_mode_files, ExpertModeData};
use num_complex::Complex64;
use std::fs;
use std::path::PathBuf;

struct Bundle(PathBuf);
impl Bundle {
    fn new(
        nsite: usize,
        keyword: &str,
        definition: &str,
        overlay_kind: &str,
        overlay: &str,
    ) -> Self {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("issue184-overlay-{}-{stamp}", std::process::id()));
        fs::create_dir(&path).unwrap();
        fs::write(
            path.join("modpara.def"),
            format!("Nsite {nsite}\nNElec 1\nNMPTrans -1\n"),
        )
        .unwrap();
        fs::write(path.join("definition.def"), definition).unwrap();
        fs::write(path.join("overlay.def"), overlay).unwrap();
        fs::write(
            path.join("namelist.def"),
            format!("ModPara modpara.def\n{keyword} definition.def\n{overlay_kind} overlay.def\n"),
        )
        .unwrap();
        Self(path)
    }
    fn load(&self) -> ExpertModeData {
        let mut data = parse_expert_mode_files(self.0.join("namelist.def")).unwrap();
        assert!(data.input_errors.is_empty());
        read_input_parameters(&mut data, self.0.join("namelist.def")).unwrap();
        data
    }
}
impl Drop for Bundle {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove exclusively owned overlay input bundle");
    }
}

fn record(header: &str, count: usize, rows: &str) -> String {
    format!("=============================================\n{header}          {count}\nComplexType         0\n=============================================\n=============================================\n{rows}")
}

#[test]
fn original_gutzwiller_overlay_updates_both_declared_slots() {
    let definition = record("NGutzwillerIdx", 2, "0 0\n1 1\n0 1\n1 1\n");
    let overlay = record("NGutzwillerIdx", 2, "0 0.5 0.0\n1 0.3 0.0\n");
    let bundle = Bundle::new(2, "Gutzwiller", &definition, "InGutzwiller", &overlay);
    let data = bundle.load();
    assert_eq!(data.gutzwiller_terms.len(), 2);
    assert_eq!(data.gutzwiller_terms[0].value, Complex64::new(0.5, 0.0));
    assert_eq!(data.gutzwiller_terms[1].value, Complex64::new(0.3, 0.0));
}

#[test]
fn original_three_orbital_overlay_values_scatter_into_complete_four_site_map() {
    let mut rows = String::new();
    for i in 0..4 {
        for j in 0..4 {
            let index = if i == 0 && j == 2 {
                1
            } else if i == 0 && j == 3 {
                2
            } else {
                0
            };
            rows.push_str(&format!("{i} {j} {index} 1\n"));
        }
    }
    rows.push_str("0 1\n1 1\n2 1\n");
    let definition = record("NOrbitalIdx", 3, &rows);
    let overlay = record("NOrbitalIdx", 3, "0 1.0 0.0\n1 2.0 0.0\n2 3.0 0.0\n");
    let bundle = Bundle::new(4, "Orbital", &definition, "InOrbital", &overlay);
    let data = bundle.load();
    assert_eq!(
        data.slater_params,
        [
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0)
        ]
    );
    for (site, expected) in [(1, 1.0), (2, 2.0), (3, 3.0)] {
        let term = data
            .orbital_terms
            .iter()
            .find(|term| term.site1 == 0 && term.site2 == site)
            .unwrap();
        assert_eq!(
            data.slater_params[term.idx as usize],
            Complex64::new(expected, 0.0)
        );
    }
}

#[test]
fn original_scrambled_normal_overlay_reuses_parameter_indices_without_sign_scaling() {
    let definition = record(
        "NOrbitalIdx",
        2,
        "0 0 1 1\n0 1 0 1\n1 0 1 -1\n1 1 0 1\n0 1\n1 1\n",
    );
    let overlay = record("NOrbitalIdx", 2, "1 20.0 -0.2\n0 10.0 -0.1\n");
    let bundle = Bundle::new(2, "Orbital", &definition, "InOrbital", &overlay);
    let data = bundle.load();
    let mapped: Vec<_> = data.orbital_terms[..3]
        .iter()
        .map(|term| data.slater_params[term.idx as usize])
        .collect();
    assert_eq!(
        mapped,
        [
            Complex64::new(20.0, -0.2),
            Complex64::new(10.0, -0.1),
            Complex64::new(20.0, -0.2)
        ]
    );
    assert_eq!(data.orbital_terms[2].sign, -1);
    assert_eq!(
        data.slater_params,
        [Complex64::new(10.0, -0.1), Complex64::new(20.0, -0.2)]
    );
}

#[test]
fn original_scrambled_general_overlay_uses_valid_complete_pairs_and_dense_slots() {
    // Original Julia three-term object includes a diagonal pair and Nsite=0.
    // Replace only spatial definitions with six valid upper-triangle pairs;
    // retain index order 1/0/1 and signs 1/1/-1 for the compared values.
    let mut rows = String::new();
    let mut pair = 0;
    for i in 0..4 {
        for j in i + 1..4 {
            let index = if pair % 2 == 0 { 1 } else { 0 };
            let sign = if pair == 2 { -1 } else { 1 };
            rows.push_str(&format!(
                "{} {} {} {} {index} {sign}\n",
                i % 2,
                i / 2,
                j % 2,
                j / 2
            ));
            pair += 1;
        }
    }
    rows.push_str("0 1\n1 1\n");
    let definition = record("NOrbitalIdx", 2, &rows);
    let overlay = record("NOrbitalIdx", 2, "1 22.0 -0.4\n0 11.0 -0.3\n");
    let bundle = Bundle::new(
        2,
        "OrbitalGeneral",
        &definition,
        "InOrbitalGeneral",
        &overlay,
    );
    let data = bundle.load();
    let mapped: Vec<_> = data.orbital_terms[..3]
        .iter()
        .map(|term| data.slater_params[term.idx as usize])
        .collect();
    assert_eq!(
        mapped,
        [
            Complex64::new(22.0, -0.4),
            Complex64::new(11.0, -0.3),
            Complex64::new(22.0, -0.4)
        ]
    );
    assert_eq!(data.orbital_terms[2].sign, -1);
    assert_eq!(
        data.slater_params,
        [Complex64::new(11.0, -0.3), Complex64::new(22.0, -0.4)]
    );
}

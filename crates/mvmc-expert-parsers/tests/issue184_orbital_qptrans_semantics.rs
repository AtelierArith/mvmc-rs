//! Source-semantic coverage for Julia M0093–103, M0109–130 and M0145–148.
//! Sparse Julia orbital constructors are completed before exercising the native
//! C reader contract; they are not claimed as valid native-C sparse inputs.
use mvmc_expert_parsers::parse_expert_mode_files;
use std::fs;
use std::path::PathBuf;

struct Bundle(PathBuf);
impl Bundle {
    fn new(nsite: usize, nmp: i64, keyword: &str, definition: &str) -> Self {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("issue184-orbital-{}-{stamp}", std::process::id()));
        fs::create_dir(&path).unwrap();
        fs::write(
            path.join("modpara.def"),
            format!("--------------------\nModel_Parameters 0\n--------------------\nVMC_Cal_Parameters\n--------------------\nCDataFileHead zvo\nCParaFileHead zqp\n--------------------\nNsite {nsite}\nNe 1\nNMPTrans {nmp}\n"),
        )
        .unwrap();
        fs::write(
            path.join("namelist.def"),
            format!("ModPara modpara.def\n{keyword} tested.def\n"),
        )
        .unwrap();
        fs::write(path.join("tested.def"), definition).unwrap();
        Self(path)
    }
    fn parse(&self) -> mvmc_expert_parsers::ExpertModeData {
        let data = parse_expert_mode_files(self.0.join("namelist.def")).unwrap();
        assert!(data.input_errors.is_empty());
        data
    }
}
impl Drop for Bundle {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove exclusively owned semantic input bundle");
    }
}

#[test]
fn normal_and_general_orbital_signs_follow_complete_c_reader_contract() {
    // readdef.c GetInfoOrbitalAntiParallel: Nsite² rows plus NArray flags,
    // APFlag=0 overwrites all signs +1; APFlag=1 preserves file signs.
    let rows: String = (0..4)
        .flat_map(|i| {
            (0..4).map(move |j| {
                let sign = if (i, j) == (0, 1) { -1 } else { 1 };
                let index = if (i, j) == (0, 1) {
                    1
                } else if (i, j) == (1, 2) {
                    2
                } else {
                    0
                };
                format!("{i} {j} {index} {sign}\n")
            })
        })
        .collect();
    let definition = format!("===\nNOrbitalIdx 3\nComplexType 0\n===\n===\n{rows}0 1\n1 1\n2 1\n");
    for nmp in [4, -4] {
        let bundle = Bundle::new(4, nmp, "Orbital", &definition);
        let data = bundle.parse();
        let (_, signs) = data.build_orbital_matrices();
        assert_eq!(signs.len(), 4);
        assert!(signs.iter().all(|row| row.len() == 4));
        assert_eq!(signs[0][0], 1);
        assert_eq!(signs[0][1], if nmp > 0 { 1 } else { -1 });
        assert_eq!(signs[1][2], 1);
        if nmp > 0 {
            assert!(signs.iter().flatten().all(|&sign| sign == 1));
        }
    }
    // C GetInfoOrbitalGeneral requires all six upper-triangle pairs for Nsite2.
    // Original Julia flattened (0,1),(2,3) pair signs are retained in this
    // complete site/spin definition; reversed entries must be antisymmetric.
    let mut rows = String::new();
    for i in 0..4 {
        for j in i + 1..4 {
            let sign = if (i, j) == (2, 3) { -1 } else { 1 };
            rows.push_str(&format!(
                "{} {} {} {} 0 {sign}\n",
                i % 2,
                i / 2,
                j % 2,
                j / 2
            ));
        }
    }
    let definition = format!("===\nNOrbitalIdx 1\nComplexType 0\n===\n===\n{rows}0 1\n");
    for nmp in [2, -2] {
        let bundle = Bundle::new(2, nmp, "OrbitalGeneral", &definition);
        let data = bundle.parse();
        let (_, signs) = data.build_orbital_matrices();
        assert_eq!(signs.len(), 4);
        assert!(signs.iter().all(|row| row.len() == 4));
        assert_eq!(signs[0][1], 1);
        assert_eq!(signs[1][0], -1);
        assert_eq!(signs[2][3], if nmp > 0 { 1 } else { -1 });
        for (i, row) in signs.iter().enumerate() {
            for (j, &sign) in row.iter().enumerate().skip(i + 1) {
                assert_eq!(signs[j][i], -sign);
            }
        }
    }
}

#[test]
fn original_four_site_identity_cyclic_and_default_opttrans_are_loaded() {
    // Literal mappings and weights from original Julia lines 166–179. Native
    // GetInfoTransSym requires exactly Nsite*NArray=8 mapping rows.
    let definition = "===\nNQPTrans 2\n===\n===\n===\n0 1.00000\n1 1.00000\n0 0 0 1\n0 1 1 1\n0 2 2 1\n0 3 3 1\n1 0 1 1\n1 1 2 1\n1 2 3 1\n1 3 0 1\n";
    let bundle = Bundle::new(4, 2, "TransSym", definition);
    let data = bundle.parse();
    assert_eq!(data.n_qp_trans, 2);
    assert_eq!(data.qp_trans_entries.len(), 2);
    assert_eq!(data.qp_trans_entries[0].site_map, [0, 1, 2, 3]);
    assert_eq!(data.qp_trans_entries[1].site_map, [1, 2, 3, 0]);
    for entry in &data.qp_trans_entries {
        assert_eq!(entry.site_map.len(), 4);
        assert_eq!(entry.site_sign, [1, 1, 1, 1]);
        assert_eq!(entry.weight, num_complex::Complex64::new(1.0, 0.0));
    }
    // Rust deliberately exposes entries rather than Julia's parallel inverse
    // arrays. Do not manufacture an inverse here and label it API coverage.
    assert_eq!(data.qp_opt_trans.len(), 1);
    assert_eq!(data.qp_opt_trans[0], [0, 1, 2, 3]);
    assert_eq!(data.qp_opt_trans_sgn[0], [1, 1, 1, 1]);
}

#[test]
fn original_antiperiodic_cyclic_input_preserves_all_four_signs() {
    let definition =
        "===\nNQPTrans 1\n===\n===\n===\n0 1.00000\n0 0 1 -1\n0 1 2 1\n0 2 3 -1\n0 3 0 1\n";
    let bundle = Bundle::new(4, -4, "TransSym", definition);
    let data = bundle.parse();
    assert_eq!(data.qp_trans_entries.len(), 1);
    assert_eq!(data.qp_trans_entries[0].site_map, [1, 2, 3, 0]);
    assert_eq!(data.qp_trans_entries[0].site_sign, [-1, 1, -1, 1]);
}

#[test]
fn original_heisenberg_sixteen_site_orbital_sample_has_periodic_signs() {
    // M0106–108: exact original sample, not the distinct fresh20 model input.
    // M0104–105 are Julia availability/skip branches, not semantic executions.
    let definition =
        include_str!("../../../tests/fixtures/issue184_heisenberg_parser/orbitalidx.def");
    let bundle = Bundle::new(16, 4, "Orbital", definition);
    let data = bundle.parse();
    assert_eq!(data.orbital_terms.len(), 16 * 16);
    assert_eq!(data.modpara.n_orbital_idx, 64);
    let (_, signs) = data.build_orbital_matrices();
    assert_eq!(signs.len(), 16);
    assert!(signs.iter().all(|row| row.len() == 16));
    assert!(signs.iter().flatten().all(|&sign| sign == 1));
}

#[test]
fn original_heisenberg_sixteen_site_translation_sample_and_default_identity() {
    // M0133–135, M0138–140 (four sign rows), M0142–144.
    // M0131–132 are availability skips; M0136–137/M0141 assert Julia inverse
    // arrays, absent from this Rust parser API and deliberately not claimed.
    let definition =
        include_str!("../../../tests/fixtures/issue184_heisenberg_parser/qptransidx.def");
    let bundle = Bundle::new(16, 4, "TransSym", definition);
    let data = bundle.parse();
    assert_eq!(data.n_qp_trans, 4);
    assert_eq!(data.qp_trans_entries.len(), 4);
    for entry in &data.qp_trans_entries {
        assert_eq!(entry.site_map.len(), 16);
        assert_eq!(entry.site_sign.len(), 16);
        assert!(entry.site_sign.iter().all(|&sign| sign == 1));
        assert_eq!(entry.weight, num_complex::Complex64::new(1.0, 0.0));
    }
    assert_eq!(data.qp_opt_trans.len(), 1);
    assert_eq!(data.qp_opt_trans[0], (0..16).collect::<Vec<_>>());
    assert_eq!(data.qp_opt_trans_sgn[0], vec![1; 16]);
}

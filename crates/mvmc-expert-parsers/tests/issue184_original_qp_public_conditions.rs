//! Original M0111–130/M0135–144 conditions through the public file loader.
//! Julia8bb test_orbital_qptrans_utils.jl:150–289 supplies the independent inputs.
//! C readdef.c:2232–2265 defines complete forward/inverse/sign tables. Rust
//! derives inverse rows with a fresh query instead of storing Julia's parallel
//! inverse array. No original helper's mutation/error API or model is certified.
use mvmc_expert_parsers::{parse_expert_mode_files, ExpertModeData};
use std::{fs, path::PathBuf};

const FOUR_SITE: &str = "===\nNQPTrans 2\n===\n===\n===\n0 1.00000\n1 1.00000\n0 0 0 1\n0 1 1 1\n0 2 2 1\n0 3 3 1\n1 0 1 1\n1 1 2 1\n1 2 3 1\n1 3 0 1\n";
const SIXTEEN_SITE: &str =
    include_str!("../../../tests/fixtures/issue184_heisenberg_parser/qptransidx.def");

struct Bundle(PathBuf);
impl Bundle {
    fn new(nsite: usize, nelec: usize, nlocspin: usize, nmp: usize, definition: &str) -> Self {
        // Exclusive allocation: a collision never grants ownership of a preexisting directory.
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "issue184-original-qp-{}-{stamp}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        let bundle = Self(path);
        fs::write(
            bundle.0.join("modpara.def"),
            format!("Nsite {nsite}\nNElec {nelec}\nNLocSpin {nlocspin}\nNMPTrans {nmp}\n"),
        )
        .unwrap();
        fs::write(
            bundle.0.join("namelist.def"),
            "ModPara modpara.def\nTransSym qptransidx.def\n",
        )
        .unwrap();
        fs::write(bundle.0.join("qptransidx.def"), definition).unwrap();
        bundle
    }

    fn parse(&self) -> ExpertModeData {
        let data = parse_expert_mode_files(self.0.join("namelist.def")).unwrap();
        assert!(data.input_errors.is_empty(), "{:?}", data.input_errors);
        data
    }
}
impl Drop for Bundle {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove only exclusively allocated input bundle");
    }
}

#[test]
fn original_four_site_public_forward_inverse_sign_and_default_opttrans_conditions() {
    let bundle = Bundle::new(4, 2, 0, 2, FOUR_SITE);
    let data = bundle.parse();
    assert_eq!(data.modpara.nsite, 4);
    assert_eq!(data.modpara.nelec, 2);
    assert_eq!(data.modpara.nlocspin, 0);
    assert_eq!(data.modpara.nmp_trans, 2);
    // M0111–120: two widths and all eight original explicit forward values.
    assert_eq!(data.qp_trans_entries.len(), 2);
    assert_eq!(data.qp_trans_entries[0].site_map.len(), 4);
    assert_eq!(data.qp_trans_entries[1].site_map.len(), 4);
    assert_eq!(data.qp_trans_entries[0].site_map, [0, 1, 2, 3]);
    assert_eq!(data.qp_trans_entries[1].site_map, [1, 2, 3, 0]);
    let before = data.qp_trans_entries.clone();
    // M0121–123: actual public query from publicly loaded rows, not a test inverse.
    let inverses: Vec<_> = data
        .qp_trans_entries
        .iter()
        .map(|entry| entry.inverse_site_map().unwrap())
        .collect();
    assert!(!inverses.is_empty());
    assert_eq!(inverses.len(), 2);
    assert_eq!(inverses[1][1], 0);
    assert_eq!(data.qp_trans_entries, before);
    // M0124–127: parallel sign rows are represented inside each public entry.
    assert!(!data.qp_trans_entries.is_empty());
    assert_eq!(data.qp_trans_entries.len(), 2);
    for entry in &data.qp_trans_entries {
        assert_eq!(entry.site_sign.len(), 4);
        assert!(entry.site_sign.iter().all(|&sign| sign == 1));
    }
    // M0128–130: disabled OptTrans retains one default identity.
    assert!(!data.qp_opt_trans.is_empty());
    assert_eq!(data.qp_opt_trans.len(), 1);
    assert_eq!(data.qp_opt_trans[0], [0, 1, 2, 3]);
}

#[test]
fn original_sixteen_site_public_inverse_identities_signs_and_default_opttrans_conditions() {
    // Original sample bytes, not newly generated Rust expectations. NLocSpin16
    // matches the original ModPara helper setting; no LocSpin file/model claim.
    let bundle = Bundle::new(16, 8, 16, 4, SIXTEEN_SITE);
    let data = bundle.parse();
    assert_eq!(data.modpara.nsite, 16);
    assert_eq!(data.modpara.nelec, 8);
    assert_eq!(data.modpara.nlocspin, 16);
    assert_eq!(data.modpara.nmp_trans, 4);
    assert_eq!(data.qp_trans_entries[0].site_map.len(), 16); // M0135
    let before = data.qp_trans_entries.clone();
    let inverses: Vec<_> = data
        .qp_trans_entries
        .iter()
        .map(|entry| entry.inverse_site_map().unwrap())
        .collect();
    assert!(!inverses.is_empty()); // M0136
    assert_eq!(inverses.len(), 4); // M0137
    assert!(!data.qp_trans_entries.is_empty()); // M0138
    assert_eq!(data.qp_trans_entries.len(), 4); // M0139
    let mut checked = 0;
    for (entry, inverse) in data.qp_trans_entries.iter().zip(&inverses) {
        assert_eq!(entry.site_map.len(), 16);
        assert_eq!(entry.site_sign.len(), 16);
        assert!(entry.site_sign.iter().all(|&sign| sign == 1)); // M0140: four rows
        assert_eq!(inverse.len(), 16);
        for (origin, &translated) in entry.site_map.iter().enumerate() {
            assert_eq!(inverse[usize::try_from(translated).unwrap()], origin); // M0141
            checked += 1;
        }
    }
    assert_eq!(checked, 64);
    assert_eq!(data.qp_trans_entries, before);
    assert!(!data.qp_opt_trans.is_empty()); // M0142
    assert_eq!(data.qp_opt_trans.len(), 1); // M0143
    assert_eq!(data.qp_opt_trans[0], (0..16).collect::<Vec<_>>()); // M0144
}

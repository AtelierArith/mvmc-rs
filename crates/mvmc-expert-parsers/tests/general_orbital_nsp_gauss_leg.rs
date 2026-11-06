//! C readdef.c:703-707 replaces `NSPGaussLeg > 1` by 1 for general orbitals (#719).

use std::fs;
use std::path::{Path, PathBuf};

use mvmc_expert_parsers::parse_expert_mode_files;

struct TempInput(PathBuf);

impl Drop for TempInput {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn parsed_nsp_gauss_leg(orbital_lines: &str, nsp_gauss_leg: i64) -> (i64, i64) {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures");
    let source = fixtures.join("c_orbital_inputs");
    let id = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("mvmc-rs-nspgaussleg-{}-{id}", std::process::id()));
    fs::create_dir(&dir).unwrap();
    let guard = TempInput(dir.clone());
    for name in [
        "ap_general_three.def",
        "p_general_three.def",
        "general_three.def",
    ] {
        fs::copy(source.join(name), dir.join(name)).unwrap();
    }
    let modpara = fs::read_to_string(fixtures.join("orbital_general/modpara.def"))
        .unwrap()
        .replace("NSPGaussLeg 1", &format!("NSPGaussLeg {nsp_gauss_leg}"));
    assert!(modpara.contains(&format!("NSPGaussLeg {nsp_gauss_leg}")));
    fs::write(dir.join("modpara.def"), modpara).unwrap();
    fs::write(
        dir.join("namelist.def"),
        format!("ModPara modpara.def\n{orbital_lines}"),
    )
    .unwrap();
    let data = parse_expert_mode_files(guard.0.join("namelist.def")).unwrap();
    (data.i_flg_orbital_general, data.modpara.nsp_gauss_leg)
}

#[test]
fn general_orbitals_force_nsp_gauss_leg_to_one() {
    for orbitals in [
        "OrbitalGeneral general_three.def\n",
        "Orbital ap_general_three.def\nOrbitalParallel p_general_three.def\n",
    ] {
        assert_eq!(parsed_nsp_gauss_leg(orbitals, 8), (1, 1), "{orbitals}");
    }
}

#[test]
fn normal_orbitals_keep_nsp_gauss_leg() {
    assert_eq!(
        parsed_nsp_gauss_leg("Orbital ap_general_three.def\n", 8),
        (0, 8)
    );
}

#[test]
fn general_orbitals_keep_nsp_gauss_leg_zero_or_one() {
    for leg in [0, 1] {
        assert_eq!(
            parsed_nsp_gauss_leg("OrbitalGeneral general_three.def\n", leg),
            (1, leg)
        );
    }
}

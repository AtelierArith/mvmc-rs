//! #350: input contract of the C ComplexUHF reader (`readdef.c`) and the C
//! defects that the Rust port deliberately does not reproduce.
use std::fs;
use std::path::{Path, PathBuf};

use mvmc_uhf::{
    c_fixed, c_space_exp18, convergence_threshold, read_definition, run, OrbitalOutputMode,
    UhfError, UhfOptions,
};

struct TestDir(PathBuf);

impl TestDir {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!("mvmc-uhf-def-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn write(&self, name: &str, text: &str) {
        fs::write(self.0.join(name), text).unwrap();
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

const RULE: &str = "=====\n";

fn header(title: &str, count: usize) -> String {
    format!("{RULE}{title} {count}\n{RULE}{RULE}{RULE}")
}

fn modpara(extra: &str) -> String {
    format!(
        "-----\nModel_Parameters 0\n-----\nVMC_Cal_Parameters\n-----\nCDataFileHead zvo\n\
         CParaFileHead zqp\n-----\nNVMCCalMode 0\nNsite 4\n{extra}"
    )
}

/// Four-site ring, `Ncond=4`, AP orbital with idx = (j-i) mod 4.
fn base(name: &str, modpara_extra: &str, namelist_extra: &str) -> TestDir {
    let dir = TestDir::new(name);
    dir.write(
        "namelist.def",
        &format!(
            "ModPara modpara.def\nLocSpin locspn.def\nTrans trans.def\n\
             CoulombIntra coulombintra.def\nOrbital orbital.def\n{namelist_extra}"
        ),
    );
    dir.write("modpara.def", &modpara(modpara_extra));
    dir.write(
        "locspn.def",
        &(header("NlocalSpin", 0) + "0 0\n1 0\n2 0\n3 0\n"),
    );
    let mut trans = String::new();
    for i in 0..4 {
        for spin in 0..2 {
            let j = (i + 1) % 4;
            trans += &format!("{i} {spin} {j} {spin} 1.0 0.0\n{j} {spin} {i} {spin} 1.0 0.0\n");
        }
    }
    dir.write("trans.def", &(header("NTransfer", 16) + &trans));
    dir.write(
        "coulombintra.def",
        &(header("NCoulombIntra", 4) + "0 3.0\n1 3.0\n2 3.0\n3 3.0\n"),
    );
    let mut orbital = String::new();
    for i in 0..4 {
        for j in 0..4 {
            orbital += &format!("{i} {j} {}\n", (j + 4 - i) % 4);
        }
    }
    dir.write("orbital.def", &(header("NOrbitalIdx", 4) + &orbital));
    dir
}

fn definition(dir: &TestDir) -> Result<mvmc_uhf::UhfDefinition, UhfError> {
    read_definition(Path::new("namelist.def"), &dir.0)
}

fn message(result: Result<mvmc_uhf::UhfDefinition, UhfError>) -> String {
    result.unwrap_err().to_string()
}

const GOOD: &str = "Ncond 4\n2Sz 0\n";

#[test]
fn defaults_and_modpara_keys_follow_c() {
    let dir = base("defaults", GOOD, "");
    let def = definition(&dir).unwrap();
    assert_eq!((def.mix, def.eps_int, def.print), (0.5, 10, 0));
    assert_eq!(
        (def.iteration_max, def.rnd_seed, def.eps_int_slater),
        (2000, 0, 6)
    );
    assert_eq!((def.nsite, def.ncond, def.ne, def.nsize), (4, 4, 2, 4));
    assert_eq!(def.orbital_mode, Some(OrbitalOutputMode::AntiParallel));
    assert_eq!(def.data_file_head, "zvo");
    assert_eq!(def.para_file_head, "zqp");

    // Keys are case-insensitive; unused mVMC keys only warn; unknown keys warn.
    let dir = base(
        "keys",
        "ncond 4\n2SZ 0\nMIX 0.25\neps 7\nITERATIONMAX 33\nrndseed 5\nepsslater 2\nPrint 1\n\
         NSROptItrStep 100\nNotAKey 3\nNelectron 9\n",
        "",
    );
    let def = definition(&dir).unwrap();
    assert_eq!((def.mix, def.eps_int, def.iteration_max), (0.25, 7, 33));
    assert_eq!((def.rnd_seed, def.eps_int_slater, def.print), (5, 2, 1));
    // Ne is overwritten by the Ncond rule when an orbital file is declared.
    assert_eq!(def.ne, 2);
}

#[test]
fn ncond_rule_and_orbital_mode_judgement_follow_c() {
    // Ne (documented) is not enough: with an orbital file Ne=(NLocSpn+Ncond)/2.
    let text = message(definition(&base("no-ncond", "Ne 4\n2Sz 0\n", "")));
    assert!(text.contains("Ncond"), "{text}");
    assert!(message(definition(&base("odd", "Ncond 3\n2Sz 0\n", "")))
        .contains("NCond (in modpara.def) must be even"));
    // 2Sz is -1 unless given, which C rejects for AP-only orbitals.
    assert!(message(definition(&base("sz", "Ncond 4\n", ""))).contains("2Sz=-1"));
    assert!(
        message(definition(&base("sz-explicit", "Ncond 4\n2Sz -1\n", "")))
            .contains("2Sz must be even")
    );
    // Nsize must leave a gap index (C reads EigenValues[-1] / [2*Nsite]).
    assert!(message(definition(&base("full", "Ncond 8\n2Sz 0\n", ""))).contains("Nsize=8"));
    // OrbitalParallel without Orbital, and General together with AP, are errors.
    let dir = base("parallel-only", GOOD, "OrbitalParallel orbital.def\n");
    let text = fs::read_to_string(dir.0.join("namelist.def")).unwrap();
    dir.write("namelist.def", &text.replace("Orbital orbital.def\n", ""));
    assert!(message(definition(&dir)).contains("Need OrbitalAP"));
    let dir = base("general-and-ap", GOOD, "OrbitalGeneral orbital.def\n");
    assert!(message(definition(&dir)).contains("Multiple definition"));
    // Orbital + OrbitalParallel selects the AP+P mode and forces General reading.
    let dir = base("ap-p", GOOD, "OrbitalParallel parallel.def\n");
    dir.write(
        "parallel.def",
        &(header("NOrbitalIdx", 1) + "0 1 0\n0 2 0\n"),
    );
    let def = definition(&dir).unwrap();
    assert_eq!(
        def.orbital_mode,
        Some(OrbitalOutputMode::AntiParallelAndParallel)
    );
    assert_eq!(
        (def.n_orbital_ap, def.n_orbital_p, def.n_orbital_idx),
        (4, 2, 6)
    );
    // Parallel indices are NOrbitalAP + 2*idx + spin, antisymmetric.
    let n2 = 8;
    assert_eq!(def.orbital_idx[1], 4); // (0,1) up-up
    assert_eq!(def.orbital_idx[4 * n2 + 5], 5); // (0,1) down-down
    assert_eq!(def.orbital_sgn[1], 1);
    assert_eq!(def.orbital_sgn[n2], -1);
    assert_eq!(def.orbital_idx[n2 + 4 + 2], 1); // AP block (1, 2+Nsite): (2-1)%4
}

#[test]
fn record_counts_pair_hop_and_lenient_readers_follow_c() {
    let dir = base("records", GOOD, "PairHop pairhop.def\nHund hund.def\n");
    dir.write("pairhop.def", &(header("NPairhop", 1) + "0 1 0.5\n"));
    dir.write("hund.def", &(header("NHund", 1) + "2 3 0.25\n"));
    let def = definition(&dir).unwrap();
    // Each file record is stored twice, as (i,j) and (j,i).
    assert_eq!(def.pair_hopping, vec![(0, 1, 0.5), (1, 0, 0.5)]);
    assert_eq!(def.hund, vec![(2, 3, 0.25)]);

    // C `ReadDefFileError` returns 0: a short Trans file is reported but
    // accepted, the missing records stay zero. Extra records are dropped
    // (C overflows its heap array).
    let dir = base("short", GOOD, "");
    dir.write("trans.def", &(header("NTransfer", 3) + "0 0 1 0 1.0 0.0\n"));
    let def = definition(&dir).unwrap();
    assert_eq!(def.transfer.len(), 3);
    assert_eq!(def.transfer[0].value.re, 1.0);
    assert_eq!(def.transfer[2].value.re, 0.0);
    dir.write(
        "trans.def",
        &(header("NTransfer", 1) + "0 0 1 0 1.0 0.0\n1 0 0 0 7.0 0.0\n"),
    );
    assert_eq!(definition(&dir).unwrap().transfer.len(), 1);
}

#[test]
fn c_crashes_become_errors() {
    // C: fopen failure followed by fclose(NULL).
    let dir = base("missing", GOOD, "TransSym qptrans.def\n");
    assert!(matches!(definition(&dir), Err(UhfError::Io { .. })));
    // C: unknown namelist keyword writes cFileNameList[-1]; Rust ignores it.
    let dir = base("unknown-keyword", GOOD, "NoSuchKeyword whatever.def\n");
    assert!(definition(&dir).is_ok());
    // Duplicate keyword: an error in C and Rust.
    let dir = base("duplicate", GOOD, "Trans trans.def\n");
    assert!(message(definition(&dir)).contains("Same keywords"));
    // C writes OrbitalIdx before checking the site index; Rust checks first.
    let dir = base("bad-orbital-site", GOOD, "");
    dir.write("orbital.def", &(header("NOrbitalIdx", 4) + "9 0 0\n"));
    assert!(message(definition(&dir)).contains("Site index"));
    // Out-of-range spin-orbital in Trans.
    let dir = base("bad-trans", GOOD, "");
    dir.write("trans.def", &(header("NTransfer", 1) + "0 0 7 0 1.0 0.0\n"));
    assert!(message(definition(&dir)).contains("Trans"));
    // Required keywords.
    let dir = base("no-modpara", GOOD, "");
    dir.write("namelist.def", "LocSpin locspn.def\n");
    assert!(message(definition(&dir)).contains("ModPara"));
}

#[test]
fn run_error_paths_do_not_write_partial_results() {
    let dir = base("run-missing", GOOD, "TransSym absent.def\n");
    let options = UhfOptions {
        base_dir: dir.0.clone(),
    };
    assert!(run(Path::new("namelist.def"), &options).is_err());
    assert!(!dir.0.join("zvo_check.dat").exists());
}

#[test]
fn c_number_formats() {
    assert_eq!(c_space_exp18(1.0), " 1.000000000000000000e+00");
    assert_eq!(
        c_space_exp18(-(0.5_f64.powi(23))),
        "-1.192092895507812500e-07"
    );
    assert_eq!(c_space_exp18(0.0), " 0.000000000000000000e+00");
    assert_eq!(
        c_space_exp18(2.0_f64.powi(400)),
        " 2.582249878086908590e+120"
    );
    assert_eq!(c_fixed(-0.0, 6), "-0.000000");
    assert_eq!(c_fixed(2.5e-7, 6), "0.000000");
    assert_eq!(c_fixed(f64::INFINITY, 6), "inf");
    // eps is 0.1 multiplied eps_int times (not 10^-eps_int).
    assert_eq!(convergence_threshold(0), 1.0);
    assert_eq!(convergence_threshold(-3), 1.0);
    let mut expected = 1.0_f64;
    for _ in 0..10 {
        expected *= 0.1;
    }
    assert_eq!(convergence_threshold(10).to_bits(), expected.to_bits());
}

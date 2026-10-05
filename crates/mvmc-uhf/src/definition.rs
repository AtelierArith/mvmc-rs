//! Port of `ComplexUHF/readdef.c`.
//!
//! The ComplexUHF reader is a dedicated, deliberately lenient reader that is
//! **not** the Expert-mode reader used by `vmc.out`: it has its own ModPara
//! keys (`Mix`, `EPS`, `Print`, `IterationMax`, `EpsSlater`, `Ncond` ...), its
//! own orbital-mode judgement, `sscanf`/`fscanf` prefix semantics and an
//! `Initial` Green-function file. The Expert-mode parsers in
//! `mvmc-expert-parsers` therefore cannot be reused without changing what the
//! C tool accepts, and this module follows the C file instead.
//!
//! Documented deviations from the C source (all of them C defects or
//! undefined behaviour, see `docs/COMPLEX_UHF.md`):
//!
//! * a missing definition file is reported as an error (C calls
//!   `fclose(NULL)` and crashes);
//! * a namelist keyword that is not in the keyword list is ignored with a
//!   warning (C writes `cFileNameList[-1]` or the previous keyword's slot);
//! * records beyond the declared counts are dropped (C writes past the end of
//!   its heap arrays); a record count mismatch is only reported, because the C
//!   `ReadDefFileError` returns 0 and therefore never fails;
//! * out-of-range site/spin/orbital indices are errors (C reads or writes out
//!   of bounds);
//! * `Nsize` outside `1..2*Nsite-1` is an error (C reads
//!   `EigenValues[-1]`/`EigenValues[2*Nsite]` for the gap).

use std::fs;
use std::path::{Path, PathBuf};

use num_complex::Complex64;

use crate::scan::Scan;
use crate::UhfError;

/// Number of header lines skipped in every definition file (`IgnoreLinesInDef`).
const IGNORE_LINES_IN_DEF: usize = 5;

/// Keyword list of `cKWListOfFileNameList` in C enum order.
const KEYWORDS: [&str; 29] = [
    "ModPara",
    "LocSpin",
    "Trans",
    "CoulombIntra",
    "CoulombInter",
    "Hund",
    "PairHop",
    "Exchange",
    "Gutzwiller",
    "Jastrow",
    "DH2",
    "DH4",
    "Orbital",
    "OrbitalAntiParallel",
    "OrbitalParallel",
    "OrbitalGeneral",
    "TransSym",
    "InGutzwiller",
    "InJastrow",
    "InDH2",
    "InDH4",
    "InOrbital",
    "OneBodyG",
    "TwoBodyG",
    "TwoBodyGEx",
    "InterAll",
    "OptTrans",
    "InOptTrans",
    "Initial",
];

const KW_MODPARA: usize = 0;
const KW_LOCSPIN: usize = 1;
const KW_TRANS: usize = 2;
const KW_COULOMB_INTRA: usize = 3;
const KW_COULOMB_INTER: usize = 4;
const KW_HUND: usize = 5;
const KW_PAIR_HOP: usize = 6;
const KW_EXCHANGE: usize = 7;
const KW_ORBITAL: usize = 12;
const KW_ORBITAL_ANTI_PARALLEL: usize = 13;
const KW_ORBITAL_PARALLEL: usize = 14;
const KW_ORBITAL_GENERAL: usize = 15;
const KW_ONE_BODY_G: usize = 22;
const KW_INTER_ALL: usize = 25;
const KW_INITIAL: usize = 28;

/// Orbital output selection (`OrbitalOutputMode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrbitalOutputMode {
    /// `OrbitalGeneral` only: `{para}_General_Fij.dat`, `{para}_GeneralOrbital_opt.dat`.
    General,
    /// `Orbital`/`OrbitalAntiParallel` only: spin-block re-diagonalisation.
    AntiParallel,
    /// `Orbital` plus `OrbitalParallel`: AP and P files.
    AntiParallelAndParallel,
}

/// A parsed one-body record `(site1, spin1, site2, spin2)` with a complex value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OneBodyTerm {
    /// Site/spin indices as in the file.
    pub index: [i32; 4],
    /// Coefficient (transfer integral or initial Green function).
    pub value: Complex64,
}

/// Everything `ReadDefFileNInt` + `ReadDefFileIdxPara` store in `DefineList`
/// that the ComplexUHF calculation reads.
#[derive(Debug, Clone)]
pub struct UhfDefinition {
    /// `Nsite`.
    pub nsite: usize,
    /// `Ne` after the `Ncond` rule (see [`read_definition`]).
    pub ne: i32,
    /// `Ncond` (default 0).
    pub ncond: i32,
    /// `2Sz` (default -1 sentinel).
    pub two_sz: i32,
    /// Linear mixing parameter `Mix` (default 0.5).
    pub mix: f64,
    /// `EPS` (default 10).
    pub eps_int: i32,
    /// `Print` (default 0).
    pub print: i32,
    /// `IterationMax` (default 2000).
    pub iteration_max: i32,
    /// `RndSeed` (default 0).
    pub rnd_seed: i32,
    /// `EpsSlater` (default 6).
    pub eps_int_slater: i32,
    /// `CDataFileHead`.
    pub data_file_head: String,
    /// `CParaFileHead`.
    pub para_file_head: String,
    /// Number of occupied spin orbitals `Nsize`.
    pub nsize: usize,
    /// Number of localized spins (`NlocalSpin` header).
    pub n_loc_spn: i32,
    /// `Trans` records.
    pub transfer: Vec<OneBodyTerm>,
    /// `CoulombIntra` records `(site, U)`.
    pub coulomb_intra: Vec<(i32, f64)>,
    /// `CoulombInter` records `(site1, site2, V)`.
    pub coulomb_inter: Vec<(i32, i32, f64)>,
    /// `Hund` records `(site1, site2, J)`.
    pub hund: Vec<(i32, i32, f64)>,
    /// `PairHop` records, each file record expanded to `(i,j)` and `(j,i)`.
    pub pair_hopping: Vec<(i32, i32, f64)>,
    /// `Exchange` records `(site1, site2, J)`.
    pub exchange: Vec<(i32, i32, f64)>,
    /// `InterAll` records.
    pub inter_all: Vec<([i32; 8], Complex64)>,
    /// Declared `NInitial` (random start only when exactly 0).
    pub n_initial: i32,
    /// `Initial` Green-function records.
    pub initial: Vec<OneBodyTerm>,
    /// A declared orbital definition exists (`iFlgOrbital == 1`).
    pub orbital_flag: bool,
    /// Orbital output mode when `orbital_flag` is set.
    pub orbital_mode: Option<OrbitalOutputMode>,
    /// `NOrbitalAP`.
    pub n_orbital_ap: usize,
    /// `NOrbitalP` (twice the declared parallel count).
    pub n_orbital_p: usize,
    /// `NOrbitalIdx`.
    pub n_orbital_idx: usize,
    /// `OrbitalIdx[2*Nsite][2*Nsite]` row-major (`-1` = not output).
    pub orbital_idx: Vec<i32>,
    /// `OrbitalSgn[2*Nsite][2*Nsite]` row-major.
    pub orbital_sgn: Vec<i32>,
}

/// Raw counts and flags of the first pass (`ReadDefFileNInt`).
#[derive(Default)]
struct Counts {
    n_transfer: i32,
    n_coulomb_intra: i32,
    n_coulomb_inter: i32,
    n_hund: i32,
    n_pair_hopping: i32,
    n_exchange: i32,
    n_inter_all: i32,
    n_initial: i32,
}

fn broken(name: &str) {
    // C `ReadDefFileError` prints this and returns 0 (so it never fails).
    println!("Error: {name} (Broken file or Not exist)");
}

fn read_text(path: &Path, shown: &str) -> Result<String, UhfError> {
    match fs::read(path) {
        Ok(bytes) => Ok(String::from_utf8_lossy(&bytes).into_owned()),
        Err(source) => {
            broken(shown);
            Err(UhfError::Io {
                path: path.to_path_buf(),
                source,
            })
        }
    }
}

/// `fgets` line iteration: lines keep their newline, the last may lack it.
fn lines(text: &str) -> Vec<&str> {
    text.split_inclusive('\n').collect()
}

fn definition_error(message: impl Into<String>) -> UhfError {
    UhfError::Definition(message.into())
}

/// `ReadBuffInt`: skip a header line, then `sscanf("%s %d")` the second line.
fn read_buff_int(name: &str, text_lines: &[&str], current: &mut i32) -> Result<(), UhfError> {
    if text_lines.len() < 2 {
        return Err(definition_error(format!(
            "Definition files(*.def) are incomplete. Error: Read File {name} ."
        )));
    }
    let mut scan = Scan::new(text_lines[1]);
    if scan.word().is_some() {
        if let Some(value) = scan.int() {
            *current = value;
        }
    }
    Ok(())
}

/// `GetFileName`: namelist keyword -> file name. Unknown keywords warn.
fn get_file_names(namelist: &Path, shown: &str) -> Result<Vec<String>, UhfError> {
    let text = read_text(namelist, shown)?;
    let mut names = vec![String::new(); KEYWORDS.len()];
    for line in lines(&text) {
        let mut scan = Scan::new(line);
        let keyword = scan.word().unwrap_or("");
        let file_name = scan.word().unwrap_or("");
        if keyword.starts_with('#')
            || line.starts_with('\n')
            || keyword.is_empty()
            || file_name.is_empty()
        {
            continue;
        }
        let Some(index) = KEYWORDS
            .iter()
            .position(|known| known.eq_ignore_ascii_case(keyword))
        else {
            eprintln!("Warning: Wrong keywords '{keyword}' in {shown}.");
            continue;
        };
        if !names[index].is_empty() {
            eprintln!("Error: Same keywords exist in {shown}.");
            return Err(definition_error(format!("Same keywords exist in {shown}.")));
        }
        names[index] = file_name.to_string();
    }
    Ok(names)
}

fn judge_orbital_mode(
    general: &mut bool,
    anti_parallel: bool,
    parallel: bool,
) -> Result<OrbitalOutputMode, UhfError> {
    if *general {
        if !anti_parallel && !parallel {
            Ok(OrbitalOutputMode::General)
        } else {
            Err(definition_error("Multiple definition of Orbital files."))
        }
    } else if anti_parallel {
        if parallel {
            *general = true;
            Ok(OrbitalOutputMode::AntiParallelAndParallel)
        } else {
            Ok(OrbitalOutputMode::AntiParallel)
        }
    } else {
        Err(definition_error(
            "Not exist any Orbital file or Need OrbitalAP file.",
        ))
    }
}

fn site_ok(site: i32, nsite: usize) -> bool {
    site >= 0 && (site as usize) < nsite
}

fn spin_site(site: i32, spin: i32, nsite: usize, what: &str) -> Result<usize, UhfError> {
    if !site_ok(site, nsite) || !(0..=1).contains(&spin) {
        return Err(definition_error(format!(
            "{what}: site {site} / spin {spin} is outside Nsite={nsite} and spin 0/1."
        )));
    }
    Ok(site as usize + spin as usize * nsite)
}

/// Read the ModPara keys, counts, files and orbital maps like C
/// `ReadDefFileNInt` followed by `ReadDefFileIdxPara`.
///
/// Relative definition-file names (including the namelist) are resolved
/// against `base`; use `Path::new("")` for the current directory as in C.
pub fn read_definition(namelist: &Path, base: &Path) -> Result<UhfDefinition, UhfError> {
    let resolve = |name: &str| -> PathBuf { base.join(name) };
    let namelist_shown = namelist.to_string_lossy().into_owned();
    println!("  Read File {namelist_shown} .");
    let names = get_file_names(&resolve(&namelist_shown), &namelist_shown).inspect_err(|_| {
        eprintln!("  error: Definition files(*.def) are incomplete.");
    })?;
    for index in [KW_MODPARA, KW_LOCSPIN] {
        if names[index].is_empty() {
            eprintln!("  Error: Need to make a def file for {}.", KEYWORDS[index]);
            return Err(definition_error(format!(
                "Need to make a def file for {}.",
                KEYWORDS[index]
            )));
        }
    }

    let mut def = UhfDefinition {
        nsite: 0,
        ne: 0,
        ncond: 0,
        two_sz: -1,
        mix: 0.5,
        eps_int: 10,
        print: 0,
        iteration_max: 2000,
        rnd_seed: 0,
        eps_int_slater: 6,
        data_file_head: String::new(),
        para_file_head: String::new(),
        nsize: 0,
        n_loc_spn: 0,
        transfer: Vec::new(),
        coulomb_intra: Vec::new(),
        coulomb_inter: Vec::new(),
        hund: Vec::new(),
        pair_hopping: Vec::new(),
        exchange: Vec::new(),
        inter_all: Vec::new(),
        n_initial: 0,
        initial: Vec::new(),
        orbital_flag: false,
        orbital_mode: None,
        n_orbital_ap: 0,
        n_orbital_p: 0,
        n_orbital_idx: 0,
        orbital_idx: Vec::new(),
        orbital_sgn: Vec::new(),
    };
    let mut counts = Counts::default();
    let mut nsite_raw: i32 = 0;
    let mut orbital_general = false;
    let mut flag_ap = false;
    let mut flag_p = false;
    let mut n_orbital_idx_raw: i32 = 0;
    let mut n_orbital_ap_raw: i32 = 0;
    let mut n_orbital_p_raw: i32 = 0;

    for (kw, name) in names.iter().enumerate() {
        if name.is_empty() {
            continue;
        }
        println!("  Read File '{name}' for {}.", KEYWORDS[kw]);
        let text = read_text(&resolve(name), name)?;
        let text_lines = lines(&text);
        match kw {
            KW_MODPARA => {
                if text_lines.len() < 8 {
                    return Err(definition_error(format!(
                        "Definition files(*.def) are incomplete. Error: Read File {name} ."
                    )));
                }
                let mut scan = Scan::new(text_lines[5]);
                let _ = scan.word();
                def.data_file_head = scan.word().unwrap_or("").to_string();
                let mut scan = Scan::new(text_lines[6]);
                let _ = scan.word();
                def.para_file_head = scan.word().unwrap_or("").to_string();
                if def.data_file_head.is_empty() || def.para_file_head.is_empty() {
                    return Err(definition_error(
                        "CDataFileHead/CParaFileHead must be given on lines 6 and 7 of ModPara.",
                    ));
                }
                let mut dtmp = 0.0_f64;
                for line in &text_lines[8..] {
                    if line.starts_with('\n') || line.starts_with('-') {
                        continue;
                    }
                    let mut scan = Scan::new(line);
                    let Some(key) = scan.word() else {
                        continue;
                    };
                    if let Some(value) = scan.float() {
                        dtmp = value;
                    }
                    parse_modpara_key(key, dtmp, &mut def, &mut nsite_raw)?;
                }
            }
            KW_LOCSPIN => read_buff_int(name, &text_lines, &mut def.n_loc_spn)?,
            KW_TRANS => read_buff_int(name, &text_lines, &mut counts.n_transfer)?,
            KW_COULOMB_INTRA => read_buff_int(name, &text_lines, &mut counts.n_coulomb_intra)?,
            KW_COULOMB_INTER => read_buff_int(name, &text_lines, &mut counts.n_coulomb_inter)?,
            KW_HUND => read_buff_int(name, &text_lines, &mut counts.n_hund)?,
            KW_PAIR_HOP => {
                read_buff_int(name, &text_lines, &mut counts.n_pair_hopping)?;
                counts.n_pair_hopping = counts.n_pair_hopping.wrapping_mul(2);
            }
            KW_EXCHANGE => read_buff_int(name, &text_lines, &mut counts.n_exchange)?,
            KW_INTER_ALL => read_buff_int(name, &text_lines, &mut counts.n_inter_all)?,
            KW_ORBITAL | KW_ORBITAL_ANTI_PARALLEL => {
                def.orbital_flag = true;
                let mut value = 0;
                read_buff_int(name, &text_lines, &mut value)?;
                n_orbital_ap_raw = value;
                flag_ap = true;
                n_orbital_idx_raw = n_orbital_idx_raw.wrapping_add(value);
            }
            KW_ORBITAL_PARALLEL => {
                def.orbital_flag = true;
                let mut value = 0;
                read_buff_int(name, &text_lines, &mut value)?;
                n_orbital_p_raw = value.wrapping_mul(2);
                flag_p = true;
                n_orbital_idx_raw = n_orbital_idx_raw.wrapping_add(value.wrapping_mul(2));
            }
            KW_ORBITAL_GENERAL => {
                def.orbital_flag = true;
                read_buff_int(name, &text_lines, &mut n_orbital_idx_raw)?;
                orbital_general = true;
            }
            KW_ONE_BODY_G => {
                let mut ignored = 0;
                read_buff_int(name, &text_lines, &mut ignored)?;
            }
            KW_INITIAL => read_buff_int(name, &text_lines, &mut counts.n_initial)?,
            _ => {}
        }
    }

    if def.orbital_flag {
        def.orbital_mode = Some(judge_orbital_mode(&mut orbital_general, flag_ap, flag_p)?);
    }

    // CalcNCond
    if def.ncond != -1 {
        if def.orbital_flag {
            if def.ncond % 2 != 0 {
                return Err(definition_error(
                    "NCond (in modpara.def) must be even number.",
                ));
            }
            def.ne = (def.n_loc_spn + def.ncond) / 2;
            def.nsize = usize::try_from(2 * def.ne).unwrap_or(0);
        } else {
            def.nsize = usize::try_from(def.n_loc_spn + def.ncond).unwrap_or(0);
        }
    }

    // CheckGeneral Orbital
    if def.two_sz != 0 && def.orbital_flag {
        if !orbital_general {
            return Err(definition_error(format!(
                "2Sz={}: OrbitalParallel or OrbitalGeneral files must be needed when 2Sz !=0 (in modpara.def).",
                def.two_sz
            )));
        } else if def.two_sz % 2 != 0 && def.two_sz != -1 {
            return Err(definition_error(
                "2Sz (in modpara.def) must be even number.",
            ));
        }
    }

    if nsite_raw <= 0 {
        return Err(definition_error(
            "Nsite must be a positive integer in ModPara.",
        ));
    }
    def.nsite = nsite_raw as usize;
    if def.nsize == 0 || def.nsize >= 2 * def.nsite {
        return Err(definition_error(format!(
            "Nsize={} (from Ncond and LocSpin) must satisfy 0 < Nsize < 2*Nsite={} \
             because the C tool reads EigenValues[Nsize-1] and EigenValues[Nsize] for the gap \
             (set Ncond in ModPara; Ne is only used when no orbital file is declared).",
            def.nsize,
            2 * def.nsite
        )));
    }
    def.n_orbital_ap = usize::try_from(n_orbital_ap_raw).unwrap_or(0);
    def.n_orbital_p = usize::try_from(n_orbital_p_raw).unwrap_or(0);
    def.n_orbital_idx = usize::try_from(n_orbital_idx_raw).unwrap_or(0);

    def.n_initial = counts.n_initial;
    read_idx_para(&names, base, &counts, orbital_general, &mut def)?;
    validate_terms(&def)?;
    Ok(def)
}

/// Reject site/spin indices that C would use out of bounds.
fn validate_terms(def: &UhfDefinition) -> Result<(), UhfError> {
    let nsite = def.nsite;
    for term in &def.transfer {
        let [s1, p1, s2, p2] = term.index;
        spin_site(s1, p1, nsite, "Trans")?;
        spin_site(s2, p2, nsite, "Trans")?;
    }
    for term in &def.initial {
        let [s1, p1, s2, p2] = term.index;
        spin_site(s1, p1, nsite, "Initial")?;
        spin_site(s2, p2, nsite, "Initial")?;
    }
    for &(site, _) in &def.coulomb_intra {
        spin_site(site, 0, nsite, "CoulombIntra")?;
    }
    for (what, list) in [
        ("CoulombInter", &def.coulomb_inter),
        ("Hund", &def.hund),
        ("PairHop", &def.pair_hopping),
        ("Exchange", &def.exchange),
    ] {
        for &(a, b, _) in list {
            spin_site(a, 0, nsite, what)?;
            spin_site(b, 0, nsite, what)?;
        }
    }
    for (index, _) in &def.inter_all {
        for quad in index.chunks(2) {
            spin_site(quad[0], quad[1], nsite, "InterAll")?;
        }
    }
    Ok(())
}

fn parse_modpara_key(
    key: &str,
    dtmp: f64,
    def: &mut UhfDefinition,
    nsite_raw: &mut i32,
) -> Result<(), UhfError> {
    const UNUSED: [&str; 19] = [
        "NVMCCalMode",
        "NLanczosMode",
        "NDataIdxStart",
        "NDataQtySmp",
        "NDataQtySmp",
        "NDataQtySmp",
        "NSPGaussLeg",
        "NSPStot",
        "NSROptItrStep",
        "NSROptItrSmp",
        "DSROptRedCut",
        "DSROptStaDel",
        "DSROptStepDt",
        "NVMCWarmUp",
        "NVMCInterval",
        "NVMCSample",
        "NExUpdatePath",
        "NSplitSize",
        "NStore",
    ];
    let is = |word: &str| key.eq_ignore_ascii_case(word);
    if UNUSED.iter().any(|word| is(word)) {
        println!("!! Warning: {key} is not used for Hatree Fock Calculation. !!");
    } else if is("Nsite") {
        *nsite_raw = dtmp as i32;
    } else if is("Ne") || is("Nelectron") {
        def.ne = dtmp as i32;
    } else if is("Ncond") {
        def.ncond = dtmp as i32;
    } else if is("2Sz") {
        def.two_sz = dtmp as i32;
        if def.two_sz == -1 {
            println!("Error: 2Sz must be even number.");
            return Err(definition_error("2Sz must be even number."));
        }
    } else if is("Mix") {
        def.mix = dtmp;
    } else if is("EPS") {
        def.eps_int = dtmp as i32;
    } else if is("Print") {
        def.print = dtmp as i32;
    } else if is("IterationMax") {
        def.iteration_max = dtmp as i32;
    } else if is("RndSeed") {
        def.rnd_seed = dtmp as i32;
    } else if is("EpsSlater") {
        def.eps_int_slater = dtmp as i32;
    } else if is("NMPTrans") {
        // Stored by C as NMPTrans/APFlag; not used by ComplexUHF.
    } else {
        println!("  Warning: keyword \" {key} \" is incorrect. ");
    }
    Ok(())
}

/// Integers and doubles of one `fscanf` record.
type Record = (Vec<i32>, Vec<f64>);

/// `fscanf` record stream of `n_ints` integers then `n_floats` doubles.
///
/// Returns each record's values; a record cut short by EOF keeps zeros for
/// the missing fields, like the C arrays.
fn scan_records(
    name: &str,
    body: &str,
    n_ints: usize,
    n_floats: usize,
) -> Result<Vec<Record>, UhfError> {
    let mut scan = Scan::new(body);
    let mut records = Vec::new();
    loop {
        let mut ints = vec![0_i32; n_ints];
        let mut floats = vec![0.0_f64; n_floats];
        let mut got = 0;
        let total = n_ints + n_floats;
        for (k, slot) in ints.iter_mut().enumerate() {
            match scan.int() {
                Some(value) => {
                    *slot = value;
                    got = k + 1;
                }
                None => break,
            }
        }
        if got == n_ints {
            for (k, slot) in floats.iter_mut().enumerate() {
                match scan.float() {
                    Some(value) => {
                        *slot = value;
                        got = n_ints + k + 1;
                    }
                    None => break,
                }
            }
        }
        if got == 0 {
            // EOF, or non-numeric text (infinite loop in C).
            if scan.word().is_some() {
                return Err(definition_error(format!(
                    "{name}: non-numeric record text (C fscanf would loop forever)."
                )));
            }
            return Ok(records);
        }
        records.push((ints, floats));
        if got < total {
            // C returns a short count (!= EOF) and the next call sees EOF.
            if scan.word().is_some() {
                return Err(definition_error(format!(
                    "{name}: malformed record (C fscanf would loop forever)."
                )));
            }
            return Ok(records);
        }
    }
}

fn read_idx_para(
    names: &[String],
    base: &Path,
    counts: &Counts,
    orbital_general: bool,
    def: &mut UhfDefinition,
) -> Result<(), UhfError> {
    let nsite = def.nsite;
    let two_n = 2 * nsite;
    if def.orbital_flag {
        def.orbital_idx = vec![0; two_n * two_n];
        def.orbital_sgn = vec![0; two_n * two_n];
        let limit = if orbital_general { two_n } else { nsite };
        for i in 0..limit {
            for j in 0..limit {
                def.orbital_idx[i * two_n + j] = -1;
                def.orbital_sgn[i * two_n + j] = 0;
            }
        }
    }

    for (kw, name) in names.iter().enumerate().skip(KW_LOCSPIN) {
        if name.is_empty() {
            continue;
        }
        let path = base.join(name);
        let text = read_text(&path, name)?;
        let text_lines = lines(&text);
        if text_lines.len() < IGNORE_LINES_IN_DEF {
            return Err(definition_error(format!(
                "{name}: fewer than {IGNORE_LINES_IN_DEF} header lines."
            )));
        }
        let body_lines = &text_lines[IGNORE_LINES_IN_DEF..];
        let body: String = body_lines.concat();
        match kw {
            KW_LOCSPIN => {
                let mut idx = 0_i64;
                for line in body_lines {
                    let mut scan = Scan::new(line);
                    let x0 = scan.int().unwrap_or(0);
                    if !site_ok(x0, nsite) {
                        return Err(definition_error(format!(
                            "{name}: LocSpin site {x0} outside Nsite={nsite}."
                        )));
                    }
                    idx += 1;
                }
                if i64::from(2 * def.ne) < i64::from(def.n_loc_spn) {
                    eprintln!("Error: 2*Ne must be (2*Ne >= NLocalSpin).");
                    return Err(definition_error("2*Ne must be (2*Ne >= NLocalSpin)."));
                }
                if idx != nsite as i64 {
                    broken(name);
                }
            }
            KW_TRANS | KW_INITIAL => {
                let declared = if kw == KW_TRANS {
                    counts.n_transfer
                } else {
                    counts.n_initial
                };
                if declared <= 0 {
                    continue;
                }
                let mut terms = vec![
                    OneBodyTerm {
                        index: [0; 4],
                        value: Complex64::new(0.0, 0.0),
                    };
                    declared as usize
                ];
                let mut d_re = 0.0_f64;
                let mut idx = 0_i64;
                for line in body_lines {
                    let mut scan = Scan::new(line);
                    let mut index = [0_i32; 4];
                    let mut d_im = 0.0_f64;
                    'fields: {
                        for slot in index.iter_mut() {
                            match scan.int() {
                                Some(value) => *slot = value,
                                None => break 'fields,
                            }
                        }
                        if let Some(value) = scan.float() {
                            d_re = value;
                            if let Some(im) = scan.float() {
                                d_im = im;
                            }
                        }
                    }
                    if let Some(term) = terms.get_mut(idx as usize) {
                        // C stores parsed prefix fields; missing ones stay zero.
                        term.index = index;
                        term.value = Complex64::new(d_re, d_im);
                    }
                    idx += 1;
                }
                if idx != i64::from(declared) {
                    broken(name);
                }
                if kw == KW_TRANS {
                    def.transfer = terms;
                } else {
                    def.initial = terms;
                }
            }
            KW_COULOMB_INTRA => {
                if counts.n_coulomb_intra > 0 {
                    let mut items = vec![(0_i32, 0.0_f64); counts.n_coulomb_intra as usize];
                    let mut idx = 0_i64;
                    for line in body_lines {
                        let mut scan = Scan::new(line);
                        let site = scan.int();
                        let value = site.and_then(|_| scan.float());
                        if let Some(item) = items.get_mut(idx as usize) {
                            *item = (site.unwrap_or(0), value.unwrap_or(0.0));
                        }
                        idx += 1;
                    }
                    if idx != i64::from(counts.n_coulomb_intra) {
                        broken(name);
                    }
                    def.coulomb_intra = items;
                }
            }
            KW_COULOMB_INTER => {
                if counts.n_coulomb_inter > 0 {
                    let mut items = vec![(0_i32, 0_i32, 0.0_f64); counts.n_coulomb_inter as usize];
                    let mut idx = 0_i64;
                    for line in body_lines {
                        let mut scan = Scan::new(line);
                        let a = scan.int();
                        let b = a.and_then(|_| scan.int());
                        let v = b.and_then(|_| scan.float());
                        if let Some(item) = items.get_mut(idx as usize) {
                            *item = (a.unwrap_or(0), b.unwrap_or(0), v.unwrap_or(0.0));
                        }
                        idx += 1;
                    }
                    if idx != i64::from(counts.n_coulomb_inter) {
                        broken(name);
                    }
                    def.coulomb_inter = items;
                }
            }
            KW_HUND | KW_EXCHANGE | KW_PAIR_HOP => {
                let declared = match kw {
                    KW_HUND => counts.n_hund,
                    KW_EXCHANGE => counts.n_exchange,
                    _ => counts.n_pair_hopping,
                };
                if declared > 0 {
                    let records = scan_records(name, &body, 2, 1)?;
                    let stored: Vec<(i32, i32, f64)> = records
                        .iter()
                        .map(|(ints, floats)| (ints[0], ints[1], floats[0]))
                        .collect();
                    if kw == KW_PAIR_HOP {
                        let mut items = Vec::new();
                        for &(a, b, v) in stored.iter().take((declared / 2) as usize) {
                            items.push((a, b, v));
                            items.push((b, a, v));
                        }
                        if stored.len() as i64 != i64::from(declared) / 2 {
                            broken(name);
                        }
                        def.pair_hopping = items;
                    } else {
                        if stored.len() as i64 != i64::from(declared) {
                            broken(name);
                        }
                        let items: Vec<_> = stored.into_iter().take(declared as usize).collect();
                        if kw == KW_HUND {
                            def.hund = items;
                        } else {
                            def.exchange = items;
                        }
                    }
                }
            }
            KW_INTER_ALL => {
                if counts.n_inter_all > 0 {
                    let records = scan_records(name, &body, 8, 2)?;
                    if records.len() as i64 != i64::from(counts.n_inter_all) {
                        broken(name);
                    }
                    def.inter_all = records
                        .into_iter()
                        .take(counts.n_inter_all as usize)
                        .map(|(ints, floats)| {
                            let mut index = [0_i32; 8];
                            index.copy_from_slice(&ints);
                            (index, Complex64::new(floats[0], floats[1]))
                        })
                        .collect();
                }
            }
            KW_ORBITAL | KW_ORBITAL_ANTI_PARALLEL => {
                read_orbital_ap(name, body_lines, orbital_general, def)?;
            }
            KW_ORBITAL_PARALLEL => read_orbital_parallel(name, body_lines, def)?,
            KW_ORBITAL_GENERAL => {
                if def.n_orbital_idx > 0 {
                    read_orbital_general(name, body_lines, def)?;
                }
            }
            KW_ONE_BODY_G => {}
            _ => {
                println!("!! Warning: {name} is not used for Hatree Fock Calculation. !!");
            }
        }
    }
    Ok(())
}

/// Parse one orbital line `"%d %d %d [%d]"`; fewer than three integers is an
/// error (C would reuse stale values).
fn orbital_line(name: &str, line: &str, want: usize) -> Result<Vec<i32>, UhfError> {
    let mut scan = Scan::new(line);
    let mut values = Vec::new();
    for _ in 0..want {
        match scan.int() {
            Some(value) => values.push(value),
            None => break,
        }
    }
    if values.len() < want - 1 {
        broken(name);
        return Err(definition_error(format!(
            "{name}: orbital record needs at least {} integers: {line:?}",
            want - 1
        )));
    }
    Ok(values)
}

fn check_orbital_index(name: &str, index: i32, def: &UhfDefinition) -> Result<(), UhfError> {
    if index == -1 || (index >= 0 && (index as usize) < def.n_orbital_idx) {
        Ok(())
    } else {
        Err(definition_error(format!(
            "{name}: orbital index {index} outside 0..{} (C indexes out of bounds).",
            def.n_orbital_idx
        )))
    }
}

fn read_orbital_ap(
    name: &str,
    body_lines: &[&str],
    general: bool,
    def: &mut UhfDefinition,
) -> Result<(), UhfError> {
    let nsite = def.nsite;
    let two_n = 2 * nsite;
    let mut idx = 0_usize;
    for line in body_lines {
        let values = orbital_line(name, line, 4)?;
        let (i, j, orbital) = (values[0], values[1], values[2]);
        let sign = if values.len() == 3 { 1 } else { values[3] };
        if !site_ok(i, nsite) || !site_ok(j, nsite) {
            eprintln!("Error: Site index is incorrect. ");
            return Err(definition_error("Site index is incorrect."));
        }
        check_orbital_index(name, orbital, def)?;
        let (i, j) = (i as usize, j as usize);
        def.orbital_idx[i * two_n + (j + nsite)] = orbital;
        def.orbital_sgn[i * two_n + (j + nsite)] = sign;
        if general {
            // Note F_IJ = -F_JI
            def.orbital_idx[(j + nsite) * two_n + i] = orbital;
            def.orbital_sgn[(j + nsite) * two_n + i] = -sign;
        }
        idx += 1;
        if idx == nsite * nsite {
            break;
        }
    }
    if !general && idx != nsite * nsite {
        broken(name);
    }
    Ok(())
}

fn read_orbital_parallel(
    name: &str,
    body_lines: &[&str],
    def: &mut UhfDefinition,
) -> Result<(), UhfError> {
    let nsite = def.nsite;
    let two_n = 2 * nsite;
    let mut idx = 0_usize;
    for line in body_lines {
        let values = orbital_line(name, line, 4)?;
        let (i, j, original) = (values[0], values[1], values[2]);
        let sign = if values.len() == 3 { 1 } else { values[3] };
        if !site_ok(i, nsite) || !site_ok(j, nsite) {
            eprintln!("Error: Site index is incorrect. ");
            return Err(definition_error("Site index is incorrect."));
        }
        for spin in 0..2_i32 {
            let all_i = i as usize + spin as usize * nsite;
            let all_j = j as usize + spin as usize * nsite;
            idx += 1;
            let fij = def.n_orbital_ap as i64 + 2 * i64::from(original) + i64::from(spin);
            if fij < 0 || fij as usize >= def.n_orbital_idx {
                return Err(definition_error(format!(
                    "{name}: parallel orbital index {original} gives {fij} outside 0..{}.",
                    def.n_orbital_idx
                )));
            }
            let fij = fij as i32;
            def.orbital_idx[all_i * two_n + all_j] = fij;
            def.orbital_sgn[all_i * two_n + all_j] = sign;
            def.orbital_idx[all_j * two_n + all_i] = fij;
            def.orbital_sgn[all_j * two_n + all_i] = -sign;
        }
        if idx == nsite * (nsite - 1) {
            break;
        }
    }
    Ok(())
}

fn read_orbital_general(
    name: &str,
    body_lines: &[&str],
    def: &mut UhfDefinition,
) -> Result<(), UhfError> {
    let nsite = def.nsite;
    let two_n = 2 * nsite;
    let mut idx = 0_usize;
    for line in body_lines {
        let values = orbital_line(name, line, 6)?;
        if values.len() < 5 {
            return Err(definition_error(format!(
                "{name}: orbital record too short."
            )));
        }
        let sign = if values.len() == 5 { 1 } else { values[5] };
        let row = spin_site(values[0], values[1], nsite, name)?;
        let col = spin_site(values[2], values[3], nsite, name)?;
        check_orbital_index(name, values[4], def)?;
        def.orbital_idx[row * two_n + col] = values[4];
        def.orbital_sgn[row * two_n + col] = sign;
        idx += 1;
        if idx == nsite * (2 * nsite - 1) {
            break;
        }
    }
    if idx != nsite * (2 * nsite - 1) {
        broken(name);
    }
    Ok(())
}

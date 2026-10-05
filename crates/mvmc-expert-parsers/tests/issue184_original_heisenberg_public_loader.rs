//! Original Julia M0193–198/M0200 through the public parser, not a runner.
use mvmc_expert_parsers::parse_expert_mode_files;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

// Pure Rust digest already tested against standard independent SHA256 vectors.
#[path = "../../mvmc-core/tests/support/fixture_sha256.rs"]
mod fixture_sha256;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/original_heisenberg_parser_184")
}

fn verify_closure(root: &Path) -> Result<(), String> {
    let bytes = fs::read(root.join("inputs.sha256")).map_err(|e| e.to_string())?;
    if fixture_sha256::sha256(&bytes)
        != "5163a694ac0e44e838dd775b76f5faea5d11b3c05135d37275bda9991f34fec8"
    {
        return Err("original manifest identity changed".into());
    }
    let text = std::str::from_utf8(&bytes).map_err(|e| e.to_string())?;
    let mut hashes = BTreeMap::new();
    for line in text.lines() {
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields.len() != 2
            || fields[1].contains(['/', '\\'])
            || hashes.insert(fields[1], fields[0]).is_some()
        {
            return Err("invalid or duplicate manifest entry".into());
        }
        let input = fs::read(root.join(fields[1])).map_err(|e| format!("{}: {e}", fields[1]))?;
        if fixture_sha256::sha256(&input) != fields[0] {
            return Err(format!("{} identity changed", fields[1]));
        }
    }
    let namelist = fs::read_to_string(root.join("namelist.def")).map_err(|e| e.to_string())?;
    let mut required = BTreeSet::from(["namelist.def"]);
    for line in namelist.lines() {
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields.len() != 2 || !required.insert(fields[1]) {
            return Err("invalid or duplicate namelist reference".into());
        }
    }
    if hashes.keys().copied().collect::<BTreeSet<_>>() != required || required.len() != 13 {
        return Err("manifest does not cover complete original namelist closure".into());
    }
    if root.join("initial.def").exists() {
        return Err("original source has no implicit initial.def".into());
    }
    Ok(())
}

struct Bundle(PathBuf);
impl Bundle {
    fn copy() -> Self {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "issue184-heisenberg-{}-{stamp}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        for entry in fs::read_dir(root()).unwrap() {
            let entry = entry.unwrap();
            fs::copy(entry.path(), path.join(entry.file_name())).unwrap();
        }
        Self(path)
    }
}
impl Drop for Bundle {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove exclusively created input bundle");
    }
}

#[test]
fn original_heisenberg_namelist_exercises_public_loader_conditions() {
    verify_closure(&root()).unwrap();
    let data = parse_expert_mode_files(root().join("namelist.def")).unwrap();
    assert!(data.input_errors.is_empty(), "{:?}", data.input_errors);
    assert_eq!(data.modpara.nsite, 16);
    assert_eq!(data.modpara.nlocspin, 16);
    assert_eq!(data.modpara.ncond, 0);
    assert_eq!(data.modpara.nelec, 8); // readdef.c:593, not a constructor override.
    assert_eq!(data.modpara.vmc_calc_mode, 0);
    assert_eq!(data.modpara.nsr_opt_itr_step, 300);
    assert_eq!(data.modpara.nsr_opt_itr_smp, 30);
    assert_eq!(data.modpara.rnd_seed, 123456789);
    assert!(!data.gutzwiller_terms.is_empty());
    assert!(!data.jastrow_terms.is_empty());
    assert!(!data
        .gutzwiller_terms
        .iter()
        .map(|t| t.site)
        .collect::<BTreeSet<_>>()
        .is_empty());
    assert!(!data
        .jastrow_terms
        .iter()
        .map(|t| (t.site1, t.site2))
        .collect::<BTreeSet<_>>()
        .is_empty());
    assert!(!data.orbital_terms.is_empty());
    let indices = data
        .orbital_terms
        .iter()
        .map(|t| t.idx)
        .collect::<BTreeSet<_>>();
    assert!(indices.len() >= 64 || data.orbital_terms.len() >= 64);
    assert_eq!(indices.len(), 64);
    assert_eq!(data.orbital_terms.len(), 256);
    // M0199's no-idx fallback is not present in Rust's typed OrbitalTerm API.
}

#[test]
fn original_fixture_preflight_rejects_each_missing_reference_and_tampering() {
    let bundle = Bundle::copy();
    verify_closure(&bundle.0).unwrap();
    for line in fs::read_to_string(root().join("namelist.def"))
        .unwrap()
        .lines()
    {
        let name = line.split_whitespace().nth(1).unwrap();
        let path = bundle.0.join(name);
        let bytes = fs::read(&path).unwrap();
        fs::remove_file(&path).unwrap();
        assert!(verify_closure(&bundle.0).is_err(), "missing {name}");
        fs::write(&path, bytes).unwrap();
    }
    fs::write(bundle.0.join("initial.def"), "unexpected overlay\n").unwrap();
    assert!(verify_closure(&bundle.0).is_err());
    fs::remove_file(bundle.0.join("initial.def")).unwrap();
    fs::write(bundle.0.join("modpara.def"), concat!("--------------------\nModel_Parameters 0\n--------------------\nVMC_Cal_Parameters\n--------------------\nCDataFileHead zvo\nCParaFileHead zqp\n--------------------\n", "Nsite 2\n")).unwrap();
    assert!(verify_closure(&bundle.0).is_err());
    fs::copy(root().join("modpara.def"), bundle.0.join("modpara.def")).unwrap();
    let manifest = fs::read_to_string(bundle.0.join("inputs.sha256")).unwrap();
    let omitted = manifest
        .lines()
        .filter(|line| !line.ends_with("orbitalidx.def"))
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(bundle.0.join("inputs.sha256"), omitted).unwrap();
    assert!(verify_closure(&bundle.0).is_err());
}

#[test]
fn public_loader_missing_optional_overlay_is_not_fixture_integrity_success() {
    let bundle = Bundle::copy();
    let path = bundle.0.join("namelist.def");
    let mut namelist = fs::read_to_string(&path).unwrap();
    namelist.push_str("InGutzwiller absent-overlay.def\n");
    fs::write(&path, namelist).unwrap();
    assert!(!bundle.0.join("absent-overlay.def").exists());
    assert!(verify_closure(&bundle.0).is_err()); // Modified source is not the original fixture.
    let data = parse_expert_mode_files(&path).unwrap();
    assert!(data.input_errors.is_empty()); // Existing Julia architecture: optional overlays.
    assert_eq!(data.modpara.nelec, 8);
    assert!(!data.gutzwiller_terms.is_empty());
    // Missing required children must never be mistaken for an optional overlay.
    fs::remove_file(bundle.0.join("greenone.def")).unwrap();
    assert!(parse_expert_mode_files(&path).is_err());
}

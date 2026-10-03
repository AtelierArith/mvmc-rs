//! Standalone corrected-General fixture guards; no historical archive/oracle dependency.
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};
#[path = "fixture_sha256.rs"]
mod digest;
use digest::sha256;

pub fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/ctest_general_pr54_3d0fd263")
}

pub fn verify(root: &Path) {
    for (file, expected) in [
        (
            "archive.sha256",
            "5d221186641197737b69840db0876462052109fa8890342ac5963d51ff9fc411",
        ),
        (
            "reviewed-source.sha256",
            "33730954e5ed3b831369723fbff577d0571fbad5067c4697c665c420a4b69605",
        ),
        (
            "reference-source/Manifest-v1.13.toml",
            "09ebd06dab244510094b99fe7c6efa2fe7a3d22221d1336b951123a5a6e8befc",
        ),
        (
            "producers/ctest_prefix_oracle.jl",
            "19ae5d405fd46d15e85892527dc7d68c81cfe30797a2fed297b12d465870c610",
        ),
        (
            "producers/ctest_direct_sr_metrics.jl",
            "b246af8cd515b6689338b1641445e5df4b5ef59056ef15ff9d4c4a0eeb38c68e",
        ),
    ] {
        assert_eq!(
            sha256(&fs::read(root.join(file)).unwrap()),
            expected,
            "{file} identity"
        );
    }
    archive(root);
    let provenance = fs::read_to_string(root.join("provenance.txt")).unwrap();
    for identity in [
        "reference_commit=3d0fd2638fd34de2a8f9609fcfaac2504caf02d2",
        "Julia=1.13.1",
        "BLAS=",
        "loaded_package=MVMCOptimizers",
        "loaded_package=MVMCExpertModeParsers",
        "loaded_package=SFMT",
    ] {
        assert!(provenance.contains(identity), "missing {identity}");
    }
    let source = fs::read_to_string(root.join("reviewed-source.sha256")).unwrap();
    let mut names = BTreeSet::new();
    for row in source
        .lines()
        .filter(|v| !v.starts_with('#') && !v.is_empty())
    {
        let t: Vec<_> = row.split_whitespace().collect();
        assert_eq!(t.len(), 2);
        safe(t[1]);
        assert!(names.insert(t[1]));
        assert_eq!(
            sha256(&fs::read(root.join("reference-source").join(t[1])).unwrap()),
            t[0]
        );
    }
    assert_eq!(names.len(), 63);
    let input = root.join("inputs/general_rbm_cmp");
    let data = mvmc_expert_parsers::parse_expert_mode_files_with_c_opt_trans(
        input.join("namelist.def"),
        false,
    )
    .unwrap();
    assert_eq!(
        (
            data.modpara.rnd_seed,
            data.modpara.nsrcg,
            data.modpara.nstore_o
        ),
        (12395, 0, 1)
    );
    assert_eq!(data.count_variational_parameters(), 102);
    for steps in [1, 2, 3, 20] {
        let case = root.join(format!("general_rbm_cmp/step-{steps}"));
        inputs(&case.join("inputs.sha256"), &input);
        settings(root, &case, steps, &data);
        assert_eq!(
            fs::read_to_string(case.join("status.txt")).unwrap().trim(),
            "0"
        );
        assert_eq!(
            fs::read(case.join("c-window-input.txt")).unwrap(),
            fs::read(case.join("c-window-declared-input.txt")).unwrap()
        );
    }
}

fn safe(name: &str) {
    assert!(Path::new(name)
        .components()
        .all(|c| matches!(c, std::path::Component::Normal(_))));
    assert!(!name.contains('\\'));
}
fn archive(root: &Path) {
    let manifest = fs::read_to_string(root.join("archive.sha256")).unwrap();
    let mut names = BTreeSet::new();
    for row in manifest.lines() {
        let t: Vec<_> = row.split_whitespace().collect();
        assert_eq!(t.len(), 2);
        safe(t[1]);
        assert!(names.insert(t[1]));
        assert_eq!(sha256(&fs::read(root.join(t[1])).unwrap()), t[0]);
    }
    assert_eq!(names.len(), 229);
}
fn inputs(manifest: &Path, input: &Path) {
    let text = fs::read_to_string(manifest).unwrap();
    let mut names = BTreeSet::new();
    for row in text.lines() {
        let t: Vec<_> = row.split_whitespace().collect();
        assert_eq!(t.len(), 2);
        safe(t[1]);
        assert_eq!(Path::new(t[1]).components().count(), 1);
        assert!(names.insert(t[1]));
        assert_eq!(sha256(&fs::read(input.join(t[1])).unwrap()), t[0]);
    }
    for file in ["namelist.def", "modpara.def", "initial.def"] {
        assert!(names.contains(file));
    }
    for (_, file) in mvmc_expert_parsers::utils::file::parse_namelist_content(
        &fs::read_to_string(input.join("namelist.def")).unwrap(),
    ) {
        assert!(names.contains(file.as_str()), "missing referenced {file}");
    }
}
pub fn settings(root: &Path, case: &Path, steps: i64, data: &mvmc_expert_parsers::ExpertModeData) {
    assert_eq!(
        fs::read(case.join("source-provenance.txt")).unwrap(),
        fs::read(root.join("provenance.txt")).unwrap()
    );
    let text = fs::read_to_string(case.join("model-settings.txt")).unwrap();
    let mut values = std::collections::BTreeMap::new();
    for (k, v) in text.split_whitespace().filter_map(|w| w.split_once('=')) {
        assert!(values.insert(k, v).is_none());
    }
    for (k, v) in [
        ("model", "general_rbm_cmp".to_string()),
        ("mode", "cmp".into()),
        ("RndSeed", data.modpara.rnd_seed.to_string()),
        ("NSRCG", data.modpara.nsrcg.to_string()),
        ("NStore", data.modpara.nstore_o.to_string()),
        (
            "canonical_NSROptItrStep",
            data.modpara.nsr_opt_itr_step.to_string(),
        ),
        (
            "canonical_NSROptItrSmp",
            data.modpara.nsr_opt_itr_smp.to_string(),
        ),
        ("steps", steps.to_string()),
        ("window", steps.to_string()),
        ("effective_NSROptItrStep", steps.to_string()),
        ("effective_NSROptItrSmp", steps.to_string()),
        ("override", "both_no_clamp".into()),
        ("Julia_default_threads", "1".into()),
        ("Julia_interactive_threads", "0".into()),
        ("BLAS_threads", "1".into()),
        ("MPI", "serial".into()),
        ("workers", "1".into()),
        ("initial_overlay", "true".into()),
    ] {
        assert_eq!(values.get(k).copied(), Some(v.as_str()), "setting {k}");
    }
}
pub fn rng(case: &Path, rng: &sfmt19937::Sfmt19937Rng) {
    let text = fs::read_to_string(case.join("rng-state.txt")).unwrap();
    let rows: Vec<_> = text.lines().collect();
    assert_eq!(rows.len(), 3);
    let words: Vec<u32> = rows[0]
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    assert_eq!(words.len(), 624);
    let index: usize = rows[1].parse().unwrap();
    assert!(index <= 624);
    let count: u128 = rows[2].parse().unwrap();
    let actual = rng.state_snapshot();
    assert_eq!(actual.0.as_slice(), words);
    assert_eq!(actual.1, index);
    assert_eq!(rng.words_consumed(), count);
    let expected: Vec<u32> = fs::read_to_string(case.join("rng.txt"))
        .unwrap()
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    assert_eq!(expected.len(), 624);
    let before = (actual, rng.words_consumed());
    let mut next = [0; 624];
    rng.dump_rand32(&mut next);
    assert_eq!(next.as_slice(), expected);
    assert_eq!(before, (rng.state_snapshot(), rng.words_consumed()));
}

#[test]
fn corrected_general_offline_closure_is_complete() {
    verify(&root());
}

#[test]
fn corrected_general_digest_known_vectors() {
    assert_eq!(
        sha256(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        sha256(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
}

#[test]
fn corrected_general_offline_mutations_fail_closed() {
    let source = root();
    verify(&source);
    let id = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let temporary = std::env::temp_dir().join(format!(
        "ctest-general-negative-{}-{id}",
        std::process::id()
    ));
    fs::create_dir(&temporary).unwrap();
    struct Owned(PathBuf);
    impl Drop for Owned {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }
    let owned = Owned(temporary);
    let root = &owned.0;
    let manifest = fs::read_to_string(source.join("archive.sha256")).unwrap();
    for row in manifest.lines() {
        let file = row.split_whitespace().nth(1).unwrap();
        let target = root.join(file);
        fs::create_dir_all(target.parent().unwrap()).unwrap();
        fs::copy(source.join(file), target).unwrap();
    }
    fs::write(root.join("archive.sha256"), &manifest).unwrap();
    verify(root);
    for file in [
        "archive.sha256",
        "reviewed-source.sha256",
        "reference-source/Manifest-v1.13.toml",
        "inputs/general_rbm_cmp/initial.def",
        "general_rbm_cmp/step-2/rng-state.txt",
    ] {
        let old = fs::read(root.join(file)).unwrap();
        fs::write(root.join(file), b"mutated\n").unwrap();
        assert!(std::panic::catch_unwind(|| verify(root)).is_err());
        fs::write(root.join(file), old).unwrap();
    }
    for invalid in [
        manifest.replace("README.md", "../README.md"),
        manifest.replace("README.md", "/README.md"),
        manifest.clone() + manifest.lines().next().unwrap() + "\n",
        manifest.lines().skip(1).map(|v| format!("{v}\n")).collect(),
    ] {
        fs::write(root.join("archive.sha256"), invalid).unwrap();
        assert!(std::panic::catch_unwind(|| archive(root)).is_err());
    }
    fs::write(root.join("archive.sha256"), manifest).unwrap();
    let case = root.join("general_rbm_cmp/step-2");
    let input = root.join("inputs/general_rbm_cmp");
    let data = mvmc_expert_parsers::parse_expert_mode_files_with_c_opt_trans(
        input.join("namelist.def"),
        false,
    )
    .unwrap();
    let original = fs::read_to_string(case.join("model-settings.txt")).unwrap();
    for (a, b) in [
        ("RndSeed=12395", "RndSeed=12396"),
        ("NSRCG=0", "NSRCG=1"),
        ("NStore=1", "NStore=0"),
        ("effective_NSROptItrSmp=2", "effective_NSROptItrSmp=20"),
        ("BLAS_threads=1", "BLAS_threads=2"),
    ] {
        assert!(original.contains(a));
        fs::write(case.join("model-settings.txt"), original.replace(a, b)).unwrap();
        assert!(std::panic::catch_unwind(|| settings(root, &case, 2, &data)).is_err());
    }
    fs::write(case.join("model-settings.txt"), original).unwrap();
    let hashes = fs::read_to_string(case.join("inputs.sha256")).unwrap();
    for missing in ["initial.def", "modpara.def"] {
        fs::write(
            case.join("inputs.sha256"),
            hashes
                .lines()
                .filter(|v| !v.ends_with(missing))
                .map(|v| format!("{v}\n"))
                .collect::<String>(),
        )
        .unwrap();
        assert!(std::panic::catch_unwind(|| inputs(&case.join("inputs.sha256"), &input)).is_err());
    }
    fs::write(case.join("inputs.sha256"), hashes).unwrap();
    let text = fs::read_to_string(case.join("rng-state.txt")).unwrap();
    let rows: Vec<_> = text.lines().collect();
    let count: usize = rows[2].parse().unwrap();
    let mut actual = sfmt19937::Sfmt19937Rng::new(12395);
    for _ in 0..count {
        actual.gen_rand32();
    }
    rng(&case, &actual);
    for changed in [
        format!("{}\n{}\n{}\n", rows[0], rows[1], count + 1),
        format!("{}\n624\n{}\n", rows[0], rows[2]),
        format!("0 {}\n{}\n{}\n", rows[0], rows[1], rows[2]),
    ] {
        fs::write(case.join("rng-state.txt"), changed).unwrap();
        assert!(std::panic::catch_unwind(|| rng(&case, &actual)).is_err());
    }
    fs::write(case.join("rng-state.txt"), text).unwrap();
}

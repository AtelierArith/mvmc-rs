//! A004 manual file lifecycle, C component rules, not runnable-model acceptance.

use mvmc_expert_parsers::{
    refresh_orbital_optimization_flags as refresh, ExpertModeData, ModParaParameters,
    OrbitalDefinitionKey as Key, OrbitalOptimizationSource as Source,
    OrbitalRawDeclaration as Declaration, OrbitalTerm,
};
use num_complex::Complex64;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT: AtomicUsize = AtomicUsize::new(0);

struct Files(PathBuf);

impl Files {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "mvmc-a004-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn write(&self, name: &str, text: &str) -> PathBuf {
        let path = self.0.join(name);
        fs::write(&path, text).unwrap();
        path
    }

    fn ap(&self, name: &str, header: i32) -> PathBuf {
        self.write(
            name,
            &format!(
                "===\nNOrbital 3\nComplex {header}\n===\n===\n\
             0 0 0 1\n0 1 1 -1\n1 0 1 1\n1 1 0 1\n\
             99 2\n7 -3\n-9 0\n"
            ),
        )
    }

    fn parallel(&self, header: i32) -> PathBuf {
        self.write(
            "p.def",
            &format!("===\nNOrbital 2\nComplex {header}\n===\n===\n0 1 0 -1\n9 4\n8 -5\n"),
        )
    }
}

impl Drop for Files {
    fn drop(&mut self) {
        // Only this exclusive test-created directory; cleanup must not double-panic.
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn data() -> ExpertModeData {
    ExpertModeData {
        modpara: ModParaParameters {
            nsite: 2,
            n_orbital_idx: 7,
            ..Default::default()
        },
        n_gutzwiller_idx: 2,
        n_jastrow_idx: 3,
        rbm_section_widths: [4, 0, 0, 0, 0, 0, 0, 0, 0],
        optimization_flags: vec![73; 34], // prefix18 + orbital14 + untouched tail2
        slater_params: vec![Complex64::new(-0.25, 0.5); 7],
        rbm_params: vec![Complex64::new(2.0, -3.0); 4],
        orbital_terms: vec![OrbitalTerm {
            site1: 0,
            site2: 1,
            idx: 1,
            is_complex: false,
            sign: -1,
        }],
        input_errors: vec!["unrelated existing diagnostic".to_owned()],
        ..Default::default()
    }
}

fn declarations(ap: i32, parallel: i32) -> [Declaration; 2] {
    [
        Declaration {
            key: Key::AntiParallel,
            parameter_count: 3,
            complex_header: ap,
        },
        Declaration {
            key: Key::Parallel,
            parameter_count: 2,
            complex_header: parallel,
        },
    ]
}

fn source(key: Key, path: &Path) -> Source<'_> {
    Source { key, path }
}

fn assert_only_flags(before: &ExpertModeData, after: &ExpertModeData) {
    let mut expected = before.clone();
    expected
        .optimization_flags
        .clone_from(&after.optimization_flags);
    assert_eq!(format!("{expected:?}"), format!("{after:?}"));
    assert_eq!(
        before.optimization_flags.len(),
        after.optimization_flags.len()
    );
}

fn reject(
    data: &mut ExpertModeData,
    declared: &[Declaration],
    sources: &[Source<'_>],
    kind: io::ErrorKind,
) {
    let before = format!("{data:?}");
    assert_eq!(refresh(data, declared, sources).unwrap_err().kind(), kind);
    assert_eq!(format!("{data:?}"), before);
}

#[test]
fn unbound_signed_modes_have_independent_literal_components_and_preserve_buffers() {
    let files = Files::new();
    for (ap_header, p_header, expected) in [
        (2, -1, [2, 2, -3, -3, 0, 0, 4, 1, 4, 1, -5, 1, -5, 1]),
        (1, -1, [2, 0, -3, 0, 0, 0, 4, 0, 4, 0, -5, 0, -5, 0]),
        (-2, 1, [2, 0, -3, 0, 0, 0, 4, -1, 4, -1, -5, -1, -5, -1]),
    ] {
        let ap = files.ap("a.def", ap_header);
        let p = files.parallel(p_header);
        let mut actual = data();
        let before = actual.clone();
        refresh(
            &mut actual,
            &declarations(ap_header, p_header),
            &[source(Key::AntiParallel, &ap), source(Key::Parallel, &p)],
        )
        .unwrap();
        assert_eq!(&actual.optimization_flags[18..32], &expected);
        assert_eq!(&actual.optimization_flags[..18], &[73; 18]);
        assert_eq!(&actual.optimization_flags[32..], &[73; 2]);
        assert_only_flags(&before, &actual);
    }
}

#[test]
fn partial_annotations_and_selected_parallel_use_explicit_unselected_ap_offset_and_header() {
    let files = Files::new();
    let p = files.parallel(1);
    for (ap_mode, p_mode, general_mode, stored_ap) in [(0, 0, 0, 0), (1, 0, 1, 3), (0, 1, 0, 3)] {
        let mut actual = data();
        actual.i_flg_orbital_anti_parallel = ap_mode;
        actual.i_flg_orbital_parallel = p_mode;
        actual.i_flg_orbital_general = general_mode;
        actual.n_orbital_anti_parallel = stored_ap;
        let before = actual.clone();
        refresh(
            &mut actual,
            &declarations(-2, 1),
            &[source(Key::Parallel, &p)],
        )
        .unwrap();
        assert_eq!(&actual.optimization_flags[18..24], &[73; 6]);
        assert_eq!(
            &actual.optimization_flags[24..32],
            &[4, -1, 4, -1, -5, -1, -5, -1]
        );
        assert_only_flags(&before, &actual);
    }
}

#[test]
fn complete_and_partial_loaded_bindings_cannot_be_overridden_or_silently_repaired() {
    let files = Files::new();
    let p = files.parallel(-1);
    let mut actual = data();
    actual
        .native_complex_headers
        .insert("OrbitalParallel".into(), -1);
    reject(
        &mut actual,
        &declarations(2, 1),
        &[source(Key::Parallel, &p)],
        io::ErrorKind::InvalidData,
    );
    actual
        .native_complex_headers
        .insert("OrbitalAntiParallel".into(), 2);
    actual
        .native_complex_declarations
        .insert("Orbitals".into(), true);
    actual.orbital_terms[0].is_complex = true;
    actual.i_flg_orbital_anti_parallel = 1;
    actual.i_flg_orbital_parallel = 1;
    actual.i_flg_orbital_general = 1;
    actual.n_orbital_anti_parallel = 3;
    let before = actual.clone();
    refresh(
        &mut actual,
        &declarations(2, -1),
        &[source(Key::Parallel, &p)],
    )
    .unwrap();
    assert_eq!(
        &actual.optimization_flags[24..32],
        &[4, 1, 4, 1, -5, 1, -5, 1]
    );
    assert_only_flags(&before, &actual);
    actual.orbital_terms[0].is_complex = false;
    reject(
        &mut actual,
        &declarations(2, -1),
        &[source(Key::Parallel, &p)],
        io::ErrorKind::InvalidData,
    );
}

#[test]
fn general_shared_and_reserved_slots_and_distinct_alias_order_are_explicit() {
    let files = Files::new();
    let g = files.write(
        "g.def",
        "===\nNOrbital 3\nComplex 1\n===\n===\n\
        0 0 1 0 0 1\n0 0 0 1 1 1\n0 0 1 1 0 -1\n\
        1 0 0 1 1 1\n1 0 1 1 0 1\n0 1 1 1 1 1\n0 2\n1 -3\n2 0\n",
    );
    let mut actual = data();
    actual.modpara.n_orbital_idx = 3;
    let before = actual.clone();
    refresh(
        &mut actual,
        &[Declaration {
            key: Key::General,
            parameter_count: 3,
            complex_header: 1,
        }],
        &[source(Key::General, &g)],
    )
    .unwrap();
    assert_eq!(&actual.optimization_flags[18..24], &[2, 2, -3, -3, 0, 0]);
    assert_eq!(&actual.optimization_flags[24..], &[73; 10]);
    assert_only_flags(&before, &actual);
    let a = files.ap("alias.def", 1);
    let b = files.write(
        "native.def",
        "===\nNOrbital 3\nComplex -1\n===\n===\n\
        0 0 0\n0 1 1\n1 0 1\n1 1 0\n0 7\n1 8\n2 9\n",
    );
    let aliases = [
        Declaration {
            key: Key::Orbital,
            parameter_count: 3,
            complex_header: 1,
        },
        Declaration {
            key: Key::AntiParallel,
            parameter_count: 3,
            complex_header: -1,
        },
    ];
    refresh(
        &mut actual,
        &aliases,
        &[source(Key::Orbital, &a), source(Key::AntiParallel, &b)],
    )
    .unwrap();
    assert_eq!(&actual.optimization_flags[18..24], &[7, 0, 8, 0, 9, 0]);
    refresh(
        &mut actual,
        &aliases,
        &[source(Key::AntiParallel, &b), source(Key::Orbital, &a)],
    )
    .unwrap();
    assert_eq!(&actual.optimization_flags[18..24], &[2, 0, -3, 0, 0, 0]);
}

#[test]
fn empty_is_true_noop_and_duplicate_keys_are_invalid_input_before_io() {
    let mut actual = data();
    actual.modpara.n_orbital_idx = -1;
    let before = format!("{actual:?}");
    refresh(&mut actual, &[], &[]).unwrap();
    assert_eq!(format!("{actual:?}"), before);
    let path = Path::new("must-not-be-opened-a004.def");
    let sources = [source(Key::Parallel, path), source(Key::Parallel, path)];
    reject(
        &mut actual,
        &declarations(1, 0),
        &sources,
        io::ErrorKind::InvalidInput,
    );
    let same = declarations(1, 0)[0];
    reject(
        &mut actual,
        &[same, same],
        &[source(Key::AntiParallel, path)],
        io::ErrorKind::InvalidInput,
    );
}

#[test]
fn later_missing_or_malformed_selected_file_rolls_back_all_staged_flags() {
    let files = Files::new();
    let a = files.ap("a.def", 1);
    let missing = files.0.join("missing.def");
    let malformed = files.write(
        "bad.def",
        "===\nNOrbital 2\nComplex 0\n===\n===\n0 1 0\n0 1\n",
    );
    let mut actual = data();
    for (path, kind) in [
        (&missing, io::ErrorKind::NotFound),
        (&malformed, io::ErrorKind::InvalidData),
    ] {
        reject(
            &mut actual,
            &declarations(1, 0),
            &[source(Key::AntiParallel, &a), source(Key::Parallel, path)],
            kind,
        );
    }
}

#[test]
fn stale_modes_widths_mapping_and_capacity_fail_without_resizing() {
    let files = Files::new();
    let p = files.parallel(0);
    let sources = [source(Key::Parallel, &p)];
    let declared = declarations(1, 0);
    let mutations: [fn(&mut ExpertModeData); 6] = [
        |d: &mut ExpertModeData| d.n_orbital_anti_parallel = 2,
        |d: &mut ExpertModeData| d.modpara.n_orbital_idx = 6,
        |d: &mut ExpertModeData| d.i_flg_orbital_parallel = -1,
        |d: &mut ExpertModeData| d.orbital_terms[0].idx = 7,
        |d: &mut ExpertModeData| d.optimization_flags.truncate(31),
        |d: &mut ExpertModeData| d.rbm_section_widths[0] = usize::MAX,
    ];
    for mutate in mutations {
        let mut actual = data();
        mutate(&mut actual);
        reject(&mut actual, &declared, &sources, io::ErrorKind::InvalidData);
    }
    let mut actual = data();
    reject(
        &mut actual,
        &[declared[1]],
        &sources,
        io::ErrorKind::InvalidData,
    );
    let mut wrong_header = declared;
    wrong_header[1].complex_header = 1;
    reject(
        &mut actual,
        &wrong_header,
        &sources,
        io::ErrorKind::InvalidData,
    );
    let mut wrong_count = declared;
    wrong_count[0].parameter_count = 1;
    wrong_count[1].parameter_count = 3; // still total7; parsed P width2 disagrees
    reject(
        &mut actual,
        &wrong_count,
        &sources,
        io::ErrorKind::InvalidData,
    );
}

#[test]
fn explicit_bindings_from_actual_strict_sections_support_ap_only_refresh() {
    use mvmc_expert_parsers::parsers::orbital::{parse_orbital_def, OrbitalKind};
    let files = Files::new();
    let a = files.ap("a.def", 1);
    let p = files.parallel(-1);
    let ap_section = parse_orbital_def(&a, 2, OrbitalKind::AntiParallel).unwrap();
    let p_section = parse_orbital_def(&p, 2, OrbitalKind::Parallel).unwrap();
    let declared = [
        Declaration {
            key: Key::AntiParallel,
            parameter_count: ap_section.n_orbital_idx as usize,
            complex_header: ap_section.complex_type,
        },
        Declaration {
            key: Key::Parallel,
            parameter_count: p_section.n_orbital_idx as usize,
            complex_header: p_section.complex_type,
        },
    ];
    let mut actual = data();
    let before = actual.clone();
    refresh(&mut actual, &declared, &[source(Key::AntiParallel, &a)]).unwrap();
    assert_eq!(&actual.optimization_flags[18..24], &[2, 0, -3, 0, 0, 0]);
    assert_eq!(&actual.optimization_flags[24..], &[73; 10]);
    assert_only_flags(&before, &actual);
}

#[test]
fn contradictory_families_missing_binding_and_unsupported_files_remain_rejected() {
    let files = Files::new();
    let a = files.ap("a.def", 1);
    let mut actual = data();
    actual.modpara.n_orbital_idx = 3;
    let ap = Declaration {
        key: Key::AntiParallel,
        parameter_count: 3,
        complex_header: 1,
    };
    let general = Declaration {
        key: Key::General,
        parameter_count: 3,
        complex_header: 1,
    };
    reject(
        &mut actual,
        &[ap, general],
        &[source(Key::AntiParallel, &a)],
        io::ErrorKind::InvalidData,
    );
    reject(
        &mut actual,
        &[],
        &[source(Key::AntiParallel, &a)],
        io::ErrorKind::InvalidData,
    );
    actual.i_flg_orbital_parallel = 1;
    reject(
        &mut actual,
        &[ap],
        &[source(Key::AntiParallel, &a)],
        io::ErrorKind::InvalidData,
    );
    actual.i_flg_orbital_parallel = 0;
    let headerless = files.write(
        "headerless.def",
        "0 0 0\n0 1 1\n1 0 1\n1 1 0\n0 1\n1 1\n2 1\n",
    );
    reject(
        &mut actual,
        &[ap],
        &[source(Key::AntiParallel, &headerless)],
        io::ErrorKind::InvalidData,
    );
    actual.native_complex_headers.insert("Orbital".into(), 1);
    reject(
        &mut actual,
        &[ap],
        &[source(Key::AntiParallel, &a)],
        io::ErrorKind::InvalidData,
    );
}

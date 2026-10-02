//! Historical Julia kernel models, constructed after validating complete inputs.
//!
//! Checked-in C replacements supply complete AP/P/General/Jastrow rows and flags.
//! Sparse spatial tables are then restored programmatically for the historical
//! coefficient/kernel regression checks. Those sparse models are not evidence
//! that C accepts the original incomplete definition files. Production parsing
//! always uses the strict reader; this helper is confined to test code.
use mvmc_expert_parsers::{ExpertModeData, OrbitalTerm};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT_MODEL: AtomicUsize = AtomicUsize::new(0);

pub fn historical_kernel_model(path: impl AsRef<Path>) -> Result<ExpertModeData, String> {
    let path = path.as_ref();
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .canonicalize()
        .map_err(|error| error.to_string())?;
    let replacements = [
        ("dh2/orbital.def", "ap_three.def"),
        ("dh4/orbital.def", "ap_three.def"),
        ("rbm/orbital.def", "ap_three.def"),
        ("dh4/parallel_orbital.def", "p_three.def"),
        ("interall/orbital.def", "ap_four.def"),
        ("orbital_general/ap.def", "ap_general_three.def"),
        ("orbital_general/parallel.def", "p_general_three.def"),
        ("orbital_general/general.def", "general_three.def"),
        ("orbital_general/sparse.def", "general_sparse_three.def"),
        (
            "orbital_general/heisenberg/general.def",
            "general_heisenberg_six.def",
        ),
        ("dh4/general_orbital.def", "general_three_real.def"),
    ];
    let content = std::fs::read_to_string(path).map_err(|error| error.to_string())?;
    let mut metadata = Vec::new();
    let mut original_orbitals = Vec::new();
    let mut rewritten = String::new();
    let mut changed = false;
    let mut changed_orbitals = false;
    for line in content.lines() {
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields.len() < 2 || fields[0].starts_with('#') {
            continue;
        }
        metadata.push((fields[0].to_owned(), fields[1].to_owned()));
        let original = path.parent().unwrap().join(fields[1]);
        let absolute = original.canonicalize().unwrap_or(original);
        let replacement = replacements
            .iter()
            .find(|(name, _)| absolute == root.join(name));
        let jastrow_replacement = fields[0] == "Jastrow"
            && ["dh2/jast.def", "dh4/jast.def", "rbm/jast.def"]
                .iter()
                .any(|name| absolute == root.join(name));
        let selected = if jastrow_replacement {
            changed = true;
            root.join("jastrow/c_three.def")
        } else if let Some((_, target)) = replacement {
            changed = true;
            changed_orbitals = true;
            root.join("c_orbital_inputs").join(target)
        } else {
            absolute.clone()
        };
        if matches!(
            fields[0],
            "Orbital" | "OrbitalAntiParallel" | "OrbitalParallel" | "OrbitalGeneral"
        ) {
            original_orbitals.push((fields[0].to_owned(), absolute));
        }
        rewritten.push_str(&format!("{} {}\n", fields[0], selected.display()));
    }
    if !changed {
        return mvmc_expert_parsers::parse_expert_mode_files(path)
            .map_err(|error| error.to_string());
    }
    let dir = std::env::temp_dir().join(format!(
        "mvmc-historical-model-{}-{}",
        std::process::id(),
        NEXT_MODEL.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).map_err(|error| error.to_string())?;
    let namelist = dir.join("namelist.def");
    std::fs::write(&namelist, rewritten).map_err(|error| error.to_string())?;
    let result =
        mvmc_expert_parsers::parse_expert_mode_files(&namelist).map_err(|error| error.to_string());
    std::fs::remove_dir_all(&dir).map_err(|error| error.to_string())?;
    let mut data = result?;
    if data.input_errors.iter().any(|error| {
        error.starts_with("error parsing Orbital") || error.starts_with("error parsing Jastrow")
    }) {
        return Err(data.input_errors.join("; "));
    }
    data.namelist = metadata;
    if !changed_orbitals {
        return Ok(data);
    }
    original_orbitals.sort_by_key(|(kind, _)| kind == "OrbitalParallel");
    let mut terms = Vec::new();
    for (kind, original) in original_orbitals {
        let text = std::fs::read_to_string(original).map_err(|error| error.to_string())?;
        let lines: Vec<_> = text.lines().collect();
        let is_complex = lines[2].split_whitespace().nth(1) == Some("1");
        for line in &lines[5..] {
            let fields: Vec<_> = line.split_whitespace().collect();
            if fields.len() < 3 {
                break;
            }
            let value = |index: usize| fields[index].parse::<i64>().unwrap();
            let term = OrbitalTerm {
                site1: value(0),
                site2: value(1),
                idx: value(2),
                sign: fields.get(3).map_or(1, |field| field.parse().unwrap()),
                is_complex,
            };
            if kind == "OrbitalParallel" {
                for spin in 0..2 {
                    terms.push(OrbitalTerm {
                        idx: data.n_orbital_anti_parallel + 2 * term.idx + spin,
                        ..term
                    });
                }
            } else {
                terms.push(term);
            }
        }
    }
    data.orbital_terms = terms;
    data.orbital_idx_matrix = None;
    data.orbital_sgn_matrix = None;
    data.ensure_orbital_idx_matrix();
    Ok(data)
}

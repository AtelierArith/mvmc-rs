// Explicit input adapter ONLY for reconstructed archived test-stage models.
// This is not a production compatibility wrapper or independent oracle.
use mvmc_expert_parsers::ExpertModeData;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT_STAGE: AtomicUsize = AtomicUsize::new(0);

pub fn read_input_parameters(data: &mut ExpertModeData, namelist: &Path) -> Result<(), String> {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .canonicalize()
        .map_err(|e| e.to_string())?;
    let inputs = fixtures.join("c_overlay_stage_184");
    let base = namelist.parent().ok_or("stage namelist has no parent")?;
    let content = std::fs::read_to_string(namelist).map_err(|e| e.to_string())?;
    let mut rewritten = String::new();
    for line in content.lines() {
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields.len() < 2 || fields[0].starts_with('#') {
            rewritten.push_str(line);
            rewritten.push('\n');
            continue;
        }
        let original = base.join(fields[1]);
        let original = original.canonicalize().unwrap_or(original);
        let name = original.file_name().and_then(|n| n.to_str()).unwrap_or("");
        let section = [
            "InChargeRBM_PhysLayer",
            "InSpinRBM_PhysLayer",
            "InGeneralRBM_PhysLayer",
            "InChargeRBM_HiddenLayer",
            "InSpinRBM_HiddenLayer",
            "InGeneralRBM_HiddenLayer",
            "InChargeRBM_PhysHidden",
            "InSpinRBM_PhysHidden",
            "InGeneralRBM_PhysHidden",
        ]
        .iter()
        .position(|&key| key == fields[0]);
        let bound_original = original.clone();
        let selected = if let Some(section) = section {
            if original == fixtures.join("rbm/production/overlay.def") {
                assert_eq!(
                    data.rbm_section_sizes()[section],
                    3,
                    "reviewed sparse-stage width"
                );
                inputs.join("sparse_width3.def")
            } else if original.parent()
                == Some(
                    fixtures
                        .join("c_orbital_inputs/historical_binary_rbm")
                        .as_path(),
                )
            {
                assert_eq!(
                    data.rbm_section_sizes()[section],
                    3,
                    "reviewed active archived stage width"
                );
                let expected_widths = if name == "rbm_input_rbm_general_cmp.def" {
                    [0, 0, 3, 0, 0, 3, 0, 0, 3]
                } else {
                    [3; 9]
                };
                assert_eq!(
                    data.rbm_section_sizes(),
                    expected_widths,
                    "reviewed named archived stage mask"
                );
                match name {
                    "rbm_input_opt_dh24_rbm_cmp.def" => inputs.join("opt_hidden_width3.def"),
                    "rbm_input_rbm_real.def" => inputs.join("hidden_rbm_real.def"),
                    "rbm_input_rbm_cmp.def" => inputs.join("hidden_rbm_cmp.def"),
                    "rbm_input_rbm_general_cmp.def" => inputs.join("hidden_rbm_general_cmp.def"),
                    "rbm_input_rbm_dh24_cmp.def" => inputs.join("hidden_rbm_dh24_cmp.def"),
                    "rbm_input_rbm_fsz.def" => inputs.join("hidden_rbm_fsz.def"),
                    _ => return Err(format!("unreviewed historical overlay {name}")),
                }
            } else {
                original
            }
        } else {
            original
        };
        if selected != bound_original {
            verify_bound_input(&bound_original, &fixtures)?;
            verify_bound_input(&selected, &fixtures)?;
        }
        rewritten.push_str(&format!("{} {}\n", fields[0], selected.display()));
    }
    let parent = std::env::temp_dir();
    let directory = loop {
        let path = parent.join(format!(
            "mvmc-overlay-stage-{}-{}",
            std::process::id(),
            NEXT_STAGE.fetch_add(1, Ordering::Relaxed)
        ));
        match std::fs::create_dir(&path) {
            Ok(()) => break path,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e.to_string()),
        }
    };
    let staged = directory.join("namelist.def");
    let result = std::fs::write(&staged, rewritten)
        .map_err(|e| e.to_string())
        .and_then(|()| {
            mvmc_expert_parsers::utils::read_input_parameters::read_input_parameters(data, &staged)
        });
    // Exclusive directory; delete only our own single known generated input.
    let cleanup = std::fs::remove_file(&staged).and_then(|()| std::fs::remove_dir(&directory));
    result.and_then(|()| cleanup.map_err(|e| e.to_string()))
}

include!("overlay_stage_sha256.rs");

fn verify_bound_input(path: &Path, fixtures: &Path) -> Result<(), String> {
    let relative = path.strip_prefix(fixtures).map_err(|e| e.to_string())?;
    let key = relative.to_str().ok_or("non-UTF8 bound stage path")?;
    let expected = match key {
        "rbm/production/overlay.def" => {
            "ff1fd783b2647c08bbcca1789cdb196eb4c0541e510541e1253fee74089f79d2"
        }
        "c_orbital_inputs/historical_binary_rbm/rbm_input_rbm_real.def" => {
            "7d56e1f6595ced49d80598fa34c8a34cb652a6aa555008f6fefa08e94e8f70d1"
        }
        "c_orbital_inputs/historical_binary_rbm/rbm_input_opt_dh24_rbm_cmp.def"
        | "c_orbital_inputs/historical_binary_rbm/rbm_input_rbm_cmp.def"
        | "c_orbital_inputs/historical_binary_rbm/rbm_input_rbm_general_cmp.def"
        | "c_orbital_inputs/historical_binary_rbm/rbm_input_rbm_dh24_cmp.def"
        | "c_orbital_inputs/historical_binary_rbm/rbm_input_rbm_fsz.def" => {
            "f641c2e2e3aff524223a44bc24fa1de4b64e8ae986ceed54456e4ce7f05a0dc7"
        }
        "c_overlay_stage_184/hidden_rbm_cmp.def" => {
            "467bd3a19231f1a89ab0c0ac1e89422413b028d9aa369074681d7715802881ba"
        }
        "c_overlay_stage_184/hidden_rbm_dh24_cmp.def" => {
            "467bd3a19231f1a89ab0c0ac1e89422413b028d9aa369074681d7715802881ba"
        }
        "c_overlay_stage_184/hidden_rbm_fsz.def" => {
            "467bd3a19231f1a89ab0c0ac1e89422413b028d9aa369074681d7715802881ba"
        }
        "c_overlay_stage_184/hidden_rbm_general_cmp.def" => {
            "467bd3a19231f1a89ab0c0ac1e89422413b028d9aa369074681d7715802881ba"
        }
        "c_overlay_stage_184/hidden_rbm_real.def" => {
            "8431ed3ee9ff43c029bff05a36874d0c47ed0639e6c837675623a66a1280ed23"
        }
        "c_overlay_stage_184/opt_hidden_width3.def" => {
            "467bd3a19231f1a89ab0c0ac1e89422413b028d9aa369074681d7715802881ba"
        }
        "c_overlay_stage_184/sparse_width3.def" => {
            "b7af4c2d955d826af617bcf23024dbe5875474ace88bcf5a24f1486fce4e3fc0"
        }
        _ => return Err(format!("unreviewed input binding {key}")),
    };
    let actual = sha256(&std::fs::read(path).map_err(|e| e.to_string())?);
    if actual != expected {
        return Err(format!("changed bound stage input {key}"));
    }
    Ok(())
}

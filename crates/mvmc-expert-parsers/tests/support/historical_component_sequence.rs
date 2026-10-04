//! Archived component sequences are not accepted C namelist duplicates.
use mvmc_expert_parsers::parsers::{doublon_holon, opttrans};
use mvmc_expert_parsers::{utils::opt_flag::set_dh_opt_flags, ExpertModeData};
use std::{
    fs, io,
    path::Path,
    sync::atomic::{AtomicUsize, Ordering},
};

static NEXT: AtomicUsize = AtomicUsize::new(0);

pub fn model(
    path: &Path,
    family: &str,
    load: impl Fn(&Path) -> Result<ExpertModeData, String>,
) -> ExpertModeData {
    let content = fs::read_to_string(path).unwrap();
    let entries = mvmc_expert_parsers::utils::file::parse_namelist_content(&content);
    let sequence: Vec<_> = entries.iter().filter(|(kind, _)| kind == family).collect();
    if sequence.is_empty() || (family == "OptTrans" && sequence.len() < 2) {
        return load(path).unwrap();
    }
    let selected = if family == "OptTrans" {
        sequence[0]
    } else {
        sequence[sequence.len() - 1]
    };
    let selected_content = fs::read_to_string(path.parent().unwrap().join(&selected.1)).unwrap();
    let selected_header: Vec<_> = selected_content.lines().take(5).collect();
    let empty_family = matches!(family, "DH2" | "DH4")
        && selected_header.len() == 5
        && selected_header[1].split_whitespace().nth(1) == Some("0");
    if sequence.len() < 2 && !empty_family {
        return load(path).unwrap();
    }
    let error = mvmc_expert_parsers::parse_expert_mode_files(path).unwrap_err();
    let mvmc_expert_parsers::ParseError::InvalidInput { message } = error else {
        panic!("archived duplicate/zero-count input must be InvalidInput: {error:?}");
    };
    if sequence.len() > 1 {
        assert!(message.contains(&format!("duplicate keyword {family}")));
    } else {
        assert!(message.contains(&format!("Error parsing required {family}")));
        assert!(message.contains("must be a positive C integer"));
    }
    // Load the last positive DH definition through production assembly, then
    // independently compare component reads and the final assembly below.
    // Zero-count archived sections require programmatic assembly. OptTrans has
    // a failing second section: load its first definition before sequential
    // public mutating component reads, which publish only successful parses.
    // C ReadBuffIntCmpFlg rejects a present zero-count DH file before GetInfoDH.
    // Omit that section from the accepted base and explicitly construct the
    // archived empty family below. It is not a newly accepted C input.
    let directory = loop {
        let directory = std::env::temp_dir().join(format!(
            "issue184-component-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        match fs::create_dir(&directory) {
            Ok(()) => break directory,
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => panic!("exclusive component model: {error}"),
        }
    };
    let mut kept_family = false;
    let rewritten: String = entries
        .iter()
        .filter(|entry| {
            if entry.0 != family {
                return true;
            }
            if empty_family {
                return false;
            }
            if *entry == selected && !kept_family {
                kept_family = true;
                return true;
            }
            false
        })
        .map(|(kind, file)| format!("{kind} {}\n", path.parent().unwrap().join(file).display()))
        .collect();
    let namelist = directory.join("namelist.def");
    fs::write(&namelist, rewritten).unwrap();
    let result = load(&namelist);
    fs::remove_dir_all(&directory).unwrap();
    let mut data = result.unwrap();
    if empty_family {
        assert!(selected_content
            .lines()
            .skip(5)
            .all(|line| line.trim().is_empty()));
        let complex_type: i32 = selected_header[2]
            .split_whitespace()
            .nth(1)
            .unwrap()
            .parse()
            .unwrap();
        assert!(
            matches!(complex_type, 0 | 1),
            "reviewed archived empty mode"
        );
        // Preserve the original empty-family declaration: it still participates
        // in initialization's complex-mode sum and hence the consumed RNG stream.
        // This replaces loaded metadata only for the explicitly constructed family.
        data.native_complex_headers.remove(family);
        data.native_complex_declarations.remove(family);
        match family {
            "DH2" => {
                data.doublon_holon_2site_indices.clear();
                data.doublon_holon_2site_params.clear();
                data.doublon_holon_2site_opt_flags.clear();
                data.doublon_holon_2site_complex = complex_type != 0;
            }
            "DH4" => {
                data.doublon_holon_4site_indices.clear();
                data.doublon_holon_4site_params.clear();
                data.doublon_holon_4site_opt_flags.clear();
                data.doublon_holon_4site_complex = complex_type != 0;
            }
            _ => unreachable!(),
        }
        set_dh_opt_flags(&mut data);
    }
    data.namelist = entries.clone(); // Preserve archived overlay metadata, not C acceptance.
    for (_, file) in sequence {
        let definition_path = path.parent().unwrap().join(file);
        match family {
            "DH2" => {
                let section = doublon_holon::parse_doublon_holon_2site_def(
                    &definition_path,
                    data.modpara.nsite,
                )
                .unwrap();
                if definition_has_zero_count(&definition_path) {
                    assert!(!section.is_success());
                    assert_eq!(section.line_number, 0);
                    assert_eq!(
                        section.error_message,
                        "NDoublonHolon2siteIdx must be a positive C integer"
                    );
                    continue;
                }
                let definition = section.data.expect("archived successful DH2 component");
                if file == &selected.1 {
                    assert_eq!(data.doublon_holon_2site_indices, definition.indices);
                    assert_eq!(data.doublon_holon_2site_complex, definition.is_complex);
                    assert_eq!(data.doublon_holon_2site_opt_flags, definition.opt_flags);
                    assert_eq!(
                        data.doublon_holon_2site_params.len(),
                        definition.opt_flags.len()
                    );
                }
            }
            "DH4" => {
                let section = doublon_holon::parse_doublon_holon_4site_def(
                    &definition_path,
                    data.modpara.nsite,
                )
                .unwrap();
                if definition_has_zero_count(&definition_path) {
                    assert!(!section.is_success());
                    assert_eq!(section.line_number, 0);
                    assert_eq!(
                        section.error_message,
                        "NDoublonHolon4siteIdx must be a positive C integer"
                    );
                    continue;
                }
                let definition = section.data.expect("archived successful DH4 component");
                if file == &selected.1 {
                    assert_eq!(data.doublon_holon_4site_indices, definition.indices);
                    assert_eq!(data.doublon_holon_4site_complex, definition.is_complex);
                    assert_eq!(data.doublon_holon_4site_opt_flags, definition.opt_flags);
                    assert_eq!(
                        data.doublon_holon_4site_params.len(),
                        definition.opt_flags.len()
                    );
                }
            }
            "OptTrans" => {
                let previous = data.clone();
                if let Err(error) = opttrans::parse_opttrans_def(&mut data, &definition_path) {
                    assert_eq!(data.n_qp_opt_trans, previous.n_qp_opt_trans);
                    assert_eq!(data.opt_trans, previous.opt_trans);
                    assert_eq!(data.para_qp_opt_trans, previous.para_qp_opt_trans);
                    assert_eq!(data.qp_opt_trans, previous.qp_opt_trans);
                    assert_eq!(data.qp_opt_trans_sgn, previous.qp_opt_trans_sgn);
                    assert_eq!(data.optimization_flags, previous.optimization_flags);
                    data.input_errors.push(format!(
                        "error parsing OptTrans file {}: {error}",
                        definition_path.display()
                    ));
                }
            }
            _ => panic!("unreviewed historical sequence family {family}"),
        }
    }
    // lib.rs postparse set_dh_opt_flags uses the final projection layout.
    // Reapplying that public assembler must not change the loaded final flags.
    if family == "DH2" || family == "DH4" {
        let flags = data.optimization_flags.clone();
        set_dh_opt_flags(&mut data);
        assert_eq!(data.optimization_flags, flags);
    }
    data
}

fn definition_has_zero_count(path: &Path) -> bool {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .nth(1)
        .unwrap()
        .split_whitespace()
        .nth(1)
        == Some("0")
}

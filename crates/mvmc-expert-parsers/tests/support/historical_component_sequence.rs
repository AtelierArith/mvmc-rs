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
    if sequence.len() < 2 {
        return load(path).unwrap();
    }
    let error = mvmc_expert_parsers::parse_expert_mode_files(path).unwrap_err();
    let mvmc_expert_parsers::ParseError::InvalidInput { message } = error else {
        panic!("archived repeated keyword must be InvalidInput: {error:?}");
    };
    assert!(message.contains(&format!("duplicate keyword {family}")));
    // DH fixtures contain successful sections only: load the last definition
    // through the original production assembly, then independently compare
    // every public component read and the final assembly below. OptTrans has
    // a failing second section: load its first definition before sequential
    // public mutating component reads, which publish only successful parses.
    let selected = if family == "OptTrans" {
        sequence[0]
    } else {
        sequence[sequence.len() - 1]
    };
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

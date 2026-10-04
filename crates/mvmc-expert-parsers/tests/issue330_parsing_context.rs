//! A093 opt-in Julia payload observations, not C definition admission.
use mvmc_expert_parsers::parsers::coulomb::{
    parse_coulomb_intra_content, parse_coulomb_intra_def, parse_coulomb_intra_definition,
};
use mvmc_expert_parsers::{ParsingContext, ParsingDiagnostic};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Input(std::path::PathBuf);
impl Input {
    fn new(content: &str) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "mvmc-issue330-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, content).unwrap();
        Self(path)
    }
}
impl Drop for Input {
    fn drop(&mut self) {
        std::fs::remove_file(&self.0).unwrap();
    }
}

#[test]
fn context_starts_with_exact_identity_and_no_observations() {
    let context = ParsingContext::new("inputs/../literal.def");
    assert_eq!(
        context.filename,
        std::path::Path::new("inputs/../literal.def")
    );
    assert_eq!(context.line_number, 0);
    assert!(context.errors.is_empty() && context.warnings.is_empty());
}

#[test]
fn empty_and_terminal_newline_follow_original_split_line_accounting() {
    let mut context = ParsingContext::new("memory");
    assert!(parse_coulomb_intra_content("", &mut context).is_empty());
    assert_eq!(context.line_number, 1);
    assert!(parse_coulomb_intra_content("\n", &mut context).is_empty());
    assert_eq!(context.line_number, 2);
}

#[test]
fn headers_comments_short_rows_and_negative_sites_keep_physical_warning_order() {
    let mut context = ParsingContext::new("literal.def");
    let terms = parse_coulomb_intra_content(
        "# comment\nNCoulombIntra 3\n\n0\n-2 8\n2 4\n// comment\n-1 9\n0 -0.5",
        &mut context,
    );
    assert_eq!(terms.len(), 2);
    assert_eq!((terms[0].site, terms[0].value), (2, 4.0));
    assert_eq!((terms[1].site, terms[1].value), (0, -0.5));
    assert_eq!(context.line_number, 9);
    assert_eq!(
        context.warnings,
        [
            ParsingDiagnostic {
                line_number: 5,
                message: "Negative site index: -2".into()
            },
            ParsingDiagnostic {
                line_number: 8,
                message: "Negative site index: -1".into()
            },
        ]
    );
    assert!(context.errors.is_empty());
}

#[test]
fn malformed_coefficient_keeps_original_safe_zero_without_new_fatal_rule() {
    let mut context = ParsingContext::new("literal");
    let terms = parse_coulomb_intra_content("1 invalid\nnot_site 4\n2 1.25", &mut context);
    assert_eq!(terms.len(), 2);
    assert_eq!((terms[0].site, terms[0].value), (1, 0.0));
    assert_eq!((terms[1].site, terms[1].value), (2, 1.25));
    assert!(context.errors.is_empty() && context.warnings.is_empty());
}

#[test]
fn a_new_parse_resets_observations_not_caller_filename() {
    let mut context = ParsingContext::new("same");
    parse_coulomb_intra_content("-1 2\n", &mut context);
    context.errors.push(ParsingDiagnostic {
        line_number: 1,
        message: "prior caller error".into(),
    });
    let terms = parse_coulomb_intra_content("0 11", &mut context);
    assert_eq!((terms[0].site, terms[0].value), (0, 11.0));
    assert_eq!(context.filename, std::path::Path::new("same"));
    assert_eq!(context.line_number, 1);
    assert!(context.errors.is_empty() && context.warnings.is_empty());
}

#[test]
fn file_payload_returns_path_context_and_io_failure_is_not_hidden() {
    let input = Input::new("-1 2\n0 11");
    let (terms, context) = parse_coulomb_intra_def(&input.0).unwrap();
    assert_eq!(context.filename, input.0);
    assert_eq!(context.warnings[0].line_number, 1);
    assert_eq!((terms[0].site, terms[0].value), (0, 11.0));
    assert!(parse_coulomb_intra_def(input.0.join("not-a-file")).is_err());
}

#[test]
fn opt_in_payload_warning_does_not_relax_strict_c_definition() {
    let input = Input::new("===\nNCoulombIntra 1\n===\n===\n===\n-1 4");
    let (terms, context) = parse_coulomb_intra_def(&input.0).unwrap();
    assert!(terms.is_empty());
    assert_eq!(
        context.warnings,
        [ParsingDiagnostic {
            line_number: 6,
            message: "Negative site index: -1".into(),
        }]
    );
    assert_eq!(
        parse_coulomb_intra_definition(&input.0, 4)
            .unwrap_err()
            .kind(),
        std::io::ErrorKind::InvalidData
    );
}

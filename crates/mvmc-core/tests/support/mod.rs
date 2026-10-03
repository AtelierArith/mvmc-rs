#![allow(dead_code)]
//! Locate the bundled `Julia-mVMC` checkout. The Rust workspace may live
//! either inside `ManyVariableVariationalMonteCarlo.jl/extern/Julia-mVMC-rs/`
//! (legacy layout) or as a sibling of `ManyVariableVariationalMonteCarlo.jl`
//! (current layout). Tests call into this helper instead of hard-coding
//! `../../../Julia-mVMC` paths so both layouts work without edits.

use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateStatus {
    Pass,
    ExplicitSkip,
    NotRun,
    MissingFixture,
    Unsupported,
    Failure,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateSelection {
    Run,
    ExplicitSkip,
    NotRun,
}

fn selection(value: Option<&str>) -> GateSelection {
    match value {
        None => GateSelection::NotRun,
        Some(value) if value.eq_ignore_ascii_case("skip") => GateSelection::ExplicitSkip,
        Some(_) => GateSelection::Run,
    }
}

pub fn report_gate(name: &str, status: GateStatus, detail: &str) {
    eprintln!("parity gate {name}: {status:?}: {detail}");
}

/// Ignored opt-in tests must call this before touching any reference inputs.
/// A runtime return cannot mark a test ignored in libtest or nextest.
pub fn require_gate(name: &str, variable: &str) {
    require_selection(name, variable, std::env::var(variable).ok().as_deref());
}

fn require_selection(name: &str, variable: &str, value: Option<&str>) {
    match selection(value) {
        GateSelection::Run if value.is_some_and(|value| !value.trim().is_empty()) => {}
        GateSelection::ExplicitSkip => {
            report_gate(
                name,
                GateStatus::ExplicitSkip,
                "requested test has a skip selector",
            );
            panic!("parity gate {name}: omit this ignored test from selection to skip it; {variable}=skip cannot pass");
        }
        _ => {
            report_gate(
                name,
                GateStatus::NotRun,
                "required opt-in selector is absent or empty",
            );
            panic!("parity gate {name}: selected ignored test requires {variable}=1 (or its documented model selector)");
        }
    }
}

pub fn missing_fixture(name: &str, detail: impl AsRef<str>) -> ! {
    report_gate(name, GateStatus::MissingFixture, detail.as_ref());
    panic!("parity gate {name} missing fixture: {}", detail.as_ref());
}

pub fn unsupported(name: &str, detail: impl AsRef<str>) -> ! {
    report_gate(name, GateStatus::Unsupported, detail.as_ref());
    panic!("parity gate {name} unsupported: {}", detail.as_ref());
}

pub fn julia_mvmc_root() -> Option<PathBuf> {
    if let Ok(custom) = std::env::var("JULIA_MVMC_ROOT") {
        let p = PathBuf::from(custom);
        return p.is_dir().then_some(p);
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest.join("../../../Julia-mVMC"),
        manifest.join("../../../ManyVariableVariationalMonteCarlo.jl/extern/Julia-mVMC"),
        manifest.join("../../../../ManyVariableVariationalMonteCarlo.jl/extern/Julia-mVMC"),
        manifest.join("../../extern/Julia-mVMC"),
        manifest.join("../../Julia-mVMC"),
        manifest.join("../Julia-mVMC"),
    ];
    candidates.into_iter().find(|c| c.is_dir())
}

#[cfg(test)]
mod tests {
    use super::{require_selection, selection, GateSelection};

    #[test]
    fn requested_gate_requires_nonempty_selection() {
        for value in [None, Some(""), Some(" "), Some("skip"), Some("SKIP")] {
            assert!(
                std::panic::catch_unwind(|| require_selection("probe", "OPT_IN", value)).is_err()
            );
        }
        require_selection("probe", "OPT_IN", Some("1"));
        require_selection("probe", "OPT_IN", Some("hubbard_chain_real"));
    }

    #[test]
    fn absent_gate_is_not_run() {
        assert_eq!(selection(None), GateSelection::NotRun);
    }

    #[test]
    fn skip_value_is_explicit_skip() {
        assert_eq!(selection(Some("skip")), GateSelection::ExplicitSkip);
        assert_eq!(selection(Some("SKIP")), GateSelection::ExplicitSkip);
    }

    #[test]
    fn any_other_value_selects_gate() {
        assert_eq!(selection(Some("1")), GateSelection::Run);
    }
}

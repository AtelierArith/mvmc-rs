//! Original Julia manual mode lifecycle with a separately reported C status.

use crate::ExpertModeData;

/// C JudgeOrbitalMode outcome for the pre-update declarations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeOrbitalModeStatus {
    /// General alone, AP alone or AP+P declarations accepted by C.
    Accepted,
    /// General conflicts with AP or P, C return -1.
    MultipleDefinitions,
    /// No AP/General definition, including P alone, C return -2.
    MissingAntiParallel,
}

/// The warning emitted by the original manual Julia operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrbitalModeWarning {
    /// General=1 and AP=1 or P=1; preserve declarations and report conflict.
    ConflictingDefinitions,
}

/// Manual result; C admission is reported, never imposed on a public caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrbitalModeReport {
    /// General mode after the original Julia update rule.
    pub general_mode: i64,
    /// C interpretation of the declarations supplied to this call.
    pub native_status: NativeOrbitalModeStatus,
    /// Structured original warning, not a new mandatory loader error.
    pub warning: Option<OrbitalModeWarning>,
}

/// Apply the original Julia manual mode rule and return its structured observations.
///
/// If General is not already1 and AP=P=1, set only General to1. Otherwise retain
/// all fields, including no-orbital/P-only and nonbinary programmatic annotations.
/// General=1 with AP=1 or P=1 reports the original conflict warning. C's stricter
/// admission result is a separate report field: this function never validates
/// files, rejects a model, repairs metadata, changes flags or consumes RNG.
/// Repeated calls with inferred General=1/AP=P=1 consequently report a conflict,
/// as the original Julia helper does; no new idempotence promise is invented.
pub fn judge_orbital_mode(data: &mut ExpertModeData) -> OrbitalModeReport {
    let general = data.i_flg_orbital_general;
    let ap = data.i_flg_orbital_anti_parallel;
    let parallel = data.i_flg_orbital_parallel;
    let native_status = if general == 1 {
        if ap == 0 && parallel == 0 {
            NativeOrbitalModeStatus::Accepted
        } else {
            NativeOrbitalModeStatus::MultipleDefinitions
        }
    } else if ap == 1 {
        NativeOrbitalModeStatus::Accepted
    } else {
        NativeOrbitalModeStatus::MissingAntiParallel
    };
    let warning = if general == 1 && (ap == 1 || parallel == 1) {
        Some(OrbitalModeWarning::ConflictingDefinitions)
    } else {
        None
    };
    if general != 1 && ap == 1 && parallel == 1 {
        data.i_flg_orbital_general = 1;
    }
    OrbitalModeReport {
        general_mode: data.i_flg_orbital_general,
        native_status,
        warning,
    }
}

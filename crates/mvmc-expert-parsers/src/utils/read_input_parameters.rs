//! Phase 4 — `In*.def` overlay reader.
//!
//! Port target: `MVMCExpertModeParsers.jl/src/utils/read_input_parameters.jl`.
//!
//! The four upstream `examples/inputs/*` cases that drive the Phase 4
//! bit-parity gate do **not** ship `In*.def` overlays, so this stub is
//! a no-op that simply tracks where the namelist file lives. A full
//! port is parked until an `In*.def`-bearing fixture is added.

use std::path::Path;

use crate::types::ExpertModeData;

/// `read_input_parameters!(data, namelist_path)` mirror. Currently a
/// no-op (see module docs).
pub fn read_input_parameters<P: AsRef<Path>>(_data: &mut ExpertModeData, _namelist_path: P) {}

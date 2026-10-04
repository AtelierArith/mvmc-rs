//! Optional, read-only diagnostics for independently mutable RBM shadow values.
//!
//! These checks do not validate dense storage, mapping ranges, or native input
//! admissibility. They are never invoked by parameter packing or initialization.

use std::collections::HashMap;
use std::fmt;

use mvmc_expert_parsers::ExpertModeData;
use num_complex::Complex64;

/// Independent RBM shadow section; indices are local to each section.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RbmShadowSection {
    /// ChargePhysLayer shadow mappings.
    ChargePhysLayer,
    /// SpinPhysLayer shadow mappings.
    SpinPhysLayer,
    /// GeneralPhysLayer shadow mappings.
    GeneralPhysLayer,
    /// ChargeHiddenLayer shadow mappings.
    ChargeHiddenLayer,
    /// SpinHiddenLayer shadow mappings.
    SpinHiddenLayer,
    /// GeneralHiddenLayer shadow mappings.
    GeneralHiddenLayer,
    /// ChargePhysHidden shadow mappings.
    ChargePhysHidden,
    /// SpinPhysHidden shadow mappings.
    SpinPhysHidden,
    /// GeneralPhysHidden shadow mappings.
    GeneralPhysHidden,
}

/// First unequal duplicate in section order and original row order.
#[derive(Debug, Clone, PartialEq)]
pub struct DuplicateParameterError {
    /// Section containing the unequal shadows.
    pub section: RbmShadowSection,
    /// Raw section-local index (not validated as a dense storage address).
    pub parameter_index: i64,
    /// Zero-based first row carrying this index.
    pub first_row: usize,
    /// Zero-based first conflicting row.
    pub conflicting_row: usize,
    /// Value from the first occurrence.
    pub first_value: Complex64,
    /// Value from the conflicting occurrence.
    pub conflicting_value: Complex64,
}

impl fmt::Display for DuplicateParameterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:?} idx={} has unequal shadow values at rows {} and {}",
            self.section, self.parameter_index, self.first_row, self.conflicting_row
        )
    }
}

impl std::error::Error for DuplicateParameterError {}

/// Diagnose unequal same-section, same-index RBM shadow values without mutation.
///
/// Numeric complex equality matches the original Julia manual diagnostic:
/// opposite signed zeros compare equal; duplicate NaNs compare unequal. A
/// singleton NaN is not a duplicate. No tolerance, repair, dense comparison,
/// allocation-width validation, or RNG operation is performed. Orbital mappings
/// have no independent shadow value in Rust and are not checked.
///
/// This explicitly optional API does not impose new pack/unpack rejection rules.
pub fn check_duplicate_consistency(data: &ExpertModeData) -> Result<(), DuplicateParameterError> {
    macro_rules! check {
        ($section:ident, $terms:expr) => {{
            let mut first: HashMap<i64, (usize, Complex64)> = HashMap::new();
            for (row, term) in $terms.iter().enumerate() {
                if let Some(&(first_row, first_value)) = first.get(&term.idx) {
                    if first_value != term.value {
                        return Err(DuplicateParameterError {
                            section: RbmShadowSection::$section,
                            parameter_index: term.idx,
                            first_row,
                            conflicting_row: row,
                            first_value,
                            conflicting_value: term.value,
                        });
                    }
                } else {
                    first.insert(term.idx, (row, term.value));
                }
            }
        }};
    }
    check!(ChargePhysLayer, data.charge_rbm_phys_layer_terms);
    check!(SpinPhysLayer, data.spin_rbm_phys_layer_terms);
    check!(GeneralPhysLayer, data.general_rbm_phys_layer_terms);
    check!(ChargeHiddenLayer, data.charge_rbm_hidden_layer_terms);
    check!(SpinHiddenLayer, data.spin_rbm_hidden_layer_terms);
    check!(GeneralHiddenLayer, data.general_rbm_hidden_layer_terms);
    check!(ChargePhysHidden, data.charge_rbm_phys_hidden_terms);
    check!(SpinPhysHidden, data.spin_rbm_phys_hidden_terms);
    check!(GeneralPhysHidden, data.general_rbm_phys_hidden_terms);
    Ok(())
}

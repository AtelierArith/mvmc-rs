//! Public, zero-based access to C-declared variational coefficient storage.
//!
//! Order is projection (Gutz/Jast/DH2/DH4), nine RBM sections, declared
//! Slater slots, then OptTrans. Access does not normalize or draw random numbers.
use crate::ExpertModeData;
use mvmc_expert_parsers::utils::parameter_init::all_complex_flag;
use num_complex::Complex64;
use std::fmt;

/// Parameter access rejected before coefficient or derived-weight mutation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParameterAccessError {
    /// Supplied vector does not cover the complete declared layout.
    LengthMismatch {
        /// Required coefficient count.
        expected: usize,
        /// Supplied coefficient count.
        actual: usize,
    },
    /// Zero-based parameter index is outside the complete layout.
    IndexOutOfRange {
        /// Requested index.
        index: usize,
        /// Declared coefficient count.
        count: usize,
    },
    /// Model storage cannot represent every declared coefficient safely.
    InvalidStorage(&'static str),
    /// A loaded projection/orbital declaration changed without clearing its binding.
    InvalidDeclaration(&'static str),
}
impl fmt::Display for ParameterAccessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch { expected, actual } => {
                write!(f, "parameter length {actual} != declared count {expected}")
            }
            Self::IndexOutOfRange { index, count } => {
                write!(f, "parameter index {index} outside declared count {count}")
            }
            Self::InvalidStorage(reason) => write!(f, "invalid parameter storage: {reason}"),
            Self::InvalidDeclaration(reason) => {
                write!(f, "invalid parameter declaration: {reason}")
            }
        }
    }
}
impl std::error::Error for ParameterAccessError {}

fn checked_count(data: &ExpertModeData) -> Result<usize, ParameterAccessError> {
    use ParameterAccessError::InvalidStorage;
    all_complex_flag(data).map_err(ParameterAccessError::InvalidDeclaration)?;
    if data.n_gutzwiller_idx < 0 || data.n_jastrow_idx < 0 || data.modpara.n_orbital_idx < 0 {
        return Err(InvalidStorage("negative declared width"));
    }
    let gutz = if data.n_gutzwiller_idx > 0 {
        data.n_gutzwiller_idx as usize
    } else {
        data.gutzwiller_terms.len()
    };
    let jast = if data.n_jastrow_idx > 0 {
        data.n_jastrow_idx as usize
    } else {
        data.jastrow_terms.len()
    };
    let dh2 = data
        .doublon_holon_2site_indices
        .len()
        .checked_mul(6)
        .ok_or(InvalidStorage("DH2 width overflow"))?;
    let dh4 = data
        .doublon_holon_4site_indices
        .len()
        .checked_mul(10)
        .ok_or(InvalidStorage("DH4 width overflow"))?;
    let rbm = data
        .rbm_section_sizes()
        .into_iter()
        .try_fold(0usize, |n, width| {
            n.checked_add(width)
                .ok_or(InvalidStorage("RBM width overflow"))
        })?;
    let slater = data.modpara.n_orbital_idx as usize;
    let count = [gutz, jast, dh2, dh4, rbm, slater, data.opt_trans.len()]
        .into_iter()
        .try_fold(0usize, |n, width| {
            n.checked_add(width)
                .ok_or(InvalidStorage("total width overflow"))
        })?;
    if data.gutzwiller_terms.len() != gutz
        || data.jastrow_terms.len() != jast
        || data.doublon_holon_2site_params.len() != dh2
        || data.doublon_holon_4site_params.len() != dh4
        || data.rbm_params.len() != rbm
        || data.slater_params.len() != slater
    {
        return Err(InvalidStorage(
            "coefficient storage differs from declared width",
        ));
    }
    if data
        .orbital_terms
        .iter()
        .any(|term| term.idx < 0 || term.idx as usize >= slater)
    {
        return Err(InvalidStorage("orbital mapping outside declared width"));
    }
    let widths = data.rbm_section_sizes();
    macro_rules! mapping {
        ($field:ident, $section:expr) => {
            if data
                .$field
                .iter()
                .any(|term| term.idx < 0 || term.idx as usize >= widths[$section])
            {
                return Err(InvalidStorage("RBM mapping outside declared section"));
            }
        };
    }
    mapping!(charge_rbm_phys_layer_terms, 0);
    mapping!(spin_rbm_phys_layer_terms, 1);
    mapping!(general_rbm_phys_layer_terms, 2);
    mapping!(charge_rbm_hidden_layer_terms, 3);
    mapping!(spin_rbm_hidden_layer_terms, 4);
    mapping!(general_rbm_hidden_layer_terms, 5);
    mapping!(charge_rbm_phys_hidden_terms, 6);
    mapping!(spin_rbm_phys_hidden_terms, 7);
    mapping!(general_rbm_phys_hidden_terms, 8);
    if let Some(weights) = &data.qp_weights {
        weights
            .qp_fix_weight
            .len()
            .checked_mul(data.opt_trans.len())
            .ok_or(InvalidStorage("QP weight width overflow"))?;
    }
    Ok(count)
}

/// Pack every declared coefficient, including unmapped RBM/Slater slots.
pub fn pack_parameters(data: &ExpertModeData) -> Result<Vec<Complex64>, ParameterAccessError> {
    checked_count(data)?;
    Ok(crate::sync::pack_variational_parameters(data))
}

/// Directly assign a complete vector, repairing every duplicate mapping.
///
/// Length/storage errors are nonmutating. Assigning OptTrans slots refreshes
/// existing QP weights even when their coefficient values are unchanged, as in
/// Julia's unpack. This does not create absent projection weights or normalize.
pub fn unpack_parameters(
    data: &mut ExpertModeData,
    values: &[Complex64],
) -> Result<(), ParameterAccessError> {
    let expected = checked_count(data)?;
    if values.len() != expected {
        return Err(ParameterAccessError::LengthMismatch {
            expected,
            actual: values.len(),
        });
    }
    crate::sync::unpack_variational_parameters(data, values);
    if !data.opt_trans.is_empty() {
        crate::qp::update_qp_weight_for(data);
    }
    Ok(())
}

/// Read a zero-based declared coefficient without inferring from mapped values.
pub fn get_parameter_value(
    data: &ExpertModeData,
    index: usize,
) -> Result<Complex64, ParameterAccessError> {
    let values = pack_parameters(data)?;
    values
        .get(index)
        .copied()
        .ok_or(ParameterAccessError::IndexOutOfRange {
            index,
            count: values.len(),
        })
}

/// Directly assign one zero-based coefficient and all mappings of that slot.
///
/// Errors occur before mutation. Unrelated duplicate mappings are not repaired
/// by this single-slot operation; bulk unpack explicitly repairs all mappings.
pub fn set_parameter_value(
    data: &mut ExpertModeData,
    index: usize,
    value: Complex64,
) -> Result<(), ParameterAccessError> {
    let count = checked_count(data)?;
    if index >= count {
        return Err(ParameterAccessError::IndexOutOfRange { index, count });
    }
    let layout = data.projection_layout();
    let rbm_end = layout.n_proj + data.count_rbm_parameters();
    let slater_end = rbm_end + data.slater_params.len();
    if index < layout.jastrow_offset {
        data.gutzwiller_terms[index].value = value;
    } else if index < layout.dh2_offset {
        data.jastrow_terms[index - layout.jastrow_offset].value = value;
    } else if index < layout.dh4_offset {
        data.doublon_holon_2site_params[index - layout.dh2_offset] = value;
    } else if index < layout.n_proj {
        data.doublon_holon_4site_params[index - layout.dh4_offset] = value;
    } else if index < rbm_end {
        data.set_rbm_parameter(index - layout.n_proj, value);
    } else if index < slater_end {
        let local = index - rbm_end;
        data.slater_params[local] = value;
    } else {
        data.opt_trans[index - slater_end] = value;
        crate::qp::update_qp_weight_for(data);
    }
    Ok(())
}

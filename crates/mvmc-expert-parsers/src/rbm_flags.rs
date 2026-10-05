//! Manual file-driven RBM flags, separate from coefficient/loader finalization.

use crate::parsers::rbm::parse_rbm_content;
use crate::types::RbmParameter;
use crate::utils::file::read_def_file;
use crate::ExpertModeData;
use std::collections::BTreeSet;
use std::io;
use std::path::Path;

/// C coefficient block order for the nine supported RBM definitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(usize)]
pub enum RbmDefinitionKey {
    /// Charge physical layer.
    ChargePhysLayer,
    /// Spin physical layer.
    SpinPhysLayer,
    /// General physical layer.
    GeneralPhysLayer,
    /// Charge hidden layer.
    ChargeHiddenLayer,
    /// Spin hidden layer.
    SpinHiddenLayer,
    /// General hidden layer.
    GeneralHiddenLayer,
    /// Charge physical/hidden connections.
    ChargePhysHidden,
    /// Spin physical/hidden connections.
    SpinPhysHidden,
    /// General physical/hidden connections.
    GeneralPhysHidden,
}

/// One selected complete RBM file; paths are resolved by the caller.
#[derive(Debug, Clone, Copy)]
pub struct RbmOptimizationSource<'a> {
    /// Typed native family.
    pub key: RbmDefinitionKey,
    /// Definition path passed to the unchanged ordinary Rust reader.
    pub path: &'a Path,
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn add(a: usize, b: usize) -> io::Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| invalid("RBM flag offset overflow"))
}

fn mul(a: usize, b: usize) -> io::Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("RBM flag offset overflow"))
}

fn projection_width(declared: i64, fallback: usize) -> io::Result<usize> {
    if declared == 0 {
        Ok(fallback)
    } else {
        usize::try_from(declared).map_err(|_| invalid("invalid projection width"))
    }
}

fn mapping_width<T: RbmParameter>(terms: &[T], width: usize) -> io::Result<()> {
    if terms.iter().any(|term| {
        usize::try_from(term.idx())
            .ok()
            .is_none_or(|index| index >= width)
    }) {
        return Err(invalid("RBM mapping index exceeds explicit section width"));
    }
    Ok(())
}

/// Transactionally refresh selected RBM flags without resizing or changing data layout.
///
/// `widths` describes all nine blocks in [`RbmDefinitionKey`] order, including
/// unselected and unmapped/reserved slots. Loaded nonzero widths must match.
/// Programmatic callers may supply explicit widths for unset declarations;
/// these widths are not installed into data and do not certify runtime storage.
/// A selected file for an explicit zero-width block is still strictly parsed,
/// then left inactive, matching the original helper's zero-section skip.
/// Selected files use the existing C-compatible complete geometry/count reader.
/// Each file's local positive complex header copies real flags into imaginary
/// flags; zero/negative local headers use established Rust deterministic zero.
/// Orbital/global mode and coefficient values do not determine these flags.
///
/// Empty sources are a true no-op. Repeated exact keys are `InvalidInput`.
/// IO/coherence/count/capacity errors leave the complete data unchanged. Success
/// changes optimization components only, retaining array length and unselected
/// bytes. No parameter initialization, mapping replacement or RNG draw occurs.
pub fn refresh_rbm_optimization_flags(
    data: &mut ExpertModeData,
    widths: &[usize; 9],
    sources: &[RbmOptimizationSource<'_>],
) -> io::Result<()> {
    if sources.is_empty() {
        return Ok(());
    }
    let mut seen = BTreeSet::new();
    for source in sources {
        if !seen.insert(source.key) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "duplicate RBM source key",
            ));
        }
    }
    for (&stored, &explicit) in data.rbm_section_widths.iter().zip(widths) {
        if explicit > i32::MAX as usize || (stored != 0 && stored != explicit) {
            return Err(invalid("explicit RBM width contradicts loaded declaration"));
        }
    }
    mapping_width(&data.charge_rbm_phys_layer_terms, widths[0])?;
    mapping_width(&data.spin_rbm_phys_layer_terms, widths[1])?;
    mapping_width(&data.general_rbm_phys_layer_terms, widths[2])?;
    mapping_width(&data.charge_rbm_hidden_layer_terms, widths[3])?;
    mapping_width(&data.spin_rbm_hidden_layer_terms, widths[4])?;
    mapping_width(&data.general_rbm_hidden_layer_terms, widths[5])?;
    mapping_width(&data.charge_rbm_phys_hidden_terms, widths[6])?;
    mapping_width(&data.spin_rbm_phys_hidden_terms, widths[7])?;
    mapping_width(&data.general_rbm_phys_hidden_terms, widths[8])?;

    let mut offset = add(
        projection_width(data.n_gutzwiller_idx, data.gutzwiller_terms.len())?,
        projection_width(data.n_jastrow_idx, data.jastrow_terms.len())?,
    )?;
    offset = add(offset, mul(data.doublon_holon_2site_indices.len(), 6)?)?;
    offset = add(offset, mul(data.doublon_holon_4site_indices.len(), 10)?)?;
    let mut offsets = [0; 9];
    for (index, &width) in widths.iter().enumerate() {
        offsets[index] = offset;
        offset = add(offset, width)?;
    }
    mul(offset, 2)?;
    let mut staged = data.optimization_flags.clone();
    for source in sources {
        let index = source.key as usize;
        let hidden = [
            data.modpara.nneuron_charge,
            data.modpara.nneuron_spin,
            data.modpara.nneuron_general,
        ][index % 3];
        let section = parse_rbm_content(
            &read_def_file(source.path)?,
            index,
            data.modpara.nsite,
            hidden,
        )?;
        if widths[index] == 0 {
            continue;
        }
        if section.width != widths[index] {
            return Err(invalid(
                "selected RBM file count differs from explicit width",
            ));
        }
        for (local, flag) in section.opt_flags {
            let local = usize::try_from(local).map_err(|_| invalid("negative RBM flag index"))?;
            let component = mul(add(offsets[index], local)?, 2)?;
            let end = add(component, 2)?;
            let pair = staged
                .get_mut(component..end)
                .ok_or_else(|| invalid("selected RBM flags exceed existing capacity"))?;
            pair[0] = flag;
            pair[1] = if section.is_complex { flag } else { 0 };
        }
    }
    data.optimization_flags = staged;
    Ok(())
}

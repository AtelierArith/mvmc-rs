//! Typed single-family definition loading, separate from namelist finalization.

use crate::parsers::{coulomb, exchange, hund, pairhop, trans};
use crate::ExpertModeData;
use std::io;
use std::path::Path;

/// Hamiltonian definitions supported by the bounded public dispatch operation.
/// InterAll and variational/projection definitions are not part of this API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HamiltonianDefinitionKind {
    /// Transfer coordinates, spins and complex coefficients.
    Transfer,
    /// On-site Coulomb coefficients.
    CoulombIntra,
    /// Pair Coulomb coefficients.
    CoulombInter,
    /// Hund pairs, without a sign transformation.
    Hund,
    /// Exchange pairs, without a sign transformation.
    Exchange,
    /// Pair-hopping rows expanded forward then reverse.
    PairHop,
}

/// Load one supported C definition into caller-owned data.
///
/// Uses `data.modpara.nsite` and the same strict readers as namelist loading.
/// Each successful read replaces only its selected Hamiltonian family. On any
/// read or consumed-record error, data is unchanged; no partial rows are stored.
/// This does not finalize projection/orbital/RBM flags, initialize parameters,
/// add namelist entries or certify a runnable model. It is the bounded public
/// update operation from Julia's dispatch architecture with C input authority,
/// not a permissive headerless payload helper.
pub fn load_hamiltonian_definition(
    data: &mut ExpertModeData,
    kind: HamiltonianDefinitionKind,
    path: impl AsRef<Path>,
) -> io::Result<()> {
    let path = path.as_ref();
    let nsite = data.modpara.nsite;
    match kind {
        HamiltonianDefinitionKind::Transfer => {
            data.transfer_terms = trans::parse_trans_definition(path, nsite)?;
        }
        HamiltonianDefinitionKind::CoulombIntra => {
            data.coulomb_intra_terms = coulomb::parse_coulomb_intra_definition(path, nsite)?;
        }
        HamiltonianDefinitionKind::CoulombInter => {
            data.coulomb_inter_terms = coulomb::parse_coulomb_inter_definition(path, nsite)?;
        }
        HamiltonianDefinitionKind::Hund => {
            data.hund_terms = hund::parse_hund_definition(path, nsite)?;
        }
        HamiltonianDefinitionKind::Exchange => {
            data.exchange_terms = exchange::parse_exchange_definition(path, nsite)?;
        }
        HamiltonianDefinitionKind::PairHop => {
            data.pair_hop_terms = pairhop::parse_pairhop_definition(path, nsite)?;
        }
    }
    Ok(())
}

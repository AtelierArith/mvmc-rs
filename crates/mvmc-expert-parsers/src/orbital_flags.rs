//! Explicit caller-owned file refresh, separate from native loader finalization.

use crate::parsers::orbital::{parse_orbital_def, OrbitalKind};
use crate::ExpertModeData;
use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::path::Path;

/// Native orbital key, including the distinct AP alias.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum OrbitalDefinitionKey {
    /// Native `Orbital` AP alias.
    Orbital,
    /// Opposite-spin mappings.
    AntiParallel,
    /// Equal-spin mappings expanded into two parameter slots.
    Parallel,
    /// Combined spin-site mappings.
    General,
}

impl OrbitalDefinitionKey {
    fn name(self) -> &'static str {
        match self {
            Self::Orbital => "Orbital",
            Self::AntiParallel => "OrbitalAntiParallel",
            Self::Parallel => "OrbitalParallel",
            Self::General => "OrbitalGeneral",
        }
    }

    fn kind(self) -> OrbitalKind {
        match self {
            Self::Orbital | Self::AntiParallel => OrbitalKind::AntiParallel,
            Self::Parallel => OrbitalKind::Parallel,
            Self::General => OrbitalKind::General,
        }
    }
}

/// Explicit raw binding for each active family, selected or unselected.
#[derive(Debug, Clone, Copy)]
pub struct OrbitalRawDeclaration {
    /// Exact native key; aliases contribute separately to the raw header sum.
    pub key: OrbitalDefinitionKey,
    /// Positive declared width before Parallel expansion.
    pub parameter_count: usize,
    /// Signed C header before aggregate positive normalization.
    pub complex_header: i32,
}

/// A selected complete definition file, resolved by the caller.
#[derive(Debug, Clone, Copy)]
pub struct OrbitalOptimizationSource<'a> {
    /// Exact declared key.
    pub key: OrbitalDefinitionKey,
    /// Path passed unchanged to the existing strict reader.
    pub path: &'a Path,
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn add(a: usize, b: usize) -> io::Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| invalid("orbital flag offset overflow"))
}

fn mul(a: usize, b: usize) -> io::Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("orbital flag offset overflow"))
}

fn width(declared: i64, fallback: usize) -> io::Result<usize> {
    if declared == 0 {
        Ok(fallback)
    } else {
        usize::try_from(declared).map_err(|_| invalid("negative or unrepresentable width"))
    }
}

fn prefix(data: &ExpertModeData) -> io::Result<usize> {
    let mut count = add(
        width(data.n_gutzwiller_idx, data.gutzwiller_terms.len())?,
        width(data.n_jastrow_idx, data.jastrow_terms.len())?,
    )?;
    count = add(count, mul(data.doublon_holon_2site_indices.len(), 6)?)?;
    count = add(count, mul(data.doublon_holon_4site_indices.len(), 10)?)?;
    for section in data.rbm_section_widths {
        count = add(count, section)?;
    }
    Ok(count)
}

/// Refresh only selected orbital optimization components, atomically and without resize.
///
/// Explicit declarations cover every active native orbital key, including unselected
/// families. Existing raw bindings cannot be overridden. Files use the unchanged
/// strict C-compatible reader; this is not a permissive Julia payload parser.
///
/// Unbound programmatic mode fields may be unset (zero); supplied declarations
/// resolve the manual layout without installing modes or headers. This does not
/// certify a runnable C model or repair loaded declaration/state consistency.
/// Loaded `Orbitals` metadata must agree with current term annotations and the
/// supplied raw header sum's nonzero predicate. This is a stale-binding/shape
/// guard, not new C model admission or coefficient-driven mode inference.
/// Empty `sources` is a true no-op. Duplicate keys are `InvalidInput`; all other
/// coherence/capacity failures are `InvalidData`. IO errors retain their kind.
/// Coefficients, mappings, all metadata, unselected flags and flag length are
/// unchanged. No initialization or RNG operations occur.
pub fn refresh_orbital_optimization_flags(
    data: &mut ExpertModeData,
    declarations: &[OrbitalRawDeclaration],
    sources: &[OrbitalOptimizationSource<'_>],
) -> io::Result<()> {
    if sources.is_empty() {
        return Ok(());
    }
    let mut declared = BTreeMap::new();
    let mut raw = 0_i64;
    for declaration in declarations {
        if declared.insert(declaration.key, declaration).is_some() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "duplicate orbital declaration key",
            ));
        }
        if declaration.parameter_count == 0 || declaration.parameter_count > i32::MAX as usize {
            return Err(invalid("orbital width must be a positive C integer"));
        }
        raw = raw
            .checked_add(i64::from(declaration.complex_header))
            .ok_or_else(|| invalid("orbital header sum overflow"))?;
    }
    let mut selected = BTreeSet::new();
    for source in sources {
        if !selected.insert(source.key) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "duplicate orbital source key",
            ));
        }
        if !declared.contains_key(&source.key) {
            return Err(invalid("selected orbital key has no explicit declaration"));
        }
    }
    let ap_alias = declared.get(&OrbitalDefinitionKey::Orbital);
    let ap_native = declared.get(&OrbitalDefinitionKey::AntiParallel);
    if let (Some(a), Some(b)) = (ap_alias, ap_native) {
        if a.parameter_count != b.parameter_count {
            return Err(invalid("AP aliases disagree on declared width"));
        }
    }
    let ap = ap_native
        .or(ap_alias)
        .map(|d| d.parameter_count)
        .unwrap_or(0);
    let parallel = declared
        .get(&OrbitalDefinitionKey::Parallel)
        .map(|d| d.parameter_count)
        .unwrap_or(0);
    let general = declared
        .get(&OrbitalDefinitionKey::General)
        .map(|d| d.parameter_count)
        .unwrap_or(0);
    if (general != 0 && (ap != 0 || parallel != 0)) || (general == 0 && ap == 0) {
        return Err(invalid("unsupported orbital family combination"));
    }
    let total = if general != 0 {
        general
    } else {
        add(ap, mul(parallel, 2)?)?
    };
    if usize::try_from(data.modpara.n_orbital_idx).ok() != Some(total) {
        return Err(invalid(
            "caller Slater width differs from explicit declaration",
        ));
    }
    let expected_modes = [
        i64::from(ap != 0),
        i64::from(parallel != 0),
        i64::from(general != 0 || parallel != 0),
    ];
    let modes = [
        data.i_flg_orbital_anti_parallel,
        data.i_flg_orbital_parallel,
        data.i_flg_orbital_general,
    ];
    let loaded = data.native_complex_declarations.contains_key("Orbitals");
    for (actual, expected) in modes.into_iter().zip(expected_modes) {
        if actual != expected && (loaded || actual != 0) {
            return Err(invalid("caller orbital mode contradicts declaration"));
        }
    }
    if data.n_orbital_anti_parallel != ap as i64 && (loaded || data.n_orbital_anti_parallel != 0) {
        return Err(invalid("caller AP offset contradicts declaration"));
    }
    for key in [
        OrbitalDefinitionKey::Orbital,
        OrbitalDefinitionKey::AntiParallel,
        OrbitalDefinitionKey::Parallel,
        OrbitalDefinitionKey::General,
    ] {
        if let Some(header) = data.native_complex_headers.get(key.name()) {
            if declared.get(&key).map(|d| d.complex_header) != Some(*header) {
                return Err(invalid(
                    "existing signed orbital binding cannot be overridden",
                ));
            }
        } else if loaded && declared.contains_key(&key) {
            return Err(invalid("loaded orbital key has no raw binding"));
        }
    }
    if let Some(&annotation) = data.native_complex_declarations.get("Orbitals") {
        if annotation != data.orbital_terms.iter().any(|term| term.is_complex)
            || annotation != (raw != 0)
        {
            return Err(invalid("loaded orbital declaration is stale"));
        }
    }
    if data.orbital_terms.iter().any(|term| {
        usize::try_from(term.idx)
            .ok()
            .is_none_or(|idx| idx >= total)
    }) {
        return Err(invalid(
            "caller mapping index exceeds declared orbital width",
        ));
    }
    let offset = prefix(data)?;
    // Validate checked global end without demanding that unselected slots exist.
    mul(add(offset, total)?, 2)?;
    let complex = if raw > 0 { 1 } else { raw };
    let mut staged = data.optimization_flags.clone();
    for source in sources {
        let declaration = declared[&source.key];
        let section = parse_orbital_def(source.path, data.modpara.nsite, source.key.kind())?;
        if section.complex_type != declaration.complex_header
            || usize::try_from(section.n_orbital_idx).ok() != Some(declaration.parameter_count)
        {
            return Err(invalid(
                "selected file header/count differs from explicit binding",
            ));
        }
        for (idx, flag) in section.opt_flags {
            let idx = usize::try_from(idx).map_err(|_| invalid("negative flag index"))?;
            let (start, slots) = if source.key == OrbitalDefinitionKey::Parallel {
                (add(ap, mul(idx, 2)?)?, 2)
            } else {
                (idx, 1)
            };
            for slot in start..add(start, slots)? {
                let component = mul(add(offset, slot)?, 2)?;
                let end = add(component, 2)?;
                let pair = staged
                    .get_mut(component..end)
                    .ok_or_else(|| invalid("selected orbital flags exceed existing capacity"))?;
                pair[0] = flag;
                pair[1] = if source.key == OrbitalDefinitionKey::Parallel {
                    complex
                } else if complex > 0 {
                    flag
                } else {
                    0
                };
            }
        }
    }
    data.optimization_flags = staged;
    Ok(())
}

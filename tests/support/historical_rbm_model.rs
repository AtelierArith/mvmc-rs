//! Programmatic sparse models for archived Julia numerical regressions.
//! These models do not establish C acceptance of the archived definitions.
use mvmc_expert_parsers::types::*;
use num_complex::Complex64;
use std::collections::{BTreeMap, BTreeSet};

pub fn restore(data: &mut ExpertModeData, sections: &[(usize, String)]) {
    let original_flags = data.optimization_flags.clone();
    let mut raw_flags = Vec::new();
    for (section, text) in sections {
        let columns = [2, 2, 3, 2, 2, 2, 3, 3, 4][*section];
        let lines: Vec<_> = text.lines().collect();
        let complex = lines
            .get(2)
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|value| value.parse::<i64>().ok())
            .unwrap_or(0)
            != 0;
        let mut maps = Vec::new();
        let mut flags = BTreeMap::new();
        let mut seen = BTreeSet::new();
        let mut in_flags = false;
        let mut failed = false;
        for line in lines.iter().skip(5) {
            let fields: Vec<_> = line.split_whitespace().collect();
            if fields.is_empty() {
                continue;
            }
            if fields.len() == 2 && !in_flags {
                in_flags = columns != 2 || !seen.insert(fields[0].to_owned());
            }
            let values: Vec<_> = fields
                .iter()
                .map(|value| value.parse::<i64>().unwrap_or(-1))
                .collect();
            if in_flags {
                if values.len() >= 2 && values[0] >= 0 && values[1] >= 0 {
                    flags.insert(values[0], values[1]);
                }
            } else if values.len() == columns {
                failed |= values.iter().any(|&v| v < 0);
                maps.push(values);
            }
        }
        if failed {
            raw_flags.push((*section, flags, complex));
            continue;
        }
        // Explicit dimensions of this internal sparse model, not C headers.
        let width = maps
            .iter()
            .map(|row| row[columns - 1])
            .max()
            .map_or(0, |v| v as usize + 1);
        data.rbm_section_widths[*section] = width;
        let zero = Complex64::new(0.0, 0.0);
        macro_rules! layer {
            ($field:ident, $kind:ident) => {{
                data.$field = maps
                    .iter()
                    .map(|v| $kind {
                        site: v[0],
                        idx: v[1],
                        value: zero,
                        is_complex: complex,
                    })
                    .collect();
            }};
        }
        macro_rules! coupling {
            ($field:ident, $kind:ident) => {{
                data.$field = maps
                    .iter()
                    .map(|v| $kind {
                        site1: v[0],
                        site2: v[1],
                        idx: v[2],
                        value: zero,
                        is_complex: complex,
                    })
                    .collect();
            }};
        }
        match section {
            0 => layer!(charge_rbm_phys_layer_terms, ChargeRBMPhysLayerTerm),
            1 => layer!(spin_rbm_phys_layer_terms, SpinRBMPhysLayerTerm),
            2 => {
                data.general_rbm_phys_layer_terms = maps
                    .iter()
                    .map(|v| GeneralRBMPhysLayerTerm {
                        site: v[0],
                        spin: v[1],
                        idx: v[2],
                        value: zero,
                        is_complex: complex,
                    })
                    .collect()
            }
            3 => layer!(charge_rbm_hidden_layer_terms, ChargeRBMHiddenLayerTerm),
            4 => layer!(spin_rbm_hidden_layer_terms, SpinRBMHiddenLayerTerm),
            5 => layer!(general_rbm_hidden_layer_terms, GeneralRBMHiddenLayerTerm),
            6 => coupling!(charge_rbm_phys_hidden_terms, ChargeRBMPhysHiddenTerm),
            7 => coupling!(spin_rbm_phys_hidden_terms, SpinRBMPhysHiddenTerm),
            8 => {
                data.general_rbm_phys_hidden_terms = maps
                    .iter()
                    .map(|v| GeneralRBMPhysHiddenTerm {
                        site1: v[0],
                        spin: v[1],
                        site2: v[2],
                        idx: v[3],
                        value: zero,
                        is_complex: complex,
                    })
                    .collect()
            }
            _ => unreachable!(),
        }
        raw_flags.push((*section, flags, complex));
    }
    let nproj = data.projection_layout().n_proj;
    let width = data.count_rbm_parameters();
    // These internal archived models explicitly choose the old divisor-one
    // normalization when no geometry was supplied. Native input tests cover
    // the actual zero/negative C divisor without a production fallback.
    let total = data.modpara.nneuron
        + data.modpara.nneuron_charge
        + data.modpara.nneuron_spin
        + data.modpara.nneuron_general;
    if width != 0 && total <= 0 {
        data.modpara.nneuron += 1 - total;
    }
    data.rbm_params = vec![Complex64::new(0.0, 0.0); width];
    let split = (2 * nproj).min(original_flags.len());
    data.optimization_flags = original_flags[..split].to_vec();
    data.optimization_flags.resize(2 * (nproj + width), 1);
    data.optimization_flags
        .extend_from_slice(&original_flags[split..]);
    for (section, flags, complex) in raw_flags {
        let offset = nproj + data.rbm_section_widths[..section].iter().sum::<usize>();
        if data.rbm_section_widths[section] != 0 {
            mvmc_expert_parsers::utils::opt_flag::set_rbm_opt_flags(data, &flags, offset, complex);
        }
    }
}

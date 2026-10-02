//! Cross-rank variational-parameter sync.
//!
//! Port target: `MVMCOptimizers.jl/src/parameter_sync.jl`. Single-process
//! today; takes `&dyn Reducer` so Phase 7 can drop in an MPI impl without
//! touching the call sites.

use mvmc_expert_parsers::utils::parameter_init::sync_modified_parameter as sync_inner;
use mvmc_expert_parsers::ExpertModeData;
use num_complex::Complex64;

use crate::reducer::Reducer;

/// `sync_modified_parameter!(data)` mirror. The `Reducer` is plumbed
/// through so a future MPI build can broadcast parameter values before
/// local correlation shifts and Slater/OptTrans normalization.
///
/// Mirrors `MVMCOptimizers.jl/src/parameter_sync.jl`.
pub fn sync_modified_parameter<R: Reducer + ?Sized>(data: &mut ExpertModeData, reducer: &R) {
    let mut values = pack_variational_parameters(data);
    reducer.broadcast_c64(0, &mut values);
    unpack_variational_parameters(data, &values);
    sync_modified_parameter_local(data, true);
}

fn pack_variational_parameters(data: &ExpertModeData) -> Vec<Complex64> {
    let mut values = data.projection_parameters();
    values.extend(data.rbm_parameters());
    values.extend(data.slater_params.iter().copied());
    values.extend(data.opt_trans.iter().copied());
    values
}

fn unpack_variational_parameters(data: &mut ExpertModeData, values: &[Complex64]) {
    let projection_len = data.projection_layout().n_proj;
    let rbm_len = data.count_rbm_parameters();
    let slater_len = data.slater_params.len();
    let opt_trans_len = data.opt_trans.len();
    let expected = projection_len + rbm_len + slater_len + opt_trans_len;
    assert_eq!(
        values.len(),
        expected,
        "variational parameter layout changed during MPI broadcast"
    );

    let mut offset = 0;
    let projection = &values[offset..offset + projection_len];
    let layout = data.projection_layout();
    for (term, value) in data
        .gutzwiller_terms
        .iter_mut()
        .take(layout.n_gutzwiller)
        .zip(&projection[layout.gutzwiller_offset..])
    {
        term.value = *value;
    }
    for (term, value) in data
        .jastrow_terms
        .iter_mut()
        .take(layout.n_jastrow)
        .zip(&projection[layout.jastrow_offset..])
    {
        term.value = *value;
    }
    data.doublon_holon_2site_params
        .iter_mut()
        .take(6 * layout.n_dh2)
        .zip(&projection[layout.dh2_offset..])
        .for_each(|(dst, src)| *dst = *src);
    data.doublon_holon_4site_params
        .iter_mut()
        .take(10 * layout.n_dh4)
        .zip(&projection[layout.dh4_offset..])
        .for_each(|(dst, src)| *dst = *src);
    offset += projection_len;

    data.set_rbm_parameters(values[offset..offset + rbm_len].to_vec());
    offset += rbm_len;
    data.slater_params
        .copy_from_slice(&values[offset..offset + slater_len]);
    offset += slater_len;
    data.opt_trans
        .copy_from_slice(&values[offset..offset + opt_trans_len]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct BroadcastReducer {
        replacement: Vec<Complex64>,
    }

    impl Reducer for BroadcastReducer {
        fn allreduce_sum_f64(&self, _buf: &mut [f64]) {}
        fn allreduce_sum_c64(&self, _buf: &mut [Complex64]) {}
        fn allreduce_sum_i64(&self, _buf: &mut [i64]) {}

        fn broadcast_c64(&self, _root: usize, buf: &mut [Complex64]) {
            buf.copy_from_slice(&self.replacement);
        }
    }

    #[test]
    fn broadcasts_and_applies_canonical_variational_parameters() {
        let mut data = ExpertModeData::new();
        data.slater_params = vec![Complex64::new(1.0, 0.0)];
        data.opt_trans = vec![Complex64::new(2.0, 0.0)];
        let replacement = vec![Complex64::new(0.5, -0.25), Complex64::new(0.8, -0.6)];
        sync_modified_parameter(
            &mut data,
            &BroadcastReducer {
                replacement: replacement.clone(),
            },
        );
        let scale = 4.0 / replacement[0].norm();
        assert!((data.slater_params[0] - replacement[0] * scale).norm() < 1.0e-15);
        // OptTrans is normalized by the Julia synchronization step after the
        // broadcast, so its amplitude is one while preserving its phase.
        assert_eq!(data.opt_trans[0], replacement[1]);
    }
}

/// Apply Julia's local optimizer synchronization, including OptTrans normalization.
/// Correlation shifts can be disabled independently of Slater/OptTrans rescaling.
pub fn sync_modified_parameter_local(data: &mut ExpertModeData, shift_correlations: bool) {
    sync_inner(data, shift_correlations);
    let mut xmax = 0.0_f64;
    for value in &data.opt_trans {
        let amplitude = mvmc_expert_parsers::utils::julia_hypot::hypot(value.re, value.im);
        if amplitude.is_nan() {
            xmax = f64::NAN;
            break;
        }
        xmax = xmax.max(amplitude);
    }
    if xmax > 0.0 {
        let ratio = 1.0 / xmax;
        for value in &mut data.opt_trans {
            *value *= ratio;
        }
    }
}

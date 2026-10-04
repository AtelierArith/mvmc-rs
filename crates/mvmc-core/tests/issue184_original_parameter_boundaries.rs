//! Original Julia M0588–M0590 first-slot signed zero and S122 serial sync.
//! These are parameter API contracts, not model/MPI numerical parity checks.
use mvmc_core::{
    pack_parameters,
    sync::{sync_modified_parameter, sync_modified_parameter_local},
    unpack_parameters, ExpertModeData, SingleProcessReducer,
};
use num_complex::Complex64;
use std::path::Path;

fn original_heisenberg() -> ExpertModeData {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/original_heisenberg_parser_184/namelist.def");
    let data = mvmc_expert_parsers::parse_expert_mode_files(path).unwrap();
    assert!(data.input_errors.is_empty(), "{:?}", data.input_errors);
    data
}

#[test]
fn original_first_projection_complex_signed_zero_survives_public_unpack() {
    let mut data = original_heisenberg();
    assert!(data.projection_layout().n_gutzwiller > 0);
    let mut values = pack_parameters(&data).unwrap();
    // Julia para_zero[1] is the same first slot as zero-based Rust values[0].
    values[0] = Complex64::new(-0.0, -0.0);
    unpack_parameters(&mut data, &values).unwrap();
    let actual = pack_parameters(&data).unwrap()[0];
    assert_eq!(actual, Complex64::new(-0.0, -0.0));
    assert_eq!(actual.re.to_bits(), (-0.0_f64).to_bits());
    assert_eq!(actual.im.to_bits(), (-0.0_f64).to_bits());
}

#[test]
fn original_heisenberg_serial_context_sync_matches_local_sync() {
    let mut contextual = original_heisenberg();
    let mut local = original_heisenberg();
    // Original Julia S122 parses two identical original inputs, no initialization
    // or RNG calls, then compares context and local synchronization pathways.
    sync_modified_parameter(&mut contextual, &SingleProcessReducer);
    sync_modified_parameter_local(&mut local, true);
    let contextual = pack_parameters(&contextual).unwrap();
    let local = pack_parameters(&local).unwrap();
    assert_eq!(contextual.len(), local.len());
    // Original array ≈ compares vector norms, not independent component budgets.
    // Retain default sqrt(eps) rtol/atol0, not a numerical model tolerance.
    let difference = contextual
        .iter()
        .zip(&local)
        .map(|(a, b)| (*a - *b).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let context_norm = contextual.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
    let local_norm = local.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
    assert!(
        contextual == local || difference <= f64::EPSILON.sqrt() * context_norm.max(local_norm)
    );
}

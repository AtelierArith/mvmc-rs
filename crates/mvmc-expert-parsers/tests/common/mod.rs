#[path = "../../../../tests/support/c_orbital_rng.rs"]
mod c_orbital_rng;
pub use c_orbital_rng::declared_slater_rng;

#[path = "../../../../tests/support/historical_orbital_model.rs"]
mod historical_orbital_model;
pub use historical_orbital_model::historical_kernel_model;

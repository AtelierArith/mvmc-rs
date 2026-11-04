//! Device-resident CUDA GPU Opt and PhysCal for mvmc-rs (Julia GPU contract).
//!
//! * [`opt::run_gpu_para_opt_from_namelist`] runs the resident walker-ensemble optimizer.
//! * [`physcal::run_gpu_phys_cal_from_namelist`] runs the resident fixed-parameter PhysCal.
//! * [`interval`] holds the batched interval sampler kernels shared by both runners.
//! * [`versions`] queries driver / CUDA / NVRTC / cuBLAS / cuSOLVER versions by `dlopen`.
//!
//! `mvmc-core` is used only for input parsing, validation and parameter I/O; the CPU C-order
//! runner stays the parity reference and nothing here changes its RNG draw order.

pub mod interval;
pub mod opt;
pub use opt::physcal;
pub mod pfaffian;
pub mod versions;

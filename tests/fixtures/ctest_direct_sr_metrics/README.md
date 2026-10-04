# Actual canonical direct-SR endpoints

52 actual Julia 1.13.1 pre-factorization/post-substitution captures: all thirteen
canonical models, independently restarted at their original seeds for 1/2/3/50
steps. `direct-sr-increment.txt` is the actual overwritten solver RHS before
parameter update/synchronization, **not** reconstructed from final parameters.
The raw matrix is column-major; the authoritative solve uses its upper triangle.
Active indices are zero-based C real/imaginary component indices. Optimization
flags, canonical modpara, input hashes, seed/overlay settings and original and
effective counts accompany every case. Both counts are explicitly overridden.

Captured by [offline probe](../../../c_toolbox/ctest_direct_sr_metrics.md) in
`/tmp/mvmc-ctest-direct-sr.55cR5X`; process 5989 completed successfully.
These are mixed C-contract/Julia solves with actual Julia ILP64 OpenBLAS, not
full native-C executable or MPI solver validation. FSZ uses the same separately
validated native C energy bridge (binary SHA-256
`dc73cadebf814984445aa5f31ccbb5388505b70b9f74628998e8c5d1fa461d72`).
Source/extraction and actual environment provenance are retained at archive root.
The environment/source audit files are explicitly post-generation observations.
All 52 configuration and next-624-word RNG records byte-match the existing
independent canonical prefix observations; no Rust-generated value is used.

Recorded 256-bit residual metrics apply to the actual stored Float64 systems
and increments. Largest normwise backward error: `4.677538293263316e-17`;
largest componentwise backward error: `3.509860472961454e-16`;
largest condition-2 estimate: `3.34976100140845e7`. Raw symmetry discrepancy
is zero in all cases. Conditions are estimates, not interval certificates.
These endpoint records supplement #190's existing `128*n*epsilon` backward
check, not a new tolerance policy. They do not independently establish a
uniform forward bound on all 50-step trajectories or on another platform;
conditioning cannot excuse discrete or RNG drift. GeneralRBM CG is not included.

The existing Rust prefix comparisons passed using their unchanged 1e-11
energy/parameter and 1e-12 SR-buffer bounds. This independent solve capture is
reference generation/evidence, **not** an additional Rust model execution.
No ordinary Rust test needs this probe, C, Julia, or toolbox reads at runtime.

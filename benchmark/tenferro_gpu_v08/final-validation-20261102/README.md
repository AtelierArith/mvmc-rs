# Post-ownership CUDA correctness verification

Source: `b1ffe5f5a343c8f094199de7889ba47470e88f88`. These are correctness gates,
not fresh before/after performance measurements. Native Linux x86_64,
Rust 1.99.0, standalone CUDA workspace's optimized `test` profile, BLAS/OMP/Rayon=1,
NVIDIA driver 580.178.04. Device identities are in `devices.txt`.

- GPU 1 (RTX 3060): 34 selected Pfaffian/extension, transfer, sampler, measurement,
  SR, routing and supplied-tape tests passed. The transfer microbenchmark is the
  single filtered test; no unavailable-device skip was used.
- GPU 1: six resident Opt gates passed, including historical Julia general-operator
  and L128 references, 20-step repeatability, failed SR and single-walker behavior.
- GPUs 0/1: one cross-device buffer/RNG rejection and subsequent owner-reuse test
  passed. `two-device-precondition.txt` verifies that the CUDA driver actually
  exposed two devices, so this was not the test's early return for missing hardware.

Full nextest logs and commands are retained. `MVMC_RS_CUDA_GATE=1` requires a
working GPU for the applicable gates. `run.sh` reproduces the three groups;
adapt its host-specific library and target paths. The two-device precondition
uses the CUDA driver only for metadata, outside Cargo tests. Fixtures are
checked-in Julia references with their existing provenance, not newly generated
or newly executed Julia 1.13 results; `fixture-hashes.txt` identifies them.

This verifies runtime behavior after the Engine resource/affinity ownership
changes. It supplements the migration's archived matched CPU/CUDA measurements
without assigning their timings to this newer revision. The audit also found
legacy arithmetic `to_bits()` assertions in the GPU SR gates; #623 separately
tracks their numerical-policy repair. Passing those old assertions is not a
claim that they implement the portable comparison policy.

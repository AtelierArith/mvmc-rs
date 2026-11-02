# Linux MPI calling-thread placement correctness

This report validates the opt-in `MVMC_RS_MPI_CPUS_PER_RANK` startup path;
it contains no performance measurements. The implementation follows placement
planner PR #619 and process-local Engine reservation PR #616 (Related to #583).

Environment: Linux x86_64, Xeon E5-2699 v3, Docker logical CPU mask 0–7,
Rust 1.99.0, MPICH 4.2, test-fast profile. Container image:
`vsc-mvmc-rs-741feed754ed4827a429135e5fcc0e7623f1f847d228698b066da094a8819e32:latest`,
image digest `sha256:503e36a6615e772db4fe47adbe6f1e1dad0f417f6bb1d1857c1f9ab6f2001615`.

`evidence.tar.gz` preserves stdout/stderr and generated outputs:

- 20 explicit integration-test launches (60 rank-local passes): 2/4 ranks,
  shared/disjoint masks with 1/2 CPUs requested, partially overlapping masks
  with 1 CPU requested, each with/without an initial invalid request on the
  final rank. Tests verify assignment, inherited worker affinity, node-local
  nonoverlap, restoration, and recovery after collective planning failure.
- 12 CLI rejection launches (36 rank-local exit statuses): invalid value,
  inconsistent values, insufficient CPUs, excessive worker budget, missing
  budget, and rank-asymmetric invalid budget. No input is read or output
  directory created. Expected statuses are 1 for inconsistent controls and
  2 for the other errors. Launcher calls have explicit timeouts.
- 12 successful CLI launches: 2/4 ranks × Opt/PhysCal × placement unset/1/2,
  using the same input and seed. Inner threads=1, requested Engine threads=4,
  Engine worker budget=0, BLAS/OMP=1, SR=tenferro. Opt uses 2 steps/1 sample;
  PhysCal uses the fixed input parameter file. `positive-summary.json` compares
  parsed finite numeric tokens in 4 Opt / 80 PhysCal files per rank count,
  excluding timing files. Maximum absolute difference is 0 for both placement
  values relative to unset. This comparison does not independently establish
  RNG draw parity or performance.

`parallel-evidence.tar.gz` adds 16 successful CLI launches: 2/4 ranks ×
Opt/PhysCal × Engine/Rayon configuration × placement unset/2. Engine mode sets
inner=1, Engine participants=2 and additional worker budget=1; Rayon mode sets
inner=2, Engine participants=1 and additional worker budget=0. Rayon capacity
is 2 and BLAS/OMP=1 in both. Other model/seed settings match the preceding
CLI checks. Every mode's finite output-token maximum absolute difference is 0
between placement unset and 2; see `parallel-summary.json`. These settings
permit parallel execution but do not measure actual kernel participation.
Reanalyse extracted results using `uv run --no-project python
compare-parallel.py EXTRACTED_ROOT/mvmc-583-runtime-parallel-positive`.
`local-parallel-replay.sh` records the exact local launches.

The `local-*-replay.sh` scripts and `wrapper.sh` preserve the actual local
launch commands, including absolute `/tmp` paths and executable hashes in
Cargo filenames. They are evidence, not portable entry points: regenerate
the test executable using `cargo nextest run -p mvmc-core --features mpi
--cargo-profile test-fast --test mpi_cpu_placement_583 --no-run`, and substitute
your target path, image, input directory and output directory. For normal
application commands see the experimental Linux MPI placement subsection in
the English/Japanese accelerated-backends manual.

These runs use one physical host. The integration test gathers affinity within
each shared-memory communicator so CPU ids on different nodes may overlap;
multi-node execution was not exercised. Application-failure rollback after an
OS affinity syscall failure was not fault-injected. Existing MPI/BLAS threads,
cgroup quotas and SMT topology remain outside this option's scope.

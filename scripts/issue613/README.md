# MPI preflight check for unsupported measurement Pfaffian (#613)

This optional Linux/dev-container check runs the real MPI-enabled CLI with 2 and 4 MPICH ranks. It covers Opt and PhysCal, both uniform `tenferro` selection and rank-inconsistent `tenferro`/`calc-m-all` selection. It verifies each rank's CLI exit status and diagnostic, absence of panic and of the requested output directory, and a timeout to catch hangs. PhysCal deliberately uses a nonexistent parameter file: the backend preflight must happen before input or output I/O. For inconsistent settings the existing CLI control agreement rejects the launch before backend preflight.

```sh
# In the dev container; use an MPI-only target to avoid a default-feature build
# overwriting the same executable while this gate runs.
CARGO_TARGET_DIR=/target/mpi-preflight cargo build -p mvmc-cli --features mpi --profile test-fast --locked
scripts/issue613/mpi-preflight.sh /target/mpi-preflight/test-fast/mvmc
```

The default launcher is `/opt/mpich/bin/mpirun`; set `MVMC_MPI_LAUNCHER` to an MPICH-compatible launcher if needed. The script requires `rg` and `timeout`. A small test-harness file barrier runs **after** each CLI process exits, ensuring every wrapper records its rank-local status before MPICH may terminate sibling wrappers. The script verifies both the launcher status and every rank-local status and log. It also repeats direct, wrapper-free 2- and 4-rank unsupported-selection launches five times each to catch false-success launcher exits. Temporary files are removed on exit and dumped on a validation failure. This gate does not implement the unsupported Pfaffian operation, cover CUDA providers, or replace numerical tests.

Before MPI finalization was added on collectively agreed preflight errors, a direct MPICH check returned launcher status 0 in 3/10 two-rank and 2/10 four-rank trials despite the per-rank unsupported diagnostics. After the change, the same direct check returned status 2 in all 20 trials on the development container. This observation is MPICH/environment-specific; the repeated direct checks above remain part of the gate so a launcher-level false success is not hidden by the rank wrappers.

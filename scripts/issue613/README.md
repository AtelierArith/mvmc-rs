# MPI check of the measurement Pfaffian selector (#613)

This optional Linux/dev-container check runs the real MPI-enabled CLI with 2 and 4 MPICH ranks over Opt and PhysCal for three selections:

- **uniform** `MVMC_RS_MEASURE_PF_BACKEND=tenferro` (supported since #613): every rank must exit 0 without an error line, and the output directory must contain `zvo_out*.dat`. PhysCal uses the real Hubbard chain `tests/fixtures/native_c_physcal_181/hubbard_chain_real` on its fixed parameters.
- **invalid** `bogus` on every rank: every rank must report `error: MVMC_RS_MEASURE_PF_BACKEND="bogus" is not one of ...` with status 2 and create no output. PhysCal deliberately uses a nonexistent parameter file: the selector preflight must happen before input or output I/O.
- **mixed** `tenferro` on rank 0 and `calc-m-all` elsewhere: the existing CLI control agreement rejects the launch (status 1) before backend preflight.

```sh
# In the dev container; use an MPI-only target to avoid a default-feature build
# overwriting the same executable while this gate runs.
CARGO_TARGET_DIR=/target/mpi-preflight cargo build -p mvmc-cli --features mpi --profile test-fast --locked
scripts/issue613/mpi-preflight.sh /target/mpi-preflight/test-fast/mvmc
```

The default launcher is `/opt/mpich/bin/mpirun`; set `MVMC_MPI_LAUNCHER` to an MPICH-compatible launcher if needed. The script requires `rg` and `timeout`. A small test-harness file barrier runs **after** each CLI process exits, ensuring every wrapper records its rank-local status before MPICH may terminate sibling wrappers. The script verifies both the launcher status and every rank-local status and log, absence of panics, and a timeout to catch hangs. It also repeats direct, wrapper-free 2- and 4-rank rejected launches five times each to catch false-success launcher exits: alternately an invalid selector (preflight) and a complex input (`heisenberg_chain_cmp`) with `tenferro`, which the CLI rejects after the collective mode agreement and before any output. Temporary files are removed on exit and dumped on a validation failure. This gate checks status and lifecycle only; the numerical comparison of the tenferro measurement Pfaffian against native C and the C-order path lives in the Rust tests (`issue181_native_c_physcal`, `issue464_backend_selector`).

Before MPI finalization was added on collectively agreed preflight errors (#614), a direct MPICH check returned launcher status 0 in 3/10 two-rank and 2/10 four-rank trials despite the per-rank diagnostics. After the change, the same direct check returned status 2 in all 20 trials on the development container. This observation is MPICH/environment-specific; the repeated direct checks above remain part of the gate so a launcher-level false success is not hidden by the rank wrappers.

# MPI rank affinity diagnostic (#583)

The `mpi_affinity_audit` example reports the Linux CPU affinity of each MPI rank's calling thread and the number of CPUs shared by each pair on the same shared-memory node. It is read-only and does not assign threads or change any budgets. It uses `MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)` via rsmpi and `/proc/self/status`; unavailable or malformed affinity is printed as `error:` and the corresponding overlap as `unavailable`. On other platforms it reports affinity as unavailable.

Affinity is an allowed CPU mask, not evidence of simultaneous CPU use or placement of workers that later change their own masks. It does not inspect cgroup CPU quotas, SMT topology, BLAS, other processes, MPI implementation threads or multiple thread pools. Pairwise overlap is not an automatic allocation algorithm: three or more masks can overlap in ways that pairwise counts alone cannot resolve. Do not interpret the result as proof of bounded execution.

## Reproduce

Build in the Linux development container with its MPI runtime and separate target volume. On a system where CPUs 0–3 are in the process's allowed set:

```sh
cargo nextest run -p mvmc-core --features mpi --example mpi_affinity_audit \
  --cargo-profile test-fast --locked
cargo build -p mvmc-core --features mpi --example mpi_affinity_audit \
  --profile test-fast --locked
# Four ranks share the same four-CPU affinity:
export AUDIT_BIN="${CARGO_TARGET_DIR:-target}/test-fast/examples/mpi_affinity_audit"
taskset -c 0-3 mpirun -n 4 "$AUDIT_BIN"
# MPICH Hydra supplies PMI_RANK; pin rank 0 to CPU 0, rank 1 to CPU 1, etc.:
mpirun -n 4 bash -c 'taskset -c "$PMI_RANK" "$AUDIT_BIN"'
```

Use an MPI launcher and library from the same installation. Choose CPUs permitted by the scheduler/container, and substitute the launcher's local-rank variable if it does not expose `PMI_RANK`. On a multi-node allocation CPU numbering is node-local; only pairs within one shared-memory communicator are compared.

## Verification

On Linux x86_64 with MPICH 4.2.0 in the development-container image `sha256:503e36a6615e772db4fe47adbe6f1e1dad0f417f6bb1d1857c1f9ab6f2001615`, four ranks sharing affinity `0-3` each reported `shared_cpus=4` for all six pairs. With rank `r` pinned to CPU `r` for `r=0,1,2,3`, all six pairs reported `shared_cpus=0`. These checks validate the diagnostic for these placements; they do not benchmark performance or validate multi-node behavior.

A third check pinned rank `r` to CPUs `r,r+1`; adjacent pairs reported one common CPU and nonadjacent pairs reported zero. Example output for all three cases is in [results/affinity-audit.txt](results/affinity-audit.txt). On MPICH Hydra this third placement can be reproduced with:

```sh
mpirun -n 4 bash -c 'next=$((PMI_RANK + 1)); taskset -c "$PMI_RANK-$next" "$AUDIT_BIN"'
```

For this last example, CPUs 0–4 must all be permitted.

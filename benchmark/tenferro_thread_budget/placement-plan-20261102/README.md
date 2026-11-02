# Node-local CPU placement feasibility (#583)

`mpi_affinity_audit --plan=N` prints a deterministic, disjoint N-CPU candidate mask for every rank on each shared-memory node, restricted to each rank's observed Linux affinity. An iterative augmenting-path matching algorithm handles constrained masks by reassigning flexible ranks. It rejects impossible requests even when the union alone appears large enough. Its complexity depends on the number of requested CPU slots and allowed CPUs; it is a setup diagnostic, not an inner-kernel operation.

This reports feasibility only. It does not apply affinity, resize pools, inspect cgroup quotas, distinguish physical cores from SMT siblings, count other threads, or validate a whole-application budget. Plan errors are printed diagnostics, not a nonzero launcher status. Non-Linux affinity remains unavailable. Every node leader prints its own plan; `N` is the node leader's command-line argument and the developer command assumes matching arguments on all ranks. This does not complete #583/#585's runtime ownership policy.

The planner's five unit tests include independent exhaustive enumeration of 4096 three-CPU/three-rank and 2304 four-CPU/two-rank mask/request cases, plus constrained reassignment, disjoint capacity and overflow cases. Every successful result is constrained to allowed CPUs; the three-rank exhaustive check also independently asserts exact counts and global non-overlap. The MPI example's seven tests (including the existing affinity parser checks) passed through nextest, and targeted Clippy with warnings denied passed.

Live MPICH checks used four ranks in the Linux dev container, with masks imposed separately by `taskset` before the diagnostic:

| Rank masks | Request/rank | Result |
|---|---:|---|
| All 0–3 | 1 | Four disjoint single-CPU masks |
| All 0–3 | 2 | Insufficient union capacity |
| 0–1, 2–3, 4–5, 6–7 | 1 | One allowed CPU per rank |
| 0–1, 2–3, 4–5, 6–7 | 2 | Original disjoint masks retained |
| 0–1, 1–2, 2–3, 3–4 | 1 | Four disjoint allowed CPUs |
| 0–1, 1–2, 2–3, 3–4 | 2 | Insufficient union capacity |

All six launches exited successfully, as expected for this report-only tool. A separate parser verified four observed rank masks, exact requested count/subset/non-overlap for feasible plans, and an error with no partial plan for infeasible cases. Logs are retained alongside the exact `run.sh`. No speed or numerical-kernel claim is made.

Build in the Linux development container with its MPI provider:

```sh
cargo build -p mvmc-core --features mpi --example mpi_affinity_audit --profile test-fast --locked
cargo nextest run -p mvmc-core --features mpi --example mpi_affinity_audit --no-fail-fast --retries 0
cargo clippy -p mvmc-core --features mpi --example mpi_affinity_audit -- -D warnings
mpirun -n 4 target/test-fast/examples/mpi_affinity_audit --plan=1
```

`run.sh` records the actual image, target volume, masks and six timeout-bounded launches; adjust those local paths and use a fresh output directory for another host. Image `sha256:503e36a6615e772db4fe47adbe6f1e1dad0f417f6bb1d1857c1f9ab6f2001615`, Linux x86_64/Xeon E5-2699 v3, Rust 1.99.0, MPICH 4.2.0. This is logical CPU allocation evidence; no multi-node or quota-enforced test is claimed.

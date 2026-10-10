# C / Julia / Rust MPI benchmark

Periodic half-filled Hubbard chain: t=1, U=4, Lsub=4, NSPGaussLeg=8.
MPI ranks: 16; compute threads/rank: 1; BLAS threads: 1.
Saved configurations: 320 total per step/group (20/rank).
Opt: 300 SR steps; PhysCal: 100 measurement groups.
Warmups: 1; measured repetitions: 3; times are medians in seconds.
PhysCal uses the same C-generated parameter file for all implementations.

Rust measurement path: calc-m-all (production default C-order kernels).

| Workload | Sites | C (s) | Julia (s) | Rust (s) | Julia/C | Rust/C |
|---|---:|---:|---:|---:|---:|---:|
| Opt | 32 | 3.501040 | 3.715684 | 3.533053 | 1.061 | 1.009 |
| Opt | 64 | 17.490420 | 16.673717 | 16.605140 | 0.953 | 0.949 |
| PhysCal | 32 | 1.203280 | 1.468175 | 1.081655 | 1.220 | 0.899 |
| PhysCal | 64 | 6.690050 | 6.366676 | 5.095055 | 0.952 | 0.762 |

C: fresh-process rank-zero internal All timer, after MPI_Init.
Julia/Rust: warmed production API maximum over ranks, including parsing,
initialization/parameter loading, sampling, optimization or observables, and output.
Process startup, JIT, build/setup and warmups are excluded. Timing boundaries differ.
Raw measurements, outputs, input hashes, source hashes, versions and linked libraries
are retained beside this report. Timing runs are sequential. Energies are not parity proof.

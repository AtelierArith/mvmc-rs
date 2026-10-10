# C / Julia / Rust MPI benchmark

Periodic half-filled Hubbard chain: t=1, U=4, Lsub=4, NSPGaussLeg=8.
MPI ranks: 4; compute threads/rank: 4; BLAS threads: 1.
Saved configurations: 300 total per step/group (75/rank).
Opt: 300 SR steps; PhysCal: 100 measurement groups.
Warmups: 1; measured repetitions: 3; times are medians in seconds.
PhysCal uses the same C-generated parameter file for all implementations.

Rust measurement path: calc-m-all (production default C-order kernels).

| Workload | Sites | C (s) | Julia (s) | Rust (s) | Julia/C | Rust/C |
|---|---:|---:|---:|---:|---:|---:|
| Opt | 32 | 3.353480 | 3.589051 | 3.071678 | 1.070 | 0.916 |
| Opt | 64 | 13.505130 | 13.400877 | 12.895514 | 0.992 | 0.955 |
| PhysCal | 32 | 1.432130 | 1.402758 | 1.345578 | 0.979 | 0.940 |
| PhysCal | 64 | 6.563780 | 5.034615 | 5.500781 | 0.767 | 0.838 |

C: fresh-process rank-zero internal All timer, after MPI_Init.
Julia/Rust: warmed production API maximum over ranks, including parsing,
initialization/parameter loading, sampling, optimization or observables, and output.
Process startup, JIT, build/setup and warmups are excluded. Timing boundaries differ.
Raw measurements, outputs, input hashes, source hashes, versions and linked libraries
are retained beside this report. Timing runs are sequential. Energies are not parity proof.

# Native MPI world observer (#495)

`world_observer.c` is a new optional MPI profiling-interface shim, not extracted
numerical code. It delegates MPI_Init to PMPI_Init and reports each actual
rank/world size and OpenMP/OpenBLAS thread count. It performs no collective,
RNG access, or arithmetic mutation. The authoritative unmodified upstream
`src/mVMC/vmcmain.c:74-78` calls MPI_Init before its primary StartTimer(0),
so these observations are outside the measured C time.

```sh
/opt/mpich/bin/mpicc -std=c11 -O2 -Wall -Wextra -Werror -shared -fPIC \
  -fopenmp c_toolbox/mpi_comparison_495/world_observer.c -lopenblas \
  -o /tmp/mvmc-world-observer.so
```
Use only in explicit developer reference runs via LD_PRELOAD. Cargo builds,
Rust tests and the Rust runtime are independent of this shim. Keep the C
executable unmodified; record its build provenance, upstream commit, vmcmain.c
SHA-256, observer source/library hashes, compiler arguments and linked providers
with the measurements. `scripts/bench_c_mpi.py` validates every rank's evidence
and retains raw outputs, launcher wall time and rank-zero internal All timer.
The C timer is not claimed to be a separately measured maximum over ranks.
# Actual OpenMP participation

`omp_observer.c` is a separate, untimed diagnostic that interposes the
`GOMP_parallel@GOMP_4.0` call imported by the GCC 13 C executable. Its trampoline
records the OpenMP logical thread IDs that actually enter parallel teams, then
reports `CEXECUTION rank parallel_calls distinct_team_thread_ids` at finalize.
These instrumented runs must never contribute timings to the comparison.

```sh
/opt/mpich/bin/mpicc -std=c11 -O2 -Wall -Wextra -Werror -shared -fPIC -fopenmp \
  c_toolbox/mpi_comparison_495/omp_observer.c -ldl -o /tmp/omp_observer.so
# Run a copied benchmark input with 20 SR steps and the same sampling settings:
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
  LD_PRELOAD=/tmp/world_observer.so:/tmp/omp_observer.so \
  /opt/mpich/bin/mpiexec -n 4 /path/to/vmc.out namelist.def
```

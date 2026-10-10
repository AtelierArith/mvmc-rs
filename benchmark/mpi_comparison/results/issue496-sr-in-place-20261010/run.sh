#!/usr/bin/env bash
set -euo pipefail
cd /workspaces/mvmc-rs
export LD_LIBRARY_PATH=/opt/mpich/lib:${LD_LIBRARY_PATH:-}
export OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export MVMC_RS_INNER_THREADS=4 UCX_MEMTYPE_CACHE=no UCX_ERROR_SIGNALS=SIGILL,SIGBUS,SIGFPE
export MVMC_RS_MEASURE_PF_BACKEND=calc-m-all MVMC_RS_SR_PF_BACKEND=c-order
unset MVMC_C_TIMER MVMC_TIMER MVMC_MAINCAL_DIAG MVMC_WEIGHTAVG_DIAG LD_PRELOAD
base=/home/vscode/.cache/mvmc/issue496-sr-in-place-confirm
mkdir "$base"
for mode in baseline inplace; do
 if [[ "$mode" = inplace ]]; then export MVMC_RS_SR_IN_PLACE_PROBE=1; else unset MVMC_RS_SR_IN_PLACE_PROBE; fi
 /opt/mpich/bin/mpiexec -n 4 /home/vscode/.cache/mvmc/target/linux-x86_64/release/examples/mpi_benchmark /workspaces/mvmc-rs/bench-out/issue502-production-opt-physcal-20261010/L64/opt-inputs/namelist.def 300 1 3 "$base/$mode" 4 > "$base/$mode.log" 2>&1
 done

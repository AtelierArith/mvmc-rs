#!/usr/bin/env bash
set -euo pipefail
cd /workspaces/mvmc-rs
export LD_LIBRARY_PATH=/opt/mpich/lib:${LD_LIBRARY_PATH:-}
export OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export MVMC_RS_INNER_THREADS=4 UCX_MEMTYPE_CACHE=no UCX_ERROR_SIGNALS=SIGILL,SIGBUS,SIGFPE
export MVMC_RS_MEASURE_PF_BACKEND=calc-m-all MVMC_RS_SR_PF_BACKEND=c-order
unset MVMC_C_TIMER MVMC_TIMER MVMC_MAINCAL_DIAG MVMC_WEIGHTAVG_DIAG LD_PRELOAD MVMC_RS_SR_IN_PLACE_PROBE
base=/home/vscode/.cache/mvmc/issue496-direct-sr-final-rng
mkdir "$base"
for size in 32 64; do
 for steps in 20 300; do
   /opt/mpich/bin/mpiexec -n 4 /home/vscode/.cache/mvmc/target/issue496-direct-sr-final/release/examples/mpi_rng_audit "/workspaces/mvmc-rs/bench-out/issue502-production-opt-physcal-20261010/L$size/opt-inputs/namelist.def" "$steps" "$base/L$size-$steps-final" 4 > "$base/L$size-$steps-final.log" 2>&1
 done
 done

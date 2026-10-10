#!/usr/bin/env bash
set -euo pipefail
root=/workspaces/mvmc-rs/bench-out/issue496-thread-granularity-20261010
export OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export UCX_ERROR_SIGNALS=SIGILL,SIGBUS,SIGFPE UCX_MEMTYPE_CACHE=no
export MVMC_RS_SR_BACKEND=c-order MVMC_RS_MEASURE_PF_BACKEND=calc-m-all MVMC_RS_INNER_THREADS=1
unset MVMC_RS_INNER_MIN_WORK_NS MVMC_RS_INNER_THRESHOLD MVMC_RS_SPINDLE_BUDGET MVMC_RS_SPINDLE_SPIN
inputs=/workspaces/mvmc-rs/bench-out/rank-thread-sweep-20261010/r1-t1/L64/opt-inputs
binary=/home/vscode/.cache/mvmc/target/linux-x86_64/release/examples/mpi_benchmark
dest="$root/Rust-measurement-regression-opt-baseline-after-L64-r1-t1"
mkdir -p "$dest"
sha256sum "$binary" > "$dest/binary-sha256.txt"
sha256sum "$inputs/"*.def > "$dest/input-sha256.txt"
timeout -k 10s 600s /opt/mpich/bin/mpiexec -n 1 "$binary" "$inputs/namelist.def" 300 1 3 "$dest" 1 > "$dest/run.log" 2>&1
rg '^BENCH' "$dest/run.log"

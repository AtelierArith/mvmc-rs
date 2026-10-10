#!/usr/bin/env bash
set -euo pipefail
root=/workspaces/mvmc-rs/bench-out/issue496-thread-granularity-20261010
export OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export UCX_ERROR_SIGNALS=SIGILL,SIGBUS,SIGFPE UCX_MEMTYPE_CACHE=no
export MVMC_RS_SR_BACKEND=c-order MVMC_RS_MEASURE_PF_BACKEND=calc-m-all MVMC_RS_INNER_THREADS=16
unset MVMC_RS_INNER_MIN_WORK_NS MVMC_RS_INNER_THRESHOLD MVMC_RS_SPINDLE_BUDGET MVMC_RS_SPINDLE_SPIN
for size in 32 64; do
    for calculation in opt phys; do
        base="/workspaces/mvmc-rs/bench-out/rank-thread-sweep-20261010/r1-t16/L$size"
        dest="$root/Rust-measurement-final-$calculation-L$size-r1-t16"
        mkdir -p "$dest"
        if [[ "$calculation" == opt ]]; then
            inputs="$base/opt-inputs"
            binary=/home/vscode/.cache/mvmc/target/issue496-measurement-team/release/examples/mpi_benchmark
            args=("$inputs/namelist.def" 300 1 3 "$dest" 1)
        else
            inputs="$base/physcal-inputs"
            binary=/home/vscode/.cache/mvmc/target/issue496-measurement-team/release/examples/mpi_physcal_benchmark
            args=("$inputs/namelist.def" "$inputs/zqp_opt.dat" 100 1 3 "$dest" 1)
            sha256sum "$inputs/zqp_opt.dat" > "$dest/params-sha256.txt"
        fi
        sha256sum "$binary" > "$dest/binary-sha256.txt"
        sha256sum "$inputs/"*.def > "$dest/input-sha256.txt"
        timeout -k 10s 600s /opt/mpich/bin/mpiexec -n 1 "$binary" "${args[@]}" > "$dest/run.log" 2>&1
        printf '%s\n' "$dest"
        rg '^BENCH' "$dest/run.log"
    done
done

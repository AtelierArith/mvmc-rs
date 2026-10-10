#!/usr/bin/env bash
set -euo pipefail
root=/workspaces/mvmc-rs/bench-out/issue496-thread-granularity-20261010
export OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export UCX_ERROR_SIGNALS=SIGILL,SIGBUS,SIGFPE UCX_MEMTYPE_CACHE=no
export MVMC_RS_SR_BACKEND=c-order MVMC_RS_MEASURE_PF_BACKEND=calc-m-all
unset MVMC_RS_INNER_MIN_WORK_NS MVMC_RS_INNER_THRESHOLD MVMC_RS_SPINDLE_BUDGET MVMC_RS_SPINDLE_SPIN
for layout in 1:1 2:8 4:4 8:2 16:1; do
    ranks=${layout%:*}; threads=${layout#*:}; export MVMC_RS_INNER_THREADS=$threads
    for size in 32 64; do
        for calculation in opt phys; do
            for version in baseline measurement; do
                case "$version" in baseline) target=linux-x86_64;; measurement) target=issue496-measurement-team;; esac
                base="/workspaces/mvmc-rs/bench-out/rank-thread-sweep-20261010/r$ranks-t$threads/L$size"
                dest="$root/Rust-measurement-regression-$calculation-$version-L$size-r$ranks-t$threads"
                mkdir -p "$dest"
                if [[ "$calculation" == opt ]]; then
                    inputs="$base/opt-inputs"
                    binary="/home/vscode/.cache/mvmc/target/$target/release/examples/mpi_benchmark"
                    args=("$inputs/namelist.def" 300 1 3 "$dest" "$ranks")
                else
                    inputs="$base/physcal-inputs"
                    binary="/home/vscode/.cache/mvmc/target/$target/release/examples/mpi_physcal_benchmark"
                    args=("$inputs/namelist.def" "$inputs/zqp_opt.dat" 100 1 3 "$dest" "$ranks")
                    sha256sum "$inputs/zqp_opt.dat" > "$dest/params-sha256.txt"
                fi
                sha256sum "$binary" > "$dest/binary-sha256.txt"
                sha256sum "$inputs/"*.def > "$dest/input-sha256.txt"
                timeout -k 10s 600s /opt/mpich/bin/mpiexec -n "$ranks" "$binary" "${args[@]}" > "$dest/run.log" 2>&1
                printf '%s\n' "$dest"
                rg '^BENCH' "$dest/run.log"
            done
        done
    done
done

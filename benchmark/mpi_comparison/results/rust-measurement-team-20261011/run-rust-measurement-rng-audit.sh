#!/usr/bin/env bash
set -euo pipefail
root=/workspaces/mvmc-rs/bench-out/issue496-thread-granularity-20261010
export OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export UCX_ERROR_SIGNALS=SIGILL,SIGBUS,SIGFPE UCX_MEMTYPE_CACHE=no
export MVMC_RS_SR_BACKEND=c-order MVMC_RS_MEASURE_PF_BACKEND=calc-m-all
unset MVMC_RS_INNER_MIN_WORK_NS MVMC_RS_INNER_THRESHOLD MVMC_RS_SPINDLE_BUDGET MVMC_RS_SPINDLE_SPIN
for layout in 1:16 4:4; do
    ranks=${layout%:*}; threads=${layout#*:}; export MVMC_RS_INNER_THREADS=$threads
    for size in 32 64; do
        inputs="/workspaces/mvmc-rs/bench-out/rank-thread-sweep-20261010/r$ranks-t$threads/L$size/opt-inputs"
        for steps in 20 300; do
            for version in ${AUDIT_VERSIONS:-baseline measurement}; do
                case "$version" in baseline) target=issue496-audit-baseline;; measurement) target=issue496-measurement-team;; esac
                binary="/home/vscode/.cache/mvmc/target/$target/release/examples/mpi_rng_audit"
                dest="$root/Rust-measurement-rng-$version-L$size-r$ranks-t$threads-s$steps"
                mkdir -p "$dest"
                sha256sum "$binary" > "$dest/binary-sha256.txt"
                sha256sum "$inputs/"*.def > "$dest/input-sha256.txt"
                timeout -k 10s 600s /opt/mpich/bin/mpiexec -n "$ranks" "$binary" "$inputs/namelist.def" "$steps" "$dest" "$ranks" > "$dest/run.log" 2>&1
                printf '%s completed\n' "$dest"
            done
        done
    done
done

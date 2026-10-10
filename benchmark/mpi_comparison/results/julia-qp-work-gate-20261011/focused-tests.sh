#!/usr/bin/env bash
set -euo pipefail
root=/workspaces/mvmc-rs/bench-out/issue496-julia-qp-work-gate-20261011
cache=/home/vscode/.cache/mvmc
export OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export JULIA_NUM_GC_THREADS=1 JULIA_MVMC_PFAPACK_THREADS=0 JULIA_MVMC_PFAPACK_LOCK=1
export UCX_ERROR_SIGNALS=SIGILL,SIGBUS,SIGFPE UCX_MEMTYPE_CACHE=no
for threads in 1 4; do
    JULIA_NUM_THREADS=$threads,0 "$cache/tools/julia-1.13.1/bin/julia" \
        --project="$cache/julia-issue496-real-qp-budget-project" --startup-file=no \
        "$cache/julia-issue496-real-qp-budget/MVMCOptimizers.jl/test_unit/test_unit_real_qp_threading.jl" \
        > "$root/real-qp-tests-t$threads.log" 2>&1
    printf 'PASS focused tests with %s threads\n' "$threads"
done

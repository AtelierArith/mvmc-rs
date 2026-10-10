#!/usr/bin/env bash
set -euo pipefail
root=/workspaces/mvmc-rs/bench-out/issue496-julia-qp-work-gate-20261011
cache=/home/vscode/.cache/mvmc
export OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export JULIA_NUM_THREADS=16,0 JULIA_NUM_GC_THREADS=1 JULIA_MVMC_MPI=1
export JULIA_MVMC_INNER_THREADS=1 JULIA_MVMC_PFAPACK_THREADS=0 JULIA_MVMC_PFAPACK_LOCK=1
export UCX_ERROR_SIGNALS=SIGILL,SIGBUS,SIGFPE UCX_MEMTYPE_CACHE=no MVMC_RNG_AUDIT_COUNT=1
for size in 32 64; do
    inputs=/workspaces/mvmc-rs/bench-out/rank-thread-sweep-20261010/r1-t16/L$size/opt-inputs
    for version in baseline candidate; do
        if [ "$version" = baseline ]; then
            project="$cache/julia-issue496-thread-budget-project"
        else
            project="$cache/julia-issue496-real-qp-budget-project"
        fi
        dest="$root/audit/$version-L$size-r1-t16-s20"
        mkdir -p "$dest"
        MVMC_STATE_AUDIT_DIRECTORY="$dest" timeout -k 10s 100s /opt/mpich/bin/mpiexec -n 1 \
            "$cache/tools/julia-1.13.1/bin/julia" --project="$project" --startup-file=no \
            "$cache/issue496-julia-state-audit.jl" "$inputs/namelist.def" "$dest" 20 \
            > "$dest/run.log" 2>&1
    done
    for kind in rng state; do
        cmp "$root/audit/baseline-L$size-r1-t16-s20/$kind-rank-0.txt" \
            "$root/audit/candidate-L$size-r1-t16-s20/$kind-rank-0.txt"
    done
    printf 'EXACT Julia real QP L%s rank1 threads16 steps20 RNG/control\n' "$size"
done

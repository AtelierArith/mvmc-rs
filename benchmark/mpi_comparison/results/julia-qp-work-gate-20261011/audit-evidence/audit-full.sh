#!/usr/bin/env bash
set -euo pipefail
root=/workspaces/mvmc-rs/bench-out/issue496-julia-qp-work-gate-20261011
cache=/home/vscode/.cache/mvmc
export OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export JULIA_NUM_GC_THREADS=1 JULIA_MVMC_MPI=1 JULIA_MVMC_INNER_THREADS=1
export JULIA_MVMC_PFAPACK_THREADS=0 JULIA_MVMC_PFAPACK_LOCK=1
export UCX_ERROR_SIGNALS=SIGILL,SIGBUS,SIGFPE UCX_MEMTYPE_CACHE=no MVMC_RNG_AUDIT_COUNT=1
for ranks in 1 4; do
    threads=$((16 / ranks))
    export JULIA_NUM_THREADS=$threads,0
    for size in 32 64; do
        inputs=/workspaces/mvmc-rs/bench-out/rank-thread-sweep-20261010/r$ranks-t$threads/L$size/opt-inputs
        for version in baseline candidate; do
            if [ "$version" = baseline ]; then
                project=$cache/julia-issue496-thread-budget-project
            else
                project=$cache/julia-issue496-real-qp-budget-project
            fi
            dest=$root/audit/$version-L$size-r$ranks-t$threads-s300
            mkdir "$dest"
            MVMC_STATE_AUDIT_DIRECTORY="$dest" timeout -k 10s 600s /opt/mpich/bin/mpiexec -n "$ranks" \
                "$cache/tools/julia-1.13.1/bin/julia" --project="$project" --startup-file=no \
                "$cache/issue496-julia-state-audit.jl" "$inputs/namelist.def" "$dest" 300 \
                > "$dest/run.log" 2>&1
        done
        for ((rank=0; rank<ranks; rank++)); do
            for kind in rng state; do
                cmp "$root/audit/baseline-L$size-r$ranks-t$threads-s300/$kind-rank-$rank.txt" \
                    "$root/audit/candidate-L$size-r$ranks-t$threads-s300/$kind-rank-$rank.txt"
            done
        done
        printf 'EXACT Julia real QP L%s ranks%s threads%s steps300 RNG/control\n' "$size" "$ranks" "$threads"
    done
done

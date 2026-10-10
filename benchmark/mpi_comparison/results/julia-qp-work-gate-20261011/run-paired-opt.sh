#!/usr/bin/env bash
set -euo pipefail
# Execute inside the Linux reference container, with exclusive benchmark CPU use.
root=/workspaces/mvmc-rs/bench-out/issue496-julia-qp-work-gate-20261011
cache=/home/vscode/.cache/mvmc
export OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export JULIA_NUM_GC_THREADS=1 JULIA_MVMC_MPI=1 JULIA_MVMC_INNER_THREADS=1
export JULIA_MVMC_PFAPACK_THREADS=0 JULIA_MVMC_PFAPACK_LOCK=1
export UCX_ERROR_SIGNALS=SIGILL,SIGBUS,SIGFPE UCX_MEMTYPE_CACHE=no
for ranks in "${@:-1}"; do
    threads=$((16 / ranks))
    export JULIA_NUM_THREADS=$threads,0
    for size in 32 64; do
        inputs=/workspaces/mvmc-rs/bench-out/rank-thread-sweep-20261010/r$ranks-t$threads/L$size/opt-inputs
        dest=$root/paired-L$size-r$ranks-t$threads-s300
        mkdir "$dest"
        git -C "$cache/julia-issue496-real-qp-budget" rev-parse HEAD > "$dest/source-head.txt"
        sha256sum "$cache/julia-issue496-real-qp-budget/MVMCOptimizers.jl/src/calculate_m_all.jl" > "$dest/source-kernel.sha256"
        git -C "$cache/julia-issue496-real-qp-budget" diff > "$dest/source.diff"
        sha256sum "$root/paired-julia.jl" "$inputs"/*.def > "$dest/input-sha256.txt"
        /opt/mpich/bin/mpiexec -n "$ranks" "$cache/tools/julia-1.13.1/bin/julia" \
            --project="$cache/julia-issue496-real-qp-budget-project" --startup-file=no \
            "$root/paired-julia.jl" "$inputs/namelist.def" 300 3 "$dest" "$ranks" \
            > "$dest/run.log" 2>&1
        printf 'COMPLETE Opt L%s ranks%s threads%s steps300\n' "$size" "$ranks" "$threads"
    done
done

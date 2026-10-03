#!/usr/bin/env bash
# Run after the CLI matrix; require explicitly configured independent Julia.
set -euo pipefail
cd "$(dirname "$0")/.."
matrix=${1:?usage: script <CLI-evidence-directory>}
: "${MPI179_JULIA:?set MPI179_JULIA to Julia 1.13.1 executable}"
export CARGO_TARGET_DIR=/tmp/mvmc-issue179-target LIBCLANG_PATH=/usr/lib/llvm-18/lib
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1 JULIA_MVMC_MPI=1
limit=${MPI179_TIMEOUT:-60}
timeout --kill-after=10s 600s cargo test --locked --profile test-fast -p mvmc-core --features mpi --test mpi_issue179_state --test mpi_issue179_collectives --no-run --message-format=json > "$matrix/state-build.json" 2> "$matrix/state-build.log"
state_bin=$(jq -r 'select(.reason=="compiler-artifact" and .target.name=="mpi_issue179_state" and .executable!=null) | .executable' "$matrix/state-build.json")
collective_bin=$(jq -r 'select(.reason=="compiler-artifact" and .target.name=="mpi_issue179_collectives" and .executable!=null) | .executable' "$matrix/state-build.json")
[[ -x $state_bin && -x $collective_bin ]]
bad=0
printf 'cell\trust_exit\tjulia_exit\tcomparison\n' > "$matrix/states.tsv"
for ranks in 2 4; do
    rc=0
    timeout --kill-after=5s "${limit}s" mpirun --oversubscribe -n "$ranks" "$collective_bin" --ignored --exact issue179_collectives --nocapture > "$matrix/collectives-$ranks.log" 2>&1 || rc=$?
    printf 'collectives-%s\t%s\tNA\tPROTOCOL_ONLY\n' "$ranks" "$rc" >> "$matrix/states.tsv"
    if (( rc != 0 )); then bad=1; fi
    for mode in real cmp fsz; do
        for split in 1 2; do
            id="r${ranks}-${mode}-s${split}-cg0-store1"
            dir="$matrix/$id"
            export MPI179_INPUT="$dir/inputs/namelist.def" MPI179_STATE_DIR="$dir/rust-state"
            rc=0 jrc=0 cmp_status=NOT_COMPARED
            timeout --kill-after=5s "${limit}s" mpirun --oversubscribe -n "$ranks" "$state_bin" --ignored --exact issue179_state --nocapture > "$dir/rust-state.log" 2>&1 || rc=$?
            timeout --kill-after=5s "${limit}s" "$MPI179_JULIA" --project=extern/Julia-mVMC scripts/verify_mpi_issue179_julia_launch.jl "$ranks" scripts/verify_mpi_issue179_state.jl "$MPI179_INPUT" "$dir/julia-state" > "$dir/julia-state.log" 2>&1 || jrc=$?
            if (( rc != 0 || jrc != 0 )); then bad=1; cmp_status=WORKER_FAILURE
            elif [[ -n ${MPI179_JUSTIFICATION:-} && -n ${MPI179_ATOL:-} && -n ${MPI179_RTOL:-} ]]; then
                if uv run --no-project python scripts/compare_mpi_issue179.py "$dir/rust-state" "$dir/julia-state" --ranks "$ranks" --atol "$MPI179_ATOL" --rtol "$MPI179_RTOL" --justification "$MPI179_JUSTIFICATION" > "$dir/comparison.log" 2>&1; then cmp_status=CAPTURED_STATE_AGREEMENT; else cmp_status=MISMATCH; bad=1; fi
            else
                # Exact-only inventory still detects trajectory drift. No computational
                # numerical gate is claimed before inspecting the first divergence.
                cmp_status=BOUNDS_REQUIRED; bad=1
                uv run --no-project python scripts/compare_mpi_issue179.py "$dir/rust-state" "$dir/julia-state" --ranks "$ranks" --atol 0 --rtol 0 --justification AGENTS.md > "$dir/first-divergence.log" 2>&1 || true
            fi
            printf '%s\t%s\t%s\t%s\n' "$id" "$rc" "$jrc" "$cmp_status" | tee -a "$matrix/states.tsv"
        done
    done
    # One-rank callback error must terminate all ranks before another step.
    export MPI179_INPUT="$matrix/r${ranks}-real-s1-cg0-store1/inputs/namelist.def"
    export MPI179_STATE_DIR="$matrix/callback-failure-$ranks" MPI179_FAIL_RANK=1
    rc=0
    timeout --kill-after=5s "${limit}s" mpirun --oversubscribe -n "$ranks" "$state_bin" --ignored --exact issue179_state --nocapture > "$matrix/callback-failure-$ranks.log" 2>&1 || rc=$?
    unset MPI179_FAIL_RANK
    printf 'callback-failure-%s\t%s\tNA\tFAILURE_PROTOCOL\n' "$ranks" "$rc" >> "$matrix/states.tsv"
    if (( rc != 0 )); then bad=1; fi
done
exit "$bad"

#!/usr/bin/env bash
# Run after the CLI matrix; require explicitly configured independent Julia.
set -euo pipefail
cd "$(dirname "$0")/.."
matrix=${1:?usage: script <CLI-evidence-directory>}
inputs_matrix=$(realpath "$matrix")
source scripts/mpi_issue179_selection.sh
# The current full input inventory has 55 expected-success cells, each with
# three prefixes and three worker settings. Capability changes require an
# explicit revised coverage declaration, not a silently smaller green matrix.
# For the width1 CG108 subset, declare EXPECTED_CELLS=12/EXPECTED_TOTAL=108
# alongside the explicit cell filter. Filtered results are not full #179.
if [[ ! ${MPI179_CELL_REGEX+x} ]]; then
    MPI179_EXPECTED_CELLS=${MPI179_EXPECTED_CELLS-55}
    MPI179_EXPECTED_TOTAL=${MPI179_EXPECTED_TOTAL-495}
fi
mpi179_validate_selection "$inputs_matrix/matrix.tsv" states
matrix=$(mktemp -d /tmp/mvmc-issue179-states.XXXXXX)
printf 'state evidence: %s\n' "$matrix"
git rev-parse HEAD > "$matrix/head.txt"
git ls-files -co --exclude-standard -z crates scripts Cargo.toml Cargo.lock .cargo | sort -zu | xargs -0 sha256sum > "$matrix/source-sha256.txt"
sha256sum "$matrix/source-sha256.txt" > "$matrix/source-manifest.sha256"
: "${MPI179_JULIA:?set MPI179_JULIA to Julia 1.13.1 executable}"
source scripts/mpi_issue179_environment.sh
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JULIA_NUM_THREADS=1 JULIA_MVMC_MPI=1
export MVMC_RS_INNER_THRESHOLD=1
limit=${MPI179_TIMEOUT:-60}
timeout --kill-after=10s 600s cargo test --locked --profile test-fast -p mvmc-core --features mpi --test mpi_issue179_state --test mpi_issue179_mapping --no-run --message-format=json > "$matrix/state-build.json" 2> "$matrix/state-build.log"
state_bin=$(jq -r 'select(.reason=="compiler-artifact" and .target.name=="mpi_issue179_state" and .executable!=null) | .executable' "$matrix/state-build.json")
collective_bin=$(jq -r 'select(.reason=="compiler-artifact" and .target.name=="mpi_issue179_mapping" and .executable!=null) | .executable' "$matrix/state-build.json")
[[ -x $state_bin && -x $collective_bin ]]
mpi179_require_test "$state_bin" issue179_state
mpi179_require_test "$collective_bin" issue179_group_width_endpoints
sha256sum "$state_bin" "$collective_bin" > "$matrix/executables.sha256"
unset MPI179_FAIL_RANK
bad=0
completed=0
printf 'cell\tworkers\trust_exit\tjulia_exit\tcomparison\n' > "$matrix/states.tsv"
for ranks in 2 4; do
    rc=0
    MPI179_EXPECT_RANKS=$ranks timeout --kill-after=5s "${limit}s" "${mpi179_mpirun[@]}" -n "$ranks" "$collective_bin" --ignored --exact issue179_group_width_endpoints --nocapture </dev/null > "$matrix/collectives-$ranks.log" 2>&1 || rc=$?
    printf 'collectives-%s\tNA\t%s\tNA\tPROTOCOL_ONLY\n' "$ranks" "$rc" >> "$matrix/states.tsv"
    if (( rc != 0 )); then bad=1; fi
done
# Every named expected-success CLI input, including CG/NStore=0, standard QP,
# OptTrans, uneven widths, and empty measurement ranks. Rejections are CLI gates.
while IFS=$'\t' read -r id ranks mode split cg store projection expected status exitcode; do
    [[ $expected == success ]] || continue
    [[ $id =~ ${MPI179_CELL_REGEX:-.*} ]] || continue
    mkdir -p "$matrix/$id"
    timeout --kill-after=5s 30s "$MPI179_JULIA" --compile=min -O0 --project=extern/Julia-mVMC scripts/mpi_issue179_c_flags.jl "$inputs_matrix/$id/inputs" "$matrix/$id/c-written-flags.txt"
    for steps in ${MPI179_STEPS_LIST:-1 2 3}; do
            export MPI179_STEPS=$steps
            dir="$matrix/$id/prefix-$steps"
            mkdir -p "$dir"
            export MPI179_INPUT="$inputs_matrix/$id/inputs/namelist.def"
            jrc=0
            sha256sum "$inputs_matrix/$id/inputs/"* > "$dir/input-sha256.txt"
            # Independent reference is immutable for this exact input/seed; reuse
            # it across worker counts, never substitute Rust worker-1 as Julia.
            timeout --kill-after=5s "${limit}s" "$MPI179_JULIA" --compiled-modules=existing --project=extern/Julia-mVMC scripts/verify_mpi_issue179_julia_launch.jl "$ranks" scripts/verify_mpi_issue179_state.jl "$MPI179_INPUT" "$dir/julia-state" </dev/null > "$dir/julia-state.log" 2>&1 || jrc=$?
            for workers in 1 2 4; do
            export MVMC_RS_INNER_THREADS=$workers MPI179_STATE_DIR="$dir/w$workers/rust-state"
            mkdir -p "$dir/w$workers"
            rc=0 cmp_status=NOT_COMPARED
            timeout --kill-after=5s "${limit}s" "${mpi179_mpirun[@]}" -n "$ranks" "$state_bin" --ignored --exact issue179_state --nocapture </dev/null > "$dir/w$workers/rust-state.log" 2>&1 || rc=$?
            if (( rc != 0 || jrc != 0 )); then bad=1; cmp_status=WORKER_FAILURE
            elif [[ -n ${MPI179_JUSTIFICATION:-} && -n ${MPI179_ATOL:-} && -n ${MPI179_RTOL:-} ]]; then
                if uv run --no-project python scripts/compare_mpi_issue179.py "$MPI179_STATE_DIR" "$dir/julia-state" --ranks "$ranks" --atol "$MPI179_ATOL" --rtol "$MPI179_RTOL" --justification "$MPI179_JUSTIFICATION" --c-flags "$matrix/$id/c-written-flags.txt" --root-output "$MPI179_STATE_DIR/zvo_out.dat" "$dir/julia-state/zvo_out.dat" --root-output "$MPI179_STATE_DIR/zvo_var.dat" "$dir/julia-state/zvo_var.dat" --root-output "$MPI179_STATE_DIR/zqp_opt.dat" "$dir/julia-state/zqp_opt.dat" > "$dir/w$workers/comparison.log" 2>&1; then cmp_status=CAPTURED_STATE_AGREEMENT; else cmp_status=MISMATCH; bad=1; fi
            else
                # Exact-only inventory still detects trajectory drift. No computational
                # numerical gate is claimed before inspecting the first divergence.
                cmp_status=BOUNDS_REQUIRED; bad=1
                if ! uv run --no-project python scripts/compare_mpi_issue179.py "$MPI179_STATE_DIR" "$dir/julia-state" --ranks "$ranks" --discrete-only --atol 0 --rtol 0 --justification AGENTS.md --c-flags "$matrix/$id/c-written-flags.txt" > "$dir/w$workers/first-divergence.log" 2>&1; then cmp_status=DISCRETE_MISMATCH; fi
            fi
            if (( rc == 0 && jrc == 0 )); then
                if ! uv run --no-project python scripts/mpi_issue179_worker_evidence.py "$MPI179_STATE_DIR" --ranks "$ranks" --workers "$workers" > "$dir/w$workers/worker-evidence.log" 2>&1; then
                    cmp_status=WORKER_EVIDENCE_FAILURE; bad=1
                fi
            fi
            printf '%s-prefix%s\t%s\t%s\t%s\t%s\n' "$id" "$steps" "$workers" "$rc" "$jrc" "$cmp_status" | tee -a "$matrix/states.tsv"
            completed=$((completed + 1))
            done
    done
done < "$inputs_matrix/matrix.tsv"
if (( completed != MPI179_SELECTED_CELLS * MPI179_SELECTED_PREFIXES * 3 )); then echo INCOMPLETE_REFERENCE_MATRIX >&2; bad=1; fi
for ranks in 2 4; do
    # One-rank callback error must terminate all ranks before another step.
    export MPI179_INPUT="$inputs_matrix/r${ranks}-real-s1-cg0-store1/inputs/namelist.def"
    export MPI179_STATE_DIR="$matrix/callback-failure-$ranks" MPI179_FAIL_RANK=1 MPI179_STEPS=3 MVMC_RS_INNER_THREADS=1
    rc=0 vrc=0
    timeout --kill-after=5s "${limit}s" "${mpi179_mpirun[@]}" -n "$ranks" "$state_bin" --ignored --exact issue179_state --nocapture </dev/null > "$matrix/callback-failure-$ranks.log" 2>&1 || rc=$?
    uv run --no-project python scripts/mpi_issue179_callback_evidence.py "$MPI179_STATE_DIR" --ranks "$ranks" --failure-rank 1 > "$matrix/callback-failure-$ranks-validation.log" 2>&1 || vrc=$?
    unset MPI179_FAIL_RANK
    printf 'callback-failure-%s\tNA\t%s\tNA\tFAILURE_PROTOCOL\n' "$ranks" "$rc" >> "$matrix/states.tsv"
    if (( rc != 0 || vrc != 0 )); then bad=1; fi
done
sha256sum -c "$matrix/executables.sha256" > "$matrix/executable-check.txt" || bad=1
git ls-files -co --exclude-standard -z crates scripts Cargo.toml Cargo.lock .cargo | sort -zu | xargs -0 sha256sum > "$matrix/source-sha256-after.txt"
if ! cmp -s "$matrix/source-sha256.txt" "$matrix/source-sha256-after.txt"; then
    printf 'SOURCE_CHANGED: shared worktree changed during validation\n' | tee "$matrix/source-changed.txt"
    bad=1
fi
exit "$bad"

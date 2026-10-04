#!/usr/bin/env bash
# MPI x inner-worker execution/state gate, BLAS=1. Not C/Julia full parity.
set -euo pipefail
cd "$(dirname "$0")/.."
matrix=$(realpath "${1:?usage: <CLI-evidence-directory>}")
source scripts/mpi_issue179_selection.sh
mpi179_validate_selection "$matrix/matrix.tsv" workers
source scripts/mpi_issue179_environment.sh
export MVMC_RS_INNER_THRESHOLD=1
out=$(mktemp -d /tmp/mvmc-issue179-workers.XXXXXX)
echo "Evidence: $out"
git -c safe.directory="$PWD" rev-parse HEAD > "$out/head.txt"
git -c safe.directory="$PWD" ls-files -co --exclude-standard -z -- crates scripts Cargo.toml Cargo.lock .cargo third_party | sort -zu | xargs -0 sha256sum > "$out/source-sha256.txt"
timeout --kill-after=10s 600s cargo build --locked --profile test-fast -p mvmc-cli --features mpi > "$out/cli-build.log" 2>&1
timeout --kill-after=10s 600s cargo test --locked --profile test-fast -p mvmc-core --features mpi --test mpi_issue179_state --no-run --message-format=json > "$out/state-build.json" 2> "$out/state-build.log"
state_bin=$(jq -r 'select(.reason=="compiler-artifact" and .target.name=="mpi_issue179_state" and .executable!=null) | .executable' "$out/state-build.json")
mpi179_require_test "$state_bin" issue179_state
cli_bin="$CARGO_TARGET_DIR/test-fast/mvmc"
sha256sum "$cli_bin" "$state_bin" > "$out/executables.sha256"
printf 'cell\tworkers\tcli_exit\tstate_exit\tcomparison\n' > "$out/matrix.tsv"
bad=0
completed=0
while IFS=$'\t' read -r id ranks mode split cg store projection expected status exitcode; do
    [[ $expected == success && ( $projection == identity || $projection == standard ) && $id != *uneven* && $id != *empty* ]] || continue
    for workers in 1 2 4; do
        export MVMC_RS_INNER_THREADS=$workers
        dir="$out/$id/w$workers"
        mkdir -p "$dir"
        input="$matrix/$id/inputs/namelist.def"
        sha256sum "$matrix/$id/inputs/"* > "$dir/input-sha256.txt"
        rc=0 src=0 comparison=NOT_COMPARED
        timeout --kill-after=5s "${MPI179_TIMEOUT:-30}s" "${mpi179_mpirun[@]}" -n "$ranks" "$cli_bin" "$input" --mode "$mode" --nsteps 1 --nsmp 1 --out-dir "$dir/cli-output" < /dev/null > "$dir/cli.log" 2>&1 || rc=$?
        MPI179_INPUT="$input" MPI179_STATE_DIR="$dir/state" timeout --kill-after=5s "${MPI179_TIMEOUT:-30}s" "${mpi179_mpirun[@]}" -n "$ranks" "$state_bin" --ignored --exact issue179_state --nocapture < /dev/null > "$dir/state.log" 2>&1 || src=$?
        if (( rc != 0 || src != 0 )) || [[ ! -s "$dir/cli-output/zvo_out.dat" ]]; then comparison=WORKER_FAILURE; bad=1
        else
            if ! uv run --no-project python scripts/mpi_issue179_worker_evidence.py "$dir/state" --ranks "$ranks" --workers "$workers" > "$dir/worker-evidence.log" 2>&1; then comparison=WORKER_EVIDENCE_FAILURE; bad=1; fi
            if [[ $comparison != WORKER_EVIDENCE_FAILURE ]]; then
                if ! uv run --no-project python scripts/compare_mpi_issue179.py "$out/$id/w1/state" "$dir/state" --ranks "$ranks" --discrete-only --atol 0 --rtol 0 --justification AGENTS.md > "$dir/discrete-comparison.log" 2>&1; then comparison=DISCRETE_MISMATCH; bad=1
                elif [[ -n ${MPI179_THREAD_ATOL:-} && -n ${MPI179_THREAD_RTOL:-} && -s ${MPI179_JUSTIFICATION:-/dev/null} ]]; then
                    if uv run --no-project python scripts/compare_mpi_issue179.py "$out/$id/w1/state" "$dir/state" --ranks "$ranks" --atol "$MPI179_THREAD_ATOL" --rtol "$MPI179_THREAD_RTOL" --justification "$MPI179_JUSTIFICATION" --root-output "$out/$id/w1/cli-output/zvo_out.dat" "$dir/cli-output/zvo_out.dat" > "$dir/numeric-comparison.log" 2>&1; then comparison=CAPTURED_WORKER_INVARIANCE; else comparison=NUMERIC_MISMATCH; bad=1; fi
                else comparison=DISCRETE_AGREEMENT_NUMERIC_BOUNDS_REQUIRED; bad=1
                fi
            fi
        fi
        printf '%s\t%s\t%s\t%s\t%s\n' "$id" "$workers" "$rc" "$src" "$comparison" | tee -a "$out/matrix.tsv"
        completed=$((completed + 1))
    done
done < "$matrix/matrix.tsv"
if (( completed != MPI179_SELECTED_CELLS * 3 )); then echo INCOMPLETE_WORKER_MATRIX >&2; bad=1; fi
sha256sum -c "$out/executables.sha256" > "$out/executable-check.txt" || bad=1
git -c safe.directory="$PWD" ls-files -co --exclude-standard -z -- crates scripts Cargo.toml Cargo.lock .cargo third_party | sort -zu | xargs -0 sha256sum > "$out/source-sha256-after.txt"
if ! cmp -s "$out/source-sha256.txt" "$out/source-sha256-after.txt"; then echo SOURCE_CHANGED > "$out/source-changed.txt"; bad=1; fi
exit "$bad"

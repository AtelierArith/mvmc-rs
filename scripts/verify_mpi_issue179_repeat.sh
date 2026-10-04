#!/usr/bin/env bash
# Fresh-process repeatability, not cross-language numerical parity.
set -euo pipefail
cd "$(dirname "$0")/.."
binary=$(realpath "${1:?usage: script frozen-state-binary input-inventory output-directory}")
self=$(realpath "$0")
inventory=$(realpath "${2:?input inventory required}")
output=${3:?new output directory required}
[[ ! -e $output && -x $binary ]]
source scripts/mpi_issue179_selection.sh
source scripts/mpi_issue179_environment.sh
mpi179_validate_selection "$inventory/matrix.tsv" states
mpi179_require_test "$binary" issue179_state
mkdir -p "$output"
output=$(realpath "$output")
sha256sum "$binary" "$self" scripts/mpi_issue179_worker_evidence.py scripts/mpi_issue179_repeat_evidence.py scripts/mpi_issue179_selection.sh scripts/mpi_issue179_environment.sh "$inventory/matrix.tsv" > "$output/executables-checkers.sha256"
mpirun --version > "$output/mpi-version.txt"
ldd "$binary" > "$output/binary-ldd.txt"
printf 'cell\tprefix\tworkers\tfirst_exit\tsecond_exit\tvalidation_exit\n' > "$output/results.tsv"
unset MPI179_FAIL_RANK
export MVMC_RS_INNER_THRESHOLD=1
printf '%s\n' "binary=$binary" "inventory=$inventory" "MPI179_STEPS_LIST=${MPI179_STEPS_LIST:-1 2 3}" "MPI179_CELL_REGEX=${MPI179_CELL_REGEX:-.*}" "MPI179_TIMEOUT=${MPI179_TIMEOUT:-60}" "MPI179_EXPECTED_CELLS=$MPI179_EXPECTED_CELLS" "MPI179_EXPECTED_TOTAL=$MPI179_EXPECTED_TOTAL" "MVMC_RS_INNER_THRESHOLD=$MVMC_RS_INNER_THRESHOLD" "workers=1 2 4" "repeats=2" "OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS" "OMP_NUM_THREADS=$OMP_NUM_THREADS" "MKL_NUM_THREADS=$MKL_NUM_THREADS" "BLIS_NUM_THREADS=$BLIS_NUM_THREADS" > "$output/configuration.txt"
bad=0 completed=0
while IFS=$'\t' read -r id ranks mode split cg store projection expected status exitcode; do
    [[ $expected == success && $id =~ ${MPI179_CELL_REGEX:-.*} ]] || continue
    export MPI179_INPUT="$inventory/$id/inputs/namelist.def"
    for steps in ${MPI179_STEPS_LIST:-1 2 3}; do
        export MPI179_STEPS=$steps
        for workers in 1 2 4; do
            export MVMC_RS_INNER_THREADS=$workers
            cell="$output/$id/prefix$steps/w$workers"
            mkdir -p "$cell"
            sha256sum "$inventory/$id/inputs/"* > "$cell/inputs.sha256"
            exits=()
            vrc=0
            for repeat in 1 2; do
                export MPI179_STATE_DIR="$cell/repeat$repeat"
                rc=0
                timeout --kill-after=5s "${MPI179_TIMEOUT:-60}s" "${mpi179_mpirun[@]}" -n "$ranks" "$binary" --ignored --exact issue179_state --nocapture </dev/null > "$cell/launch$repeat.log" 2>&1 || rc=$?
                exits+=("$rc")
                if (( rc == 0 )); then
                    uv run --no-project python scripts/mpi_issue179_worker_evidence.py "$MPI179_STATE_DIR" --ranks "$ranks" --workers "$workers" > "$cell/workers$repeat.log" 2>&1 || vrc=1
                else vrc=1; fi
            done
            # Scheduling IDs are checked above, not required to repeat.
            uv run --no-project python scripts/mpi_issue179_repeat_evidence.py "$cell/repeat1" "$cell/repeat2" --ranks "$ranks" --cg "$cg" --first-exit "${exits[0]}" --second-exit "${exits[1]}" > "$cell/repeat-validation.log" 2>&1 || vrc=1
            sha256sum -c "$cell/inputs.sha256" > "$cell/input-check.txt" || vrc=1
            printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$id" "$steps" "$workers" "${exits[0]}" "${exits[1]}" "$vrc" | tee -a "$output/results.tsv"
            if (( vrc != 0 )); then bad=1; fi
            completed=$((completed + 1))
        done
    done
done < "$inventory/matrix.tsv"
(( completed == MPI179_SELECTED_CELLS * MPI179_SELECTED_PREFIXES * 3 )) || bad=1
sha256sum -c "$output/executables-checkers.sha256" > "$output/executable-check.txt" || bad=1
printf 'REPEATABILITY cells=%s expected=%s bad=%s root=%s\n' "$completed" "$MPI179_EXPECTED_TOTAL" "$bad" "$output" | tee "$output/terminal-summary.txt"
exit "$bad"

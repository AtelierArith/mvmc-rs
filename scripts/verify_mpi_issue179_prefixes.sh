#!/usr/bin/env bash
# Review before launching. Direct/store0, workers1, prefixes1/2/3 only.
set -euo pipefail
cd "$(dirname "$0")/.."
binary=$(realpath "${1:?frozen state binary required}")
inventory=$(realpath "${2:?inventory required}")
output=${3:?new output directory required}
manifest=$(realpath "${4:?frozen source manifest required; paths relative to snapshot root}")
[[ ! -e $output && -x $binary && -s $manifest && -s $inventory/matrix.tsv ]]
[[ ${MPI179_SOURCE_COMMIT:?committed snapshot SHA required} =~ ^[0-9a-f]{40}$ ]]
[[ ${MPI179_TIMEOUT:-60} =~ ^[1-9][0-9]*$ ]]
proof=$(realpath "${5:?frozen build association JSON required}")
mkdir -p "$output"
output=$(realpath "$output")
phase=preflight
finish() {
    local rc=$?
    printf 'phase=%s exit=%s source_commit=%s\n' "$phase" "$rc" "$MPI179_SOURCE_COMMIT" > "$output/terminal-status.txt"
    return "$rc"
}
trap finish EXIT
[[ -s $proof && $(git rev-parse --show-toplevel) == "$PWD" && $(git rev-parse HEAD) == "$MPI179_SOURCE_COMMIT" ]]
git diff --exit-code "$MPI179_SOURCE_COMMIT" -- crates third_party xtask Cargo.toml Cargo.lock extern tests > "$output/production-baseline-diff.txt"
git diff --exit-code "$MPI179_SOURCE_COMMIT" -- scripts ':!scripts/mpi_issue179_prefix_evidence.py' ':!scripts/test_mpi_issue179_prefix_evidence.py' ':!scripts/verify_mpi_issue179_prefixes.sh' > "$output/nonowner-scripts-diff.txt"
cp "$manifest" "$output/sources-before.sha256"
sha256sum -c "$manifest" > "$output/source-before-check.txt"
uv run --no-project python scripts/mpi_issue179_prefix_evidence.py "$PWD" --build-proof "$proof" --binary "$binary" --manifest "$manifest" --commit "$MPI179_SOURCE_COMMIT" > "$output/build-association-check.txt" 2>&1
cp "$proof" "$output/build-association.json"
sha256sum -c baseline.sha256 > "$output/baseline-check.txt"
source scripts/mpi_issue179_environment.sh
source scripts/mpi_issue179_selection.sh
mpi179_require_test "$binary" issue179_state
listing=$("$binary" --list)
[[ $(printf '%s\n' "$listing" | rg -c '^issue179_state: test$') == 1 ]]
# Check the entire declared inventory before creating evidence or launching.
declare -A cells=() axes=()
while IFS=$'\t' read -r id ranks mode width cg store projection expected status exitcode extra; do
    [[ $id == cell || $id == id ]] && continue
    [[ $id =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ && ! ${cells[$id]+present} && -z $extra ]]
    cells[$id]=1
    [[ $expected == success && $cg == 0 && $store == 0 && $projection == identity &&
       ( $ranks == 2 || $ranks == 4 ) && ( $width == 1 || $width == 2 ) &&
       ( $mode == real || $mode == cmp || $mode == fsz ) && -n $status && -n $exitcode ]]
    axis="$ranks-$mode-$width"
    [[ ! ${axes[$axis]+present} && -s $inventory/$id/inputs/namelist.def ]]
    axes[$axis]=1
done < "$inventory/matrix.tsv"
(( ${#axes[@]} == 12 ))
# Preserve ALL input files before any launch, including non-flat include files.
uv run --no-project python scripts/mpi_issue179_prefix_evidence.py "$inventory" --input-closure "$output/input-closure-before.sha256" > "$output/input-closure-check.log" 2>&1
[[ -s $output/input-closure-before.sha256 ]]
printf '%s\n' "$listing" > "$output/selected-test-list.txt"
printf '%s\n' "$MPI179_SOURCE_COMMIT" > "$output/source-commit.txt"
sha256sum "$binary" "$0" scripts/mpi_issue179_prefix_evidence.py scripts/test_mpi_issue179_prefix_evidence.py scripts/compare_mpi_issue179.py scripts/mpi_issue179_repeat_evidence.py scripts/mpi_issue179_environment.sh scripts/mpi_issue179_selection.sh scripts/run_optional_gates_183.py "$inventory/matrix.tsv" "$manifest" "$proof" > "$output/executables-checkers.sha256"
export MVMC_RS_INNER_THREADS=1
unset MVMC_RS_INNER_THRESHOLD MPI179_FAIL_RANK
mpirun --version > "$output/mpi-version.txt"
ldd "$binary" > "$output/binary-ldd.txt"
awk '/=> \/|^[[:space:]]*\// { for (i=1;i<=NF;i++) if ($i ~ /^\//) print $i }' "$output/binary-ldd.txt" | sort -u | xargs -r sha256sum > "$output/libraries-before.sha256"
[[ -s $output/libraries-before.sha256 ]]
uv run --no-project python scripts/mpi_issue179_prefix_evidence.py "$output" --backend-binary "$binary" > "$output/runtime-backend-check.log" 2>&1
printf '%s\n' "binary=$binary" "inventory=$inventory" "source_manifest=$manifest" "source_commit=$MPI179_SOURCE_COMMIT" "prefixes=1 2 3" "workers=1" "repeats=2" "MVMC_RS_INNER_THRESHOLD=UNSET (production default)" "MPI179_TIMEOUT=${MPI179_TIMEOUT:-60}" "CARGO_TARGET_DIR=$CARGO_TARGET_DIR" "OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS" "OMP_NUM_THREADS=$OMP_NUM_THREADS" "MKL_NUM_THREADS=$MKL_NUM_THREADS" "BLIS_NUM_THREADS=$BLIS_NUM_THREADS" > "$output/environment.txt"
rustc -Vv > "$output/rust-version.txt"
uv --version > "$output/uv-version.txt"
printf 'cell\tprefix\tfirst_exit\tsecond_exit\n' > "$output/results.tsv"
printf 'cell\tvalidation_exit\n' > "$output/checker-results.tsv"
bad=0 completed=0
phase=launches
while IFS=$'\t' read -r id ranks mode width cg store projection expected status exitcode; do
    [[ $id == cell || $id == id ]] && continue
    export MPI179_INPUT="$inventory/$id/inputs/namelist.def"
    for steps in 1 2 3; do
        export MPI179_STEPS=$steps
        cell="$output/$id/prefix$steps/w1"
        mkdir -p "$cell"
        sha256sum "$inventory/$id/inputs/"* > "$cell/inputs.sha256"
        exits=()
        for repeat in 1 2; do
            export MPI179_STATE_DIR="$cell/repeat$repeat"
            rc=0
            printf 'MPI179_INPUT=%q MPI179_STATE_DIR=%q MPI179_STEPS=%q MVMC_RS_INNER_THREADS=1 ' "$MPI179_INPUT" "$MPI179_STATE_DIR" "$MPI179_STEPS" >> "$output/commands.txt"
            printf '%q ' timeout --kill-after=5s "${MPI179_TIMEOUT:-60}s" "${mpi179_mpirun[@]}" -n "$ranks" "$binary" --ignored --exact issue179_state --nocapture >> "$output/commands.txt"
            printf '\n' >> "$output/commands.txt"
            timeout --kill-after=5s "${MPI179_TIMEOUT:-60}s" "${mpi179_mpirun[@]}" -n "$ranks" "$binary" --ignored --exact issue179_state --nocapture </dev/null > "$cell/launch$repeat.log" 2>&1 || rc=$?
            printf '%s\n' "$rc" > "$cell/launch$repeat.exit"
            exits+=("$rc")
            (( rc == 0 )) || bad=1
            completed=$((completed + 1))
        done
        sha256sum -c "$cell/inputs.sha256" > "$cell/input-check.txt" || bad=1
        printf '%s\t%s\t%s\t%s\n' "$id" "$steps" "${exits[0]}" "${exits[1]}" >> "$output/results.tsv"
    done
    vrc=0
    uv run --no-project python scripts/mpi_issue179_prefix_evidence.py "$output/$id" --ranks "$ranks" --width "$width" > "$output/$id/validation.log" 2>&1 || vrc=$?
    printf '%s\t%s\n' "$id" "$vrc" >> "$output/checker-results.tsv"
    (( vrc == 0 )) || bad=1
done < "$inventory/matrix.tsv"
(( completed == 72 )) || bad=1
sha256sum -c "$output/executables-checkers.sha256" > "$output/executable-check.txt" || bad=1
sha256sum -c "$output/sources-before.sha256" > "$output/source-after-check.txt" || bad=1
sha256sum -c "$output/libraries-before.sha256" > "$output/library-after-check.txt" || bad=1
sha256sum -c "$output/input-closure-before.sha256" > "$output/input-closure-after-check.txt" || bad=1
phase=complete
printf 'launches=%s expected=72 exit=%s\n' "$completed" "$bad" > "$output/terminal-summary.txt"
exit "$bad"

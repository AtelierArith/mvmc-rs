#!/usr/bin/env bash
# Capture all supported identity CLI combinations; not an independent parity gate.
set -euo pipefail
cd "$(dirname "$0")/.."
matrix=$(realpath "${1:?usage: <CLI-evidence-directory> <state-build.json>}")
source scripts/mpi_issue179_selection.sh
mpi179_validate_selection "$matrix/matrix.tsv" rust
build=$(realpath "${2:?state build artifact JSON required}")
source scripts/mpi_issue179_environment.sh
bin=$(jq -r 'select(.reason=="compiler-artifact" and .target.name=="mpi_issue179_state" and .executable!=null) | .executable' "$build")
[[ -x $bin && ( $bin == /tmp/mvmc-issue179-target/* || $bin == /home/vscode/.cache/mvmc/target/issue179-*/* ) ]]
mpi179_require_test "$bin" issue179_state
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
out=$(mktemp -d /tmp/mvmc-issue179-rust-states.XXXXXX)
echo "Evidence: $out"
cp "$build" "$out/build-artifacts.json"
printf '%s\n' "$matrix" "$build" > "$out/origin.txt"
# Build/source provenance comes from the supplied build's evidence directory,
# not the potentially edited shared worktree at execution time.
sha256sum "$bin" > "$out/executable.sha256"
printf 'cell\texit\tstatus\n' > "$out/matrix.tsv"
bad=0
completed=0
while IFS=$'\t' read -r id ranks mode split cg store projection expected status exitcode; do
    [[ $projection == identity && $expected == success && $id == *-cg* ]] || continue
    dir="$out/$id"
    mkdir -p "$dir"
    input="$matrix/$id/inputs/namelist.def"
    sha256sum "$matrix/$id/inputs/"* > "$dir/input-sha256.txt"
    rc=0
    MPI179_INPUT="$input" MPI179_STATE_DIR="$dir/state" timeout --kill-after=5s "${MPI179_TIMEOUT:-30}s" "${mpi179_mpirun[@]}" -n "$ranks" "$bin" --ignored --exact issue179_state --nocapture < /dev/null > "$dir/launch.log" 2>&1 || rc=$?
    status=CAPTURED_NOT_COMPARED
    if (( rc != 0 )); then status=FAIL; bad=1; fi
    for (( rank=0; rank<ranks; rank++ )); do
        if [[ ! -s "$dir/state/rank-$rank.txt" ]]; then status=MISSING_EVIDENCE; bad=1; fi
    done
    printf '%s\t%s\t%s\n' "$id" "$rc" "$status" | tee -a "$out/matrix.tsv"
    completed=$((completed + 1))
done < "$matrix/matrix.tsv"
if (( completed != MPI179_SELECTED_CELLS )); then echo INCOMPLETE_STATE_MATRIX >&2; bad=1; fi
exit "$bad"

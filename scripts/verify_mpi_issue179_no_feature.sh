#!/usr/bin/env bash
# Actual launch-policy rejection: a CLI built without MPI cannot run world2/4.
set -euo pipefail
cd "$(dirname "$0")/.."
binary=$(realpath "${1:?usage: script frozen-no-MPI-CLI namelist new-evidence-directory}")
input=$(realpath "${2:?namelist required}")
output=${3:?new evidence directory required}
[[ -x $binary && -f $input && ! -e $output ]]
source scripts/mpi_issue179_environment.sh
mkdir -p "$output"
output=$(realpath "$output")
sha256sum "$binary" "$0" scripts/mpi_issue179_environment.sh > "$output/binary-checker.sha256"
sha256sum "$(dirname "$input")/"* > "$output/inputs.sha256"
mpiexec --version > "$output/mpi-version.txt"
printf 'ranks\tlaunch_exit\tboundary_exit\n' > "$output/results.tsv"
bad=0
for ranks in 2 4; do
    rc=0 vrc=0
    timeout --kill-after=5s 30s "${mpi179_mpirun[@]}" -n "$ranks" "$binary" "$input" --nsteps 1 --nsmp 1 --mode real --out-dir "$output/world$ranks" </dev/null > "$output/world$ranks.log" 2>&1 || rc=$?
    # A timeout/signal is not proof of the expected fail-fast policy.
    (( rc > 0 && rc < 124 )) || vrc=1
    count=$(grep -F -c 'MPI launcher detected; rebuild mvmc-cli with --features mpi' "$output/world$ranks.log" || true)
    [[ $count == "$ranks" && ! -e $output/world$ranks ]] || vrc=1
    printf '%s\t%s\t%s\n' "$ranks" "$rc" "$vrc" | tee -a "$output/results.tsv"
    if (( vrc != 0 )); then bad=1; fi
done
sha256sum -c "$output/binary-checker.sha256" > "$output/binary-checker-check.txt" || bad=1
sha256sum -c "$output/inputs.sha256" > "$output/input-check.txt" || bad=1
printf 'NO_FEATURE_LAUNCH_REJECTION worlds=2,4 bad=%s root=%s\n' "$bad" "$output" | tee "$output/terminal-summary.txt"
exit "$bad"

#!/usr/bin/env bash
# Public CLI rank/output boundaries; not independent numerical accuracy.
set -euo pipefail
cd "$(dirname "$0")/.."
binary=$(realpath "${1:?usage: script frozen-MPI-CLI namelist new-evidence-directory}")
input=$(realpath "${2:?namelist required}")
output=${3:?new evidence directory required}
[[ -x $binary && -f $input && ! -e $output ]]
source scripts/mpi_issue179_environment.sh
mkdir -p "$output"
output=$(realpath "$output")
sha256sum "$binary" "$0" scripts/mpi_issue179_environment.sh > "$output/binary-checker.sha256"
sha256sum "$(dirname "$input")/"* > "$output/inputs.sha256"
mpiexec --version > "$output/mpi-version.txt"
ldd "$binary" > "$output/binary-ldd.txt"
printf 'ranks\tlaunch_exit\tboundary_exit\n' > "$output/results.tsv"
bad=0
for ranks in 2 4; do
    rc=0 vrc=0
    timeout --kill-after=5s 60s "${mpi179_mpirun[@]}" -n "$ranks" "$binary" "$input" --nsteps 1 --nsmp 1 --mode real --out-dir "$output/world$ranks" </dev/null > "$output/world$ranks.log" 2>&1 || rc=$?
    if (( rc != 0 )); then vrc=1; else
        # Rust CLI architecture: banner and public summary once, not invented
        # Julia logging strings or one summary per independently seeded chain.
        for message in 'model    :' '=== mvmc — Julia-mVMC Rust port ===' '=== Completed 1 SR steps' 'Output files written to:' 'Final energy / site:' 'Final-window means (1 steps):'; do
            count=$(grep -F -c "$message" "$output/world$ranks.log" || true)
            [[ $count == 1 ]] || vrc=1
        done
        for name in zvo_out.dat zvo_var.dat; do
            file="$output/world$ranks/$name"
            [[ -s $file ]] && [[ $(wc -l < "$file") == 1 ]] || vrc=1
        done
        [[ -s $output/world$ranks/zqp_opt.dat ]] || vrc=1
    fi
    printf '%s\t%s\t%s\n' "$ranks" "$rc" "$vrc" | tee -a "$output/results.tsv"
    if (( vrc != 0 )); then bad=1; fi
done
sha256sum -c "$output/binary-checker.sha256" > "$output/binary-checker-check.txt" || bad=1
sha256sum -c "$output/inputs.sha256" > "$output/input-check.txt" || bad=1
printf 'CLI_ROOT_BOUNDARIES worlds=2,4 bad=%s root=%s\n' "$bad" "$output" | tee "$output/terminal-summary.txt"
exit "$bad"

#!/usr/bin/env bash
# SOURCE proposal only. Do not invoke before parent approves build/preflight.
set -euo pipefail
: "${MPI178_BINARY:?exact approved MPI-enabled test executable}"
: "${MPI178_EVIDENCE:?fresh exclusive output root}"
: "${MPI178_LAUNCHER:?absolute audited native MPI launcher}"
mkdir "$MPI178_EVIDENCE"
trap 'printf "%s\n" "$?" > "$MPI178_EVIDENCE/launcher-script.status"' EXIT
"$MPI178_LAUNCHER" --version > "$MPI178_EVIDENCE/launcher-version.txt" 2>&1
"$MPI178_LAUNCHER" -help > "$MPI178_EVIDENCE/launcher-help.txt" 2>&1
flags=()
if rg -q 'HYDRA|Hydra' "$MPI178_EVIDENCE/launcher-version.txt"; then
    rg -q -- '-disable-auto-cleanup' "$MPI178_EVIDENCE/launcher-help.txt"
    flags=(-disable-auto-cleanup)
elif rg -q 'Open MPI|OpenRTE' "$MPI178_EVIDENCE/launcher-version.txt"; then
    printf '%s\n' 'This reviewed per-rank capture variant requires Hydra; OpenMPI capture needs separate source review.' >&2
    exit 2
else
    printf '%s\n' 'Unsupported/unaudited launcher metadata; do not silently skip.' >&2
    exit 2
fi
sha256sum "$MPI178_BINARY" > "$MPI178_EVIDENCE/binary-before.sha256"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for ranks in 2 4; do
    for cell in opt:1 opt:2 physcal:1; do
        api=${cell%:*}; width=${cell#*:}
        for fault in root last-owner; do
            name="n${ranks}-${api}-w${width}-${fault}"
            rankdir="$MPI178_EVIDENCE/$name-ranks"
            mkdir "$rankdir"
            # Explicit pre-launch empty stderr artifacts: never synthesize missing
            # capture files after execution. Noclobber makes creation exclusive.
            stderr_receipt="$MPI178_EVIDENCE/$name-stderr-initial.txt"
            for ((rank=0; rank<ranks; rank++)); do
                stderr_file="$rankdir/rank-$rank.stderr"
                (set -o noclobber; : > "$stderr_file")
                test "$(stat -c %s "$stderr_file")" -eq 0
                printf 'rank=%s bytes=0 path=%s\n' "$rank" "$stderr_file" >> "$stderr_receipt"
            done
            sha256sum "$rankdir"/*.stderr > "$MPI178_EVIDENCE/$name-stderr-initial.sha256"
            capture=(-outfile-pattern "$rankdir/rank-%r.stdout" -errfile-pattern "$rankdir/rank-%r.stderr" -print-all-exitcodes)
            output="$MPI178_EVIDENCE/$name-output"
            export MPI_ISSUE178_API="$api" MPI_ISSUE178_WIDTH="$width"
            export MPI_ISSUE178_FAIL_RANK="$fault" MPI_ISSUE178_OUTPUT="$output"
            printf '%q ' timeout --signal=TERM --kill-after=10s 120s "$MPI178_LAUNCHER" "${flags[@]}" "${capture[@]}" -n "$ranks" "$MPI178_BINARY" --ignored --exact mpi_phase2::actual_sampling_failure_reaches_comm1_then_all_global_ranks --nocapture --test-threads=1 > "$MPI178_EVIDENCE/$name-command.txt"
            printf '\n' >> "$MPI178_EVIDENCE/$name-command.txt"
            set +e
            timeout --signal=TERM --kill-after=10s 120s "$MPI178_LAUNCHER" "${flags[@]}" "${capture[@]}" -n "$ranks" "$MPI178_BINARY" --ignored --exact mpi_phase2::actual_sampling_failure_reaches_comm1_then_all_global_ranks --nocapture --test-threads=1 > "$MPI178_EVIDENCE/$name.log" 2>&1
            status=$?
            set -e
            printf '%s\n' "$status" > "$MPI178_EVIDENCE/$name.status"
            # Preserve failure; no retries or continuation that claims success.
            test "$status" -eq 0 || exit "$status"
            bad=0; if test "$fault" = last-owner; then bad=$((ranks-width)); fi
            physcal=false; if test "$api" = physcal; then physcal=true; fi
            test "$(find "$rankdir" -maxdepth 1 -type f | wc -l)" -eq "$((ranks*2))"
            for ((rank=0; rank<ranks; rank++)); do
                test -s "$rankdir/rank-$rank.stdout"
                test -f "$rankdir/rank-$rank.stderr"
                awk -v expected_rank="$rank" -v ranks="$ranks" -v width="$width" -v physcal="$physcal" -v bad="$bad" -f "$(dirname "$0")/mpi_issue178_validate_ranks.awk" "$rankdir/rank-$rank.stdout" "$rankdir/rank-$rank.stderr" > "$MPI178_EVIDENCE/$name-rank-$rank-validation.log"
            done
        done
    done
done
sha256sum -c --quiet "$MPI178_EVIDENCE/binary-before.sha256" > "$MPI178_EVIDENCE/binary-post.log"

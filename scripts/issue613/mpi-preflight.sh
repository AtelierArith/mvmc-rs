#!/usr/bin/env bash
# Optional live MPICH preflight check; build the MPI-enabled CLI first.
set -euo pipefail
if [[ $# != 1 ]]; then
    echo "usage: $0 PATH_TO_MPI_CLI" >&2
    exit 2
fi
binary=$(realpath "$1")
root=$(cd "$(dirname "$0")/../.." && pwd)
launcher=${MVMC_MPI_LAUNCHER:-/opt/mpich/bin/mpirun}
test -x "$binary"
test -x "$launcher"
out=$(mktemp -d "${TMPDIR:-/tmp}/mvmc-613-mpi.XXXXXXXX")
trap 'rm -rf "$out"' EXIT
fail() {
    printf 'failed: %s status=%s\n' "$label" "${status:-unknown}" >&2
    cat "$out/$label.log" >&2
    for log in "$out/$label"-rank*.log; do
        [[ ! -f "$log" ]] || cat "$log" >&2
    done
    for rank_file in "$out/$label"-rank*.status; do
        [[ ! -f "$rank_file" ]] || { printf '%s: ' "$rank_file" >&2; cat "$rank_file" >&2; }
    done
    exit 1
}
for ranks in 2 4; do
    for selection in uniform mixed; do
        for mode in opt phys; do
            label="$ranks-$selection-$mode"
            set +e
            OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
            MVMC_RS_INNER_THREADS=1 MVMC_RS_TENFERRO_THREADS=1 MVMC_RS_SR_BACKEND=tenferro \
                timeout --kill-after=5s 45s "$launcher" -n "$ranks" bash -c '
                    set -euo pipefail
                    selection=$1; mode=$2; binary=$3; root=$4; out=$5; label=$6
                    if [[ "$selection" = uniform || "$PMI_RANK" = 0 ]]; then
                        export MVMC_RS_MEASURE_PF_BACKEND=tenferro
                    else
                        export MVMC_RS_MEASURE_PF_BACKEND=calc-m-all
                    fi
                    args=("$root/tests/fixtures/issue347_varbin/opt_inputs/namelist.def" --out-dir "$out/result-$label")
                    if [[ "$mode" = phys ]]; then
                        args+=(--physcal "$out/missing-parameters.dat")
                    else
                        args+=(--nsteps 1 --nsmp 1)
                    fi
                    set +e
                    "$binary" "${args[@]}" > "$out/$label-rank$PMI_RANK.log" 2>&1
                    rank_status=$?
                    set -e
                    printf "%s\n" "$rank_status" > "$out/$label-rank$PMI_RANK.status"
                    # Prevent the launcher from terminating a sibling before it can record its
                    # own exit status. This is a test-harness file barrier after the CLI exited.
                    for ((other = 0; other < PMI_SIZE; other++)); do
                        for ((poll = 0; poll < 500; poll++)); do
                            [[ -s "$out/$label-rank$other.status" ]] && break
                            sleep 0.01
                        done
                        [[ -s "$out/$label-rank$other.status" ]] || exit 98
                    done
                    exit "$rank_status"
                ' _ "$selection" "$mode" "$binary" "$root" "$out" "$label" > "$out/$label.log" 2>&1
            status=$?
            set -e
            [[ "$status" != 124 && "$status" != 137 ]] || fail
            if [[ "$selection" = uniform ]]; then
                [[ "$status" = 2 ]] || fail
            else
                [[ "$status" = 1 ]] || fail
            fi
            for ((rank = 0; rank < ranks; rank++)); do
                [[ -f "$out/$label-rank$rank.status" ]] || fail
                rank_status=$(cat "$out/$label-rank$rank.status")
                if [[ "$selection" = uniform ]]; then
                    [[ "$rank_status" = 2 ]] || fail
                    rg -q 'error: MVMC_RS_MEASURE_PF_BACKEND: unsupported: tenferro has no skew-symmetric factorization or Pfaffian' "$out/$label-rank$rank.log" || fail
                else
                    [[ "$rank_status" = 1 ]] || fail
                    rg -q 'CLI (control agreement failed|run controls differ)' "$out/$label-rank$rank.log" || fail
                fi
                ! rg -q 'panicked|thread .* panic' "$out/$label-rank$rank.log" || fail
            done
            [[ ! -e "$out/result-$label" ]] || fail
            printf '%s launcher_status=%s, per-rank statuses verified: passed\n' "$label" "$status"
        done
    done
done

# Check the actual launcher exit code without rank wrappers. Abrupt process::exit
# before MPI_Finalize previously made MPICH occasionally return 0 despite errors.
for ranks in 2 4; do
    for repetition in 1 2 3 4 5; do
        label="direct-$ranks-$repetition"
        set +e
        OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
        MVMC_RS_INNER_THREADS=1 MVMC_RS_TENFERRO_THREADS=1 MVMC_RS_SR_BACKEND=tenferro \
        MVMC_RS_MEASURE_PF_BACKEND=tenferro \
            timeout --kill-after=5s 45s "$launcher" -n "$ranks" "$binary" \
            "$root/tests/fixtures/issue347_varbin/opt_inputs/namelist.def" \
            --nsteps 1 --nsmp 1 --out-dir "$out/result-$label" > "$out/$label.log" 2>&1
        status=$?
        set -e
        [[ "$status" = 2 ]] || fail
        rg -q 'error: MVMC_RS_MEASURE_PF_BACKEND: unsupported:' "$out/$label.log" || fail
        ! rg -q 'panicked|thread .* panic' "$out/$label.log" || fail
        [[ ! -e "$out/result-$label" ]] || fail
        printf '%s launcher_status=%s: passed\n' "$label" "$status"
    done
done

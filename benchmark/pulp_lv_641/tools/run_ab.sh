#!/usr/bin/env bash
# Whole-workload A/B of two mvmc binaries on the L128 inputs, with provenance (#698).
#
#   tools/run_ab.sh OUT MAIN_BIN PULP_BIN [PHYS_REPS=5] [OPT_REPS=3] [FIRST_CPU=0]
#
# Runs tools/ab.py (one warmup plus REPS measured repetitions, alternating the binary order)
# for PhysCal and Opt at 1 and 8 threads, pinned to CPUs FIRST_CPU.., BLAS 1 thread. Writes
# OUT/ab-{phys,opt}-t{1,8}.txt (raw per-repetition wall times and C timers), OUT/env.txt
# (date, host, CPU, toolchain, revision, binary SHA-256, BLAS library) and OUT/load.txt
# (uptime before and after every case), then OUT/summary.md via tools/ab_summary.py.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$here/../../.." && pwd)
out=$(mkdir -p "$1" && cd -- "$1" && pwd)
main_bin=$(realpath "$2")
pulp_bin=$(realpath "$3")
phys_reps=${4:-5}
opt_reps=${5:-3}
first=${6:-0}

{
    echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) first_cpu=$first"
    lscpu | grep -E 'Model name|^CPU\(s\)|Thread\(s\) per core'
    rustc -Vv | grep -E '^rustc|^LLVM'
    echo "repo_head=$(git -C "$repo" rev-parse HEAD)"
    echo "phys_reps=$phys_reps opt_reps=$opt_reps (each after one warmup)"
    sha256sum "$main_bin" "$pulp_bin"
    echo "blas: $(ldd "$main_bin" | grep -Ei 'blas|lapack' | tr -s ' \t' ' ' | tr '\n' ';')"
    echo "env: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MVMC_RS_INNER_THREADS=RAYON_NUM_THREADS=threads" \
         "MVMC_RS_SR_BACKEND=c-order MVMC_RS_MEASURE_PF_BACKEND=c-order MVMC_C_TIMER=1 (tools/ab.py)"
} > "$out/env.txt"

work=$(mktemp -d)
# Load trace every 10 s (1-minute load average and the busiest other processes), so a
# repetition disturbed by other jobs can be identified afterwards.
(while true; do
    echo "$(date +%T) $(cut -d' ' -f1-3 /proc/loadavg) | $(ps -eo pcpu,comm --sort=-pcpu --no-headers \
        | grep -v -E ' (mvmc|ps)$' | head -3 | tr -s ' ' | tr '\n' ';')"
    sleep 10
done) > "$out/load-trace.txt" &
tracer=$!
trap 'kill $tracer 2>/dev/null; rm -rf "$work"' EXIT
: > "$out/load.txt"
for case in "phys 1 $phys_reps" "phys 8 $phys_reps" "opt 1 $opt_reps" "opt 8 $opt_reps"; do
    set -- $case
    echo "before $1 t$2: $(uptime)" >> "$out/load.txt"
    echo "command: uv run --no-project python tools/ab.py $1 $2 $first $3 main=$main_bin pulp=$pulp_bin" \
        > "$out/ab-$1-t$2.txt"
    (cd "$work" && uv run --no-project python "$here/ab.py" "$1" "$2" "$first" "$3" \
        main="$main_bin" pulp="$pulp_bin") >> "$out/ab-$1-t$2.txt"
    echo "after $1 t$2: $(uptime)" >> "$out/load.txt"
done
uv run --no-project python "$here/ab_summary.py" "$out" > "$out/summary.md"
cat "$out/summary.md"

#!/usr/bin/env bash
# Independent processes with bounded affinity; this does not emulate mVMC walkers or MPI.
set -euo pipefail
if [[ $# != 2 ]]; then
    echo "usage: $0 /path/to/sr_backend_bench OUTPUT_DIRECTORY" >&2
    exit 2
fi
binary=$1
output=$2
mkdir -p "$output"
if [[ -e "$output/groups.txt" ]]; then
    echo "output directory already contains a run: $output" >&2
    exit 2
fi
for threads in 1 4; do
    for processes in 1 4; do
        for repetition in 1 2 3; do
            start_ns=$(date +%s%N)
            pids=()
            for ((i=1; i<=processes; i++)); do
                RAYON_NUM_THREADS=$threads OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
                    taskset -c 0-3 /usr/bin/time -v "$binary" 256 \
                    > "$output/t${threads}-p${processes}-r${repetition}-${i}.txt" 2>&1 &
                pids+=("$!")
            done
            for pid in "${pids[@]}"; do wait "$pid"; done
            end_ns=$(date +%s%N)
            printf 'threads=%s processes=%s repetition=%s elapsed_ns=%s\n' \
                "$threads" "$processes" "$repetition" "$((end_ns - start_ns))" \
                >> "$output/groups.txt"
        done
    done
done

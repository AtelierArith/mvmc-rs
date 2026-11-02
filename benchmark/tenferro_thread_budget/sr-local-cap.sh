#!/usr/bin/env bash
# Compare standalone SR stage behavior before/after the inferred local cap.
set -euo pipefail
if [[ $# != 4 ]]; then
    echo "usage: $0 DEVELOPMENT_IMAGE BEFORE_VOLUME AFTER_VOLUME EMPTY_OUTPUT_DIRECTORY" >&2
    exit 2
fi
image=$1
before=$2
after=$3
out=$4
base=${MVMC_BENCH_CPU_BASE:-0}
hoist=${MVMC_BENCH_SR_HOIST:-0}
[[ $base =~ ^[0-9]+$ ]] || { echo "MVMC_BENCH_CPU_BASE must be a nonnegative CPU number" >&2; exit 2; }
last=$((base + 3))
mkdir -p "$out"
out=$(cd "$out" && pwd)
[[ ! -e "$out/config.env" ]] || { echo "output directory is not fresh: $out" >&2; exit 2; }
printf 'cpu_base=%s\ncpuset=%s-%s\nrayon_threads=4\ntenferro_threads=unset\nhoist=%s\n' "$base" "$base" "$last" "$hoist" > "$out/config.env"
for inner in 1 4; do
    for rep in 0 1 2 3; do
        if ((rep % 2 == 0)); then versions='before after'; else versions='after before'; fi
        for version in $versions; do
            if [[ $version == before ]]; then volume=$before; else volume=$after; fi
            docker run --rm --platform=linux/amd64 --cpuset-cpus="$base-$last" --shm-size=1g \
                -v "$volume:/target" -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 \
                -e "MVMC_RS_INNER_THREADS=$inner" -e RAYON_NUM_THREADS=4 -e "MVMC_BENCH_SR_HOIST=$hoist" \
                --entrypoint bash "$image" -lc '
                    set -euo pipefail
                    unset MVMC_RS_TENFERRO_THREADS MVMC_RS_TENFERRO_SPIN_POLLS
                    TIMEFORMAT="wall=%R user=%U sys=%S"
                    time /target/test-fast/examples/sr_backend_bench 64 128
                ' > "$out/${version}-i${inner}-r${rep}.txt" 2>&1
        done
    done
done

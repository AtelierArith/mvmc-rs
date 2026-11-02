#!/usr/bin/env bash
# Matched L32 CLI runs with default and effectively serial inner-kernel work gates.
set -euo pipefail
if [[ $# != 4 ]]; then
    echo "usage: $0 DEVELOPMENT_IMAGE TARGET_VOLUME INPUT_DIRECTORY EMPTY_OUTPUT_DIRECTORY" >&2
    exit 2
fi
image=$1
volume=$2
inputs=$(cd "$3" && pwd)
out=$4
base=${MVMC_BENCH_CPU_BASE:-0}
[[ $base =~ ^[0-9]+$ ]] || exit 2
mkdir -p "$out"
out=$(cd "$out" && pwd)
[[ ! -e "$out/results" ]] || { echo 'results directory exists' >&2; exit 2; }
mkdir "$out/results"
sha256sum "$inputs"/{opt,phys}/* > "$out/input-sha256.txt"
printf 'cpu_base=%s\ninner_threads=4\ntenferro_threads=4\n' "$base" > "$out/config.env"
for mode in opt phys; do
    for rep in 0 1 2 3 4; do
        if ((rep % 2 == 0)); then gates='default serial'; else gates='serial default'; fi
        for gate in $gates; do
            docker run --rm --platform=linux/amd64 --cpuset-cpus="$base-$((base+3))" --shm-size=1g \
                -v "$inputs:/input:ro" -v "$out/results:/results" -v "$volume:/target:ro" \
                -w "/input/$mode" -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 \
                -e MVMC_RS_INNER_THREADS=4 -e RAYON_NUM_THREADS=4 -e MVMC_RS_TENFERRO_THREADS=4 \
                -e MVMC_RS_SR_BACKEND=tenferro -e "BENCH_GATE=$gate" -e "BENCH_MODE=$mode" \
                -e "BENCH_DEST=/results/${mode}-${gate}-${rep}" --entrypoint bash "$image" -lc '
                    set -euo pipefail
                    unset MVMC_RS_INNER_PROFILE MVMC_RS_TENFERRO_SPIN_POLLS
                    if [[ "$BENCH_GATE" == serial ]]; then export MVMC_RS_INNER_MIN_WORK_NS=1000000000000; else unset MVMC_RS_INNER_MIN_WORK_NS; fi
                    TIMEFORMAT="wall=%R user=%U sys=%S"
                    if [[ "$BENCH_MODE" == opt ]]; then
                        time timeout --kill-after=5s 120s /target/release/mvmc /input/opt/namelist.def --nsteps 20 --nsmp 1 --initial-def none --out-dir "$BENCH_DEST"
                    else
                        time timeout --kill-after=5s 120s /target/release/mvmc /input/phys/namelist.def --physcal /input/phys/zqp_opt.dat --out-dir "$BENCH_DEST"
                    fi
                ' > "$out/${mode}-${gate}-${rep}.log" 2>&1
        done
    done
done

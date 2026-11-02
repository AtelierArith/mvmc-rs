#!/usr/bin/env bash
# Compare inferred Engine budgets in single-process Opt and PhysCal.
set -euo pipefail
if [[ $# != 5 ]]; then
    echo "usage: $0 DEVELOPMENT_IMAGE BEFORE_VOLUME AFTER_VOLUME INPUT_DIRECTORY EMPTY_OUTPUT_DIRECTORY" >&2
    exit 2
fi
image=$1
before=$2
after=$3
inputs=$(cd "$4" && pwd)
out=$5
base=${MVMC_BENCH_CPU_BASE:-0}
[[ $base =~ ^[0-9]+$ ]] || { echo "MVMC_BENCH_CPU_BASE must be a nonnegative CPU number" >&2; exit 2; }
last=$((base + 15))
mkdir -p "$out"
out=$(cd "$out" && pwd)
[[ ! -e "$out/results" ]] || { echo "output directory already contains results: $out" >&2; exit 2; }
mkdir "$out/results"
printf 'cpu_base=%s\ncpuset=%s-%s\nrayon_threads=16\ntenferro_threads=inner\n' "$base" "$base" "$last" > "$out/config.env"
sha256sum "$inputs"/{opt,phys}/* > "$out/input-sha256.txt"
for mode in opt phys; do
    for inner in 4 16; do
        for rep in 0 1 2 3; do
            if ((rep % 2 == 0)); then versions='before after'; else versions='after before'; fi
            for version in $versions; do
                if [[ $version == before ]]; then volume=$before; else volume=$after; fi
                docker run --rm --platform=linux/amd64 --cpuset-cpus="$base-$last" --shm-size=1g \
                    -v "$inputs:/input:ro" -v "$out/results:/results" -v "$volume:/target" \
                    -w "/input/$mode" -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 \
                    -e "MVMC_RS_INNER_THREADS=$inner" -e RAYON_NUM_THREADS=16 -e "MVMC_RS_TENFERRO_THREADS=$inner" \
                    -e "MVMC_EXPERIMENT_SHARED_CPU_TEAM=$([[ $version == after ]] && echo 1 || echo 0)" -e MVMC_RS_SR_BACKEND=tenferro -e "BENCH_MODE=$mode" \
                    -e "BENCH_DEST=/results/${mode}-${version}-i${inner}-r${rep}" \
                    --entrypoint bash "$image" -lc '
                        set -euo pipefail
                        unset MVMC_RS_TENFERRO_SPIN_POLLS
                        TIMEFORMAT="wall=%R user=%U sys=%S"
                        if [[ "$BENCH_MODE" == opt ]]; then
                            time timeout --kill-after=5s 120s /target/release/mvmc /input/opt/namelist.def --nsteps 20 --nsmp 1 --initial-def none --out-dir "$BENCH_DEST"
                        else
                            time timeout --kill-after=5s 120s /target/release/mvmc /input/phys/namelist.def --physcal /input/phys/zqp_opt.dat --out-dir "$BENCH_DEST"
                        fi
                    ' > "$out/${mode}-${version}-i${inner}-r${rep}.txt" 2>&1
            done
        done
    done
done

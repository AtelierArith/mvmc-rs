#!/usr/bin/env bash
# Compare per-engine budgets while each MPI rank has four dedicated CPUs.
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
[[ $base =~ ^[0-9]+$ ]] || { echo "MVMC_BENCH_CPU_BASE must be a nonnegative CPU number" >&2; exit 2; }
last=$((base + 15))
mkdir -p "$out"
out=$(cd "$out" && pwd)
[[ ! -e "$out/results" ]] || { echo "output directory already contains results: $out" >&2; exit 2; }
mkdir "$out/results"
printf 'cpu_base=%s\ncpuset=%s-%s\ninner_threads=4\n' "$base" "$base" "$last" > "$out/config.env"
sha256sum "$inputs"/{opt,phys}/* > "$out/input-sha256.txt"
for mode in opt phys; do
    for rep in 0 1 2 3; do
        if ((rep % 2 == 0)); then settings='1 4'; else settings='4 1'; fi
        for threads in $settings; do
            docker run --rm --platform=linux/amd64 --cpuset-cpus="$base-$last" --shm-size=1g \
                -v "$inputs:/input:ro" -v "$out/results:/results" -v "$volume:/target" \
                -w "/input/$mode" -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 \
                -e MVMC_RS_INNER_THREADS=4 -e RAYON_NUM_THREADS=4 -e "MVMC_RS_TENFERRO_THREADS=$threads" \
                -e MVMC_RS_SR_BACKEND=tenferro -e "BENCH_CPU_BASE=$base" \
                -e "BENCH_MODE=$mode" -e "BENCH_DEST=/results/${mode}-t${threads}-r${rep}" \
                --entrypoint bash "$image" -lc '
                    set -euo pipefail
                    TIMEFORMAT="wall=%R user=%U sys=%S"
                    launcher=(/opt/mpich/bin/mpirun -n 4 bash -c '\''first=$((BENCH_CPU_BASE + 4 * PMI_RANK)); last=$((first + 3)); exec taskset -c "$first-$last" "$@"'\'' _ /target/release/mvmc)
                    if [[ "$BENCH_MODE" == opt ]]; then
                        time timeout --kill-after=5s 120s "${launcher[@]}" /input/opt/namelist.def --nsteps 20 --nsmp 1 --initial-def none --out-dir "$BENCH_DEST"
                    else
                        time timeout --kill-after=5s 120s "${launcher[@]}" /input/phys/namelist.def --physcal /input/phys/zqp_opt.dat --out-dir "$BENCH_DEST"
                    fi
                ' > "$out/${mode}-t${threads}-r${rep}.txt" 2>&1
        done
    done
done

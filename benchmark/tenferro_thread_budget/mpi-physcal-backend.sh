#!/usr/bin/env bash
# Compare SR backend choices during fixed-parameter MPI PhysCal.
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
printf 'cpu_base=%s\ncpuset=%s-%s\ninner_threads=4\ntenferro_threads=4\n' "$base" "$base" "$last" > "$out/config.env"
sha256sum "$inputs"/phys/* > "$out/input-sha256.txt"
for rep in 0 1 2 3; do
    if ((rep % 2 == 0)); then settings='c-order tenferro'; else settings='tenferro c-order'; fi
    for backend in $settings; do
        docker run --rm --platform=linux/amd64 --cpuset-cpus="$base-$last" --shm-size=1g \
            -v "$inputs:/input:ro" -v "$out/results:/results" -v "$volume:/target" \
            -w /input/phys -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 \
            -e MVMC_RS_INNER_THREADS=4 -e RAYON_NUM_THREADS=4 -e MVMC_RS_TENFERRO_THREADS=4 \
            -e "MVMC_RS_SR_BACKEND=$backend" -e "BENCH_CPU_BASE=$base" \
            -e "BENCH_DEST=/results/phys-${backend}-r${rep}" \
            --entrypoint bash "$image" -lc '
                set -euo pipefail
                TIMEFORMAT="wall=%R user=%U sys=%S"
                time timeout --kill-after=5s 120s /opt/mpich/bin/mpirun -n 4 bash -c '\''first=$((BENCH_CPU_BASE + 4 * PMI_RANK)); last=$((first + 3)); exec taskset -c "$first-$last" "$@"'\'' _ /target/release/mvmc /input/phys/namelist.def --physcal /input/phys/zqp_opt.dat --out-dir "$BENCH_DEST"
            ' > "$out/phys-${backend}-r${rep}.txt" 2>&1
    done
done

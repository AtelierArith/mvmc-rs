#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 4 ]]; then
    echo "usage: $0 DEVELOPMENT_IMAGE TARGET_VOLUME OPT_INPUT_DIRECTORY EMPTY_OUTPUT_DIRECTORY" >&2
    exit 2
fi
image=$1
volume=$2
input=$(cd "$3" && pwd)
out=$4
base=${MVMC_BENCH_CPU_BASE:-0}
[[ $base =~ ^[0-9]+$ ]] || exit 2
mkdir -p "$out"
out=$(cd "$out" && pwd)
[[ ! -e "$out/results" ]] || { echo 'results directory exists' >&2; exit 2; }
mkdir -p "$out/results" "$out/input"
cp "$input"/* "$out/input/"
sed -i -E 's/^NSRCG[[:space:]].*/NSRCG          1/' "$out/input/modpara.def"
sha256sum "$out/input"/* > "$out/input-sha256.txt"
for rep in 0 1 2 3; do
 if ((rep % 2 == 0)); then modes='held stage'; else modes='stage held'; fi
 for mode in $modes; do
  docker run --rm --platform=linux/amd64 --cpuset-cpus="$base-$((base+15))" --shm-size=1g \
   -v "$out/input:/input:ro" -v "$out/results:/results" -v "$volume:/target:ro" -w /input \
   -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e MVMC_RS_INNER_THREADS=4 -e RAYON_NUM_THREADS=4 \
   -e MVMC_RS_SR_BACKEND=tenferro -e "MVMC_RS_TENFERRO_TEAM_LIFETIME=$mode" \
   -e "BENCH_CPU_BASE=$base" -e "BENCH_DEST=/results/$mode-$rep" --entrypoint bash "$image" -lc '
    set -euo pipefail
    unset MVMC_RS_TENFERRO_THREADS MVMC_RS_TENFERRO_SPIN_POLLS
    TIMEFORMAT="wall=%R user=%U sys=%S"
    launcher=(/opt/mpich/bin/mpirun -n 4 bash -c '\''first=$((BENCH_CPU_BASE + 4 * PMI_RANK)); last=$((first + 3)); exec taskset -c "$first-$last" "$@"'\'' _ /target/release/mvmc)
    time timeout --kill-after=5s 120 "${launcher[@]}" /input/namelist.def --nsteps 20 --nsmp 1 --initial-def none --out-dir "$BENCH_DEST"
   ' > "$out/$mode-$rep.log" 2>&1
 done
done

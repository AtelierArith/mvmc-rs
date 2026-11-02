#!/usr/bin/env bash
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
[[ $base =~ ^[0-9]+$ ]] || exit 2
mkdir -p "$out"
out=$(cd "$out" && pwd)
[[ ! -e "$out/results" ]] || { echo 'results directory exists' >&2; exit 2; }
mkdir "$out/results"
sha256sum "$inputs"/* > "$out/input-sha256.txt"
for rep in 0 1 2 3; do
 if ((rep % 2 == 0)); then versions='before after'; else versions='after before'; fi
 for version in $versions; do
  if [[ $version == before ]]; then volume=$before; else volume=$after; fi
  docker run --rm --platform=linux/amd64 --cpuset-cpus="$base-$((base+3))" --shm-size=1g \
    -v "$inputs:/input:ro" -v "$out/results:/results" -v "$volume:/target:ro" -w /input \
    -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e MVMC_RS_INNER_THREADS=4 -e RAYON_NUM_THREADS=4 \
    -e MVMC_RS_TENFERRO_THREADS=4 -e MVMC_RS_SR_BACKEND=tenferro -e "BENCH_DEST=/results/$version-$rep" \
    --entrypoint bash "$image" -lc '
     set -euo pipefail
     unset MVMC_RS_INNER_MIN_WORK_NS MVMC_RS_INNER_PROFILE MVMC_RS_TENFERRO_SPIN_POLLS
     TIMEFORMAT="wall=%R user=%U sys=%S"
     time timeout --kill-after=5s 300 /target/release/mvmc /input/namelist.def --physcal /input/zqp_opt.dat --out-dir "$BENCH_DEST"
    ' > "$out/$version-$rep.log" 2>&1
 done
done

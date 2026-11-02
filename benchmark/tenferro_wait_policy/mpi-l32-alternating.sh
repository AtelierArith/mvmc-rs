#!/usr/bin/env bash
# Alternate default/zero spin and 1/4 tenferro threads for a fixed four-rank workload.
set -euo pipefail
if [[ $# != 4 ]]; then
    echo "usage: $0 DEVELOPMENT_IMAGE TARGET_VOLUME INPUT_DIRECTORY EMPTY_OUTPUT_DIRECTORY" >&2
    exit 2
fi
image=$1
volume=$2
inputs=$(cd "$3" && pwd)
out=$4
cpu_set=${MVMC_BENCH_CPUSET:-0-15}
inner_threads=${MVMC_BENCH_INNER_THREADS:-4}
mkdir -p "$out"
out=$(cd "$out" && pwd)
if [[ -e "$out/results" ]]; then
    echo "output directory already contains results: $out" >&2
    exit 2
fi
mkdir "$out/results"
printf 'cpuset=%s\ninner_threads=%s\n' "$cpu_set" "$inner_threads" > "$out/config.env"
sha256sum "$inputs"/{opt,phys}/* > "$out/input-sha256.txt"
for mode in opt phys; do
    for repetition in 0 1 2 3; do
        if (( repetition % 2 == 0 )); then configs='default:1 zero:1 default:4 zero:4'; else configs='zero:4 default:4 zero:1 default:1'; fi
        for config in $configs; do
            policy=${config%:*}
            threads=${config#*:}
            wait_env=()
            if [[ $policy == zero ]]; then wait_env=(-e MVMC_RS_TENFERRO_SPIN_POLLS=0); fi
            docker run --rm --platform=linux/amd64 --cpuset-cpus="$cpu_set" --shm-size=1g \
                -v "$inputs:/input:ro" -v "$out/results:/results" -v "$volume:/target" \
                -w "/input/$mode" \
                -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e MVMC_RS_INNER_THREADS="$inner_threads" \
                -e RAYON_NUM_THREADS=4 -e MVMC_RS_TENFERRO_THREADS="$threads" \
                "${wait_env[@]}" \
                -e MVMC_RS_SR_BACKEND=tenferro -e "BENCH_MODE=$mode" \
                -e "BENCH_DEST=/results/${mode}-${policy}-t${threads}-r${repetition}" \
                --entrypoint bash "$image" -lc '
                    TIMEFORMAT="wall=%R user=%U sys=%S"
                    if [[ "$BENCH_MODE" == opt ]]; then
                        time timeout --kill-after=5s 120s /opt/mpich/bin/mpirun -n 4 /target/release/mvmc \
                            /input/opt/namelist.def --nsteps 20 --nsmp 1 --initial-def none --out-dir "$BENCH_DEST"
                    else
                        time timeout --kill-after=5s 120s /opt/mpich/bin/mpirun -n 4 /target/release/mvmc \
                            /input/phys/namelist.def --physcal /input/phys/zqp_opt.dat --out-dir "$BENCH_DEST"
                    fi
                ' > "$out/${mode}-${policy}-t${threads}-r${repetition}.txt" 2>&1
        done
    done
done

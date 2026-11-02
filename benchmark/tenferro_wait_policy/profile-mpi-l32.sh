#!/usr/bin/env bash
# Diagnostic-only profile: instrumentation changes timing; keep separate from benchmark runs.
set -euo pipefail
if [[ $# != 4 ]]; then
    echo "usage: $0 DEVELOPMENT_IMAGE TARGET_VOLUME INPUT_DIRECTORY EMPTY_OUTPUT_DIRECTORY" >&2
    exit 2
fi
image=$1
volume=$2
inputs=$(cd "$3" && pwd)
out=$4
mkdir -p "$out"
out=$(cd "$out" && pwd)
if [[ -e "$out/opt-default" ]]; then
    echo "output directory already contains a profile: $out" >&2
    exit 2
fi
for mode in opt phys; do
    for policy in default zero; do
        wait_env=()
        if [[ $policy == zero ]]; then wait_env=(-e MVMC_RS_TENFERRO_SPIN_POLLS=0); fi
        docker run --rm --platform=linux/amd64 --cpuset-cpus=0-15 --shm-size=1g \
            -v "$inputs:/input:ro" -v "$out:/results" -v "$volume:/target" -w "/input/$mode" \
            -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e MVMC_RS_INNER_THREADS=4 \
            -e RAYON_NUM_THREADS=4 -e MVMC_RS_TENFERRO_THREADS=4 "${wait_env[@]}" \
            -e MVMC_RS_SR_BACKEND=tenferro -e MVMC_RS_INNER_PROFILE=1 \
            -e MVMC_RS_INNER_OBSERVE=1 -e MVMC_C_TIMER=1 \
            -e "BENCH_MODE=$mode" -e "BENCH_NAME=${mode}-${policy}" \
            --entrypoint bash "$image" -lc '
                if [[ "$BENCH_MODE" == opt ]]; then
                    timeout --kill-after=5s 120s /opt/mpich/bin/mpirun \
                        -errfile-pattern "/results/${BENCH_NAME}-rank-%r.stderr" \
                        -outfile-pattern "/results/${BENCH_NAME}-rank-%r.stdout" -n 4 \
                        /target/release/mvmc /input/opt/namelist.def --nsteps 20 --nsmp 1 \
                        --initial-def none --out-dir "/results/$BENCH_NAME"
                else
                    timeout --kill-after=5s 120s /opt/mpich/bin/mpirun \
                        -errfile-pattern "/results/${BENCH_NAME}-rank-%r.stderr" \
                        -outfile-pattern "/results/${BENCH_NAME}-rank-%r.stdout" -n 4 \
                        /target/release/mvmc /input/phys/namelist.def --physcal /input/phys/zqp_opt.dat \
                        --out-dir "/results/$BENCH_NAME"
                fi
            ' > "$out/${mode}-${policy}-launcher.txt" 2>&1
    done
done

#!/usr/bin/env bash
# Short fixed-work Opt and PhysCal probe; see multi-process.md for scope and limits.
set -euo pipefail
if [[ $# != 3 ]]; then
    echo "usage: $0 DEVELOPMENT_IMAGE TARGET_VOLUME EMPTY_OUTPUT_DIRECTORY" >&2
    exit 2
fi
image=$1
volume=$2
out=$3
cpu_set=${MVMC_BENCH_CPUSET:-0-3}
inner_threads=${MVMC_BENCH_INNER_THREADS:-1}
wait_env=()
if [[ ${MVMC_RS_TENFERRO_SPIN_POLLS+x} ]]; then
    wait_env=(-e "MVMC_RS_TENFERRO_SPIN_POLLS=$MVMC_RS_TENFERRO_SPIN_POLLS")
fi
repo=$(git rev-parse --show-toplevel)
mkdir -p "$out"
out=$(cd "$out" && pwd)
if [[ -e "$out/inputs" ]]; then
    echo "output already contains inputs: $out" >&2
    exit 2
fi
mkdir -p "$out/inputs/opt" "$out/inputs/phys" "$out/results"
printf 'cpuset=%s\ninner_threads=%s\n' "$cpu_set" "$inner_threads" > "$out/config.env"
printf 'spin_polls=%s\n' "${MVMC_RS_TENFERRO_SPIN_POLLS:-upstream-default}" >> "$out/config.env"
cp "$repo"/benchmark/hubbard_chain/inputs/hubbard_chain_L32/*.def "$out/inputs/opt/"
cp "$repo"/benchmark/physcal/inputs/hubbard_chain_L32/*.def "$out/inputs/phys/"
cp "$repo/benchmark/physcal/inputs/hubbard_chain_L32/zqp_opt.dat" "$out/inputs/phys/"
sed -i -E 's/^NDataQtySmp[[:space:]].*/NDataQtySmp    20/' "$out/inputs/phys/modpara.def"
sha256sum "$out"/inputs/{opt,phys}/* > "$out/input-sha256.txt"
for mode in opt phys; do
    for repetition in 0 1 2 3; do
        if (( repetition % 2 == 0 )); then settings='1 4'; else settings='4 1'; fi
        for threads in $settings; do
            docker run --rm --platform=linux/amd64 --cpuset-cpus="$cpu_set" --shm-size=1g \
                -v "$out/inputs:/input:ro" -v "$out/results:/results" -v "$volume:/target" \
                -w "/input/$mode" \
                -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e MVMC_RS_INNER_THREADS="$inner_threads" \
                -e RAYON_NUM_THREADS=4 -e MVMC_RS_TENFERRO_THREADS="$threads" \
                "${wait_env[@]}" \
                -e MVMC_RS_SR_BACKEND=tenferro -e "BENCH_MODE=$mode" \
                -e "BENCH_DEST=/results/${mode}-t${threads}-r${repetition}" \
                --entrypoint bash "$image" -lc '
                    TIMEFORMAT="wall=%R user=%U sys=%S"
                    if [[ "$BENCH_MODE" == opt ]]; then
                        time timeout --kill-after=5s 120s /opt/mpich/bin/mpirun -n 4 /target/release/mvmc \
                            /input/opt/namelist.def --nsteps 20 --nsmp 1 --initial-def none --out-dir "$BENCH_DEST"
                    else
                        time timeout --kill-after=5s 120s /opt/mpich/bin/mpirun -n 4 /target/release/mvmc \
                            /input/phys/namelist.def --physcal /input/phys/zqp_opt.dat --out-dir "$BENCH_DEST"
                    fi
                ' > "$out/${mode}-t${threads}-r${repetition}.txt" 2>&1
        done
    done
done

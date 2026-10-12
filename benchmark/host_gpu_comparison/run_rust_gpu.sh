#!/usr/bin/env bash
# Rust GPU whole-runner E2E (Opt + PhysCal) on the host (issue #544).
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=env.sh
source "$here/env.sh"
# shellcheck source=lib.sh
source "$here/lib.sh"

[ -x "$RUST_GPU_BIN" ] || {
    echo "missing $RUST_GPU_BIN; build with: (cd gpu/mvmc-gpu-cuda && cargo build --release --locked --bin mvmc-cuda)" >&2
    exit 1
}

export LD_LIBRARY_PATH="$CUDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MVMC_RS_SR_BACKEND=cuda
export MVMC_RS_SAMPLER_BACKEND=cuda
export MVMC_RS_MEASURE_BACKEND=cuda
export MVMC_RS_MEASURE_PF_BACKEND=cuda
export MVMC_C_TIMER=1

case=rust-gpu
dir="$RESULT_DIR/$case"
mkdir -p "$dir"
csv="$dir/$case.csv"
printf 'case,phase,rep,wall_s,all_s\n' >"$csv"
record_env "$dir/environment.txt"

# PhysCal fixed-parameter checkpoint: a two-point window writes parameter triples.
ckpt=$(make_checkpoint "$RUST_GPU_BIN" "$dir/checkpoint-input" "$dir/checkpoint-out")
echo "checkpoint: $ckpt"

opt_in="$dir/opt/input"
mkdir -p "$dir/opt"
prepare_input "$opt_in" "$SAMPLES" 0 1
for rep in $(seq 1 $((WARMUPS + REPS))); do
    tag=warmup
    [ "$rep" -gt "$WARMUPS" ] && tag=measure
    run_timed "$csv" "$case" "opt-$tag" "$rep" "$dir/opt/run-$rep" \
        bash -c "cd '$opt_in' && '$RUST_GPU_BIN' namelist.def --nsteps $STEPS --nsmp 1 --seed $SEED --out-dir '$dir/opt/run-$rep'"
done

phys_in="$dir/phys/input"
mkdir -p "$dir/phys"
prepare_input "$phys_in" "$SAMPLES" 1 1
for rep in $(seq 1 $((WARMUPS + REPS))); do
    tag=warmup
    [ "$rep" -gt "$WARMUPS" ] && tag=measure
    run_timed "$csv" "$case" "physcal-$tag" "$rep" "$dir/phys/run-$rep" \
        bash -c "cd '$phys_in' && '$RUST_GPU_BIN' namelist.def --physcal '$ckpt' --out-dir '$dir/phys/run-$rep'"
done

echo "results: $csv"

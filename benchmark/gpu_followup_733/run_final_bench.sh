#!/bin/bash
set -euo pipefail
repo=$(git rev-parse --show-toplevel)
cd "$repo"
if [ ! -d benchmark/gpu_followup_733/inputs/complex-phys ]; then
 uv run --no-project python benchmark/gpu_followup_733/prepare_inputs.py benchmark/gpu_followup_733/inputs/complex-phys
fi
for kind in complex-opt complex-phys real-opt real-phys; do
 mode=${kind#*-}
 case "$kind" in
 complex-opt) input="$repo/gpu/mvmc-gpu-cuda/tests/fixtures/complex/hubbard_complex_L128/namelist.def";;
 complex-phys) input="$repo/benchmark/gpu_followup_733/inputs/complex-phys/namelist.def";;
 real-opt) input="$repo/benchmark/cpu_scaling_l128/inputs/opt/namelist.def";;
 real-phys) input="$repo/benchmark/cpu_scaling_l128/inputs/phys/namelist.def";;
 esac
 steps=20
 if [ "$mode" = phys ]; then steps=1; fi
 taskset -c 0 uv run --no-project python benchmark/pfaffian_faer/run_gpu_e2e.py --before /tmp/mvmc733-baseline --after "$repo" --before-volume mvmc-733-before-cuda-target --after-volume mvmc-731-cuda-target --input "$input" --output "benchmark/gpu_followup_733/final-$kind" --mode "$mode" --image mvmc731-cuda-toolchain:local --driver-cache-volume mvmc-733-driver-cache --device 1 --cpu 17 --walkers 320 --steps "$steps" --warmup 128 --interval 128 --reps 3
done

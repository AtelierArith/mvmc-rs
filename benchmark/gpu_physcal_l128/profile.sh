#!/usr/bin/env bash
# Optional diagnostic, never called by Cargo or the uninstrumented speed driver.
set -euo pipefail
bench_input=$(realpath "${1:?input directory}")
bench_binary=$(realpath "${2:?mvmc-cuda binary}")
bench_out=$(realpath -m "${3:?fresh output directory}")
bench_nsys=$(realpath "${4:?nsys CLI}")
bench_importer=${5:-}
if [[ -n "$bench_importer" ]]; then bench_importer=$(realpath "$bench_importer"); fi
test ! -e "$bench_out"
mkdir -p "$bench_out"
cp -a "$bench_input/." "$bench_out/"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 MVMC_RS_INNER_THREADS=1
export MVMC_RS_SR_BACKEND=cuda MVMC_RS_MEASURE_PF_BACKEND=cuda MVMC_RS_MEASURE_BACKEND=cuda MVMC_RS_SAMPLER_BACKEND=cuda MVMC_RS_CUDA_RNG=sfmt
cd "$bench_out"
"$bench_nsys" --version > profiler-version.txt
sha256sum "$bench_binary" namelist.def modpara.def zqp_opt.dat > source-sha256.txt
"$bench_nsys" profile --sample=none --cpuctxsw=none --trace=cuda,osrt \
 --output="$bench_out/trace" taskset -c 0 "$bench_binary" namelist.def zqp_opt.dat > profile.log 2>&1
if [[ ! -f trace.nsys-rep ]]; then
 test -n "$bench_importer"
 "$bench_importer" -i trace.qdstrm -o trace.nsys-rep > import.log 2>&1
fi
"$bench_nsys" stats --report cudaapisum,gpukernsum,gpumemtimesum,gpumemsizesum \
 --format csv trace.nsys-rep > stats.csv 2> stats.stderr

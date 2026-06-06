#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

timestamp="$(date +%Y%m%d_%H%M%S)"
out_dir="${OUT_DIR:-benchmark/pfapack_compare/results/$timestamp}"
mkdir -p "$out_dir"

manifest="benchmark/pfapack_compare/Cargo.toml"
julia_project="extern/PfaPack.jl"
julia_bench="benchmark/pfapack_compare/bench_julia.jl"

run_julia() {
  local threads="$1"
  local out="$out_dir/julia_openblas${threads}.csv"

  echo "== Julia / OPENBLAS_NUM_THREADS=$threads =="
  OPENBLAS_NUM_THREADS="$threads" \
    julia --project="$julia_project" "$julia_bench" | tee "$out"
}

run_rust_scalar() {
  local threads="$1"
  local out="$out_dir/rust_scalar_openblas${threads}.csv"
  local target_dir="$out_dir/target_scalar_openblas${threads}"

  echo "== Rust scalar / OPENBLAS_NUM_THREADS=$threads =="
  OPENBLAS_NUM_THREADS="$threads" \
  CARGO_TARGET_DIR="$target_dir" \
    cargo run --release --manifest-path "$manifest" --offline | tee "$out"
}

run_rust_simd() {
  local threads="$1"
  local out="$out_dir/rust_simd_openblas${threads}.csv"
  local target_dir="$out_dir/target_simd_openblas${threads}"

  echo "== Rust SIMD / OPENBLAS_NUM_THREADS=$threads =="
  OPENBLAS_NUM_THREADS="$threads" \
  CARGO_TARGET_DIR="$target_dir" \
    cargo run --release --manifest-path "$manifest" \
      --features pfapack/simd-backend --offline | tee "$out"
}

run_rust_blas() {
  local threads="$1"
  local out="$out_dir/rust_blas_openblas${threads}.csv"
  local target_dir="$out_dir/target_blas_openblas${threads}"

  echo "== Rust BLAS / OPENBLAS_NUM_THREADS=$threads =="
  OPENBLAS_NUM_THREADS="$threads" \
  CARGO_TARGET_DIR="$target_dir" \
    cargo run --release --manifest-path "$manifest" \
      --features pfapack/blas-backend --offline | tee "$out"
}

append_with_config() {
  local config="$1"
  local variant="$2"
  local file="$3"

  awk -F, -v config="$config" -v variant="$variant" '
    NR == 1 {
      next
    }
    {
      row_variant = variant
      if (variant == "from_impl") {
        row_variant = $1
      }
      print config "," row_variant "," $0
    }
  ' "$file" >> "$out_dir/all_results.csv"
}

generate_report() {
  local report="$out_dir/report.md"

  {
    echo "# PfaPack Benchmark Report"
    echo
    echo "- Date: $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "- Output directory: \`$out_dir\`"
    echo "- Metric: median wall-clock time in milliseconds"
    echo "- Iteration behavior: each timed iteration copies/clones the input matrix first"
    echo "- Julia LV uses \`julia_zsktf2_turbo!\` and is only available for complex LTL-derived operations"
    echo "- Fastest value in each row is marked with \`★\`"
    echo
    echo "## Commands"
    echo
    echo '```sh'
    for threads in 1 4; do
      echo "OPENBLAS_NUM_THREADS=$threads julia --project=$julia_project $julia_bench"
      echo "OPENBLAS_NUM_THREADS=$threads cargo run --release --manifest-path $manifest --offline"
      echo "OPENBLAS_NUM_THREADS=$threads cargo run --release --manifest-path $manifest --features pfapack/simd-backend --offline"
      echo "OPENBLAS_NUM_THREADS=$threads cargo run --release --manifest-path $manifest --features pfapack/blas-backend --offline"
    done
    echo '```'
    echo

    awk -F, '
      BEGIN {
        configs[1] = "openblas_threads_1"
        configs[2] = "openblas_threads_4"
        config_titles["openblas_threads_1"] = "OPENBLAS_NUM_THREADS=1"
        config_titles["openblas_threads_4"] = "OPENBLAS_NUM_THREADS=4"
      }
      NR > 1 {
        key = $1 SUBSEP $4 SUBSEP $5 SUBSEP $6
        value[key SUBSEP $2] = $7
      }
      function marked(v, min_v) {
        if (v == "") {
          return "-"
        }
        if ((v + 0) == min_v) {
          return "★ **" v "**"
        }
        return v
      }
      function consider(v, current) {
        if (v == "") {
          return current
        }
        if (current == "" || (v + 0) < current) {
          return v + 0
        }
        return current
      }
      END {
        kinds[1] = "real"
        kinds[2] = "complex"
        ns[1] = "32"
        ns[2] = "64"
        ns[3] = "128"
        ns[4] = "256"
        ops[1] = "pfaffian_ltl"
        ops[2] = "ltl"
        ops[3] = "ltl_utu2pfa"
        ops[4] = "ltl_utu2inv"

        for (c = 1; c <= 2; c++) {
          config = configs[c]
          print "## " config_titles[config]
          print ""
          print "| kind | n | op | Julia ms | Julia LV ms | Rust scalar ms | Rust SIMD ms | Rust BLAS ms |"
          print "|---|---:|---|---:|---:|---:|---:|---:|"
          for (ki = 1; ki <= 2; ki++) {
            kind = kinds[ki]
            for (ni = 1; ni <= 4; ni++) {
              n = ns[ni]
              for (oi = 1; oi <= 4; oi++) {
                op = ops[oi]
                base = config SUBSEP kind SUBSEP n SUBSEP op
                julia = value[base SUBSEP "julia"]
                julia_lv = value[base SUBSEP "julia_lv"]
                rust_scalar = value[base SUBSEP "rust_scalar"]
                rust_simd = value[base SUBSEP "rust_simd"]
                rust_blas = value[base SUBSEP "rust_blas"]
                min = ""
                min = consider(julia, min)
                min = consider(julia_lv, min)
                min = consider(rust_scalar, min)
                min = consider(rust_simd, min)
                min = consider(rust_blas, min)
                printf("| %s | %s | %s | %s | %s | %s | %s | %s |\n",
                  kind, n, op,
                  marked(julia, min),
                  marked(julia_lv, min),
                  marked(rust_scalar, min),
                  marked(rust_simd, min),
                  marked(rust_blas, min))
              }
            }
          }
          print ""
        }
      }
    ' "$out_dir/all_results.csv"
  } > "$report"
}

printf 'config,variant,impl,kind,n,op,median_ms\n' > "$out_dir/all_results.csv"

for threads in 1 4; do
  run_julia "$threads"
  append_with_config "openblas_threads_${threads}" "from_impl" "$out_dir/julia_openblas${threads}.csv"

  run_rust_scalar "$threads"
  append_with_config "openblas_threads_${threads}" "rust_scalar" "$out_dir/rust_scalar_openblas${threads}.csv"

  run_rust_simd "$threads"
  append_with_config "openblas_threads_${threads}" "rust_simd" "$out_dir/rust_simd_openblas${threads}.csv"

  run_rust_blas "$threads"
  append_with_config "openblas_threads_${threads}" "rust_blas" "$out_dir/rust_blas_openblas${threads}.csv"
done

generate_report

echo "== Results =="
echo "$out_dir"
echo "$out_dir/all_results.csv"
echo "$out_dir/report.md"

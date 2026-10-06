#!/usr/bin/env bash
# Reproduce the Rust-vs-Julia PhysCal benchmark on the Hubbard-chain inputs
# (issue #442, AtelierArith/mvmc-rs PR #444).
#
# Usage:
#   scripts/run_physcal_benchmark.sh [reps] [warmups] [threads]
#
#   reps     measured repetitions per side   (default: 3)
#   warmups  warm-up repetitions per side    (default: 1)
#   threads  BLAS / OpenMP / Julia threads   (default: 1)
#
# Outputs:
#   target/bench/physcal_chain.csv              four `physcal_ref` fixtures
#   target/bench/physcal_hubbard.csv            L16/L24/L32 raw timings
#   target/bench/physcal_hubbard_report.md      L16/L24/L32 report + observables
#
# Reference-runner prerequisites (submodules, Julia 1.13.1, SFMT/PfaPack native
# libraries, OpenBLAS/LAPACK) are documented in docs/DEVELOPMENT.md. The
# `bench-physcal` tasks also compare the measured Green-function observables, so
# they fail loudly on any shape or numeric mismatch.

set -euo pipefail

REPS=${1:-3}
WARMUPS=${2:-1}
THREADS=${3:-1}

cd "$(dirname "$0")/.."

echo "=== Rust vs Julia PhysCal benchmark (reps=$REPS warmups=$WARMUPS threads=$THREADS) ==="

echo
echo "--- four physcal_ref fixtures (real / complex / fsz / hubbard+lanczos) ---"
cargo run -p xtask -- bench-physcal --reps "$REPS" --warmups "$WARMUPS" --threads "$THREADS"

echo
echo "--- Hubbard chain L16/L24/L32 (NDataQtySmp=100) ---"
cargo run -p xtask -- bench-physcal-hubbard --reps "$REPS" --warmups "$WARMUPS" --threads "$THREADS"

echo
echo "raw timings : target/bench/physcal_chain.csv"
echo "              target/bench/physcal_hubbard.csv"
echo "report      : target/bench/physcal_hubbard_report.md"
echo "archived    : benchmark/physcal/results/"

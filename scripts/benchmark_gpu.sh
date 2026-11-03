#!/usr/bin/env bash
# Shared L128 Opt/PhysCal: Julia and Rust on one NVIDIA GPU.
set -euo pipefail
benchmark_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
command -v uv >/dev/null || { echo 'uv is required: https://docs.astral.sh/uv/' >&2; exit 1; }
exec uv run --no-project python "$benchmark_root/scripts/benchmark/run.py" gpu "$@"

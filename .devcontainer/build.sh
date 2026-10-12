#!/usr/bin/env bash
set -euo pipefail

if [[ $# == 0 ]]; then
    echo 'Usage: bash .devcontainer/build.sh COMMAND [ARG ...]' >&2
    exit 2
fi
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
jobs=$(bash "$script_dir/build-jobs.sh")
export MVMC_BUILD_JOBS="$jobs"

# Native build scripts launched by Cargo inherit its token-based jobserver.
# Replacing those tokens with -jN would oversubscribe nested builds. CMake can
# consume the inherited make jobserver when no parallel level is forced.
if [[ -n ${CARGO_MAKEFLAGS:-} ]]; then
    export MAKEFLAGS="$CARGO_MAKEFLAGS"
    unset CARGO_BUILD_JOBS CMAKE_BUILD_PARALLEL_LEVEL
    export JULIA_NUM_PRECOMPILE_TASKS=1
elif [[ ${MAKEFLAGS:-} == *--jobserver-* ]]; then
    unset CARGO_BUILD_JOBS CMAKE_BUILD_PARALLEL_LEVEL
    export JULIA_NUM_PRECOMPILE_TASKS=1
elif [[ ${1##*/} == cargo ]]; then
    # Cargo creates the top-level jobserver. Its build scripts receive tokens
    # via CARGO_MAKEFLAGS; an inherited -jN/parallel level would create another
    # native worker pool independently of those tokens.
    export CARGO_BUILD_JOBS="$jobs"
    unset MAKEFLAGS CMAKE_BUILD_PARALLEL_LEVEL
    export JULIA_NUM_PRECOMPILE_TASKS=1
else
    export CARGO_BUILD_JOBS="$jobs"
    export CMAKE_BUILD_PARALLEL_LEVEL="$jobs"
    export JULIA_NUM_PRECOMPILE_TASKS="$jobs"
    export MAKEFLAGS="${MAKEFLAGS:+$MAKEFLAGS }-j$jobs"
fi

# Deliberately preserve JULIA_NUM_THREADS and BLAS/OpenMP runtime settings.
exec "$@"

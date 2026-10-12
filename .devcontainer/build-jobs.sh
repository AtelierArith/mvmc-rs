#!/usr/bin/env sh
set -eu

# nproc respects process CPU affinity. Runtime numerical thread limits must not
# turn a single-thread benchmark setting into a single-job compiler setting.
jobs=${MVMC_BUILD_JOBS:-$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)}
case "$jobs" in
    ''|0*|*[!0-9]*)
        echo 'MVMC_BUILD_JOBS must be a positive decimal integer (without leading zeros).' >&2
        exit 2
        ;;
esac
printf '%s\n' "$jobs"

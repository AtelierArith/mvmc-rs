#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
fixture=$(mktemp -d)
trap 'rm -rf "$fixture"' EXIT
original_path=$PATH
cat > "$fixture/nproc" <<'NPROC'
#!/usr/bin/env sh
[ -z "${OMP_NUM_THREADS+x}" ] && [ -z "${OMP_THREAD_LIMIT+x}" ] || exit 9
printf '7\n'
NPROC
chmod +x "$fixture/nproc"
export PATH="$fixture:$PATH"
unset MVMC_BUILD_JOBS CARGO_MAKEFLAGS MAKEFLAGS CARGO_BUILD_JOBS CMAKE_BUILD_PARALLEL_LEVEL
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
[[ $(sh "$script_dir/build-jobs.sh") == 7 ]]
[[ $(MVMC_BUILD_JOBS=3 sh "$script_dir/build-jobs.sh") == 3 ]]
for invalid in 0 -1 01 text '2 3'; do
    if MVMC_BUILD_JOBS="$invalid" sh "$script_dir/build-jobs.sh" >/dev/null 2>&1; then
        echo "Accepted invalid build jobs: $invalid" >&2
        exit 1
    fi
done
bash "$script_dir/build.sh" bash -c '
    set -e
    [[ $MVMC_BUILD_JOBS == 7 && $CARGO_BUILD_JOBS == 7 && $CMAKE_BUILD_PARALLEL_LEVEL == 7 ]]
    [[ $JULIA_NUM_PRECOMPILE_TASKS == 7 && $MAKEFLAGS == -j7 ]]
    [[ $JULIA_NUM_THREADS == 1 && $OMP_NUM_THREADS == 1 && $OPENBLAS_NUM_THREADS == 1 ]]
'
MVMC_BUILD_JOBS=3 bash "$script_dir/build.sh" bash -c '
    set -e
    [[ $CARGO_BUILD_JOBS == 3 && $CMAKE_BUILD_PARALLEL_LEVEL == 3 && $JULIA_NUM_PRECOMPILE_TASKS == 3 ]]
'
cat > "$fixture/cargo" <<'CARGO'
#!/usr/bin/env bash
set -euo pipefail
[[ $CARGO_BUILD_JOBS == $EXPECTED_JOBS ]]
[[ ! -v MAKEFLAGS && ! -v CMAKE_BUILD_PARALLEL_LEVEL ]]
[[ $JULIA_NUM_PRECOMPILE_TASKS == 1 ]]
[[ $JULIA_NUM_THREADS == 1 && $OMP_NUM_THREADS == 1 && $OPENBLAS_NUM_THREADS == 1 ]]
[[ $1 == build && $2 == --locked ]]
CARGO
chmod +x "$fixture/cargo"
EXPECTED_JOBS=7 MAKEFLAGS=-j99 CMAKE_BUILD_PARALLEL_LEVEL=99 \
    bash "$script_dir/build.sh" cargo build --locked
EXPECTED_JOBS=3 MVMC_BUILD_JOBS=3 MAKEFLAGS=-j99 CMAKE_BUILD_PARALLEL_LEVEL=99 \
    bash "$script_dir/build.sh" "$fixture/cargo" build --locked
for origin in cargo make; do
    export CARGO_BUILD_JOBS=99 CMAKE_BUILD_PARALLEL_LEVEL=99
    if [[ $origin == cargo ]]; then
        export CARGO_MAKEFLAGS='-j --jobserver-auth=3,4'
        export MAKEFLAGS=-j99
    else
        unset CARGO_MAKEFLAGS
        export MAKEFLAGS='-j --jobserver-auth=3,4'
    fi
    bash "$script_dir/build.sh" bash -c '
        set -e
        [[ $MAKEFLAGS == "-j --jobserver-auth=3,4" ]]
        [[ ! -v CARGO_BUILD_JOBS && ! -v CMAKE_BUILD_PARALLEL_LEVEL ]]
        [[ $JULIA_NUM_PRECOMPILE_TASKS == 1 && $JULIA_NUM_THREADS == 1 ]]
    '
done
if bash "$script_dir/build.sh" >/dev/null 2>&1; then
    echo 'Accepted a missing build command' >&2
    exit 1
fi
if command -v taskset >/dev/null 2>&1; then
    first_cpu=$(taskset -pc $$ | sed 's/.*: //; s/[-,].*//')
    [[ $(PATH="$original_path" taskset -c "$first_cpu" sh "$script_dir/build-jobs.sh") == 1 ]]
fi
printf 'Build parallelism policy checks passed.\n'

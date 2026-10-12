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
# Exercise the actual installer's budget, receipt and make command lines with a
# recording make stub. No download, configure or native compilation is run.
repo=$(cd "$script_dir/.." && pwd)
installer="$repo/scripts/issue234/native/install-pmi1-mpich.sh"
cat > "$fixture/make" <<'MAKE'
#!/usr/bin/env sh
printf '%s\n' "$*" >> "$MAKE_ARGS_LOG"
MAKE
chmod +x "$fixture/make"
build_section=$(awk '
    /^build_jobs=/ || /^sha256sum.*source[.]sha256/ ||
    /^printf.*jobs=.*settings[.]txt/ || /^timeout.*make/ {print}
' "$installer")
tool_names=$(sed -n 's/^for name in \(.*\);do$/\1/p' "$installer" | head -1)
for tool in sh env nproc; do
    [[ " $tool_names " == *" $tool "* ]]
done
for selected in default 3; do
    proof="$fixture/proof-$selected"
    mkdir "$proof"
    if [[ $selected == default ]]; then
        unset MVMC_BUILD_JOBS
        expected=7
    else
        export MVMC_BUILD_JOBS=$selected
        expected=$selected
    fi
    repo="$repo" here="${installer%/*}" proof="$proof" prefix="$fixture/prefix" \
        version=4.2.0 digest=fixture MAKE_ARGS_LOG="$proof/make.args" \
        bash -e -c "$build_section"
    [[ $(cat "$proof/make.args") == "-j$expected"$'\n'"-j$expected install" ]]
    grep -F "jobs=$expected" "$proof/settings.txt" >/dev/null
    grep -F "$script_dir/build-jobs.sh" "$proof/source.sha256" >/dev/null
    sha256sum -c --quiet "$proof/source.sha256"
done
unset MVMC_BUILD_JOBS
if command -v taskset >/dev/null 2>&1; then
    first_cpu=$(taskset -pc $$ | sed 's/.*: //; s/[-,].*//')
    [[ $(PATH="$original_path" taskset -c "$first_cpu" sh "$script_dir/build-jobs.sh") == 1 ]]
fi
printf 'Build parallelism policy checks passed.\n'

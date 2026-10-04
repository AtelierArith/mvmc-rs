#!/bin/bash
# Existing Linux default CI job only. No local supervisor/cache redesign.
set -euo pipefail
[[ $- == *p* ]] || { printf 'privileged nonlogin Bash required\n' >&2; exit 64; }
for name in BASH_ENV ENV NODE_OPTIONS NODE_PATH; do
    [[ ! -v $name ]] || { printf 'unexpected environment key %s\n' "$name" >&2; exit 64; }
done
test "$#" = 1
out=$1
[[ $out = /* ]] && test ! -e "$out"
mkdir "$out"
finish() {
    prior=$?; trap - EXIT; set +e
    source_post=NOT_STARTED; artifact_post=NOT_STARTED
    if test -s "$out/source.before.json"; then
        "$node" scripts/issue176/pfapack-ci-source.mjs "$out/source.after.json" > "$out/source.post.stdout" 2> "$out/source.post.stderr"
        source_post=$?
        if test "$source_post" = 0; then cmp "$out/source.before.json" "$out/source.after.json"; source_post=$?; fi
    fi
    if test -s "$out/selected.json.sha256"; then
        sha256sum -c --quiet "$out/selected.json.sha256" > "$out/selected.post.stdout" 2> "$out/selected.post.stderr"
        artifact_post=$?
    fi
    printf '%s\n' "$source_post" > "$out/source.post.status"
    printf '%s\n' "$artifact_post" > "$out/artifact.post.status"
    final=$prior
    if test "$final" = 0 && { test "$source_post" != 0 || test "$artifact_post" != 0; }; then final=74; fi
    printf 'primary=%s source_post=%s artifact_post=%s terminal=%s scientific_acceptance=false\n' "$prior" "$source_post" "$artifact_post" "$final" > "$out/terminal.txt"
    exit "$final"
}
node=''
trap finish EXIT
node=$(command -v node)
test "$(git rev-parse HEAD)" = "$GITHUB_SHA"
printf '%s\n' "$GITHUB_SHA" > "$out/head.txt"
test "$(df -Pk . | awk 'NR==2{print $4}')" -ge 655360
# File-size guard includes the bounded raw stream; child logs have narrower
# prospective 1MiB caps in acquire.mjs. No workspace test assertion changes.
ulimit -f 131072
export CARGO_BUILD_JOBS=2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
"$node" scripts/issue176/pfapack-ci-source.mjs "$out/source.before.json"
controls_status=0
timeout -k 5s 60s "$node" scripts/pfapack-factor-acquisition/controls.mjs > "$out/controls.stdout" 2> "$out/controls.stderr" || controls_status=$?
printf '%s\n' "$controls_status" > "$out/controls.status"
test "$controls_status" = 0
selection='test(=ltl::factor_acquisition::independent_c59_four_factor_inverse_stage_stream)'
list_status=0
timeout -k 5s 1800s cargo nextest list -p pfapack --lib --locked --cargo-profile ci \
    --run-ignored ignored-only -E "$selection" --message-format json > "$out/list.json" 2> "$out/list.stderr" || list_status=$?
printf '%s\n' "$list_status" > "$out/list.status"
test "$list_status" = 0
"$node" scripts/pfapack-factor-acquisition/bind-selection.mjs "$out/list.json" "$PWD" "$out/selected.json"
# A single selected test call produces all four cases through its FD3 pipe.
# Outer timeout supervises the Node process and its non-detached child group.
# Acquire preserves the first failure and does not rerun any case.
acquisition_status=0
timeout -k 10s 1200s "$node" scripts/pfapack-factor-acquisition/acquire.mjs "$out/selected.json" "$out" \
    > "$out/analysis.stdout" 2> "$out/analysis.stderr" || acquisition_status=$?
printf '%s\n' "$acquisition_status" > "$out/acquisition.status"
test "$acquisition_status" = 0

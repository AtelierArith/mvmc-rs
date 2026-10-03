#!/usr/bin/env bash
# Explicit developer verification; never called by Cargo tests/build scripts.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
for tool in cargo jq sha256sum tar awk; do
    command -v "$tool" >/dev/null || { echo "Unsupported: missing $tool" >&2; exit 1; }
done
artifact=$(mktemp -d "${TMPDIR:-/tmp}/mvmc-threaded-182.XXXXXX")
echo "artifact=$artifact"
terminal() {
    local status=$?
    trap - EXIT
    set +e
    if declare -F source_hashes >/dev/null; then
        source_hashes > "$artifact/source.after.sha256" || status=1
        fixture_hashes > "$artifact/fixtures.after.sha256" || status=1
        if [[ -d $artifact/fixtures ]]; then
            (cd "$artifact/fixtures"; find . -type f -print0 | LC_ALL=C sort -z |
                xargs -0 -r sha256sum) > "$artifact/frozen-fixtures.after.sha256"
        fi
    fi
    printf 'exit_status=%s\nfinished_utc=%s\n' "$status" "$(date -u +%FT%TZ)" > "$artifact/terminal.txt"
    echo "terminal exit=$status artifact=$artifact"
    exit "$status"
}
trap terminal EXIT
gate=runner_workers_preserve_rng_configurations_direct_store_cg_and_physcal
filter="test($gate)"
if [[ -n ${MVMC_RS_THREADED_FILTER:-} ]]; then
    filter="($filter) & (${MVMC_RS_THREADED_FILTER})"
fi
fixture_root=${MVMC_RS_THREADED_FIXTURE_ROOT:-$repo}
paths=(tests/fixtures/ctest_model_prefixes/provenance.txt)
paths+=(tests/fixtures/ctest_model_prefixes/general_rbm_cmp_cg/step-1
    tests/fixtures/reviewed_cg_62b/canonical_general_rbm
    extern/Julia-mVMC/test/integration/reference/general_rbm_cmp/inputs)
for model in heisenberg_chain_real heisenberg_chain_cmp heisenberg_chain_fsz hubbard_chain_real; do
    paths+=("tests/fixtures/ctest_model_prefixes/$model/step-1")
    paths+=("extern/Julia-mVMC/test/integration/reference/$model/inputs")
    paths+=("extern/Julia-mVMC/test/integration/reference/$model/physcal_ref")
done
# Hash the exact immutable inputs consumed by the long Rust-only matrix.
fixture_hashes() {
    (cd "$fixture_root" || { echo "MissingFixture: $fixture_root" >&2; return 1; }; for path in "${paths[@]}"; do
        if [[ ! -e $path ]]; then echo "MissingFixture: $fixture_root/$path" >&2; return 1; fi
        if [[ -f $path ]]; then sha256sum "$path"; else
            find "$path" -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum
        fi
    done)
}
source_hashes() {
    git ls-files -co --exclude-standard -z -- Cargo.toml Cargo.lock crates scripts/verify_threaded_issue182.sh |
        LC_ALL=C sort -zu | xargs -0 -r sha256sum
}
{
    printf 'started_utc=%s\nsource_head=%s\n' "$(date -u +%FT%TZ)" "$(git rev-parse HEAD)"
    printf 'profile=test-fast\nfeatures=default\nrequested_workers=1,2,4\nthreshold=32\nsizes=31,32,33\nmpi_world=1 (non-MPI)\n'
    printf 'oracle_execution=none; checked-in reference fixtures only\nfilter=%s\nfixture_root=%s\n' "$filter" "$fixture_root"
    printf 'actual_activation=run.log actual runner observation; capacity is not activation\n'
    uname -a
    rustc -Vv
    cargo nextest --version
    git submodule status extern/Julia-mVMC extern/mVMC-1.3.0
    [[ ! -f extern/Julia-mVMC/Manifest-v1.13.toml ]] || sha256sum extern/Julia-mVMC/Manifest-v1.13.toml
    printf 'BLAS runtime not inferred from environment; binary linkage captured separately\n'
} > "$artifact/metadata.txt"
source_hashes > "$artifact/source.before.sha256"
git ls-files -co --exclude-standard -z -- Cargo.toml Cargo.lock crates scripts/verify_threaded_issue182.sh |
    LC_ALL=C sort -zu | tar --null -T - -cf "$artifact/source.tar"
fixture_hashes > "$artifact/fixtures.before.sha256" 2> "$artifact/preflight.log" || {
    cat "$artifact/preflight.log" >&2
    exit 1
}
tar -C "$fixture_root" -cf "$artifact/fixtures.tar" "${paths[@]}"
mkdir "$artifact/fixtures"
tar -C "$artifact/fixtures" -xf "$artifact/fixtures.tar"
(cd "$artifact/fixtures"; find . -type f -print0 | LC_ALL=C sort -z |
    xargs -0 -r sha256sum) > "$artifact/frozen-fixtures.before.sha256"
archive="$artifact/tests.tar.zst"
if [[ -n ${MVMC_RS_THREADED_ARCHIVE:-} ]]; then
    # Replaying a supplied binary never claims it was built from current sources.
    cp "$MVMC_RS_THREADED_ARCHIVE" "$archive"
    printf 'archive_origin=supplied; current source association unverified\n' >> "$artifact/metadata.txt"
else
    export CARGO_TARGET_DIR=${MVMC_RS_THREADED_TARGET_DIR:-$artifact/target}
    printf 'archive_origin=built here\ntarget=%s\n' "$CARGO_TARGET_DIR" >> "$artifact/metadata.txt"
    cargo nextest archive --locked -p mvmc-core --cargo-profile test-fast --test threaded_issue182 \
        --archive-file "$archive" > "$artifact/build.log" 2>&1
fi
sha256sum "$archive" "$artifact/fixtures.tar" "$artifact/source.tar" > "$artifact/artifacts.sha256"
cargo nextest list --archive-file "$archive" --run-ignored only --message-format json \
    -E "$filter" > "$artifact/selection.json" 2> "$artifact/list.log"
selected=$(jq '[.["rust-suites"][] | .testcases | to_entries[] |
    select(.value["filter-match"].status == "matches")] | length' "$artifact/selection.json")
printf 'selected=%s\n' "$selected" >> "$artifact/metadata.txt"
printf 'command=MVMC_RS_THREADED_182=1 MVMC_RS_THREADED_FIXTURE_ROOT=%q cargo nextest run --archive-file %q --run-ignored only -E %q --no-fail-fast --retries 0 --success-output immediate --failure-output immediate\n' \
    "$artifact/fixtures" "$archive" "$filter" >> "$artifact/metadata.txt"
if [[ $selected != 1 ]]; then
    echo "NotRun: expected exactly one complete runner matrix; selected=$selected" >&2
    exit 1
fi
# Record linkage when available, without treating ldd as a provider/version oracle.
binary=$(jq -r '.["rust-suites"][] | .["binary-path"]' "$artifact/selection.json")
if command -v ldd >/dev/null; then ldd "$binary" > "$artifact/linkage.txt" 2>&1 || true; fi
set +e
MVMC_RS_THREADED_182=1 MVMC_RS_THREADED_FIXTURE_ROOT="$artifact/fixtures" \
    cargo nextest run --archive-file "$archive" --run-ignored only -E "$filter" \
    --no-fail-fast --retries 0 --success-output immediate --failure-output immediate \
    > "$artifact/run.log" 2>&1
run_status=$?
set -e
printf 'nextest_exit=%s\n' "$run_status" >> "$artifact/metadata.txt"
source_hashes > "$artifact/source.after.sha256"
fixture_hashes > "$artifact/fixtures.after.sha256"
cmp "$artifact/source.before.sha256" "$artifact/source.after.sha256" || {
    echo 'SourceChanged: source association is not stable' >&2; exit 1;
}
cmp "$artifact/fixtures.before.sha256" "$artifact/fixtures.after.sha256" || {
    echo 'FixtureChanged: reference inputs changed' >&2; exit 1;
}
(cd "$artifact/fixtures"; find . -type f -print0 | LC_ALL=C sort -z |
    xargs -0 -r sha256sum) > "$artifact/frozen-fixtures.after.sha256"
cmp "$artifact/frozen-fixtures.before.sha256" "$artifact/frozen-fixtures.after.sha256" || {
    echo 'FixtureChanged: frozen reference inputs changed' >&2; exit 1;
}
[[ $run_status == 0 ]] || exit "$run_status"
# Validate actual entered QP/transfer jobs, not install calls or pool capacity.
awk '
 /actual runner observation workers=/ {
     line=$0; sub(/^.*workers=/,"",line); split(line,a,":"); w=a[1]+0
     qp=$0; sub(/^.*parallel_qp_items: /,"",qp); sub(/,.*/,"",qp)
     term=$0; sub(/^.*parallel_term_items: /,"",term); sub(/,.*/,"",term)
     ids=$0; sub(/^.*distinct_workers: /,"",ids); sub(/,.*/,"",ids)
     if (w==1 && qp+0==0 && term+0==0 && ids+0==0) seen[1]=1
     if ((w==2 || w==4) && qp+0>0 && term+0>0 && ids+0>0 && ids+0<=w) seen[w]=1
 }
 END { if (!seen[1] || !seen[2] || !seen[4]) exit 1 }
' "$artifact/run.log" || { echo 'Unsupported: actual worker evidence missing/invalid' >&2; exit 1; }
echo "Passed: complete Rust-only matrix and actual worker observations; artifact=$artifact"

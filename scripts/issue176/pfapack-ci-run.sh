#!/bin/bash
set -euo pipefail
test "$#" = 3
feature=$1;mode=$2;out=$3
test ! -e "$out";mkdir "$out"
root=$(git rev-parse --show-toplevel)
test "$(git rev-parse HEAD)" = "$GITHUB_SHA"
export CARGO_TARGET_DIR="$RUNNER_TEMP/pfapack176-target-$feature-$mode" CARGO_BUILD_JOBS=2 CARGO_INCREMENTAL=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 MVMC_RS_INNER_THREADS=1
test ! -e "$CARGO_TARGET_DIR"
case "$feature" in default)features=();;simd-backend|blas-backend)features=(--features "$feature");;both)features=(--features simd-backend,blas-backend);;*)exit 2;;esac
case "$mode" in ordinary)filter=();runflags=();;ignored)filter=(-E 'binary(=golden_vs_julia)');runflags=(--run-ignored ignored-only);;*)exit 2;;esac
finish(){
 prior=$?;trap - EXIT;set +e;post=0
 node scripts/issue176/pfapack-ci-source.mjs "$out/source-after.json" >"$out/source-after.stdout" 2>"$out/source-after.stderr" || post=1
 cmp "$out/source-before.json" "$out/source-after.json" >"$out/source-post.log" 2>&1 || post=1
 if test -f "$out/selected.json";then
  node --input-type=module -e 'import fs from "node:fs";import crypto from "node:crypto";import assert from "node:assert/strict";const j=JSON.parse(fs.readFileSync(process.argv[1]));for(const [p,h]of j.artifacts)assert.equal(crypto.createHash("sha256").update(fs.readFileSync(p)).digest("hex"),h,p)' "$out/selected.json" >"$out/artifact-post.log" 2>&1 || post=1
 fi
 printf 'prior=%s\npost=%s\n' "$prior" "$post" >"$out/terminal.txt"
 if test "$prior" != 0;then exit "$prior";fi;test "$post" = 0
}
trap finish EXIT
node scripts/issue176/pfapack-ci-source.mjs "$out/source-before.json"
{ uname -a;rustc -Vv;cargo nextest --version;node --version;pkg-config --modversion openblas;git rev-parse HEAD; } >"$out/environment.txt"
step(){ local label=$1 seconds=$2;shift 2;local status=0
 timeout --foreground --kill-after=5s "${seconds}s" "$@" >"$out/$label.stdout" 2>"$out/$label.stderr" || status=$?
 printf '%s\n' "$status" >"$out/$label.status";test "$status" = 0
}
step list 1800 cargo nextest list -p pfapack --locked --cargo-profile test-fast "${features[@]}" "${filter[@]}" "${runflags[@]}" --message-format json
node scripts/issue176/pfapack-ci-list.mjs "$out/list.stdout" "$root" "$CARGO_TARGET_DIR" "$feature" "$mode" "$out/selected.json"
step run 600 cargo nextest run -p pfapack --locked --cargo-profile test-fast "${features[@]}" "${filter[@]}" "${runflags[@]}" --no-fail-fast --retries 0 --no-tests fail
step relist 120 cargo nextest list -p pfapack --locked --cargo-profile test-fast "${features[@]}" "${filter[@]}" "${runflags[@]}" --message-format json
node scripts/issue176/pfapack-ci-list.mjs "$out/relist.stdout" "$root" "$CARGO_TARGET_DIR" "$feature" "$mode" "$out/reselected.json"
cmp "$out/selected.json" "$out/reselected.json"

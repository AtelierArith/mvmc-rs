#!/usr/bin/env bash
# Extract the C candidate retry loops verbatim, build the raw-word probe and
# compare (or, with --write, regenerate) the checked-in Rust fixture. Optional
# developer tool; Cargo builds and Rust tests never run it.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
dir="$root/c_toolbox/hopping_retry_559"
src="$root/extern/mVMC-1.3.0/src/mVMC"
fixture="$root/tests/fixtures/candidate/hopping_retry_cap_559.txt"

check() { echo "$1  $2" | sha256sum --check --quiet; }
check 8431b58eaec53e5325c801b69aa8a56306ec6a154b54e099266f4e21e5dd42ed "$src/vmcmake.c"
check 2b43a47a85ec13833d453a010962334147024a6d6aae01b99bb8b2a0b80c15db "$src/vmcmake_fsz.c"

extract() { # file first last
  printf '/* Verbatim: extern/mVMC-1.3.0/src/mVMC/%s lines %s-%s (GPLv3, see that file). */\n' \
    "$(basename "$1")" "$2" "$3"
  sed -n "$2,$3p" "$1"
}
{
  extract "$src/vmcmake.c" 515 545
} > "$dir/hopping_upstream.inc"
{
  extract "$src/vmcmake_fsz.c" 594 664
  extract "$src/vmcmake_fsz.c" 756 788
} > "$dir/fsz_upstream.inc"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cc -std=c11 -O0 -Wall -Wextra -Wno-unused-parameter -Werror \
  "$dir/probe.c" -o "$work/probe"
"$work/probe" > "$work/out.txt"
if [[ "${1:-}" == "--write" ]]; then
  cp "$work/out.txt" "$fixture"
  echo "wrote $fixture"
else
  diff -u "$fixture" "$work/out.txt"
  echo "fixture matches C"
fi

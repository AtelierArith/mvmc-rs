#!/usr/bin/env bash
# Extract the C candidate retry loops verbatim, build the raw-word probe and
# compare (or, with --write, regenerate) the checked-in Rust fixture. Optional
# developer tool; Cargo builds and Rust tests never run it.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
dir="$root/c_toolbox/hopping_retry_559"
src="$root/extern/mVMC-1.4.0/src/mVMC"
fixture="$root/tests/fixtures/candidate/hopping_retry_cap_559.txt"

check() { echo "$1  $2" | sha256sum --check --quiet; }
check a9518a622b4c170e0f0d6bca98270425a0042ccbe1c1deb1106c8bbec0a28fc9 "$src/vmcmake.c"
check f6322ed48eb186c097074ea9287906fcf58129a44d91b932d0a6750ea9daba66 "$src/vmcmake_fsz.c"

extract() { # file first last
  printf '/* Verbatim: extern/mVMC-1.4.0/src/mVMC/%s lines %s-%s (GPLv3, see that file). */\n' \
    "$(basename "$1")" "$2" "$3"
  sed -n "$2,$3p" "$1"
}
{
  extract "$src/vmcmake.c" 620 650
} > "$dir/hopping_upstream.inc"
{
  extract "$src/vmcmake_fsz.c" 633 703
  extract "$src/vmcmake_fsz.c" 866 898
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

#!/bin/bash
# Generate tests/fixtures/multidef_348/split_c_probe.txt (issue #348).
# Explicit developer command; never invoked by Cargo or Rust tests. Run from the repo root:
#   c_toolbox/multidef_348/generate_split.sh
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
work=$(mktemp -d)
gcc -O0 -Wall -o "$work/split_probe" "$root/c_toolbox/multidef_348/split_probe.c"
"$work/split_probe" > "$root/tests/fixtures/multidef_348/split_c_probe.txt"
{
  echo "generator=c_toolbox/multidef_348/generate_split.sh (standalone kernel check, no MPI)"
  echo "generated_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "formula_source=extern/mVMC-1.3.0/src/mVMC/vmcmain.c lines 752-759 (verbatim)"
  echo "vmcmain.c_sha256=$(sha256sum "$root/extern/mVMC-1.3.0/src/mVMC/vmcmain.c" | cut -d' ' -f1)"
  echo "split_probe.c_sha256=$(sha256sum "$root/c_toolbox/multidef_348/split_probe.c" | cut -d' ' -f1)"
  echo "compiler=$(gcc --version | head -1)  flags=-O0 -Wall"
  echo "arch=$(uname -m) kernel=$(uname -sr)"
  echo "split_c_probe.txt_sha256=$(sha256sum "$root/tests/fixtures/multidef_348/split_c_probe.txt" | cut -d' ' -f1)"
} > "$root/tests/fixtures/multidef_348/split_PROVENANCE.txt"
rm -rf "$work"

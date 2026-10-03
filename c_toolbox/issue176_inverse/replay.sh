#!/usr/bin/env bash
set -euo pipefail
probe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$probe_dir/../.." && pwd)
replay_dir=$(mktemp -d /tmp/mvmc-issue176-inverse.XXXXXX)
native_dir="$repo_dir/extern/mVMC-1.3.0/src"
printf 'artifacts=%s\n' "$replay_dir"
c++ --version > "$replay_dir/compiler.txt"
gfortran --version >> "$replay_dir/compiler.txt"
sha256sum "$probe_dir/replay.cc" "$probe_dir/replay.sh" \
  "$repo_dir/tests/fixtures/real_fsz/setup.txt" \
  "$native_dir"/ltl2inv/{ltl2inv.cc,invert.tcc,pfaffian.tcc,trmmt.tcc,ilaenv_lauum.cc,ilaenv_lauum.hh,ilaenv.h,ilaenv_wrap.f90} \
  "$native_dir"/common/{blalink.hh,blalink_fort.h,colmaj.hh,deps/blis.h} \
  > "$replay_dir/source-sha256.txt"
set -x
c++ -std=c++11 -O0 -ffp-contract=off -DBLAS_EXTERNAL \
  -I"$native_dir/common" -I"$native_dir/common/deps" -I"$native_dir/ltl2inv" \
  -c "$native_dir/ltl2inv/ltl2inv.cc" -o "$replay_dir/ltl2inv.o"
c++ -std=c++11 -O0 -ffp-contract=off -DBLAS_EXTERNAL \
  -I"$native_dir/common" -I"$native_dir/common/deps" -I"$native_dir/ltl2inv" \
  -c "$native_dir/ltl2inv/ilaenv_lauum.cc" -o "$replay_dir/ilaenv_lauum.o"
gfortran -O0 -ffp-contract=off -J"$replay_dir" \
  -c "$native_dir/ltl2inv/ilaenv_wrap.f90" -o "$replay_dir/ilaenv_wrap.o"
c++ -std=c++11 -O0 -ffp-contract=off "$probe_dir/replay.cc" \
  "$replay_dir/ltl2inv.o" "$replay_dir/ilaenv_lauum.o" "$replay_dir/ilaenv_wrap.o" \
  -lopenblas -lgfortran -o "$replay_dir/replay"
OPENBLAS_NUM_THREADS=1 "$replay_dir/replay" | tee "$replay_dir/result.txt"
set +x
ldd "$replay_dir/replay" > "$replay_dir/libraries.txt"
sha256sum "$replay_dir/replay" "$replay_dir/result.txt"

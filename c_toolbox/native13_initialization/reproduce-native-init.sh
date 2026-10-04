#!/usr/bin/env bash
# Explicit optional developer acquisition; NEVER invoked by Cargo.
# mVMC source/extraction is GPL-3.0-or-later; included SFMT retains BSD license.
set -euo pipefail
source_tree=$(realpath "${1:?original native source checkout}")
build=$(realpath "${2:?original CMake build with pinned providers}")
inputs=$(realpath "${3:?checked-in fixture inputs directory}")
root=${4:?new exclusive output directory}
packet=$(cd "$(dirname "$0")";pwd)
test ! -e "$root";mkdir "$root";root=$(realpath "$root")
export TMPDIR="$root"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
printf '%s  %s\n' fdcd661c4eb028786fc58ae5e1f5232426c943b547f95a0d74e888e602256d63 "$source_tree/src/mVMC/vmcmain.c" b61f0f0239193fd4c06f2879e6e8452225bec45b3bca426902c244fd31ad4e3a "$source_tree/src/sfmt/SFMT.c" > "$root/original.sha256"
sha256sum -c "$root/original.sha256"
perl "$packet/extract-family.pl" "$source_tree/src/mVMC/vmcmain.c" "$root/vmcmain.family.c" > "$root/extraction-family.log"
perl "$packet/extract-init-stop.pl" "$root/vmcmain.family.c" "$root/vmcmain.stop.c" > "$root/extraction-stop.log"
flags=(-DBLAS_EXTERNAL -DMEXP=19937 -D_lapack -D_mVMC -D_mpi_use
 -I"$build/include/blis" -I"$source_tree/src/mVMC/include" -I"$source_tree/src/common"
 -I"$source_tree/src/StdFace/src" -I"$source_tree/src/mVMC" -I"$source_tree/src/sfmt"
 -I"$packet" -O3 -DNDEBUG -fopenmp -O3 -DNDEBUG)
for spec in "vmcmain:$root/vmcmain.stop.c" "family:$packet/family.c" "sfmt_observer:$packet/sfmt_observer.c";do
 /opt/mpich/bin/mpicc "${flags[@]}" -c "${spec#*:}" -o "$root/${spec%%:*}.o"
done
/opt/mpich/bin/mpicxx -O3 -DNDEBUG -fopenmp -rdynamic "$root/vmcmain.o" "$root/family.o" "$root/sfmt_observer.o" \
 "$build/src/mVMC/CMakeFiles/vmc.out.dir/physcal_lanczos.c.o" "$build/src/mVMC/CMakeFiles/vmc.out.dir/splitloop.c.o" \
 -o "$root/vmc-init-stop" -L"$source_tree/src/pfapack" -Wl,-rpath,"$source_tree/src/pfapack" \
 "$build/src/StdFace/src/libStdFace_mvmc.a" -lm "$build/src/pfapack/fortran/libpfapack.a" \
 "$build/src/ltl2inv/libltl2inv.a" "$build/lib/libblis.a" -lpthread -lopenblas -lm -ldl -lgfortran -lquadmath
sha256sum "$root/vmc-init-stop" > "$root/binary.sha256"
ldd "$root/vmc-init-stop" > "$root/runtime.ldd";! rg 'not found' "$root/runtime.ldd"
for model in HeisenbergChain HubbardChain HubbardTetragonal HubbardTetragonal_MomentumProjection KondoChain HeisenbergChain_cmp HubbardChain_cmp KondoChain_cmp KondoChain_Stot1_cmp GeneralRBM_cmp HeisenbergChain_fsz HubbardChain_fsz KondoChain_fsz;do
 mkdir "$root/$model" "$root/$model/checkpoints"
 cp "$inputs/$model/"*.def "$root/$model/"
 (cd "$root/$model";export MVMC_NATIVE13_INIT_OBSERVER_ROOT="$root/$model/checkpoints";timeout -k 5 120 "$root/vmc-init-stop" -e namelist.def) > "$root/$model.stdout" 2> "$root/$model.stderr"
done
sha256sum -c "$root/original.sha256"
# NEW outputs require NEW independently recorded build/runtime/source bindings.
# Do not substitute them into the historical strict-pinned generator without review.

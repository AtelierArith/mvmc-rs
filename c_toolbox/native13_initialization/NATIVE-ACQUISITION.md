# Durable native acquisition provenance and reproduction

All commands below are optional developer commands, NOT run while preparing this
candidate. Ordinary Cargo tests require none of these sources/programs/libraries.

## Measured historical environment

Native environment Linux x86_64, devcontainer73c57e563c61. Retained build receipts
record GCC13.3.0 Ubuntu13.3.0-6ubuntu2~24.04.1, GNU Fortran13.3.0, CMake3.28.3.
Read-only current pinned MPICH metadata:4.2.0, ABI16:0:4, ch4:ucx,
prefix/opt/mpich, hydra/pmi1, withoutPMIx. Original mpicc-show:
gcc -I/opt/mpich/include -L/opt/mpich/lib -Wl,-rpath,/opt/mpich/lib
-Wl,--enable-new-dtags -lmpi. These checks are separate from Rust validation.
Julia was NOT used in native generation or this ordinary Rust regression.

Actual retained provider-generation.zDKjNP metadata:
OpenBLAS0.3.26, NO_LAPACKE,DYNAMIC_ARCH,NO_AFFINITY,Haswell,MAX_THREADS64,
/usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblasp-r0.3.26.so, getterthreads1.
StaticBLIS version sv0.8.1+arm, dim_t8, raw_thread_getter=-1; actualthreads
UNVERIFIED. Getter probe metadata is NOT proof of live VMC worker settings.
OMP/OPENBLAS/MKL environment values1. Actual native ELF/runtime manifests in
fixture provenance retain before/after SHA; no complete backend dispatch claim.

Original native tree d73d06bd529d3b2573f38eb5817c4a5f52971006;
StdFace6fa4ef1f6809001a24b6501bd08f988bdc8cb7c4;
PfaPackebaa11b722dbb5f1ddc62da607592d2e301c0c4a.
BLIS gitlinka32257eeab2e9946e71546a05a1847a39341ec6b is source inventory ONLY:
the LINKED provider was the downloaded static sv0.8.1+arm artifact.
Original download recipe download_blis_artifact.cmake selected architecture,
then https://github.com/xrq-phys/blis/releases/download/sv0.8.1+arm/libblis_intel64_gcc.tar.gz
or amd64 variant. Retained actual archiveSHA
6d52d23b00dfdb104a8d6ac7285b6596aabe681cadca029bdf7609f420cde486,
linked libblis.a498373d374b25c6241311ac745bf3a098b7cc472063b7f71142980fb38bfe53b,
header531ebb25642ce857698a8f27dcb2db2c7273f7e55b8e787179822f43eda8c83b.
This records acquired bytes, NOT an upstream-authenticated EXPECTED_HASH claim.

## Exact source extraction and passive behavior

extract-family.pl pins original vmcmain SHA and six unique insertion anchors:
init_gen_rand(RndSeed+group1); LAPACK workspace query; InitParameter;
End Initialize parameters (after original optional input loads);
SyncModifiedParameter; InitQPWeight. Seed/query capture no uninitialized Para.
extract-init-stop.pl pins complete family derivative4f40056f... and inserts
termination immediately AFTER the pre-sampling checkpoint: MPI_Finalize/return0.
Original vmcmain main/initializer/math call bodies are preserved; sampler NOT run.
Original GPL-3.0-or-later copyright/license remains in vendored native tree.

family.c/h/family_hook.h derive defined masks from native readdef GetInfoOpt
2101-2113 and GetInfoOptOrbitalParalell2084-2098. No int read from unwritten
imaginary flags. Unsupported DH/BF/OptTrans layouts fail, not guessed.
original13-descriptors.tsv preserves all13 expected layouts/stage headers.
sfmt_observer.c wraps the sole original static SFMT state, counts exact word
entrypoints and peeks a COPY recurrence without reseeding/live state restoration.
Wide/bulk paths fail snapshot rather than misreport drawcount. Original included
SFMT.c and SFMT LICENSE.txt retain Saito/Matsumoto/Hiroshima BSD notices.
observer.h declares the exact compiled ABI. No native binaries are checked in.

## Reproduction commands and independent bindings

Obtain a separate original native checkout at the tree/gitlinks above. Do NOT
modify vendored references. Original build invocation (one worker):

```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
cmake -S ORIGINAL -B BUILD -DCMAKE_C_COMPILER=/opt/mpich/bin/mpicc \
 -DCMAKE_CXX_COMPILER=/opt/mpich/bin/mpicxx -DCMAKE_Fortran_COMPILER=/usr/bin/gfortran \
 -DGIT_SUBMODULE_UPDATE=OFF -DTesting=OFF -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build BUILD --target vmcdry.out vmc.out --parallel 1
bash c_toolbox/native13_initialization/reproduce-native-init.sh ORIGINAL BUILD \
 tests/fixtures/native13_initialization/inputs NEW_EXCLUSIVE_OUTPUT
```

Before a regeneration freeze complete recursive native/gitlink files, compiled
object/library/provider/header/tool/input manifests and actual provider metadata;
recheck after. Do not trust only post-run or selfdeclared manifests. Check the
exact acquired BLIS artifact before build; no silent provider substitution.
reproduce-native-init.sh explicitly compiles ONLY replacement vmcmain/family/SFMT
objects with original defines/includes/O3/NDEBUG/OpenMP and uses original native
physcal_lanczos/splitloop objects and StdFace/PfaPack/ltl2inv/BLIS libraries.
Original SFMT.c.o is replaced, NOT additionally linked (avoid two static states).
Historical compiled family.o and sfmt_observer.o were reused for OVRLu0; new
reproduction recompiles those exact source files and records its NEW identity.
These recipes are SOURCE-only here, not newly executed or certified builds.

The strict historical converter takes six positional arguments:

```
perl c_toolbox/native13_initialization/generate-fixtures.pl \
 FIRST3_CAPTURE REMAINING10_CAPTURE ORIGINAL_INPUT_ROOT \
 original13-descriptors.tsv VERIFIED_ORIGIN_JSON NEW_EXCLUSIVE_DEST
```

Its --verify-only destination performs independent artifact/archive/manifest
checks. Historical binary/source/provider identities are deliberately pinned;
NEW reproduction has NEW provenance and must receive reviewed prospective pins,
not masquerade as old archival acquisition or silently replace expectations.
Independent validate-generated.pl verifies actual C raw/future CONTENT/order,
parameter text roundtrip, QP text and exact original definition hashes.

Rust-only WP5PTo validation remains separately main4ce/frozen rlibs. The ordinary
candidate initially used0130, advanced to83fd before its first Cargo build, and
then to0f64bcb5f65fc7eb84af1487e978e9d3c74c3db8 after the83fd run had ended.
On83fd, run97c23c70-d319-4f6c-bdc4-55debe167c8c passed1test/0skip; a later focused
clippy failed on a redundant same-type conversion, with all post hashes0.
After that lint-only repair,0f64 run448c5191-3047-4c71-b8f8-e7820a79dfff passed
1test/0skip in0.030s; focused clippy/fmt and source/binary/tools posts were all0.
Selected binarySHA4926fce008feee26d6e1b30488ce9ee53b04b424c46250e8a8ec909587a7fd33.
The user-authorized absolute1e-18 bound applies only to fixed GeneralRBM random
RBM components; all raw/discrete contracts stay exact. These results do not
claim native20/13sampling/thread completion. This paragraph is a results-only
update after validation; the developer reproduction recipe itself was not run.

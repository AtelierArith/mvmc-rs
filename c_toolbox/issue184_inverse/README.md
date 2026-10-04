# Bounded issue184 inverse acquisition

Developer-only standalone kernel probe; never invoked/read by Cargo. Numerical
bodies are included unchanged, not extracted/rewritten: authoritative
`extern/mVMC-1.3.0/src/ltl2inv/invert.tcc` lines12–33 sktdsmx and76–120 utu2inv,
SHA256 `d477c7d2bb6d2b0c29872d06ec75fd7e804e1e30ace00888b77d31d2c74b821c`.
Its MPL-2.0 notice and included common headers remain in upstream sources.
ILAENV uses unchanged ilaenv_lauum.cc and ilaenv_wrap.f90, hashes
`2e2f764cb9315088e397a649bd4487f1ff342e8c54549c0bc9695d0b63be3a5e` and
`13292b4a29358488f5a4bbac300bc09bd4b500c9af6e5579fc7a0b95d5e7b2ca`.

Reproduce from repository root with a fresh external output destination:

```sh
probe_stage=$(mktemp -d /tmp/mvmc-issue184-inverse.XXXXXX)
gfortran -O0 -ffp-contract=off -J"$probe_stage" -c extern/mVMC-1.3.0/src/ltl2inv/ilaenv_wrap.f90 -o "$probe_stage/ilaenv.o"
c++ -std=c++17 -O0 -ffp-contract=off -DBLAS_EXTERNAL -Iextern/mVMC-1.3.0/src/common -Iextern/mVMC-1.3.0/src/common/deps -Iextern/mVMC-1.3.0/src/ltl2inv c_toolbox/issue184_inverse/probe.cc extern/mVMC-1.3.0/src/ltl2inv/ilaenv_lauum.cc "$probe_stage/ilaenv.o" -lopenblas -lgfortran -o "$probe_stage/probe"
OPENBLAS_NUM_THREADS=1 julia +1.13.1 --startup-file=no --project=extern/Julia-mVMC c_toolbox/issue184_inverse/generate.jl "$probe_stage/probe" "$probe_stage/acquired"
```

Actual acquisition stage `/tmp/mvmc-issue184-inverse.tUbfFV/residual-acquired`,
Julia1.13.1 Linux x86_64, project `extern/Julia-mVMC`, Manifest-v1.13 SHA256
`09ebd06dab244510094b99fe7c6efa2fe7a3d22221d1336b951123a5a6e8befc`.
C++/Fortran GCC13.3.0, native LP64 libopenblas.so.0; Julia ILP64 OpenBLAS0.3.30
USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS512, actual BLAS threads1
(generator explicitly sets1). Compiler emits upstream diag2char return warning.
Native OpenBLAS package is0.3.26+ds-1ubuntu0.1 (dpkg-query actual).
Julia library SHA256 `4ee5ad9dcc4082b918d3e5bfc434b5a74fb43025f102447eb16a7f04ba21a12b`.
Metadata-only follow-up queried threads1 and the version string; it did not
regenerate expected values. Current generator SHA256
`4d0c29860d84bf2467e9fbaef1c8223b5dccd8fe5b2a545f05cc2b1ae3ea52a0`.
Julia utu2.jl SHA `1352a3edbc413611ad99877ea32f7faf5e54e003f9a3fa088f8dbdbdb2723a43`.
Latest metadata collector adds compiler/library/source identities; it does not
change expectations. Historical small goldens cite a currently absent generator;
they contain inverse A only, not M/vT. These new fixtures capture all three.

Real4 uses the original literal upper entries1,.5,.3,2,.7,3 and identity pivot.
Complex6 is NEW fixed dyadic input (not the original unseeded Julia matrix),
with pivot2,1,4,3,6,5. Adjacent paired upper entries2,3,4; other upper entries
real(i+j)/16, imaginary(i-j)/32, one-based indices. This pivot sequence exercises
swaps but each pair swaps twice, so its net permutation is identity; no coverage
claim for every nonidentity permutation, panel kernel or original12 assertions.

Native C/J maximum absolute differences: real0, complex2.8609792490763984e-17.
Independent BigFloat256 reconstruction of U*T*transpose(U), with the sequential
pivot permutation, yields C infinity residual2.7755575615628914e-16/2.078233767074223e-16,
backward errors1.826024711554534e-17/1.6306938730227955e-17 and infinity-condition
estimates14.2/11.74447522895164. These are estimates for these small inputs only.
256epsilon abs+rel accounts for four <=6-term solve/product stages and complex
scalar arithmetic (8*4*6=192 rounded up to256); separate scaled residual checks
prevent forward comparison alone from concealing a wrong inverse. No bitwise
computed-value comparison, tolerance increase, sampler/MPI or full model claim.

Initial Rust poisoned workspace run e7845932-38d6-4c5f-877c-d69c82c01d6c exit100:
0PASS/2FAIL. Partial M reset left final strict-upper column17; real first bad A
entry3 was17.666666666666668 vs.6666666666666666. Full M reset fixes the algorithm
without changing bounds. Final strict-schema/reuse tests and feature gates must
be reported separately from this preserved failure and early passing snapshots.

## Final bounded validation

Frozen production SHA256
`c386c9929e962058f5560312df38c81072613d731bb1e05cc4907cc03a243380`;
strict-schema/zero-poison/reused-workspace/residual test SHA256
`9d1d3af1169a9526330fd1218769035d0199f5baca0fb420cb6df3f98b8f1d5e`.
No numeric bound changed to repair the defect. Commands for each feature row:

```sh
cargo nextest run -p pfapack --cargo-profile test-fast --locked --no-fail-fast --retries 0
# Independently add --features simd-backend, --features blas-backend,
# or --features 'simd-backend blas-backend'.
# Run the existing ignored large fixtures separately with --run-ignored only.
cargo clippy -p pfapack --all-targets --features 'simd-backend blas-backend' --locked -- -D warnings
```

| Feature configuration | Ordinary tests, exit0 | Separate ignored large tests, exit0 |
| --- | --- | --- |
| default | d4879b8d-35e7-483b-b28b-786189870b66:24PASS/8skipped,.010s | f861f0f5-80fb-4386-ad37-db4012df0ce4:8PASS/24skipped,.208s |
| SIMD | 89c1fa59-1ef2-4339-a494-0622340e825d:24PASS/8skipped,.010s | 0531a117-72a1-4b52-9a06-ebed54431344:8PASS/24skipped,.199s |
| BLAS | bc3cf5b8-f44b-4401-a0c0-feaac2c9bf33:27PASS/8skipped,.028s | 547d8198-4e96-46c0-93ae-0133fdbaee95:8PASS/27skipped,.802s |
| SIMD+BLAS | 57eda877-9ca9-42be-bf79-7de0e0212584:27PASS/8skipped,.025s | 8f2dd60a-6726-4b33-8579-ce13affe44a6:8PASS/27skipped,.824s |

Independent parent final combined proof
`a907e243-2a98-4288-a25e-64ed7aa335fa`:exit0,27PASS/8excluded,.030s;
parent FULL strict-fixture/reuse/residual source and minimal production diff review.
Agent strict all-targets combined clippy command above exited0,.57s.
Parent additionally reports clippy/docs success; exact parent commands and
doctest counts were not supplied in this handoff, so no behavioral doctest or
whole-workspace success is inferred. The two new small fixtures do not establish
all12 original unseeded assertions or panel A/M/vT workspace parity. Existing
large golden tests compare inverse A, not all workspace arrays.

## Scalar benchmark execution, not a before/after comparison

Session68539 terminal0, final tool chunk969b4c. Exact command:

```sh
OPENBLAS_NUM_THREADS=1 cargo run --release --manifest-path benchmark/pfapack_compare/Cargo.toml --offline
```

Final output is preserved in `benchmark-68539-tool-transcription.txt`, explicitly
a tool-output transcription rather than a native capture. Release compilation
took4m23s. Existing benchmark `ltl_utu2inv` includes LTL decomposition plus
inverse; it is not an inverse-only microbenchmark. Median milliseconds:

| n | real | complex |
| --- | ---: | ---: |
| 32 | 0.041786 | 0.059674 |
| 64 | 0.128092 | 0.387000 |
| 128 | 0.848494 | 1.964286 |
| 256 | 4.982521 | 14.371287 |

Scalar/default feature configuration only. No pre-fix baseline, before/after,
SIMD/BLAS benchmark result, performance improvement or regression-size claim.
The separate transfer_green line is unmodified benchmark output, not evidence
about this inverse fix. No numerical-policy or fixture-bound relaxation.

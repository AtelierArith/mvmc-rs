# Two-hop bilinear form of the C two-electron Pfaffian update (issue #449)

Standalone kernel check, not a full `vmc.out` run: it evaluates the reduction of the
authoritative C kernel `calculateNewPfMTwo_child_real` for the inputs of
`tests/fixtures/pfaffian_cg/two_hop_bilinear.txt` (size-6 cases of the Julia fixture plus sizes
2 to 64) and stores the resulting bit patterns as the independent expectation of
`two_hop_bilinear_is_bit_identical_to_the_c_reduction`
(`crates/mvmc-core/src/sampling/updates.rs`).

## Origin

* Upstream: `extern/mVMC-1.4.0` (submodule `v1.4.0`, commit
  `e83aa6cb1e37bfc4876848787ed2f61721a7e972`).
* Source file: `src/mVMC/pfupdate_two_real.c`, SHA-256
  `79c98d367003d1f2acb304dcddad6e7c2bfa7e3f0b0a4706099ebd84856b6513`.
* Extraction boundary: lines 169-180 (the `bMa` double loop), copied verbatim into `probe.c`
  between the `--- ... ---` markers; only the surrounding I/O is new. The `p_a/p_b/q_a/q_b`
  loop (lines 154-167) is not extracted (it is already sequential in Rust and not changed).
  1.4.0 adds `#pragma omp simd reduction(+:tmp)` to the inner loop; the probe is compiled
  without OpenMP, so C keeps the sequential order. Regenerated under 1.4.0 (#695): the data
  lines are identical to the former 1.3.0 extraction (lines 167-177).

```c
for(msi=0;msi<nsize;msi++) {
  invM_i = invM + msi*Nsize;
  tmp = 0.0;
  for(msj=0;msj<nsize;msj++) {
    tmp += invM_i[msj] * vec_a[msj];
  }
  bMa += vec_b[msi] * tmp;
}
```

## Compiler options and the FMA finding

`gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0`, `-O2 -ffp-contract=off` and `-O2` (default
x86-64 target): identical bits (checked by `generate.sh`). With `-O3 -march=native` (FMA
contraction enabled) 50 of the 132 case results differ in the last bits, so the fixture is
valid only for builds without FMA contraction. The reference C builds of this repository
(`c_toolbox/*/build.sh`) use CMake defaults without `-march`, i.e. no FMA.

## Reproduce (explicit developer command)

```sh
c_toolbox/two_hop_bilinear_449/generate.sh   # rewrites tests/fixtures/pfaffian_cg/two_hop_bilinear_c.txt
```

Rust builds and tests never run this; they read the checked-in fixture only.

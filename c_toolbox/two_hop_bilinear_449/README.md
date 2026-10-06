# Two-hop bilinear form of the C two-electron Pfaffian update (issue #449)

Standalone kernel check, not a full `vmc.out` run: it evaluates the reduction of the
authoritative C kernel `calculateNewPfMTwo_child_real` for the inputs of
`tests/fixtures/pfaffian_cg/two_hop_bilinear.txt` (size-6 cases of the Julia fixture plus sizes
2 to 64) and stores the resulting bit patterns as the independent expectation of
`two_hop_bilinear_is_bit_identical_to_the_c_reduction`
(`crates/mvmc-core/src/sampling/updates.rs`).

## Origin

* Upstream: `extern/mVMC-1.3.0` (submodule `v1.3.0`, commit
  `d73d06bd529d3b2573f38eb5817c4a5f52971006`).
* Source file: `src/mVMC/pfupdate_two_real.c`, SHA-256
  `e30feeb3d1710be4db69a792fbd73408bbb0e95438af0268745acf49556c1efe`.
* Extraction boundary: lines 167-177 (the `bMa` double loop), copied verbatim into `probe.c`
  between the `--- ... ---` markers; only the surrounding I/O is new. The `p_a/p_b/q_a/q_b`
  loop (lines 154-164) is not extracted (it is already sequential in Rust and not changed).

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

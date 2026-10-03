# Exact original S104 Heisenberg input

These 13 Expert input files are byte-for-byte copies of the original sample at
`MVMCExpertModeParsers.jl/test/samples/HeisenbergChain` in
[Julia-mVMC revision 8bb1b9e8ae47b1512c00b321be05664ddcac0fd1](https://github.com/tmisawa/Julia-mVMC/tree/8bb1b9e8ae47b1512c00b321be05664ddcac0fd1/MVMCExpertModeParsers.jl/test/samples/HeisenbergChain).
The namelist and every referenced definition are included, without editing
settings, indices, signs, coefficients or line endings. Source Julia sample
provenance is not a fresh native C execution claim. No computed Rust expectation
is captured in this fixture.

Original test `MVMCOptimizers.jl/test/test_slater_update.jl` SHA-256:
`840d0d4d6a1474ff3d10188033c29ae969b837a32d638d4f64d1af9dedeed613`.
S104/M0512–M0517 uses public parse, seeded initialization, QP weight initialization
and normal Slater update, then checks success and table size. It does not run
sampling or optimization. Input has Nsite16, NSPGaussLeg8, NMPTrans-1,
RndSeed123456789. The regression deliberately preserves the original input's
NSROptItrStep300 and NVMCSample1000 but never invokes these workloads; it does
not replace the current long-runner acceptance baseline20.

Rust regression:
`issue184_slater_contracts::original_s104_sixteen_site_seeded_input_wires_all_slater_planes`.
The actual public initialization/update path checks the exact input settings,
eight QP planes and all 8192 cells finite/real, nonzero table, C algebraic
antisymmetry and same-input/seed within-implementation repeatability of
parameters, Slater table and RNG debug state. No new numerical tolerance is used.
This is not a cross-language numerical table or sampling-trajectory oracle.
C `slater.c::UpdateSlaterElm_fcmp` spin-block formula is the algebraic authority;
the sibling literal noncommuting-translation regression checks its four blocks
against independent enumerated expectations.

## Offline reproduction / copy verification

Ordinary test (no Julia, C or toolbox runtime):

```sh
cargo nextest run --locked --cargo-profile test-fast -p mvmc-core \
  --test issue184_slater_contracts --no-fail-fast --retries 0
```

Run `bcd7ef44-d8cc-402a-8d8a-2f5d52343f86`: terminal0, 5/5 passed,
0 skipped, 0.009s. Test SHA-256
`5b8e45aa1e255bf4c513de24722badd883bf6144ed9bcac6190df377aaba2b4a`.
Source-copy verification from repository root (optional developer check only):

```sh
for file in tests/fixtures/physcal_181/slater-s104/inputs/*.def; do
  cmp "$file" "extern/Julia-mVMC/MVMCExpertModeParsers.jl/test/samples/HeisenbergChain/${file##*/}" || exit
 done
```

All thirteen `cmp` checks completed terminal0. SHA-256 copied/source bytes:

```text
3cb308455d59aab724fa073a4cd146c068d64845aebdd4e6a17903f9e835d9bd  inputs/coulombinter.def
c2585bb73fa671f02e28c8c717064fe6f2f9b4f541f8b5d270f063e6b16f46c8  inputs/exchange.def
79855770f5dec7e429e53a65ba25a8edee2070d717c4d75033ab3d6efd9584d6  inputs/greenone.def
877eeb8d62d90684586d83845451aff60678a26d637f82967c4744e4bf4675a5  inputs/greentwo.def
0c11a62af3563973a46f182b00c034c24b789a25434ebed148a514ec1e0d5647  inputs/gutzwilleridx.def
547179e56cd8f43ecf92391dd45af7caba2b39276c6f5a7b755f4db5b49f4a5a  inputs/hund.def
f3bddbf191d5ebc87025e2d24e88352c98add860357fa2869db8c0754d1b2a18  inputs/jastrowidx.def
4a4d66a102799b584cba6f2bba63e8038924ec8e50b46946f15bcf019549fda3  inputs/locspn.def
db730a6e74c2884bc97842ffe0f3386d95b1831bd8b12f0c8bfc7c674e1ca63e  inputs/modpara.def
9a72042e4587c41249c052ec469eebee10864dedc3e1d7fdb5d8e9e6a3db2c81  inputs/namelist.def
1d8e8134cb3e2ff7242c74365ea9c8894fbce9b87d2df0d65d836b36f0199828  inputs/orbitalidx.def
25715a81bbc8127724cbd666f55a4c9187c7907240ce03e201989fce963c0d16  inputs/qptransidx.def
bfadd2c5fee83496acda945324dec4dd2620c7295d1cc94a631e47a8efef24a5  inputs/trans.def
```

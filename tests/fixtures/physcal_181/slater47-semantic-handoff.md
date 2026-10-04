# Slater47 semantic handoff (#184)

This is a source-level mapping, not a claim that 47 assertions have run in Rust.
No verification TSV is modified. The original main test has conditional success,
failure and skip branches; the unit test also expands assertions in a loop.

Original pinned Julia sources:

- `MVMCOptimizers.jl/test/test_slater_update.jl`, SHA-256
  `840d0d4d6a1474ff3d10188033c29ae969b837a32d638d4f64d1af9dedeed613`.
- `MVMCOptimizers.jl/test_unit/test_unit_slater_update.jl`, SHA-256
  `918404343ce6a4025254d13a4bbf84ef7052ae9fea859bc2c3f1b3431db4b19e`.
- Authoritative C `src/mVMC/slater.c`, SHA-256
  `cba5ddb9b8f42894dfa808aa867b05b358caab17175fb39931caf4625037db70`.
  `UpdateSlaterElm_fcmp` composes optimized translation before fixed translation
  and multiplies both site signs (lines 70–91). Periodic orbital signs follow
  `readdef.c`'s `GetInfoOrbitalAntiParallel`, not Julia's synthetic fallback.

All new Rust tests are in `crates/mvmc-core/tests/issue184_slater_contracts.rs`.

| Original IDs / setting | Actual Rust evidence / distinction |
| --- | --- |
| M0487–M0499 / S099: four sites, two fixed translations, one optimized translation, identity/shift and inverse shift | Rust retains parsed fixed translations rather than returning five arrays from a private builder. `public_slater_update_composes_opttrans_before_fixed_translation_in_each_plane` checks every cell of four QP planes, including independent enumerated composed maps and signs. This does **not** assert an inverse-map return value. |
| M0500 / S100: missing fixed maps, two declared translations | Julia private helper throws `ArgumentError`. Rust has no corresponding public helper; supported-file requirements belong to parser/runtime validation. Do not classify Julia's helper exception as a C numerical requirement or claim this literal boundary tested here. |
| M0501–M0506 / S101: two explicit optimized maps/sign rows | New composition test checks complete maps through the public updater; its noncommuting permutations strengthen order detection. It is not the original identity/cyclic-shift tuple-return test. |
| M0507–M0509 / S102: four sites, sixteen constant .1 terms, two QP planes | Original tests only success (or failure catch) and table length, not numerical values. New composition test checks all four spin blocks and table shape with binary-exact independent integer expectations. Original constant-.1 input is not reproduced. |
| M0510–M0511 / S103: QP weights absent | `supplied_orbital_cache_is_preserved_and_missing_weights_leave_state_unchanged` exercises actual public update and complete state snapshot equality. Julia log wording is not reproduced. Data-cache preparation occurs before the return, so no blanket data atomicity claim is made. |
| M0512–M0517 / S104: original Heisenberg sample, sixteen sites, NSPGaussLeg=8, NMPTrans=-1, seed123456789 | `original_s104_sixteen_site_seeded_input_wires_all_slater_planes` now uses the **exact original thirteen Expert files**, copied byte-identically into `slater-s104/inputs` with hashes/provenance. Actual public parse → seed initialization → QP weights → update checks eight planes/all8192 cells, nonzero finite real data, C algebraic antisymmetry and within-implementation repeatability. No full sampler or optimization loop is run, matching original S104's boundary. The earlier `parsed_heisenberg_initialization_wires_all_slater_planes_repeatably` remains separately labelled six-site representative coverage. |
| M0748–M0756 / S163: three sites, indices0/2/2, coefficients1/2.5/zero duplicate | `declared_slater_slots_and_duplicate_mapping_preserve_coefficients` uses the original mapping triples, declared coefficient width3 and values1/0/2.5; checks dimensions, indices, signs and retained dense coefficients. C coefficients are separate from mapping records: Julia's nonzero term-gather rule is an API/storage distinction. Negative-sign case requires antiperiodic boundary; periodic C input overrides signs to+1, explicitly tested. |
| M0757 / S164: supplied sign matrix object identity | New cache/missing-weights test preserves both supplied matrix contents. Julia `===` object identity has no matching Rust owned-matrix public API; do not claim pointer-identity parity. |
| M0758–M0759 / S165: missing fixed maps, error log plus exception | Same private-helper architecture distinction as M0500; no fabricated test-only public API. |
| M0760–M0763 / S166: empty optimized maps become two identity maps/sign rows | Julia private helper fallback is not a declared C file-input requirement. Rust's private translated-site fallback is not directly asserted by these tests. Explicit complete maps and the public composition are tested instead. |

## Validation and remaining actionable gaps

Command:

```sh
cargo nextest run --locked --cargo-profile test-fast -p mvmc-core \
  --test issue184_slater_contracts --no-fail-fast --retries 0
```

Terminal 0, run `bcd7ef44-d8cc-402a-8d8a-2f5d52343f86`: 5 passed,
0 skipped, 0.009 s. Test source SHA-256
`5b8e45aa1e255bf4c513de24722badd883bf6144ed9bcac6190df377aaba2b4a`.
No numerical tolerance is added: the literal kernel wiring uses integer/binary
inputs; other equality checks are structural or within-implementation
repeatability. No C/Julia runtime or toolbox is invoked by these tests.

The original sixteen-site S104 input lifecycle gap is resolved by the new
exact-input test; see `slater-s104/README.md` for hashes and copy verification.
It does not establish a numerical cross-language Slater table oracle or full
sampling validation (neither was an assertion in original S104). Private helper tuple,
inverse-map and exception/log identity claims should be classified as API
differences, not silently marked numerically verified. Full sampling is covered
separately by PhysCal model tests, not by these five Slater tests.

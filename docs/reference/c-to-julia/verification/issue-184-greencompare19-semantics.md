# GreenCompare19 semantic handoff (#184)

This is a per-contract handoff, not 19 Julia assertions reproduced or a new
production comparison API. No assertion-audit TSV was edited by this owner.

## Original source and settings

Read both complete files in the pinned shared Julia reference
`8bb1b9e8ae47b1512c00b321be05664ddcac0fd1`:

- `test/integration/tools/test_green_compare.jl`: 117 lines, SHA256
  `5f5239bdbb09d30828b625014193ebcd9ab1c63a06123c9230e2f2f31c0b2678`.
- `test/integration/tools/green_compare.jl`: 193 lines, SHA256
  `8d20b64bf7a3567e5ad0ea4598c65a2208b95c52c08baec0c8ca859bf26fd0a7`.

The tests have no enclosing loops or stochastic draws. `ONE_A` is two six-field
rows with four discrete indices and two numeric fields, trailing spaces and a
blank line. `FAC_A` is the single value-only row `2.5 -1.0 0.3 0.1`.
Defaults are absolute 1e-12, linear relative 1e-10 and product relative 1e-9.
The declared `DC_A` is unused by these 19 assertions; this batch does not prove
the direct-two-body ten-field helper. C `vmcmain.c` PhysCal output uses the
single ordered factored row and indexed six-/ten-field rows; the generic
numeric helper must not replace that schema check.

Julia's test-only `GreenCompareResult` reports exact/fallback/count/errors and
detail strings. Rust has an assertion-based schema-aware gate and a generic
test-only numerical helper, not this result/file API. Their architectural
difference is explicit below. Computed-float raw-byte equality is not adopted
as a portable numerical requirement; fixed-value writer byte tests are separate.

## Assertion mapping

`N` below is Banach's
`green_helper_numeric_fields_are_bounded_but_indices_and_shape_are_exact`;
`P` is `green_helper_quantity_bounds_do_not_permit_factored_layout_drift`.
`S` is the new owned
`green_reference_schema_rejects_identical_malformed_factored_files`;
`F` is `green_reference_gate_requires_candidate_and_reference_files`.
All names refer to crate-local Rust integration tests, not Julia execution.

| ID | Original literal condition | Rust evidence / remaining API distinction |
|---|---|---|
| M1316 | Identical `ONE_A`: ok/exact/no fallback | F accepts identical literal two-row payload; exact/fallback fields are an intentional test-helper API difference. |
| M1317 | Identical two rows have four numeric values | F checks the literal two-row payload through strict six-column gate; no `n_values` result field. |
| M1318 | Identical `FAC_A`: ok/exact/four values | S accepts literal single-row four-field positive control; exact/count metadata not reproduced. |
| M1319 | First real `1` -> `1.00000000005`: numeric success, nonexact/fallback | N uses this exact perturbation and accepts; nonexact/fallback flags are not Rust API. |
| M1320 | Same perturbation: max absolute error <1e-10 | N proves bounded numeric acceptance; no returned maximum-error metric, so not literal metric-field proof. |
| M1321 | First real `1` -> `1.000000005`: failure/out-of-tolerance detail | N rejects this exact payload; Rust panic/result wording differs. |
| M1322 | One-body `1` -> `1.0000000005` fails | P rejects the exact payload at original linear bound. |
| M1323 | Factored same perturbation passes | P accepts the exact payload at original product bound. |
| M1324 | ONE_A loses its second row | N rejects exact one-row payload; no Julia detail-field API. |
| M1325 | Second index tuple `(1,1,0,1)` -> `(1,1,0,0)` | N rejects the exact discrete change despite identical numbers; no detail-field API. |
| M1326 | First row loses imaginary column | N rejects original five-field first row; no detail-field API. |
| M1327 | FAC_A four values -> `2.5 -1.0` | P rejects original width change; no detail-field API. |
| M1328 | Candidate `2.5 -1.0` / `0.3 0.1` on two rows vs one-row reference | S rejects exact literal candidate/reference at schema boundary; no detail-field API. |
| M1329 | Both files byte-identical but malformed two-row factored layout | S first proves fixed-input byte equality, then requires schema rejection. This closes the generic helper's known identical-malformed acceptance gap without altering that generic API. |
| M1330 | One identical numeric row differs in trailing space/blank line: nonexact | F proves fixed-input bytes differ; no returned `exact` flag. |
| M1331 | Same whitespace-different pair is numerically accepted | F accepts the exact original one-row payloads after strict shape/index checks. |
| M1332 | Same whitespace-different pair reports fallback | Acceptance is covered by F; no fallback metadata API. |
| M1333 | Existing ONE_A candidate, missing reference | F rejects an absent reference with supported Green basename (not Julia's arbitrary `nope.dat` reference basename); failure is asserted, not a returned `missing` detail. |
| M1334 | Missing `nope.dat` candidate, existing reference | F rejects the literal missing candidate; failure is asserted, not a returned `missing` detail. |

Julia's columnar raw-exact shortcut does not validate identical malformed
columnar content, and it compares index tokens lexically. Rust always validates
schema and integer indices. Those stronger schema/numeric-index contracts are
not claimed as literal Julia shortcut behavior or an upstream C file-reader
rejection rule. No production shim is needed merely to duplicate helper flags.

## Actual focused proof

Final owner command:

```sh
cargo nextest run -p mvmc-core --cargo-profile test-fast --locked \
  --no-fail-fast --retries 0 \
  -E 'test(green_reference_) | test(green_helper_) | test(factored_output_requires_one_ordered_row_of_pairs)'
cargo clippy -p mvmc-core --test physcal_issue181 --locked -- -D warnings
```

Owner handle66451, terminal0, run
`0cdf90c8-81a2-4313-8b3f-1ee0e7b61f57`: five tests passed, 618 outside the
selection, 0.016s. Focused clippy handle93302 terminal0. The unrelated vendored
tenferro-runtime deprecation warning was observed, not suppressed or edited.

Final source SHA256:

- `crates/mvmc-core/tests/physcal_issue181.rs`:
  `fc39e4901470b13650bf31e11cead4c821730d1c32b4affeb789c4a5e8798b60`.
- `crates/mvmc-core/tests/issue184_integration_tools_contracts.rs` (read-only,
  Banach-owned): `e00643994ce5caaa7f46f2f8b8c410777219dbc23b01a0891208bd5d97e8048e`.

This is five bounded Rust test results, not a full gate, macOS numerical proof,
19 literal Julia result-field assertions, or evidence for the unused DC_A API.

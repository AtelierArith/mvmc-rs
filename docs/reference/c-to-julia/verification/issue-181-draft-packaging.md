# #181 / owned IO and #184 tests: draft packaging, not completion

Related to #181, #180 and #184; part of #185. Do not use `Closes` based on this
focused handoff. Parent owns Rust staging/commit/push and final milestone audit.

## Exact owned paths

`issue-181-owned-paths.txt` is the explicit sorted changed/new path list for this
owner at restored HEAD `52d025e`. It includes this report and itself: 1157 paths.
It excludes unchanged already-tracked fixture files and every unrelated shared
change. The optional toolbox paths are developer probes, never Cargo oracles.

Top-level source/test/script/document paths:

- `crates/mvmc-core/src/io.rs`
- `crates/mvmc-core/tests/output_blocks.rs`
- `crates/mvmc-core/tests/physcal_issue181.rs` — bodies/helpers owned here;
  gate declarations are #183-owned and require joint packaging review.
- `crates/mvmc-core/tests/issue184_public_rejection_boundaries.rs`
- `crates/mvmc-core/tests/issue184_slater_contracts.rs`
- `crates/mvmc-cli/tests/issue184_fixed_file_boundaries.rs`
- `scripts/generate_physcal_181.jl`
- `docs/reference/c-to-julia/verification/issue-180-model-coverage.md`
- `docs/reference/c-to-julia/verification/issue-181-owned-paths.txt`
- `docs/reference/c-to-julia/verification/issue-181-draft-packaging.md`

Exact toolbox and fixture paths are enumerated in the manifest, not selected
with a broad `git add .`. Fixture subsets include six models, DH/RBM/OptTrans
and In-overlay combinations, two measurement frames, all-six-term Lanczos1/2,
C reader/empty-file/format/parameter/normalization probes, original third
Hubbard19, exact original S104 sixteen-site input and semantic handoffs.
Do not include generated `crates/mvmc-core/zvo_out.dat`, `zvo_var.dat`,
`wrapper.mod`, target/cache files, manifests from the Julia fork, or other
agents' runner/parser/MPI/SR/CG/reference-consumer changes via this manifest.

## Implemented scope

- Strict row/column and ordered discrete-index comparison with per-output
  numerical budgets; fixed parameter records are independently interpreted
  from original zqp input, not copied from a Rust-before snapshot.
- C single ordered factored-Green row in normal/Lanczos output, C trailing
  whitespace/blank-line formatting, signed indexed out/var lifecycle and
  zero-Green file-open contracts.
- C optimization-window energy/energy-square/post-update parameter aggregation
  and full declared parameter storage output (including DH and unmapped slots).
  Runner collection/callers remain another owner's implementation.
- Offline independent references for saved configurations/counters/next624,
  all supported nonInterAll terms, overlays/flags/declared widths and weighted
  measurements; provenance distinguishes Julia references, C reader/kernel
  probes and historical native C model outputs.
- IO134 analytic/public-writer and loader cases, exact original third-Hubbard
  consumed19/all-five-output regression, Slater47 semantic mapping and exact
  S104 public parse/init/QP/update lifecycle, public runtime/CLI rejection
  atomicity with count and next624 preservation.

The separate Julia aggregation fixes are already on the authorized single
upstream PR54 lineage through `62b0f97`. They are not Rust files to stage here;
the shared pinned Julia reference was not moved. No new Rust commit/push is
performed by this owner.

## Fresh bounded verification after restoration

Updated `AGENTS.md` and `docs/NUMERICAL_COMPARISONS.md` were read completely.
Computed cross-language floating-point fields use justified explicit bounds;
primitive RNG init/conversions/count/state stay exact on identical control
paths. Demonstrated numerical acceptance divergence may explain subsequent
trajectory differences; long acceptance uses same-input/seed 20-step
repeatability, not historical 50-step endpoint forcing. No tolerance was changed
by this packaging pass.

All commands use `--locked --cargo-profile test-fast --no-fail-fast --retries 0`.

| Command scope | Terminal result / run |
| --- | --- |
| Core `output_blocks`, `issue184_public_rejection_boundaries`, `issue184_slater_contracts`, selected IO134/third-Hubbard/singular/duplicate cases in `physcal_issue181` | 36/36 PASS, 27 explicitly outside selection; `9a449fa7-2cd6-4ac6-a1c7-dbb53e4c69ec`, 0.243s |
| Core lib `-E 'test(io::)'` | 6/6 PASS, 203 outside selection; `fd275c50-1654-4e20-9a5f-9131f23dd9c4`, 0.025s |
| CLI `--test issue184_fixed_file_boundaries` | 1/1 PASS covering four actual subprocess cases; `e8e4cd94-cb3c-4ac9-b7db-27f67d42fb89`, 0.057s |
| `git diff --check` | terminal0 |

Core combined filter:

```sh
cargo nextest run --locked --cargo-profile test-fast -p mvmc-core \
  --test output_blocks --test issue184_public_rejection_boundaries \
  --test issue184_slater_contracts --test physcal_issue181 \
  -E 'binary(output_blocks) | binary(issue184_public_rejection_boundaries) | binary(issue184_slater_contracts) | test(original_io134) | test(third_hubbard_lanczos) | test(singular_lanczos_alpha) | test(no_factored_terms)' \
  --no-fail-fast --retries 0
```

Source SHA-256 at these checks:

```text
3db39cf9cf69ab1124d378c91ab820ccce2be9ad75a4741ad2322b9da3eda6e4  crates/mvmc-core/src/io.rs
c922b28f4fcd5f78f53507f0fd2cf6e0d40eb0adcf596ffe7355c801b364e0e0  crates/mvmc-core/tests/output_blocks.rs
5f0960d3bdd368dc8564fcadc713b249d39dd3c6d7dbf00e3427e1c30eb5dc42  crates/mvmc-core/tests/physcal_issue181.rs
f4fe7edda438d17dd20a48300211c45d0680cf19498573f5b37fb6f1f0cab212  crates/mvmc-core/tests/issue184_public_rejection_boundaries.rs
5b8e45aa1e255bf4c513de24722badd883bf6144ed9bcac6190df377aaba2b4a  crates/mvmc-core/tests/issue184_slater_contracts.rs
19ffb7fba0ba079ee39cd6941cf7b4c154a7975e2dacf34333c8cabec5219ce0  crates/mvmc-cli/tests/issue184_fixed_file_boundaries.rs
368f15c43206db983d87d71cfc7a35747d104be5a0b2c8ea825726b3ed9e0ba4  scripts/generate_physcal_181.jl
```

Parent separately reported restored workspace check and fifteen SR tests passed.
Those are parent evidence, not this owner's full-workspace validation. Focused
builds emit the unrelated tenferro `Atomic::fetch_update` deprecation warning;
this report does not claim all-target Clippy is clean.

## Remaining acceptance / coordination

1. Following parent approval, **one** integrated #181 binary run completed:
   `ea2061c5-f99e-4ca8-8ef0-b1e6fa0df790`, handle `79674` terminal0, isolated TMPDIR
   `/tmp/mvmc-181-integrated.y0cVGV`. Command adds
   `MVMC_RS_PHYSCAL_181=1 --run-ignored all` to the locked/test-fast full
   `--test physcal_issue181` selection. Final result: **42/42 PASS, 0 skipped**,
   84.561s. `non_interall_lanczos_modes_one_and_two_write_the_defined_files`
   completed in 84.560s. The six models, combinations/two frames and all-six-term
   cases passed in that run. This is this binary's full gate, not all #181/#185
   acceptance or the later raw-Green observer patch.
   Parent's separate exchange Lanczos callback test is not selected or restarted.
   Launch-observed run source SHA is
   `a4f586e9f7b7a8bcd0ea87b512d9d698631606dbf859192dab9a58e12a293ad3`;
   IO/test hashes remain those above. At harvest shared `run.rs` had changed to
   `0c7896619d58d7f4379dd4d9b08b666b65735ca714a531da4c78f4e100ea9da3`.
   Therefore this is a completed compiled-run result, **not** a source-frozen
   latest-worktree proof or validation of that later patch/raw-Green hook. Do
   not assign a retroactive source SHA to the compiled binary; final integrated
   validation awaits the coordinated source freeze.
2. Actual **raw Green before division inside the public runner** still awaits
   the runner owner's observation hook. Current evidence has independent raw
   Julia sums, actual C normalization, actual runner raw-energy recording and
   normalized Green comparisons; the fixture-fed averaging test is not an
   observed raw runner Green checkpoint. See native-c-weighted-green README.
3. Accepted S196/M0998 singleton OptTrans grouped PhysCal now has bounded actual
   MPI proof: parent reports Wegener's corrected frozen-external run `98d22d`
   terminal0, worlds2/4 × workers1/2/4 × two repeats, through public PhysCal.
   This resolves the pending frozen-checkpoint runtime evidence, **not** current
   main or whole #179 verification. The earlier filename bug was test-only and
   was corrected; external handle4573 was never restarted by this owner.
   Exact MPI source/run metadata belongs to Wegener's #179 packaging report.
4. Full #185 serial/CLI/native-MPI/thread 20-step matrix, remaining CG consumers,
   all-feature/Linux/macOS portability, workspace/Clippy/fmt/doctests and final
   #184 per-assertion inventory are broader parent/other-owner acceptance.
5. Meaningful Rust error context/settings are mapped separately from Julia
   literal exception strings. Do not implement cosmetic exception copying to
   close rows; private helper tuple/log/object-identity differences are explicit
   architecture distinctions. Original S104 exact-input lifecycle is resolved,
   not conditional, but is not a full sampler run.

Full native C composed-input sampling is stronger optional evidence, not a
blanket prerequisite for every cell; C supported-input and kernel authorities
remain required and separately labelled. No native C full-combination trajectory
claim is made. Owner handle `79674` is terminal0; no owner live process remains.
Retain its isolated output for review; no duplicate full gate was launched.

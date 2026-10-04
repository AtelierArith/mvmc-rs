---
date: 2026-07-08
datetime: 2026-07-08 17:03 JST
model: Claude Fable 5 (claude-fable-5); GPT-5 Codex review edit
status: review
topic: GeneralRBM_cmp ctest fixture 有効化 (Julia-mVMC release/v0.5.0 commit 2518320)
---

# GeneralRBM_cmp ctest fixture 有効化レビュー

## 対象

- Branch: `Julia-mVMC` `release/v0.5.0`
- Commit: `2518320` feat(rbm): enable GeneralRBM ctest fixture
- 変更範囲:
  - `Julia-mVMC/MVMCOptimizers.jl/src/initial_params.jl`
    （`_load_para_triples!` の RBM block 読み込み対応）
  - `Julia-mVMC/MVMCOptimizers.jl/test_unit/test_unit_read_opt_para.jl`
    （RBM 読み順・duplicate idx を固定する unit test）
  - `Julia-mVMC/test/integration/ctest_models.jl`
    （`general_rbm_cmp` を deferred から standard へ移動）
  - `Julia-mVMC/test/integration/reference/general_rbm_cmp/`
    （inputs 一式 + `ctest_ref/ref_mean.dat` / `ref_std.dat`）
  - `Julia-mVMC/CHANGELOG.md`,
    `Julia-mVMC/docs/manual/02_input_files.md`,
    `Julia-mVMC/docs/manual/03_optimization.md`,
    `Julia-mVMC/docs/manual/05_compatibility.md`,
    `Julia-mVMC/test/integration/reference/README.md`
- 関連ドキュメント: `docs/roadmaps/2026-07-08-c-mvmc-v13-parity-roadmap.html`
  （GeneralRBM_cmp カードを Done、legend を Done 7/10 · Active 1/10 · Partial 0/10 · Pending 2/10 へ更新済み）

## 結論

**問題なし。** `initial.def` / `zqp_opt.dat` の RBM triples 読み込みは C-mVMC の
parameter layout に忠実で、unit / integration の両ゲートが green。軽微な指摘 1 件
（helper 重複）のみで、修正は必須ではない。

## 確認内容

### 1. C layout 忠実性

読み順 `NProj -> NRBM -> NSlater -> NOptTrans` を C 実装と突き合わせて確認した。

- 全体配置: `mVMC/src/mVMC/setmemory.c:330-334` で
  `Proj = Para`, `RBM = Para + NProj`,
  `Slater = Para + NProj + FlagRBM*NRBM + NProjBF`,
  `OptTrans = Para + NProj + FlagRBM*NRBM + NProjBF + NSlater`。
  `NPara` の構成は `mVMC/src/mVMC/readdef.c:901`
  （`NProj + NSlater + NOptTrans + NProjBF + NRBM * FlagRBM`）。
  Julia loader の読み順はこれと一致する。
- NRBM block 内の layer 順: `mVMC/src/mVMC/rbm.c:202-203` の pointer offset
  （`RBM_Hidden = RBM + NRBM_PhysLayerIdx`,
  `RBM_PhysHidden = RBM + NRBM_PhysLayerIdx + NRBM_HiddenLayerIdx`）より
  PhysLayer -> HiddenLayer -> PhysHidden。
- 各 layer 内の subsection 順: `mVMC/src/mVMC/readdef.c:1074-1091` の `fidx`
  offset（`NProj`, `NProj+NChargeRBM_PhysLayerIdx`,
  `NProj+NChargeRBM_PhysLayerIdx+NSpinRBM_PhysLayerIdx`）より
  Charge -> Spin -> General。
- Julia 側の section 列挙
  `Julia-mVMC/MVMCOptimizers.jl/src/stochastic_opt.jl:28-40`
  （`_rbm_parameter_sections`）は
  PhysLayer(C,S,G) -> HiddenLayer(C,S,G) -> PhysHidden(C,S,G) の 9-tuple で
  上記 C の並びと完全一致。

### 2. 宣言量と消費量の整合

- `expected_floats` の `n_rbm` は parser 側
  `Julia-mVMC/MVMCExpertModeParsers.jl/src/utils/read_input_parameters.jl:132-149`
  （`_rbm_section_nparam` = 空なら 0、それ以外は max idx + 1、の section 和）。
- loader 側の実消費は
  `Julia-mVMC/MVMCOptimizers.jl/src/initial_params.jl` の
  `_initial_param_section_width`（同じく max idx + 1）で section ごとに読む。
- 両者が同一定義のため、宣言 float 数と消費 triples 数はずれない。
  too-short / trailing-garbage の検証が mutation より先に走る
  validate-before-commit の contract も維持されている。

### 3. duplicate RBM idx の保持

- `_set_indexed_terms_from_values!` は section_values から各 term へ
  `term.idx` で scatter するため、同一 idx を持つ複数 term は同じ値を受け取る。
  C の InParam scatter と同じ semantics。
- unit test
  `Julia-mVMC/MVMCOptimizers.jl/test_unit/test_unit_read_opt_para.jl`
  の新 testset「RBM block loads before Slater」が、idx 0 を共有する
  `GeneralRBMPhysLayerTerm` 2 件が同値になることと、
  NProj(3) -> NRBM(5) -> NSlater(2) の境界（n = 10）を固定している。

### 4. テスト実行結果（本レビューで再実行）

- unit: `julia --project=. test_unit/test_unit_read_opt_para.jl`
  （`MVMCOptimizers.jl` 直下）— 全 testset pass。
  新 testset「RBM block loads before Slater」9/9。
- integration:
  `JULIA_MVMC_CTEST_MODELS=general_rbm_cmp julia --project=@. test/integration/ctest_equivalent.jl`
  — **6/6 pass**（58.3 秒）。
  `NPara=102 (NProj=6 + NRBM=86 + NOrbitalIdx=10 + NOptTrans=0)` と
  RBM block が正しく計上されている。

### 5. fixture / ドキュメント整合

- `general_rbm_cmp/ctest_ref/` が `ref_mean.dat` / `ref_std.dat` のみなのは
  既存 fixture（例: `heisenberg_chain_real`）と同じ規約。provenance と
  regenerate 手順（`python3 runtest.py GeneralRBM_cmp`）は
  `Julia-mVMC/test/integration/reference/README.md` に追記済み。
- `Julia-mVMC/docs/manual/05_compatibility.md` から「RBM block refused」の
  記述と Known limitation 項が削除され、`read_initial_def!` の対応範囲が
  projection + RBM + Slater + OptTrans に更新されている。
- `Julia-mVMC/CHANGELOG.md` に Added 項目あり。
- `docs/roadmaps/2026-07-08-c-mvmc-v13-parity-roadmap.html` は
  Done 7/10 · Active 1/10 · Partial 0/10 · Pending 2/10、
  GeneralRBM_cmp カード Done、matrix 行に読み順と duplicate idx 保持を明記。

## 指摘事項

### 軽微（対応任意）: section 幅 helper の 3 重定義

max idx + 1 の section 幅計算が 3 箇所に重複している:

- `Julia-mVMC/MVMCExpertModeParsers.jl/src/utils/read_input_parameters.jl:132`
  （`_rbm_section_nparam`）
- `Julia-mVMC/MVMCOptimizers.jl/src/stochastic_opt.jl:42`
  （`_parameter_section_width`）
- `Julia-mVMC/MVMCOptimizers.jl/src/initial_params.jl`
  （`_initial_param_section_width`、本 commit で追加）

`expected_floats` と実消費量の整合はこの 3 定義が一致していることに依存する。
現状は全て同一定義で問題ないが、将来 sparse index を許す等の変更を行う場合は
1 箇所（package 境界を跨ぐため、最低でも MVMCOptimizers 内の 2 つ）に
集約してから変更するのが安全。今回の release では対応不要。

## 備考

- 本 commit は wavefunction 側の RBM 実装（`stochastic_opt.jl` の RBM section
  参照など）が既に存在する前提で、parameter file I/O と fixture 昇格のみを
  追加している。RBM 計算本体の正しさは ctest-equivalent gate（C reference との
  最終 opt summary 比較）で間接的に検証されている。
- unit test は General 系 section のみ使用しており、Charge/Spin/General の
  interleave 順そのものは test で固定されていない。ただし section tuple が
  C の `fidx` offset と一致していることをソース対照で確認済み。

## 2026-10-04 Rust General限定5-cell実行証拠（追補）

以下は上記2026-07-08の歴史レビューとは別reference・別実行の限定追補。原本文・global status・台帳rowは変更しない。Related to #183; Related to #185; no closure. PR312はcommit `c3909f86d8825811da2639c9e79472d1e7d30ac0`でmerge済み。以下の実行証拠はmerge前head06f850の履歴として保持する。

### Shared execution identity and authority

Actual head `06f8509d8d0a3f11ac9782be38abd6e74426ec7c`, [dispatch37197873853](https://github.com/AtelierArith/mvmc-rs/actions/runs/37197873853), attempt1; [General job111423423107](https://github.com/AtelierArith/mvmc-rs/actions/runs/37197873853/job/111423423107), [aggregate111423698205](https://github.com/AtelierArith/mvmc-rs/actions/runs/37197873853/job/111423698205). Actual Rust identities in `crates/mvmc-core/tests/ctest_general_reference.rs`: prefix identity `corrected_general_all_four_prefixes_match_independent_reference`; public identity `corrected_general_twenty_step_public_runner_is_repeatable`.

Reference: checked-in `tests/fixtures/ctest_general_pr54_3d0fd263/`, acquisition reference commit `3d0fd2638fd34de2a8f9609fcfaac2504caf02d2`, historical Julia1.13.1 ILP64 OpenBLAS acquisition, Manifest SHA `09ebd06dab244510094b99fe7c6efa2fe7a3d22221d1336b951123a5a6e8befc`. This run used offline fixtures only: Julia runtime NotRun, version/BLAS null; no C/Julia oracle. Preserve per-fixture provenance, not a newly acquired numerical reference claim.

Common actual settings: seed12395, samples100, frames1, warmup10, interval1, NSRCG0, NStore1, complex mode, ranks1/workers1, default features/test-fast/locked. Original Modpara declaration steps1500/window100 is unchanged and separate from effective gate overrides below. Actual OpenBLAS0.3.26 Haswell LP64 threads1; resolved library SHA `bfc7492adbf84a8f567720a9e1fae2afc18f3d817da233e7f4d453683485308e`. Rust1.99.0 b940084d7/LLVM23.1.1/Linuxx86_64.

### Five-cell evidence join (not five separate Rust tests)

| Scoped cell | Effective steps/window | Calls/repeats | Actual evidence | Result and limits | Owner |
|---|---|---|---|---|---|
| General prefix1 | 1/1 | 1 | prefix identity; raw Generalprefix1 and settings kind=prefix steps=1 | Included in prefix test PASS1.233s; original exact raw/cursor/count/next624/config assertions retained | Goodall implementation; parent review; #183/#185 |
| General prefix2 | 2/2 | 1 | prefix identity; raw Generalprefix2 and settings kind=prefix steps=2 | Same prefix test PASS; not an additional Rust identity | Same |
| General prefix3 | 3/3 | 1 | prefix identity; raw Generalprefix3 and settings kind=prefix steps=3 | Same prefix test PASS; original numerical budgets unchanged | Same |
| General prefix20 | 20/20 | 1 | prefix identity; raw Generalprefix20 and settings kind=prefix steps=20 | Same prefix test PASS; not whole model/settings matrix | Same |
| General public20 | 20/20 | 2 fresh high-level calls | public identity; settings kind=public steps=20 repeats=2 after both calls/output assertions | Public test PASS1.745s; marker reports call arguments/input, not internal public RNG instrumentation | Same |

The raw nextest result is **two tests passed,16 excluded**, UUID `6d5333dc-a6ad-4ea1-90a3-d58d7585165d`; exclusions are not model PASS or ExplicitSkip. No tolerance, RNG, independent assertion or fixture edits support these joins.

Actual selected command (one invocation serving all five cells):

```sh
cargo nextest run --locked -p mvmc-core --cargo-profile test-fast --test ctest_general_reference --run-ignored only -E 'test(=corrected_general_all_four_prefixes_match_independent_reference) | test(=corrected_general_twenty_step_public_runner_is_repeatable)' --no-fail-fast --retries 0 --success-output immediate --failure-output immediate
```

Actual environment includes `MVMC_RS_CTEST_GENERAL=1` and BLAS thread controls1; backend actual_threads independently reports1. commands.json records start/end, Completed, returncode0, exact selected identities/profile/features. Version/discovery commands are not selected completion. Selected Cargo ELF recorded SHA `1eab64b0be0dc0caf3cbb0629241f88b08975ac268f3a866a2ad9e15fca89be6`; binary closure before/after identical. No uploaded ELF independently rehash claim.

### Artifact pins and remaining classifications

Local retained HOST packet `/tmp/issue183-general-06f850-actual.qduCpV` (temporary filesystem, not a durable attachment); original downloaded artifacts remain unchanged. External acquisition: [plan artifact11302090205](https://api.github.com/repos/AtelierArith/mvmc-rs/actions/artifacts/11302090205), [General artifact11301389785](https://api.github.com/repos/AtelierArith/mvmc-rs/actions/artifacts/11301389785), [aggregate artifact11300719852](https://api.github.com/repos/AtelierArith/mvmc-rs/actions/artifacts/11300719852), associated with the linked exact run/attempt/head; GH retention/authentication limits apply. General artifact `evidence/artifacts.json` supplies all28 member SHA records (manifest SHA `395c2cdc94850ecc2c08f2d195ed4cb7632f233548891f7561ba7567eb6bb0c0`), not merely workflow status. Raw general.stderr SHA `a6cf1a7199488d2ddab121e34f240647d9b4b16ebb047ca8444dc38ae385025a`; five settings JSON SHA `746c9da4533b1f35e368df3cf8f26a2bd97f9a970388ac9baaf7d4740bab527c`; actual commands SHA `245fcc928821d859afd40b6f23e761afc849c981a1f129b8858435d6badb01e3`; backend SHA `cb86986d76af09aa543db1fefb4ba83e844686e056ab2b4343f7d59ff1439ae1`; closure-posts SHA `e6b302e7f2528a4a0bc34c0e3169b167c6853faacceae54a7be25f337b70b5bf`. All28 stored artifact hash records replay0; all source/fixture/binary JSON before-after comparisons0. terminal0/Pass, three closure records UNCHANGED/errornull.

Aggregate is GeneralPass with **comparison_evidence Unverified / numeric_reference_comparisons null**. Preserve that exact classification: successful test assertions are not permission to set whole comparison inventory Verified. Lanczos/MPI/thread remain NotRun for this dispatch; their historical receipts are not erased or relabelled current-head results. Full13 model matrix, full public API inventory, genuine MPI/public-RNG coverage and #183/#185 overall acceptance remain outside this bounded join.

Keep historical2cc failures: V2 inventory1 and V3 fmt1, and separate V4 compile/LIST/lint prior/post0. They are not current06 model execution. Ordinary PR312 CI37197823229 remains a separate requirement; lint/docsSUCCESS and other jobs pending at acquisition. No ordinary CI PASS inferred from optional workflow.

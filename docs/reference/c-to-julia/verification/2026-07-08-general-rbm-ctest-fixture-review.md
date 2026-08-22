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

---
date: 2026-08-22
datetime: 2026-08-22 16:34 JST
model: GPT-5 Codex
status: reference
topic: C-mVMC to Julia-mVMC porting document archive
summary: |
  C-mVMC v1.3 までの Rust 移植計画に再利用するため、C→Julia 移植時の
  roadmap、設計、検証、デバッグ記録、公式仕様の計47点を
  原文スナップショットとして整理する。
---

# C-mVMC → Julia-mVMC 移植資料索引

## 目的

このディレクトリは、Rust-mVMC を C-mVMC `v1.3.0` まで追従させる際の参考資料を保存する。
現在の Rust port は `docs/PORTING_PLAN.md` で Julia-mVMC v0.1 相当を対象としているため、
Julia-mVMC v0.5.0 までに蓄積された C-reference parity の設計・検証方法を次期計画へ引き継ぐ。

主な基準点:

- C-mVMC: tag `v1.3.0`, commit `d73d06bd529d3b2573f38eb5817c4a5f52971006`
  （2024-10-05）
- Julia-mVMC: tag `v0.5.0`, commit `c2ea432785bc14364a3cd5e9eef44db464289cc9`
  （2026-07-09）
- Julia-mVMC bilingual manual snapshot: commit
  `72a8ab32ea6e6f9703eec74db21aed5c3ff8c112`（2026-07-09）

## 保存方針

- 各ファイルは原典を編集しない原文スナップショットとする。
- 当時のローカル path、branch、commit、HPC 環境、未完了項目も履歴として保持する。
- 設計正本は通常の `docs/{plans,specs}/` だけでなく、当時の運用上
  `docs/superpowers/{plans,specs}/` に置かれたものも収録対象とする。
- 記載された「現在」「次の作業」「未対応」は原典更新日時点の状態であり、Rust 側の現在の判断を拘束しない。
- 原典内の相対リンクは元の配置を前提としているため、この archive 内では解決しない場合がある。
- Rust 実装へ反映する前に、C `v1.3.0`、Julia `v0.5.0`、現在の Rust source の三者で再監査する。
- `SOURCE_SHA256SUMS` は収録した原文スナップショットの内容確認用である。

## 推奨読書順

1. [C v1.3 parity dashboard](roadmaps/2026-07-08-c-mvmc-v13-parity-roadmap.html)
2. [Julia v0.4/v0.5 roadmap](roadmaps/2026-06-14-julia-mvmc-v0.4-summary-and-v0.5-roadmap.md)
3. [初期 VMCParaOpt 移植記録](historical/MVMCOptimizers-TODO.md)
4. [integration reference data の作成・再生成方法](verification/integration-reference-data.md)
5. 対象機能に対応する `plans/` と `verification/` の資料
6. C v1.3公式仕様の [algorithm](c-mvmc-v1.3/algorithm.ja.rst)、
   [expert input](c-mvmc-v1.3/expert.ja.rst)、[output](c-mvmc-v1.3/output.ja.rst)

## 資料一覧

### Roadmap・全体像

| 収録ファイル | 原典 | 対象 | 原典更新日 | 用途 |
|---|---|---|---|---|
| `roadmaps/2026-07-08-c-mvmc-v13-parity-roadmap.html` | `Shin-mVMC/docs/roadmaps/2026-07-08-c-mvmc-v13-parity-roadmap.html` | C v1.3 / Julia v0.5 release直前 | 2026-07-08 | 10のparity trackと残件を俯瞰するdashboard |
| `roadmaps/2026-06-14-julia-mvmc-v0.4-summary-and-v0.5-roadmap.md` | `Shin-mVMC/docs/plans/2026-06-14-julia-mvmc-v0.4-summary-and-v0.5-roadmap.md` | C v1.2～v1.4監査 / Julia v0.4～v0.5 | 2026-07-08 | 機能依存、取り込み順、unsupported scopeの正本 |

### 初期移植履歴

| 収録ファイル | 原典 | 対象 | 原典更新日 | 用途 |
|---|---|---|---|---|
| `historical/MVMCOptimizers-TODO.md` | `Shin-mVMC/private-mVMC/MVMCOptimizers.jl/TODO.md` | 初期VMCParaOpt移植 | 2026-02-02 | RNG、SR、Pfaffian、parameter layoutの一致記録 |
| `historical/MVMCOptimizers-VMCPhysCal-PLAN.md` | `Shin-mVMC/private-mVMC/MVMCOptimizers.jl/PLAN.md` | 初期VMCPhysCal移植 | 2026-02-02 | C処理フロー、Julia module対応、段階実装、検証項目 |
| `historical/HeisenbergChain-cmp-debug-progress.md` | `Shin-mVMC/private-mVMC/docs/progress_1.md` | complex Heisenberg parity debug | 2026-02-04 | RNG、burn sample、BLAS、丸め差からtrajectory分岐までの切り分け |

### 機能別plan

| 収録ファイル | 原典 | 対象 | 原典更新日 | 用途 |
|---|---|---|---|---|
| `plans/2026-05-25-julia-mvmc-c-compatible-timer-plan.md` | `Shin-mVMC/docs/plans/2026-05-25-julia-mvmc-c-compatible-timer-plan.md` | C-compatible timer / Julia v0.2 | 2026-05-25 | `MVMC_C_TIMER`によるsection timerとC互換出力の設計 |
| `plans/2026-06-05-julia-mvmc-physcal-plan1-greentwoex-parser.md` | `Shin-mVMC/docs/superpowers/plans/2026-06-05-julia-mvmc-physcal-plan1-greentwoex-parser.md` | C v1.3 PhysCal / non-FSZ TwoBodyGEx | 2026-06-05 | `greentwoex.def` parser、index基準、入力validationの実装plan |
| `plans/2026-06-05-julia-mvmc-physcal-plan2-factored-green-compute.md` | `Shin-mVMC/docs/superpowers/plans/2026-06-05-julia-mvmc-physcal-plan2-factored-green-compute.md` | C v1.3 PhysCal / factored Green | 2026-06-05 | factored two-body Green計算とC semanticsの実装plan |
| `plans/2026-06-05-julia-mvmc-physcal-plan3-fixtures-and-e2e.md` | `Shin-mVMC/docs/superpowers/plans/2026-06-05-julia-mvmc-physcal-plan3-fixtures-and-e2e.md` | C v1.3 PhysCal / C-reference gate | 2026-06-05 | runner、比較helper、fixture、end-to-end gateの実装plan |
| `plans/2026-06-05-julia-mvmc-physcal-factored-green-and-fixtures-design.md` | `Shin-mVMC/docs/superpowers/specs/2026-06-05-julia-mvmc-physcal-factored-green-and-fixtures-design.md` | C v1.3 PhysCal / factored Green | 2026-06-05 | 非FSZ TwoBodyGExとreference fixtureの設計正本 |
| `plans/2026-06-06-julia-mvmc-dh2-dh4-compat-plan.md` | `Shin-mVMC/docs/superpowers/plans/2026-06-06-julia-mvmc-dh2-dh4-compat-plan.md` | C v1.3 DH2/DH4 / Julia v0.3 | 2026-06-06 | doublon-holon projectionのparser、layout、runtime、fixture計画 |
| `plans/2026-06-06-julia-mvmc-dh2-dh4-dh2-runtime-plan.md` | `Shin-mVMC/docs/superpowers/plans/2026-06-06-julia-mvmc-dh2-dh4-dh2-runtime-plan.md` | C v1.3 DH2 runtime | 2026-06-06 | DH2 count・ratio更新とC index式の実装plan |
| `plans/2026-06-10-julia-mvmc-v0.4-mpi-design.md` | `Shin-mVMC/docs/specs/2026-06-10-julia-mvmc-v0.4-mpi-design.md` | Julia v0.4 MPI / Rust Phase 7 | 2026-06-10 | communicator、reduction、RNG、serial fallbackを含むMPI設計正本 |
| `plans/2026-06-10-julia-mvmc-v0.4-mpi-r0-plan.md` | `Shin-mVMC/docs/plans/2026-06-10-julia-mvmc-v0.4-mpi-r0-plan.md` | Julia v0.4 MPI R0 | 2026-06-10 | MPI基盤の段階実装、test、導入順序 |
| `plans/2026-06-11-julia-mvmc-v0.4-mpi-detection-policy.md` | `Shin-mVMC/docs/specs/2026-06-11-julia-mvmc-v0.4-mpi-detection-policy.md` | Julia v0.4 MPI launch detection | 2026-06-11 | MPI起動検出とserial fallbackのpolicy |
| `plans/2026-06-22-julia-mvmc-v0.5-cg-implementation-plan.md` | `Shin-mVMC/docs/plans/2026-06-22-julia-mvmc-v0.5-cg-implementation-plan.md` | C SR-CG / Julia v0.5 | 2026-06-22 | SR-CG solver、residual、fallback、C-reference検証の実装plan |
| `plans/2026-06-22-julia-mvmc-v0.5-nsplit-nstore-plan.md` | `Shin-mVMC/docs/plans/2026-06-22-julia-mvmc-v0.5-nsplit-nstore-plan.md` | `NSplitSize` / `NStore` / Julia v0.5 | 2026-06-22 | SR-CG分割・保存semanticsとunsupported combinationの整理 |
| `plans/2026-06-23-julia-mvmc-full-lanczos-plan.md` | `Shin-mVMC/docs/plans/2026-06-23-julia-mvmc-full-lanczos-plan.md` | C v1.2 Full Lanczos / Julia V05-1 | 2026-07-01 | mode 1/2の段階実装、出力契約、C-reference gate |
| `plans/2026-06-30-julia-mvmc-v05-3-nsplit-standard-projection-design.md` | `Shin-mVMC/docs/superpowers/specs/2026-06-30-julia-mvmc-v05-3-nsplit-standard-projection-design.md` | Julia V05-3 / grouped QP split | 2026-06-30 | standard projectionに対する`NSplitSize`分割の設計正本 |
| `plans/2026-06-30-julia-mvmc-v05-3-nsplit-standard-projection-plan.md` | `Shin-mVMC/docs/superpowers/plans/2026-06-30-julia-mvmc-v05-3-nsplit-standard-projection-plan.md` | Julia V05-3 / grouped QP split | 2026-06-30 | parserからkernel、MPI reduction、fixtureまでの実装plan |
| `plans/2026-07-07-julia-mvmc-v05-3-srcg-nsplit-design-plan.md` | `Shin-mVMC/docs/plans/2026-07-07-julia-mvmc-v05-3-srcg-nsplit-design-plan.md` | SR-CG + `NSplitSize>1` | 2026-07-07 | 壊れたC oracleを模倣せずcorrected semanticsを採る判断 |
| `plans/2026-07-07-julia-mvmc-v05-4-physcal-nsplit-plan.md` | `Shin-mVMC/docs/plans/2026-07-07-julia-mvmc-v05-4-physcal-nsplit-plan.md` | PhysCal grouped split | 2026-07-07 | sz-conserved normal Greenに限定したMPI split設計 |
| `plans/2026-07-07-julia-mvmc-v05-4-fsz-twobodygex-plan.md` | `Shin-mVMC/docs/plans/2026-07-07-julia-mvmc-v05-4-fsz-twobodygex-plan.md` | FSZ/general-orbital TwoBodyGEx | 2026-07-08 | FSZ Green dispatch、fixture、negative gateの設計 |
| `plans/2026-07-07-julia-mvmc-full-lanczos-mode2-tier-b-plan.md` | `Shin-mVMC/docs/plans/2026-07-07-julia-mvmc-full-lanczos-mode2-tier-b-plan.md` | Full Lanczos mode 2 Tier B | 2026-07-07 | TwoBodyGEx gateを含むFull Lanczos後続plan |

### 検証・fixture・実装監査

| 収録ファイル | 原典 | 対象 | 原典更新日 | 用途 |
|---|---|---|---|---|
| `verification/issue-184-evidence-matrix.md` | #184/#185 GitHub API and current repository inventory | Issue #184 Julia exports/tests/examples and Rust caller evidence foundation | 2026-10-03 | #174–#182 owners; independent executed-evidence criteria; reference/settings/command/result ledger; partial/unrun runner/CLI/callback and #181 oracle gaps |
| `verification/15_model_ctest_integration_plan.md` | `Shin-mVMC/private_docs/15_model_ctest_integration_plan.md` | C v1.3 15-model ctest / Julia integration | 2026-05-14（mtime） | C ctestと同じ統計判定、対象model、fixture移植、段階gateの計画 |
| `verification/15_model_ctest_integration_implementation_summary.md` | `Shin-mVMC/private_docs/15_model_ctest_integration_implementation_summary.md` | C v1.3 15-model ctest / Julia integration | 2026-05-14（mtime） | 12 standard modelの実装結果とRBM/Lanczosの保留条件 |
| `verification/testing-plan-julia-mvmc.md` | `Shin-mVMC/docs/testing_plan_julia_mvmc.md` | Julia unit/contract/integration | 2026-02-13 | test階層と、差分を最小再現へ落とす運用 |
| `verification/2026-06-06-julia-mvmc-dh2-dh4-compat-plan-review.md` | `Shin-mVMC/docs/reviews/2026-06-06-julia-mvmc-dh2-dh4-compat-plan-review.md` | C v1.3 DH2/DH4 / Julia v0.3 | 2026-06-06 | Cのindex式・layout offsetをground truthと照合したplan監査 |
| `verification/2026-06-12-julia-mvmc-nsrcg1-serial-cg-residual-rootcause.md` | `Shin-mVMC/docs/reports/2026-06-12-julia-mvmc-nsrcg1-serial-cg-residual-rootcause.md` | C/Julia `NSRCG=1` serial CG | 2026-06-12 | SR-CG residual差のroot causeと数値検証上の注意 |
| `verification/integration-reference-data.md` | `Julia-mVMC/test/integration/reference/README.md` | Julia v0.5 C-reference fixtures | 2026-07-08 | model一覧、provenance、再生成手順、tolerance |
| `verification/2026-07-08-general-rbm-ctest-fixture-review.md` | `Shin-mVMC/docs/reviews/2026-07-08-general-rbm-ctest-fixture-review.md` | C v1.3 GeneralRBM / Julia v0.5 | 2026-07-08 | RBM triples layoutとctest-equivalent gateの監査 |
| `verification/2026-06-20-c-mvmc-nstore-nsplit-srcg-investigation.md` | `Dev_mVMC/docs/2026-06-20-c-mvmc-nstore-nsplit-srcg-investigation.md` | C develop `8c7f95a` | 2026-06-20 | `NStore`/`NSplitSize`/SR-CGのsilent wrong調査とJuliaへの含意 |

### 性能

| 収録ファイル | 原典 | 対象 | 原典更新日 | 用途 |
|---|---|---|---|---|
| `performance/2026-05-26-c-vs-julia-mvmc-bottleneck-and-fix.md` | `Shin-mVMC/docs/reports/2026-05-26-c-vs-julia-mvmc-bottleneck-and-fix.md` | C/Julia Pfaffian kernel | 2026-05-27 | 同一algorithmでもSIMD・compiler・BLAS条件で性能が変わる事例 |
| `performance/2026-06-08-julia-mvmc-threading-prep-plan.md` | `Shin-mVMC/docs/plans/2026-06-08-julia-mvmc-threading-prep-plan.md` | Julia threading / Rust Phase 6 | 2026-06-08 | thread local state、reduction、再現性を保つ並列化準備plan |
| `performance/2026-06-09-maincal-thread-race-rootcause-review.md` | `Shin-mVMC/docs/reviews/2026-06-09-maincal-thread-race-rootcause-review.md` | Julia `VMCMainCal` threading | 2026-06-09 | shared buffer raceのroot causeとsequential既定の根拠 |
| `performance/2026-06-15-julia-mvmc-hubbard-performance-gap-plan.md` | `Shin-mVMC/docs/plans/2026-06-15-julia-mvmc-hubbard-performance-gap-plan.md` | Hubbard real-mode C/Julia gap | 2026-06-15 | workload分解、profiling、correctness gateを定めた最適化plan |
| `performance/2026-06-17-julia-mvmc-hubbard-calham1-pfproj-optimization.md` | `Shin-mVMC/docs/reports/2026-06-17-julia-mvmc-hubbard-calham1-pfproj-optimization.md` | Hubbard `CalHamiltonian1` / Pfaffian | 2026-06-17 | Pfaffian/projector hot pathの測定と最適化記録 |
| `performance/2026-06-17-julia-mvmc-hubbard-locenergy-slater-calham1-optimization-record.md` | `Shin-mVMC/docs/reports/2026-06-17-julia-mvmc-hubbard-locenergy-slater-calham1-optimization-record.md` | Hubbard LocEnergy / Slater / CalHamiltonian1 | 2026-06-17 | 複数kernelにまたがる性能変更とparity確認の記録 |
| `performance/2026-06-17-julia-mvmc-hubbard-optimization-final-summary.md` | `Shin-mVMC/docs/reports/2026-06-17-julia-mvmc-hubbard-optimization-final-summary.md` | Hubbard real-mode optimization | 2026-06-17 | 最適化結果、残存gap、再現条件の最終summary |

### Julia v0.5公開資料

| 収録ファイル | 原典 | 対象 | 原典更新日 | 用途 |
|---|---|---|---|---|
| `julia-v0.5/CHANGELOG.md` | `Julia-mVMC/CHANGELOG.md` | Julia v0.1～v0.5 | 2026-07-08 | 各releaseで追加されたC parity機能の時系列 |
| `julia-v0.5/compatibility.md` | `Julia-mVMC/docs/src/en/compatibility.md` | Julia v0.5 + bilingual manual commit | 2026-07-09 | C/Julia関数対応、公開gate、未対応組み合わせ |

### C-mVMC v1.3公式仕様

| 収録ファイル | 原典 | 対象 | 原典更新日 | 用途 |
|---|---|---|---|---|
| `c-mvmc-v1.3/algorithm.en.rst` | `mVMC@v1.3.0:doc/en/source/algorithm.rst` | C v1.3 algorithm baseline | 2024-10-05 | 英語版の波動関数、SR、RBM、Lanczos数式仕様 |
| `c-mvmc-v1.3/algorithm.ja.rst` | `mVMC@v1.3.0:doc/ja/source/algorithm.rst` | C v1.3 algorithm baseline | 2024-10-05 | 日本語版の波動関数、SR、RBM、Lanczos数式仕様 |
| `c-mvmc-v1.3/expert.en.rst` | `mVMC@v1.3.0:doc/en/source/expert.rst` | C v1.3 expert input | 2024-10-05 | 英語版input file仕様。General RBMを含む |
| `c-mvmc-v1.3/expert.ja.rst` | `mVMC@v1.3.0:doc/ja/source/expert.rst` | C v1.3 expert input | 2024-10-05 | 日本語版input file仕様。General RBMを含む |
| `c-mvmc-v1.3/output.en.rst` | `mVMC@v1.3.0:doc/en/source/output.rst` | C v1.3 output | 2024-10-05 | 英語版output contract |
| `c-mvmc-v1.3/output.ja.rst` | `mVMC@v1.3.0:doc/ja/source/output.rst` | C v1.3 output | 2024-10-05 | 日本語版output contract |

## 今回収録しなかったもの

- 個々のcommit reviewやplan review: 最終planまたは実装reviewと内容が重複するため原則除外。
  ただしDH2/DH4 compat plan reviewは、Cのindex式とlayoutを直接監査した一次的な
  ground-truth記録として例外収録した。
- Julia公開manualのinstallation/tutorial全体: porting仕様より利用者向け情報が中心のため除外。
- C v1.3 tutorial画像・build生成物: 容量が大きくRust実装判断との関連が薄いため除外。
- C v1.3 `standard.rst`: Rustでstandard modeをscopeに含めるか未確定のため保留。
- OptTrans reviewとC open-issue inventory: Rust側のreject/known-issue方針を決める段階まで保留。
- fork版Fortran PfaPack `sktf2`境界バグ資料: Rustはpure-Julia subsetのみを移植し、
  Fortran wrapper/sourceは対象外であるため保留。必要なら`n=2` regression test設計時に参照する。
- `Dev_mVMC/docs` のBackFlow、NBody、2nd/Power Lanczos群: 主にC v1.4以降のため、v1.3追従とは別archiveにする。
- 実際のC/Julia fixtureデータ: 本ディレクトリはdocument archiveに限定する。fixtureの導入先とlicenseは次期実装計画で決める。

## Rust計画へ転用するときの原則

- Cを仕様、Juliaを既存の移植判断・分解例、Rustを実装対象として三者比較する。
- C出力をblindにgolden化せず、source revision、build条件、seed、MPI/thread/BLAS条件、再生成commandを記録する。
- stochastic trajectoryの完全一致と、統計量・toleranceで判定すべき経路を分ける。
- C側に既知のsilent wrongがある組み合わせはparity対象にせず、unsupportedまたはcorrected semanticsとして明示する。
- parserだけ通る状態、runtimeで消費できる状態、C-reference gateを通る状態を別々に管理する。
- 対応済み機能だけでなく、組み合わせごとのreject scopeをtestで固定する。

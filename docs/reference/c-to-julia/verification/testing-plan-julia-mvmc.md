# Julia-mVMC ユニットテスト導入計画（`docs/superpower.md` 参考）

## 目的（brainstorming 相当）
- C/J 比較の統合テストで見つかった差分を、**関数レベル（unit/contract）で早期検知**できるようにする。
- fsz/cmp 分岐、RBM 9セクションのレイアウト、`NProj + NRBM + NSlater` のインデックス規約、`trans.def` の `spin1/spin2` など、**壊れやすい“仕様”をテストで固定**する。

## “完了”の定義（verification-before-completion 相当）
- `private-mVMC/MVMCOptimizers.jl` で、C 側のビルド成果物（`../mVMC/build/...`）無しでも
  - `julia --project=@. -e 'using Pkg; Pkg.test()'` が安定して PASS（= unit/contract は外部依存なし）
- 重い統合テスト（C-mVMC の `work/*` を使う比較や長い最適化）はデフォルト無効で、
  - `MVMC_INTEGRATION_TESTS=1` を明示したときのみ実行される

## テスト階層（ワークフローの固定）
- **unit（常時）**: 小関数・小状態・外部ファイルなし・1秒級・seed 固定
- **contract（常時）**: “仕様の対応”を小さな fixture で固定（例: `spin1/spin2`、RBM offset、配列の並び）
- **integration（任意）**: C ref / work を使う比較、ベースライン（長い最適化）など

## 既存資産（再利用）
- `private-mVMC/MVMCOptimizers.jl/test_unit/plan.md`
  - ここに **ユニットテストの追加対象・優先順位・helper 方針**がまとまっているため、本計画の一次情報とする。
- 既存 integration（例）:
  - `private-mVMC/MVMCOptimizers.jl/test/test_vmc_models*.jl`（C ref との比較）
  - `private-mVMC/MVMCOptimizers.jl/test/test_zvo_out_vs_c.jl`
  - `private-mVMC/MVMCOptimizers.jl/test/test_example_*_baseline.jl`（最適化を回すベースライン）

## 実装方針（writing-plans: 2〜5分粒度）
### Phase 0: 導線の整理（最初に“ゲート化”）
- `private-mVMC/MVMCOptimizers.jl/test/runtests.jl` を整理し、デフォルトで unit/contract のみ実行する。
- integration は `MVMC_INTEGRATION_TESTS=1` のときだけ `include(...)` する。

### Phase 1: unit/contract の土台（fixture と helper）
- `test_unit/helpers/` を用意して「最小の `ExpertModeData` / state」を生成できるようにする。
- 方針:
  - ファイル I/O を避ける（必要なら `mktempdir()` のみ）
  - 乱数は `SFMT19937RNG()` + `Random.seed!` で固定

### Phase 2: 重要バグクラスから TDD で固定（RED→GREEN→REFACTOR）
優先度は `private-mVMC/MVMCOptimizers.jl/test_unit/plan.md` に従う。特に以下は“回帰しやすい”ため先に固定する。
- `trans.def` の `spin1/spin2`（fsz の transfer 差分の再発防止）
- RBM の 9 セクション offset と opt-flag の割当
- Proj / RBM / Slater の更新マップ（`NProj + NRBM + NSlater`）

### Phase 3: integration は別ゲートで運用
- `MVMC_INTEGRATION_TESTS=1` のときのみ実行
- 実行前提（例）:
  - C 側で `ctest -V` を回し `../mVMC/build/test/python/work/...` を用意（ref 自動生成に必要）

## 実行コマンド（証拠ログとして残す）
- unit/contract（デフォルト）:
  - `cd private-mVMC/MVMCOptimizers.jl && julia --project=@. -e 'using Pkg; Pkg.test()'`
- integration（明示有効化）:
  - `cd private-mVMC/MVMCOptimizers.jl && MVMC_INTEGRATION_TESTS=1 julia --project=@. -e 'using Pkg; Pkg.test()'`

## 運用ルール（systematic-debugging）
- 差分が出たら、まず **最小再現を作る → unit/contract に落とす → 最後に integration で確認**の順で進める。
- ref 更新が必要なケースは、更新手順をドキュメント化し、**無意識に ref を動かさない**（更新ゲートを明示する）。


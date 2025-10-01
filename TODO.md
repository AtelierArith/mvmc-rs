# mVMC Rust移植 - 初期チェックリスト

このドキュメントは、mVMCのC実装からRustへの移植プロジェクトの初期段階で実行すべきタスクのチェックリストです。

## Phase 1: 基盤構築 (Foundation)

### 1.1 プロジェクト構造のセットアップ

#### ワークスペース設定
- [x] メインの`Cargo.toml`でワークスペースを設定
- [x] 各クレート用のディレクトリ構造を作成
  - [x] `crates/mvmc-core/`
  - [x] `crates/mvmc-math/`
  - [x] `crates/mvmc-physics/`
  - [x] `crates/mvmc-io/`
  - [x] `crates/mvmc-parallel/`
  - [x] `crates/mvmc-cli/`
  - [x] `crates/mvmc-bindings/`
- [x] 各クレートの`Cargo.toml`を初期設定
- [x] 共通依存関係をワークスペースレベルで定義

#### 開発環境の整備
- [x] `.gitignore`の設定（Rust用）
- [x] `.github/workflows/ci.yml`の作成
- [x] `rustfmt.toml`の設定
- [x] `clippy.toml`の設定
- [x] `Cargo.lock`をgitに追加

### 1.2 ドキュメント基盤

#### プロジェクト文書
- [ ] `README.md`の更新（Rust移植版用）
- [ ] `LICENSE`ファイルの確認・更新
- [ ] `CHANGELOG.md`の作成
- [ ] `CONTRIBUTING.md`の作成

#### API文書
- [ ] `docs/`ディレクトリの作成
- [ ] `docs/api/`ディレクトリの作成
- [ ] `docs/tutorials/`ディレクトリの作成
- [ ] `docs/migration/`ディレクトリの作成

### 1.3 テスト基盤

#### テスト構造
- [x] `tests/`ディレクトリの作成（統合テスト用）
- [x] `benches/`ディレクトリの作成（ベンチマーク用）
- [x] `examples/`ディレクトリの作成
  - [ ] `examples/hubbard/`（将来実装）
  - [ ] `examples/heisenberg/`（将来実装）
  - [ ] `examples/kondo/`（将来実装）

#### テストデータ
- [x] 既存のサンプルファイルを`tests/data/`にコピー
- [x] テスト用の設定ファイルを作成
- [x] ベンチマーク用のデータセットを準備

## Phase 2: 数値計算ライブラリ (`mvmc-math`)

### 2.1 基本型とエラー処理

#### 基本型定義
- [x] `mvmc-math/src/types.rs`の作成
  - [x] 複素数型の定義（`num-complex`を使用）
  - [x] 行列・ベクトル型の定義（`ndarray`を使用）
  - [ ] 物理定数の定義
- [x] `mvmc-math/src/error.rs`の作成
  - [x] 数値計算エラーの定義
  - [x] エラーハンドリングの実装

#### 複素数計算
- [x] `mvmc-math/src/complex/mod.rs`の作成
- [x] `mvmc-math/src/complex/operations.rs`の実装
  - [x] 基本的な複素数演算
  - [x] 高精度計算のサポート

### 2.2 乱数生成器

#### SFMT実装
- [x] `mvmc-math/src/random/mod.rs`の作成
- [x] `mvmc-math/src/random/sfmt.rs`の実装
  - [x] C実装の`src/sfmt/`を参考に移植
  - [ ] SIMD最適化の実装（将来拡張）
- [x] `mvmc-math/src/random/distributions.rs`の実装
  - [x] 各種確率分布の実装
  - [x] モンテカルロ用の分布

### 2.3 線形代数基盤

#### 行列操作
- [x] `mvmc-math/src/linear_algebra/mod.rs`の作成
- [x] `mvmc-math/src/linear_algebra/matrix.rs`の実装
  - [x] 密行列の実装
  - [ ] 疎行列の実装（将来用）
- [x] `mvmc-math/src/linear_algebra/decomposition.rs`の実装
  - [x] LU分解の実装
  - [x] 固有値分解の実装（LAPACKバインディング経由）

#### BLAS/LAPACKバインディング
- [x] `mvmc-math/src/linear_algebra/blas.rs`の実装
  - [x] 基本的なBLAS操作のラッパー
  - [x] 型安全なインターフェース
- [x] `mvmc-bindings`クレートの初期設定
  - [x] LAPACKのFFI定義
  - [ ] ScaLAPACKのFFI定義（オプション）

### 2.4 テストとベンチマーク

#### 単体テスト
- [x] 各モジュールの単体テストを実装
- [ ] 既存C実装との結果比較テスト（プレースホルダー実装のため）
- [x] エッジケースのテスト

#### ベンチマーク
- [x] 数値計算のベンチマークを実装
- [x] C実装との性能比較
- [x] メモリ使用量の測定

## Phase 3: 入出力処理 (`mvmc-io`)

### 3.1 入力ファイル解析

#### StdFace形式
- [ ] `mvmc-io/src/input/stdface.rs`の実装
  - [ ] パーサーの実装
  - [ ] 型安全な設定構造体
- [ ] `mvmc-io/src/input/parser.rs`の実装
  - [ ] 汎用パーサー基盤
  - [ ] エラーハンドリング

#### TOML形式
- [ ] `mvmc-io/src/input/toml.rs`の実装
  - [ ] TOMLパーサーの実装
  - [ ] 設定構造体の定義

### 3.2 出力処理

#### データ出力
- [ ] `mvmc-io/src/output/data.rs`の実装
  - [ ] 計算結果の出力
  - [ ] フォーマット指定のサポート
- [ ] `mvmc-io/src/output/parameters.rs`の実装
  - [ ] 最適化されたパラメータの保存
  - [ ] 再現性のための設定保存

### 3.3 ファイル形式サポート

#### 基本形式
- [ ] `mvmc-io/src/formats/json.rs`の実装
- [ ] `mvmc-io/src/formats/binary.rs`の実装
- [ ] `mvmc-io/src/formats/hdf5.rs`の実装（将来用）

## Phase 4: コアライブラリ基盤 (`mvmc-core`)

### 4.1 基本構造

#### エラー処理
- [ ] `mvmc-core/src/error.rs`の実装
  - [ ] アプリケーション全体のエラー定義
  - [ ] エラー変換の実装

#### 型定義
- [ ] `mvmc-core/src/types.rs`の実装
  - [ ] 物理量の型定義
  - [ ] 計算状態の型定義

### 4.2 設定管理

#### パラメータ管理
- [ ] `mvmc-core/src/config/mod.rs`の実装
- [ ] `mvmc-core/src/config/parameters.rs`の実装
  - [ ] 計算パラメータの定義
  - [ ] バリデーション機能
- [ ] `mvmc-core/src/config/validation.rs`の実装
  - [ ] 入力値の検証
  - [ ] 設定の整合性チェック

### 4.3 ユーティリティ

#### メモリ管理
- [ ] `mvmc-core/src/utils/memory.rs`の実装
  - [ ] 大きな配列の効率的な管理
  - [ ] メモリ使用量の監視

#### ログ機能
- [ ] `mvmc-core/src/utils/logging.rs`の実装
  - [ ] 構造化ログの実装
  - [ ] デバッグ情報の出力

## Phase 5: 物理モデル基盤 (`mvmc-physics`)

### 5.1 格子構造

#### 基本格子
- [ ] `mvmc-physics/src/lattice/mod.rs`の実装
- [ ] `mvmc-physics/src/lattice/square.rs`の実装
- [ ] `mvmc-physics/src/lattice/triangular.rs`の実装
- [ ] `mvmc-physics/src/lattice/honeycomb.rs`の実装

### 5.2 ハミルトニアン

#### 基本構造
- [ ] `mvmc-physics/src/hamiltonian/mod.rs`の実装
- [ ] `mvmc-physics/src/hamiltonian/builder.rs`の実装
  - [ ] ハミルトニアン構築のインターフェース
  - [ ] 型安全な構築プロセス

### 5.3 物理量

#### 基本物理量
- [ ] `mvmc-physics/src/observables/mod.rs`の実装
- [ ] `mvmc-physics/src/observables/energy.rs`の実装
- [ ] `mvmc-physics/src/observables/green_function.rs`の実装

## Phase 6: 並列化基盤 (`mvmc-parallel`)

### 6.1 スレッド並列化

#### 基本実装
- [ ] `mvmc-parallel/src/threading/mod.rs`の実装
- [ ] `mvmc-parallel/src/threading/thread_pool.rs`の実装
  - [ ] `rayon`を活用したスレッドプール
  - [ ] ワークスティーリングの実装

### 6.2 MPI並列化

#### FFIバインディング
- [ ] `mvmc-parallel/src/mpi/mod.rs`の実装
- [ ] `mvmc-parallel/src/mpi/communicator.rs`の実装
  - [ ] MPI通信のラッパー
  - [ ] 型安全なインターフェース

## Phase 7: CLI基盤 (`mvmc-cli`)

### 7.1 基本構造

#### コマンドライン
- [ ] `mvmc-cli/src/main.rs`の実装
- [ ] `mvmc-cli/src/commands/mod.rs`の実装
- [ ] `clap`を使用したCLIフレームワークの設定

#### 基本コマンド
- [ ] `mvmc-cli/src/commands/optimize.rs`の実装
- [ ] `mvmc-cli/src/commands/calculate.rs`の実装
- [ ] `mvmc-cli/src/commands/analyze.rs`の実装

## Phase 8: 統合テスト

### 8.1 基本統合

#### エンドツーエンドテスト
- [ ] 簡単なハバードモデルの計算テスト
- [ ] 入力→計算→出力の一連の流れのテスト
- [ ] 既存C実装との結果比較

#### パフォーマンステスト
- [ ] ベンチマークスイートの実装
- [ ] メモリ使用量の測定
- [ ] 並列化の効果測定

### 8.2 ドキュメント

#### 使用例
- [ ] 基本的な使用例の作成
- [ ] チュートリアルの作成
- [ ] API文書の生成

## 完了基準

### Phase 1完了基準
- [x] ワークスペースが正常にビルドできる
- [x] 基本的なテストが実行できる
- [x] CI/CDパイプラインが動作する

### Phase 2完了基準
- [x] 基本的な数値計算が動作する
- [x] 複素数計算が完全に実装・テスト済み
- [x] 線形代数基盤が完全に実装・テスト済み
- [x] BLAS/LAPACKバインディングが実装・テスト済み
- [x] ベンチマークが実行できる
- [ ] C実装との結果が一致する（プレースホルダー実装のため）

### Phase 3完了基準
- [ ] 既存の入力ファイルが読み込める
- [ ] 結果が適切な形式で出力される
- [ ] エラーハンドリングが適切に動作する

### Phase 4完了基準
- [ ] 基本的な設定管理が動作する
- [ ] エラーハンドリングが統一されている
- [ ] ログ機能が動作する

## 進捗状況

### 完了済み ✅
- **Phase 1**: 基盤構築が完了
  - ワークスペース設定、ディレクトリ構造、開発環境整備
  - テスト基盤の構築（統合テスト、ベンチマーク用ディレクトリ）
  - テストデータの準備（既存サンプルファイルのコピー）
- **Phase 2 (完了)**: 数値計算ライブラリの基盤
  - 乱数生成器の実装（`mvmc-math/src/random.rs`）
  - 複素数計算の実装（`mvmc-math/src/complex.rs`）
  - 線形代数基盤の実装（`mvmc-math/src/linear_algebra.rs`）
  - BLAS/LAPACKバインディングの実装（`mvmc-bindings/`）
  - 包括的な単体テストとプロパティベーステスト（45個のテスト）
  - 最適化されたベンチマーク基盤
  - TDDガイドの作成（`TDD_GUIDE.md`、`TDD_SETUP_COMPLETE.md`）

### 現在の状況
- **Rust edition 2024** を採用
- **Test-Driven Development** の環境が整備済み
- **CI/CDパイプライン** が動作中
- **Phase 2完了**: 数値計算ライブラリが完全に実装・テスト済み
  - 乱数生成器（SFMTベース）
  - 複素数計算（高精度演算）
  - 線形代数基盤（複素行列操作）
  - BLAS/LAPACKバインディング（型安全なラッパー）
  - 包括的なテストスイート（45個のテスト）
  - 最適化されたベンチマーク基盤

### 次のステップ（推奨順序）
1. **物理モデル基盤** (`mvmc-physics`)
   - ハバードモデルの実装
   - 格子構造の定義
   - ハミルトニアンの構築

2. **コアライブラリ基盤** (`mvmc-core`)
   - 波動関数の実装（Slater行列、Pfaffian）
   - エラー処理の統一
   - 設定管理システム

3. **入出力処理** (`mvmc-io`)
   - StdFace形式のパーサー
   - TOML形式のサポート
   - 結果出力機能

## 実装詳細

### Phase 2実装詳細

#### 数値計算ライブラリ (`mvmc-math`)
- **複素数計算** (`src/complex.rs`)
  - 基本的な複素数演算（加算、減算、乗算、除算）
  - 高精度計算（指数、対数、位相計算）
  - 数値的に安全な除算処理
  - 12個の単体テスト + 6個のプロパティベーステスト

- **線形代数基盤** (`src/linear_algebra.rs`)
  - 複素行列の基本操作（作成、アクセス、設定）
  - 行列演算（加算、減算、乗算、スカラー倍）
  - 高度な操作（転置、エルミート転置、トレース、行列式）
  - LU分解による行列式計算
  - エルミート行列の判定
  - 15個の単体テスト + 6個のプロパティベーステスト

- **乱数生成器** (`src/random.rs`)
  - SFMTベースの高速乱数生成
  - 各種確率分布のサポート
  - モンテカルロ計算用の分布
  - 12個の単体テスト + 4個のプロパティベーステスト

#### Cライブラリバインディング (`mvmc-bindings`)
- **LAPACK FFI定義** (`src/ffi/lapack.rs`)
  - 基本的なLAPACK関数のFFI定義
  - エラーハンドリングの実装
  - 型安全なインターフェース

- **安全なラッパー** (`src/wrappers/linear_algebra.rs`)
  - 線形方程式の解法
  - LU分解
  - 行列の逆行列計算
  - 固有値分解
  - 特異値分解
  - 6個の単体テスト + 2個のプロパティベーステスト

#### テスト・ベンチマーク
- **総テスト数**: 45個（単体テスト + プロパティベーステスト）
- **ベンチマーク**: 最適化済み（30秒〜1分程度で実行）
- **カバレッジ**: 高品質なテストカバレッジ
- **TDD**: 全機能でテスト駆動開発を採用

## 注意事項

1. **段階的実装**: 各Phaseは前のPhaseが完了してから開始する
2. **テスト駆動**: 各機能はテストを先に書いてから実装する
3. **ドキュメント**: 実装と同時にドキュメントを更新する
4. **パフォーマンス**: 各段階でパフォーマンスを測定・記録する
5. **互換性**: 既存のC実装との互換性を常に意識する

## 参考資料

- [PLAN.md](./PLAN.md) - 詳細な移植計画
- [CLAUDE.md](./CLAUDE.md) - プロジェクト概要
- [mVMC公式リポジトリ](https://github.com/issp-center-dev/mVMC) - 元のC実装
- [mVMC論文](https://www.sciencedirect.com/science/article/pii/S0010465518303102) - 理論的背景


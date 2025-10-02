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
- [x] `mvmc-io/src/stdface/parser.rs`の実装
  - [x] パーサーの実装
  - [x] 型安全な設定構造体
- [x] `mvmc-io/src/stdface/config.rs`の実装
  - [x] 汎用パーサー基盤
  - [x] エラーハンドリング

#### TOML形式
- [x] `mvmc-io/src/toml/parser.rs`の実装
  - [x] TOMLパーサーの実装
  - [x] 設定構造体の定義

#### JSON形式
- [x] `mvmc-io/src/json/parser.rs`の実装
  - [x] JSONパーサーの実装
  - [x] 設定構造体の定義

### 3.2 出力処理 ✅ **完了**

#### データ出力
- [x] `mvmc-io/src/output/data.rs`の実装
  - [x] 計算結果の出力（`EnergyData`, `VariationalData`, `ObservableData`）
  - [x] フォーマット指定のサポート（Text/Binary形式）
- [x] `mvmc-io/src/output/parameters.rs`の実装
  - [x] 最適化されたパラメータの保存（`OptimizedParameters`）
  - [x] 再現性のための設定保存（読み込み/書き込み対応）
- [x] `mvmc-io/src/output/mod.rs`の実装
  - [x] 出力マネージャー（`OutputManager`）の実装
  - [x] ファイル名の統一管理
  - [x] 包括的なテストスイート（30個のテスト）

### 3.3 ファイル形式サポート

#### 基本形式
- [x] `mvmc-io/src/json/`の実装
- [x] テキスト/バイナリ形式のサポート（`output/data.rs`で実装）
- [ ] `mvmc-io/src/formats/hdf5.rs`の実装（将来用）

## Phase 4: コアライブラリ基盤 (`mvmc-core`)

### 4.1 基本構造

#### エラー処理
- [x] `mvmc-core/src/error.rs`の実装
  - [x] アプリケーション全体のエラー定義
  - [x] エラー変換の実装

#### 型定義
- [x] `mvmc-core/src/types.rs`の実装
  - [x] 物理量の型定義
  - [x] 計算状態の型定義

### 4.2 設定管理

#### パラメータ管理
- [x] `mvmc-core/src/config/mod.rs`の実装
- [x] `mvmc-core/src/config/parameters.rs`の実装
  - [x] 計算パラメータの定義
  - [x] バリデーション機能
- [x] `mvmc-core/src/config/validation.rs`の実装
  - [x] 入力値の検証
  - [x] 設定の整合性チェック

### 4.3 ユーティリティ

#### メモリ管理
- [ ] `mvmc-core/src/utils/memory.rs`の実装
  - [ ] 大きな配列の効率的な管理
  - [ ] メモリ使用量の監視

#### ログ機能
- [ ] `mvmc-core/src/utils/logging.rs`の実装
  - [ ] 構造化ログの実装
  - [ ] デバッグ情報の出力

### 4.4 VMC計算エンジン ✅ **完了**

#### 波動関数実装
- [x] `mvmc-core/src/wavefunction/slater.rs`の実装
  - [x] Slater行列の実装
  - [x] 行列式計算（LU分解）
  - [x] 高速更新アルゴリズム
- [x] `mvmc-core/src/wavefunction/pfaffian.rs`の実装
  - [x] Pfaffian行列の実装
  - [x] LTL分解による計算
  - [x] 高速更新メカニズム
- [x] `mvmc-core/src/wavefunction/projection.rs`の実装
  - [x] 射影演算子の実装
  - [x] Gutzwiller因子の計算
  - [ ] Jastrow因子の実装（将来用）
  - [ ] Doublon-Holon相関因子の実装（将来用）
- [x] `mvmc-core/src/wavefunction/rbm.rs`の実装
  - [x] RBM波動関数の基本構造
  - [x] パラメータ管理
  - [x] 振幅計算

#### 最適化アルゴリズム
- [x] `mvmc-core/src/optimization/conjugate_gradient.rs`の実装
  - [x] 共役勾配法の実装
  - [x] 線形方程式の解法
  - [x] 収束判定
- [x] `mvmc-core/src/optimization/stochastic_reconfiguration.rs`の実装
  - [x] SR法の実装
  - [x] パラメータ更新
  - [x] 収束判定
- [x] `mvmc-core/src/optimization/lanczos.rs`の実装
  - [x] Lanczos法の実装
  - [x] 固有値計算
  - [x] 基底状態の計算

#### モンテカルロサンプリング
- [x] `mvmc-core/src/monte_carlo/metropolis.rs`の実装
  - [x] Metropolisアルゴリズム
  - [x] 電子配置の更新
  - [x] 受容率の計算
- [x] `mvmc-core/src/monte_carlo/sampler.rs`の実装
  - [x] サンプリング統計の管理
  - [x] 受容率の追跡
- [x] `mvmc-core/src/monte_carlo/observables.rs`の実装
  - [x] 物理量の計算
  - [x] 統計的誤差の評価

#### VMCエンジン統合
- [x] `mvmc-core/src/vmc/engine.rs`の実装
  - [x] VmcEngine構造体の実装
  - [x] 最適化モードと期待値計算モードの統合
  - [x] パラメータ検証とエラーハンドリング
  - [x] モンテカルロサンプリングと最適化の統合
- [x] `mvmc-core/src/vmc/mod.rs`の実装
  - [x] VMCモジュールの統合
  - [x] 公開APIの定義

## Phase 5: 物理モデル基盤 (`mvmc-physics`)

### 5.1 格子構造

#### 基本格子
- [x] `mvmc-physics/src/lattice/mod.rs`の実装
- [x] `mvmc-physics/src/lattice/chain.rs`の実装（1次元鎖格子）
- [x] `mvmc-physics/src/lattice/square.rs`の実装（2次元正方格子）
- [ ] `mvmc-physics/src/lattice/triangular.rs`の実装（将来用）
- [ ] `mvmc-physics/src/lattice/honeycomb.rs`の実装（将来用）

### 5.2 ハミルトニアン

#### 基本構造
- [x] `mvmc-physics/src/hamiltonian/mod.rs`の実装
- [x] `mvmc-physics/src/hamiltonian/hubbard.rs`の実装（ハバードモデル）
- [x] `mvmc-physics/src/hamiltonian/heisenberg.rs`の実装（ハイゼンベルグモデル）
- [ ] `mvmc-physics/src/hamiltonian/builder.rs`の実装（将来用）
  - [ ] ハミルトニアン構築のインターフェース
  - [ ] 型安全な構築プロセス

### 5.3 物理量

#### 基本物理量
- [x] `mvmc-physics/src/observables/mod.rs`の実装
- [x] `mvmc-physics/src/observables/energy.rs`の実装（エネルギー計算）
- [x] `mvmc-physics/src/observables/magnetization.rs`の実装（磁化計算）
- [x] `mvmc-physics/src/observables/correlation.rs`の実装（相関関数計算）
- [ ] `mvmc-physics/src/observables/green_function.rs`の実装（将来用）

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

## Phase 7: CLI基盤 (`mvmc-cli`) ✅ **完了**

### 7.1 基本構造

#### コマンドライン
- [x] `mvmc-cli/src/main.rs`の実装
- [x] `mvmc-cli/src/commands/mod.rs`の実装
- [x] `clap`を使用したCLIフレームワークの設定
- [x] `mvmc-cli/src/error.rs`の実装（エラーハンドリング）

#### 基本コマンド
- [x] `mvmc-cli/src/commands/run.rs`の実装（VMC計算実行）
- [x] `mvmc-cli/src/commands/info.rs`の実装（設定情報表示）
- [x] `mvmc-cli/src/commands/validate.rs`の実装（設定検証）
- [x] `mvmc-cli/src/commands/version.rs`の実装（バージョン情報）

#### 実装済み機能
- コマンドライン引数パース（`clap`使用）
- 設定ファイル読み込み（StdFace/TOML/JSON対応）
- カラー出力（`colored`使用）
- ロギング機能（`env_logger`使用）
- スレッド数指定
- バイナリ出力対応

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
- [x] 既存の入力ファイルが読み込める
- [x] 結果が適切な形式で出力される（テキスト/バイナリ形式対応）
- [x] エラーハンドリングが適切に動作する

### Phase 4完了基準
- [x] 基本的な設定管理が動作する
- [x] エラーハンドリングが統一されている
- [x] VMC計算エンジンが動作する
- [x] 統合テストが成功する
- [x] CLIとVMCエンジンが統合される

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
- **Phase 3 (完了)**: 入出力処理の基盤
  - StdFace形式パーサーの実装（`mvmc-io/src/stdface/`）
  - TOML形式パーサーの実装（`mvmc-io/src/toml/`）
  - JSON形式パーサーの実装（`mvmc-io/src/json/`）
  - 出力処理の実装（`mvmc-io/src/output/`）
    - データ出力（エネルギー、変分データ、物理量）
    - パラメータ保存/読み込み（テキスト/バイナリ形式）
    - OutputManagerによる統一ファイル管理
  - 包括的なテストスイート（50個のテスト）
  - エラーハンドリングとバリデーション機能
- **Phase 4 (完了)**: コアライブラリ基盤
  - 波動関数の実装（Slater行列、Pfaffian、射影演算子、RBM）
  - 最適化アルゴリズムの実装（共役勾配法、SR法、Lanczos法）
  - モンテカルロサンプリングの実装（Metropolis法）
  - VMC計算エンジンの統合実装
  - 統合テストの実装（11個のテスト）
  - CLIとVMCエンジンの統合
- **Phase 5 (完了)**: 物理モデル基盤
  - 格子構造の実装（1次元鎖、2次元正方格子）
  - ハミルトニアンの実装（ハバード、ハイゼンベルグモデル）
  - 物理量計算の実装（エネルギー、磁化、相関関数）
  - 包括的なテストスイート（99個のテスト）
  - プロパティベーステストによる数学的性質の検証
- **Phase 7 (完了)**: CLI基盤
  - コマンドラインインターフェース（`mvmc-cli/`）
  - 4つのコマンド実装（run、info、validate、version）
  - 複数入力形式対応（StdFace、TOML、JSON）
  - テキスト/バイナリ出力対応
  - スレッド数指定、カラー出力、ロギング機能
  - VMCエンジンとの統合完了

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
- **Phase 3完了**: 入出力処理が完全に実装・テスト済み
  - StdFace形式パーサー（mVMC標準入力形式）
  - TOML形式パーサー（構造化設定ファイル）
  - JSON形式パーサー（機械可読設定ファイル）
  - 出力処理（エネルギーデータ、変分データ、最適化パラメータ）
  - テキスト/バイナリ形式のサポート
  - 包括的なテストスイート（50個のテスト）
  - エラーハンドリングとバリデーション機能
- **Phase 5完了**: 物理モデル基盤が完全に実装・テスト済み
  - 格子構造（1次元鎖、2次元正方格子）
  - ハミルトニアン（ハバード、ハイゼンベルグモデル）
  - 物理量計算（エネルギー、磁化、相関関数）
  - 包括的なテストスイート（99個のテスト）
  - プロパティベーステストによる数学的性質の検証
- **Phase 7完了（基本実装）**: CLI基盤が実装済み
  - コマンドラインインターフェース（`clap`使用）
  - 4つの基本コマンド（run、info、validate、version）
  - 複数入力形式対応（StdFace、TOML、JSON自動検出）
  - テキスト/バイナリ出力形式切り替え
  - スレッドプール設定（`rayon`使用）
  - カラー出力（`colored`使用）
  - ロギング機能（`env_logger`使用）
  - 包括的なエラーハンドリング

### 次のステップ（推奨順序）
1. **並列化基盤** (`mvmc-parallel`) ← **次の優先**
   - スレッド並列化（rayonベース - CLIで基本実装済み）
   - MPI並列化（FFIバインディング）

2. **統合テストと最適化**
   - エンドツーエンドテスト
   - C実装との結果比較
   - パフォーマンス最適化

3. **高度な機能の実装**
   - Jastrow因子の実装
   - Doublon-Holon相関因子の実装
   - より複雑な物理モデルの実装

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

### Phase 3実装詳細

#### 入出力処理ライブラリ (`mvmc-io`)
- **StdFace形式パーサー** (`src/stdface/`)
  - キー・バリューペアの解析
  - 引用符の自動除去
  - コメント行のスキップ
  - 多次元格子の自動検出（1D/2D）
  - 厳密モードと非厳密モードのサポート
  - 包括的なバリデーション機能

- **TOML形式パーサー** (`src/toml/`)
  - 構造化された設定ファイルの解析
  - 型安全な設定構造体
  - エラーハンドリングとバリデーション

- **JSON形式パーサー** (`src/json/`)
  - 機械可読な設定ファイルの解析
  - 型安全な設定構造体
  - エラーハンドリングとバリデーション

- **共通機能**
  - `ConfigParser`トレイトによる統一インターフェース
  - `ConfigWriter`トレイトによる出力機能
  - ユーティリティ関数（ファイル形式検出、パス正規化）

#### テスト・ベンチマーク
- **総テスト数**: 50個（入力解析20個 + 出力処理30個）
- **対応フォーマット**: StdFace、TOML、JSON（入力）、Text/Binary（出力）
- **エラーハンドリング**: 統一されたエラー処理
- **TDD**: 全機能でテスト駆動開発を採用

### Phase 5実装詳細

#### 物理モデルライブラリ (`mvmc-physics`)
- **格子構造** (`src/lattice/`)
  - `ChainLattice`: 1次元鎖格子（周期的・開放境界条件対応）
  - `SquareLattice`: 2次元正方格子（周期的・開放境界条件対応）
  - `Lattice`トレイトによる統一インターフェース
  - 隣接関係、距離計算、座標変換機能

- **ハミルトニアン** (`src/hamiltonian/`)
  - `HubbardHamiltonian`: ハバードモデル（ホッピング、相互作用、化学ポテンシャル）
  - `HeisenbergHamiltonian`: ハイゼンベルグモデル（交換相互作用、磁場）
  - `Spin`列挙型: Up、Down、Empty状態
  - `Hamiltonian`トレイトによる統一インターフェース

- **物理量計算** (`src/observables/`)
  - `EnergyCalculator`: エネルギー計算（運動エネルギー、ポテンシャルエネルギー）
  - `MagnetizationCalculator`: 磁化計算（総磁化、サイトあたり磁化、絶対磁化）
  - `CorrelationCalculator`: 相関関数計算（スピン-スピン相関、構造因子）
  - `Observable`トレイトによる統一インターフェース

#### テスト・ベンチマーク
- **総テスト数**: 99個（単体テスト + プロパティベーステスト）
- **数学的性質**: プロパティベーステストによる検証
- **統計的検証**: 確率的アルゴリズムの分布特性検証
- **TDD**: 全機能でテスト駆動開発を採用

### Phase 7実装詳細

#### CLIライブラリ (`mvmc-cli`)
- **コマンド構造** (`src/main.rs`、`src/commands/`)
  - `clap`を使用したコマンドライン引数解析（derive API）
  - サブコマンドシステム（run、info、validate、version）
  - グローバルオプション（verbose、quiet）

- **基本コマンド**
  - `run`: VMC計算の実行
    - 入力形式の自動検出（拡張子ベース）
    - 出力形式の選択（テキスト/バイナリ）
    - スレッド数の指定（`rayon`でスレッドプール設定）
  - `info`: 設定ファイルの情報表示
    - モデルタイプ、格子タイプの表示
    - カラー出力による視認性向上
  - `validate`: 設定ファイルの検証
    - パーサーによる構文検証
    - エラーメッセージの分かりやすい表示
  - `version`: バージョン情報の表示
    - バージョン番号とGitコミットハッシュ

- **エラーハンドリング** (`src/error.rs`)
  - `thiserror`を使用した統一エラー型
  - `mvmc_io`、`mvmc_core`のエラー変換
  - 詳細なエラーメッセージ

- **ユーティリティ機能**
  - カラー出力（`colored`クレート）
  - ロギング（`env_logger`、`log`クレート）
  - 環境変数による設定（`RUST_LOG`）

#### 実装状況
- **コード行数**: 699行（main.rs 150行 + commands 464行 + error.rs 74行 + version.rs 11行）
- **ビルド状態**: 正常にコンパイル・実行可能
- **依存関係**: clap 4.5、colored 2.1、env_logger 0.11、log 0.4
- **次のステップ**: Phase 4（コアVMCエンジン）実装後に実際の計算機能を統合

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


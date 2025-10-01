# mVMC Rust移植 - 初期チェックリスト

このドキュメントは、mVMCのC実装からRustへの移植プロジェクトの初期段階で実行すべきタスクのチェックリストです。

## Phase 1: 基盤構築 (Foundation)

### 1.1 プロジェクト構造のセットアップ

#### ワークスペース設定
- [ ] メインの`Cargo.toml`でワークスペースを設定
- [ ] 各クレート用のディレクトリ構造を作成
  - [ ] `crates/mvmc-core/`
  - [ ] `crates/mvmc-math/`
  - [ ] `crates/mvmc-physics/`
  - [ ] `crates/mvmc-io/`
  - [ ] `crates/mvmc-parallel/`
  - [ ] `crates/mvmc-cli/`
  - [ ] `crates/mvmc-bindings/`
- [ ] 各クレートの`Cargo.toml`を初期設定
- [ ] 共通依存関係をワークスペースレベルで定義

#### 開発環境の整備
- [ ] `.gitignore`の設定（Rust用）
- [ ] `.github/workflows/ci.yml`の作成
- [ ] `rustfmt.toml`の設定
- [ ] `clippy.toml`の設定
- [ ] `Cargo.lock`をgitに追加

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
- [ ] `tests/`ディレクトリの作成（統合テスト用）
- [ ] `benches/`ディレクトリの作成（ベンチマーク用）
- [ ] `examples/`ディレクトリの作成
  - [ ] `examples/hubbard/`
  - [ ] `examples/heisenberg/`
  - [ ] `examples/kondo/`

#### テストデータ
- [ ] 既存のサンプルファイルを`tests/data/`にコピー
- [ ] テスト用の設定ファイルを作成
- [ ] ベンチマーク用のデータセットを準備

## Phase 2: 数値計算ライブラリ (`mvmc-math`)

### 2.1 基本型とエラー処理

#### 基本型定義
- [ ] `mvmc-math/src/types.rs`の作成
  - [ ] 複素数型の定義
  - [ ] 行列・ベクトル型の定義
  - [ ] 物理定数の定義
- [ ] `mvmc-math/src/error.rs`の作成
  - [ ] 数値計算エラーの定義
  - [ ] エラーハンドリングの実装

#### 複素数計算
- [ ] `mvmc-math/src/complex/mod.rs`の作成
- [ ] `mvmc-math/src/complex/operations.rs`の実装
  - [ ] 基本的な複素数演算
  - [ ] 高精度計算のサポート

### 2.2 乱数生成器

#### SFMT実装
- [ ] `mvmc-math/src/random/mod.rs`の作成
- [ ] `mvmc-math/src/random/sfmt.rs`の実装
  - [ ] C実装の`src/sfmt/`を参考に移植
  - [ ] SIMD最適化の実装
- [ ] `mvmc-math/src/random/distributions.rs`の実装
  - [ ] 各種確率分布の実装
  - [ ] モンテカルロ用の分布

### 2.3 線形代数基盤

#### 行列操作
- [ ] `mvmc-math/src/linear_algebra/mod.rs`の作成
- [ ] `mvmc-math/src/linear_algebra/matrix.rs`の実装
  - [ ] 密行列の実装
  - [ ] 疎行列の実装（将来用）
- [ ] `mvmc-math/src/linear_algebra/decomposition.rs`の実装
  - [ ] LTL分解の実装
  - [ ] 固有値分解の実装

#### BLAS/LAPACKバインディング
- [ ] `mvmc-math/src/linear_algebra/blas.rs`の実装
  - [ ] 基本的なBLAS操作のラッパー
  - [ ] 型安全なインターフェース
- [ ] `mvmc-bindings`クレートの初期設定
  - [ ] LAPACKのFFI定義
  - [ ] ScaLAPACKのFFI定義（オプション）

### 2.4 テストとベンチマーク

#### 単体テスト
- [ ] 各モジュールの単体テストを実装
- [ ] 既存C実装との結果比較テスト
- [ ] エッジケースのテスト

#### ベンチマーク
- [ ] 数値計算のベンチマークを実装
- [ ] C実装との性能比較
- [ ] メモリ使用量の測定

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
- [ ] ワークスペースが正常にビルドできる
- [ ] 基本的なテストが実行できる
- [ ] CI/CDパイプラインが動作する

### Phase 2完了基準
- [ ] 基本的な数値計算が動作する
- [ ] C実装との結果が一致する
- [ ] ベンチマークが実行できる

### Phase 3完了基準
- [ ] 既存の入力ファイルが読み込める
- [ ] 結果が適切な形式で出力される
- [ ] エラーハンドリングが適切に動作する

### Phase 4完了基準
- [ ] 基本的な設定管理が動作する
- [ ] エラーハンドリングが統一されている
- [ ] ログ機能が動作する

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


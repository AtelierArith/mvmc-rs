# mVMC Rust移植計画書

## 概要

この文書は、C言語で実装されたmVMC（many-variable Variational Monte Carlo method）をRustに移植するための包括的な計画書です。既存のC実装の構造分析に基づき、Rustの利点を最大限活用した設計を提案します。

## 1. 既存C実装の構造分析

### 1.1 プロジェクト概要

mVMCは量子格子モデルの高精度変分モンテカルロ計算を行うソフトウェアパッケージです。

**対象モデル:**
- ハバードモデル
- ハイゼンベルグモデル
- 近藤格子モデル
- 多軌道ハバードモデル

**主要機能:**
- 変分パラメータの最適化
- 期待値の計算
- グリーン関数の計算
- 物理量の評価

### 1.2 既存ディレクトリ構造

```
mVMC/src/
├── mVMC/           # メインのVMC計算エンジン
├── ComplexUHF/     # 制限なしハートリー・フォック法
├── ltl2inv/        # 線形代数計算（LTL分解と逆行列）
├── pfupdates/      # Pfaffian更新（C++実装）
├── pfapack/        # Pfaffian計算ライブラリ
├── sfmt/           # 高速乱数生成器
├── common/         # 共通ライブラリ
└── StdFace/        # 入力ファイル生成ツール
```

### 1.3 コアモジュールの詳細

#### メインVMCモジュール (`src/mVMC/`)
- `vmcmain.c`: メインプログラム
- `vmccal.c`: VMC計算の核心部分
- `vmcmake.c`: 波動関数の構築
- `slater.c`: Slater行列の計算
- `pfupdate.c`: Pfaffian更新
- `projection.c`: 射影演算子の計算
- `rbm.c`: 制限ボルツマンマシン

#### ComplexUHFモジュール (`src/ComplexUHF/`)
- `UHFmain.c`: UHFメインプログラム
- `diag.c`: 対角化計算
- `green.c`: グリーン関数計算
- `makeham.c`: ハミルトニアン構築

#### 線形代数モジュール (`src/ltl2inv/`)
- `ltl2inv.cc`: LTL分解と逆行列計算
- `invert.tcc`: 逆行列計算テンプレート
- `pfaffian.tcc`: Pfaffian計算

### 1.4 データ構造とアルゴリズム

**主要なデータ構造:**
```c
// グローバル変数（global.h）
char CDataFileHead[D_FileNameMax];  // 出力ファイルプレフィックス
int NVMCCalMode;                    // 計算モード（0:最適化, 1:期待値計算）
int NLanczosMode;                   // Lanczos法のモード
int NStoreO;                        // ストアOの選択
```

**波動関数の表現:**
- Slater行列: 一粒子波動関数の行列
- Pfaffian: ペアリング波動関数
- 射影演算子: 局所スピン制約
- RBM: 制限ボルツマンマシンによる補正

### 1.5 ビルドシステム

**主要な設定オプション:**
- `USE_SCALAPACK`: ScaLAPACKライブラリの使用
- `PFAFFIAN_BLOCKED`: ブロック更新Pfaffianの使用
- `USE_GEMMT`: GEMMTの使用
- `Testing`: テストの有効化

**依存関係:**
- LAPACK/ScaLAPACK
- MPI
- OpenMP
- BLIS（オプション）

## 2. Rust移植用ディレクトリ構造

### 2.1 プロジェクト全体構造

```
mvmc-rs/
├── Cargo.toml                    # ワークスペース設定
├── README.md
├── LICENSE
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml
├── docs/                         # ドキュメント
│   ├── api/
│   ├── tutorials/
│   └── migration/
├── examples/                     # 使用例
│   ├── hubbard/
│   ├── heisenberg/
│   └── kondo/
├── benches/                      # ベンチマーク
├── tests/                        # 統合テスト
├── tools/                        # ユーティリティツール
│   ├── input_generator/
│   └── result_analyzer/
└── crates/                       # 各クレート
    ├── mvmc-core/               # コアライブラリ
    ├── mvmc-cli/                # CLIアプリケーション
    ├── mvmc-math/               # 数値計算ライブラリ
    ├── mvmc-physics/            # 物理モデル
    ├── mvmc-io/                 # 入出力処理
    ├── mvmc-parallel/           # 並列計算
    └── mvmc-bindings/           # Cライブラリバインディング
```

### 2.2 コアライブラリ (`crates/mvmc-core/`)

```
crates/mvmc-core/
├── Cargo.toml
├── src/
│   ├── lib.rs                   # ライブラリエントリーポイント
│   ├── error.rs                 # エラー定義
│   ├── types.rs                 # 基本型定義
│   ├── config/                  # 設定管理
│   │   ├── mod.rs
│   │   ├── parameters.rs
│   │   └── validation.rs
│   ├── wavefunction/            # 波動関数
│   │   ├── mod.rs
│   │   ├── slater.rs           # Slater行列
│   │   ├── pfaffian.rs         # Pfaffian
│   │   ├── projection.rs       # 射影演算子
│   │   └── rbm.rs              # 制限ボルツマンマシン
│   ├── optimization/            # 最適化アルゴリズム
│   │   ├── mod.rs
│   │   ├── conjugate_gradient.rs
│   │   ├── stochastic_reconfiguration.rs
│   │   └── lanczos.rs
│   ├── monte_carlo/             # モンテカルロ計算
│   │   ├── mod.rs
│   │   ├── sampler.rs
│   │   ├── metropolis.rs
│   │   └── observables.rs
│   └── utils/                   # ユーティリティ
│       ├── mod.rs
│       ├── memory.rs
│       └── logging.rs
├── tests/
└── benches/
```

### 2.3 数値計算ライブラリ (`crates/mvmc-math/`)

```
crates/mvmc-math/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── linear_algebra/          # 線形代数
│   │   ├── mod.rs
│   │   ├── matrix.rs
│   │   ├── decomposition.rs     # LTL分解など
│   │   ├── inverse.rs
│   │   └── blas.rs             # BLAS/LAPACKラッパー
│   ├── random/                  # 乱数生成
│   │   ├── mod.rs
│   │   ├── sfmt.rs             # SFMT実装
│   │   └── distributions.rs
│   ├── fft/                     # FFT計算
│   │   ├── mod.rs
│   │   └── fftw.rs
│   └── complex/                 # 複素数計算
│       ├── mod.rs
│       └── operations.rs
├── tests/
└── benches/
```

### 2.4 物理モデル (`crates/mvmc-physics/`)

```
crates/mvmc-physics/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── models/                  # 物理モデル
│   │   ├── mod.rs
│   │   ├── hubbard.rs          # ハバードモデル
│   │   ├── heisenberg.rs       # ハイゼンベルグモデル
│   │   ├── kondo.rs            # 近藤格子モデル
│   │   └── multi_orbital.rs    # 多軌道モデル
│   ├── lattice/                 # 格子構造
│   │   ├── mod.rs
│   │   ├── square.rs
│   │   ├── triangular.rs
│   │   ├── honeycomb.rs
│   │   └── kagome.rs
│   ├── hamiltonian/             # ハミルトニアン
│   │   ├── mod.rs
│   │   ├── builder.rs
│   │   └── matrix_elements.rs
│   └── observables/             # 物理量
│       ├── mod.rs
│       ├── energy.rs
│       ├── green_function.rs
│       ├── structure_factor.rs
│       └── correlation.rs
├── tests/
└── examples/
```

### 2.5 入出力処理 (`crates/mvmc-io/`)

```
crates/mvmc-io/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── input/                   # 入力ファイル処理
│   │   ├── mod.rs
│   │   ├── stdface.rs          # StdFace形式
│   │   ├── toml.rs             # TOML形式
│   │   └── parser.rs
│   ├── output/                  # 出力ファイル処理
│   │   ├── mod.rs
│   │   ├── data.rs
│   │   ├── parameters.rs
│   │   └── visualization.rs
│   ├── formats/                 # ファイル形式
│   │   ├── mod.rs
│   │   ├── hdf5.rs
│   │   ├── json.rs
│   │   └── binary.rs
│   └── validation/              # 入力検証
│       ├── mod.rs
│       └── schema.rs
├── tests/
└── schemas/                     # JSONスキーマなど
```

### 2.6 並列計算 (`crates/mvmc-parallel/`)

```
crates/mvmc-parallel/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── mpi/                     # MPI並列化
│   │   ├── mod.rs
│   │   ├── communicator.rs
│   │   ├── collective.rs
│   │   └── point_to_point.rs
│   ├── threading/               # スレッド並列化
│   │   ├── mod.rs
│   │   ├── thread_pool.rs
│   │   └── work_stealing.rs
│   ├── distributed/             # 分散計算
│   │   ├── mod.rs
│   │   ├── data_distribution.rs
│   │   └── load_balancing.rs
│   └── gpu/                     # GPU計算（将来拡張）
│       ├── mod.rs
│       └── cuda.rs
├── tests/
└── benchmarks/
```

### 2.7 CLIアプリケーション (`crates/mvmc-cli/`)

```
crates/mvmc-cli/
├── Cargo.toml
├── src/
│   ├── main.rs
│   ├── commands/                # CLIコマンド
│   │   ├── mod.rs
│   │   ├── optimize.rs
│   │   ├── calculate.rs
│   │   ├── analyze.rs
│   │   └── convert.rs
│   ├── config/                  # 設定管理
│   │   ├── mod.rs
│   │   └── cli_config.rs
│   └── ui/                      # ユーザーインターフェース
│       ├── mod.rs
│       ├── progress.rs
│       └── output.rs
├── tests/
└── man/                         # マニュアルページ
```

### 2.8 Cバインディング (`crates/mvmc-bindings/`)

```
crates/mvmc-bindings/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── ffi/                     # FFI定義
│   │   ├── mod.rs
│   │   ├── lapack.rs
│   │   ├── scalapack.rs
│   │   └── mkl.rs
│   └── wrappers/                # 安全なラッパー
│       ├── mod.rs
│       ├── linear_algebra.rs
│       └── parallel.rs
├── build.rs                     # ビルドスクリプト
├── c/                           # Cヘッダーファイル
└── tests/
```

## 3. ワークスペース設定

### 3.1 メインCargo.toml

```toml
[workspace]
members = [
    "crates/mvmc-core",
    "crates/mvmc-cli",
    "crates/mvmc-math",
    "crates/mvmc-physics",
    "crates/mvmc-io",
    "crates/mvmc-parallel",
    "crates/mvmc-bindings",
]
resolver = "2"

[workspace.package]
edition = "2024"
rust-version = "1.85"

[workspace.dependencies]
# 共通依存関係
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
toml = "0.8"
anyhow = "1.0"
thiserror = "1.0"
rayon = "1.7"
ndarray = "0.15"
num-complex = "0.4"
nalgebra = "0.32"
```

## 4. 移行戦略

### 4.1 段階的移行計画

#### Phase 1: 基盤構築
- `mvmc-math` クレートから開始
- 基本的な数値計算機能をRustで実装
- CライブラリとのFFIバインディング

#### Phase 2: コア機能
- `mvmc-core` クレートの実装
- 波動関数、最適化アルゴリズムの移植
- 段階的にCコードを置き換え

#### Phase 3: 物理モデル
- `mvmc-physics` クレートの実装
- 各物理モデルのRust実装

#### Phase 4: 統合・最適化
- CLI、入出力、並列化の実装
- パフォーマンス最適化
- テスト・ベンチマークの充実

### 4.2 移行の優先順位

1. **高優先度**
   - 基本的な数値計算（線形代数、乱数生成）
   - 波動関数の基本構造（Slater行列、Pfaffian）
   - 入力ファイル解析

2. **中優先度**
   - 最適化アルゴリズム
   - 物理モデルの実装
   - 並列化機能

3. **低優先度**
   - 高度な可視化機能
   - GPU計算サポート
   - 実験的なアルゴリズム

## 5. 設計原則

### 5.1 基本原則

- **モジュール性**: 各クレートは独立して使用可能
- **型安全性**: Rustの型システムを活用した安全な設計
- **パフォーマンス**: ゼロコスト抽象化の活用
- **並列性**: `rayon`、`tokio`などを活用した効率的な並列化
- **拡張性**: 新しい物理モデルやアルゴリズムの追加が容易
- **互換性**: 既存のC実装との段階的移行

### 5.2 技術的考慮事項

#### メモリ管理
- Rustの所有権システムを活用した安全なメモリ管理
- 大きな配列の効率的な管理（`ndarray`、`nalgebra`の活用）

#### 並列化
- `rayon`によるデータ並列化
- MPIバインディングによる分散並列化
- 非同期処理（`tokio`）の活用

#### 数値計算
- `ndarray`による多次元配列操作
- `nalgebra`による線形代数計算
- BLAS/LAPACKのFFIバインディング

#### エラーハンドリング
- `anyhow`、`thiserror`による統一されたエラー処理
- 型安全なエラー伝播

## 6. 入力ファイル形式

### 6.1 既存形式のサポート

#### StdFace形式（`.def`）
```
W = 4
L = 2
model = "FermionHubbard"
lattice = "Tetragonal"
t = 1.0
U = 4.0
nelec = 8
```

#### TOML形式（`.toml`）
```toml
[lattice]
Lx = 6
Ly = 1
model_type = "Hubbard"
[mVMC]
sub_x = 2
sub_y = 1
```

### 6.2 新形式の提案

#### JSON形式（`.json`）
```json
{
  "lattice": {
    "type": "square",
    "size": [4, 4]
  },
  "model": {
    "type": "hubbard",
    "parameters": {
      "t": 1.0,
      "U": 4.0
    }
  },
  "calculation": {
    "electrons": 8,
    "optimization": {
      "method": "cg",
      "max_iterations": 500
    }
  }
}
```

## 7. テスト戦略

### 7.1 Test Driven Development (TDD) アプローチ

このプロジェクトでは、Test Driven Development（テスト駆動開発）を採用し、以下の原則に従って開発を進めます：

#### TDDの基本サイクル
1. **Red**: 失敗するテストを書く
2. **Green**: テストが通る最小限のコードを書く
3. **Refactor**: コードをリファクタリングして品質を向上させる

#### TDDの適用範囲
- **単体テスト**: 各関数・メソッドの動作を保証
- **統合テスト**: モジュール間の連携を検証
- **プロパティベーステスト**: 数値計算の性質を検証
- **回帰テスト**: 既存C実装との結果一致を保証

### 7.2 テストの種類

#### 7.2.1 単体テスト
- **各モジュールの個別テスト**: 各関数・メソッドの動作検証
- **境界値テスト**: エッジケースでの動作確認
- **エラーハンドリングテスト**: 異常系の処理確認

#### 7.2.2 統合テスト
- **モジュール間の連携テスト**: 複数クレート間の連携確認
- **エンドツーエンドテスト**: 入力から出力までの一連の流れ
- **パフォーマンステスト**: 計算時間・メモリ使用量の測定

#### 7.2.3 プロパティベーステスト
- **数値計算の性質**: 線形代数の性質（結合則、分配則など）
- **物理量の保存**: エネルギー保存則などの物理法則
- **並列計算の一貫性**: 並列・直列計算の結果一致

#### 7.2.4 回帰テスト
- **既存C実装との結果比較**: 同一入力での出力一致
- **ベンチマーク**: パフォーマンスの維持・向上確認
- **既知の解析解との比較**: 理論値との一致確認

### 7.3 テストデータ戦略

#### 7.3.1 既存データの活用
- **既存のサンプルファイル**: mVMCのサンプルデータをテストに活用
- **既知の解析解**: 小さな系での理論解との比較
- **ベンチマークデータ**: 標準的なテストケース

#### 7.3.2 自動生成データ
- **ランダムテストケース**: プロパティベーステスト用
- **境界値データ**: 極値・特殊ケースの自動生成
- **ストレステストデータ**: 大規模系での動作確認

#### 7.3.3 テストデータ管理
- **データバージョニング**: テストデータの変更履歴管理
- **データ検証**: テストデータの妥当性確認
- **データ最適化**: テスト実行時間の短縮

### 7.4 TDD実装ガイドライン

#### 7.4.1 テストファーストの原則
```rust
// 例: 複素数演算のテストファースト実装

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_addition() {
        // 1. Red: 失敗するテストを書く
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);
        let expected = Complex::new(4.0, 6.0);
        assert_eq!(a + b, expected);
    }
}

// 2. Green: 最小限の実装
impl Add for Complex {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Complex::new(self.re + other.re, self.im + other.im)
    }
}

// 3. Refactor: 最適化・改善
```

#### 7.4.2 テスト構造化
- **Given-When-Then**: テストの構造化
- **テストケースの独立性**: 各テストが独立して実行可能
- **テストデータの分離**: テスト用データの明確な分離

#### 7.4.3 継続的テスト
- **CI/CD統合**: 全テストの自動実行
- **テストカバレッジ**: コードカバレッジの監視
- **テストパフォーマンス**: テスト実行時間の最適化

### 7.5 数値計算特有のテスト戦略

#### 7.5.1 浮動小数点数の比較
```rust
// 浮動小数点数の近似比較
fn assert_approx_eq(a: f64, b: f64, epsilon: f64) {
    assert!((a - b).abs() < epsilon, "{} != {} (epsilon: {})", a, b, epsilon);
}

#[test]
fn test_matrix_multiplication() {
    let a = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = Matrix::from_vec(vec![5.0, 6.0, 7.0, 8.0], 2, 2);
    let result = &a * &b;
    let expected = Matrix::from_vec(vec![19.0, 22.0, 43.0, 50.0], 2, 2);

    for i in 0..result.rows() {
        for j in 0..result.cols() {
            assert_approx_eq(result[(i, j)], expected[(i, j)], 1e-10);
        }
    }
}
```

#### 7.5.2 プロパティベーステスト
```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_matrix_transpose_property(m in matrix_strategy()) {
        let transposed = m.transpose();
        let double_transposed = transposed.transpose();

        // 転置の転置は元の行列と等しい
        prop_assert_eq!(m, double_transposed);
    }

    #[test]
    fn test_complex_multiplication_distributive(
        a in complex_strategy(),
        b in complex_strategy(),
        c in complex_strategy()
    ) {
        // 分配則: a * (b + c) = a * b + a * c
        let left = a * (b + c);
        let right = a * b + a * c;
        prop_assert_approx_eq!(left.re, right.re, epsilon = 1e-10);
        prop_assert_approx_eq!(left.im, right.im, epsilon = 1e-10);
    }
}
```

#### 7.5.3 物理量の保存則テスト
```rust
#[test]
fn test_energy_conservation() {
    let mut system = HubbardModel::new(4, 4, 8);
    let initial_energy = system.calculate_energy();

    // モンテカルロステップを実行
    for _ in 0..1000 {
        system.metropolis_step();
    }

    let final_energy = system.calculate_energy();

    // エネルギーは保存される（統計的誤差の範囲内）
    assert_approx_eq!(initial_energy, final_energy, 0.01);
}
```

### 7.6 テスト環境の構築

#### 7.6.1 テスト用クレート
```toml
# Cargo.toml
[dev-dependencies]
proptest = "1.0"
criterion = "0.5"
mockall = "0.11"
```

#### 7.6.2 ベンチマーク設定
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_matrix_multiplication(c: &mut Criterion) {
    let a = Matrix::random(100, 100);
    let b = Matrix::random(100, 100);

    c.bench_function("matrix_mul_100x100", |bencher| {
        bencher.iter(|| {
            black_box(&a * &b)
        })
    });
}

criterion_group!(benches, benchmark_matrix_multiplication);
criterion_main!(benches);
```

### 7.7 テスト品質の保証

#### 7.7.1 テストカバレッジ
- **目標カバレッジ**: 90%以上
- **カバレッジ測定**: `cargo tarpaulin`の使用
- **カバレッジレポート**: 定期的なレポート生成

#### 7.7.2 テストメンテナンス
- **テストの可読性**: 明確なテスト名とコメント
- **テストの保守性**: 重複の排除と共通化
- **テストの安定性**: 非決定的なテストの排除

#### 7.7.3 テストドキュメント
- **テスト戦略の文書化**: テスト方針の明確化
- **テストケースの説明**: 各テストの目的と期待値
- **テスト実行手順**: 開発者向けの実行ガイド

## 8. ドキュメント戦略

### 8.1 ドキュメントの種類

- **API文書**: `cargo doc`による自動生成
- **チュートリアル**: 段階的な使用例
- **移行ガイド**: C実装からの移行手順
- **パフォーマンスガイド**: 最適化のヒント

### 8.2 ドキュメント生成

- `rustdoc`によるAPI文書の自動生成
- `mdbook`によるチュートリアル文書
- コード例の自動テスト

## 9. 今後の拡張計画

### 9.1 短期目標（6ヶ月）

- 基本的な数値計算ライブラリの実装
- 単純なハバードモデルの実装
- 基本的なCLIの実装

### 9.2 中期目標（1年）

- 全物理モデルの実装
- 並列化機能の実装
- 既存C実装との互換性確保

### 9.3 長期目標（2年）

- GPU計算サポート
- 高度な可視化機能
- 機械学習との統合

## 10. リスク管理

### 10.1 技術的リスク

- **パフォーマンス**: C実装との性能差
- **互換性**: 既存データ形式との互換性
- **複雑性**: アルゴリズムの複雑さ

### 10.2 軽減策

- 段階的な移行によるリスク分散
- 包括的なテストによる品質確保
- 既存実装との並行開発

## 11. 結論

この計画書は、mVMCのC実装をRustに移植するための包括的なロードマップを提供します。段階的なアプローチにより、既存の機能を保持しながら、Rustの利点を最大限活用した保守性の高いコードベースを構築できます。

Rustの型安全性、メモリ安全性、並列性の利点を活用することで、より安全で効率的な量子格子モデル計算ソフトウェアの実現が期待されます。


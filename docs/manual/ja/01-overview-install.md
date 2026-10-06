# 1. 概要とインストール

[目次](README.md) · 次へ: [2. 理論I: 変分モンテカルロ法とハミルトニアン](02-theory-vmc-hamiltonian.md)

## 1.1 このパッケージについて

`mvmc-rs` は、格子フェルミオン系およびスピン系のための多変数変分モンテカルロ法ソルバー **mVMC** の、純粋な Rust による移植版です。mVMC と同様に、次のことを行います。

1. *Expertモード*の入力一式(`namelist.def` と、そこに列挙された定義ファイル)を読み込みます。
2. 相関因子と量子数射影を伴うパフィアンペア積波動関数のパラメータを、確率的再構成法(SR法)により最適化します(`NVMCCalMode = 0`)。
3. 固定されたパラメータセットに対して物理量(グリーン関数、およびオプションで単一ステップの Lanczos 補正)を評価します(`NVMCCalMode = 1`)。

Rust ワークスペースは、本マニュアル全体で用いる階層的な基準規則に従います([AGENTS.md](../../../AGENTS.md) を参照)。

| 項目 | 基準 |
|------|------|
| 物理、アルゴリズム、入力規約、出力フォーマット、パラメータ配置、RNG の乱数生成順序、演算順序と符号 | C 実装 `extern/mVMC-1.3.0` |
| 公開 Rust API の形、ランナーの構造、ライフサイクル、テスト構成 | Julia 移植版 `extern/Julia-mVMC` |
| 浮動小数点の比較 | 明示的な絶対/相対許容誤差。ビット単位の比較は行わない([11.4](11-compatibility.md#114-数値比較ポリシー)を参照) |

本プロジェクトは*純粋な Rust* 実装です。BLAS/LAPACK(OpenBLAS)にはリンクしますが、mVMC や PFAPACK に対する C や Fortran の外部関数依存はなく、Rust のビルドとテストに `c_toolbox/` は不要です。

## 1.2 ワークスペース構成

| クレート | 役割 |
|-------|------|
| `crates/sfmt19937` | C 互換のシード設定、32 ビット乱数、`genrand_real2` 変換を備えた SFMT-19937 乱数生成器(`Sfmt19937Rng`, `crates/sfmt19937/src/lib.rs:74`) |
| `crates/pfapack` | パフィアン、歪対称 LTL 分解、逆行列カーネル(`pfaffian_ltl_real`, `zsktf2_c_compat`, `utu2inv_complex`)。スカラーバックエンドと BLAS バックエンドを持ちます |
| `crates/mvmc-expert-parsers` | Expertモードのファイルパーサー、`ExpertModeData`、パラメータ初期化、量子射影の重み |
| `crates/mvmc-core` | VMC エンジン本体: サンプリング、物理量、SR、Lanczos、出力、MPI リデューサー、検証 |
| `crates/mvmc-cli` | `mvmc` バイナリと 4 つのサンプルプログラム |
| `crates/mvmc-greenr2k` | `greenr2k` バイナリ: グリーン関数のフーリエ変換(`tool/greenr2k.F90` の移植、[9.9](09-output-files.md#99-後処理-greenr2k-グリーン関数のフーリエ変換)) |
| `xtask` | ベンチマークと回帰テストの自動化 |

## 1.3 サポート範囲

| 機能 | `mvmc-rs` での状況 |
|------------|---------------------|
| 直接SR(`NSRCG = 0`)またはCG SR(`NSRCG = 1`)によるパラメータ最適化(`NVMCCalMode = 0`) | サポート |
| 固定パラメータでの物理量計算(`NVMCCalMode = 1`): `OneBodyG`, `TwoBodyG`, `TwoBodyGEx` | サポート |
| 単一ステップ Lanczos法(`NLanczosMode = 1, 2`) | 任意の `NSplitSize` で `InterAll` なし、スピンを変える `Trans` なしの sz 保存経路でサポート([7.5](07-input-files.md#75-サポートされる入力と拒否される入力)を参照) |
| 実数および複素数の波動関数 | サポート(入力宣言により決まります。[3.3](03-theory-wavefunction.md#33-実数モードと複素数モード)を参照) |
| `Orbital`/`OrbitalAntiParallel`, `OrbitalParallel`, `OrbitalGeneral` (FSZ) | サポート |
| Gutzwiller因子、Jastrow因子、2/4サイトのダブロン-ホロン相関因子、電荷/スピン/一般のRBM、`OptTrans` | サポート |
| `InterAll` ハミルトニアン項 | パース済みで最適化時に評価されます。この項目群の所有は本マニュアルとは別です |
| BackFlow(`BF`, `BFRange`)、`SpinJastrow`、`NSRCG >= 2`、`useDiagScale`、`RescaleSmat` | 拒否 |
| 複数定義モード(`-m N`、MPI グループとディレクトリごとに 1 計算) | `mpi` フィーチャー、またはシリアルでの `-m 1` でサポート([8.1](08-running.md#multidef-モード-m)) |
| グループ実行(`NSplitSize > 1`)を含む MPI | `mpi` フィーチャーでサポート。制約あり([8.4](08-running.md#84-mpiとグループ実行)) |

## 1.4 動作要件

- 最近の安定版 Rust ツールチェーン。`rust-toolchain.toml` は `rustfmt` と `clippy` を含む `stable` チャンネルを選択します。固定された MSRV はありません。
- **OpenBLAS と LAPACK(LP64 インターフェース)**。`crates/mvmc-core/build.rs` は `cargo:rustc-link-lib=dylib=openblas` を出力します。Linux ではシステムパッケージ(Debian/Ubuntu では `libopenblas-dev liblapack-dev`)をインストールしてください。macOS では `brew install openblas` を実行します。この formula は keg-only であり、ビルドスクリプトが Homebrew のライブラリパス(Apple Silicon では `/opt/homebrew/opt/openblas`、Intel では `/usr/local/opt/openblas`)を追加します。
- `mpi` フィーチャーを使う場合: C コンパイララッパーを備えた MPI 実装と `libclang`(`mpi` クレートがバインディングを生成します)。CI のセットアップアクションは、Ubuntu では `libopenblas-dev liblapack-dev libclang-dev libmpich-dev mpich pkg-config` を、macOS では `openblas mpich` をインストールします
  (`.github/actions/setup-rust-ci/action.yml`)。
- Linux x86_64 が数値のリファレンスプラットフォームです。Dev Container が用意されており([docs/DEV_CONTAINER.md](../../DEV_CONTAINER.md))、macOS は移植性の確認に使われます([docs/NUMERICAL_COMPARISONS.md](../../NUMERICAL_COMPARISONS.md))。

## 1.5 Cargoフィーチャー

| フィーチャー | 対象 | 意味 |
|---------|-------|---------|
| `blas-backend` | `pfapack` | パフィアンカーネルに BLAS/LAPACK ルーチン(`dger`, `zgeru`, `dtrtri`, `dtrmm`, `dscal`, ...)を使用します。**`mvmc-core` は常にこれを有効にします**(`crates/mvmc-core/Cargo.toml` の `pfapack = { features = ["blas-backend"] }`)。そのため、オプティマイザーは常に BLAS/LAPACK バックエンドを使います。単体の `pfapack` クレートは、デフォルトではスカラーのリファレンスバックエンドを使います。 |
| `simd-backend` | `pfapack` | 単体の `pfapack` クレートで SIMD(`pulp`)カーネルを有効にします。`mvmc-core` や `mvmc-cli` からは**転送されません**(`crates/mvmc-cli/Cargo.toml` が定義するのは `mpi` と `gpu-cuda` のみです)。PfaPack のテストとベンチマークに使用します: `cargo nextest run -p pfapack --features 'simd-backend blas-backend'`。 |
| `mpi` | `mvmc-core`, `mvmc-cli` | MPI サポート(`MpiContext`, `MpiGroupContext`)をコンパイルします。これがない場合、ランチャーが検出したマルチランク実行は、`--features mpi` で再ビルドするよう求めるエラーで拒否されます。 |
| `gpu-cuda` | `mvmc-core`, `mvmc-cli` | デフォルトで無効。依存を追加せず `Cargo.lock` も変更しません。`mvmc_core::backend` に tenferro CUDA プロバイダーを登録できるようにします(無効時の `BackendKind::Cuda` はエラーで、CPU への暗黙のフォールバックはありません)。CUDA の依存ツリーは独立したワークスペース `gpu/mvmc-gpu-cuda` で別にビルドします。オプションのゲートは `scripts/run_cuda_gate.sh` です(`MVMC_RS_CUDA_GATE=1` ではデバイスがなければ失敗し、未設定では「skipped, no device」と明示します)。`docs/design/gpu-readiness.md` の第 10 節を参照してください。 |

## 1.6 ビルドとスモークテスト

```bash
# type-check everything
cargo check --workspace

# build the optimized CLI (serial)
cargo build --release -p mvmc-cli
# ... with MPI support
cargo build --release -p mvmc-cli --features mpi

# the binary is target/release/mvmc
target/release/mvmc --help
```

[AGENTS.md](../../../AGENTS.md) にある開発用チェック:

```bash
cargo nextest run --workspace --cargo-profile test-fast   # all unit/integration tests
cargo test --workspace --doc                              # doc tests (not run by nextest)
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
```

エンドツーエンドの動作例は[第10章](10-tutorial.md)に示します。本マニュアルのために行った `cargo build --release -p mvmc-cli` のビルドは、Linux x86_64 のクリーンな target ディレクトリから約 6 分で完了しました **(観測)**。`--features mpi` のビルドは、本マニュアルの執筆中には実行して**いません** **(未検証)**。

## 1.7 次に読むもの

- プログラムが何を計算するかを理解するには: [第2〜6章](02-theory-vmc-hamiltonian.md)。
- 計算を実行するには: [第7章](07-input-files.md)、[第8章](08-running.md)、[第9章](09-output-files.md)、および[チュートリアル](10-tutorial.md)。
- 開発者向けドキュメント: [docs/DEVELOPMENT.md](../../DEVELOPMENT.md)、
  [docs/NUMERICAL_COMPARISONS.md](../../NUMERICAL_COMPARISONS.md)、
  [docs/PORTING_PLAN.md](../../PORTING_PLAN.md)。

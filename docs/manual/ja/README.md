# mvmc-rs ユーザーマニュアル

**言語:** [English](../README.md) · 日本語

このマニュアルは、多変数変分モンテカルロ法パッケージ mVMC の Rust 版(`mvmc-rs`)について説明します。
理論から出発し、主要な式のそれぞれについて、基準となる動作を定義する C 関数と、それを実装する
Rust 関数との対応を示します。後半の章は利用リファレンスで、入力ファイル、コマンドライン、並列実行、
出力ファイル、実際の計算例(チュートリアル)、C 実装および Julia 実装との差異、そしてオプションの高速化(tenferro/CUDA)バックエンドを扱います。

マニュアルは GitHub 上で表示されるプレーンな Markdown で書かれています。数式には
`$...$` および `$$...$$`(GitHub がレンダリングします)を使います。この日本語版は英語版
([`../README.md`](../README.md))と同じファイル名・同じ構成の翻訳です。コード識別子、ファイルパス、
環境変数、数式、`file:line` 引用は翻訳せず、英語版と同一です。

## 目次

| # | 章 | 内容 |
|---|----|------|
| 1 | [概要とインストール](01-overview-install.md) | ワークスペースの構成、C コードが基準であること、動作要件、Cargo フィーチャー(`blas-backend`、`simd-backend`、`mpi`)、ビルドとスモークテスト |
| 2 | [理論 I: VMC とハミルトニアン](02-theory-vmc-hamiltonian.md) | 変分モンテカルロ法の推定量、ローカルエネルギー、サポートされるハミルトニアンの項、1体・2体グリーン関数の比 |
| 3 | [理論 II: 変分波動関数](03-theory-wavefunction.md) | パフィアンペア積部分(normal、parallel、general/FSZ 軌道、実数・複素数)、Gutzwiller、Jastrow、ダブロン-ホロン、RBM、量子数射影、パラメータ配置と同期 |
| 4 | [理論 III: マルコフ連鎖サンプリング](04-theory-sampling.md) | 配置、初期サンプル、ホッピング/交換/局所スピン反転の更新、採択、パフィアンと逆行列の更新、バーンイン |
| 5 | [理論 IV: 確率的再構成法](05-theory-sr.md) | 対数微分 $O_k$、$S$ 行列と力、カットオフと対角シフト、直接法と CG 法、パラメータ更新、最終平均 |
| 6 | [理論 V: 物理量と Lanczos 法](06-theory-observables-lanczos.md) | 1体・2体グリーン関数、重み付き平均、シングルステップ Lanczos のエネルギーとグリーン関数(`NLanczosMode` 1/2) |
| 7 | [入力ファイル](07-input-files.md) | `namelist.def`、`modpara.def`、Expert モードの定義ファイル、初期パラメータファイル。Rust がサポートするもの・拒否するもの |
| 8 | [実行方法](08-running.md) | `mvmc` CLI、シリアル/MPI/グループ(`NSplitSize`)実行、環境変数 |
| 9 | [出力ファイル](09-output-files.md) | Rust が書き出すすべてのファイル、その列、どのランクが書き出すか |
| 10 | [チュートリアル](10-tutorial.md) | Hubbard 鎖の最適化に続く物理量計算の完全な例(実際の出力つき) |
| 11 | [互換性と差異](11-compatibility.md) | C および Julia との差異、数値比較の方針、未解決の観測事項 |
| 12 | [アクセラレータ(GPU)バックエンド](12-accelerated-backends.md) | オプションの tenferro/CUDA バックエンド: 何があるか、ビルドと実行(native、docker、`gpu-cuda`)、`MVMC_RS_SR_BACKEND`、検証と許容誤差、マルチウォーカー実行、既知の制限、#450 のベンチマークスイート |
| A | [付録: 引用の検証方法](appendix-checks.md) | 引用チェッカー、観測出力の出所、未検証の事項 |

## 「実装」ボックスの読み方

主要な式のあとには、次の形式のボックスが置かれます。

> **実装**
> - C: `Function` — `extern/mVMC-1.3.0/src/mVMC/file.c:LINE`
> - Rust: `function` — `crates/<crate>/src/file.rs:LINE`
> - 整合性: 数値比較で重要になる演算順序や契約に関する注記。

規約:

- 数値的な動作(パラメータ配置、乱数の引き順、演算順序、符号、ファイル形式)については
  **C が基準(正)** です。Julia-mVMC は Rust の公開 API とランナー構造の設計リファレンスです。
  [AGENTS.md](../../../AGENTS.md) と [第 11 章](11-compatibility.md) を参照してください。
- ファイルと行番号は次に対して確認しました。
  - C リファレンス `extern/mVMC-1.3.0`(git サブモジュール、コミット `d73d06bd`)
  - このリポジトリのコミット `37348eac`(`main`、"Add public RBM flag refresh and manual orbital mode (#336)")

  行番号はコードの変更とともにずれていきます。安定したキーはシンボル名です。
  引用をどのように検証したか、および検証していない事項は
  [`appendix-checks.md`](appendix-checks.md) に記載しています。
- **(未検証)** と記した記述は、ドキュメントやコードの読解に基づくもので、この作業中には実行して
  確認していません(たとえば MPI 実行や macOS での動作)。**(観測)** と記した記述は、このマニュアルの
  執筆中に `mvmc` のリリースビルドで再現したものです。
- Rust の API は活発に開発中であり、後方互換性は目標ではありません([AGENTS.md](../../../AGENTS.md)
  参照)。ここに書かれた内容は変更される可能性があります。

## 参照した資料

- C マニュアル: `extern/mVMC-1.3.0/doc/en/source/{algorithm,expert,output,standard,start,tutorial}.rst`
  (日本語版マニュアルは `doc/ja`)。
- C 実装: `extern/mVMC-1.3.0/src/mVMC/`。
- Julia ドキュメント(構成のみ): `extern/Julia-mVMC/docs/src/en/*.md`。
- Rust ワークスペース: `crates/`。

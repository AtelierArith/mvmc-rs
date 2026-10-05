# 7. 入力ファイル

[目次](README.md) · 前へ: [6. 理論V: 物理量とLanczos法による補正](06-theory-observables-lanczos.md) · 次へ: [8. 実行](08-running.md)

`mvmc` は、C プログラムとまったく同じように **Expert モード**の入力を読み込みます。1 つのリストファイル(`namelist.def`)が定義ファイルの集合を指名し、各ファイルは固定のキーワードで導入されます。各ファイル形式の基準となる記述は C マニュアル
[`extern/mVMC-1.3.0/doc/en/source/expert.rst`](../../../extern/mVMC-1.3.0/doc/en/source/expert.rst)
(日本語版: `doc/ja`)です。この章では構造を要約し、Rust 実装が異なる点を記録し、サポート/拒否の一覧を示します。この一覧は `crates/mvmc-core/src/validation.rs` から直接取ったものです。

C パッケージの Standard モードのフロントエンド(`vmcdry`/StdFace、`-s` オプション)は Rust ワークスペースの一部では**ありません**。Expert モードのファイルは C のツールで生成するか、`benchmark/hubbard_chain/inputs/` および `tests/fixtures/` にコミットされている例を使ってください([チュートリアル](10-tutorial.md)を参照)。

## 7.1 `namelist.def`

コメントでない各行は `Keyword filename` の形式です。キーワードは任意の順序で現れてよく、`#` で始まる行はスキップされ、キーワードの照合は ASCII の大文字小文字を区別しません(`CheckWords`)。
C のキーワードリストは `cKWListOfFileNameList` (`extern/mVMC-1.3.0/src/mVMC/include/readdef.h:33`)です。

```text
ModPara LocSpin Trans CoulombIntra CoulombInter Hund PairHop Exchange Gutzwiller Jastrow DH2 DH4
{Charge,Spin,General}RBM_{HiddenLayer,PhysLayer,PhysHidden}   (9 keywords)
Orbital OrbitalAntiParallel OrbitalParallel OrbitalGeneral TransSym
InGutzwiller InJastrow InDH2 InDH4  In{Charge,Spin,General}RBM_{HiddenLayer,PhysLayer,PhysHidden}
InOrbital InOrbitalAntiParallel InOrbitalParallel InOrbitalGeneral
OneBodyG TwoBodyG TwoBodyGEx InterAll OptTrans InOptTrans BF BFRange
```

Rust のパーサ(`C_NAMELIST_KEYWORDS`, `crates/mvmc-expert-parsers/src/lib.rs:322`)は同じテーブルを同じ順序で使います。さらに、エイリアス `DoublonHolon2Site`, `DoublonHolon4Site`(`DH2`, `DH4` に対応)および `QPTrans`(`TransSym` に対応)を受け付けます。これらは Rust/Julia の拡張であり、C の綴りではありません。

例(`benchmark/hubbard_chain/inputs/hubbard_chain_L16/namelist.def`):

```text
ModPara  modpara.def
LocSpin  locspn.def
Trans    trans.def
CoulombIntra  coulombintra.def
OneBodyG greenone.def
TwoBodyG greentwo.def
Gutzwiller gutzwilleridx.def
Jastrow  jastrowidx.def
Orbital  orbitalidx.def
TransSym qptransidx.def
```

## 7.2 `modpara.def`

各行は `Key value` の形式です(Rust パーサではオプションとして `Key = value` も受け付けます。`-` で始まる行は装飾です)。パーサは `parse_modpara_content`
(`crates/mvmc-expert-parsers/src/parsers/modpara.rs:20`)で、キーのテーブルは `apply_param`
(`modpara.rs:45`)にあります。C のリーダは `GetInfoFromModPara` (`readdef.c:1825`)です。

**デフォルト値。** 指定されていないキーは以下の値を取ります。C の値は
`SetDefaultValuesModPara` (`readdef.c:1756`)に由来し、いくつかのキーで **C マニュアルの記述と異なります**
(マニュアル: `NSPGaussLeg` 8、`NVMCSample` 1000。コード: 1 と 10)。基準となるのはコードです。Rust の値は
`impl Default for ModParaParameters` (`crates/mvmc-expert-parsers/src/types.rs:233`)であり、Julia のパーサに従っているため **C とは異なります**。重要なキーはすべて明示的に指定してください。

| キー | 意味 | C のデフォルト | Rust のデフォルト |
|-----|---------|-----------|--------------|
| `CDataFileHead`, `CParaFileHead` | データ(`zvo`)/パラメータ(`zqp`)出力ファイルの接頭辞 | –(必須の行) | `zvo`, `zqp` |
| `NVMCCalMode` | 0: 最適化、1: 物理量(CLI では `--physcal` と一致している必要があります。[8.1](08-running.md#計算の選択)) | 0 | 0 |
| `NLanczosMode` | 0 なし、1 エネルギー、2 エネルギー + グリーン関数(PhysCal のみ) | 0 | 0 |
| `NDataIdxStart`, `NDataQtySmp` | 出力ファイルの番号付け。PhysCal のサンプル数 | 0, 1 | 0, 1 |
| `Nsite` | サイト数 | 16 | 0 |
| C: `Nelectron` または `Ne`; Rust: `NElec` または `Nelec` | スピンあたりの電子数($N_e$)。**2 つの綴りは互いに素です**。Rust は `Nelectron`/`Ne` を黙って無視し(観測: $N_e=0$ となり、その後に無関係な後発のエラーが出ます)、C は `NElec`/`Nelec` を読みません。両者が読む `Ncond` を推奨します | 8 | 0 |
| `Ncond` | 伝導電子数。設定された場合は $N_e=(N_{\rm locspin}+N_{\rm cond})/2$ | -1 | -1 |
| `2Sz` | $2S_z$。-1 は「固定しない」を意味します(一般軌道が必要) | -1 | -1 |
| `NSPGaussLeg`, `NSPStot` | スピン射影のメッシュと $S$ | 1, 0 | 1, 0 |
| `NMPTrans` | $\lvert N_{\rm MP}\rvert$ 個の並進。負の値 = 反周期的。**ゼロ以外でなければなりません**(1 = なし) | 0 | 0 |
| `NSROptItrStep`, `NSROptItrSmp` | SR のステップ数と最終平均化ウィンドウ | 1000, step/10 | 1000, 1000 |
| `DSROptRedCut`, `DSROptStaDel`, `DSROptStepDt` | SR のカットオフ、対角シフト、時間ステップ | 0.001, 0.02, 0.02 | 1e-6, 0.0, 0.01 |
| `NSRCG`, `DSROptCGTol`, `NSROptCGMaxIter` | CG の切り替え、許容誤差、最大反復回数 | 0, 1e-10, 0 | 0, 1e-10, 0 |
| `NStore` | サンプルごとの $O$ を保存する(0/1) | 1 | 1 |
| `NVMCWarmUp`, `NVMCInterval`, `NVMCSample` | バーンイン、間隔、サンプル数 | 10, 1, 10 | 1000, 1, 10000 |
| `NExUpdatePath` | 0 ホッピング、1 ホッピング+交換、2 交換(スピン)、3 近藤 | 0 | 1 |
| `RndSeed` | SFMT のシード。ランク/グループ $g$ のシードは `RndSeed + g`。負の値 = 時計 | 11272 | 11272 |
| `NSplitSize` | MPI グループの幅([8.4](08-running.md#84-mpiとグループ実行)) | 1 | 1 |
| `Nneuron`, `NneuronCharge`, `NneuronSpin`, `NneuronGeneral`, `NBlockSize_RBMRatio` | RBM の隠れニューロンとブロックサイズ | 0, 0, 0, 0, 200 | 0, 0, 0, 0, 200 |
| `NFileFlushInterval` | フラッシュ間隔(C の `-F` オプション) | 1 | 1(パースされるが**使われません**) |
| `ComplexType` | レガシーの複素数フラグ | – | 0 |
| `NOneBodyG`, `NTwoBodyG`, `NTwoBodyGEx` | 個数。通常はグリーン関数ファイルから取られます | 0 | 0 |
| `useDiagScale`, `RescaleSmat` | Julia リファレンスのサポートされないオプション | – | 0(ゼロ以外は拒否) |

`NOrbitalIdx` は `modpara.def` のキーではありません。軌道パラメータの個数は軌道ファイルから導出されます。
`NSROptFixSmp` は Rust ではパースされます(デフォルト 0)が、C の `GetInfoFromModPara` のキーではありません。

## 7.3 定義ファイル

すべての定義ファイルは同じ骨格を持ちます。2 行目に個数(`NTransfer 64`)を持つ 5 行のヘッダ、オプションの型の行(`ComplexType 0`)、そして行のテーブルです。C マニュアルでの形式は次のとおりです。

| キーワード | 行の形式 | Cマニュアル |
|---------|-----------|----------|
| `LocSpin` | `site flag`(flag 1 = 局在スピン、0 = 遍歴) | `expert.rst:637` |
| `Trans` | `i σ_i j σ_j Re(t) Im(t)` ($-t\,c^\dagger_{i\sigma_i}c_{j\sigma_j}$) | `expert.rst:721` |
| `InterAll` | `i σ_i j σ_j k σ_k l σ_l Re(I) Im(I)` | `expert.rst:837` |
| `CoulombIntra` | `i U_i` | `expert.rst:960` |
| `CoulombInter`, `Hund`, `Exchange` | `i j value` | `expert.rst:1039`, `1118`, `1278` |
| `PairHop` | `i j value`($(i,j)$ と $(j,i)$ に展開されます) | `expert.rst:1197` |
| `Gutzwiller` | $N_s$ 行の `site idx`、続いて $N_G$ 行の `idx optflag` | `expert.rst:1360` |
| `Jastrow` | サイト対の行 `i j idx`、続いて `idx optflag` | `expert.rst:1486` |
| `DH2`, `DH4` | パートナーサイトのテーブルと `idx optflag` | `expert.rst:1608`, `1744` |
| `*RBM_*` | インデックステーブルと `idx optflag` | `expert.rst:1885-2285` |
| `Orbital`/`OrbitalAntiParallel` | $N_s^2$ 行の `i j idx`(符号の規約はマニュアルを参照)、続いて `idx optflag` | `expert.rst:2285` |
| `OrbitalParallel`, `OrbitalGeneral` | スピン分解版 | `expert.rst:2420`, `2557` |
| `TransSym` | `NQPTrans` 行の `idx Re(weight) Im(weight)`、続いて `idx site mapped_site sign` の行 | `expert.rst:2705` |
| `OneBodyG` | `i σ j σ'` | `expert.rst:2921` |
| `TwoBodyG` | `i σ j σ' k τ l τ'` | `expert.rst:3019` |
| `TwoBodyGEx` | 1 体エントリの対を特定する行(C: `GetInfoTwoBodyGEx`) | Cマニュアルには記述なし |
| `OptTrans`, `InOptTrans` | 最適化される並進の重みと初期値 | Cマニュアルには記述なし |

注記:

- スピンインデックスは 0(アップ)と 1(ダウン)です。サイトインデックスは 0 から始まります。`idx optflag` テーブルのパラメータインデックスは 0 から始まり、$0\ldots N-1$ をカバーしていなければなりません。
- `Trans`, `Hund`, `Exchange`, `CoulombInter`, `CoulombIntra`, `PairHop` は、`crates/mvmc-expert-parsers/src/parsers/`(`coulomb.rs`, `hund.rs`, `exchange.rs`, `pairhop.rs`, `trans.rs`)の型付きローダでパースされます。C のリーダは `GetTransferInfo` (`readdef.c:1973`)、`ReadPairHopValue` (`readdef.c:2038`)、`ReadPairDValue`
  (`readdef.c:2063`)、`GetInfoInterAll` (`readdef.c:2420`)です。`load_hamiltonian_definition` API
  (`crates/mvmc-expert-parsers/src/definition.rs`)は、これらの族のひとつを `ExpertModeData` に読み込みます。
- インデックスファイルの `ComplexType` 行は、そのブロックのパラメータが実数(0)か複素数(1)かを選択し、それにより実数/複素数の実行モードも決まります([3.3](03-theory-wavefunction.md#33-実数モードと複素数モード))。
- パラメータの最適化フラグ(`optflag`)は、1 で最適化、0 で固定です。初期値が与えられていない固定パラメータは 0 に設定され、乱数を消費しません([3.7](03-theory-wavefunction.md#37-初期値と同期))。

## 7.4 初期パラメータ値

初期値は 3 つの供給源から、この順序で適用されます(C: `InitParameter`, `ReadInitParameter`, `ReadInputParameters`。
Rust: `crates/mvmc-core/src/run.rs:1358` に記された "init → initial.def → In\*.def → sync" の順序)。

1. 最適化される RBM および Slater パラメータの**ランダム初期化**([3.7](03-theory-wavefunction.md#37-初期値と同期))。
2. `zqp_opt.dat` の形式の**初期パラメータファイル**(6 個の浮動小数点数のヘッダと、パラメータあたり 3 個の浮動小数点数。最後の完全なレコードが優先されます)。C はこれをドライバの 2 番目の位置引数として渡します
   (`vmcmain.c`, `ReadInitParameter`, `parameter.c:95`)。Rust は `read_initial_def`
   (`crates/mvmc-core/src/initial_params.rs:117`)と CLI オプション `--initial-def auto|none|PATH`
   で読み込みます(`auto` は隣接する `initial.def` があれば読み込みます)。浮動小数点数の個数は
   $6+3\,(N_{\rm proj}+N_{\rm RBM}+N_{\rm Slater}+N_{\rm OptTrans})$ と一致しなければなりません(`load_para_triples`)。
3. **`In*` ファイル**(`InGutzwiller`, `InJastrow`, `InDH2`, `InDH4`, `In*RBM_*`, `InOrbital*`, `InOptTrans`): ヘッダと
   行 `idx Re Im` — 実行によって書き出されるブロックごとのファイル `zqp_*_opt.dat`
   ([9.3](09-output-files.md#93-最適化されたパラメータ))と同じ形式なので、実行の出力をそのまま入力に戻すことができます。

固定パラメータの PhysCal 実行では、`--physcal` に与えられたパラメータファイルが `read_opt_para_file`
(`crates/mvmc-core/src/initial_params.rs:141`)で読み込まれます。

## 7.5 サポートされる入力と拒否される入力

以下のチェックは `crates/mvmc-core/src/validation.rs`(コミット `37348eac`)にあります。これらは初期化や出力より前に実行され、拒否された入力は `error: ...` とゼロ以外の終了ステータスで終了します。

### ネームリストのキーワード(`validate_para_opt`, `crates/mvmc-core/src/validation.rs:162-214`)

| キーワード | パラメータ最適化 | PhysCal |
|----------|-----------------------|---------|
| `ModPara`, `LocSpin`, `Trans`, `CoulombIntra`, `CoulombInter`, `Hund`, `Exchange`, `Gutzwiller`, `Jastrow`, `Orbital`, `OrbitalAntiParallel`, `OrbitalParallel`, `OrbitalGeneral`, `OneBodyG`, `TwoBodyG`, `TransSym` (alias `QPTrans`) | 受理 | 受理 |
| `PairHop`, `InterAll`, `DH2` (`DoublonHolon2Site`), `DH4` (`DoublonHolon4Site`), `OptTrans`, `InOptTrans`, all `{Charge,Spin,General}RBM_*`, `In{Gutzwiller,Jastrow,Orbital,OrbitalAntiParallel,OrbitalParallel,OrbitalGeneral,DH2,DH4}` (and aliases `InDoublonHolon*`), `In{Charge,Spin,General}RBM_*` | 受理 | 受理 |
| `TwoBodyGEx` | 受理され無視される | 受理 **(観測)** |
| その他の `In…` キーワード | 拒否("not implemented yet (issue #20)") | チェックされない |
| `SpinJastrow` | 拒否("projection layout would be wrong") | チェックされない |
| `BF`, `BFRange` (BackFlow) および未知のキーワード | 拒否("unsupported namelist section …") | **チェックされない: `BF` エントリは受理され、BackFlow は適用されません(現在の `main` で観測)** |

PhysCal の列は `validate_phys_cal` (`crates/mvmc-core/src/validation.rs:244`)に由来し、そこではキーワードチェックが繰り返されません。この非対称性はリリースビルドで観測されたもので、[11.5](11-compatibility.md#115-未解決の観測事項)に挙げています。

### ModPara の値(`validate_supported_modpara`, `crates/mvmc-core/src/validation.rs:77-105`。両方のエントリポイント)

| 条件 | 結果 |
|-----------|--------|
| `NMPTrans == 0` | 拒否(「並進射影なし」には 1 を使います。C の規約) |
| `NSplitSize < 1` | 拒否 |
| `NLanczosMode` が 0..2 にない | 拒否 |
| `NSRCG >= 2` | 拒否("not supported by Julia-mVMC") |
| `useDiagScale != 0` | 拒否 |
| `RescaleSmat != 0` | 拒否 |

### パラメータ最適化(`validate_para_opt`, `crates/mvmc-core/src/validation.rs:108-241`)

| 条件 | 結果 |
|-----------|--------|
| `NLanczosMode > 0` | 拒否("use PhysCal") |
| `NVMCCalMode != 0` | 最適化のエントリポイントでは拒否("cannot run parameter optimization; use fixed-parameter PhysCal")。PhysCal は `--physcal` と `NVMCCalMode 1` で選択します([8.1](08-running.md#計算の選択)) |
| `InterAll` のサイトが $0\ldots N_s-1$ の範囲外、スピンが {0,1} にない、または(通常軌道または固定 `2Sz`)でスピンを変えるペア($\sigma_1\ne\sigma_2$ または $\sigma_3\ne\sigma_4$)がある | 拒否 |
| `NSite`, `NElec`, `NVMCWarmUp`, `NSROptItrStep` が負。`NVMCSample` または `NVMCInterval` が $\le0$ | 拒否 |
| 不完全な Expert 入力(`input_errors` が空でない) | 拒否(PhysCal でも) |

### 物理量(`validate_phys_cal`, `crates/mvmc-core/src/validation.rs:244-291`)

| 条件 | 結果 |
|-----------|--------|
| 一般(FSZ)軌道での `NLanczosMode > 0` | 拒否 |
| スピンを変える `Trans` 行がある場合の `NLanczosMode > 0` | 拒否 |
| 任意の `InterAll` 項がある場合の `NLanczosMode > 0` | 拒否 |
| `TwoBodyGEx` がなく `OneBodyG` のエントリが重複している場合の `NLanczosMode = 2` | 拒否 |

### グループ実行(`validate_grouped_runtime`, `crates/mvmc-core/src/validation.rs:23-44`。`NSplitSize > 1` のとき適用)

| 条件 | 結果 |
|-----------|--------|
| `NSRCG != 0`(CG)での最適化 | 拒否: C で未定義(`vmccal.c:241,248` は保存した `O` をグローバルなサンプル位置に書くが、`vmccal.c:314-318` は先頭列からローカル数だけ読む) |
| 一般(FSZ)軌道での PhysCal/最適化(任意の `NQPFull`) | 受理(C が定義。#349 でネイティブ C の 2/4 ランクと照合) |
| `NLanczosMode > 0` の PhysCal | 受理(C が定義) |
| `NQPOptTrans > 1` / `OptTrans` | 受理(C が定義) |
| `NSplitSize > 1` だがリデューサがグループ通信子でない(`validate_reducer_rank`, `crates/mvmc-core/src/validation.rs:47-66`) | 拒否("requires an MPI group communicator") |

### コマンドラインレベル(`crates/mvmc-core/src/run.rs`, `crates/mvmc-cli/src/main.rs`)

| 条件 | 結果 |
|-----------|--------|
| `NVMCCalMode = 0` での `--physcal`、`--physcal` なしでの `NVMCCalMode = 1`、または 0/1 以外の `NVMCCalMode` | パース直後、初期化や出力より前に `select_calculation` (`crates/mvmc-cli/src/main.rs:429`)で拒否 |
| `--nsteps <= 0` で PhysCal でない(`NSROptItrStep = 0` も含む) | 拒否("nothing to run") |
| ウィンドウ `nsmp > nsteps` | 拒否(`validate_optimization_window`, `run.rs:1306`) |
| シリアルの `--physcal` 実行でない `--physcal-trace` | 拒否 |

### C プログラムの動作のうち Rust が提供しないもの

バイナリ出力(`-b`, `NFileFlushInterval`/`-F`)、複数定義モード(`-m`)、Standard モード(`-s`)、`--version` (`-v`)、
進捗ファイル `zvo_time_NNN.dat`、バックフロー(`BF`)、および `InterAll` の Lanczos は利用できません([11.3](11-compatibility.md#113-cリファレンスとの既知の相違)を参照)。

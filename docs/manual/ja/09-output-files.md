# 9. 出力ファイル

[目次](README.md) · 前へ: [8. 実行](08-running.md) · 次へ: [10. チュートリアル: 16サイトのHubbard鎖](10-tutorial.md)

すべてのファイルは、**world rank 0 のみ**が出力ディレクトリ(`--out-dir`)に書き出します
([8.4](08-running.md#84-mpiとグループ実行))。以下のファイル名で、`zvo` は `modpara.def` の `CDataFileHead`、
`zqp` は `CParaFileHead` です(デフォルトは `zvo`、`zqp`)。`NNN` は
`NDataIdxStart + k` を `%03d` で出力したものです(負のインデックスは `-01` のように出力されます)。

数値は C の書式 `"% .18e"` で書き出されます(`+` の代わりに先頭スペース、18桁、
指数部は2桁以上。`format_c_double`、`crates/mvmc-core/src/io.rs:50`)。各書式の C 側のリファレンスは
[`extern/mVMC-1.3.0/doc/en/source/output.rst`](../../../extern/mVMC-1.3.0/doc/en/source/output.rst) と `initfile.c`、`vmcmain.c:outputData`、`avevar.c` です。

出力されるファイルの一覧です("opt" = パラメータ最適化、"phys" = PhysCal)。

| ファイル | opt | phys | 条件 |
|------|-----|------|-----------|
| `zvo_out.dat` | yes | – | 常に |
| `zvo_var.dat` | yes | – | 常に |
| `zqp_opt.dat` | yes | – | 常に(最後に) |
| `zqp_<block>_opt.dat` | yes | – | `NSROptItrSmp > 1` かつブロックが空でない |
| `zvo_SRinfo.dat` | CG のみ | – | `NSRCG != 0` |
| `zvo_out_NNN.dat`, `zvo_var_NNN.dat` | – | yes | 常に |
| `zvo_cisajs_NNN.dat` | – | yes | `OneBodyG` にエントリがある |
| `zvo_cisajscktalt_NNN.dat` | – | yes | `TwoBodyG` にエントリがある |
| `zvo_cisajscktaltex_NNN.dat` | – | yes | `TwoBodyGEx` にエントリがある |
| `zvo_ls_out_NNN.dat`, `zvo_ls_qqqq_NNN.dat` | – | yes | `NLanczosMode > 0` |
| `zvo_ls_cisajs_NNN.dat`, `zvo_ls_cisajscktalt_NNN.dat`, `zvo_ls_cisajscktaltex_NNN.dat` | – | yes | `NLanczosMode = 2` |
| `zvo_CalcTimer.dat`, `zvo_CalcTimerDiag.dat` | yes | yes | タイマー用環境変数 ([8.5](08-running.md#85-環境変数)) |

## 9.1 ステップごとのエネルギー: `zvo_out.dat` (opt) と `zvo_out_NNN.dat` (phys)

SR ステップごとに1行(opt。ファイルはステップ0で切り詰められ、以降は追記されます)、または1行のみ(phys)です。6列からなります。

| # | 物理量 |
|---|----------|
| 1 | $\operatorname{Re}\langle H\rangle$ |
| 2 | $\operatorname{Im}\langle H\rangle$ |
| 3 | $\operatorname{Re}\langle H^2\rangle_{\rm est}=\langle\lvert E_{\rm loc}\rvert^2\rangle$ |
| 4 | 相対分散 $\operatorname{Re}[(\langle H^2\rangle-\langle H\rangle^2)/\langle H\rangle^2]$ |
| 5 | $\langle S_z\rangle$ |
| 6 | $\langle S_z^2\rangle$ |

C の書式は `"% .18e % .18e  % .18e % .18e %.18e %.18e\n"` で、3列目の前にスペースが2つあり、5列目と6列目の前には先頭スペースがありません
(`vmcmain.c:640`)。Rust はこれをバイト単位で再現します。例(チュートリアル実行の最初の行):

```text
 1.719867648718613040e+01  0.000000000000000000e+00   3.255315646316159359e+02  1.005329525872698249e-01 0.000000000000000000e+00 0.000000000000000000e+00
```

> **実装**
> - C: `outputData` — `extern/mVMC-1.3.0/src/mVMC/vmcmain.c:640`
> - C: `InitFile` — `extern/mVMC-1.3.0/src/mVMC/initfile.c:33`
> - Rust: `output_data` — `crates/mvmc-core/src/io.rs:89`
> - Rust: `output_phys_data` — `crates/mvmc-core/src/io.rs:161`
> - 整合性: (**命名の違い**) 最適化では、C はデータインデックス付きでファイルを書き出します(`zvo_out_001.dat` と `zvo_var_001.dat`、`initfile.c:55,59`、`NDataIdxStart`)。Rust は**インデックスなしの `zvo_out.dat` と `zvo_var.dat`** を書き出します(Julia の規約、`io.rs:113,128`)。PhysCal のファイルは両者ともインデックス付きです。最適化出力の分散列は、$\lvert\langle H\rangle\rvert\le10^{-14}$ のとき `0.0` になります(`io.rs:103`)。

## 9.2 ステップごとのパラメータ: `zvo_var.dat` と `zvo_var_NNN.dat`

1ステップにつき1行です。行は $\operatorname{Re}\langle H\rangle,\ \operatorname{Im}\langle H\rangle,\ 0.0,\ \operatorname{Re}\langle H^2\rangle,\ \operatorname{Im}\langle H^2\rangle,\ 0.0$ で始まり、続いて
[3.1](03-theory-wavefunction.md#パラメータのレイアウト)のレイアウト(射影、RBM、Slater、OptTrans。未割り当てスロットと予約スロットも書き出されます)に従うパラメータベクトルの**すべての**エントリについて、3つ組
`Re Im 0.0` が並びます。したがって1行は
$6+3N_{\rm para}$ 個の数値からなります(パラメータ100個のチュートリアルモデルでは306個)。行 $t$ のパラメータは、ステップ $t$ のサンプルに*使用された*もの、すなわち SR 更新の前の値です。
行の書式は `"% .18e % .18e 0.0 ..."` です(`vmcmain.c:655-657`)。

## 9.3 最適化されたパラメータ

**`zqp_opt.dat`** は最適化の最後に1度だけ書き出され(`OutputOptData`)、PhysCal の入力となります。3つ組が並ぶ1行です。$\texttt{NSROptItrSmp}>1$ の場合、$\langle H\rangle$、$\langle H^2\rangle$ および各パラメータ(パラメータベクトルの順)について、
ウィンドウ平均(実部、虚部)とウィンドウ標準偏差(実部)が並びます。`NSROptItrSmp = 1` の場合は(値、`0.0`)のペアが並び、ブロックごとのファイルは書き出されません。

**ブロックごとのファイル** `zqp_gutzwiller_opt.dat`、`zqp_jastrow_opt.dat`、`zqp_doublonHolon2site_opt.dat`、`zqp_doublonHolon4site_opt.dat`、
`zqp_{charge,spin,general}RBM_{physlayer,hiddenlayer,physhidden}_opt.dat`、`zqp_orbital_opt.dat`(通常の軌道)、
`zqp_orbitalAntiParallel_opt.dat` + `zqp_orbitalParallel_opt.dat`、または `zqp_orbital_general_opt.dat`(FSZ)、そして `zqp_trans_opt.dat`(OptTrans)は、次のレイアウトを持ちます。

```text
======================
NGutzwillerIdx  4
======================
======================
======================
0 -1.556663106607272473e+00  0.000000000000000000e+00 
1 -1.337680080798275606e+00  0.000000000000000000e+00 
...
```

すなわち、5行のヘッダー(2行目にブロック名と宣言された個数が入ります)と、`index mean_re mean_im` の行からなります。これは
`In*` 初期値ファイル([7.4](07-input-files.md#74-初期パラメータ値))とまったく同じ書式なので、ブロックファイルを初期値ファイルとして再利用できます。DH ブロックのヘッダーに宣言された個数
はパターン数であり、行数はその個数の $6\times$(DH2)または $10\times$(DH4)です。

> **実装**
> - C: `StoreOptData` — `extern/mVMC-1.3.0/src/mVMC/avevar.c:82`
> - C: `OutputOptData` — `extern/mVMC-1.3.0/src/mVMC/avevar.c:94`
> - Rust: `store_opt_data` — `crates/mvmc-core/src/io.rs:21`
> - Rust: `output_opt_data` — `crates/mvmc-core/src/io.rs:441`
> - Rust: `output_parameter_block` — `crates/mvmc-core/src/io.rs:580`
> - 整合性: ブロックの順序、ファイル名(`RBM_OUTPUT_BLOCKS`)、ヘッダーのテキスト、および `NSROptItrSmp = 1` に対する「ペアで補助ファイルなし」の規則は、`OutputOptData` をそのまま踏襲しています。ウィンドウ統計量は両者とも $\sqrt{\sum|x-\bar x|^2/(n-1)}$ です。

## 9.4 ソルバー情報: `zvo_SRinfo.dat`

**CG** ソルバー(`NSRCG != 0`)のみが書き出します。1行のヘッダーに続いて、SR ステップごとに1行です。

```text
#Npara Msize optCut diagCut sDiagMax  sDiagMin    absRmax       imax
  100   100     0     0  1.22546e+00  1.14682e-02 -6.31767e-02    76, 100
```

列は次のとおりです。パラメータ数 `Npara`(複素数としての個数)、解く系のサイズ `Msize`($n_S$)、最適化フラグにより固定されるパラメータ数 `optCut`、
`DSROptRedCut` により除去されるパラメータ数 `diagCut`、$S$ の対角要素の最大値と最小値、絶対値が最大の解成分 `absRmax`、その
パラメータインデックス `imax`、そしてカンマの後に CG の反復回数です。書式は `"%5d %5d %5d %5d % .5e % .5e % .5e %5d, %d"` です。
C のマニュアルによれば、実数モードのパラメータの虚部成分は `optCut` に数えられます。

> **実装**
> - C: `fn_StochasticOptCG` (writes the CG row) — `extern/mVMC-1.3.0/src/mVMC/stcopt_cg_impl.c:73`
> - C: `StochasticOpt` (writes the direct-solver row) — `extern/mVMC-1.3.0/src/mVMC/stcopt.c:33`
> - C: `InitFile` (header, opened for every optimization) — `extern/mVMC-1.3.0/src/mVMC/initfile.c:33`
> - Rust: `stochastic_opt_cg_with_reducer` — `crates/mvmc-core/src/sr_cg.rs:101`
> - 整合性: (**相違**) C は*直接法*ソルバーでも `zvo_SRinfo.dat` を書き出します(`stcopt.c:157`、反復回数の列なし)。Rust は CG の経路でのみ書き出します(`crates/mvmc-core/src/sr_cg.rs:208-241`、接頭辞は `CDataFileHead`)。直接法ソルバーの実行では `zvo_SRinfo.dat` は生成されません **(観測)**。Rust では、ヘッダー行はファイルが新規または空の場合にのみ書き出されます。

## 9.5 タイマー

`zvo_CalcTimer.dat` は、計測対象のセクションごとに1行からなります。C のタイマー ID を括弧で含むラベルと、累積秒数
(`"%12.5f"`)です。例:

```text
All                         [0]      4.02645
Initialization              [1]      0.00102
  read options             [10]      0.00000
  ReadDefFile              [11]      0.00094
  SetMemory                [12]      0.00000
  InitParameter            [13]      0.00005
VMCParaOpt                  [2]      4.02351
  VMCMakeSample             [3]      1.46436
    makeInitialSample      [30]      0.00858
    make candidate         [31]      0.03867
    hopping update         [32]      0.95590
      UpdateProjCnt        [60]      0.05897
```

(最適化実行では51行、[0] から SR のサブタイマーまで **(観測)**。PhysCal では `VMCPhysCal [2]` を使います)。`zvo_CalcTimerDiag.dat` は、さらに診断用の
スロット(ID は 966 まで。計測されないスロットは 0)を列挙します。Rust ではファイル名は**常に `zvo_...`** です。タイマーの書き出し関数はリテラルの接頭辞
`"zvo"` で呼び出されます(`run.rs`、`crates/mvmc-cli/src/main.rs`)。一方 C は `CDataFileHead` を使います(`vmcclock.c:79`)。タイマーは包含的です。親セクションは、子セクションが計測されている間も動き続けます。

> **実装**
> - C: `OutputTimerParaOpt` — `extern/mVMC-1.3.0/src/mVMC/vmcclock.c:79`
> - C: `OutputTimerPhysCal` — `extern/mVMC-1.3.0/src/mVMC/vmcclock.c:140`
> - Rust: `TimerEnv::from_lookup` — `crates/mvmc-core/src/c_timer.rs:174`

## 9.6 グリーン関数: `zvo_cisajs_NNN.dat`, `zvo_cisajscktalt_NNN.dat`, `zvo_cisajscktaltex_NNN.dat`

PhysCal のサンプルごとに1度書き出され、ファイルはインデックスごとに再作成されます(`"w"`)。

- **`zvo_cisajs_NNN.dat`** — `OneBodyG` のエントリごとに1行: `i s j s' Re Im`。C の書式は `"%d %d %d %d % .18e  % .18e \n"` で、最後に空行が1行続きます。
  値は $\langle c^\dagger_{is}c_{js'}\rangle$ です。
- **`zvo_cisajscktalt_NNN.dat`** — `TwoBodyG` のエントリごとに1行: `i s j s' k t l t' Re Im`(`"%d ... % .18e % .18e\n"`)で、最後に空行が続きます。
  値は $\langle c^\dagger_{is}c_{js'}c^\dagger_{kt}c_{lt'}\rangle$ です。
- **`zvo_cisajscktaltex_NNN.dat`** — 各 `TwoBodyGEx` エントリについて順に `Re  Im ` を並べた**1行**(因数分解された積、[6.1](06-theory-observables-lanczos.md#61-グリーン関数))で、改行1つで終わります。

例(チュートリアル実行、`NVMCSample = 3000`):

```text
0 0 0 0  4.156666666666666288e-01   0.000000000000000000e+00 
0 0 1 0  3.677613437662305418e-01   0.000000000000000000e+00 
0 0 0 0 0 0 0 0  4.156666666666666288e-01  0.000000000000000000e+00
0 0 0 0 0 1 0 1  1.019999999999999934e-01  0.000000000000000000e+00
```

個数がゼロの場合、そのファイルは作成されません(チュートリアルには `TwoBodyGEx` がないため、`zvo_cisajscktaltex_001.dat` もありません)。

> **実装**
> - C: `outputData` — `extern/mVMC-1.3.0/src/mVMC/vmcmain.c:640`
> - C: `InitFilePhysCal` — `extern/mVMC-1.3.0/src/mVMC/initfile.c:72`
> - Rust: `output_phys_data` — `crates/mvmc-core/src/io.rs:161`
> - 整合性: 行のテキスト、および `ex` ファイルの末尾の空行/1行レイアウトが再現されています(`io.rs:244-301`、コメント "C vmcmain.c:671-675 emits pairs in term order, newline after the loop")。

## 9.7 Lanczos ファイル

`NLanczosMode > 0` の場合([6.2](06-theory-observables-lanczos.md#62-1ステップ-lanczos-波動関数)):

- **`zvo_ls_out_NNN.dat`** — 3つの数値 $E_{\rm LS},\ \sigma^2_{\rm LS}/E^2_{\rm LS},\ \alpha$。それぞれ書式 `"% .18e  "` で、**末尾の改行なし**です(C と同じ)。例: `-8.893351117533208949e+00   6.611265253057749952e-03   3.885060397405710741e-01`。
- **`zvo_ls_qqqq_NNN.dat`** — `QQQQ` の16個の実部(それぞれ `"% .18e  "`)と改行。
- `NLanczosMode = 2` では、**`zvo_ls_cisajs_NNN.dat`**(行 `i s j s' Re Im` の後に空行。実数モードでは虚部を `0.0` として書き出します)、**`zvo_ls_cisajscktalt_NNN.dat`**(8個のインデックス、`Re Im`、その後に空行)、
  **`zvo_ls_cisajscktaltex_NNN.dat`**(`Re Im` のペアの1行)が追加されます。`ex` ファイルは `TwoBodyGEx` が空でも作成されます(その場合は改行1つだけを含みます)。

> **実装**
> - C: `PhysCalLanczos_fcmp` — `extern/mVMC-1.3.0/src/mVMC/physcal_lanczos.c:149`
> - Rust: `output_phys_data` — `crates/mvmc-core/src/io.rs:161`
> - Rust: `lanczos_energy` — `crates/mvmc-core/src/lanczos.rs:61`
> - 整合性: デバッグ専用の C ファイル `zvo_ls_qcisajsq_NNN.dat` と `zvo_ls_qcisajscktaltq_NNN.dat`(`#ifdef _DEBUG`)は生成されません。$\alpha$ の決定に失敗した場合、C は何も書き出しませんが、Rust は `NaN` を書き出します([6.2](06-theory-observables-lanczos.md#62-1ステップ-lanczos-波動関数))。

## 9.8 Rust が書き出さない C のファイル

| C のファイル | 内容 | Rust |
|--------|---------|------|
| `zvo_time_NNN.dat` | ステップごとのサンプリング進捗、採択率、タイムスタンプ(`OutputTime`) | 書き出さない |
| `zvo_varbin_NNN.dat` | `zvo_var` のバイナリ版(`-b`) | 書き出さない |
| 直接法ソルバーの `zvo_SRinfo.dat` | [9.4](#94-ソルバー情報-zvo_srinfodat) を参照 | 書き出さない |
| `zvo_ls_qcisajsq_*`, `zvo_ls_qcisajscktaltq_*` | `_DEBUG` ビルドのみ | 書き出さない |
| 最適化中の `zvo_out_NNN.dat`/`zvo_var_NNN.dat` | インデックス接尾辞 | `zvo_out.dat`/`zvo_var.dat` として書き出す |

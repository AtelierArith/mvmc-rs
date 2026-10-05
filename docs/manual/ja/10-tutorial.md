# 10. チュートリアル: 16サイトのHubbard鎖

[目次](README.md) · 前へ: [9. 出力ファイル](09-output-files.md) · 次へ: [11. 互換性と相違点](11-compatibility.md)

このチュートリアルでは、半充填の1次元Hubbardモデルの変分波動関数を最適化し、続いて最適化されたパラメータを用いて
グリーン関数とシングルステップLanczosエネルギーを測定します。すべてのコマンドは `mvmc` のリリースビルド
(`main` のコミット `3e9024ee`、Linux x86_64、`cargo build --release -p mvmc-cli`)で実行しました。以下の抜粋は実際の出力です
**(観測)**。実行時間はマシンとその負荷に依存します(ここでは2回の実行で最適化にそれぞれ13秒と31秒かかりました)。

## 10.1 モデルと入力

入力はリポジトリにコミットされているので、サブモジュールも C のツールも不要です:
`benchmark/hubbard_chain/inputs/hubbard_chain_L16/`(同じディレクトリの `StdFace.def` から C の StdFace ツールで生成。由来は
`benchmark/hubbard_chain/README.md` を参照)。モデルは次のとおりです。

$$
\mathcal H=-t\sum_{i\sigma}\big(c^\dagger_{i\sigma}c_{i+1\sigma}+{\rm h.c.}\big)+U\sum_in_{i\uparrow}n_{i\downarrow},\qquad t=1,\ U=4,\ N_s=16,\ N=16\ \text{(half filling)} .
$$

```bash
mkdir work && cd work
cp -r <repo>/benchmark/hubbard_chain/inputs/hubbard_chain_L16 hubbard_L16
cat hubbard_L16/namelist.def
```

```text
         ModPara  modpara.def
         LocSpin  locspn.def
           Trans  trans.def
    CoulombIntra  coulombintra.def
        OneBodyG  greenone.def
        TwoBodyG  greentwo.def
      Gutzwiller  gutzwilleridx.def
         Jastrow  jastrowidx.def
         Orbital  orbitalidx.def
        TransSym  qptransidx.def
```

各ファイルが定義する内容です([第7章](07-input-files.md)を参照)。

| ファイル | この例での内容 |
|------|-------------------------|
| `trans.def` | `NTransfer 64`: 16サイトのリング上の、両スピンに対する最近接ホッピング $t=1$ |
| `coulombintra.def` | `NCoulombIntra 16`、$U_i=4$ |
| `locspn.def` | `NlocalSpin 0`: すべての電子が遍歴 |
| `gutzwilleridx.def` | `NGutzwillerIdx 4`: 周期4で共有される Gutzwiller パラメータ(`Lsub = 4`) |
| `jastrowidx.def` | `NJastrowIdx 32`: 距離クラスごとの Jastrow パラメータ |
| `orbitalidx.def` | `NOrbitalIdx 64`: ペア軌道 $f_{ij}$(周期4の部分格子クラス)、実数 |
| `qptransidx.def` | `NQPTrans 4`: 並進パターン(最初の $\lvert N_{\rm MP}\rvert$ 個のみ使用) |
| `greenone.def` | `NCisAjs 32`: $\langle c^\dagger_{0\sigma}c_{j\sigma}\rangle$、$j=0..15$、両スピン |
| `greentwo.def` | `NCisAjsCktAltDC 96`: 2体相関関数の集合 |

また `modpara.def` の設定は次のとおりです(C のキー `Ncond` により $N_e=8$ となります。`2Sz 0`、`NSPGaussLeg 8`、`NSPStot 0` は8点のメッシュによる
一重項射影を与えます。`NMPTrans -1` は反周期境界条件の単一の並進セクターを選択します)。

```text
Nsite 16   Ncond 16   2Sz 0   NSPGaussLeg 8   NSPStot 0   NMPTrans -1
NSROptItrStep 300   NSROptItrSmp 30   DSROptRedCut 1e-8   DSROptStaDel 0.01   DSROptStepDt 0.003
NVMCWarmUp 10   NVMCInterval 1   NVMCSample 300   NExUpdatePath 0   RndSeed 1   NSplitSize 1   NStore 1   NSRCG 0
```

実数の変分パラメータは $N_{\rm para}=4+32+64=100$ 個あり(すべてのインデックスファイルが `ComplexType 0` なので、実数カーネルが使われます)、
$N_{\rm QP}=N_{\rm GL}\lvert N_{\rm MP}\rvertN_{\rm opt}=8$ 個の射影セクター、スピンごとに $N_e=8$ 個の電子があります。

## 10.2 パラメータを最適化する

```bash
mvmc hubbard_L16/namelist.def --out-dir out_opt
```

(`mvmc` = `<repo>/target/release/mvmc`)。コンソール出力:

```text
model    : Nsite=16 Nelec=8 NSROptItrStep=300
mode     : NVMCCalMode=0
sample   : NVMCSample=300 NVMCWarmUp=10

=== mvmc — Julia-mVMC Rust port ===
namelist : hubbard_L16/namelist.def
out-dir  : out_opt


=== Completed 300 SR steps in 30.67s ===
Output files written to: /…/out_opt
Final energy / site: -0.5418807043
Final-window means (30 steps): [-8.607008448725692, 0.0]
```

エネルギーのウィンドウ平均は $-8.6070$、すなわち1サイトあたり $-0.5379$ です。最後の1ステップ($-0.5419$)は平均ではなく、揺らぎます。
書き出されたファイル(`ls -l`)は、`zvo_out.dat`(46 500 バイト、300行)、`zvo_var.dat`(1.7 MB、306個の数値からなる300行)、
`zqp_opt.dat`、`zqp_gutzwiller_opt.dat`、`zqp_jastrow_opt.dat`、`zqp_orbital_opt.dat` です。直接法の SR ソルバーは `zvo_SRinfo.dat` を書き出しません
([9.4](09-output-files.md#94-ソルバー情報-zvo_srinfodat))。

### 収束: `zvo_out.dat`

列は $\operatorname{Re}\langle H\rangle$、$\operatorname{Im}\langle H\rangle$、$\langle\lvert E_{\rm loc}\rvert^2\rangle$、相対分散、$\langle S_z\rangle$、$\langle S_z^2\rangle$ です
([9.1](09-output-files.md#91-ステップごとのエネルギー-zvo_outdat-opt-と-zvo_out_nnndat-phys))。抜粋した行:

| ステップ (1始まり) | $\langle H\rangle$ | $\langle H\rangle/N_s$ | 相対分散 |
|----------------|--------------------|------------------------|-------------------|
| 1 | +17.1987 | +1.075 | 0.1005 |
| 100 | −4.5697 | −0.286 | 0.5347 |
| 200 | −8.3101 | −0.519 | 0.0253 |
| 300 | −8.6701 | −0.542 | 0.0261 |

最初の行はランダムな初期状態のエネルギー(1サイトあたり $+1.07$)です。SR によりこれが下がり、相対分散
(固有状態ではゼロ)は $10^{-1}$ から $2.6\times10^{-2}$ に減少します。ステップ100の行は最適化の途中にあたるため、収束していません。

### 最適化されたパラメータ: `zqp_opt.dat`

$\langle H\rangle$、$\langle H^2\rangle$ および100個のパラメータについての、3つ組(平均の実部、平均の虚部、最後の30ステップにわたる標準偏差)が並ぶ1行です。

```text
-8.607008448725691707e+00  0.000000000000000000e+00  1.152418507247792157e-01  7.590574786855339084e+01  0.000000000000000000e+00  1.892906971655880399e+00 -1.556663106607272473e+00 …
```

したがって、エネルギーのウィンドウ平均は $-8.6070\pm0.115$ です(30ステップのエネルギーの標準偏差)。`zqp_gutzwiller_opt.dat` には4つの Gutzwiller パラメータが入っています。

```text
======================
NGutzwillerIdx  4
======================
======================
======================
0 -1.556663106607272473e+00  0.000000000000000000e+00 
1 -1.337680080798275606e+00  0.000000000000000000e+00 
2 -1.734280992033364166e+00  0.000000000000000000e+00 
3 -1.676363487003552644e+00  0.000000000000000000e+00 
```

## 10.3 最適化されたパラメータでの物理量

PhysCal には `ModPara` ファイルで `NVMCCalMode 1` が必要です。入力のコピーを作ってモードを切り替え、より良い推定のためにサンプル数を増やします。

```bash
cp -r hubbard_L16 phys_in
sed -i 's/^NVMCCalMode    0/NVMCCalMode    1/; s/^NVMCSample     300/NVMCSample     3000/' phys_in/modpara.def
mvmc phys_in/namelist.def --physcal out_opt/zqp_opt.dat --out-dir out_phys
```

```text
model    : Nsite=16 Nelec=8 NSROptItrStep=300
mode     : NVMCCalMode=1
sample   : NVMCSample=3000 NVMCWarmUp=10

=== mvmc — Julia-mVMC Rust port ===
namelist : phys_in/namelist.def
out-dir  : out_phys
physcal  : out_opt/zqp_opt.dat


=== Completed 1 PhysCal samples in 1.41s ===
Output files written to: out_phys
```

1サンプル(`NDataQtySmp 1`、`NDataIdxStart 1`)で、ファイル `zvo_out_001.dat`、`zvo_var_001.dat`、`zvo_cisajs_001.dat`、`zvo_cisajscktalt_001.dat` が生成されます
(`TwoBodyGEx` は要求されていません)。[8.1](08-running.md#計算の選択)のディスパッチ規則は厳密に適用されます。`hubbard_L16`(モード0)に `--physcal` を付けて実行した場合や、`phys_in`(モード1)をそれなしで実行した場合は、直ちに停止します。

```text
error: --physcal requires NVMCCalMode=1 in ModPara (found NVMCCalMode=0); set NVMCCalMode=1 for fixed-parameter PhysCal
error: NVMCCalMode=1 selects fixed-parameter PhysCal; supply the fixed parameter file with --physcal <PATH>
```

`zvo_out_001.dat` には、3000サンプルで測定した固定パラメータのエネルギーが入っています。

```text
-8.604176648473060851e+00  0.000000000000000000e+00   7.579059132541173938e+01  2.375646954033361000e-02 0.000000000000000000e+00 0.000000000000000000e+00
```

また `zvo_cisajs_001.dat` の最初の数行($i\,\sigma\,j\,\sigma'$ に続いて $\langle c^\dagger_{i\sigma}c_{j\sigma'}\rangle$ の実部と虚部)は次のとおりです。

```text
0 0 0 0  4.156666666666666288e-01   0.000000000000000000e+00 
0 0 1 0  3.677613437662305418e-01   0.000000000000000000e+00 
0 0 2 0 -2.158749322082987102e-02   0.000000000000000000e+00 
0 0 3 0 -8.735104226616838274e-02   0.000000000000000000e+00 
```

最初の行は密度 $\langle n_{0\uparrow}\rangle$ で、続く行は $j=1,2,3$ に対する同じスピン間の1体相関です。`zvo_cisajscktalt_001.dat` では、`0 0 0 0 0 0 0 0` が
$\langle n_{0\uparrow}n_{0\uparrow}\rangle=\langle n_{0\uparrow}\rangle$、`0 0 0 0 0 1 0 1` が $\langle n_{0\uparrow}n_{0\downarrow}\rangle$(二重占有、0.102)です。

### 統計誤差: 複数のシードを使う

連続するサンプルは相関しているため、1本のチェーンの統計誤差は単純な二項分布による見積りよりもはるかに大きくなります。異なる
シード(`--seed N`。各実行は約1.4秒)で測定を繰り返すと、同じパラメータと3000サンプルで次のようになります **(観測)**。

| `--seed` | $\langle H\rangle$ | 相対分散 | $\langle n_{0\uparrow}\rangle$ |
|----------|--------------------|-------------------|-------------------------------|
| 1 (= `RndSeed`) | −8.6042 | 0.0238 | 0.4157 |
| 2 | −8.6305 | 0.0224 | 0.5247 |
| 3 | −8.6261 | 0.0239 | 0.4580 |
| 4 | −8.6483 | 0.0201 | 0.4690 |

エネルギーは $10^{-2}$ のレベルで再現されますが、局所密度はチェーン間で約 $\pm0.05$ ばらつきます(Gutzwiller パラメータと軌道パラメータは周期4で共有されているため、この状態の密度が一様である必要はありません)。4つの密度の平均 0.467 は、個々の値から想像されるよりも充填率 0.5 に近い値です。本番の計算では、`NVMCSample` を増やすか、`NDataQtySmp > 1`(番号付きの出力セットが複数)や複数の MPI ランク
([8.4](08-running.md#84-mpiとグループ実行))を使ってください。シード `1` の行は、`--seed 1` が `modpara.def` のシードと等しいため、上の実行と同一です。

## 10.4 シングルステップLanczosエネルギー

`NLanczosMode 1` を設定します(この入力には `InterAll` もスピンフリップホッピングもないため、Lanczos の経路はサポートされます、[7.5](07-input-files.md#75-サポートされる入力と拒否される入力))。

```bash
cp -r phys_in phys_ls
sed -i 's/^NLanczosMode   0/NLanczosMode   1/' phys_ls/modpara.def
mvmc phys_ls/namelist.def --physcal out_opt/zqp_opt.dat --out-dir out_ls
```

これにより `zvo_ls_out_001.dat` と `zvo_ls_qqqq_001.dat` が追加されました(2次モーメント $F(x,H^2)$ をホップした配置から評価するため、23秒かかりました、[6.2](06-theory-observables-lanczos.md#62-1ステップ-lanczos-波動関数))。

```text
$ cat out_ls/zvo_ls_out_001.dat
-8.893351117533208949e+00   6.611265253057749952e-03   3.885060397405710741e-01
```

$E_{\rm LS}=-8.8934$(1サイトあたり $-0.5558$)で、相対分散は $2.4\times10^{-2}$ から $6.6\times10^{-3}$ に下がり、最適な混合は $\alpha=0.3885$ です。`zvo_ls_qqqq_001.dat` の16個のエントリは
`1, -8.6042, -8.6042, 75.79, -8.6042, 75.76, 75.79, ...` で始まります。エントリ 2, 3, 10, 11, 15(0始まり)は $h_1,h_{2(11)},h_{2(20)},h_{3(12)},h_4$ です
([6.2](06-theory-observables-lanczos.md#62-1ステップ-lanczos-波動関数))。当然ながら、$h_{2(11)}=75.79059\ldots$ は `zvo_out_001.dat` の3列目
($\langle\lvert E_{\rm loc}\rvert^2\rangle$)に等しく、エントリ10の $h_{2(20)}=75.76$ は別途測定された $\langle F^\dagger(H^2)\rangle$ です。

## 10.5 バリエーション

- **CG ソルバー。** `modpara.def` に `NSRCG 1` を追加し、5ステップ実行します(`--nsteps 5 --nsmp 5`)。`zvo_SRinfo.dat` が現れ、ステップごとに `Npara Msize optCut diagCut sDiagMax sDiagMin absRmax imax, iterations` が書かれます。

  ```text
  #Npara Msize optCut diagCut sDiagMax  sDiagMin    absRmax       imax
    100   100     0     0  1.22546e+00  1.14682e-02 -6.31767e-02    76, 100
    100   100     0     0  1.40307e+00  1.05528e-02 -6.39915e-02    76, 100
  ```

  ここでは CG ソルバーが最大の100反復まで実行されており(`NSROptCGMaxIter` のデフォルトは $=n_S$)、`DSROptCGTol` には到達しませんでした。
- **タイマー。** `MVMC_C_TIMER=1 mvmc hubbard_L16/namelist.def --nsteps 30 --nsmp 5 --out-dir out_t` は、追加で `zvo_CalcTimer.dat` を書き出します
  ([9.5](09-output-files.md#95-タイマー))。このモデルでは、時間の大部分が `hopping update` と `UpdateMAll` にかかります。
- **拒否される入力。** `NSRCG 2` を追加すると、`mvmc` は `error: NSRCG >= 2 is not supported by Julia-mVMC; use NSRCG = 0 or 1` を出力し、終了ステータス1で停止します。
- **再現性。** 同じコマンドを `--seed 5` で2回実行すると、バイト単位で同一の `zvo_out.dat` が得られます。

サンプルプログラム(`cargo run -p mvmc-cli --example hubbard_chain`、`heisenberg_chain_real`、`heisenberg_chain_cmp`、`heisenberg_chain_fsz`)は、Julia リファレンスの入力
(`extern/Julia-mVMC/examples/inputs`。サブモジュールまたは `JULIA_MVMC_ROOT` 経由で検出)を数 SR ステップだけ実行します(`JULIA_MVMC_EXAMPLE_STEPS`、`MVMC_OUT_DIR`)。これらはこのマニュアルのためには実行していません **(未検証)**。

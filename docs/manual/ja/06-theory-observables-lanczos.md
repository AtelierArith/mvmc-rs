# 6. 理論V: 物理量とLanczos法による補正

[目次](README.md) · 前へ: [5. 理論IV: 確率的再構成法](05-theory-sr.md) · 次へ: [7. 入力ファイル](07-input-files.md)

固定パラメータモード(`NVMCCalMode = 1`、*PhysCal*)では、プログラムは最適化されたパラメータ
(通常は `zqp_opt.dat`。[8.2](08-running.md#82-固定パラメータでの物理量計算) を参照)を読み込み、
[第4章](04-theory-sampling.md) と同様に配置をサンプリングして、エネルギーと
`greenone.def`、`greentwo.def`、`greentwoex.def` で要求された相関関数を測定します。この
章では、推定量、重み付き平均、および1ステップの Lanczos
補正(`NLanczosMode = 1, 2`)を説明します。

## 6.1 グリーン関数

要求された各インデックス集合について、サンプル $x$ 上の*局所*値はグリーン関数の比
([2.5](02-theory-vmc-hamiltonian.md#25-グリーン関数の比)) です。

| Namelist キーワード | ファイル | 観測量 | 局所値 | 出力 |
|---------------------|----------|--------|--------|------|
| `OneBodyG` | `greenone.def` | $\langle c^\dagger_{i\sigma_1}c_{j\sigma_2}\rangle$ | `GreenFunc1` による $G^{(1)}$ | `zvo_cisajs_NNN.dat` |
| `TwoBodyG` | `greentwo.def` | $\langle c^\dagger_{i\sigma_1}c_{j\sigma_2}c^\dagger_{k\sigma_3}c_{l\sigma_4}\rangle$ | `GreenFunc2` による $G^{(2)}$ | `zvo_cisajscktalt_NNN.dat` |
| `TwoBodyGEx` | `greentwoex.def` | 2つの1体エントリの因子化された積 | $G^{(1)}_a\,\overline{G^{(1)}_b}$ | `zvo_cisajscktaltex_NNN.dat` |

測定値は、(統合された)すべてのチェーンの保存されたすべてのサンプルにわたる重み付き平均です。

$$
\langle A\rangle=\frac{1}{W}\sum_xw\,A_{\rm loc}(x),\qquad
\langle G_{\rm ex}\rangle=\frac{1}{W}\sum_xw\,G^{(1)}_a(x)\,\overline{G^{(1)}_b(x)}
$$

($w=1$)。因子化された(`TwoBodyGEx`)量は、積形式の推定量
$\langle F^\dagger(x,A)F(x,B)\rangle$ です(C マニュアル([*Power Lanczos method*](../../../extern/mVMC-1.3.0/doc/en/source/algorithm.rst)))。
これは*同じ*サンプルの**1体**局所値から構築されるため(各 `TwoBodyGEx` の行は1体リスト内の2つの位置を保持します。C では `CisAjsCktAltIdx[idx][0..1]`、Rust では `green_two_ex_indices`)、2電子
振幅を必要としません。すべての局所値はまずサンプルごとに保存され
(`LocalCisAjs`, `LocalCisAjsCktAltDC`)、その後で累積されます。

PhysCal は `NDataQtySmp` 回、サンプリング(最初の呼び出しには `NVMCWarmUp` のバーンインが含まれ、以降の呼び出しは直前の配置から続けます。[4.1](04-theory-sampling.md#41-サンプラーのループ))、測定、ランクにわたる平均を繰り返し、
番号付きの出力ファイル一式(`NDataIdxStart`, `NDataIdxStart+1`, ...)を1組書き出します。

> **実装**
> - C: `VMCPhysCal` — `extern/mVMC-1.3.0/src/mVMC/vmcmain.c:531`
> - C: `CalculateGreenFunc` — `extern/mVMC-1.3.0/src/mVMC/calgrn.c:32`
> - C: `WeightAverageGreenFunc` — `extern/mVMC-1.3.0/src/mVMC/average.c:209`
> - C: `GreenFunc1` — `extern/mVMC-1.3.0/src/mVMC/locgrn.c:41`
> - C: `GreenFunc2` — `extern/mVMC-1.3.0/src/mVMC/locgrn.c:86`
> - C: `CalculateGreenFunc_fsz` — `extern/mVMC-1.3.0/src/mVMC/calgrn_fsz.c:33`
> - Rust: `vmc_phys_cal_in_place_timed` — `crates/mvmc-core/src/run.rs:840`
> - Rust: `prepare_phys_cal_from_namelist` — `crates/mvmc-core/src/run.rs:517`
> - Rust: `ordinary_green_values` — `crates/mvmc-core/src/observables/green_measurements.rs:233`
> - Rust: `accumulate_two_body_gex_sample` — `crates/mvmc-core/src/observables.rs:109`
> - Rust: `normalize_physcal_green` — `crates/mvmc-core/src/run.rs:249`
> - Rust: `calculate_green_func_fsz` — `crates/mvmc-core/src/observables/fsz_measurements.rs:17`
> - Rust: `output_phys_data` — `crates/mvmc-core/src/io.rs:259`
> - 整合性: C は `PhysCisAjsCktAlt[idx] += w*LocalCisAjs[idx0]*conj(LocalCisAjs[idx1])` を累積します(`calgrn.c:110`)。Rust の `accumulate_two_body_gex_sample` は同じ順序で `weight * one_body[first] * one_body[second].conj()` を形成します。規格化では、事前に計算した $W$ の逆数を掛けます(`const double complex invW = 1.0/Wc`、`average.c:278`。ランク0での `SafeMpiReduce_fcmp` の後に適用、`average.c:287-290`)。Rust は C99 方式の `c_complex::divide(1, wc)` を使って各値に掛けます(`run.rs:249-278`)。要素ごとの除算は行いません。C では、平均されたグリーン関数はルートランクにのみ存在し(`weightAverageReduce` はランク0へ縮約します)、ファイルを書くのもルートのみです。サンプルごとの結果は*同じ*番号付きファイル一式に出力され、ファイルは `NDataIdxStart + sample` ごとに再作成(`"w"`)されます。

## 6.2 1ステップ Lanczos 波動関数

power-Lanczos 状態 $|\phi\rangle=(1+\alpha H)|\psi\rangle$ は、ハミルトニアンを1回作用させることで $|\psi\rangle$ を改善します。必要なモーメントはすべて、
*元の* $\rho(x)$ 上で局所演算子を使って評価されます。

$$
h_1=\langle F^\dagger(H)\rangle,\quad
h_{2(11)}=\langle F^\dagger(H)F(H)\rangle,\quad
h_{2(20)}=\langle F^\dagger(H^2)\rangle,\quad
h_{3(12)}=\langle F^\dagger(H)F(H^2)\rangle,\quad
h_4=\langle F^\dagger(H^2)F(H^2)\rangle .
$$

$\langle\phi|\phi\rangle=1+2\alpha h_1+\alpha^2h_{2(11)}$ なので、Lanczos エネルギーは

$$
E_{\rm LS}(\alpha)=\frac{h_1+\alpha\,(h_{2(11)}+h_{2(20)})+\alpha^2h_{3(12)}}{1+2\alpha h_1+\alpha^2h_{2(11)}}
$$

です([Cマニュアル, *Determination of $\alpha$*](../../../extern/mVMC-1.3.0/doc/en/source/algorithm.rst))。$dE_{\rm LS}/d\alpha=0$ とおくと2次
方程式が得られます(商の微分法則から直接導出)。

$$
-A\,\alpha^2+2B\,\alpha+C_0=0,\quad
A=h_{2(11)}(h_{2(11)}+h_{2(20)})-2h_1h_{3(12)},\quad
B=h_{3(12)}-h_1h_{2(11)},\quad
C_0=h_{2(11)}+h_{2(20)}-2h_1^2,
$$

$$
\alpha_\pm=\frac{B\pm\sqrt{\Delta}}{A},\qquad
\Delta=B^2+A\,C_0 .
$$

コードは $\Delta$ を展開形
$h_{2(11)}(h_{2(11)}+h_{2(20)})^2-h_1^2h_{2(11)}(h_{2(11)}+2h_{2(20)})+4h_1^3h_{3(12)}-2h_1(2h_{2(11)}+h_{2(20)})h_{3(12)}+h_{3(12)}^2$
で評価します($B^2+AC_0$ に等しい。この展開は本マニュアルの執筆時に手計算で確認しました)。エネルギーの低い方の根が使われ、

$$
\frac{\sigma^2_{\rm LS}}{E^2_{\rm LS}}=\frac{\dfrac{h_{2(11)}+2\alpha h_{3(12)}+\alpha^2h_4}{D}-E_{\rm LS}^2}{E_{\rm LS}^2},\qquad D=1+2\alpha h_1+\alpha^2h_{2(11)},
$$

が、$E_{\rm LS}$ および $\alpha$ とともに `zvo_ls_out_NNN.dat` に書き出される相対分散です。推定は、いずれかの根について
$\Delta<0$ または $|D/h_1|<10^{-12}$ の場合に失敗します(エラーが報告されます)。

**モーメントの測定方法。** $\nu=2$(`NLSHam`)の場合、コードはサンプルごとに局所値の $2\times2$ 配列

$$
\mathrm{LSLQ}=\begin{pmatrix}1 & F(x,H)\\ F(x,H) & F(x,H^2)\end{pmatrix}
\quad(\text{flat order }[1,\ E_{\rm loc},\ E_{\rm loc},\ F(x,H^2)]),
$$

を保存し、16成分のテンソル

$$
\mathrm{QQQQ}[r_q][r_p][r_i][r_j]\mathrel{+}=w\,\overline{\mathrm{LSLQ}[r_q][r_i]}\;\mathrm{LSLQ}[r_p][r_j]\qquad(\text{flat index }8r_q+4r_p+2r_i+r_j),
$$

を累積します。その(0始まりの)成分 $2,3,10,11,15$ が $h_1,\ h_{2(11)},\ h_{2(20)},\ h_{3(12)},\ h_4$(実部)です。局所的な2乗
$F(x,H^2)=\langle\psi|HH|x\rangle/\langle\psi|x\rangle$ は、$H$ の対角部分 $V$ と
1ステップのホッピングから構築されます(`LSLocalQ`)。

$$
F(x,H^2)=E_{\rm loc}(x)\,V(x)\;-\;\sum_{(ij\sigma)}t_{ij}\,\frac{\langle\psi|H\,c^\dagger_{i\sigma}c_{j\sigma}|x\rangle}{\langle\psi|x\rangle}
\;+\;\sum_{\rm 2\text{-}body}(\ldots),
$$

ここで各 $\langle\psi|Hc^\dagger_ic_j|x\rangle/\langle\psi|x\rangle=E_{\rm loc}(x')\,\overline{\psi(x')/\psi(x)}$ は、ホッピング後の配置 $x'$ の
ローカルエネルギーに共役な振幅比を掛けたものです(`calHCA1`)。$\psi(x')=0$ の場合は、ゼロの振幅で割ることがないよう、別の分岐
(`calHCA2`)を使います。交換、ペアホップ、`InterAll` の各項は
2体版の `calHCACA` を使います。

> **実装**
> - C: `LSLocalQ` — `extern/mVMC-1.3.0/src/mVMC/lslocgrn.c:76`
> - C: `calculateHK` — `extern/mVMC-1.3.0/src/mVMC/lslocgrn.c:122`
> - C: `calculateHW` — `extern/mVMC-1.3.0/src/mVMC/lslocgrn.c:139`
> - C: `calHCA` — `extern/mVMC-1.3.0/src/mVMC/lslocgrn.c:180`
> - C: `calHCACA` — `extern/mVMC-1.3.0/src/mVMC/lslocgrn.c:449`
> - C: `calculateQQQQ` — `extern/mVMC-1.3.0/src/mVMC/vmccal.c:832`
> - C: `CalculateEne` — `extern/mVMC-1.3.0/src/mVMC/physcal_lanczos.c:270`
> - C: `CalculateEneByAlpha` — `extern/mVMC-1.3.0/src/mVMC/physcal_lanczos.c:298`
> - C: `PhysCalLanczos_fcmp` — `extern/mVMC-1.3.0/src/mVMC/physcal_lanczos.c:149`
> - Rust: `accumulate_lanczos_qqqq` — `crates/mvmc-core/src/lanczos.rs:44`
> - Rust: `lanczos_energy` — `crates/mvmc-core/src/lanczos.rs:73`
> - Rust: `energy_by_alpha` — `crates/mvmc-core/src/lanczos.rs:128`
> - Rust: `calculate_lanczos_h2_transfer` — `crates/mvmc-core/src/observables.rs:2597`
> - 整合性: 判別式の式、`if (ene_p > ene_m) alpha = alpha_m` という選択、および許容条件 `fabs(dnorm/H1) < pow(10.0,-12)` は文字通りに再現されています(`lanczos.rs:61-114`、`lanczos.rs:117-143`)。`QQQQ` は、`calculateQQQQ` と `calculateQQQQ_real` の違いと同じく、複素の実行(`all_complex`)でのみ*左*因子に共役を使い、実数の実行では共役を使いません。Rust の演算子の適用は Julia の順序に従います(PairHop は down スピンのホップを先に適用し、Exchange は各スピンチャネルを up-down、down-up の順に適用します。テスト `lanczos_pair_hop_applies_down_then_up_like_julia`、`lanczos_exchange_applies_each_spin_channel_in_julia_order`)。**Rust の制限:** Lanczos は `InterAll` がなく、かつパスが FSZ でない場合にのみ累積されます(`run.rs:2779`)。また検証でスピンを変える `Trans`、`NSplitSize > 1`、一般軌道がさらに拒否されます(`crates/mvmc-core/src/validation.rs:255-291`)。これは C がサポートするもののサブセットです。**失敗時の挙動が異なります:** 2次方程式に許容される根がない場合、C はエラーを出力して `zvo_ls_*` ファイルには何も書きません。Rust は `zvo_ls_qqqq_NNN.dat` を書き出し、`zvo_ls_out_NNN.dat` には `NaN, NaN, NaN` を書き出します(`io.rs:311-316`)。

## 6.3 Lanczos ステップ後の物理量

エルミート演算子 $A$ に対する Lanczos 期待値は

$$
A_{\rm LS}(\alpha)=\frac{\langle\phi|A|\phi\rangle}{\langle\phi|\phi\rangle}
=\frac{A_0+\alpha\,(A_{1(10)}+A_{1(01)})+\alpha^2A_{2(11)}}{1+2\alpha h_1+\alpha^2h_{2(11)}},
$$

です。ここで $A_0=\langle F(A)\rangle$、$A_{1(10)}=\langle F^\dagger(H)F(A)\rangle$、$A_{1(01)}=\langle F(AH)\rangle$、
$A_{2(11)}=\langle F^\dagger(H)F(AH)\rangle$ です([Cマニュアル](../../../extern/mVMC-1.3.0/doc/en/source/algorithm.rst))。
$A=c^\dagger_ic_j$ および $A=c^\dagger_ic_jc^\dagger_kc_l$ について、局所値
$F(A)$ と $F(HA)=\langle\psi|HA|x\rangle/\langle\psi|x\rangle$ は `LSLCisAjs` と `LSLCisAjsCktAlt`
(`LSLocalCisAjs`、`calHCA`、`calHCACA`)です。モーメントテンソル `QCisAjsQ`、`QCisAjsCktAltQ`、
`QCisAjsCktAltQDC` は形状 $\nu\times\nu\times N_{\rm phys}$ を持ち、`QQQQ` とともに $W$ で規格化されます。
`CalculatePhysVal_fcmp` は、各量 $i$(`NPhys` 個)について次を組み立てます。

$$
A_{\rm LS,i}=\frac{Q[i]+\alpha\,(Q[N_{\rm phys}+i]+Q[\nu N_{\rm phys}+i])+\alpha^2\,Q[\nu N_{\rm phys}+N_{\rm phys}+i]}{\operatorname{Re}\big(1+2\alpha h_1+\alpha^2h_{2(11)}\big)} .
$$

`NLanczosMode = 1` はエネルギーファイルと `QQQQ` モーメントのみを書き出します。`NLanczosMode = 2` は
これに加えて、Lanczos の1体、直接2体、および因子化2体のグリーン関数を書き出します
(`zvo_ls_cisajs_NNN.dat`、`zvo_ls_cisajscktalt_NNN.dat`、`zvo_ls_cisajscktaltex_NNN.dat`。[第9章](09-output-files.md)を参照)。

> **実装**
> - C: `CalculatePhysVal_fcmp` — `extern/mVMC-1.3.0/src/mVMC/physcal_lanczos.c:336`
> - C: `CalculatePhysVal_real` — `extern/mVMC-1.3.0/src/mVMC/physcal_lanczos.c:313`
> - C: `LSLocalCisAjs` — `extern/mVMC-1.3.0/src/mVMC/lslocgrn.c:98`
> - C: `calculateQCAQ` — `extern/mVMC-1.3.0/src/mVMC/vmccal.c:852`
> - C: `calculateQCACAQ` — `extern/mVMC-1.3.0/src/mVMC/vmccal.c:871`
> - Rust: `lanczos_phys_values` — `crates/mvmc-core/src/io.rs:537`
> - Rust: `calculate_lanczos_green` — `crates/mvmc-core/src/observables.rs:2369`
> - Rust: `output_phys_data` — `crates/mvmc-core/src/io.rs:259`
> - 整合性: `lanczos_phys_values` は、`dnorm = (1 + 2*alpha*h1 + alpha*alpha*h2_1).re` として `(Q[i] + alpha*(Q[n+i] + Q[2n+i]) + alpha*alpha*Q[3n+i]) / dnorm` を評価します。これは C の `CalculatePhysVal_fcmp`(`physcal_lanczos.c:336-358`)と同じ式・同じ結合順序です。実モードでは虚部はリテラルの `0.0` として書き出されます(C には別の `_real` ライターがあります)。C マニュアルの記号 $A_{1(01)}$/$A_{1(10)}$ は2つの交差スロット `Q[n+i]` と `Q[2n+i]` に対応します。スロットとマニュアル記号の対応付けは `calculateQCAQ`(共役された左因子)に従っており、ここでは再導出していません **(未検証)**。

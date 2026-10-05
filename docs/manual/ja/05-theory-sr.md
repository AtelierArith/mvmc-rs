# 5. 理論IV: 確率的再構成法

[目次](README.md) · 前へ: [4. 理論III: マルコフ連鎖サンプリング](04-theory-sampling.md) · 次へ: [6. 理論V: 物理量とLanczos法による補正](06-theory-observables-lanczos.md)

パラメータ最適化(`NVMCCalMode = 0`)は、$E(\alpha)=\langle\psi_\alpha|H|\psi_\alpha\rangle/\langle\psi_\alpha|\psi_\alpha\rangle$ を
**確率的再構成法**(SR法)で最小化します。虚時間ステップ
$e^{-\Delta t H}|\psi_\alpha\rangle$ を変分多様体の接空間へ射影し、量子幾何計量 $S$ を持つ線形方程式系を得ます。
mVMC は [Tahara and Imada, J. Phys. Soc. Jpn. 77, 114701 (2008)] の定式化を用います。

## 5.1 対数微分 $O_k$

パラメータの各実成分 $x_a$($\operatorname{Re}\alpha_k$ と $\operatorname{Im}\alpha_k$)について、コードは
$O_a(x)=\partial\ln\psi(x)/\partial x_a$ を必要とします。振幅は $\alpha_k$ について正則なので、
$\partial_{\operatorname{Im}\alpha_k}\ln\psi=i\,\partial_{\operatorname{Re}\alpha_k}\ln\psi$ が成り立ち、各パラメータは組
$(O_k,\ iO_k)$ を与えます。SR ステップのベクトルは
[3.1](03-theory-wavefunction.md#パラメータのレイアウト) の配置になります。$O=[\,1\ (\text{energy slot}),\ \text{proj},\ \text{RBM},\ \text{Slater},\ \text{OptTrans}\,]$
で、各パラメータにつき2つの要素を持ちます。各ブロックは次のとおりです。

| ブロック | $O_k(x)$ |
|----------|----------|
| Gutzwiller, Jastrow, DH2, DH4 | 整数カウンタ $c_k(x)$(組 $(c_k,0)$。$\ln P$ には $\operatorname{Re}\alpha_k$ だけが入るため) |
| RBM, 可視バイアス $a_p$ | $m_p(x)$ |
| RBM, 隠れバイアス $b_h$ | $\tanh\theta_h(x)$ |
| RBM, 結合 $W_{ph}$ | $m_p(x)\tanh\theta_h(x)$ |
| Slater $f_k$ | $\displaystyle\frac{1}{\mathrm{IP}(x)}\sum_qw_q\,\mathrm{Pf}X_q\;\tfrac12\,\mathrm{Tr}\!\Big[X_q^{-1}\frac{\partial X_q}{\partial f_k}\Big]$ |
| OptTrans $p^{\rm opt}_o$ | $\displaystyle\frac{1}{\mathrm{IP}(x)}\sum_{K,s}p_K\tfrac12\sin\beta_s\omega_sP_S\,\mathrm{Pf}X_{o,K,s}$ |

Slater ブロックでは、コードは保存されている逆行列からトレースを解析的に評価します。
スピン軌道が軌道インデックス $k$ に対応する電子の各ペア $(m,n)$ は、
$\texttt{InvM}[m][n]$ にセクターのスピン回転因子
($cs$, $-cc$, $ss$, $-cs$。それぞれ up–up, up–down, down–up, down–down ブロックに対応し、
[3.4](03-theory-wavefunction.md#34-量子数射影) を参照)と並進の符号を掛けた項を加えます。
その後、セクターにわたる $\mathrm{Pf}X_q$ 重み付き・$w_q$ 重み付きの和を
$\mathrm{IP}$ で割ります。$\beta=0$ ではこれは $O_k=-\sum_{i\uparrow,j\downarrow}(X^{-1})_{ij}\,\sigma^{(k)}_{ij}$ に帰着します。

> **実装**
> - C: `VMCMainCal` (assembles `SROptO`) — `extern/mVMC-1.3.0/src/mVMC/vmccal.c:82`
> - C: `SlaterElmDiff_fcmp` — `extern/mVMC-1.3.0/src/mVMC/slater.c:100`
> - C: `SlaterElmDiff_fsz` — `extern/mVMC-1.3.0/src/mVMC/slater_fsz.c:119`
> - C: `RBMDiff` — `extern/mVMC-1.3.0/src/mVMC/rbm.c:323`
> - C: `calculateOptTransDiff` — `extern/mVMC-1.3.0/src/mVMC/vmccal.c:639`
> - Rust: `set_projection_diff` — `crates/mvmc-core/src/observables.rs:201`
> - Rust: `set_rbm_diff` — `crates/mvmc-core/src/sampling/rbm.rs:693`
> - Rust: `slater_elm_diff_with_scratch_timed` — `crates/mvmc-core/src/slater_derivative.rs:176`
> - Rust: `slater_elm_diff_fsz_with_scratch` — `crates/mvmc-core/src/slater_derivative.rs:473`
> - Rust: `opt_trans_diff` — `crates/mvmc-core/src/observables.rs:998`
> - 整合性: C ではセクターごとのバッファを `buf[orbidx] += invM_i[msj]*cs*tOrbSgn_i[msj]` などで累積し、`QPFullWeight` でセクターについて和をとり、最後に $1/\mathrm{IP}$ を掛けます(`slater.c:194-238`)。Rust の累積はこの順序を保ちます(セクター和にはテンソル縮約ヘルパー `qp_weighted_orbital_sum_einsum`、`observables.rs:537` を使用。テスト `qp_weighted_orbital_sum_einsum_matches_manual_complex_reference`)。また除算ではなく `julia_complex::reciprocal(ip)` を掛けます。結果は許容誤差付きで比較されます。OptTrans の配置は C と Rust で異なります。[3.4](03-theory-wavefunction.md#34-量子数射影) を参照してください。

## 5.2 SR方程式

$\langle\cdot\rangle$ を保存されたサンプル(重み1)にわたるモンテカルロ平均とします。
コードは SR の1ステップごとに次を累積します。

$$
\mathrm{OO}_{ab}=\big\langle O_b\,\overline{O_a}\big\rangle\ (a\ge\text{first parameter}),\qquad
\mathrm{OO}_{0b}=\langle O_b\rangle,\qquad
\mathrm{HO}_b=\langle E_{\rm loc}\,O_b\rangle,\ \ \mathrm{HO}_0=\langle E_{\rm loc}\rangle .
$$

計量と力は実部のみを使い、パラメータの実部と虚部を独立な実変数として扱います。

$$
\boxed{\;S_{ab}=\operatorname{Re}\mathrm{OO}_{ab}-\operatorname{Re}\mathrm{OO}_{0a}\,\operatorname{Re}\mathrm{OO}_{0b},\qquad
g_a=-2\,\Delta t\,\big(\operatorname{Re}\mathrm{HO}_a-\operatorname{Re}\mathrm{HO}_0\operatorname{Re}\mathrm{OO}_{0a}\big)\;}
$$

ここで $\Delta t=\texttt{DSROptStepDt}$ です。$g_a$ は、エネルギー勾配
$\partial E/\partial x_a=2\operatorname{Re}\big(\langle E_{\rm loc}O_a\rangle-\langle E_{\rm loc}\rangle\langle O_a\rangle\big)$ の $-\Delta t$ 倍です。
解く前に2つの安定化を行います。

1. **冗長な方向をカットします。** $S_{\max}=\max_aS_{aa}$ とします。$S_{aa}<S_{\max}\cdot\texttt{DSROptRedCut}$ を満たす成分は
   解から除かれ、更新されません(`zvo_SRinfo.dat` の $\texttt{diagCut}$)。最適化フラグが $\ne1$ の成分は固定されます
   ($\texttt{optCut}$)。残りの $n_S$ 個の成分は `smatToParaIdx` に列挙され、系を構成します。
2. **対角シフト。** $S_{aa}\leftarrow S_{aa}\,(1+\texttt{DSROptStaDel})$(乗法的シフトで、Tahara–Imada の $\varepsilon$ に相当します)。

パラメータの変化量は次を解いて得ます。

$$
\sum_b S'_{ab}\,\delta x_b=g_a,\qquad
x_a\leftarrow x_a+\delta x_a,
$$

ここで成分 $a=2k$ は $\operatorname{Re}\alpha_k$ を、$a=2k+1$ は $\operatorname{Im}\alpha_k$ を更新します
(複素モード。実モードでは $a=k$)。いずれかの $\delta x_a$ が有限でない場合、または分解・ソルバがエラーを報告した場合は、
パラメータは更新されず、実行はエラーで停止します。その後、パラメータは
同期されます([3.7](03-theory-wavefunction.md#37-初期値と同期))。

> **実装**
> - C: `StochasticOpt` (cut, diag shift via `stcOptInit`, solve, update) — `extern/mVMC-1.3.0/src/mVMC/stcopt.c:33`
> - C: `stcOptInit` (builds $S$ and $g$) — `extern/mVMC-1.3.0/src/mVMC/stcopt_dposv.c:53`
> - C: `stcOptMain` (LAPACK `dposv`) — `extern/mVMC-1.3.0/src/mVMC/stcopt_dposv.c:33`
> - Rust: `stochastic_opt_real_timed` — `crates/mvmc-core/src/sr.rs:78`
> - Rust: `stochastic_opt_complex_timed` — `crates/mvmc-core/src/sr.rs:172`
> - Rust: `build_s_g_real` — `crates/mvmc-core/src/sr.rs:394`
> - Rust: `build_s_g_complex` — `crates/mvmc-core/src/sr.rs:427`
> - Rust: `collect_active_real` — `crates/mvmc-core/src/sr.rs:357`
> - Rust: `component_is_optimized` — `crates/mvmc-core/src/sr.rs:25`
> - Rust: `cholesky_solve` — `crates/mvmc-core/src/sr.rs:856`
> - Rust: `update_parameter_value` — `crates/mvmc-core/src/sr.rs:760`
> - 整合性: `S[idx] = OO[(pi+2)*(2*size)+(pj+2)].re - OO[pi+2].re*OO[pj+2].re` および `S[ii] *= 1+DSROptStaDel`(`stcopt_dposv.c:69-74`)、`g[si] = -DSROptStepDt*2.0*(HO[pi+2].re - HO[0].re*OO[pi+2].re)`(`stcopt_dposv.c:81`)です。Rust の `build_s_g_complex` は同じ式を同じ順序で評価します(実モードではインデックスオフセットが2ではなく1)。実モードでは C は実行列を虚部ゼロの複素配置に埋め込むため、$S_{\max}$ にはゼロの虚部分散が含まれます。Rust の `collect_active_real` も同じ理由で `0.0` から畳み込みます。Rust は LAPACK の `dpotrf_`/`dpotrs_`(上三角のコレスキー分解。`sr.rs:45-52` で宣言)で解き、C は `dposv('U')` を使います。$S$ が正定値でない場合、Rust は更新を拒否し(`cholesky_solve` が `Err` を返す)、C は `dposv` の `info` を返します。C が直接ソルバー向けに書き出す $\texttt{zvo\_SRinfo.dat}$ の行(`stcopt.c:157`)は、Rust では書き出し**ません**([9](09-output-files.md))。

## 5.3 $\mathrm{OO}$ と $\mathrm{HO}$ の累積と `NStore` オプション

`NSRCG = 0` かつ `NStore = 0` の場合、各サンプルで全行列のランク1更新を行います。
`calculateOO`(複素、明示的ループ)または `calculateOO_real`(`DGER`)を使い、
$\mathrm{OO}\mathrel{+}=w\,O\,O^{\dagger}$、$\mathrm{HO}\mathrel{+}=wE_{\rm loc}O$ となります(サンプルあたりのコストは $O(N_p^2)$)。
`NStore = 1`(デフォルト)または `NSRCG = 1` の場合は、スケールされたサンプル $\sqrt wO_s$ を保存し
(`SROptO_Store`)、サンプリングループの後に行列–行列積で一度だけ行列を形成します
(`calculateOO_Store`, `calculateOO_Store_real`)。この方法は
$O(N_pN_{\rm smp})$ の追加メモリを要しますが、はるかに高速で、CG ソルバーでは必須です。MPI ランクにわたって、累積量は
合計され、総重み $W$ で割られます(`WeightAverageSROpt`)。

> **実装**
> - C: `calculateOO` — `extern/mVMC-1.3.0/src/mVMC/vmccal.c:769`
> - C: `calculateOO_real` — `extern/mVMC-1.3.0/src/mVMC/vmccal.c:796`
> - C: `calculateOO_Store` — `extern/mVMC-1.3.0/src/mVMC/vmccal.c:692`
> - C: `calculateOO_Store_real` — `extern/mVMC-1.3.0/src/mVMC/vmccal.c:656`
> - C: `WeightAverageSROpt` — `extern/mVMC-1.3.0/src/mVMC/average.c:78`
> - C: `WeightAverageSROpt_real` — `extern/mVMC-1.3.0/src/mVMC/average.c:115`
> - Rust: `calculate_oo` — `crates/mvmc-core/src/observables.rs:263`
> - Rust: `calculate_oo_real` — `crates/mvmc-core/src/observables.rs:219`
> - Rust: `calculate_oo_store` — `crates/mvmc-core/src/observables.rs:425`
> - Rust: `calculate_oo_store_real` — `crates/mvmc-core/src/observables.rs:322`
> - Rust: `finalize_oo_store` — `crates/mvmc-core/src/observables.rs:447`
> - Rust: `weight_average_sr_opt` — `crates/mvmc-core/src/average.rs:31`
> - Rust: `weight_average_sr_opt_real` — `crates/mvmc-core/src/average.rs:50`
> - Rust: `reduce_accumulators` — `crates/mvmc-core/src/run.rs:1750`
> - 整合性: C は保存するサンプルを `sqrt(w)` でスケールし(`SROptO_Store[...] = sqrtw*SROptO[...]`、`vmccal.c:241,248`)、グラム積はステップごとに一度だけ形成します。実数の保存行列については、Rust は Julia の SYRK ディスパッチと上三角コピーに従います。複素の保存積は逐次的なサンプル和を保ちます(`sr_store_gram_julia`、`observables.rs:497`。テスト `stored_direct_sr_gram_matches_sampled_julia_values`、`real_gram_matches_julia_generic_and_syrk_dispatch_boundary`)。これらの浮動小数点の総和順序は BLAS プロバイダーによって異なるため、許容誤差付きで比較されます。`vmcmain.c` が `WeightAverageSROpt` の一方の分岐を選ぶのと同様に、MPI で縮約されるのはアクティブな分岐(実または複素)だけです。

## 5.4 共役勾配法ソルバー(`NSRCG = 1`)

$S'$ を形成して分解する代わりに、CG ソルバーは保存されたサンプルを使ってベクトルにそれを作用させます。

$$
S'x=\frac1W\sum_s O^{(s)}\big(O^{(s)T}x\big)\;-\;\langle O\rangle\,\big(\langle O\rangle\!\cdot\!x\big)\;+\;\texttt{DSROptStaDel}\;\mathrm{diag}(S)\,x ,
$$

(複素モードではサンプル行列が実部と虚部に分かれ、`dgemv` が2組になります。ベクトルは $2N_{\rm para}$ 個の実要素を持ちます)。
標準的な CG 漸化式で $S'x=g$ を解きます。

$$
r_0=d_0=g,\ \ \delta_0=r_0\!\cdot\!r_0;\qquad
\alpha_i=\frac{\delta_i}{d_i\!\cdot\!S'd_i},\ \ x_{i+1}=x_i+\alpha_id_i,\ \ r_{i+1}=r_i-\alpha_iS'd_i,\ \
\beta_i=\frac{r_{i+1}\!\cdot\!r_{i+1}}{\delta_i},\ \ \delta_{i+1}=\beta_i\delta_i,\ \ d_{i+1}=r_{i+1}+\beta_id_i ,
$$

C を再現する上で重要な3つの規則があります。

- $\delta<\texttt{DSROptCGTol}^2\,n_S^2$ で停止します(反復の*開始時*に確認)。
- 最大 `NSROptCGMaxIter` 回反復します($\le 0$ の場合は `n_S`)。
- **20回目の反復ごと**に、残差を漸化式ではなく $r=g-S'x$ で厳密に再計算します。

カットと力は直接ソルバーと同一です([5.2](#52-sr方程式))。カットに使う対角要素は
$\mathrm{OO}_{aa}-\mathrm{OO}_{0a}^2$ です。CG の反復回数は `zvo_SRinfo.dat` の最終列に出力されます。
Rust では CG に対して `NSplitSize = 1` のみサポートされます(グループ CG は C で未定義)([7.5](07-input-files.md#75-サポートされる入力と拒否される入力))。

> **実装**
> - C: `StochasticOptCG` (wrapper selecting real/complex) — `extern/mVMC-1.3.0/src/mVMC/stcopt_cg.c:42`
> - C: `fn_StochasticOptCG` — `extern/mVMC-1.3.0/src/mVMC/stcopt_cg_impl.c:73`
> - C: `fn_StochasticOptCG_Main` — `extern/mVMC-1.3.0/src/mVMC/stcopt_cg_impl.c:258`
> - C: `fn_operate_by_S` — `extern/mVMC-1.3.0/src/mVMC/stcopt_cg_impl.c:356`
> - C: `fn_StochasticOptCG_Init` — `extern/mVMC-1.3.0/src/mVMC/stcopt_cg_impl.c:428`
> - Rust: `stochastic_opt_cg_with_reducer` — `crates/mvmc-core/src/sr_cg.rs:101`
> - Rust: `solve_with_reducer` — `crates/mvmc-core/src/sr_cg.rs:328`
> - Rust: `apply_with_reducer` — `crates/mvmc-core/src/sr_cg.rs:442`
> - Rust: `sequential_dot` — `crates/mvmc-core/src/sr_cg.rs:264`
> - Rust: `SampledSrOperator` — `crates/mvmc-core/src/sr_cg.rs:288`
> - 整合性: ここは演算順序が最も重要になる箇所です。C は `cg_thresh = DSROptCGTol*DSROptCGTol * (double)nSmat * (double)nSmat`(`stcopt_cg_impl.c:265`)を計算し、漸化式は `beta = xdot(r,r)/delta; delta = beta*delta;` です。古い $\delta$ は $r\!\cdot\!r$ で置き換えられ*ません*(`stcopt_cg_impl.c:333-336`)。Rust は両方を再現します(`crates/mvmc-core/src/sr_cg.rs:346`、`crates/mvmc-core/src/sr_cg.rs:387`: "C:336 rounds the quotient and then multiplies it by the old norm")。内積は BLAS の `ddot` ではなく逐次(`sequential_dot`)で、積は C と同じ `dgemv` の組を使います。このため CG は FMA と縮約順序に敏感であり、打ち切られた CG の結果はビット単位の整合ではなく許容誤差のゲートで検証します。
> - 整合性: 対角シフトは、行列を修正するのではなく `z += sdiag[si]*DSROptStaDel*x[si]`(`stcopt_cg_impl.c:420`)として現れます。$\langle O\rangle\cdot x$ には `xdot` を使います。サンプル積の MPI 縮約は、大域的な重み、平均、シフトの補正より前に行われます(`apply_with_reducer` のドキュメント)。

## 5.5 最適化ループと最終平均ウィンドウ

`VMCParaOpt` は $t=0\ldots\texttt{NSROptItrStep}-1$ について次を繰り返します。

1. `UpdateSlaterElm`(現在のパラメータから Slater テーブルを再構築)、`UpdateQPWeight`。
2. サンプリング(`VMCMakeSample`、[第4章](04-theory-sampling.md))と測定(`VMCMainCal`): $E_{\rm loc}$, $O$, $\mathrm{OO}$, $\mathrm{HO}$。
3. ランクにわたる重み付き平均(`WeightAverageWE`, `WeightAverageSROpt`)。
4. そのステップのエネルギーとパラメータを**書き出し**(`zvo_out`, `zvo_var`)ます。パラメータはこのステップの更新の*前*のものです。
5. SR 更新(直接法または CG)、続いて `SyncModifiedParameter`。
6. $t\ge\texttt{NSROptItrStep}-\texttt{NSROptItrSmp}$ なら、最終ウィンドウのために $(\langle H\rangle,\langle H^2\rangle,\text{all parameters})$ を保存します(`StoreOptData`)。

最後のステップの後、`OutputOptData` が `zqp_opt.dat` を書き出します。$\langle H\rangle$、$\langle H^2\rangle$ および各パラメータについて、ウィンドウの
平均(実部、虚部)と標本標準偏差 $\sqrt{\sum|x_s-\bar x|^2/(n-1)}$(1量あたり3つの数)を書き、続いてパラメータブロックごとに、ウィンドウ平均のみを含む1ファイルを書きます。
`NSROptItrSmp = 1` の場合は最後の値だけが(値, 0)の組として書かれ、ブロックごとのファイルは書かれません。`zqp_opt.dat` の最適化されたパラメータは、
後の PhysCal 実行の入力になります([第6章](06-theory-observables-lanczos.md)、[8.2](08-running.md#82-固定パラメータでの物理量計算))。

> **実装**
> - C: `VMCParaOpt` — `extern/mVMC-1.3.0/src/mVMC/vmcmain.c:331`
> - C: `StoreOptData` — `extern/mVMC-1.3.0/src/mVMC/avevar.c:82`
> - C: `OutputOptData` — `extern/mVMC-1.3.0/src/mVMC/avevar.c:94`
> - Rust: `vmc_para_opt_timed` — `crates/mvmc-core/src/run.rs:1113`
> - Rust: `vmc_para_opt` — `crates/mvmc-core/src/run.rs:1092`
> - Rust: `run_para_opt_from_namelist` — `crates/mvmc-core/src/run.rs:1444`
> - Rust: `store_opt_data` — `crates/mvmc-core/src/io.rs:21`
> - Rust: `output_opt_data` — `crates/mvmc-core/src/io.rs:561`
> - 整合性: ウィンドウは `step >= NSROptItrStep - NSROptItrSmp` です。Rust は `NSROptItrSmp > NSROptItrStep` を事前に拒否します("nsteps must be >= nsmp; C leaves oversized-window rows unwritten"、`validate_optimization_window`、`run.rs:1306`)。標準偏差の式は、どちらも `creal(data*conj(data))` の `sqrt(var/(n-1))` です。コマンドラインの `--nsteps`/`--nsmp` による上書きは、ウィンドウについて `NSROptItrStep`/`NSROptItrSmp` を変更します。

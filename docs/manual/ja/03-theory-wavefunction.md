# 3. 理論II: 変分波動関数

[目次](README.md) · 前へ: [2. 理論I: 変分モンテカルロ法とハミルトニアン](02-theory-vmc-hamiltonian.md) · 次へ: [4. 理論III: マルコフ連鎖サンプリング](04-theory-sampling.md)

## 3.1 試行状態

mVMC の変分状態は次のとおりです。

$$
|\psi\rangle=\mathcal N_{\rm RBM}\;\mathcal P_G\,\mathcal P_J\,\mathcal P^{(2)}_{d\text{-}h}\,\mathcal P^{(4)}_{d\text{-}h}\;
\mathcal L^{S}\mathcal L^{K}\mathcal L^{P}\;|\phi_{\rm pair}\rangle
$$

([Cマニュアル, *Expertモードの入力ファイル*](../../../extern/mVMC-1.3.0/doc/en/source/expert.rst), 75-110行目) 各因子は次のとおりです。

| 因子 | 定義 | 節 |
|--------|------------|---------|
| $\lvert\phi_{\rm pair}\rangle$ | パフィアンペア積 $\big[\sum_{ij\sigma\sigma'}f_{i\sigma j\sigma'}c^\dagger_{i\sigma}c^\dagger_{j\sigma'}\big]^{N/2}\lvert0\rangle$ | [3.2](#32-パフィアンペア積部分) |
| $\mathcal P_G$ | $\exp\big[\sum_i g_i\,n_{i\uparrow}n_{i\downarrow}\big]$ | [3.5](#35-gutzwiller因子jastrow因子ダブロン-ホロン相関因子) |
| $\mathcal P_J$ | $\exp\big[\tfrac12\sum_{i\ne j}v_{ij}(n_i-1)(n_j-1)\big]$ | [3.5](#35-gutzwiller因子jastrow因子ダブロン-ホロン相関因子) |
| $\mathcal P^{(2)}_{d\text{-}h}$, $\mathcal P^{(4)}_{d\text{-}h}$ | 2サイトおよび4サイトのダブロン-ホロン相関因子 | [3.5](#35-gutzwiller因子jastrow因子ダブロン-ホロン相関因子) |
| $\mathcal N_{\rm RBM}$ | 制限ボルツマンマシン因子 | [3.6](#36-制限ボルツマンマシン因子) |
| $\mathcal L^S,\mathcal L^K,\mathcal L^P$ | スピン射影、運動量射影、格子対称性射影 | [3.4](#34-量子数射影) |

実空間では、配置 $x$ に対してコードは次の**振幅**を評価します。

$$
\boxed{\;\psi(x)=\langle x|\psi\rangle
= \exp\Big[\sum_{k\in{\rm proj}}\alpha_k\,c_k(x)\Big]\;
\exp\big[\ln\mathcal N_{\rm RBM}(x)\big]\;
\underbrace{\sum_{q=1}^{N_{\rm QP}}w_q\,\mathrm{Pf}\,X_q(x)}_{\mathrm{IP}(x)}\;}
$$

ここで $c_k(x)$ は整数の*射影カウンタ* ($\texttt{eleProjCnt}$)、$X_q(x)$
は $q$ 番目の量子射影セクターにおける [3.2](#32-パフィアンペア積部分) の
歪対称行列、$w_q$ は [3.4](#34-量子数射影) のセクター重みです。メトロポリス判定に入るのは
$\ln|\psi|$ の差だけなので、コードは $\ln\mathrm{IP}$ (`logIp`) とカウンタを
別々に保持し、$\psi$ 自体は決して構成しません。

### パラメータのレイアウト

すべての変分パラメータは 1 本の複素ベクトル (C では `Para`、Rust では
`pack_parameters` で詰められたもの) に、次の順で格納されます。

$$
\texttt{Para}=\big[\underbrace{g_i}_{N_G}\;\underbrace{v_{ij}}_{N_J}\;\underbrace{\alpha^{(2)}}_{6N_{\rm DH2}}\;\underbrace{\alpha^{(4)}}_{10N_{\rm DH4}}\;\big|\;\text{RBM}\;\big|\;\underbrace{f}_{N_{\rm Slater}}\;\big|\;\underbrace{p^{\rm opt}}_{N_{\rm OptTrans}}\big].
$$

最初のグループは $N_{\rm proj}=N_G+N_J+6N_{\rm DH2}+10N_{\rm DH4}$ 個の要素
(`NProj`) を持ちます。C の読み込み側のオフセットは `ReadInputParameters` で確認できます
(DH2 は `NGutzwillerIdx + NJastrowIdx` から、DH4 は
`NGutzwillerIdx + NJastrowIdx + 2*3*NDoublonHolon2siteIdx` から始まります。`readdef.c`)。Rust のレイアウトは
`ProjectionLayout` (`projection_layout`,
`crates/mvmc-expert-parsers/src/types.rs:1311`) で、`dh2_offset`、
`dh4_offset`、`n_proj` を持ちます。各パラメータには 2 つの*最適化フラグ* (実部、
虚部) が連続して格納されます (`OptFlag[2k]`, `OptFlag[2k+1]`)。パラメータ成分は、
そのフラグが 1 のときに限り最適化されます。

SR ベクトル $O$ ([第5章](05-theory-sr.md)) は同じ順序を使い、パラメータごとに
2 スロット (実部、虚部の微分) を持ち、先頭に恒等成分用の組が追加されます:
$[1,\ \text{proj}\ (2N_{\rm proj}),\ \text{RBM}\ (2N_{\rm RBM}),\ \text{Slater}\ (2N_{\rm Slater}),\ \text{OptTrans}]$
(`VMCMainCal`, `vmccal.c:197-215`)。

> **実装**
> - C: `ReadInputParameters` — `extern/mVMC-1.3.0/src/mVMC/readdef.c:1183`
> - Rust: `projection_layout` — `crates/mvmc-expert-parsers/src/types.rs:1311`
> - Rust: `accumulate_observables_local` — `crates/mvmc-core/src/run.rs:2591`
> - 整合性: Rust は、定義ファイルの行数が宣言より少ない場合でも、各ブロックに*宣言された*幅を確保します (「スパース射影」規則、`ProjectionLayout` のドキュメント)。FSZ のメイン計算では Slater の微分が射影ブロックの直後に置かれます (RBM スロットなし)。通常のパスでは先にすべての RBM スロットを確保します (`run.rs:2887-2900`)。

## 3.2 パフィアンペア積部分

1体部分は一般化された (BCS 型の) ペア積です。

$$
|\phi_{\rm pair}\rangle=\Big(\sum_{I,J=1}^{2N_s}F_{IJ}\,c^\dagger_Ic^\dagger_J\Big)^{N/2}|0\rangle ,
\qquad F_{IJ}=-F_{JI},
$$

ここで $I=(i,\sigma)$ はスピン軌道です。通常のスピン一重項ペアリングの場合は
$f_{ij}\equiv F_{(i\uparrow)(j\downarrow)}$ だけが非零で、
$|\phi_{\rm pair}\rangle=(\sum_{ij}f_{ij}c^\dagger_{i\uparrow}c^\dagger_{j\downarrow})^{N_e}|0\rangle$ となります。
単一のスレーター行列式は、$F$ の $N_e$ 個の非零特異値がすべて 1 に等しい特別な場合です ([Cマニュアル, *パフィアン-スレーター行列式の性質*](../../../extern/mVMC-1.3.0/doc/en/source/algorithm.rst)):

$$
f_{ij}=\sum_{n=1}^{N_e}\Phi_{in\uparrow}\Phi_{jn\downarrow},\qquad
F_{IJ}=\sum_{n=1}^{N/2}\big(\Phi_{I,2n-1}\Phi_{J,2n}-\Phi_{J,2n-1}\Phi_{I,2n}\big).
$$

配置との重なりはパフィアンになります。$x$ の $N=2N_e$ 個の電子に
$m=1\ldots N$ と番号を付け (通常のパスではアップスピン電子が先)、
$r_m$ を電子 $m$ のスピン軌道とします。このとき

$$
\langle x|\phi_{\rm pair}\rangle=\mathrm{Pf}\,X(x),\qquad X_{mn}(x)=F_{r_mr_n},
$$

であり、これは $N\times N$ の歪対称行列です。コードは SR の 1 ステップごとに 1 回、
各射影セクター $q$ ([3.4](#34-量子数射影)) について、サイズ $2N_s\times2N_s$ の*スレーター要素テーブル*
$\texttt{SlaterElm}[q][I][J]$ (パラメータ $f_{ij}$ から導かれる、並進とスピン回転を施した
$F^{(q)}_{IJ}$) を構築し、パフィアンを再計算するたびに、占有されたスピン軌道の行と
列を $X_q$ に集めます。C は集めたブロックを
$\texttt{invM}[m\cdot N+n]=-\texttt{sltE}[r_m][r_n]$ として格納します。テーブルが歪対称である
($X_{nm}=F_{r_nr_m}=-F_{r_mr_n}$) ため、このバッファを列優先行列として読むとちょうど $X$ になり、
それが分解ルーチンに渡されます。

パフィアンと逆行列は、歪対称 LTL 分解から得られます
(C では `M_ZSKTRF` + `utu2pfa_z` + `utu2inv_z`、`crates/pfapack` では `zsktf2`、
`utu2pfa_complex`、`utu2inv_complex`)。最後の
`M_ZSCAL(..., -1)` の後、格納された配列は行優先行列として
$\texttt{InvM}_{mn}=(X^{-1})_{mn}$ を満たします。これはすべての更新式
([4.5](04-theory-sampling.md#45-パフィアン比と逆行列の更新)) で使われる量です。

> **実装**
> - C: `CalculateMAll_fcmp` — `extern/mVMC-1.3.0/src/mVMC/matrix.c:285`
> - C: `calculateMAll_child_fcmp` — `extern/mVMC-1.3.0/src/mVMC/matrix.c:332`
> - C: `CalculateMAll_real` — `extern/mVMC-1.3.0/src/mVMC/matrix.c:516`
> - C: `calculateMAll_child_real` — `extern/mVMC-1.3.0/src/mVMC/matrix.c:557`
> - C: `CalculateMAll_fsz` — `extern/mVMC-1.3.0/src/mVMC/matrix.c:78`
> - Rust: `calc_m_all_complex` — `crates/mvmc-core/src/pfaffian.rs:281`
> - Rust: `calc_m_all_real` — `crates/mvmc-core/src/pfaffian.rs:134`
> - Rust: `calc_m_all_fsz_complex` — `crates/mvmc-core/src/pfaffian.rs:509`
> - Rust: `calc_m_all_fsz_real` — `crates/mvmc-core/src/pfaffian.rs:623`
> - Rust: `calc_m_all_child_complex` — `crates/mvmc-core/src/pfaffian.rs:721`
> - Rust: `zsktf2_c_compat` — `crates/pfapack/src/ltl.rs:54`
> - Rust: `utu2pfa_complex` — `crates/pfapack/src/utu2.rs:57`
> - Rust: `utu2inv_complex` — `crates/pfapack/src/utu2.rs:339`
> - 整合性: C は、分解がゼロピボットを報告した場合、またはパフィアンが有限でない場合にサンプルを失敗とします (`info != 0`、`matrix.c:371-373`)。Rust は `CalcMAllError::{ZeroPivot, NonFinitePfaffian, AllZero}` を返します。通常の複素数オプティマイザは Julia の `zsktf2_turbo` の演算順序に従い、`c_compat` 版 (`calc_m_all_complex_c_compat`, `crates/mvmc-core/src/pfaffian.rs:301`) は C の PFAPACK カーネルに従います。実数パスは BLAS の `dger`/`dtrtri`/`dtrmm` を使います。パフィアンと逆行列の結果はビット単位ではなく、明示的な許容誤差で比較します ([11.4](11-compatibility.md#114-数値比較ポリシー) を参照)。

### 軌道モード

非零の $F_{IJ}$ の集合とそのパラメータ化の方法は、どの軌道定義ファイルが存在するかに依存します。
パーサーはそれらを `i_flg_orbital_anti_parallel`、`i_flg_orbital_parallel`、
`i_flg_orbital_general` に記録します (`crates/mvmc-expert-parsers/src/orbital_mode.rs`)。

| モード | Namelist キーワード | 非零のペア振幅 | パフィアンのパス |
|------|-------------------|--------------------------|---------------|
| 通常 (sz 保存) | `Orbital` または `OrbitalAntiParallel` | $i$ 上の $\uparrow$ と $j$ 上の $\downarrow$ の間の $f_{ij}$ | `*_fcmp`/`*_real` カーネル、$N_\uparrow=N_\downarrow=N_e$ |
| 平行スピンペアリング | `OrbitalAntiParallel` **と** `OrbitalParallel` | 追加で $F_{i\sigma j\sigma}$ | *一般*として扱われる (ペア $AP=P=1$ は `General=1` を設定、`judge_orbital_mode`) |
| 一般 (FSZ) | `OrbitalGeneral` | $I,J\in\{1..2N_s\}$ のすべての $F_{I J}$ | スピンラベル `eleSpn` を明示的に持つ `*_fsz` カーネル |

C のリーダーは $2S_z\neq0$ のときは常に `OrbitalParallel`/`OrbitalGeneral` を要求し、
一般軌道では `NSPGaussLeg = 1` を強制します (スピン射影は通常のパスにだけ
実装されています)。一般軌道では Lanczos法 は拒否されます
(`readdef.c:597-617`)。C の `JudgeOrbitalMode` は Rust の
`judge_orbital_mode` に対応しており、C の許可結果を報告します (が、強制はしません)。

FSZ/一般軌道では、テーブルはスピン回転なしで構築されます。
$\texttt{sltE}[I][J]=F_{IJ}-F_{JI}$ (4 つのスピンブロックそれぞれ。
`UpdateSlaterElm_fsz`, `slater_fsz.c:32`)。並進の符号と
`OrbitalSgn` 行列は、通常のパスとまったく同じように適用されます。

## 3.3 実数モードと複素数モード

mVMC には実数用と複素数用のカーネルが別々にあります (`*_real`, `*_fcmp`)。モードは
数値から推定**されません**。C は次のように設定します。

$$
\texttt{AllComplexFlag}=\texttt{iComplexFlgGutzwiller}+\texttt{iComplexFlgJastrow}+\texttt{iComplexFlgDH2}+\texttt{iComplexFlgDH4}+\texttt{iComplexFlgOrbital},
$$

ここで各フラグは対応するインデックスファイルの `ComplexType` ヘッダ値です
(`readdef.c:641-642`)。すべて 0 なら実数カーネルが選択されます。RBM の
ファイルはこの和に寄与しません。Rust は
`get_all_complex_flag` (`crates/mvmc-core/src/run.rs:1713`) と
`all_complex_flag` (`crates/mvmc-expert-parsers/src/utils/parameter_init.rs:23`) でこれを再現しており、
どちらも同じヘッダ宣言を受け取ります。`In*` ファイルから読み込んだ虚数値はモードを
変更せず、格納されたヘッダを消去せずに変更された読み込み宣言はエラーになります。

実数モードでは各パラメータの実部だけが最適化され、SR
行列は $N_{\rm para}\times N_{\rm para}$ の実行列になります。複素数モードでは
実部と虚部が $2N_{\rm para}$ 個の独立な実変数になります
([5.2](05-theory-sr.md#52-sr方程式))。

> **実装**
> - C: `ReadInputParameters` (`AllComplexFlag` を設定) — `extern/mVMC-1.3.0/src/mVMC/readdef.c:1183`
> - Rust: `get_all_complex_flag` — `crates/mvmc-core/src/run.rs:1713`
> - Rust: `all_complex_flag` — `crates/mvmc-expert-parsers/src/utils/parameter_init.rs:23`
> - 整合性: 実数モードの実行では、C は `SlaterElm_real`/`InvM_real`/`PfM_real` のコピーを保持します。Rust は実数バッファ (`pf_m_real`, `sr_opt_oo_real`, ...) を保持し、共有コードが必要とする箇所では複素数のシャドウも保持します。

## 3.4 量子数射影

射影演算子 $\mathcal L^S\mathcal L^K\mathcal L^P$ は、*セクター* $q=(o,K,s)$ にわたる
求積/和によって評価されます。

- $s=1\ldots N_{\rm GL}$、$N_{\rm GL}=\texttt{NSPGaussLeg}$: スピン射影 ($y$ 軸まわりの回転) のための Gauss–Legendre 節点
  $\beta_s\in[0,\pi]$ と重み $\omega_s$。
  $S=\texttt{NSPStot}$ が Legendre 多項式 $P_S$ を選びます。
- $K=1\ldots\lvert N_{\rm MP}\rvert$、$N_{\rm MP}=\texttt{NMPTrans}$: `TransSym`/`qptransidx.def` で与えられる格子並進 (または任意の
  サイト置換群) と重み $p_K$
  (複素位相が運動量を選びます)。`NMPTrans` が負の場合は
  反周期境界条件 (折り返されたサイトでの符号因子) が有効になります。$N_{\rm MP}=1$
  は「並進射影なし」を意味します。
- $o=1\ldots N_{\rm opt}$、$N_{\rm opt}=\texttt{NQPOptTrans}$: *最適化される*並進
  (`OptTrans`、`-o` フラグで有効化。その重み $p^{\rm opt}_o$ は
  変分パラメータです。通常は $N_{\rm opt}=1$)。

セクター添字は $q=o\,N_{\rm fix}+K\,N_{\rm GL}+s$ (0 始まり、$N_{\rm fix}=N_{\rm GL}\lvert N_{\rm MP}\rvert$)
で、$N_{\rm QP}=N_{\rm GL}\,\lvert N_{\rm MP}\rvert\,N_{\rm opt}$ となります。重みは

$$
w_q=p^{\rm opt}_o\;p_K\;\tfrac12\sin\beta_s\;\omega_s\;P_S(\cos\beta_s)
\qquad(N_{\rm GL}>1),\qquad
w_q=p^{\rm opt}_o\,p_K\quad(N_{\rm GL}=1,\ \beta=0).
$$

です。

セクター $q$ のスレーター要素テーブルは、並進と符号を施したペア振幅と
$\beta_s$ によるスピン回転を使います。
$f^{(q)}_{ij}=\mathrm{sgn}_i\,\mathrm{sgn}_j\,f_{\,T_q(i)\,T_q(j)}$ (サイト写像
$T_q$ と符号は `QPTrans`、`QPOptTrans`、`OrbitalSgn` から) と
$c=\cos\tfrac{\beta_s}2$、$s=\sin\tfrac{\beta_s}2$ を用いると、

$$
\begin{aligned}
F^{(q)}(i\uparrow,j\uparrow)&=-(f_{ij}-f_{ji})\,cs, &
F^{(q)}(i\uparrow,j\downarrow)&=f_{ij}c^2+f_{ji}s^2,\\
F^{(q)}(i\downarrow,j\uparrow)&=-f_{ij}s^2-f_{ji}c^2, &
F^{(q)}(i\downarrow,j\downarrow)&=(f_{ij}-f_{ji})\,cs .
\end{aligned}
$$

となります。$\beta=0$ ($N_{\rm GL}=1$) では $F(i\uparrow,j\downarrow)=f_{ij}$ と
$F(i\downarrow,j\uparrow)=-f_{ji}$ だけが残ります。すべての振幅と
すべてのメトロポリス比に入る射影された内積は

$$
\mathrm{IP}(x)=\sum_{q}w_q\,\mathrm{Pf}\,X_q(x),\qquad
\ln\mathrm{IP}(x)\ \text{is stored as}\ \texttt{logIp}.
$$

です。

> **実装**
> - C: `InitQPWeight` — `extern/mVMC-1.3.0/src/mVMC/qp.c:38`
> - C: `UpdateQPWeight` — `extern/mVMC-1.3.0/src/mVMC/qp.c:129`
> - C: `GaussLeg` — `extern/mVMC-1.3.0/src/mVMC/gauleg.c:32`
> - C: `LegendrePoly` — `extern/mVMC-1.3.0/src/mVMC/legendrepoly.c:32`
> - C: `UpdateSlaterElm_fcmp` — `extern/mVMC-1.3.0/src/mVMC/slater.c:37`
> - C: `UpdateSlaterElm_fsz` — `extern/mVMC-1.3.0/src/mVMC/slater_fsz.c:32`
> - C: `CalculateIP_fcmp` — `extern/mVMC-1.3.0/src/mVMC/qp.c:110`
> - C: `CalculateLogIP_fcmp` — `extern/mVMC-1.3.0/src/mVMC/qp.c:90`
> - Rust: `init_qp_weight` — `crates/mvmc-core/src/qp.rs:12`
> - Rust: `init_qp_weight_inplace` — `crates/mvmc-expert-parsers/src/utils/qp_weight.rs:86`
> - Rust: `update_qp_weight` — `crates/mvmc-expert-parsers/src/utils/qp_weight.rs:151`
> - Rust: `gauss_legendre` — `crates/mvmc-expert-parsers/src/utils/qp_weight.rs:24`
> - Rust: `legendre_poly` — `crates/mvmc-expert-parsers/src/utils/qp_weight.rs:65`
> - Rust: `update_slater_elm` — `crates/mvmc-core/src/slater_update.rs:17`
> - Rust: `update_slater_elm_fsz` — `crates/mvmc-core/src/slater_update.rs:150`
> - Rust: `calculate_ip_complex` — `crates/mvmc-core/src/observables.rs:147`
> - Rust: `calculate_log_ip_complex` — `crates/mvmc-core/src/observables.rs:178`
> - Rust: `translated_site` — `crates/mvmc-core/src/qp.rs:33`
> - 整合性: C は `w = 0.5*sin(beta[i])*weight[i]*LegendrePoly(cos(beta[i]), NSPStot)` と `QPFixWeight[i + j*NSPGaussLeg] = w*ParaQPTrans[j]` を評価し (`qp.c:73-77`)、`FlagOptTrans > 0` のときは `QPFullWeight = OptTrans[i]*QPFixWeight` とします (`qp.c:129-146`)。和 `ip += QPFullWeight[q]*pfM[q]` は `CalculateIP_fcmp` でセクター順に累積され、`CalculateLogIP_fcmp` は `clog(ip)` を返します。Rust も同じ順序に従います。Slater テーブルは、Julia の歴史的な $10^{-14}$ の振幅カットオフなしに、*宣言された*係数から再構築されます (`slater_update.rs:46`)。並進の符号 (反周期モード) は `NMPTrans < 0` のときにだけ適用されます (`qp.rs:67`)。
> - 整合性: `OptTrans` には 2 種類のフラグレイアウトがあります。C ドライバのフラグ `-o` は「連続」フラグ書き込みを選択します (`c_opt_trans_flags`、`crates/mvmc-core/src/sr.rs:23` を参照)。C の `calculateOptTransDiff` (`vmccal.c:639`) は微分を連続した複素数インデックスに書き込みます (C のコメントでは「this part will not be used」)。一方 Rust の `opt_trans_diff` (`observables.rs:957`) は Julia と同様に (value, i·value) の組を書き込みます。OptTrans は `tests/fixtures/opttrans` のフィクスチャでカバーされています。このマニュアルでは、これ以上の対応付けは行いません **(未検証)**。

## 3.5 Gutzwiller因子、Jastrow因子、ダブロン-ホロン相関因子

4 つの因子はすべて、*整数カウンタ*と (実数の) 係数の積の指数関数です。

$$
\ln P(x)=\sum_{k=1}^{N_{\rm proj}}\operatorname{Re}\alpha_k\;c_k(x).
$$

C は $\operatorname{Re}\,\texttt{Proj}[k]$ だけを取ります (`LogProjVal`, `LogProjRatio`: "we assume gutzwiller and jastrow is real")。カウンタは
`MakeProjCnt` で計算され、電子がホッピングするときに `UpdateProjCnt` で逐次更新されます。

- **Gutzwiller** (インデックス写像 $\texttt{GutzwillerIdx}[i]$): $c_{\rm idx(i)}\mathrel{+}=n_{i\uparrow}n_{i\downarrow}$、すなわち $\mathcal P_G=\exp[\sum_ig_{{\rm idx}(i)}n_{i\uparrow}n_{i\downarrow}]$。
- **Jastrow** (インデックス写像 $\texttt{JastrowIdx}[i][j]$): $i<j$ について $c_{{\rm idx}(i,j)}\mathrel{+}=(n_i-1)(n_j-1)$、すなわち $\mathcal P_J=\exp[\sum_{i<j}v_{{\rm idx}(i,j)}(n_i-1)(n_j-1)]$、ここで $n_i=n_{i\uparrow}+n_{i\downarrow}$。インデックス写像が対称なら、これはマニュアルの $\tfrac12\sum_{i\ne j}$ に等しくなります。$n_i=1$ のサイトは何も寄与しません。
- **2サイトのダブロン-ホロン** (`DH2`、インデックステーブル $\texttt{DoublonHolon2siteIdx}[t][2i..2i+1]$ = パターン $t$ におけるサイト $i$ の 2 つのパートナーサイト $r_0,r_1$): $n_i\ne1$ のすべてのサイトについて、$\xi=n_i/2\in\{0\ (\text{holon}),1\ (\text{doublon})\}$ とし、$m$ をパートナーのうち*反対の*欠陥の数 (サイト $i$ がホロンならダブロン、ダブロンならホロン)、$m=0,1,2$ とします。このとき $c_{\,{\rm offset}+t+(\xi+2m)N_{\rm DH2}}\mathrel{+}=1$ となり、パターンごとに $2\times3=6$ 個の係数が得られます。
- **4サイトのダブロン-ホロン** (`DH4`): パートナーサイトが 4 つになる以外は同様で、$m=0\ldots4$、パターンごとに $2\times5=10$ 個の係数です。

> **実装**
> - C: `MakeProjCnt` — `extern/mVMC-1.3.0/src/mVMC/projection.c:58`
> - C: `UpdateProjCnt` — `extern/mVMC-1.3.0/src/mVMC/projection.c:155`
> - C: `LogProjVal` — `extern/mVMC-1.3.0/src/mVMC/projection.c:32`
> - C: `LogProjRatio` — `extern/mVMC-1.3.0/src/mVMC/projection.c:41`
> - C: `ProjRatio` — `extern/mVMC-1.3.0/src/mVMC/projection.c:50`
> - Rust: `make_proj_cnt` — `crates/mvmc-core/src/sampling/projection.rs:106`
> - Rust: `update_proj_cnt` — `crates/mvmc-core/src/sampling/projection.rs:205`
> - Rust: `log_proj_val` — `crates/mvmc-core/src/sampling/projection.rs:320`
> - Rust: `log_proj_ratio` — `crates/mvmc-core/src/sampling/projection.rs:330`
> - Rust: `recompute_dh_counts` — `crates/mvmc-core/src/sampling/projection.rs:20`
> - 整合性: `LogProjRatio` は $k=0..N_{\rm proj}-1$ にわたり、添字順に `z += creal(Proj[idx]) * (double)(projCntNew[idx]-projCntOld[idx])` を累積します。Rust の `log_proj_ratio` も同じ順序と精度で和を取ります (テスト `log_proj_val_and_ratio_align`、`update_proj_cnt_matches_make_proj_cnt_after_hop`)。`ProjRatio` は和の $\exp$ です。C は libm の `exp` を使い、Rust は `julia_exp::exp` (Julia の `exp` の移植) を使います。これは最下位ビットの差の文書化された原因であり、メトロポリス判定を反転させうるものです ([11.3](11-compatibility.md#113-cリファレンスとの既知の相違))。

## 3.6 制限ボルツマンマシン因子

RBM 因子は振幅に次を掛けます。

$$
\mathcal N_{\rm RBM}(x)=\exp\Big[\sum_{p}a_p\,m_p(x)\Big]\;\prod_{h=1}^{N_h}\cosh\theta_h(x),\qquad
\theta_h(x)=b_h+\sum_{p}W_{ph}\,m_p(x),
$$

可視特徴量 $m_p$ には 3 種類があります (マニュアルの単一の「General RBM」は 3 番目です)。

| 種類 | 可視特徴量 $m_p$ | 隠れニューロン |
|------|------------------------|----------------|
| 電荷 | $n_i-1$ (サイト $i=0..N_s-1$) | `NneuronCharge` |
| スピン | $n_{i\uparrow}-n_{i\downarrow}$ | `NneuronSpin` |
| 一般 | $2N_s$ 個のスピン軌道 $I$ にわたる $2n_I-1$ | `NneuronGeneral` |

パラメータは **9 個**のインデックスブロック — `{Charge,Spin,General}RBM_{PhysLayer, HiddenLayer, PhysHidden}` —
にまとめられ、$a$ (物理層バイアス)、$b$ (隠れ層バイアス)、$W$ (結合) を保持します。
`Para` 内の順序は、すべての `PhysLayer` ブロック (電荷、スピン、一般)、次にすべての `HiddenLayer`
ブロック、次にすべての `PhysHidden` ブロックです。コードは
ベクトル $\texttt{rbmCnt}=[\,\sum_pa\text{-weighted counts}\ (N^{\rm phys}_{\rm RBM}),\ \theta_h\ (N_h)\,]$ を保持し、
ホッピング時に逐次更新します (`UpdateRBMCnt`)。対数振幅は

$$
\ln\mathcal N_{\rm RBM}=\sum_{p}\texttt{RBM}[p]\,\texttt{rbmCnt}[p]+\sum_{h}\ln\cosh\texttt{rbmCnt}[N^{\rm phys}+h].
$$

です。

メトロポリスの指数に入るのは $\Delta\ln\mathcal N_{\rm RBM}$ の**実部**だけです
([4.4](04-theory-sampling.md#44-採択判定))。グリーン関数では、
複素数の比 `RBMRatio` 全体が振幅比に掛けられます。`NBlockSize_RBMRatio`
(デフォルト 200) は、ベクトル化された比の評価における C のブロックサイズです。

> **実装**
> - C: `MakeRBMCnt` — `extern/mVMC-1.3.0/src/mVMC/rbm.c:187`
> - C: `UpdateRBMCnt` — `extern/mVMC-1.3.0/src/mVMC/rbm.c:386`
> - C: `LogRBMRatio` — `extern/mVMC-1.3.0/src/mVMC/rbm.c:125`
> - C: `RBMRatio` — `extern/mVMC-1.3.0/src/mVMC/rbm.c:64`
> - C: `WeightRBM` — `extern/mVMC-1.3.0/src/mVMC/rbm.c:30`
> - Rust: `make_rbm_cnt` — `crates/mvmc-core/src/sampling/rbm.rs:160`
> - Rust: `update_rbm_cnt_hopping` — `crates/mvmc-core/src/sampling/rbm.rs:311`
> - Rust: `log_rbm_ratio` — `crates/mvmc-core/src/sampling/rbm.rs:461`
> - Rust: `log_rbm_val` — `crates/mvmc-core/src/sampling/rbm.rs:538`
> - Rust: `log_cosh_stable` — `crates/mvmc-core/src/sampling/rbm.rs:453`
> - 整合性: C は隠れニューロンごとに `clog(ccosh(theta))` と `cexp` を評価します (`rbm.c:30-60`)。Rust は数値的に安定な `log_cosh_stable` を使うため、$|\operatorname{Re}\theta|$ が大きい場合は `clog(ccosh)` と丸め誤差レベルで異なることがあります。メトロポリスの指数は、RBM の項を左結合の順序 `(proj + rbm.re + ip_new.re) - ip_old.re` で加えます (`metropolis.rs:41`、テスト `rbm_acceptance_preserves_julia_left_associative_log_additions`)。RBM ブロックのパーサーは、出力の前に「アーカイブされたスパース」RBM 定義を拒否します (`crates/mvmc-core/src/run.rs:2074`、テスト `public_runner_rejects_archived_sparse_rbm_definitions_before_output`)。

## 3.7 初期値と同期

**初期化** (`InitParameter`。すべてのランクで*同じ*乱数状態を用いて 1 回実行され、
乱数の引き順がどこでも同一になります): すべての射影
パラメータは 0 から始まります。RBM と Slater の係数は、**フラグが正の場合に限り**ランダムで、
そうでなければ 0 で乱数は消費されません。

| ブロック | 実数モード | 複素数モード | パラメータあたりの乱数の引き数 |
|-------|-----------|--------------|---------------------|
| RBM | $0.01\,(r-\tfrac12)/N_{\rm neuron}$ | $10^{-2}\,r_1\,e^{2\pi i r_2}$ | 1 / 2 |
| Slater | $2(r-\tfrac12)$ | $[2(r_1-\tfrac12)+2i(r_2-\tfrac12)]/\sqrt2$ | 1 / 2 |
| OptTrans | `ParaQPOptTrans` | 同じ | 0 |

ここで $r$ は `genrand_real2` で引かれます。RBM の乱数は Slater の乱数より先に引かれます。その後、
任意の*初期パラメータファイル* (C ドライバの 2 番目の位置引数、Rust では `--initial-def`) と
`In*` 定義ファイルが、この順で値を
上書きします。

**同期** (`SyncModifiedParameter`。初期化後、および
SR 更新のたびに行われます):

1. `Para` をルートからブロードキャストします (MPI)。
2. ダブロン-ホロンのシフト: すべての DH2 (または DH4) 係数が最適化される場合、
   3 個 (または 5 個) ずつの各グループのビンから平均を引き、
   取り除いた量を Gutzwiller パラメータに加えます (`shiftDH2`, `shiftDH4`)。
3. Gutzwiller–Jastrow のシフト: すべての Gutzwiller *および* Jastrow 係数が
   最適化される場合、$N_G+N_J$ 個すべての値の共通の平均を引きます (`shiftGJ`)。
   これらのシフトは、規格化を除いて $|\psi|^2$ を変えません。フラグは
   `SetFlagShift` (`parameter.c:255`、`readdef.c:1170` から呼ばれる) で 1 回だけ設定されます。
4. Slater パラメータを、$\max_k|f_k|=4$ となるように再スケールします (`D_AmpMax`)。
5. OptTrans が有効な場合、OptTrans の重みを最大絶対値が 1 になるように再スケールします。

> **実装**
> - C: `InitParameter` — `extern/mVMC-1.3.0/src/mVMC/parameter.c:35`
> - C: `ReadInitParameter` — `extern/mVMC-1.3.0/src/mVMC/parameter.c:95`
> - C: `SyncModifiedParameter` — `extern/mVMC-1.3.0/src/mVMC/parameter.c:134`
> - C: `shiftGJ` — `extern/mVMC-1.3.0/src/mVMC/parameter.c:181`
> - C: `shiftDH2` — `extern/mVMC-1.3.0/src/mVMC/parameter.c:202`
> - C: `shiftDH4` — `extern/mVMC-1.3.0/src/mVMC/parameter.c:228`
> - C: `D_AmpMax` (4.0) — `extern/mVMC-1.3.0/src/mVMC/parameter.c:32`
> - Rust: `init_parameter` — `crates/mvmc-expert-parsers/src/utils/parameter_init.rs:63`
> - Rust: `sync_modified_parameter` — `crates/mvmc-expert-parsers/src/utils/parameter_init.rs:167`
> - Rust: `sync_modified_parameter` (reducer-aware wrapper) — `crates/mvmc-core/src/sync.rs:18`
> - Rust: `sync_modified_parameter_local` — `crates/mvmc-core/src/sync.rs:87`
> - Rust: `read_initial_def` — `crates/mvmc-core/src/initial_params.rs:117`
> - Rust: `read_opt_para_file` — `crates/mvmc-core/src/initial_params.rs:141`
> - 整合性: 乱数の**引き数と順序** (RBM ブロックが先で正規のセクション/インデックス順、次に Slater。実数: 有効なパラメータごとに `genrand_real2` を 1 回、複素数: 半径、次に位相 / 実部、次に虚部) は保存されなければなりません。これは厳密であり、許容誤差の対象ではありません ([11.4](11-compatibility.md#114-数値比較ポリシー))。複素数の Slater 値は 2 回の乱数の後に `sqrt(2.0)` で割られます。Rust の複素数 RBM の位相は Julia 互換の `sin`/`cos` (`julia_trig`) を使います。Rust の Slater 振幅の上限処理は $|f|$ に `julia_hypot::hypot` を使い (`parameter_init.rs:252`)、C は `cabs` を使います。また `sync_modified_parameter_local` の OptTrans 振幅も `hypot` を使います (`sync.rs:91`)。C は `cabs` に対して `1.0/xmax` で正規化します。DH4 のシフト後の累積順序は、ビン順に和を取ってから 5 で割ります (`parameter_init.rs:197-207`)。
> - 整合性: `sync_modified_parameter_local(data, shift_correlations)` における相関のシフトは、呼び出し側が無効にできます (Julia のライフサイクル)。CLI は常にデフォルトを渡します。
